"""The chapter backfill against a fake catalog: what it asks, what it reads
and what it reports with the claim's token; how long it waits when the
catalog hands out nothing; the titles it gives back unread (a deadline too
near, the worker stopping, a read the deadline cut short)."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from analyzer import chapter_backfill
from analyzer.katalog import ClaimTimedOut, KatalogClient
from analyzer.pipelines import chapters


class Catalog:
    """The catalog's side of the backfill: the claims it answers, in turn,
    and every report."""

    def __init__(self, *claims: dict[str, Any] | None) -> None:
        self.claims = list(claims)
        self.asked: list[tuple[str, int]] = []
        self.timeouts: list[float | None] = []
        self.put: list[tuple[str, list[dict], dict, str | None]] = []
        self.failed: list[tuple[str, dict, str | None]] = []
        self.released: list[tuple[str, str]] = []

    def claim_chapters(
        self, worker: str, limit: int = 1, timeout: float | None = None
    ) -> dict[str, Any] | None:
        self.asked.append((worker, limit))
        self.timeouts.append(timeout)
        claim = (
            self.claims.pop(0)
            if self.claims
            else {"items": [], "idle": "done", "retryAfterSeconds": 3600}
        )
        if isinstance(claim, Exception):
            raise claim
        return claim

    def put_chapters(
        self, item_id: str, chs: list[dict], *, read: dict | None = None, token: str | None = None
    ) -> dict | None:
        self.put.append((item_id, chs, read or {}, token))
        return {"state": "read" if chs else "none"}

    def report_chapters_failed(self, item_id: str, read: dict, *, token: str | None = None) -> bool:
        self.failed.append((item_id, read, token))
        return True

    def release_chapters(self, item_id: str, token: str) -> bool:
        self.released.append((item_id, token))
        return True


def at(seconds: float) -> str:
    """A deadline seconds from now, as the catalog writes one."""
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def job(path: str, **fields: Any) -> dict[str, Any]:
    return {
        "id": "m1",
        "type": "movie",
        "path": path,
        "sourceId": "s1",
        "durationMs": 60_000,
        "token": "tok-1",
        **fields,
    }


def backfill(
    catalog: Catalog, stop: threading.Event | None = None
) -> chapter_backfill.ChapterBackfill:
    return chapter_backfill.ChapterBackfill(catalog, stop or threading.Event(), "analyzer-0")  # type: ignore[arg-type]


def a_read(monkeypatch: pytest.MonkeyPatch, result: chapters.ChapterRead) -> list[dict]:
    """chapters.read answers result; returns the reads it was asked for."""
    asked: list[dict] = []

    def read(path: str, **kwargs: Any) -> chapters.ChapterRead:
        asked.append({"path": path, **kwargs})
        return result

    monkeypatch.setattr(chapters, "read", read)
    return asked


def test_a_title_handed_out_is_read_and_reported_with_its_token(
    monkeypatch, tmp_path: Path
) -> None:
    found = chapters.ChapterRead(
        path="/m.mkv",
        ok=True,
        found=3,
        probe="ffprobe version 4.4.2",
        chapters=[{"startMs": 0, "endMs": 1000, "title": "Opening", "ordinal": 1}],
    )
    asked = a_read(monkeypatch, found)
    catalog = Catalog({"items": [job("/m.mkv")], "deadline": at(1800)})
    # a second at least before the next claim
    assert backfill(catalog).claim_once() == chapter_backfill.CLAIM_GAP_S
    assert catalog.asked == [("analyzer-0", 1)]
    [(item, chs, read, token)] = catalog.put
    assert item == "m1" and token == "tok-1" and chs == found.chapters
    assert (
        read["found"] == 3 and read["probe"] == "ffprobe version 4.4.2" and read["path"] == "/m.mkv"
    )
    # the read is of the original the claim named
    assert read["sourceId"] == "s1"
    # read with the title's length, and no longer than the read's own timeout
    assert asked[0]["duration_ms"] == 60_000 and asked[0]["timeout"] == chapters.read_timeout()


def test_a_read_that_failed_is_reported_with_its_token(monkeypatch) -> None:
    a_read(
        monkeypatch,
        chapters.ChapterRead(
            path="/m.mkv", ok=False, error="ffprobe exited 1: moov atom not found"
        ),
    )
    catalog = Catalog({"items": [job("/m.mkv")], "deadline": at(1800)})
    backfill(catalog).claim_once()
    assert catalog.put == [] and catalog.released == []
    [(item, read, token)] = catalog.failed
    assert (item, token) == ("m1", "tok-1") and read[
        "error"
    ] == "ffprobe exited 1: moov atom not found"
    assert read["transient"] is False and read["sourceId"] == "s1"


def test_a_claim_that_settled_titles_from_their_records_is_followed_at_once() -> None:
    catalog = Catalog(
        {"items": [], "settled": 50, "idle": "50 titles were settled", "retryAfterSeconds": 1}
    )
    assert backfill(catalog).claim_once() == chapter_backfill.SETTLED_WAIT_S


def test_a_read_of_a_real_file(tmp_path: Path) -> None:
    # a file whose original is gone between the claim and the read
    catalog = Catalog({"items": [job(str(tmp_path / "gone.mkv"))], "deadline": at(1800)})
    backfill(catalog).claim_once()
    [(_item, read, _token)] = catalog.failed
    assert read["error"].startswith("no file to read")


def test_a_deadline_too_near_gives_the_title_back_unread(monkeypatch) -> None:
    asked = a_read(monkeypatch, chapters.ChapterRead(path="/m.mkv", ok=True))
    catalog = Catalog({"items": [job("/m.mkv")], "deadline": at(5)})
    backfill(catalog).claim_once()
    assert asked == [] and catalog.released == [("m1", "tok-1")] and catalog.put == []


def test_a_read_runs_no_longer_than_its_deadline(monkeypatch) -> None:
    asked = a_read(
        monkeypatch,
        chapters.ChapterRead(
            path="/m.mkv", ok=False, timed_out=True, error="ffprobe did not answer within 60 s"
        ),
    )
    catalog = Catalog({"items": [job("/m.mkv")], "deadline": at(60)})
    backfill(catalog).claim_once()
    assert 50 < asked[0]["timeout"] <= 60
    # cut short by the deadline: no failure of the title
    assert catalog.released == [("m1", "tok-1")] and catalog.failed == []


def test_a_read_that_ran_out_of_its_own_time_failed(monkeypatch) -> None:
    a_read(
        monkeypatch,
        chapters.ChapterRead(
            path="/m.mkv", ok=False, timed_out=True, error="ffprobe did not answer within 120 s"
        ),
    )
    catalog = Catalog({"items": [job("/m.mkv")], "deadline": at(1800)})
    backfill(catalog).claim_once()
    assert catalog.released == [] and [f[1]["error"] for f in catalog.failed] == [
        "ffprobe did not answer within 120 s"
    ]


def test_a_worker_that_stops_gives_its_title_back(monkeypatch) -> None:
    stop = threading.Event()
    stop.set()
    asked = a_read(monkeypatch, chapters.ChapterRead(path="/m.mkv", ok=True))
    catalog = Catalog({"items": [job("/m.mkv")], "deadline": at(1800)})
    backfill(catalog, stop).claim_once()
    assert asked == [] and catalog.released == [("m1", "tok-1")]


def test_a_read_stopped_with_its_worker_gives_the_title_back(monkeypatch) -> None:
    a_read(
        monkeypatch,
        chapters.ChapterRead(
            path="/m.mkv", ok=False, interrupted=True, error="the read was stopped"
        ),
    )
    catalog = Catalog({"items": [job("/m.mkv")], "deadline": at(1800)})
    backfill(catalog).claim_once()
    assert catalog.released == [("m1", "tok-1")] and catalog.failed == []


@pytest.mark.parametrize(
    ("claim", "wait"),
    [
        ({"items": [], "idle": "chapters.backfill is off", "retryAfterSeconds": 900}, 900.0),
        (
            {"items": [], "idle": "outside chapters.backfill.window", "retryAfterSeconds": 14_400},
            3600.0,
        ),
        ({"items": [], "idle": "1 titles are being read", "retryAfterSeconds": 5}, 30.0),
        ({"items": [], "idle": "?", "retryAfterSeconds": "soon"}, 3600.0),
        (None, 600.0),  # a catalog without the backfill
    ],
)
def test_a_catalog_that_hands_out_nothing_says_how_long_to_wait(claim, wait: float) -> None:
    catalog = Catalog(claim)
    assert backfill(catalog).claim_once() == wait


def test_a_claim_the_catalog_does_not_answer_in_time_is_asked_again_soon() -> None:
    catalog = Catalog(ClaimTimedOut("timed out"))  # type: ignore[arg-type]
    # not "can't be asked" (NO_BACKFILL_WAIT_S): the catalog is there, and slow
    assert backfill(catalog).claim_once() == chapter_backfill.WAIT_MIN_S == 30.0
    # its answer waited for well past the catalog's own bound of a claim (20 s)
    assert (
        catalog.timeouts == [chapter_backfill.CLAIM_TIMEOUT_S]
        and chapter_backfill.CLAIM_TIMEOUT_S >= 60
    )


def test_the_client_waits_a_claims_timeout_and_says_it_ran_out() -> None:
    seen: list[dict] = []

    def catalog(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        seen.append(request.extensions["timeout"])
        if request.url.path == "/api/chapters/claim":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(404)

    client = KatalogClient(
        "http://catalog.test", "http://catalog.test/token", "worker", "not-a-secret"
    )
    client._http = httpx.Client(transport=httpx.MockTransport(catalog))
    with pytest.raises(ClaimTimedOut):
        client.claim_chapters("analyzer-0", limit=1, timeout=60.0)
    assert seen == [{"connect": 60.0, "read": 60.0, "write": 60.0, "pool": 60.0}]


def test_a_catalog_that_fails_a_claim_is_asked_again_later() -> None:
    def catalog(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        raise httpx.ConnectError("refused", request=request)

    client = KatalogClient(
        "http://catalog.test", "http://catalog.test/token", "worker", "not-a-secret"
    )
    client._http = httpx.Client(transport=httpx.MockTransport(catalog))
    assert client.claim_chapters("analyzer-0", limit=1, timeout=60.0) is None
    worker = chapter_backfill.ChapterBackfill(client, threading.Event(), "analyzer-0")
    assert worker.claim_once() == chapter_backfill.NO_BACKFILL_WAIT_S


def test_a_malformed_title_is_passed_over() -> None:
    catalog = Catalog({"items": [{"id": "m1"}], "deadline": at(1800)})
    backfill(catalog).claim_once()
    assert catalog.put == catalog.failed == catalog.released == []


def test_the_thread_asks_until_it_stops(monkeypatch) -> None:
    a_read(monkeypatch, chapters.ChapterRead(path="/m.mkv", ok=True))
    stop = threading.Event()
    catalog = Catalog(
        {"items": [job("/m.mkv")], "deadline": at(1800)},
        {"items": [job("/e.mkv", id="e1", token="tok-2")], "deadline": at(1800)},
    )
    worker = backfill(catalog, stop)
    t = threading.Thread(target=worker.run)
    t.start()
    for _ in range(200):
        if len(catalog.asked) >= 3:
            break
        time.sleep(0.01)
    stop.set()
    t.join(timeout=5)
    assert not t.is_alive()
    assert [(i, tok) for i, _c, _r, tok in catalog.put] == [("m1", "tok-1"), ("e1", "tok-2")]


@pytest.mark.parametrize(
    ("value", "on"), [(None, True), ("true", True), ("false", False), ("FALSE", False)]
)
def test_whether_the_backfill_asks(monkeypatch, value, on: bool) -> None:
    if value is None:
        monkeypatch.delenv("CHAPTER_BACKFILL", raising=False)
    else:
        monkeypatch.setenv("CHAPTER_BACKFILL", value)
    assert chapter_backfill.enabled() is on
