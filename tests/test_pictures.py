"""The picture worker (pictures.py) against a fake catalog: the analyze
pass's titles go first and are measured unless the catalog has an area of
what they play now of this method; the backfill's claims are measured and
reported with their tokens, nothing handed out waits as long as the catalog
says, and a measurement past its deadline, or stopped, gives its title back.
The measurement itself is replaced by a recorder."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime

import httpx
import pytest

from analyzer import pictures
from analyzer.katalog import ClaimedItem, KatalogClient
from analyzer.pipelines import picture

BASE = "http://catalog.test"
FILM = "5b0e1c2d-0000-4000-8000-000000000001"
EXTRA = "5b0e1c2d-0000-4000-8000-000000000002"
EPISODE = "5b0e1c2d-0000-4000-8000-000000000003"


class Catalog:
    """POST /api/picture/claim answers the claims queued, one per call (an
    empty claim once they are used up); GET /api/picture/items/{id} the
    areas set; every PUT is recorded."""

    def __init__(
        self, claims: list[dict] | None, status: int = 200, areas: dict | None = None
    ) -> None:
        self.claims = list(claims or [])
        self.status = status
        self.areas = areas or {}
        self.asked: list[dict] = []
        self.reports: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        body = json.loads(request.content or b"null")
        if request.method == "POST" and path == "/api/picture/claim":
            self.asked.append(body)
            if self.status != 200:
                return httpx.Response(self.status, text="no backfill")
            if not self.claims:
                return httpx.Response(
                    200, json={"items": [], "idle": "every title has one", "retryAfterSeconds": 900}
                )
            return httpx.Response(200, json=self.claims.pop(0))
        if request.method == "GET" and path.startswith("/api/picture/items/"):
            have = self.areas.get(path.rsplit("/", 1)[1])
            return httpx.Response(200, json=have) if have else httpx.Response(404)
        if request.method == "PUT" and path.startswith("/api/picture/"):
            self.reports.append((path, body))
            return httpx.Response(200, json={})
        return httpx.Response(404)


def client(catalog: Catalog) -> KatalogClient:
    c = KatalogClient(BASE, f"{BASE}/token", "worker", "not-a-secret")
    c._http = httpx.Client(transport=httpx.MockTransport(catalog))
    return c


def worker(catalog: Catalog, **kw) -> pictures.PictureWorker:
    return pictures.PictureWorker(client(catalog), threading.Event(), "analyzer-0", **kw)


@pytest.fixture
def measured(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int | None, float | None]]:
    """What the worker measured: each path, its duration and deadline."""
    calls: list[tuple[str, int | None, float | None]] = []

    def measure(
        path: str,
        duration_ms: int | None = None,
        *,
        deadline: float | None = None,
        stop=None,
        probe: bool = False,
    ) -> picture.Measurement:
        calls.append((path, duration_ms, deadline))
        if (deadline is not None and deadline <= 0) or (stop is not None and stop()):
            return picture.Measurement(
                path=path, interrupted=True, reason="stopped before it was done", seconds=21_600
            )
        if probe:
            return picture.Measurement(
                path=path,
                frame=picture.Box(0, 0, 1920, 1080),
                video={"codec": "hevc", "sar": "1:1"},
                measured_on={"kind": "package"},
                sampling={"mode": "probe", "durationS": 7200.0},
            )
        return picture.Measurement(
            path=path,
            frame=picture.Box(0, 0, 1920, 1080),
            area=picture.Box(0, 138, 1920, 804),
            samples=300,
            informative=290,
        )

    monkeypatch.setattr(picture, "measure", measure)
    return calls


def claim(*jobs: dict, deadline: str = "2026-11-03T17:00:00Z") -> dict:
    return {"items": list(jobs), "deadline": deadline}


FILM_JOB = {
    "kind": "item",
    "id": FILM,
    "itemId": FILM,
    "type": "movie",
    "path": "/library/movies/5b/film/versions/v1/hls/v0/playlist.m3u8",
    "durationMs": 7_200_000,
    "token": "token-of-the-film",
}
EXTRA_JOB = {
    "kind": "extra",
    "id": EXTRA,
    "itemId": FILM,
    "type": "extra",
    "path": "/library/movies/5b/film/extras/x1/hls/v0/playlist.m3u8",
    "durationMs": 120_000,
    "token": "token-of-the-extra",
}


def item(item_id: str = EPISODE, kind: str = "episode") -> ClaimedItem:
    return ClaimedItem(
        id=item_id,
        type=kind,
        title="Pilot",
        year=None,
        duration_ms=2_700_000,
        path=f"/media/{item_id}.mkv",
    )


# ------------------------------------------------------------- the backfill
def test_a_claims_titles_are_measured_and_reported_with_their_tokens(measured) -> None:
    catalog = Catalog([claim(FILM_JOB, EXTRA_JOB)])
    assert worker(catalog).claim_once() == 0  # titles were handed out: ask again at once
    assert catalog.asked == [{"worker": "analyzer-0", "limit": 1, "method": "picture/4"}]
    deadline = datetime(2026, 11, 3, 17, 0, tzinfo=UTC).timestamp()
    assert measured == [
        (FILM_JOB["path"], 7_200_000, deadline),
        (EXTRA_JOB["path"], 120_000, deadline),
    ]
    assert [p for p, _b in catalog.reports] == [
        f"/api/picture/items/{FILM}",
        f"/api/picture/extras/{EXTRA}",
    ]
    body = catalog.reports[0][1]
    assert body["area"] == {"x": 0, "y": 138, "width": 1920, "height": 804}
    assert body["frame"] == {"width": 1920, "height": 1080} and body["path"] == FILM_JOB["path"]
    assert body["token"] == "token-of-the-film" and body["method"] == "picture/4"


def test_a_measurement_past_its_deadline_gives_its_title_back(measured) -> None:
    catalog = Catalog([claim(FILM_JOB, deadline="1970-01-01T00:00:00Z")])
    worker(catalog).claim_once()
    assert catalog.reports == [
        (
            f"/api/picture/items/{FILM}",
            {"released": True, "path": FILM_JOB["path"], "token": "token-of-the-film"},
        )
    ]


def test_the_last_run_of_a_title_too_costly_to_measure_says_so(measured) -> None:
    # Its runs before had their whole lease and did not finish either: past
    # its deadline the last run reports the title too costly to measure, an
    # unknown of what a probe of the file says it is.
    job = {**FILM_JOB, "costlyRuns": 2, "last": True}
    catalog = Catalog([claim(job, deadline="1970-01-01T00:00:00Z")])
    worker(catalog).claim_once()
    [(path, body)] = catalog.reports
    assert path == f"/api/picture/items/{FILM}" and body["token"] == "token-of-the-film"
    assert "released" not in body and body["area"] is None and body["seconds"] == 21_600
    assert body["frame"] == {"width": 1920, "height": 1080} and body["video"]["codec"] == "hevc"
    assert body["sampling"]["doubt"] == ["cost"] and body["sampling"]["costlyRuns"] == 3
    assert body["reason"] == (
        "can't tell: too costly to measure: 3 runs of 360 min each did not finish"
    )
    assert measured[-1][2] > time.time()  # the probe has a deadline of its own


def test_a_last_run_stopped_with_its_worker_gives_its_title_back(measured) -> None:
    catalog = Catalog([claim({**FILM_JOB, "costlyRuns": 2, "last": True})])
    w = worker(catalog)
    w.stop.set()
    w.claim_once()
    assert [b.get("released") for _p, b in catalog.reports] == [True]


def test_titles_in_hand_when_the_worker_stops_are_given_back(measured) -> None:
    catalog = Catalog([claim(FILM_JOB, EXTRA_JOB)])
    w = worker(catalog)
    w.stop.set()
    w.claim_once()
    assert [b.get("released") for _p, b in catalog.reports] == [True, True]


@pytest.mark.parametrize(
    ("after", "wait"), [(900, 900.0), (5, pictures.WAIT_MIN_S), (86_400, pictures.WAIT_MAX_S)]
)
def test_nothing_handed_out_waits_as_long_as_the_catalog_says(measured, after, wait) -> None:
    catalog = Catalog([{"items": [], "idle": "inside the quiet hours", "retryAfterSeconds": after}])
    assert worker(catalog).claim_once() == wait
    assert measured == [] and catalog.reports == []


@pytest.mark.parametrize("status", [404, 503])
def test_a_catalog_without_a_backfill_is_asked_again_later(measured, status) -> None:
    catalog = Catalog(None, status=status)
    assert worker(catalog).claim_once() == pictures.NO_BACKFILL_WAIT_S


def test_a_malformed_title_is_skipped(measured) -> None:
    catalog = Catalog([claim({"kind": "item", "id": FILM})])
    worker(catalog).claim_once()
    assert measured == [] and catalog.reports == []


# --------------------------------------------------------------- the passes
def test_a_passs_title_is_measured_and_reported_without_a_token(measured) -> None:
    catalog = Catalog(None)
    w = worker(catalog)
    assert w.submit(item())
    t0 = time.time()
    w.measure_pass(w._queue.get_nowait())
    [(path, ms, deadline)] = measured
    assert path == f"/media/{EPISODE}.mkv" and ms == 2_700_000
    assert t0 + pictures.PASS_DEADLINE_S - 5 <= deadline <= time.time() + pictures.PASS_DEADLINE_S
    [(where, body)] = catalog.reports
    assert where == f"/api/picture/items/{EPISODE}" and "token" not in body


@pytest.mark.parametrize("state", ["measured", "unknown"])
def test_a_current_area_of_this_method_is_not_measured_again(measured, state) -> None:
    have = {"state": state, "current": True, "method": picture.METHOD}
    w = worker(Catalog(None, areas={EPISODE: have}))
    w.measure_pass(pictures.PassTitle(EPISODE, "/media/e.mkv", None))
    assert measured == []


@pytest.mark.parametrize(
    "have",
    [
        {"state": "measured", "current": False, "method": "picture/4"},
        {"state": "measured", "current": True, "method": "picture/3"},
        {"state": "failed", "current": True, "method": "picture/4"},
    ],
)
def test_an_area_of_another_file_or_method_is_measured_again(measured, have) -> None:
    w = worker(Catalog(None, areas={EPISODE: have}))
    w.measure_pass(pictures.PassTitle(EPISODE, "/media/e.mkv", None))
    assert len(measured) == 1


def test_only_movies_and_episodes_are_handed_over(measured) -> None:
    w = worker(Catalog(None))
    assert w.submit(item(FILM, "movie")) and w.submit(item())
    assert not w.submit(item(kind="season"))
    assert not worker(Catalog(None), passes=False).submit(item())


def test_a_full_queue_leaves_the_title_to_the_backfill(measured, monkeypatch) -> None:
    monkeypatch.setattr(pictures, "QUEUE_MAX", 2)
    w = worker(Catalog(None))
    assert [w.submit(item(f"{EPISODE[:-1]}{i}")) for i in range(3)] == [True, True, False]


def test_a_passs_title_goes_before_the_backfills(measured) -> None:
    # The worker measures the pass's title first, then claims; stop ends it.
    catalog = Catalog([claim(FILM_JOB)])
    w = worker(catalog)
    w.submit(item())
    t = threading.Thread(target=w.run)
    t.start()
    for _ in range(500):
        if len(catalog.asked) >= 2:
            break
        time.sleep(0.01)
    w.stop.set()
    t.join(timeout=5)
    assert not t.is_alive()
    assert [p for p, _b in catalog.reports] == [
        f"/api/picture/items/{EPISODE}",
        f"/api/picture/items/{FILM}",
    ]


def test_a_title_handed_over_wakes_a_waiting_worker(measured) -> None:
    catalog = Catalog([])  # nothing to hand out: the catalog says wait 900 s
    w = worker(catalog)
    t = threading.Thread(target=w.run)
    t.start()
    for _ in range(500):
        if catalog.asked:
            break
        time.sleep(0.01)
    w.submit(item())
    for _ in range(500):
        if catalog.reports:
            break
        time.sleep(0.01)
    w.stop.set()
    t.join(timeout=5)
    assert [p for p, _b in catalog.reports] == [f"/api/picture/items/{EPISODE}"]


def test_a_worker_without_the_backfill_never_claims(measured) -> None:
    catalog = Catalog([claim(FILM_JOB)])
    w = worker(catalog, backfill=False)
    w.submit(item())
    t = threading.Thread(target=w.run)
    t.start()
    for _ in range(500):
        if catalog.reports:
            break
        time.sleep(0.01)
    w.stop.set()
    t.join(timeout=5)
    assert catalog.asked == [] and len(catalog.reports) == 1
