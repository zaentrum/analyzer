"""The consumer loop against a fake broker and a fake catalog API: which
`enriched` events run which passes, what a pass writes back, and what it
leaves alone.

The real KatalogClient talks to the fake catalog through an httpx mock
transport, so the guards read the step statuses exactly as in production
and every write is recorded. The detectors are replaced by recorders that
return fixed signals — they are what the passes find, not what is tested.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx
import pytest
from structlog.testing import capture_logs

from analyzer import kafka, worker
from analyzer.katalog import KatalogClient
from analyzer.pipelines import blackframe, chapters, chromaprint, silence, subtitles, tidb

ITEM = "7a1c0de0-0000-4000-8000-000000000003"
SIBLING = "7a1c0de0-0000-4000-8000-000000000004"
BASE = "http://catalog.test"

INTRO = {"kind": "intro", "startMs": 0, "endMs": 60_000, "source": "tidb",
         "confidence": 0.98, "label": None}
RECAP = {"kind": "recap", "startMs": 60_000, "endMs": 90_000, "source": "subtitle",
         "confidence": 0.9, "label": "previously on"}
CREDITS = {"kind": "credits", "startMs": 2_500_000, "endMs": 2_600_000,
           "source": "blackframe", "confidence": 0.7, "label": "black 3.0s"}
THEME = {"kind": "intro", "startMs": 5_000, "endMs": 55_000, "source": "chromaprint",
         "confidence": 0.8, "label": "chromaprint x1"}

DONE = {"tidb": "done", "chapter": "done", "subtitle": "done", "blackframe": "done",
        "silence": "skipped"}


class Message:
    """A consumed record, as confluent-kafka hands it to the loop."""

    def __init__(self, value: dict, offset: int) -> None:
        self._value = json.dumps(value).encode()
        self._offset = offset

    def value(self) -> bytes:
        return self._value

    def error(self) -> None:
        return None


class Broker:
    """One partition for the consumer and the topic the producer writes.
    Sets `stop` once every message has been polled, which ends the loop."""

    def __init__(self, events: list[dict], stop: threading.Event) -> None:
        self.pending = [Message(e, i) for i, e in enumerate(events)]
        self.stop = stop
        self.committed: list[int] = []
        self.produced: list[tuple[str, dict]] = []

    def subscribe(self, _topics: list[str]) -> None:
        pass

    def poll(self, _timeout: float) -> Message | None:
        if not self.pending:
            self.stop.set()
            return None
        return self.pending.pop(0)

    def commit(self, message: Message) -> None:
        self.committed.append(message._offset)

    def close(self) -> None:
        pass

    def produce(self, topic: str, key: bytes, value: bytes) -> None:
        self.produced.append((topic, json.loads(value)))

    def flush(self, *_args: object) -> int:
        return 0


class Catalog:
    """The catalog's worker protocol. Step writes update the statuses the
    next read returns, as the real one does."""

    def __init__(self, path: Path, steps: dict[str, str], *, siblings: bool | str = False) -> None:
        self.path = path
        self.steps = dict(steps)
        self.siblings = siblings
        self.writes: list[tuple[str, str, object]] = []

    def detail(self, item_id: str) -> dict:
        return {"id": item_id, "type": "episode", "title": "Clip", "year": 2020,
                "durationMs": 2_600_000, "path": str(self.path), "seasonNumber": 1,
                "episodeNumber": 2, "seriesTmdbId": "1", "movieTmdbId": "2",
                "hasOwnPoster": True, "hasOwnBackdrop": True}

    def step_writes(self, step: str | None = None) -> list[tuple[str, str]]:
        return [(path.rsplit("/", 1)[1], body["status"]) for method, path, body in self.writes
                if "/steps/" in path and (step is None or path.endswith(f"/{step}"))]

    def segment_writes(self) -> list[list[dict]]:
        return [body["segments"] for _m, path, body in self.writes
                if path.startswith("/api/segments/")]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        if request.method == "GET" and path == f"/api/analyze/items/{ITEM}":
            return httpx.Response(200, json=self.detail(ITEM))
        if request.method == "GET" and path == f"/api/analyze/items/{ITEM}/steps":
            return httpx.Response(200, json={"itemId": ITEM, "steps": self.steps})
        if request.method == "GET" and path == f"/api/analyze/items/{ITEM}/siblings":
            if self.siblings == "error":
                return httpx.Response(503, text="catalog busy")
            items = [self.detail(SIBLING)] if self.siblings else []
            return httpx.Response(200, json={"itemId": ITEM, "items": items})
        body = json.loads(request.content or b"null")
        self.writes.append((request.method, path, body))
        if request.method == "PUT" and "/steps/" in path:
            self.steps[path.rsplit("/", 1)[1]] = body["status"]
        return httpx.Response(200, json={})


class Detectors:
    """What each pass finds, and which passes were asked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, slow: dict[str, float] | None = None,
                 **found: list[dict]) -> None:
        self.calls: list[str] = []
        self.found = {"tidb": [INTRO], "subtitle": [RECAP], "blackframe": [CREDITS],
                      "silence": [], "chromaprint": [THEME], **found}
        self.slow = slow or {}

        def recorder(name: str):
            def detect(*_args: object, **_kwargs: object) -> list[dict]:
                self.calls.append(name)
                time.sleep(self.slow.get(name, 0))
                return [dict(s) for s in self.found[name]]
            return detect

        monkeypatch.setattr(tidb, "detect", recorder("tidb"))
        monkeypatch.setattr(subtitles, "detect", recorder("subtitle"))
        monkeypatch.setattr(blackframe, "detect", recorder("blackframe"))
        monkeypatch.setattr(silence, "detect", recorder("silence"))
        monkeypatch.setattr(chromaprint, "detect", recorder("chromaprint"))
        monkeypatch.setattr(chapters, "detect", lambda _path: chapters.ChapterResult(
            segments=[], chapters=[{"title": "Act 1", "startMs": 0, "endMs": 2_600_000}]))


def event(**fields: str) -> dict:
    """An `enriched` envelope as the catalog produces it."""
    return {"eventId": "e1", "itemId": ITEM, "type": "episode", "step": "analyze",
            "status": "done", "occurredAt": "2026-10-04T00:00:00Z",
            "source": "katalog-manager", **fields}


def retry() -> dict:
    """The same trigger as the catalog's retry sends it again."""
    return event(status="retry", source="retry")


def run(monkeypatch: pytest.MonkeyPatch, catalog: Catalog, events: list[dict]) -> Broker:
    stop = threading.Event()
    broker = Broker(events, stop)
    monkeypatch.setattr(kafka, "build_consumer", lambda *a, **k: broker)
    monkeypatch.setattr(kafka, "build_producer", lambda *a, **k: broker)
    client = KatalogClient(BASE, f"{BASE}/token", "worker", "not-a-secret")
    client._http = httpx.Client(transport=httpx.MockTransport(catalog))
    worker.run_event_consumer(client, "broker.test:9092", "analyzer-workers",
                              "stube.catalog.item.enriched", "stube.catalog.item.analyzed",
                              "PLAINTEXT", "transcode", 0.0, stop)
    return broker


@pytest.fixture
def media(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"\x1a\x45\xdf\xa3")
    return path


# ------------------------------------------- a pass that skips done passes
def test_full_pass_writes_what_it_found(monkeypatch, media) -> None:
    catalog = Catalog(media, {})
    found = Detectors(monkeypatch)
    broker = run(monkeypatch, catalog, [event()])
    assert found.calls == ["tidb", "subtitle", "blackframe", "silence"]
    [segments] = catalog.segment_writes()
    assert {s["source"] for s in segments} == {"tidb", "subtitle", "blackframe"}
    assert [t for t, _e in broker.produced] == ["stube.catalog.item.analyzed"]
    assert broker.committed == [0]


def test_partial_pass_that_found_nothing_leaves_the_segments(monkeypatch, media) -> None:
    # A redelivery after a crash in blackframe: tidb, chapter and subtitle
    # are done, and their segments are in the catalog. The catalog replaces
    # an item's whole set, so writing what blackframe + silence found —
    # nothing — would have wiped them.
    catalog = Catalog(media, {**DONE, "blackframe": "in_progress", "silence": "pending"})
    found = Detectors(monkeypatch, blackframe=[])
    run(monkeypatch, catalog, [event()])
    assert found.calls == ["blackframe", "silence"]
    assert catalog.segment_writes() == []
    assert catalog.step_writes() == [("blackframe", "in_progress"), ("blackframe", "skipped"),
                                     ("silence", "in_progress"), ("silence", "skipped")]


def test_partial_pass_that_found_something_writes_the_whole_set(monkeypatch, media) -> None:
    # blackframe finds the credits now: the set written must still hold what
    # tidb and subtitle found, so their detectors run again — without a word
    # to their steps.
    catalog = Catalog(media, {**DONE, "blackframe": "in_progress"})
    found = Detectors(monkeypatch)
    run(monkeypatch, catalog, [event()])
    # (silence, skipped, runs again on a redelivery, as before.)
    assert found.calls == ["blackframe", "silence", "tidb", "subtitle"]
    [segments] = catalog.segment_writes()
    assert sorted(s["source"] for s in segments) == ["blackframe", "subtitle", "tidb"]
    assert {step for step, _status in catalog.step_writes()} == {"blackframe", "silence"}


@pytest.mark.parametrize("outage", ["nothing", "error"])
def test_partial_pass_keeps_the_segments_when_a_done_pass_cant_find_them_again(
    monkeypatch, media, outage: str,
) -> None:
    # tidb answers an outage with nothing; writing the set without its intro
    # would lose it.
    catalog = Catalog(media, {**DONE, "blackframe": "in_progress"})
    found = Detectors(monkeypatch, tidb=[])
    if outage == "error":
        def broken(**_kwargs: object) -> list[dict]:
            raise RuntimeError("tidb down")
        monkeypatch.setattr(tidb, "detect", broken)
    run(monkeypatch, catalog, [event()])
    assert "blackframe" in found.calls
    assert catalog.segment_writes() == []
    assert catalog.step_writes("blackframe")[-1] == ("blackframe", "done")


# ----------------------------------------------- chromaprint waiting to run
@pytest.mark.parametrize(("steps", "analyzed"), [
    (DONE, True),                                   # no chromaprint step: not required
    ({**DONE, "chromaprint": "done"}, True),
    ({**DONE, "chromaprint": "failed"}, True),      # failed is terminal; the retry re-claims it
    ({**DONE, "chromaprint": "pending"}, False),    # a retry claimed it
    ({**DONE, "chromaprint": "in_progress"}, False),
    ({**DONE, "silence": "pending"}, False),
    ({}, False),
])
def test_already_analyzed(steps: dict[str, str], analyzed: bool) -> None:
    assert worker._already_analyzed(steps) is analyzed


def test_a_chromaprint_waiting_for_its_run_runs(monkeypatch, media) -> None:
    # The catalog retried a failed chromaprint: claimed (pending), its
    # trigger sent again. Every other pass is finished, which the guard
    # used to take for "already analyzed": the step stayed pending until
    # the reaper timed it out.
    catalog = Catalog(media, {**DONE, "silence": "done", "chromaprint": "pending"}, siblings=True)
    found = Detectors(monkeypatch, silence=[CREDITS | {"source": "silence"}])
    broker = run(monkeypatch, catalog, [event()])
    assert found.calls[0] == "chromaprint"
    assert catalog.step_writes() == [("chromaprint", "in_progress"), ("chromaprint", "done")]
    [segments] = catalog.segment_writes()
    # The whole set: tidb's intro outranks the fingerprinted one.
    assert sorted((s["kind"], s["source"]) for s in segments) == [
        ("credits", "blackframe"), ("intro", "tidb"), ("recap", "subtitle")]
    assert [t for t, _e in broker.produced] == ["stube.catalog.item.analyzed"]


@pytest.mark.parametrize(("siblings", "answer"), [(False, "skipped"), ("error", "failed")])
def test_a_chromaprint_this_pass_cant_run_gets_its_answer(
    monkeypatch, media, siblings, answer: str,
) -> None:
    # The episode has no siblings any more, or the catalog can't list them:
    # the waiting step is answered instead of left pending for the reaper.
    catalog = Catalog(media, {**DONE, "silence": "done", "chromaprint": "pending"},
                      siblings=siblings)
    found = Detectors(monkeypatch)
    run(monkeypatch, catalog, [event()])
    assert "chromaprint" not in found.calls
    assert catalog.step_writes() == [("chromaprint", answer)]
    assert catalog.segment_writes() == []  # nothing new: the stored set stays


# ------------------------------------------------------------------ retries
@pytest.mark.parametrize("steps", [
    DONE,
    {**DONE, "chromaprint": "done"},
    {**DONE, "chromaprint": "skipped", "tidb": "not_applicable"},
])
def test_retry_of_finished_passes_is_only_acked(monkeypatch, media, steps) -> None:
    # The reaper took a long run for dead and the catalog sent the trigger
    # again; the run reported its end before the retry was consumed. The
    # retry runs nothing, writes nothing, sends nothing: one log line.
    catalog = Catalog(media, steps, siblings=True)
    found = Detectors(monkeypatch)
    with capture_logs() as logs:
        broker = run(monkeypatch, catalog, [retry()])
    assert found.calls == []
    assert catalog.writes == []
    assert broker.produced == []
    assert broker.committed == [0]
    assert [e["event"] for e in logs if e.get("item_id") == ITEM] == [
        "consumer.retry_already_finished"]


def test_retry_runs_only_the_unfinished_passes(monkeypatch, media) -> None:
    # blackframe failed and the catalog claimed it for a retry (pending).
    # Unlike a redelivery, the retry leaves the skipped silence pass alone
    # too: the catalog retries none of done, skipped and not_applicable.
    catalog = Catalog(media, {**DONE, "blackframe": "pending"})
    found = Detectors(monkeypatch)
    broker = run(monkeypatch, catalog, [retry()])
    assert found.calls == ["blackframe", "tidb", "subtitle"]  # the last two only to fuse
    assert catalog.step_writes() == [("blackframe", "in_progress"), ("blackframe", "done")]
    [segments] = catalog.segment_writes()
    assert sorted(s["source"] for s in segments) == ["blackframe", "subtitle", "tidb"]
    # A retry that did work passes the chain on; downstream guards skip
    # what is done there.
    assert [t for t, _e in broker.produced] == ["stube.catalog.item.analyzed"]
    assert broker.committed == [0]


def test_retry_of_a_waiting_chromaprint_runs_it(monkeypatch, media) -> None:
    catalog = Catalog(media, {**DONE, "chromaprint": "pending"}, siblings=True)
    found = Detectors(monkeypatch)
    run(monkeypatch, catalog, [retry()])
    assert found.calls[0] == "chromaprint" and "silence" not in found.calls
    assert catalog.step_writes() == [("chromaprint", "in_progress"), ("chromaprint", "done")]


def test_retry_marker() -> None:
    assert kafka.is_retry(kafka.parse_envelope(json.dumps(retry()).encode()))
    assert not kafka.is_retry(kafka.parse_envelope(json.dumps(event()).encode()))
    assert kafka.parse_envelope(b"not json") == {}
    assert kafka.parse_envelope(b"[1]") == {}
    assert kafka.parse_envelope(None) == {}


# ---------------------------------------------------------------- heartbeat
def test_a_long_pass_says_it_is_alive_until_it_ends(monkeypatch, media) -> None:
    # The reaper takes a step silent past its timeout for a dead run and
    # retries it beside the live one. A pass that runs long reports in
    # progress again every beat, and no beat lands after its end.
    monkeypatch.setenv("STEP_HEARTBEAT_SECONDS", "0.05")
    catalog = Catalog(media, {})
    Detectors(monkeypatch, slow={"blackframe": 0.4})
    run(monkeypatch, catalog, [event()])
    time.sleep(0.2)  # a beat that outlived its pass would land by now
    beats = catalog.step_writes("blackframe")
    assert beats[0] == ("blackframe", "in_progress")
    assert beats[-1] == ("blackframe", "done")
    assert len(beats) >= 1 + 3 + 1  # the start, beats, the end
    assert set(beats[1:-1]) == {("blackframe", "in_progress")}


def test_a_failing_pass_stops_beating_before_it_reports_failure(monkeypatch, media) -> None:
    monkeypatch.setenv("STEP_HEARTBEAT_SECONDS", "0.05")
    catalog = Catalog(media, {})
    Detectors(monkeypatch)

    def crash(*_args: object) -> list[dict]:
        time.sleep(0.3)
        raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(silence, "detect", crash)
    run(monkeypatch, catalog, [event()])
    time.sleep(0.2)
    beats = catalog.step_writes("silence")
    assert beats[-1] == ("silence", "failed")
    assert len(beats) >= 3 and set(beats[:-1]) == {("silence", "in_progress")}


@pytest.mark.parametrize(("value", "every"), [(None, 600.0), ("120", 120.0), ("0", 600.0),
                                              ("-5", 600.0), ("soon", 600.0)])
def test_heartbeat_interval(monkeypatch, value, every: float) -> None:
    if value is None:
        monkeypatch.delenv("STEP_HEARTBEAT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("STEP_HEARTBEAT_SECONDS", value)
    assert worker._heartbeat_seconds() == every
    # Far inside the reaper's 2 h timeout of the analyzer's steps.
    assert worker.HEARTBEAT_SECONDS * 6 <= 2 * 3600
