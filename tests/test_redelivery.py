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
from pathlib import Path

import httpx
import pytest

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

    def __init__(self, path: Path, steps: dict[str, str], *, siblings: bool = False) -> None:
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
            items = [self.detail(SIBLING)] if self.siblings else []
            return httpx.Response(200, json={"itemId": ITEM, "items": items})
        body = json.loads(request.content or b"null")
        self.writes.append((request.method, path, body))
        if request.method == "PUT" and "/steps/" in path:
            self.steps[path.rsplit("/", 1)[1]] = body["status"]
        return httpx.Response(200, json={})


class Detectors:
    """What each pass finds, and which passes were asked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, **found: list[dict]) -> None:
        self.calls: list[str] = []
        self.found = {"tidb": [INTRO], "subtitle": [RECAP], "blackframe": [CREDITS],
                      "silence": [], "chromaprint": [THEME], **found}

        def recorder(name: str):
            def detect(*_args: object, **_kwargs: object) -> list[dict]:
                self.calls.append(name)
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
