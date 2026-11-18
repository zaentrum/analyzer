"""The captions backfill against a fake catalog: what it asks, what it reads
(the original) and what it reports with the claim's token; how long it waits when the catalog
hands out nothing; the originals it gives back unread; that it refuses to
start with any ffprobe but the pipeline's 7.1 - the image's 4.4.2 above all.
And the client: the claim and the PUTs on the wire."""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from captions_fake import FFPROBE_442
from structlog.testing import capture_logs

from analyzer import captions_backfill
from analyzer.katalog import CaptionsClaim, CaptionsJob, ClaimTimedOut, KatalogClient
from analyzer.pipelines import captions

ORIGINAL = "/m/Film (2024).mkv"
PROBE = "ffprobe version n7.1.5-12-g1fdbca85aa-20260731"


class Catalog:
    """The catalog's side of the backfill: the claims it answers, in turn,
    and every report."""

    def __init__(self, *claims: CaptionsClaim | Exception | None) -> None:
        self.claims = list(claims)
        self.asked: list[tuple[str, int, float | None]] = []
        self.put: list[tuple[CaptionsJob, captions.CaptionsRead]] = []
        self.released: list[CaptionsJob] = []
        self.takes = True

    def claim_captions(
        self, worker: str, limit: int = 1, timeout: float | None = None
    ) -> CaptionsClaim | None:
        self.asked.append((worker, limit, timeout))
        claim = self.claims.pop(0) if self.claims else idle("done", 3600)
        if isinstance(claim, Exception):
            raise claim
        return claim

    def put_captions(self, job: CaptionsJob, original: captions.CaptionsRead) -> bool:
        self.put.append((job, original))
        return self.takes

    def release_captions(self, job: CaptionsJob) -> bool:
        self.released.append(job)
        return True


def at(seconds: float) -> float:
    return time.time() + seconds


def idle(why: str, retry: Any) -> CaptionsClaim:
    try:
        r = float(retry)
    except (TypeError, ValueError):
        r = None
    return CaptionsClaim(idle=why, retry_after_s=r)


def job(**fields: Any) -> CaptionsJob:
    return CaptionsJob(
        **{"source_id": "s1", "item_id": "m1", "path": ORIGINAL, "token": "tok-1", **fields}
    )


def claim(*jobs: CaptionsJob, deadline: float | None = None) -> CaptionsClaim:
    return CaptionsClaim(jobs=list(jobs), deadline=at(1800) if deadline is None else deadline)


def backfill(
    catalog: Catalog, stop: threading.Event | None = None, ffprobe: str = "/x/ffprobe"
) -> captions_backfill.CaptionsBackfill:
    return captions_backfill.CaptionsBackfill(
        catalog, stop or threading.Event(), "analyzer-0", ffprobe=ffprobe  # type: ignore[arg-type]
    )


SIZE = 1_500_000_000  # the size a read found the file at


def found(path: str, has: bool | None, frames: int = 48, **kw: Any) -> captions.CaptionsRead:
    """A read of path, a ten-minute video: its start window, `frames`
    decoded at 25 a second from 0, showing captions or not; none, when it
    failed as a whole (has None)."""
    ok = has is not None
    ws = [captions.Window(0.0, 0.0, frames=frames, captions=has, start=0.0,
                          end=round((frames - 1) * 0.04, 3))] if ok else []  # fmt: skip
    kw.setdefault("file_size", SIZE)
    return captions.CaptionsRead(
        path=path, ok=ok, captions=has, frames=frames if ok else 0, windows=ws,
        duration=600.0 if ok else None, unreadable=not ok, probe=PROBE, **kw,
    )  # fmt: skip


def reads(monkeypatch, **by_path: captions.CaptionsRead) -> list[dict]:
    """captions.read answers each path its read; returns the reads asked for."""
    asked: list[dict] = []

    def read(path: str, **kwargs: Any) -> captions.CaptionsRead:
        asked.append({"path": path, **kwargs})
        return by_path[path]

    monkeypatch.setattr(captions, "read", read)
    return asked


# ------------------------------------------------------------ the claim


def test_an_original_handed_out_is_read_and_reported_with_its_token(monkeypatch) -> None:
    asked = reads(monkeypatch, **{ORIGINAL: found(ORIGINAL, True)})
    catalog = Catalog(claim(job()))
    assert backfill(catalog).claim_once() == captions_backfill.CLAIM_GAP_S
    assert catalog.asked == [("analyzer-0", 1, captions_backfill.CLAIM_TIMEOUT_S)]
    [(j, original)] = catalog.put
    assert j.token == "tok-1" and j.source_id == "s1"
    assert (original.captions, original.frames) == (True, 48)
    # read with its ffprobe, 48 packets, no longer than the read's own timeout
    assert asked == [{"path": ORIGINAL, "packets": 48, "timeout": captions.read_timeout(),
                      "stop": asked[0]["stop"], "ffprobe": "/x/ffprobe"}]  # fmt: skip


def test_a_claim_may_name_how_many_packets(monkeypatch) -> None:
    asked = reads(monkeypatch, **{ORIGINAL: found(ORIGINAL, True)})
    backfill(Catalog(claim(job(packets=96)))).claim_once()
    assert asked[0]["packets"] == 96


def test_a_read_that_failed_is_reported_failed_never_no_captions(monkeypatch) -> None:
    reads(monkeypatch, **{ORIGINAL: found(ORIGINAL, None, error="no video stream to read")})
    catalog = Catalog(claim(job()))
    backfill(catalog).claim_once()
    [(_j, original)] = catalog.put
    assert (original.ok, original.captions, original.frames) == (False, None, 0)
    assert catalog.released == []


def test_a_read_that_cant_name_its_ffprobe_is_given_back_never_reported(monkeypatch) -> None:
    unnamed = captions.CaptionsRead(path=ORIGINAL, ok=True, captions=True, frames=48, probe=None)
    reads(monkeypatch, **{ORIGINAL: unnamed})
    catalog = Catalog(claim(job()))
    with capture_logs() as logs:
        backfill(catalog).claim_once()
    assert catalog.put == [] and len(catalog.released) == 1
    assert "captions.backfill_unnamed_ffprobe" in [x["event"] for x in logs]


def test_a_deadline_too_near_gives_the_original_back_unread(monkeypatch) -> None:
    asked = reads(monkeypatch, **{ORIGINAL: found(ORIGINAL, True)})
    catalog = Catalog(claim(job(), deadline=at(5)))
    backfill(catalog).claim_once()
    assert asked == [] and [j.source_id for j in catalog.released] == ["s1"] and catalog.put == []


def test_a_read_runs_no_longer_than_its_deadline(monkeypatch) -> None:
    asked = reads(monkeypatch, **{ORIGINAL: found(
        ORIGINAL, None, timed_out=True, transient=True, error="ffprobe did not answer within 60 s"
    )})  # fmt: skip
    catalog = Catalog(claim(job(), deadline=at(60)))
    backfill(catalog).claim_once()
    assert 50 < asked[0]["timeout"] <= 60
    # cut short by the deadline: no failure of the original
    assert len(catalog.released) == 1 and catalog.put == []


def test_a_read_that_ran_out_of_its_own_time_failed(monkeypatch) -> None:
    reads(monkeypatch, **{ORIGINAL: found(
        ORIGINAL, None, timed_out=True, transient=True, error="ffprobe did not answer within 120 s"
    )})  # fmt: skip
    catalog = Catalog(claim(job()))
    backfill(catalog).claim_once()
    assert catalog.released == [] and [o.error for _j, o in catalog.put] == [
        "ffprobe did not answer within 120 s"
    ]


def test_a_worker_that_stops_gives_its_original_back(monkeypatch) -> None:
    stop = threading.Event()
    stop.set()
    asked = reads(monkeypatch, **{ORIGINAL: found(ORIGINAL, True)})
    catalog = Catalog(claim(job()))
    backfill(catalog, stop).claim_once()
    assert asked == [] and len(catalog.released) == 1


def test_a_read_stopped_with_its_worker_gives_the_original_back(monkeypatch) -> None:
    reads(monkeypatch, **{ORIGINAL: found(
        ORIGINAL, None, interrupted=True, error="the read was stopped"
    )})  # fmt: skip
    catalog = Catalog(claim(job()))
    backfill(catalog).claim_once()
    assert len(catalog.released) == 1 and catalog.put == []


def test_a_report_the_catalog_did_not_take_is_left_to_its_lease(monkeypatch) -> None:
    reads(monkeypatch, **{ORIGINAL: found(ORIGINAL, True)})
    catalog = Catalog(claim(job()))
    catalog.takes = False
    with capture_logs() as logs:
        backfill(catalog).claim_once()
    assert catalog.released == []
    assert "captions.backfill_report_failed" in [x["event"] for x in logs]


@pytest.mark.parametrize(
    ("answer", "wait"),
    [
        (idle("captions.backfill is off", 900), 900.0),
        (idle("outside captions.backfill.window", 14_400), 3600.0),
        (idle("2 originals are being read", 5), 30.0),
        (idle("?", "soon"), 3600.0),
        (CaptionsClaim(malformed=1), captions_backfill.CLAIM_GAP_S),
        (None, 600.0),  # a catalog without the backfill
    ],
)
def test_a_catalog_that_hands_out_nothing_says_how_long_to_wait(answer, wait: float) -> None:
    assert backfill(Catalog(answer)).claim_once() == wait


def test_a_claim_the_catalog_does_not_answer_in_time_is_asked_again_soon() -> None:
    catalog = Catalog(ClaimTimedOut("timed out"))
    assert backfill(catalog).claim_once() == captions_backfill.WAIT_MIN_S == 30.0


def test_the_thread_asks_until_it_stops(monkeypatch, fake_ffprobe) -> None:
    ff = fake_ffprobe()
    reads(monkeypatch, **{ORIGINAL: found(ORIGINAL, True), "/m/e.mkv": found("/m/e.mkv", False)})
    stop = threading.Event()
    catalog = Catalog(claim(job()), claim(job(source_id="s2", path="/m/e.mkv", token="tok-2")))
    t = threading.Thread(target=backfill(catalog, stop, str(ff.path)).run, daemon=True)
    t.start()
    for _ in range(300):
        if len(catalog.asked) >= 3:
            break
        time.sleep(0.01)
    stop.set()
    t.join(timeout=5)
    assert not t.is_alive()
    assert [(j.source_id, j.token, o.captions) for j, o in catalog.put] == [
        ("s1", "tok-1", True), ("s2", "tok-2", False)]  # fmt: skip


# ------------------------------------------------------------ its ffprobe


def test_the_backfill_refuses_to_start_with_the_image_s_ffprobe_4_4_2(
    monkeypatch, fake_ffprobe
) -> None:
    # The image's own ffprobe (Ubuntu 22.04's 4.4.2) must never read the
    # captions: run with it, the backfill asks the catalog for nothing.
    ff = fake_ffprobe(FFPROBE_442)
    asked = reads(monkeypatch, **{ORIGINAL: found(ORIGINAL, True)})
    catalog = Catalog(claim(job()))
    stop = threading.Event()
    worker = backfill(catalog, stop, ffprobe=str(ff.path))
    with capture_logs() as logs:
        t = threading.Thread(target=worker.run, daemon=True)
        t.start()
        t.join(timeout=5)
        alive = t.is_alive()
        stop.set()  # a backfill that did start ends here
    assert not alive  # it ended at once, without being stopped
    assert catalog.asked == [] and asked == [] and catalog.put == [] and ff.runs == []
    [refused] = [x for x in logs if x["event"] == "captions.backfill_refused"]
    assert refused["log_level"] == "error" and "ffprobe 4.4.2" in refused["why"]
    assert "not the pipeline's 7.1.x" in refused["why"]
    assert worker.refusal() == refused["why"]


def test_the_backfill_refuses_to_start_without_an_ffprobe_where_it_is_named(
    monkeypatch, tmp_path: Path, fake_ffprobe
) -> None:
    # a 7.1 on the PATH changes nothing: CAPTIONS_FFPROBE names the one it runs
    on_path = fake_ffprobe()
    monkeypatch.setenv("PATH", f"{on_path.dir}:/usr/bin:/bin")
    monkeypatch.setenv("CAPTIONS_FFPROBE", str(tmp_path / "missing" / "ffprobe"))
    catalog = Catalog(claim(job()))
    worker = captions_backfill.CaptionsBackfill(catalog, threading.Event(), "analyzer-0")  # type: ignore[arg-type]
    stop = threading.Event()
    worker.stop = stop
    with capture_logs() as logs:
        t = threading.Thread(target=worker.run, daemon=True)
        t.start()
        t.join(timeout=5)
        stop.set()
    assert catalog.asked == [] and on_path.runs == []
    assert "no ffprobe to run at" in logs[-1]["why"]


def test_the_backfill_starts_with_the_pipeline_s_7_1(monkeypatch, fake_ffprobe) -> None:
    ff = fake_ffprobe()
    stop = threading.Event()
    stop.set()  # it starts, and stops at once
    with capture_logs() as logs:
        backfill(Catalog(), stop, str(ff.path)).run()
    started = next(x for x in logs if x["event"] == "captions.backfill_started")
    assert started["probe"] == "ffprobe version n7.1.5-12-g1fdbca85aa-20260731"


def test_an_ffprobe_that_changed_under_it_gives_the_original_back_and_stops(
    monkeypatch, fake_ffprobe
) -> None:
    ff = fake_ffprobe()

    def read(path: str, **kwargs: Any) -> captions.CaptionsRead:
        raise captions.WrongProbe("it is ffprobe 4.4.2 now")

    monkeypatch.setattr(captions, "read", read)
    catalog = Catalog(claim(job()), claim(job()))
    stop = threading.Event()
    t = threading.Thread(target=backfill(catalog, stop, str(ff.path)).run, daemon=True)
    t.start()
    t.join(timeout=5)
    alive = t.is_alive()
    stop.set()
    assert not alive and len(catalog.asked) == 1 and len(catalog.released) == 1


@pytest.mark.parametrize(
    ("value", "on"), [(None, True), ("true", True), ("false", False), ("FALSE", False)]
)
def test_whether_the_backfill_asks(monkeypatch, value, on: bool) -> None:
    if value is None:
        monkeypatch.delenv("CAPTIONS_BACKFILL", raising=False)
    else:
        monkeypatch.setenv("CAPTIONS_BACKFILL", value)
    assert captions_backfill.enabled() is on


# ------------------------------------------------------------ the client


class Wire:
    """The catalog on the wire: what it was sent, and what it answers."""

    def __init__(self, claim: httpx.Response | Exception | None = None, put: int = 200) -> None:
        self.sent: list[tuple[str, str, Any, Any]] = []
        self.claim, self.put = claim, put

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        body = json.loads(request.content) if request.content else None
        self.sent.append((request.method, request.url.path, body, request.extensions["timeout"]))
        if request.url.path == "/api/captions/claim":
            if isinstance(self.claim, Exception):
                raise self.claim
            return self.claim or httpx.Response(404)
        return httpx.Response(self.put, json={"state": "read"})

    def client(self) -> KatalogClient:
        c = KatalogClient(
            "http://catalog.test", "http://catalog.test/token", "worker", "not-a-secret"
        )
        c._http = httpx.Client(transport=httpx.MockTransport(self))
        return c


def test_the_claim_on_the_wire() -> None:
    deadline = datetime.now(UTC) + timedelta(minutes=10)
    wire = Wire(httpx.Response(200, json={
        "items": [
            {"sourceId": "s1", "itemId": "m1", "path": ORIGINAL, "packets": 48,
             "token": "tok-1"},
            {"sourceId": "s2", "itemId": "m2", "path": "/m/b.mkv", "packets": 48,
             "token": "tok-2"},
            {"itemId": "m3", "path": "/m/c.mkv"},  # no sourceId: passed over
        ],
        "deadline": deadline.isoformat().replace("+00:00", "Z"),
    }))  # fmt: skip
    c = wire.client().claim_captions("analyzer-0", limit=1, timeout=60.0)
    assert "settled" not in CaptionsClaim.__dataclass_fields__
    assert wire.sent[0][:3] == ("POST", "/api/captions/claim", {"worker": "analyzer-0", "limit": 1})
    assert wire.sent[0][3]["read"] == 60.0
    assert c is not None and c.malformed == 1
    assert c.jobs == [
        CaptionsJob(source_id="s1", item_id="m1", path=ORIGINAL, packets=48, token="tok-1"),
        CaptionsJob(source_id="s2", item_id="m2", path="/m/b.mkv", packets=48, token="tok-2"),
    ]  # fmt: skip
    assert abs(c.deadline - deadline.timestamp()) < 1


def test_an_idle_claim_on_the_wire() -> None:
    wire = Wire(httpx.Response(200, json={"items": [], "idle": "captions.backfill is off",
                                          "retryAfterSeconds": 900}))  # fmt: skip
    c = wire.client().claim_captions("analyzer-0")
    assert (c.jobs, c.idle, c.retry_after_s, c.deadline) == ([], "captions.backfill is off", 900.0,
                                                             None)  # fmt: skip


@pytest.mark.parametrize("answer", [None, httpx.Response(503, json={"error": "migration"}),
                                    httpx.ConnectError("refused")])  # fmt: skip
def test_a_catalog_without_the_backfill_or_failing_it_hands_out_nothing(answer) -> None:
    assert Wire(answer).client().claim_captions("analyzer-0") is None


def test_a_claim_not_answered_in_time_says_so() -> None:
    wire = Wire(httpx.ReadTimeout("timed out"))
    with pytest.raises(ClaimTimedOut):
        wire.client().claim_captions("analyzer-0", timeout=60.0)


def test_what_the_read_found_on_the_wire() -> None:
    wire = Wire()
    assert wire.client().put_captions(job(), found(ORIGINAL, True))
    [(method, path, body, _t)] = wire.sent
    assert (method, path) == ("PUT", "/api/captions/sources/s1")
    # as katalog-manager takes it: the windows, the length they were placed
    # by, the size of the file read, the ffprobe; no summary, no package
    assert body == {"token": "tok-1", "original": {
        "captionWindows": [{"at": 0.0, "start": 0.0, "end": 1.88, "frames": 48, "captions": True,
                            "interlacedFrames": 0, "repeatFrames": 0, "keyframeBefore": False}],
        "duration": 600.0, "sizeBytes": SIZE, "ffprobe": PROBE}}  # fmt: skip


def test_a_failed_read_on_the_wire_is_an_error_never_no_captions() -> None:
    wire = Wire()
    failed = found(ORIGINAL, None, error="its frames have no presentation times (a raw stream)")
    assert wire.client().put_captions(job(), failed)
    [(_m, _p, body, _t)] = wire.sent
    # a file it could not read: why, the ffprobe, and the size it found
    assert body == {"token": "tok-1", "original": {
        "error": "its frames have no presentation times (a raw stream)", "ffprobe": PROBE,
        "sizeBytes": SIZE}}  # fmt: skip
    # a window that decoded no frame is sent as one that failed, never "none"
    odd = captions.CaptionsRead(path=ORIGINAL, ok=True, captions=False, frames=0, probe=PROBE,
                                duration=600.0,
                                windows=[captions.Window(0.0, 0.0, frames=0, captions=False)])
    wire.client().put_captions(job(), odd)
    assert wire.sent[-1][2]["original"]["captionWindows"] == [
        {"at": 0.0, "start": None, "end": None, "frames": 0, "captions": None,
         "interlacedFrames": None, "repeatFrames": None, "keyframeBefore": None,
         "error": "the window decoded no frame"}]  # fmt: skip
    # a read that ran out of time is a file it could not read, whatever
    # windows it read before
    late = found(ORIGINAL, True, timed_out=True, error="ffprobe did not answer within 120 s")
    wire.client().put_captions(job(), late)
    assert wire.sent[-1][2]["original"] == {
        "error": "ffprobe did not answer within 120 s", "ffprobe": PROBE, "sizeBytes": SIZE}
    # and one that failed as a whole (a length its last packet belies) too
    whole = found(ORIGINAL, True)
    whole.unreadable, whole.error = True, "its file says it ends at 30.000 s, ..."
    wire.client().put_captions(job(), whole)
    assert set(wire.sent[-1][2]["original"]) == {"error", "ffprobe", "sizeBytes"}


def test_an_original_given_back_on_the_wire() -> None:
    wire = Wire()
    assert wire.client().release_captions(job())
    assert wire.sent == [("PUT", "/api/captions/sources/s1", {"token": "tok-1", "released": True},
                          wire.sent[0][3])]  # fmt: skip


def test_no_report_is_sent_without_its_ffprobe() -> None:
    wire = Wire()
    unnamed = found(ORIGINAL, True)
    unnamed.probe = None
    with pytest.raises(ValueError, match="names the ffprobe"):
        wire.client().put_captions(job(), unnamed)
    assert wire.sent == []


def test_the_header_probe_s_findings_on_the_wire() -> None:
    # streams of captions of their own, and what is off the closed lists,
    # with a read and with a failed one
    wire = Wire()
    r = found(ORIGINAL, True)
    r.caption_streams, r.off_list = 1, []
    wire.client().put_captions(job(), r)
    original = wire.sent[-1][2]["original"]
    assert (original["captionStreams"], original["offList"]) == (1, [])
    off = found(ORIGINAL, None, error="off the closed lists: codec:vc1, stream:data:bin_data")
    off.caption_streams, off.off_list = 0, ["codec:vc1", "stream:data:bin_data"]
    wire.client().put_captions(job(), off)
    assert wire.sent[-1][2]["original"] == {
        "error": "off the closed lists: codec:vc1, stream:data:bin_data", "ffprobe": PROBE,
        "sizeBytes": SIZE, "captionStreams": 0,
        "offList": ["codec:vc1", "stream:data:bin_data"]}  # fmt: skip
    # a read that never got to the header probe says neither, and one that
    # found no file no size
    wire.client().put_captions(job(), found(ORIGINAL, None, error="no file to read",
                                            file_size=None))  # fmt: skip
    assert set(wire.sent[-1][2]["original"]) == {"error", "ffprobe"}


def windows_read(path: str) -> captions.CaptionsRead:
    """A read whose windows decoded: the start and 10 % with captions, 30 %
    failed, 50 % without (of a 600 s video)."""
    ws = [
        captions.Window(0.0, 0.0, frames=48, captions=True, start=0.0, end=1.88),
        captions.Window(0.1, 60.0, frames=48, captions=True, start=58.0, end=59.88,
                        interlaced=48),
        captions.Window(0.3, 180.0, frames=0, error="its first frame is at 1.400 s, 178.6 s from "
                        "where it was placed (180.000 s): it landed elsewhere"),
        captions.Window(0.5, 300.0, frames=48, captions=False, start=298.0, end=299.88,
                        repeats=12, keyframe_before=True),
    ]  # fmt: skip
    return captions.CaptionsRead(path=path, ok=True, captions=True, frames=144, windows=ws,
                                 duration=600.0, probe=PROBE, file_size=SIZE)


def test_every_window_goes_on_the_wire_and_a_failed_one_stays_failed() -> None:
    wire = Wire()
    wire.client().put_captions(job(), windows_read(ORIGINAL))
    body = wire.sent[0][2]
    want = [
        {"at": 0.0, "start": 0.0, "end": 1.88, "frames": 48, "captions": True,
         "interlacedFrames": 0, "repeatFrames": 0, "keyframeBefore": False},
        {"at": 0.1, "start": 58.0, "end": 59.88, "frames": 48, "captions": True,
         "interlacedFrames": 48, "repeatFrames": 0, "keyframeBefore": False},
        {"at": 0.3, "start": None, "end": None, "frames": 0, "captions": None,
         "interlacedFrames": None, "repeatFrames": None, "keyframeBefore": None,
         "error": "its first frame is at 1.400 s, 178.6 s from where it was placed "
                  "(180.000 s): it landed elsewhere"},
        {"at": 0.5, "start": 298.0, "end": 299.88, "frames": 48, "captions": False,
         "interlacedFrames": 0, "repeatFrames": 12, "keyframeBefore": True},
    ]  # fmt: skip
    assert body["original"]["captionWindows"] == want
    assert set(body) == {"token", "original"}
    assert set(body["original"]) == {"captionWindows", "duration", "sizeBytes", "ffprobe"}
    # round trip: the windows read back from the JSON are those read
    back = [captions.Window(w["at"], frames=w["frames"], captions=w["captions"],
                            start=w["start"], end=w["end"], error=w.get("error"),
                            interlaced=w["interlacedFrames"] or 0,
                            repeats=w["repeatFrames"] or 0,
                            keyframe_before=bool(w["keyframeBefore"]))
            for w in json.loads(json.dumps(body["original"]["captionWindows"]))]  # fmt: skip

    def key(w: captions.Window) -> tuple:
        return (w.at, w.start, w.end, w.frames, w.captions, w.error,
                0 if w.error else w.interlaced, 0 if w.error else w.repeats,
                False if w.error else w.keyframe_before)  # fmt: skip

    assert [key(w) for w in back] == [key(w) for w in windows_read(ORIGINAL).windows]


def test_a_window_that_failed_is_said_with_its_original_and_why(monkeypatch) -> None:
    # the catalog hands the original out again; the log names the window
    reads(monkeypatch, **{ORIGINAL: windows_read(ORIGINAL)})
    catalog = Catalog(claim(job()))
    with capture_logs() as logs:
        backfill(catalog).claim_once()
    failed = [(x["source_id"], x["window"], x["why"]) for x in logs
              if x["event"] == "captions.backfill_window_failed"]  # fmt: skip
    assert failed == [("s1", "30%", "its first frame is at 1.400 s, 178.6 s from where it was "
                                    "placed (180.000 s): it landed elsewhere")]  # fmt: skip
    assert len(catalog.put) == 1  # reported all the same, the failed window as failed


def test_a_read_without_the_size_of_its_file_is_refused() -> None:
    # the catalog takes a read of the file it handed out alone: a read of
    # frames says the size it found; a failed one, or one of no frame, needn't
    wire = Wire()
    wire.client().put_captions(job(), found(ORIGINAL, True, file_size=None))
    why = catalog_refuses(wire.sent[-1][2]["original"])
    assert why == "a read says the size of the file it read"
    for r in (found(ORIGINAL, None, error="no file to read", file_size=None),
              found(ORIGINAL, False, 0, file_size=None)):  # fmt: skip
        wire.client().put_captions(job(), r)
        assert catalog_refuses(wire.sent[-1][2]["original"]) is None


@pytest.mark.parametrize("status", [400, 404, 503])
def test_a_report_the_catalog_refuses_is_false(status: int) -> None:
    assert Wire(put=status).client().put_captions(job(), found(ORIGINAL, True)) is False


def catalog_refuses(report: dict) -> str | None:
    """katalog-manager's check of an original's report (internal/captions,
    FileReport.check, at 02adbf7), a null read as Go decodes it (0, false):
    why it answers 400, None when it takes it."""
    version = str(report.get("ffprobe") or "").strip()
    if not version:
        return "no ffprobe"
    m = re.match(r"^(?i:ffprobe\s+version\s+)?n?(\d+)\.(\d+)", version)
    if not m or (m.group(1), m.group(2)) != ("7", "1"):
        return "not ffprobe 7.1"
    if str(report.get("error") or "").strip():
        return None
    d = report.get("duration")
    if d is not None and not d > 0:
        return "duration not above 0"
    windows = report.get("captionWindows") or []
    if not windows:
        return "no windows"
    streams = report.get("captionStreams")
    if streams is not None and streams < 0:
        return "negative captionStreams"
    for w in windows:
        at, start, end = (w.get(k) or 0.0 for k in ("at", "start", "end"))
        frames, shown = w.get("frames") or 0, bool(w.get("captions"))
        if not 0 <= at <= 1:
            return "at out of range"
        if start < 0 or end < start:
            return "end before start"
        if frames < 0:
            return "negative frames"
        if frames == 0 and shown:
            return "captions in no frame"
    if not sum(w.get("frames") or 0 for w in windows):
        return None  # a read of no frame: a failed one
    size = report.get("sizeBytes")
    if size is None or size < 0:
        return "a read says the size of the file it read"
    return None


@pytest.mark.parametrize(
    "read",
    [
        windows_read(ORIGINAL),  # captions, a window that failed, one without
        found(ORIGINAL, True),
        found(ORIGINAL, False, 1),
        found(ORIGINAL, None, error="no video stream to read"),
        found(ORIGINAL, True, timed_out=True, error="ffprobe did not answer within 120 s"),
        captions.CaptionsRead(path=ORIGINAL, ok=False, probe="ffprobe version 7.1.5", windows=[
            captions.Window(0.0, None, frames=12, error="ffprobe found no captions and said: "
                            "Read error", start=0.0, end=0.44),
            captions.Window(0.9, ("time", 540.0), frames=0, captions=False)]),
        captions.CaptionsRead(path=ORIGINAL, ok=True, captions=True, probe="ffprobe version 7.1.5",
                              windows=[captions.Window(0.0, None, frames=3, captions=True,
                                                       start=-0.04, end=0.04)]),
    ],
)
def test_what_goes_on_the_wire_is_what_the_catalog_takes(read) -> None:
    wire = Wire()
    wire.client().put_captions(job(), read)
    body = wire.sent[0][2]
    assert set(body) == {"token", "original"}
    assert catalog_refuses(body["original"]) is None, body["original"]
    # a window that failed decoded no frame and shows none: never "none"
    for w in body["original"].get("captionWindows") or []:
        if w["frames"] == 0:
            assert w["captions"] is None and w["error"]
