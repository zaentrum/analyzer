"""What the captions backfill sends the catalog, held to a fixture: for each
kind of read - windows that show none and captions, a long group of
pictures, a picture that starts late, a short picture read whole (with
leading pictures, and with a start window replaced by it), HEVC that is not
known and HEVC whose packets carry captions, a window that failed, a start
window that placed no frame, reads that failed as a whole (a keyframes'
listing that is none, a claim of another count of packets among them) - the
body of its PUT
/api/captions/sources/{sourceId}, byte for byte as the backfill sends it
after a claim (through KatalogClient, the made-up ffprobe answering), with
what the catalog holds it to: the stored probe of the original (the header
the made-up ffprobe gives), its record's durationMs, the size of its file,
and what the analyzer means by the read (verdict: "none", "captions",
"unscanned" or "not known").

The fixture (REPORTS, beside this file) is the analyzer's side of the
contract: a test of the catalog's can take every body in it through its own
decoding and checks, with each case's records, and say whether it reads each
as the analyzer means it. A change to what goes on the wire fails here until
the fixture is written again (CAPTIONS_REPORTS_WRITE=1, this module alone)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from captions_fake import FakeProbe, Video, frames

from analyzer import captions_backfill
from analyzer.katalog import KatalogClient
from analyzer.pipelines import captions

REPORTS = Path(__file__).with_name("captions-reports.json")
TOKEN = "the-claim-s-token"
SIZE = 600 * 25 * 20_000  # the made-up original's bytes
FILM = {"index": 0, "codec_type": "video", "codec_name": "h264", "height": 1080}
HEVC = {**FILM, "codec_name": "hevc"}
SAID = "[h264 @ 0x1] error while decoding MB 12 7, bytestream -5\n"

# the start window's decode of a 90 s picture, as it would answer, and a line on stderr
START_SAID = {"frames": frames(48, with_captions=False), "streams": [{"closed_captions": 0}]}

# case: (the video the made-up ffprobe answers of, its record's durationMs,
# and how the case differs: the packets its claim names, or one run's own
# answer - its entries, its stdout and its stderr)
CASES: dict[str, tuple[Video, int, dict]] = {
    "windows-none": (Video(), 600_000, {}),
    "windows-captions": (Video(captions=[[290.0, 310.0]]), 600_000, {}),
    # groups of 13 s: windows up to 11 s before their places, which the gop
    # it reports lets them lie (max(10 s, 13 s + 0.5 s))
    "windows-gop-13s": (Video(gop=13.0), 600_000, {}),
    "windows-start-late": (
        Video(first_key=5.0, streams=[{**FILM, "start_time": "5.000000", "duration": "595.0"}]),
        600_000,
        {},
    ),
    "whole-none": (Video(length=90.0), 90_000, {}),
    "whole-captions": (Video(length=90.0, captions=[[30.0, 35.0]]), 90_000, {}),
    # 10 packets before its first keyframe: its leading pictures, counted
    "whole-leading-10": (Video(length=90.0, first_key=0.4), 90_000, {}),
    # its start window's decode says something and is replaced by a clean
    # whole read: the verdict rests on the whole run
    "whole-start-window-said": (
        Video(length=90.0),
        90_000,
        {"answer": ("start", START_SAID, SAID)},
    ),
    "hevc-not-known": (Video(streams=[HEVC]), 600_000, {}),
    "hevc-captions": (Video(streams=[HEVC], bytes_captions=[[290.0, 310.0]]), 600_000, {}),
    "window-failed": (Video(untimed=[[300.03, 302.0]]), 600_000, {}),
    # its first keyframe 60 s in: the start window places no frame, and the
    # read says no span (a span of none is no body the catalog takes)
    "start-places-no-frame": (Video(first_key=60.0, gop=4.0), 600_000, {}),
    "failed-header-said": (Video(said={"header": SAID}), 600_000, {}),
    "failed-off-list": (Video(streams=[{**FILM, "codec_name": "vc1"}]), 600_000, {}),
    "failed-rate-below-one": (Video(streams=[{**FILM, "avg_frame_rate": "1/2"}]), 600_000, {}),
    # its keyframes' listing answers nothing, exit 0: not known, never a gop of 0
    "failed-keyframes-listing": (Video(), 600_000, {"answer": ("packet=pts_time,flags", "", "")}),
    # a claim that names 24 packets a window: one the read can't serve
    "failed-claim-24-packets": (Video(), 600_000, {"packets": 24}),
}


def verdict(r: captions.CaptionsRead) -> str:
    """What the analyzer means by a read: "captions", "none", "unscanned"
    (HEVC whose windows can't tell), else "not known"."""
    if r.ok:
        return "captions" if r.captions else "none"
    return "unscanned" if not r.captions_known else "not known"


def _sent(
    ff: FakeProbe, path: Path, duration_ms: int, monkeypatch: pytest.MonkeyPatch, packets: int
) -> tuple[bytes, captions.CaptionsRead]:
    """The body of the PUT the backfill sends after a claim of path, and the
    read it reports."""
    put: list[bytes] = []

    def catalog(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        if request.url.path == "/api/captions/claim":
            deadline = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600))
            return httpx.Response(200, json={"items": [{
                "sourceId": "src-1", "itemId": "item-1", "path": str(path), "packets": packets,
                "token": TOKEN, "durationMs": duration_ms}], "deadline": deadline})  # fmt: skip
        assert (request.method, request.url.path) == ("PUT", "/api/captions/sources/src-1")
        put.append(request.content)
        return httpx.Response(200, json={"state": "read"})

    client = KatalogClient("http://catalog.test", "http://catalog.test/token", "w", "not-a-secret")
    client._http = httpx.Client(transport=httpx.MockTransport(catalog))
    reads: list[captions.CaptionsRead] = []
    real = captions.read

    def read(*args: Any, **kwargs: Any) -> captions.CaptionsRead:
        reads.append(real(*args, **kwargs))
        return reads[-1]

    with monkeypatch.context() as m:
        m.setattr(captions, "read", read)
        captions_backfill.CaptionsBackfill(
            client, threading.Event(), "analyzer-test", ffprobe=str(ff.path)
        ).claim_once()
    [body], [r] = put, reads
    return body, r


def _header(ff: FakeProbe, path: Path) -> dict:
    """The header the made-up ffprobe gives of the file: the stored probe a
    record of it keeps."""
    out = subprocess.run([str(ff.path), *captions.header_args(), str(path)],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    return json.loads(out)


def _call(ff: FakeProbe, path: Path, entries: str) -> int:
    """Which run of a read lists entries (its number, from 1); "start" the
    start window's decode."""
    captions.read(str(path), ffprobe=str(ff.path))
    for n, run in enumerate(ff.runs, 1):
        shown = run[run.index("-show_entries") + 1]
        if shown == entries or (entries == "start" and "-show_frames" in run):
            return n
    raise AssertionError(f"no run of {entries}")


def reports(fake_ffprobe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    """Every case's report and what the catalog holds it to."""
    out = {}
    for name, (video, duration_ms, how) in CASES.items():
        path = tmp_path / name / "Film (2024).mkv"
        path.parent.mkdir()
        with path.open("wb") as f:
            f.truncate(SIZE)
        ff = fake_ffprobe(video=video)
        if "answer" in how:
            entries, answer, err = how["answer"]
            call = _call(fake_ffprobe(video=video), path, entries)
            ff = fake_ffprobe(video=video).answers(answer, err=err, call=call)
        body, r = _sent(ff, path, duration_ms, monkeypatch, how.get("packets", captions.PACKETS))
        out[name] = {
            "verdict": verdict(r),
            "durationMs": duration_ms,
            "sizeBytes": SIZE,
            "probe": _header(fake_ffprobe(video=video), path),
            "sent": body.decode(),  # the body as sent, its bytes
        }
    return out


def test_what_goes_on_the_wire_is_the_fixture(fake_ffprobe, tmp_path: Path, monkeypatch) -> None:
    got = reports(fake_ffprobe, tmp_path, monkeypatch)
    if os.environ.get("CAPTIONS_REPORTS_WRITE") == "1":
        REPORTS.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    assert got == json.loads(REPORTS.read_text()), "CAPTIONS_REPORTS_WRITE=1 writes it again"


def test_the_fixture_has_every_kind_of_read() -> None:
    cases = json.loads(REPORTS.read_text())
    assert set(cases) == set(CASES)
    verdicts = {c["verdict"] for c in cases.values()}
    assert verdicts == {"none", "captions", "unscanned", "not known"}
    for name, c in cases.items():
        body = c["body"] = json.loads(c["sent"])
        assert set(body) == {"token", "original"} and body["token"] == TOKEN, name
        assert c["sent"].count(TOKEN) == 1, name  # a catalog's test puts its own claim's there
        original = body["original"]
        assert original["ffprobe"].startswith("ffprobe version n7.1."), name
        if c["verdict"] in ("none", "captions", "unscanned"):
            # a read: its windows, the span it measured and its gop
            assert {"captionWindows", "duration", "span", "gop", "sizeBytes"} <= set(original), name
            assert "error" not in original, name
        if c["verdict"] == "none":
            assert all(w["frames"] >= captions.MIN_WINDOW_FRAMES and w["captions"] is False
                       for w in original["captionWindows"]), name  # fmt: skip
    said = cases["whole-start-window-said"]
    assert said["verdict"] == "none" and said["body"]["original"]["captionWindows"][0]["whole"]
    whole = cases["whole-none"]["body"]["original"]["captionWindows"]
    assert [(w["at"], w.get("whole")) for w in whole] == [(0.0, True)]
    assert cases["windows-gop-13s"]["body"]["original"]["gop"] == pytest.approx(13.0)
    lead = cases["whole-leading-10"]["body"]["original"]["captionWindows"][0]
    assert (lead["leading"], lead["packets"] - lead["frames"]) == (10, 10)
    assert "span" not in cases["start-places-no-frame"]["body"]["original"]
    assert "gop" not in cases["failed-keyframes-listing"]["body"]["original"]
    claim = cases["failed-claim-24-packets"]["body"]["original"]
    assert claim["error"].endswith("a read it can't serve")
    unscanned = cases["hevc-not-known"]["body"]["original"]
    assert (unscanned["captionsKnown"], unscanned["reason"]) == (False, captions.HEVC_REASON)
