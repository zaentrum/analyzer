"""The captions read and backfill with the pipeline's real ffprobe 7.1, on
originals ffmpeg makes: five minutes of H.264 with an A53 caption SEI before
every picture (in Matroska, MP4 and MPEG-TS), the same without, captions that
start 100 s in (Matroska, MPEG-TS), HEVC that libx265 encoded with -a53cc 1
and without, a clip cut inside a group of pictures with and without
captions; and the reads that can't vouch for "none" and fail: a transport
stream whose clock starts again (read as one that wraps), a Matroska file
whose header says a tenth of its length, a raw stream (frames without
times), a short clip whose windows lie on one another, a file that isn't a
video and one without video; and the backfill reading an original as the
catalog hands it out, through the client, to the JSON it sends. An ffprobe
of another release on the PATH (the image's 4.4.2, a CI runner's, a
workstation's) is refused.

They run with CAPTIONS_FFPROBE naming an ffprobe of the 7.1 release, and an
ffmpeg 7.1 with libx264 and libx265 to make the clips (CAPTIONS_TEST_FFMPEG,
else the ffmpeg beside that ffprobe); skipped otherwise. CAPTIONS_NOKEY may
name a file whose first packets come before its first keyframe, to read too.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import struct
import subprocess
import threading
from pathlib import Path

import httpx
import pytest

from analyzer import captions_backfill
from analyzer.katalog import KatalogClient, _captions_read
from analyzer.pipelines import captions

FFPROBE = os.environ.get("CAPTIONS_FFPROBE") or ""
FFMPEG = os.environ.get("CAPTIONS_TEST_FFMPEG") or (
    str(Path(FFPROBE).with_name("ffmpeg")) if FFPROBE else ""
)


def _usable() -> str | None:
    """Why these tests can't run here; None when they can."""
    if not FFPROBE:
        return "CAPTIONS_FFPROBE names no ffprobe 7.1"
    captions.probe.cache_clear()
    why = captions.probe(FFPROBE).refusal()
    if why:
        return why
    if not FFMPEG or not os.access(FFMPEG, os.X_OK):
        return f"no ffmpeg to make the clips at {FFMPEG or '(none)'} (CAPTIONS_TEST_FFMPEG)"
    encoders = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True,
                              text=True, check=False).stdout  # fmt: skip
    if "libx264" not in encoders or "libx265" not in encoders:
        return f"{FFMPEG} has no libx264 or libx265"
    return None


pytestmark = pytest.mark.skipif(_usable() is not None, reason=str(_usable()))

KEYS = "keyint=48:min-keyint=48:scenecut=0"
X264 = ["-c:v", "libx264", "-preset", "ultrafast", "-x264-params", f"{KEYS}:slices=1"]
X265 = ["-c:v", "libx265", "-preset", "ultrafast", "-x265-params", f"log-level=error:{KEYS}"]
# cc_data as a broadcast carries it with each picture: process_cc_data_flag
# and two field-1 byte pairs, a caption's control codes in odd parity.
CC_DATA = bytes([0x40 | 2, 0xFF, 0xFC, 0x94, 0x2C, 0xFC, 0xC8, 0x49, 0xFF])
A53 = bytes([0xB5, 0x00, 0x31]) + b"GA94" + bytes([0x03]) + CC_DATA  # ITU-T T.35, ATSC
SIDE = "ATSC A53 Part 4 Closed Captions"
LONG = 300  # seconds: a length whose windows lie apart


def _ff(*args: str) -> None:
    subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "fatal", "-y", *args],
                   check=True)  # fmt: skip


def _with_captions(raw: Path, out: Path, mode: str = "all") -> None:
    """The H.264 elementary stream at `raw` again at `out`, an A53 cc_data
    SEI (payload type 4) before the slice of each picture (mode "all"), of
    its IDR pictures ("idr"), or of the n-th picture and those after it
    ("late:n"); "nokey" drops the first picture's slice instead (an IDR),
    so the stream starts before its next keyframe."""
    sei = b"\x00\x00\x00\x01" + bytes([0x06, 0x04, len(A53)]) + A53 + b"\x80"
    late = int(mode.split(":")[1]) if mode.startswith("late:") else None
    data, made, pos, n = raw.read_bytes(), bytearray(), 0, 0
    for m in re.finditer(rb"\x00\x00\x01(.)", data, re.DOTALL):
        start = m.start() - 1 if m.start() and data[m.start() - 1] == 0 else m.start()
        kind = m.group(1)[0] & 0x1F
        if kind not in (1, 5):  # a slice: one per picture
            continue
        if mode == "nokey" and n == 0:
            made += data[pos:start]
            nxt = re.compile(rb"\x00\x00\x01").search(data, m.end())
            pos = len(data) if nxt is None else nxt.start() - (data[nxt.start() - 1] == 0)
        elif mode == "all" or (mode == "idr" and kind == 5) or (late is not None and n >= late):
            made += data[pos:start] + sei
            pos = start
        n += 1
    out.write_bytes(bytes(made + data[pos:]))


def _captioned(path: Path) -> tuple[int, int]:
    """(video frames, those whose SEI carries A53 closed captions) of a file,
    every frame of it decoded."""
    got = subprocess.run(
        [FFPROBE, "-v", "fatal", "-select_streams", "v:0", "-show_frames", "-show_entries",
         "frame_side_data=side_data_type", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    frames = json.loads(got).get("frames") or []
    return len(frames), sum(
        1
        for f in frames
        if any(x.get("side_data_type") == SIDE for x in f.get("side_data_list") or [])
    )


def _first_key(path: Path) -> int:
    flags = subprocess.run(
        [FFPROBE, "-v", "fatal", "-select_streams", "v:0", "-read_intervals", "%+#200",
         "-show_entries", "packet=flags", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    return next(i for i, f in enumerate(flags) if f.startswith("K"))


@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, Path]:
    """The originals, 25 fps, 160x120, a keyframe every 48 pictures; skips
    where this ffmpeg makes them otherwise."""
    d = tmp_path_factory.mktemp("captions")
    c: dict[str, Path] = {}

    def at(name: str) -> Path:
        c[name] = d / name
        return c[name]

    picture = ["-f", "lavfi", "-i", f"testsrc2=size=160x120:rate=25:duration={LONG}"]
    tone = ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={LONG}"]
    audio = ["-c:a", "aac", "-ac", "2", "-metadata:s:a:0", "language=eng"]
    raw, cc, late = d / "plain.h264", d / "cc.h264", d / "late.h264"
    _ff(*picture, *X264, "-f", "h264", str(raw))
    _with_captions(raw, cc)
    _with_captions(raw, late, "late:2500")  # from 100 s on
    _ff("-framerate", "25", "-i", str(cc), *tone, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
        *audio, str(at("h264-cc.mkv")))
    for name in ("h264-cc.mp4", "h264-cc.ts"):
        _ff("-framerate", "25", "-i", str(cc), "-c:v", "copy", str(at(name)))
    _ff("-framerate", "25", "-i", str(raw), "-c:v", "copy", str(at("h264.mkv")))
    for name in ("late.mkv", "late.ts"):
        _ff("-framerate", "25", "-i", str(late), "-c:v", "copy", str(at(name)))
    _ff("-i", str(c["h264-cc.mkv"]), "-map", "0:v", *X265, "-a53cc", "1", str(at("hevc-cc.mkv")))
    _ff("-i", str(c["h264.mkv"]), "-map", "0:v", *X265, str(at("hevc.mkv")))
    # cut inside a group of pictures, as a capture is: a keyframe every 100
    # pictures, the first one left out and the 99 after it kept (-copyinkf)
    long_gop = d / "long-gop.h264"
    _ff(*picture, "-c:v", "libx264", "-preset", "ultrafast", "-x264-params",
        "keyint=100:min-keyint=100:scenecut=0:slices=1", "-f", "h264", str(long_gop))
    _with_captions(long_gop, d / "long-gop-cc.h264")
    for src, name in (("long-gop-cc.h264", "mid-cc.mkv"), ("long-gop.h264", "mid.mkv")):
        _ff("-framerate", "25", "-i", str(d / src), "-c:v", "copy", str(d / f"full-{name}"))
        _ff("-i", str(d / f"full-{name}"), "-ss", "0.04", "-copyinkf", "-map", "0:v", "-c",
            "copy", str(at(name)))
    # a capture whose clock starts again: 30 s on a clock at 1000 s without
    # captions, then the rest on a clock from 1.4 s, captions 100 s in
    a, b = d / "reset-a.ts", d / "reset-b.ts"
    _ff("-framerate", "25", "-i", str(raw), "-t", "30", "-c:v", "copy", "-output_ts_offset",
        "1000", str(a))
    _ff("-framerate", "25", "-i", str(late), "-c:v", "copy", str(b))
    at("reset.ts").write_bytes(a.read_bytes() + b.read_bytes())
    # a Matroska file whose header says 30 s of its 300
    mkv = bytearray(c["late.mkv"].read_bytes())
    i = mkv.find(bytes([0x44, 0x89, 0x88]))  # Duration, an 8-byte float
    mkv[i + 3:i + 11] = struct.pack(">d", 30_000.0)
    at("short-header.mkv").write_bytes(bytes(mkv))
    c["raw-cc"], c["raw-plain"] = cc, raw
    _ff("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=8", *X264,
        "-a53cc", "1", str(at("short.mkv")))
    _ff(*tone[:3], "sine=frequency=440:sample_rate=48000:duration=4", *audio, str(at("audio.mka")))
    at("junk.mkv").write_bytes(b"\x1a\x45\xdf\xa3 not a Matroska file")
    made = {n: _captioned(c[n]) for n in ("hevc-cc.mkv", "hevc.mkv")}
    if made != {"hevc-cc.mkv": (LONG * 25, LONG * 25), "hevc.mkv": (LONG * 25, 0)} \
            or _first_key(c["mid.mkv"]) != 99:  # fmt: skip
        pytest.skip(f"this ffmpeg makes other clips than these tests need: {made}")
    return c


def read(path: Path, **kw) -> captions.CaptionsRead:
    # the claim's durationMs: the file's length, as its source record takes it
    kw.setdefault("duration_ms", _record_ms(path))
    return captions.read(str(path), ffprobe=FFPROBE, **kw)


def _record_ms(path: Path) -> int | None:
    args = [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    out = subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()
    try:
        return round(float(out) * 1000)
    except ValueError:
        return None


@pytest.mark.parametrize("name", ["h264-cc.mkv", "h264-cc.mp4", "h264-cc.ts", "hevc-cc.mkv"])
def test_an_original_with_a53_captions_says_so_in_every_window(clips, name: str) -> None:
    r = read(clips[name])
    assert (r.ok, r.captions, r.error) == (True, True, None), r.note
    # 90 % at most 30 s short of the picture's end (its span a frame short of 300 s)
    assert [w.at for w in r.windows] == pytest.approx([0.0, 0.1, 0.3, 0.5, 0.7, 0.9], abs=1e-3)
    assert all(w.captions and w.frames > 0 for w in r.windows)
    assert abs(r.duration - LONG) < 1
    assert captions.probe(FFPROBE).branch == "7.1" and r.probe == captions.probe(FFPROBE).name


@pytest.mark.parametrize("name", ["h264.mkv", "hevc.mkv"])
def test_an_original_without_captions_says_none_every_window_decoded(clips, name: str) -> None:
    r = read(clips[name])
    if name.startswith("hevc"):  # never "none": every window read, and not known
        assert (r.ok, r.captions, r.captions_known) == (False, None, False), r.note
        assert r.reason == "hevc: needs the packets' signature" and len(r.windows) == 6
    else:
        assert (r.ok, r.captions, r.error) == (True, False, None), r.note
    assert len(r.windows) == 6 and all(w.frames > 0 and not w.error for w in r.windows)
    # the frame step its header probe states (25 a second), not one measured
    assert (r.step, r.step_from) == (pytest.approx(0.04), "avg_frame_rate")
    # each window where it was placed: its first frame the keyframe before
    for w in r.windows:
        assert w.aim - 2.0 <= w.start <= w.aim + 0.04 and w.end > w.start


@pytest.mark.parametrize("name", ["late.mkv", "late.ts"])
def test_captions_that_start_later_are_found_in_a_later_window(clips, name: str) -> None:
    r = read(clips[name])
    assert (r.ok, r.captions) == (True, True), r.note
    # 100 s in: the start and 10 % (30 s) show none, 30 % (90 s) on show them
    assert [w.captions for w in r.windows] == [False, False, False, True, True, True]


def test_an_original_cut_inside_a_group_of_pictures_states_no_start_and_is_not_read(clips):
    # Matroska cut inside a group of pictures (-copyinkf): its first packets
    # have no decoding time, and ffprobe states no start_time, of its picture
    # or its format: no plan, the read fails (not known, captions or not)
    for name in ("mid-cc.mkv", "mid.mkv"):
        r = read(clips[name])
        assert (r.ok, r.captions, r.unreadable, r.windows) == (False, None, True, []), r.note
        assert r.error.startswith("its records state no start of its picture"), r.error


# ------------------------------------------------- reads that fail


def test_a_clock_that_starts_again_fails_the_read_never_none(clips) -> None:
    # read as a clock that wrapped: some 26 hours, the windows past the end
    r = read(clips["reset.ts"])
    assert (r.ok, r.captions, r.unreadable) == (False, None, True), r.note
    assert r.duration > 20_000 and "more than 10 times" in r.error
    assert set(_captions_read(r)) == {"error", "ffprobe", "sizeBytes", "captionStreams", "offList"}


def test_a_header_that_says_a_tenth_of_the_length_fails_the_read_never_none(clips) -> None:
    r = read(clips["short-header.mkv"])
    assert (r.ok, r.captions, r.unreadable) == (False, None, True), r.note
    # its record takes the header's 30 s, and the picture runs on past it
    assert r.error.startswith("its picture ends at 30.000 s as its records state it"), r.error


def test_a_raw_stream_fails_the_read(clips) -> None:
    # its format is no container of the list: frames without times
    r = read(clips["raw-cc"])
    assert (r.ok, r.captions, r.unreadable, r.off_list) == (False, None, True, ["container:h264"])


def test_a_short_clip_whose_windows_can_t_lie_apart_is_read_whole(clips) -> None:
    # 8 s, a keyframe every 2 s: the short-picture rule reads every packet
    r = read(clips["short.mkv"])
    [w] = r.windows
    assert (r.ok, r.captions, w.whole, w.error) == (True, False, True, None), r.note or r.error
    assert w.frames == w.packets > 0 and w.start == 0.0


@pytest.mark.skipif(not os.environ.get("CAPTIONS_NOKEY"), reason="CAPTIONS_NOKEY names no file")
def test_a_file_before_its_first_keyframe_never_says_none_unread() -> None:
    r = read(Path(os.environ["CAPTIONS_NOKEY"]))
    assert r.start_packet > 0 or r.windows[0].error
    if r.captions is False:
        assert all(w.frames > 0 and not w.error for w in r.windows)


@pytest.mark.parametrize(
    ("name", "starts"),
    [("junk.mkv", "ffprobe exited 1"), ("audio.mka", "no video stream to read")],
)
def test_a_file_that_cant_tell_fails_never_no_captions(clips, name: str, starts: str) -> None:
    r = read(clips[name])
    assert (r.ok, r.captions, r.frames) == (False, None, 0) and r.error.startswith(starts)


def test_the_backfill_reads_an_original_to_the_json_it_sends(clips) -> None:
    sent: list[tuple[str, str, dict]] = []
    claims = [
        {"items": [{"sourceId": "s1", "itemId": "m1", "path": str(clips["late.mkv"]),
                    "packets": 48, "token": "tok-1", "durationMs": _record_ms(clips["late.mkv"])}],
         "deadline": "2999-01-01T00:00:00Z"},
        {"items": [{"sourceId": "s2", "itemId": "m2", "path": str(clips["reset.ts"]),
                    "packets": 48, "token": "tok-2", "durationMs": _record_ms(clips["reset.ts"])}],
         "deadline": "2999-01-01T00:00:00Z"},
    ]  # fmt: skip

    def catalog(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        body = json.loads(request.content)
        sent.append((request.method, request.url.path, body))
        if request.url.path == "/api/captions/claim":
            if claims:
                return httpx.Response(200, json=claims.pop(0))
            stop.set()
            return httpx.Response(200, json={"items": [], "idle": "done", "retryAfterSeconds": 60})
        return httpx.Response(200, json={"state": "read"})

    client = KatalogClient("http://catalog.test", "http://catalog.test/token", "w", "not-a-secret")
    client._http = httpx.Client(transport=httpx.MockTransport(catalog))
    stop = threading.Event()
    t = threading.Thread(
        target=captions_backfill.CaptionsBackfill(client, stop, "analyzer-0", ffprobe=FFPROBE).run,
        daemon=True,
    )
    t.start()
    t.join(timeout=60)
    stop.set()
    assert not t.is_alive()
    puts = [(path, body) for method, path, body in sent if method == "PUT"]
    name = captions.probe(FFPROBE).name
    assert [(path, set(body), set(body["original"])) for path, body in puts] == [
        ("/api/captions/sources/s1", {"token", "original"},
         {"captionWindows", "duration", "span", "gop", "sizeBytes", "ffprobe", "captionStreams",
          "offList"}),
        # (its clock is held before its keyframes are listed: no gop said)
        ("/api/captions/sources/s2", {"token", "original"},
         {"error", "ffprobe", "sizeBytes", "captionStreams", "offList"}),
    ]  # fmt: skip
    original = puts[0][1]["original"]
    assert original["ffprobe"] == name and abs(original["duration"] - LONG) < 1
    # a keyframe every 48 pictures at 25 a second: its gop, as the plan reads it
    assert original["gop"] == pytest.approx(48 / 25)
    ws = original["captionWindows"]
    assert [w["at"] for w in ws] == pytest.approx([0.0, 0.1, 0.3, 0.5, 0.7, 0.9], abs=1e-3)
    assert [set(w) for w in ws] == [
        {"at", "start", "end", "frames", "captions", "interlacedFrames", "repeatFrames",
         "keyframeBefore"}] * 6
    assert [w["captions"] for w in ws] == [False, False, False, True, True, True]
    assert all(w["frames"] > 0 and 0 <= w["start"] <= w["end"] for w in ws)
    # in the order of the video, none on another
    assert all(a["end"] < b["start"] for a, b in itertools.pairwise(ws))
    assert "more than 10 times" in puts[1][1]["original"]["error"]


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="no ffprobe on the PATH")
def test_the_path_s_own_ffprobe_of_another_release_is_refused() -> None:
    # The image's ffprobe is Ubuntu 22.04's 4.4.2; a CI runner's or a
    # workstation's another release again. None of them reads the captions.
    own = shutil.which("ffprobe") or ""
    captions.probe.cache_clear()
    p = captions.probe(own)
    if p.ok:
        pytest.skip(f"the PATH's ffprobe is of the 7.1 release ({p.line})")
    worker = captions_backfill.CaptionsBackfill(None, threading.Event(), "w", ffprobe=own)  # type: ignore[arg-type]
    assert worker.refusal() is not None and "not the pipeline's 7.1.x" in worker.refusal()
    worker.run()  # returns at once: it never asks the catalog (None here) for anything
    with pytest.raises(captions.WrongProbe):
        captions.read(str(Path(__file__)), ffprobe=own)


# ------------------------------------------------- the closed lists, real

def _nals(data: bytes):
    """An Annex B stream's NAL units, start codes stripped, in order."""
    starts = [m.start() for m in re.finditer(rb"\x00\x00\x01", data)]
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(data)
        nal = data[s + 3:end]
        if i + 1 < len(starts) and nal.endswith(b"\x00"):
            nal = nal[:-1]
        yield nal


def _to_suffix_sei(src: Path, out: Path) -> int:
    """The HEVC stream at src again at out, each prefix SEI (NAL 39) that
    carries the A53 payload moved after its picture's slice as a suffix SEI
    (NAL 40); returns how many moved."""
    held, made, moved = [], bytearray(), 0
    for nal in _nals(src.read_bytes()):
        kind = (nal[0] >> 1) & 0x3F
        if kind == 39 and b"GA94" in nal:
            held.append(bytes([(40 << 1) | (nal[0] & 1), nal[1]]) + nal[2:])
            moved += 1
            continue
        made += b"\x00\x00\x00\x01" + nal
        if kind < 32:  # a slice
            for h in held:
                made += b"\x00\x00\x00\x01" + h
            held = []
    out.write_bytes(bytes(made))
    return moved


def _leb128(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _with_t35(src: bytes) -> bytes:
    """A low-overhead AV1 stream (ffmpeg's -f obu) again, an ITU-T T.35
    metadata OBU (type 5, metadata type 4) with an A53 cc_data payload after
    the temporal delimiter of each temporal unit, as an encoder that carries
    captions writes it."""
    payload = _leb128(4) + A53 + b"\x80"  # its trailing bits
    meta = bytes([(5 << 3) | 0x02]) + _leb128(len(payload)) + payload
    out, i = bytearray(), 0
    while i < len(src):
        head = src[i]
        j = i + 1 + ((head >> 2) & 1)  # its extension byte
        assert (head >> 1) & 1, "an OBU without its size"
        size = shift = 0
        while True:
            x = src[j]
            size, shift, j = size | (x & 0x7F) << shift, shift + 7, j + 1
            if not x & 0x80:
                break
        out += src[i : j + size]
        if (head >> 3) & 0xF == 2:  # a temporal delimiter
            out += meta
        i = j + size
    return bytes(out)


def _encoders() -> str:
    return subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True,
                          check=False).stdout  # fmt: skip


@pytest.fixture(scope="module")
def listed(clips, tmp_path_factory) -> dict[str, Path]:
    """Originals for every entry of the closed lists a clip can be made of,
    five minutes each where they must read, and a short one of each axis
    off the lists."""
    d = tmp_path_factory.mktemp("listed")
    c: dict[str, Path] = {}
    raw, cc = clips["raw-plain"], clips["raw-cc"]
    picture = ["-f", "lavfi", "-i", f"testsrc2=size=176x144:rate=25:duration={LONG}"]

    def at(name: str) -> Path:
        c[name] = d / name
        return c[name]

    # containers: QuickTime with a timecode track, AVI; and MPEG-1/2 in a
    # program stream and a transport stream (A53 in MPEG-2's user data)
    _ff("-framerate", "25", "-i", str(cc), "-c:v", "copy", "-timecode", "01:00:00:00",
        str(at("tmcd.mov")))
    _ff("-i", str(clips["h264-cc.mkv"]), "-map", "0:v", "-c:v", "copy", "-bsf:v",
        "h264_mp4toannexb", str(at("cc.avi")))
    for name in ("mpeg2-cc.mpg", "mpeg2-cc.ts"):
        _ff("-i", str(clips["h264-cc.mkv"]), "-map", "0:v", "-c:v", "mpeg2video", "-a53cc", "1",
            "-g", "12", "-b:v", "1M", str(at(name)))
    _ff("-i", str(clips["h264.mkv"]), "-c:v", "mpeg1video", "-g", "12", "-b:v", "1M",
        str(at("mpeg1.mpg")))
    # streams on the list: subtitles and a font attached; a stream of
    # captions of its own (c608) beside a picture without them
    srt = d / "subs.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nA line\n\n")
    font = d / "font.ttf"
    font.write_bytes(b"\x00\x01\x00\x00" + b"\x00" * 64)
    _ff("-framerate", "25", "-i", str(raw), "-i", str(srt), "-map", "0:v", "-map", "1:s",
        "-c:v", "copy", "-c:s", "srt", "-attach", str(font), "-metadata:s:t",
        "mimetype=application/x-truetype-font", str(at("subs-font.mkv")))
    sub = d / "c608-track.mov"
    _ff("-f", "lavfi", "-i", f"movie={clips['h264-cc.mkv']}[out0+subcc]", "-map", "0:s",
        "-c:s", "copy", "-f", "mov", str(sub))
    _ff("-framerate", "25", "-i", str(raw), "-i", str(sub), "-map", "0:v", "-map", "1:s",
        "-c", "copy", str(at("c608.mov")))
    # HEVC whose A53 SEI follows its picture (a suffix SEI)
    prefix, suffix = d / "prefix.hevc", d / "suffix.hevc"
    # (no B-frames: a raw stream's pictures go into Matroska in their order)
    _ff("-i", str(clips["h264-cc.mkv"]), "-map", "0:v", "-c:v", "libx265", "-preset", "ultrafast",
        "-x265-params", f"log-level=error:{KEYS}:bframes=0", "-a53cc", "1", "-f", "hevc",
        str(prefix))
    if not _to_suffix_sei(prefix, suffix):
        pytest.skip("this libx265 wrote no A53 prefix SEI to move")
    for name in ("hevc-suffix.mkv", "hevc-suffix.ts"):
        _ff("-framerate", "25", "-i", str(suffix), "-c:v", "copy", str(at(name)))
    # attached pictures: an MP4 cover before the film, a poster and a
    # landscape cover attached to a Matroska file; and the same picture as a
    # video stream no container marks as attached, off the lists
    jpg, png = d / "poster.jpg", d / "cover_land.png"
    _ff("-f", "lavfi", "-i", "testsrc=size=320x480:rate=1:duration=1", "-frames:v", "1", str(jpg))
    _ff("-f", "lavfi", "-i", "testsrc=size=640x360:rate=1:duration=1", "-frames:v", "1", str(png))
    _ff("-i", str(jpg), "-framerate", "25", "-i", str(cc), "-map", "0:v", "-map", "1:v",
        "-c", "copy", "-disposition:v:0", "attached_pic", str(at("cover-first.mp4")))
    _ff("-framerate", "25", "-i", str(cc), "-c:v", "copy", "-attach", str(jpg), "-attach",
        str(png), "-metadata:s:t:0", "mimetype=image/jpeg", "-metadata:s:t:1",
        "mimetype=image/png", str(at("posters.mkv")))
    _ff("-i", str(jpg), "-framerate", "25", "-i", str(cc), "-map", "0:v", "-map", "1:v",
        "-c", "copy", str(at("off-still.mkv")))
    # in fields: H.264 whose Matroska header says progressive, MPEG-2 in a
    # transport stream
    _ff(*picture, "-vf", "setfield=tff", "-c:v", "libx264", "-preset", "ultrafast",
        "-x264-params", "tff=1:keyint=48:min-keyint=48:scenecut=0", str(d / "fields.mkv"))
    mkv = bytearray((d / "fields.mkv").read_bytes())
    i = mkv.find(bytes([0x9A, 0x81]))  # FlagInterlaced, one byte
    if i >= 0:
        mkv[i + 2] = 2  # progressive
    at("fields-said-progressive.mkv").write_bytes(bytes(mkv))
    _ff(*picture, "-vf", "setfield=tff", "-c:v", "mpeg2video", "-flags", "+ilme+ildct", "-top",
        "1", "-g", "12", "-b:v", "2M", str(at("fields-mpeg2.ts")))
    # HEVC coded as field pictures, where this libx265 has its interlace mode
    try:
        _ff(*picture, "-vf", "setfield=tff", "-c:v", "libx265", "-preset", "ultrafast",
            "-x265-params", f"log-level=error:interlace=tff:{KEYS}", str(at("fields-hevc.mkv")))
    except subprocess.CalledProcessError:
        c.pop("fields-hevc.mkv")
    # an arrival read while it is copied: a transport stream whose captions
    # start 100 s in, its first 40 % of bytes
    whole = clips["late.ts"].read_bytes()
    at("arriving.ts").write_bytes(whole[: (len(whole) * 4 // 10) // 188 * 188])
    # AV1, where this ffmpeg encodes it and the ffprobe decodes it
    enc = _encoders()
    av1 = next((e for e in ("libsvtav1", "libaom-av1") if e in enc), None)
    if av1:
        obu, cc_obu = d / "av1.obu", d / "av1-cc.obu"
        _ff("-i", str(clips["h264.mkv"]), "-map", "0:v", "-c:v", av1, *(["-preset", "12"]
            if av1 == "libsvtav1" else ["-cpu-used", "8"]), "-g", "48", "-f", "obu",
            str(obu))  # fmt: skip
        cc_obu.write_bytes(_with_t35(obu.read_bytes()))
        for src, name in ((obu, "av1.mkv"), (cc_obu, "av1-cc.mkv")):
            _ff("-framerate", "25", "-f", "obu", "-i", str(src), "-c", "copy", str(at(name)))
    # off the lists, one of each axis: a container, a codec, a stream kind,
    # a second film, a picture that keeps the blanking lines
    short = ["-f", "lavfi", "-i", "testsrc2=size=176x144:rate=25:duration=4"]
    _ff(*short, "-c:v", "libx264", "-preset", "ultrafast", str(at("off-container.flv")))
    _ff(*short, "-c:v", "mpeg4", str(at("off-codec.avi")))
    blob = d / "blob.bin"
    blob.write_bytes(bytes(range(256)) * 16)
    _ff(*short, "-f", "data", "-i", str(blob), "-map", "0:v", "-map", "1", "-c:v", "libx264",
        "-preset", "ultrafast", "-c:d", "copy", str(at("off-stream.ts")))
    _ff(*short, *short, "-map", "0:v", "-map", "1:v", "-c:v", "libx264", "-preset", "ultrafast",
        str(at("off-two-films.mkv")))
    _ff("-f", "lavfi", "-i", "testsrc2=size=720x486:rate=30000/1001:duration=4", "-c:v",
        "libx264", "-preset", "ultrafast", str(at("off-vbi.mkv")))
    return c


@pytest.mark.parametrize(
    ("name", "shown"),
    [("tmcd.mov", True), ("cc.avi", True), ("mpeg2-cc.mpg", True), ("mpeg2-cc.ts", True),
     ("mpeg1.mpg", False), ("subs-font.mkv", False)],
)
def test_containers_codecs_and_streams_on_the_list_are_read(listed, name, shown) -> None:
    r = read(listed[name])
    assert (r.ok, r.captions, r.off_list) == (True, shown, []), r.note or r.error
    assert all(w.frames > 0 and not w.error for w in r.windows)


def test_a_stream_of_captions_of_its_own_is_captions(listed) -> None:
    # the picture carries none, the c608 track does
    r = read(listed["c608.mov"])
    assert (r.ok, r.captions, r.caption_streams) == (True, True, 1), r.note or r.error
    assert not any(w.captions for w in r.windows)


@pytest.mark.parametrize("name", ["hevc-suffix.mkv", "hevc-suffix.ts"])
def test_hevc_captions_in_a_suffix_sei_are_found_in_its_bytes(listed, name: str) -> None:
    # 7.1 exports no side data of a suffix SEI: the packets' bytes hold them
    r = read(listed[name])
    assert (r.ok, r.captions, r.codec) == (True, True, "hevc"), r.note or r.error
    assert all(w.captions for w in r.windows)


@pytest.mark.parametrize("name", ["cover-first.mp4", "posters.mkv"])
def test_attached_pictures_are_passed_over(listed, name: str) -> None:
    r = read(listed[name])
    assert (r.ok, r.captions, r.codec, r.off_list) == (True, True, "h264", []), r.note or r.error


@pytest.mark.parametrize("name", ["fields-said-progressive.mkv", "fields-mpeg2.ts",
                                  "fields-hevc.mkv"])
def test_frames_in_fields_are_counted_whatever_the_header_says(listed, name: str) -> None:
    if name not in listed:
        pytest.skip("this libx265 codes no field pictures")
    r = read(listed[name])
    if r.codec in captions.SCANNED_CODECS:  # HEVC: never "none", its fields counted all the same
        assert (r.ok, r.captions) == (False, None), r.note or r.error
    else:
        assert r.ok and r.captions is False, r.note or r.error
    assert all(w.interlaced == w.frames > 0 for w in r.windows)


def test_an_arrival_read_while_it_is_copied_reads_every_window_whole(listed) -> None:
    # its windows land inside groups of pictures: each is read from the next
    # keyframe, never of a few frames
    r = read(listed["arriving.ts"])
    assert r.captions is not True, r.note or r.error
    assert all(w.frames >= captions.PACKETS / 2 for w in r.windows if not w.error)
    # a window whose decode said anything (a reference missing where it was
    # read from) fails it, never "none" (MUST 1); every other one reads whole
    assert all("ffprobe" in w.error for w in r.windows if w.error)
    assert r.ok is (not any(w.error for w in r.windows))


@pytest.mark.parametrize(("name", "shown"), [("av1.mkv", False), ("av1-cc.mkv", True)])
def test_av1_is_read(listed, name: str, shown: bool) -> None:
    # A53 in ITU-T T.35 metadata, which libdav1d gives as side data
    if name not in listed:
        pytest.skip("this ffmpeg encodes no AV1")
    decoders = subprocess.run([FFPROBE, "-hide_banner", "-decoders"], capture_output=True,
                              text=True, check=False).stdout  # fmt: skip
    if "libdav1d" not in decoders:
        pytest.skip("this ffprobe decodes AV1 without libdav1d (a hardware decoder only)")
    r = read(listed[name])
    assert (r.ok, r.captions, r.codec) == (True, shown, "av1"), r.note or r.error
    assert all(w.captions is shown and w.frames >= captions.PACKETS / 2 for w in r.windows)


@pytest.mark.parametrize(
    ("name", "off"),
    [
        ("off-container.flv", ["container:flv"]),
        ("off-codec.avi", ["codec:mpeg4"]),
        ("off-stream.ts", ["stream:data:bin_data"]),
        ("off-two-films.mkv", ["video-streams:2"]),
        ("off-vbi.mkv", ["picture:vbi-lines"]),
        ("off-still.mkv", ["video-streams:2", "codec:mjpeg"]),
    ],
)
def test_what_is_off_the_lists_is_not_known(listed, name: str, off: list[str]) -> None:
    r = read(listed[name])
    assert (r.ok, r.captions, r.unreadable, r.off_list) == (False, None, True, off)
    assert set(_captions_read(r)) == {"error", "ffprobe", "sizeBytes", "captionStreams", "offList"}


# ------------------------------------------- where long groups of pictures land


def _ebml_id(b: bytes, i: int) -> tuple[int, int]:
    n = 1
    while not b[i] & (0x80 >> (n - 1)):
        n += 1
    return int.from_bytes(b[i : i + n], "big"), i + n


def _ebml_size(b: bytes, i: int) -> tuple[int, int]:
    n = 1
    while not b[i] & (0x80 >> (n - 1)):
        n += 1
    v = b[i] & (0xFF >> n)
    for j in range(1, n):
        v = v << 8 | b[i + j]
    return v, i + n


def _sparse_cues(src: Path, out: Path, every_ms: int) -> int:
    """The Matroska file at src again at out, its index (Cues) holding only
    the keyframes every every_ms milliseconds: each other CuePoint overwritten
    by a Void element of its length, so that a seek lands on an indexed
    keyframe and passes those between over. Returns how many it voided."""
    b = bytearray(src.read_bytes())
    i = 0
    _id, i = _ebml_id(b, i)  # the EBML header
    size, i = _ebml_size(b, i)
    i += size
    _id, i = _ebml_id(b, i)  # the Segment
    _size, i = _ebml_size(b, i)
    voided = 0
    while i < len(b):
        eid, j = _ebml_id(b, i)
        size, j = _ebml_size(b, j)
        if eid == 0x1C53BB6B:  # Cues
            k = j
            while k < j + size:
                pid, m = _ebml_id(b, k)
                psize, m = _ebml_size(b, m)
                end = m + psize
                if pid == 0xBB:  # a CuePoint: its CueTime (0xB3) first
                    tid, n = _ebml_id(b, m)
                    tsize, n = _ebml_size(b, n)
                    when = int.from_bytes(b[n : n + tsize], "big") if tid == 0xB3 else 0
                    if when % every_ms:
                        whole = end - k
                        b[k : end] = bytes([0xEC, 0x80 | (whole - 2)]) + bytes(whole - 2)
                        voided += 1
                k = end
        i = j + size
    out.write_bytes(bytes(b))
    return voided


@pytest.fixture(scope="module")
def gops(tmp_path_factory) -> dict[str, Path]:
    """Matroska originals of long groups of pictures: of 15 s and 25 s over
    five minutes, of 40 s over six; and one of 10 s whose index holds every
    fourth keyframe alone."""
    d = tmp_path_factory.mktemp("gops")
    c: dict[str, Path] = {}
    for name, gop, length in (("gop15.mkv", 15, LONG), ("gop25.mkv", 25, LONG),
                              ("gop40.mkv", 40, 360), ("gop10.mkv", 10, LONG)):  # fmt: skip
        frames = gop * 25
        c[name] = d / name
        _ff("-f", "lavfi", "-i", f"testsrc2=size=160x120:rate=25:duration={length}", "-c:v",
            "libx264", "-preset", "ultrafast", "-x264-params",
            f"keyint={frames}:min-keyint={frames}:scenecut=0", str(c[name]))  # fmt: skip
    c["skipping.mkv"] = d / "skipping.mkv"
    assert _sparse_cues(c["gop10.mkv"], c["skipping.mkv"], 40_000) > 0
    return c


@pytest.mark.parametrize("name", ["gop15.mkv", "gop25.mkv"])
def test_long_groups_of_pictures_land_on_the_keyframe_before(gops, name: str) -> None:
    r = read(gops[name])
    assert (r.ok, r.captions) == (True, False), r.note or r.error
    assert all(w.keyframe_before and 0 <= w.aim - w.start <= captions.BACK_SLACK_S
               for w in r.windows)  # fmt: skip
    if name == "gop25.mkv":  # the 90 % window 20 s before its place
        assert round(r.windows[-1].aim - r.windows[-1].start) == 20


def test_a_group_of_pictures_longer_than_the_windows_reach_is_read_whole(gops) -> None:
    # groups of 40 s in six minutes: the 10 % window (36 s) could land on the
    # start's frames, and the short-picture rule reads every packet
    r = read(gops["gop40.mkv"])
    [w] = r.windows
    assert (r.ok, r.captions, w.whole, w.error) == (True, False, True, None), r.note or r.error


def test_a_seek_that_passes_a_keyframe_over_fails(gops) -> None:
    # an index of every fourth keyframe: the 10 % window lands at 0 s, 30 s
    # before its place, with the keyframes at 10 and 20 s between
    r = read(gops["skipping.mkv"])
    assert (r.ok, r.captions) == (False, None)
    assert "a keyframe at 10.000 s lies between" in r.error, r.error
    # the same file indexed whole reads
    assert read(gops["gop10.mkv"]).ok
