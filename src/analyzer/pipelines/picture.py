"""The picture area: the part of the coded frame a title's picture fills.

Many sources keep a full frame with a wider picture letterboxed inside it,
the black bars baked into the video (a 1920x1080 frame whose picture is
1920x804), or a narrower one pillarboxed (a 4:3 picture in a 16:9 frame).
The players fit the picture area to the screen rather than the frame. The
bars are measured once, here, and the catalog keeps the area; the package is
left as it is.

What the area is. The union of the picture over the whole title, every
scene: it is never smaller than the picture of a scene sampled, so a film
whose aspect changes (wider frames in places) keeps the largest, and a bar
counts only where it is black throughout. Cutting the picture is far worse
than missing a crop, so a side whose edge is in doubt keeps the full frame.

How it is measured. Frames spread over the whole title (one every
INTERVAL_S seconds, at least MIN_SAMPLES and at most MAX_SAMPLES) are
decoded one at a time, so a sample reads a little of the file, never all of
it:

  * of an original, ffmpeg seeks to the time and decodes the first frame it
    can show there (a keyframe), reading on only as far as the decoder needs
    to give it out;
  * of a package's video rendition (its HLS playlist), a sample is a
    segment's first frame, a keyframe: its init segment and the segment cut
    right after that frame (the fragment's header says where), so a sample
    reads one frame and decodes one.

ffmpeg reduces each frame to SEGMENTS segment means per row and per column
(gray, 8 bits), and the rules below run on those, per side:

  * black is the level of the frame's outermost lines (the median over the
    samples of their darkest segment): a segment holds picture when it is
    CONTENT levels above it. A row (a column) is picture when half of its
    segments do, within the picture's extent the other axis found; noise
    and grain in a bar average out in a segment of tens of pixels.
  * the picture's edge on a side is the outermost picture row of any
    sample: a dark or a black frame adds nothing, so dark scenes and fades
    never shrink the area, and a scene with a wider picture widens it.
  * the edge is kept only where enough samples (SUPPORT_MIN, and
    SUPPORT_SHARE of those with a picture) show the picture reaching it,
    within a few pixels; otherwise that side keeps the full frame.
  * anything in the bar beyond a guard band at the edge, in any sample, a
    black one too, makes the bar not black throughout: subtitles or
    captions burned into it, a dim picture (a dark scene of a wider frame
    looks like this), specks. That side keeps the full frame. A segment
    counts there from NOT_BLACK levels above black, and only on two
    adjacent lines, so noise in a bar, averaged over a segment, does not.
    What sits in the guard band (ringing at a bright edge) widens the area
    to take it in.
  * a logo is left out: a small mark (at most a quarter of the side's
    length) in the bar in at least half of the samples is static, and the
    bar is black around it.

The edges are rounded outwards to even pixels. A measurement with too few
samples that show a picture, or of a rotated video, can't tell (reason); one
that can't read the video failed (error).
"""

from __future__ import annotations

import bisect
import json
import math
import os
import re
import statistics
import struct
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# The measurement's version, kept with every area.
METHOD = "picture/1"

# Segment means per row and per column.
SEGMENTS = 32
# Levels (8 bits) above black at which a segment holds picture, and at which
# one in the bar beyond the guard band makes the bar not black.
CONTENT = 10
NOT_BLACK = 8
# A black above this is no black bar's: the frame has none.
BLACK_MAX = 40
# The share of segments of a row (a column), within the picture's extent,
# that hold picture for the row to be picture.
BROAD = 0.5
# How many samples, and which share of those with a picture, must show the
# picture reaching an edge (within SUPPORT_TOLERANCE of the side's length)
# for it to be kept.
SUPPORT_MIN = 3
SUPPORT_SHARE = 0.03
SUPPORT_TOLERANCE = 0.004
# The band beyond an edge, as a share of the side's length, where ringing
# at a bright edge widens the area instead of making the bar not black.
GUARD = 0.008
# A logo: in the bar in at least this share of the samples, and at most
# LOGO_SPAN of the side's length.
LOGO_SHARE = 0.5
LOGO_SPAN = 0.25
# Sampling: one frame every INTERVAL_S seconds (PICTURE_INTERVAL_S overrides
# it), MIN_SAMPLES to MAX_SAMPLES of them; and how many samples with a
# picture there must be at least to tell anything (MIN_INFORMATIVE, and a
# tenth of the samples).
INTERVAL_S = 10.0
MIN_SAMPLES = 24
MAX_SAMPLES = 720
MIN_INFORMATIVE = 6
# One sample's ffmpeg run, at most.
SAMPLE_TIMEOUT_S = 60
# Samples decoded at once (PICTURE_WORKERS overrides it).
WORKERS = 2
# How much of a segment is read to find its first frame (the fragment's
# header comes first).
FRAGMENT_HEAD = 256 * 1024
# Containers whose headers describe their streams, so a sample need not
# probe them at length.
_HEADED = (".mkv", ".mk3d", ".webm", ".mp4", ".m4v", ".mov")


@dataclass(frozen=True)
class Box:
    """A rectangle of the coded frame, in pixels: its left and top edge,
    its width and its height."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class Measurement:
    """What a measurement found of the video at path.

    frame is its coded size (a Box at 0, 0), area the picture's within it:
    the frame itself when there are no bars, None when the measurement can't
    tell (reason says why) or failed (error says why). sides says, per side
    (top, bottom, left, right), where its edge is and why. samples are the
    frames decoded, informative those that show a picture; seconds the time
    it took, cpu_seconds the CPU its ffmpeg runs and its rules used."""

    path: str
    frame: Box | None = None
    area: Box | None = None
    reason: str | None = None
    error: str | None = None
    samples: int = 0
    informative: int = 0
    sides: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    seconds: float = 0.0
    cpu_seconds: float = 0.0
    # The measurement was stopped (its deadline passed, the worker stops):
    # it says nothing, and the title is measured again later.
    interrupted: bool = False

    @property
    def full(self) -> bool:
        """The picture fills the frame: there are no bars."""
        return self.area is not None and self.area == self.frame

    def report(self) -> dict[str, Any]:
        """The catalog's body for it (PUT /api/picture/items/{id})."""
        frame = area = None
        if self.frame is not None:
            frame = {"width": self.frame.width, "height": self.frame.height}
        if self.area is not None:
            a = self.area
            area = {"x": a.x, "y": a.y, "width": a.width, "height": a.height}
        body: dict[str, Any] = {
            "path": self.path,
            "method": METHOD,
            "frame": frame,
            "area": area,
            "samples": self.samples,
            "informative": self.informative,
            "durationMs": round(self.duration_s * 1000),
            "seconds": round(self.seconds, 1),
            "cpuSeconds": round(self.cpu_seconds, 1),
            "sides": self.sides,
        }
        if self.reason:
            body["reason"] = self.reason[:500]
        if self.error:
            body["error"] = self.error[:500]
        return body


class Interrupted(Exception):
    """The measurement's deadline passed, or the worker stops."""


class _Unreadable(Exception):
    """The video can't be read: ffprobe or every sample failed."""


class _Undecidable(Exception):
    """The video is read, and can't be measured (rotated, too small)."""


# ------------------------------------------------------------------ measure
def measure(
    path: str,
    duration_ms: int | None = None,
    *,
    interval_s: float | None = None,
    deadline: float | None = None,
    stop: Callable[[], bool] | None = None,
    workers: int | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> Measurement:
    """Measure the picture area of the video at path: a file ffmpeg reads
    (an original), or a package's video rendition, its HLS media playlist
    (.m3u8). deadline (time.time()) and stop end it early: the result is
    interrupted and says nothing. Never raises for what the file holds: a
    video that can't be read says why in error."""
    t0 = time.monotonic()
    m = Measurement(path=path)
    cpu = _Cpu()
    try:
        _measure(m, cpu, path, duration_ms, interval_s, deadline, stop, workers, ffmpeg, ffprobe)
    except Interrupted:
        m.interrupted, m.area, m.reason = True, None, "stopped before it was done"
    except _Undecidable as e:
        m.area, m.reason = None, str(e)
    except _Unreadable as e:
        m.area, m.error = None, str(e)
    m.seconds = time.monotonic() - t0
    m.cpu_seconds = cpu.total()
    log.info(
        "picture.measured",
        path=path,
        frame=None if m.frame is None else f"{m.frame.width}x{m.frame.height}",
        area=None if m.area is None else f"{m.area.width}x{m.area.height}+{m.area.x}+{m.area.y}",
        full=m.full,
        reason=m.reason,
        error=m.error,
        samples=m.samples,
        informative=m.informative,
        seconds=round(m.seconds, 1),
        cpu_seconds=round(m.cpu_seconds, 1),
        interrupted=m.interrupted,
    )
    return m


class _Cpu:
    """The CPU seconds of the ffmpeg runs of one measurement, each read when
    it is reaped (os.wait4), and of the rules, which run in this thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._children = 0.0
        self._t0 = time.thread_time()

    def add(self, seconds: float) -> None:
        with self._lock:
            self._children += seconds

    def total(self) -> float:
        return self._children + (time.thread_time() - self._t0)


def _measure(
    m: Measurement,
    cpu: _Cpu,
    path: str,
    duration_ms: int | None,
    interval_s: float | None,
    deadline: float | None,
    stop: Callable[[], bool] | None,
    workers: int | None,
    ffmpeg: str,
    ffprobe: str,
) -> None:
    def check() -> None:
        if (deadline is not None and time.time() >= deadline) or (stop is not None and stop()):
            raise Interrupted()

    playlist = path.lower().endswith(".m3u8")
    if playlist:
        init, segments = read_playlist(path)
        if init is None or not segments:
            raise _Unreadable("the playlist names no fMP4 init segment and segments in its folder")
        probe_input = _concat(init, segments[0][1]) or segments[0][1]
        duration_s = segments[-1][0] + segments[-1][2]
    else:
        init, segments = None, []
        probe_input = path
        duration_s = (duration_ms or 0) / 1000.0
    check()
    info = _probe(probe_input, ffprobe, cpu)
    width, height = info["width"], info["height"]
    m.frame = Box(0, 0, width, height)
    if info["rotation"]:
        raise _Undecidable(f"the video is rotated by {info['rotation']} degrees")
    if width < 64 or height < 64:
        raise _Undecidable(f"the frame is too small ({width}x{height})")
    if not playlist and duration_s <= 0:
        duration_s = info["duration"]
    if duration_s <= 0:
        raise _Unreadable("the video's duration is not known")
    m.duration_s = duration_s

    times = sample_times(duration_s, interval_s if interval_s is not None else _interval())
    if playlist:
        jobs = [_Job(t, init=init, segment=seg) for t, seg in pick_segments(segments, times)]
    else:
        quick = path.lower().endswith(_HEADED)
        jobs = [_Job(t, args=_seek_args(path, t, quick)) for t in times]

    n = workers if workers is not None else _workers()
    frames: list[tuple[float, bytes]] = []
    seen: set[int] = set()
    failed = 0
    expected = SEGMENTS * (width + height)

    def one(job: _Job) -> tuple[float, bytes | None]:
        check()
        return job.t, job.run(ffmpeg, cpu)

    with ThreadPoolExecutor(max_workers=max(1, n)) as pool:
        for t, data in pool.map(one, jobs):
            if data is None or len(data) != expected:
                failed += 1
                continue
            key = hash(data)
            if key in seen:  # the same frame, sampled twice
                continue
            seen.add(key)
            frames.append((t, data))
    check()
    m.samples = len(frames)
    if not frames:
        raise _Unreadable(f"none of {len(jobs)} samples could be decoded")
    if failed:
        log.info("picture.samples_failed", path=path, failed=failed, of=len(jobs))
    d = decide([f for _t, f in frames], width, height, times=[t for t, _f in frames])
    m.area, m.reason, m.sides, m.informative = d.area, d.reason, d.sides, d.informative


def _interval() -> float:
    """The sampling interval: INTERVAL_S, or PICTURE_INTERVAL_S (1 to 120 s)."""
    try:
        v = float(os.environ.get("PICTURE_INTERVAL_S") or INTERVAL_S)
    except ValueError:
        return INTERVAL_S
    return min(max(v, 1.0), 120.0)


def _workers() -> int:
    try:
        n = int(os.environ.get("PICTURE_WORKERS") or WORKERS)
    except ValueError:
        return WORKERS
    return min(max(n, 1), 8)


def sample_times(duration_s: float, interval_s: float = INTERVAL_S) -> list[float]:
    """The times sampled of a title duration_s long: evenly spread over it,
    each in the middle of its share of the runtime, one every interval_s
    seconds (MIN_SAMPLES to MAX_SAMPLES of them)."""
    if duration_s <= 0:
        return []
    n = round(duration_s / max(interval_s, 0.1))
    n = min(max(n, MIN_SAMPLES), MAX_SAMPLES)
    step = duration_s / n
    return [round((i + 0.5) * step, 3) for i in range(n)]


def _seek_args(path: str, t: float, quick: bool) -> list[str]:
    """ffmpeg's input of the frame at t of an original: the first it can show
    from the keyframe before t. A container whose header describes its
    streams is not probed at length (a probe decodes frames of its start)."""
    probe = ["-probesize", "65536", "-analyzeduration", "0"] if quick else []
    return [*probe, "-noaccurate_seek", "-ss", f"{t:.3f}", "-i", path]


# ----------------------------------------------------------------- playlists
_EXTINF = re.compile(r"^#EXTINF:\s*([0-9.]+)")
_MAP = re.compile(r'^#EXT-X-MAP:.*URI="([^"]+)"')


def read_playlist(path: str) -> tuple[str | None, list[tuple[float, str, float]]]:
    """The init segment of an HLS media playlist and its segments, each with
    the time it starts at and its duration, as paths in the playlist's
    folder. A playlist of byte ranges (one file), or of a URI that is a URL
    or leaves the folder, gives none."""
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [ln.strip() for ln in f]
    init: str | None = None
    out: list[tuple[float, str, float]] = []
    at, pending = 0.0, None
    for ln in lines:
        if not ln:
            continue
        if ln.startswith("#EXT-X-BYTERANGE") or (ln.startswith("#EXT-X-MAP") and "BYTERANGE" in ln):
            return None, []
        mm = _MAP.match(ln)
        if mm:
            init = _beside(base, mm.group(1))
            if init is None:
                return None, []
            continue
        mi = _EXTINF.match(ln)
        if mi:
            pending = float(mi.group(1))
            continue
        if ln.startswith("#"):
            continue
        seg = _beside(base, ln)
        if seg is None or pending is None:
            return None, []
        out.append((at, seg, pending))
        at += pending
        pending = None
    return init, out


def _beside(base: str, uri: str) -> str | None:
    """uri of a playlist in base as a path in that folder; None when it is a
    URL or leaves the folder."""
    if "://" in uri or uri.startswith("/") or ".." in uri.split("/"):
        return None
    return os.path.join(base, uri)


def pick_segments(
    segments: list[tuple[float, str, float]], times: list[float]
) -> list[tuple[float, str]]:
    """The segments the times fall in, each once, with the time it starts."""
    starts = [s for s, _p, _d in segments]
    out: list[tuple[float, str]] = []
    used: set[int] = set()
    for t in times:
        i = max(0, bisect.bisect_right(starts, t) - 1)
        if i not in used:
            used.add(i)
            out.append((segments[i][0], segments[i][1]))
    return out


def _concat(init: str | None, segment: str) -> str | None:
    """ffmpeg's input of one fMP4 segment with its init segment: the concat
    protocol, which a "|" in a path would break (None then)."""
    if init is None or "|" in init or "|" in segment:
        return None
    return f"concat:{init}|{segment}"


def first_frame_end(head: bytes) -> int | None:
    """Where the first sample of an fMP4 segment ends, as an offset in it:
    from its movie fragment (moof), the first track fragment's base, the
    run's data offset and the first sample's size. None when head holds no
    such fragment."""
    off = 0
    while off + 8 <= len(head):
        size, kind = struct.unpack(">I4s", head[off : off + 8])
        if size == 1 and off + 16 <= len(head):
            size = struct.unpack(">Q", head[off + 8 : off + 16])[0]
        if size < 8:
            return None
        if kind == b"moof":
            if off + size > len(head):
                return None
            return _fragment_first(head, off, off + size)
        off += size
    return None


def _fragment_first(b: bytes, moof: int, end: int) -> int | None:
    off = moof + 8
    while off + 8 <= end:
        size, kind = struct.unpack(">I4s", b[off : off + 8])
        if size < 8:
            return None
        if kind == b"traf":
            return _track_first(b, off + 8, off + size, moof)
        off += size
    return None


def _track_first(b: bytes, off: int, end: int, moof: int) -> int | None:
    base, default_size = moof, None
    while off + 8 <= end:
        size, kind = struct.unpack(">I4s", b[off : off + 8])
        if size < 8:
            return None
        body = off + 8
        flags = int.from_bytes(b[body + 1 : body + 4], "big")
        if kind == b"tfhd":
            p = body + 8  # version and flags, track_ID
            if flags & 0x1:
                base = struct.unpack(">Q", b[p : p + 8])[0]
                p += 8
            if flags & 0x2:
                p += 4
            if flags & 0x8:
                p += 4
            if flags & 0x10:
                default_size = struct.unpack(">I", b[p : p + 4])[0]
        elif kind == b"trun":
            count = struct.unpack(">I", b[body + 4 : body + 8])[0]
            p, data = body + 8, 0
            if flags & 0x1:
                data = struct.unpack(">i", b[p : p + 4])[0]
                p += 4
            if flags & 0x4:
                p += 4
            if flags & 0x100:
                p += 4
            first = default_size
            if flags & 0x200:
                first = struct.unpack(">I", b[p : p + 4])[0]
            if count == 0 or first is None or not flags & 0x1:
                return None
            return base + data + first
        off += size
    return None


# -------------------------------------------------------------------- ffmpeg
# One frame reduced to SEGMENTS segment means per row (the rows, top to
# bottom), then per column (left to right; transpose=cclock_flip is a plain
# transpose): a SEGMENTS x (height + width) gray image.
_REDUCE = (
    f"format=gray,split[a][b];[a]scale={SEGMENTS}:ih:flags=area[r];"
    f"[b]transpose=cclock_flip,scale={SEGMENTS}:ih:flags=area[c];[r][c]vstack"
)


@dataclass
class _Job:
    """One sample: ffmpeg's input arguments, or a package segment's first
    frame (init and segment), which is fed on stdin."""

    t: float
    args: list[str] | None = None
    init: str | None = None
    segment: str | None = None

    def run(self, ffmpeg: str, cpu: _Cpu) -> bytes | None:
        if self.args is not None:
            return _reduce(ffmpeg, self.args, None, cpu)
        data = _segment_head(self.init, self.segment)
        if data is not None:
            got = _reduce(
                ffmpeg,
                ["-probesize", "65536", "-analyzeduration", "0", "-f", "mp4", "-i", "pipe:0"],
                data,
                cpu,
            )
            if got is not None:
                return got
        src = _concat(self.init, self.segment) if self.segment else None
        if src is None:
            return None
        return _reduce(ffmpeg, ["-i", src], None, cpu)


def _segment_head(init: str | None, segment: str | None) -> bytes | None:
    """A package segment's init segment and the segment up to the end of its
    first frame; None when the fragment can't be read."""
    if init is None or segment is None:
        return None
    try:
        with open(segment, "rb") as f:
            head = f.read(FRAGMENT_HEAD)
            end = first_frame_end(head)
            if end is None:
                return None
            if end > len(head):
                head += f.read(end - len(head))
            if end > len(head):
                return None
        with open(init, "rb") as f:
            return f.read() + head[:end]
    except OSError:
        return None


def _reduce(ffmpeg: str, input_args: list[str], stdin: bytes | None, cpu: _Cpu) -> bytes | None:
    """The first frame of the input, reduced (_REDUCE); None when ffmpeg
    fails or runs past SAMPLE_TIMEOUT_S."""
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        *(["-nostdin"] if stdin is None else []),
        "-threads",
        "1",
        *input_args,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        "1",
        *_passthrough(ffmpeg),
        "-vf",
        _REDUCE,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    out, ok = _run(cmd, stdin, cpu)
    return out if ok and out else None


_PASSTHROUGH: dict[str, list[str]] = {}


def _passthrough(ffmpeg: str) -> list[str]:
    """The output option that hands frames on with their own timestamps.
    The default for raw video, a constant rate, drops a fast seek's frames
    before the time sought, which have negative timestamps, and so decodes
    the whole group of pictures up to it. -fps_mode from ffmpeg 5.1 on,
    -vsync before (which 5.1 and later still take, deprecated)."""
    if ffmpeg not in _PASSTHROUGH:
        opt = "-vsync"
        try:
            r = subprocess.run(
                [ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True, timeout=10
            )
            m = re.search(r"version n?(\d+)\.(\d+)", r.stdout)
            if m and (int(m.group(1)), int(m.group(2))) >= (5, 1):
                opt = "-fps_mode"
        except (OSError, subprocess.TimeoutExpired):
            pass
        _PASSTHROUGH[ffmpeg] = [opt, "passthrough"]
    return _PASSTHROUGH[ffmpeg]


def _run(cmd: list[str], stdin: bytes | None, cpu: _Cpu) -> tuple[bytes, bool]:
    """Run cmd, feeding it stdin, and return its stdout and whether it ended
    well; its CPU is added to cpu when it is reaped."""
    try:
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return b"", False
    timer = threading.Timer(SAMPLE_TIMEOUT_S, p.kill)
    timer.start()
    out = b""
    try:
        if stdin is not None:
            feeder = threading.Thread(target=_feed, args=(p, stdin), daemon=True)
            feeder.start()
        err_reader = threading.Thread(target=p.stderr.read, daemon=True)
        err_reader.start()
        out = p.stdout.read()
        _pid, status, usage = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        cpu.add(usage.ru_utime + usage.ru_stime)
    except OSError:
        p.kill()
        p.wait()
        return b"", False
    finally:
        timer.cancel()
        for s in (p.stdout, p.stderr, p.stdin):
            if s is not None:
                s.close()
    return out, p.returncode == 0


def _feed(p: subprocess.Popen, data: bytes) -> None:
    try:
        p.stdin.write(data)
        p.stdin.close()
    except OSError:
        pass  # it stopped reading: it has what it needs, or failed


def _probe(path: str, ffprobe: str, cpu: _Cpu) -> dict[str, Any]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        path,
    ]
    out, ok = _run(cmd, None, cpu)
    try:
        body = json.loads(out or b"{}")
    except json.JSONDecodeError as e:
        raise _Unreadable("ffprobe gave no JSON") from e
    streams = body.get("streams") or []
    if not streams:
        raise _Unreadable(
            "ffprobe found no video stream" if ok else "ffprobe could not read the file"
        )
    s = streams[0]
    width, height = int(s.get("width") or 0), int(s.get("height") or 0)
    if width <= 0 or height <= 0:
        raise _Unreadable("the video stream has no size")
    rotation = 0
    for sd in s.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rotation = int(float(sd["rotation"])) % 360
            except (TypeError, ValueError):
                pass
    try:
        rotation = rotation or int(float((s.get("tags") or {}).get("rotate") or 0)) % 360
    except (TypeError, ValueError):
        pass
    duration = 0.0
    for v in ((body.get("format") or {}).get("duration"), s.get("duration")):
        try:
            duration = float(v)
            break
        except (TypeError, ValueError):
            continue
    return {"width": width, "height": height, "rotation": rotation, "duration": duration}


# ------------------------------------------------------------------- decide
@dataclass
class Decision:
    """The area decide found, or None and why; each side's edge and why;
    and how many frames showed a picture."""

    area: Box | None
    reason: str | None
    sides: dict[str, Any]
    informative: int


def decide(
    frames: list[bytes],
    width: int,
    height: int,
    *,
    times: list[float] | None = None,
    segments: int = SEGMENTS,
) -> Decision:
    """The picture area of frames reduced as _REDUCE reduces them (segments
    means per row, then per column), by the rules at the top of the file."""
    k = segments
    v = _View(frames, width, height, k, times)
    black = v.black()
    if black > BLACK_MAX:
        # The frame's outermost lines are never black: there are no bars.
        return Decision(
            Box(0, 0, width, height),
            None,
            {"black": black, "note": "the frame's edges are never black"},
            len(frames),
        )
    sides: dict[str, Any] = {}
    top, bottom, left, right = 0, height, 0, width
    informative = 0
    # Each axis within the extent the other found: a picture letterboxed and
    # pillarboxed at once (windowboxed) fills half its rows' segments only
    # within its columns. Twice is enough for them to agree.
    for _round in range(2):
        rows = v.lines("row", black, _span(left, right, width, k))
        top_s, bottom_s = rows.side("top"), rows.side("bottom")
        cols = v.lines("col", black, _span(top_s.edge, bottom_s.edge, height, k))
        left_s, right_s = cols.side("left"), cols.side("right")
        top, bottom, left, right = top_s.edge, bottom_s.edge, left_s.edge, right_s.edge
        informative = min(rows.informative, cols.informative)
        sides = {s.name: s.why() for s in (top_s, bottom_s, left_s, right_s)}
    sides["black"] = black
    need = max(MIN_INFORMATIVE, math.ceil(0.1 * len(frames)))
    if informative < need:
        why = f"too few frames show a picture ({informative} of {len(frames)}, {need} needed)"
        return Decision(None, why, sides, informative)
    if bottom - top < height // 8 or right - left < width // 8:
        why = f"the picture found is too small ({right - left}x{bottom - top})"
        return Decision(None, why, sides, informative)
    x0, y0 = _even_down(left), _even_down(top)
    x1, y1 = min(width, _even_up(right)), min(height, _even_up(bottom))
    return Decision(Box(x0, y0, x1 - x0, y1 - y0), None, sides, informative)


def _even_down(n: int) -> int:
    return n - n % 2


def _even_up(n: int) -> int:
    return n + n % 2


def _span(lo: int, hi: int, length: int, k: int) -> tuple[int, int]:
    """The segments that lie within [lo, hi) of a line length long, cut in k."""
    a = math.ceil(lo * k / length)
    b = math.floor(hi * k / length)
    if b - a < 1:
        return 0, k
    return a, b


class _View:
    """The frames' reduced lines, read per axis."""

    def __init__(
        self, frames: list[bytes], width: int, height: int, k: int, times: list[float] | None
    ) -> None:
        self.frames = frames
        self.width, self.height, self.k = width, height, k
        self.times = times or [0.0] * len(frames)

    def black(self) -> int:
        """The black level: the median over the frames of the darkest
        segment of their outermost lines (rows 0 and height-1, columns 0
        and width-1)."""
        k, h, w = self.k, self.height, self.width
        darkest = [
            min(
                min(f[0:k]),
                min(f[(h - 1) * k : h * k]),
                min(f[h * k : (h + 1) * k]),
                min(f[(h + w - 1) * k : (h + w) * k]),
            )
            for f in self.frames
        ]
        return int(statistics.median(darkest))

    def lines(self, axis: str, black: int, span: tuple[int, int]) -> _Lines:
        if axis == "row":
            return _Lines(self, 0, self.height, ("top", "bottom"), black, span)
        return _Lines(self, self.height, self.width, ("left", "right"), black, span)


def _table(level: int) -> bytes:
    """A translation table: 1 for a byte above level, 0 for the rest."""
    return bytes(1 if b > level else 0 for b in range(256))


@dataclass
class _Side:
    name: str
    length: int
    edge: int  # the picture's first line from this side, as a line of the axis
    support: int = 0
    note: str = ""
    seen: list[dict[str, Any]] = field(default_factory=list)

    def why(self) -> dict[str, Any]:
        cut = self.edge if self.name in ("top", "left") else self.length - self.edge
        out: dict[str, Any] = {"bar": cut, "support": self.support, "note": self.note}
        if self.seen:
            out["seen"] = self.seen[:5]
        return out


class _Lines:
    """The lines of one axis (rows, or columns) of every frame."""

    def __init__(
        self,
        v: _View,
        offset: int,
        count: int,
        names: tuple[str, str],
        black: int,
        span: tuple[int, int],
    ) -> None:
        self.v, self.count, self.names, self.span = v, count, names, span
        k = v.k
        a, b = span
        self.need = max(1, math.ceil((b - a) * BROAD))
        content, not_black = _table(black + CONTENT), _table(black + NOT_BLACK)
        # Per frame, the axis' segments that hold picture, and those that are
        # not black (1, else 0).
        axis = [f[offset * k : (offset + count) * k] for f in v.frames]
        self.content = [x.translate(content) for x in axis]
        self.lit = [x.translate(not_black) for x in axis]
        self.first = [self._first(c, False) for c in self.content]
        self.last = [self._first(c, True) for c in self.content]
        self.informative = sum(1 for x in self.first if x is not None)

    def _broad(self, c: bytes, line: int) -> bool:
        k = self.v.k
        a, b = self.span
        return sum(c[line * k + a : line * k + b]) >= self.need

    def _first(self, c: bytes, reverse: bool) -> int | None:
        """A frame's first picture line from the start (or the end), as its
        distance from that side; None when it has none (a black frame)."""
        lines: Iterable[int] = range(self.count - 1, -1, -1) if reverse else range(self.count)
        for d, line in enumerate(lines):
            if self._broad(c, line):
                return d
        return None

    def side(self, name: str) -> _Side:
        reverse = name == self.names[1]
        dists = [d for d in (self.last if reverse else self.first) if d is not None]
        length = self.count
        s = _Side(name, length, length if reverse else 0)
        if not dists:
            s.note = "no frame shows a picture"
            return s
        union = min(dists)
        tol = max(2, round(length * SUPPORT_TOLERANCE))
        s.support = sum(1 for d in dists if d <= union + tol)
        if union == 0:
            s.note = "the picture reaches the frame's edge"
            return s
        need = max(SUPPORT_MIN, math.ceil(SUPPORT_SHARE * len(dists)))
        if s.support < need:
            s.note = (
                f"{s.support} frames show the picture reaching its edge, {need} needed: kept whole"
            )
            return s
        guard = max(2, round(length * GUARD))
        zone = union - guard  # lines 0..zone-1 from the side: the bar beyond the guard band
        if zone > 0:
            found = self._beyond(reverse, zone)
            if found:
                s.seen = found
                s.note = "the bar is not black throughout: kept whole"
                return s
        # The guard band: what is not black there (ringing at a bright edge)
        # widens the area to take it in.
        d = self._outermost(reverse, max(0, zone), union)
        s.note = "bar" if d == union else f"bar, widened by {union - d} lines at its edge"
        s.edge = length - d if reverse else d
        return s

    def _line(self, reverse: bool, d: int) -> int:
        return self.count - 1 - d if reverse else d

    def _beyond(self, reverse: bool, zone: int) -> list[dict[str, Any]]:
        """What is not black in the bar's first zone lines from the side, in
        any frame (a black one too: a subtitle over a fade is one): segments
        on two adjacent lines, a logo left out. Each finding with the frame's
        time and its lines (as distances from the side)."""
        k = self.v.k
        hits: list[tuple[int, int, bytes]] = []  # frame, distance, the line's lit segments
        for fi, lit in enumerate(self.lit):
            for d in range(zone):
                line = self._line(reverse, d)
                seg = lit[line * k : (line + 1) * k]
                if 1 in seg:
                    hits.append((fi, d, seg))
        if not hits:
            return []
        logo = self._logo(hits, len(self.lit))
        by_frame: dict[int, dict[int, bytes]] = {}
        for fi, d, seg in hits:
            by_frame.setdefault(fi, {})[d] = seg
        found: list[dict[str, Any]] = []
        for fi, lines in sorted(by_frame.items()):
            marked = []
            for d, seg in lines.items():
                nxt = lines.get(d + 1)
                if nxt is not None and any(
                    seg[j] and nxt[j] and (d, j) not in logo and (d + 1, j) not in logo
                    for j in range(k)
                ):
                    marked.append(d)
            if marked:
                found.append(
                    {"at": round(self.v.times[fi], 1), "lines": [min(marked), max(marked) + 2]}
                )
        if logo and not found:
            log.info("picture.logo_left_out", segments=len(logo))
        return found

    def _logo(self, hits: list[tuple[int, int, bytes]], frames: int) -> set[tuple[int, int]]:
        """The segments of the bar that are lit in at least LOGO_SHARE of the
        frames, when together they span at most LOGO_SPAN of the line."""
        k = self.v.k
        counts: dict[tuple[int, int], int] = {}
        for _fi, d, seg in hits:
            for j, bit in enumerate(seg):
                if bit:
                    counts[(d, j)] = counts.get((d, j), 0) + 1
        logo = {cell for cell, c in counts.items() if c >= LOGO_SHARE * frames}
        if logo:
            cols = sorted({j for _d, j in logo})
            if cols[-1] - cols[0] + 1 > LOGO_SPAN * k:
                return set()
        return logo

    def _outermost(self, reverse: bool, lo: int, union: int) -> int:
        """The outermost line, between lo and union (distances from the side),
        that is not black in a segment of any frame."""
        k = self.v.k
        for d in range(lo, union):
            line = self._line(reverse, d)
            for lit in self.lit:
                if 1 in lit[line * k : (line + 1) * k]:
                    return d
        return union
