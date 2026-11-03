"""Where a measurement's samples come from: the frames of a title, decoded
from a file opened once.

What is sampled.

  * keyframes: a shot of TARGET_GAP_S seconds or more holds a sample
    wherever the video's keyframes allow it. They are picked so that no two
    picked are further apart than that (from each, the latest within it,
    else the next), and an encoder puts one at most cuts. Each is decoded
    alone, its decoder flushed after it.
  * stretches: what changes within a shot (a subtitle burned into a bar, a
    caption) has no keyframe of its own, so STRETCHES stretches spread over
    the title, STRETCH_SHARE of it together (STRETCH_MIN_S to STRETCH_MAX_S
    each), are decoded through, their reference frames, a sample every
    STRETCH_STEP_S.
  * a title of at most SHORT_S seconds is decoded through, a sample every
    SEQUENTIAL_STEP_S, when its file is at most SEQUENTIAL_MAX_BYTES.

Where from.

  * an original whose container has an index of its keyframes (Matroska's
    cues, MP4's sample table, AVI's index; none further apart than
    INDEX_GAP_MAX_S): its keyframes, a seek to each;
  * an original without one (MPEG-TS, MPEG-PS): a grid of the same step, a
    seek to each time and the first keyframe from there;
  * a package's video rendition (its HLS playlist of fMP4 segments): its
    sync samples, each read by its byte range (fmp4.py), and decoded with
    the init segment's decoder configuration.

What it costs. The keyframes are read coarse to fine, every 16th pick over
the whole title first, then every 8th, then the stretches, then every 4th
and the rest, each pass in the order of the file, until the read budget is
spent: READ_MB_PER_HOUR of the title (PICTURE_READ_MB_PER_HOUR overrides
it), at least READ_MIN_MB, besides what opening the file takes. A title
whose keyframes are more than its budget keeps an even coverage, a coarser
one: the samples' largest gap is reported, and the budget's stop.
"""

from __future__ import annotations

import bisect
import io
import itertools
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import av

from . import fmp4

TARGET_GAP_S = 2.9
SHORT_S = 300.0
SEQUENTIAL_STEP_S = 1.0
SEQUENTIAL_MAX_BYTES = 512 << 20
STRETCHES = 12
STRETCH_SHARE = 0.02
STRETCH_MIN_S = 4.0
STRETCH_MAX_S = 20.0
STRETCH_STEP_S = 0.5
READ_MB_PER_HOUR = 1024
READ_MIN_MB = 96
# The most keyframes a title takes: a longer one's are further apart.
MAX_SAMPLES = 6000
# A container's index lists the keyframes when none of its entries are
# further apart than this (an MPEG-TS's is a few points its probe found).
INDEX_GAP_MAX_S = 30.0
# A grid sample reads at most this much to find a keyframe, a stretch this
# much to decode through.
GRID_READ_MAX = 48 << 20
STRETCH_READ_MAX = 256 << 20
# The keyframe passes before the stretches, and after them: every n-th pick.
COARSE, FINE = (16, 8), (4, 2, 1)

Sink = Callable[[float, Any, bool], None]


class Unreadable(Exception):
    """The video can't be read."""


class Undecidable(Exception):
    """The video is read, and can't be measured (rotated, none)."""


@dataclass
class Sampling:
    """What a source did: how it sampled (keyframes, grid, sequential, of an
    original or a package), the title's duration as its file says, the
    keyframes it meant to take, the samples it tried, decoded and failed,
    the bytes it read to open the file and in all, the budget and whether
    it stopped there, and the samples' times."""

    mode: str = ""
    duration_s: float = 0.0
    planned: int = 0
    attempted: int = 0
    decoded: int = 0
    failed: int = 0
    duplicates: int = 0
    stretches: int = 0
    bytes_open: int = 0
    bytes_read: int = 0
    budget: int = 0
    stopped: str | None = None
    times: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def fail(self, why: str) -> None:
        self.failed += 1
        if len(self.errors) < 5:
            self.errors.append(why[:200])

    def max_gap(self) -> float | None:
        """The longest stretch of the title without a sample, from its start
        to its end."""
        if not self.times or self.duration_s <= 0:
            return None
        return _max_gap(self.times, self.duration_s)


@dataclass
class Video:
    """What a source says of its video: its decoder, its coded frame, its
    pixel aspect, its rotation, its duration, its container."""

    codec: str
    width: int
    height: int
    sar: tuple[int, int] = (1, 1)
    rotation: int = 0
    duration_s: float = 0.0
    container: str = ""


def budget_bytes(duration_s: float) -> int:
    """The read budget of a title of duration_s seconds."""
    try:
        per_hour = float(os.environ.get("PICTURE_READ_MB_PER_HOUR") or READ_MB_PER_HOUR)
    except ValueError:
        per_hour = READ_MB_PER_HOUR
    per_hour = min(max(per_hour, 16.0), 65536.0)
    return int(max(READ_MIN_MB, per_hour * duration_s / 3600.0) * (1 << 20))


def cover(times: list[float], gap: float) -> list[int]:
    """The indexes of times (ascending) to sample so that no two picked are
    more than gap apart where times allow it: from each pick, the latest
    within gap, else the next."""
    picks: list[int] = []
    i, n = 0, len(times)
    while i < n:
        j = i
        if picks:
            j = max(i, bisect.bisect_right(times, times[picks[-1]] + gap, lo=i) - 1)
        picks.append(j)
        i = j + 1
    return picks


def passes(picks: list[int], strides: tuple[int, ...], done: set[int]) -> Iterator[list[int]]:
    """The picks in passes: every strides[0]-th first, then every
    strides[1]-th of those left, and so on, each in the order of the
    title; done holds the picks given out."""
    for stride in strides:
        p = [picks[k] for k in range(0, len(picks), stride) if picks[k] not in done]
        done.update(p)
        if p:
            yield p


def gap_for(duration_s: float) -> float:
    """The largest gap the keyframes picked of a title this long aim at."""
    return max(TARGET_GAP_S, duration_s / MAX_SAMPLES)


def stretches(duration_s: float) -> list[tuple[float, float]]:
    """The stretches of a title decoded through: their starts and ends."""
    if duration_s <= 0:
        return []
    length = min(STRETCH_MAX_S, max(STRETCH_MIN_S, STRETCH_SHARE * duration_s / STRETCHES))
    out = []
    for k in range(STRETCHES):
        a = max(0.0, (k + 0.5) * duration_s / STRETCHES - length / 2)
        out.append((a, min(duration_s, a + length)))
    return out


def _max_gap(times: list[float], duration_s: float) -> float:
    if not times:
        return duration_s
    t = sorted(times)
    gaps = [t[0]] + [b - a for a, b in itertools.pairwise(t)]
    if duration_s > 0:
        gaps.append(duration_s - t[-1])
    return max(0.0, max(gaps))


class _Counting(io.RawIOBase):
    """A file that counts the bytes read from it."""

    def __init__(self, path: str) -> None:
        self._f = open(path, "rb", buffering=0)
        self.bytes = 0

    def readinto(self, b: Any) -> int:
        n = self._f.readinto(b) or 0
        self.bytes += n
        return n

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._f.seek(offset, whence)

    def tell(self) -> int:
        return self._f.tell()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def size(self) -> int:
        return os.fstat(self._f.fileno()).st_size

    def close(self) -> None:
        self._f.close()
        super().close()


def _threads() -> int:
    try:
        return min(max(int(os.environ.get("PICTURE_THREADS") or 1), 1), 8)
    except ValueError:
        return 1


class _Source:
    """What an original and a package share: what is sampled, in which
    order, within the budget."""

    sampling: Sampling
    video: Video
    _catalog_s: float

    def _spent(self) -> int:
        """The bytes read so far, besides opening the file."""
        raise NotImplementedError

    def _key(self, i: int) -> tuple[float, Any] | None:
        raise NotImplementedError

    def _stretch(self, a: float, b: float, sink: Sink) -> None:
        raise NotImplementedError

    def _over(self) -> bool:
        sm = self.sampling
        if self._spent() >= sm.budget:
            sm.stopped = "budget"
            return True
        return False

    def _keyframes_and_stretches(
        self, times: list[float], sink: Sink, seen: set[Any] | None
    ) -> None:
        """Sample the keyframes at times (ascending) and the stretches, in the
        order of the top of the file."""
        sm = self.sampling
        dur = max(sm.duration_s, self._catalog_s)
        picks = cover(times, gap_for(dur))
        sm.planned = len(picks)
        done: set[int] = set()

        def keys(strides: tuple[int, ...]) -> bool:
            for p in passes(picks, strides, done):
                for i in p:
                    if self._over():
                        return False
                    sm.attempted += 1
                    got = self._key(i)
                    if got is None:
                        continue
                    t, frame = got
                    if seen is not None:
                        if frame.pts is not None and frame.pts in seen:
                            sm.duplicates += 1
                            continue
                        seen.add(frame.pts)
                    self._take(frame, t, sink)
            return True

        if not keys(COARSE):
            return
        for a, b in stretches(sm.duration_s or dur):
            if self._over():
                return
            sm.stretches += 1
            self._stretch(a, b, sink)
        keys(FINE)

    def _take(self, frame: Any, t: float, sink: Sink, counted: bool = True) -> None:
        sm = self.sampling
        if frame.width != self.video.width or frame.height != self.video.height:
            if sm.decoded == 0:
                # what the frames are is what the players show
                self.video.width, self.video.height = frame.width, frame.height
            else:
                sm.fail(
                    f"a frame of {frame.width}x{frame.height} in a video of "
                    f"{self.video.width}x{self.video.height}"
                )
                return
        if frame.rotation:
            raise Undecidable(f"the video is rotated by {frame.rotation % 360} degrees")
        sm.decoded += 1
        sm.times.append(t)
        sink(t, frame, counted)


# ------------------------------------------------------------- originals
class Original(_Source):
    """An original: a file libavformat reads, opened once."""

    def __init__(self, path: str, duration_s: float = 0.0) -> None:
        self.path = path
        self.sampling = Sampling()
        self._io = _Counting(path)
        try:
            self._c = av.open(
                self._io, mode="r", options={"probesize": "1000000", "analyzeduration": "1000000"}
            )
        except (av.error.FFmpegError, OSError, ValueError) as e:
            self._io.close()
            raise Unreadable(f"the file can't be opened: {e}") from e
        videos = [
            s
            for s in self._c.streams.video
            if not (s.disposition & av.stream.Disposition.attached_pic)
        ]
        if not videos:
            self.close()
            raise Undecidable("the file has no video stream (a cover picture is none)")
        self._v = v = videos[0]
        for s in self._c.streams:
            if s is not v:
                s.discard = av.stream.Discard.all
        cc = v.codec_context
        cc.thread_count = _threads()
        if not cc.width or not cc.height:
            self.close()
            raise Unreadable("the video stream has no size")
        sar = v.sample_aspect_ratio or cc.sample_aspect_ratio
        file_s = 0.0
        if self._c.duration:
            file_s = self._c.duration / av.time_base
        elif v.duration and v.time_base:
            file_s = float(v.duration * v.time_base)
        self.video = Video(
            codec=cc.name,
            width=cc.width,
            height=cc.height,
            sar=(sar.numerator, sar.denominator) if sar else (1, 1),
            duration_s=file_s,
            container=self._c.format.name,
        )
        self.sampling.duration_s = file_s or duration_s
        self._catalog_s = duration_s
        self.sampling.bytes_open = self._io.bytes
        self._start = v.start_time or 0
        self._keys: list[tuple[float, int]] = []

    def close(self) -> None:
        try:
            self._c.close()
        except Exception:
            pass
        self._io.close()

    def _spent(self) -> int:
        return self._io.bytes - self.sampling.bytes_open

    def _ts(self, t: float) -> int:
        return self._start + int(Fraction(t).limit_denominator(1000) / self._v.time_base)

    def _time(self, pts: int) -> float:
        return float((pts - self._start) * self._v.time_base)

    def _index(self) -> list[tuple[float, int]]:
        """The video's keyframes, from its container's index; [] without one."""
        v = self._v
        try:
            # Matroska reads its cues at the first seek
            self._c.seek(self._start, stream=v, backward=True)
        except av.error.FFmpegError:
            return []
        keys = {e.timestamp for e in (v.index_entries or []) if e.is_keyframe}
        return sorted((self._time(ts), ts) for ts in keys)

    def run(self, sink: Sink) -> Sampling:
        """Decode the samples, handing each to sink(time, frame)."""
        sm = self.sampling
        dur = max(sm.duration_s, self._catalog_s)
        sm.budget = budget_bytes(dur)
        if dur <= SHORT_S and self._io.size() <= SEQUENTIAL_MAX_BYTES:
            sm.mode = "sequential"
            sm.budget = max(sm.budget, self._io.size())
            self._sequential(sink)
        else:
            self._keys = self._index()
            # an index of the keyframes, not of a few seek points: none
            # further apart than INDEX_GAP_MAX_S over the whole title
            times = [t for t, _ts in self._keys]
            covered = len(times) >= 2 and _max_gap(times, sm.duration_s) <= INDEX_GAP_MAX_S
            self._v.codec_context.skip_frame = "NONKEY"
            if covered:
                sm.mode = "keyframes"
                self._keyframes_and_stretches([t for t, _ts in self._keys], sink, None)
            else:
                sm.mode = "grid"
                step = gap_for(dur)
                self._keys = [(k * step, self._ts(k * step)) for k in range(int(dur / step) + 1)]
                self._keyframes_and_stretches([t for t, _ts in self._keys], sink, set())
        sm.bytes_read = self._io.bytes
        return sm

    def _key(self, i: int) -> tuple[float, Any] | None:
        t, ts = self._keys[i]
        grid = self.sampling.mode == "grid"
        frame = self._first_frame(ts, keyframe_only=not grid)
        if frame is None:
            return None
        if grid and frame.pts is not None:
            t = self._time(frame.pts)
        return t, frame

    def _first_frame(self, ts: int, keyframe_only: bool) -> Any | None:
        """The first keyframe decoded at or after a seek to ts (the stream's
        time base); None when there is none (a failure is counted)."""
        sm, v, c = self.sampling, self._v, self._c
        cc = v.codec_context
        try:
            cc.flush_buffers()
            c.seek(ts, stream=v, backward=True, any_frame=False)
            limit = self._io.bytes + GRID_READ_MAX
            for pkt in c.demux(v):
                if pkt.size == 0:
                    break
                if keyframe_only and not pkt.is_keyframe:
                    continue
                frames = cc.decode(pkt)
                if not frames and (keyframe_only or pkt.is_keyframe):
                    frames = cc.decode(None)
                if frames:
                    if frames[0].is_corrupt:
                        sm.fail("a keyframe decoded broken")
                        return None
                    return frames[0]
                if self._io.bytes > limit:
                    sm.fail("no keyframe within the read limit of a sample")
                    return None
            sm.fail("no frame at a sample's time")
        except (av.error.FFmpegError, OSError, ValueError) as e:
            sm.fail(f"a sample could not be decoded: {e}")
        return None

    def _through(
        self, a: float, b: float, step: float, limit: int, sink: Sink, counted: bool
    ) -> None:
        """Decode from the keyframe at or before a to b, the reference
        frames, a sample every step."""
        sm, v, c = self.sampling, self._v, self._c
        cc = v.codec_context
        skip = cc.skip_frame
        nxt, keyed = a, False
        try:
            cc.flush_buffers()
            cc.skip_frame = "NONREF"
            c.seek(self._ts(a), stream=v, backward=True, any_frame=False)
            stop = self._io.bytes + limit
            for pkt in c.demux(v):
                try:
                    frames = cc.decode(pkt if pkt.size else None)
                except (av.error.FFmpegError, ValueError) as e:
                    sm.fail(f"a frame could not be decoded: {e}")
                    continue
                for frame in frames:
                    # nothing before the first keyframe, nothing broken: a
                    # decoder conceals what it lacks with grey
                    keyed = keyed or frame.key_frame
                    if frame.pts is None or not keyed or frame.is_corrupt:
                        continue
                    t = self._time(frame.pts)
                    if t > b:
                        return
                    if t + 1e-6 >= nxt:
                        sm.attempted += 1
                        self._take(frame, t, sink, counted)
                        nxt = t + step
                if self._io.bytes > stop:
                    sm.stopped = sm.stopped or "a stretch's read limit"
                    return
        except (av.error.FFmpegError, OSError) as e:
            sm.fail(f"the video could not be read through: {e}")
        finally:
            cc.flush_buffers()
            cc.skip_frame = skip

    def _stretch(self, a: float, b: float, sink: Sink) -> None:
        self._through(a, b, STRETCH_STEP_S, STRETCH_READ_MAX, sink, False)

    def _sequential(self, sink: Sink) -> None:
        sm = self.sampling
        sm.planned = max(1, int(sm.duration_s / SEQUENTIAL_STEP_S))
        self._through(0.0, float("inf"), SEQUENTIAL_STEP_S, 1 << 62, sink, True)


# -------------------------------------------------------------- packages
class Package(_Source):
    """A package's video rendition: its HLS media playlist."""

    def __init__(self, playlist: str, duration_s: float = 0.0) -> None:
        self.path = playlist
        self.sampling = sm = Sampling(mode="package")
        try:
            init_path, self._segments = fmp4.read_playlist(playlist)
            sm.bytes_open += os.path.getsize(playlist)
            with open(init_path, "rb") as f:
                data = f.read(4 << 20)
            sm.bytes_open += len(data)
            self._init = fmp4.parse_init(data)
        except fmp4.Unreadable as e:
            raise Unreadable(str(e)) from e
        except OSError as e:
            raise Unreadable(f"the package can't be read: {e}") from e
        self.files = [playlist, init_path] + [s.path for s in self._segments]
        last = self._segments[-1]
        sm.duration_s = last.start + last.duration
        sm.bytes_read = sm.bytes_open
        self._catalog_s = duration_s
        i = self._init
        self.video = Video(
            codec=i.codec,
            width=i.width,
            height=i.height,
            sar=i.sar,
            rotation=i.rotation,
            duration_s=sm.duration_s,
            container="hls-fmp4",
        )
        if i.rotation:
            raise Undecidable(f"the video is rotated by {i.rotation} degrees")
        try:
            self._cc = av.CodecContext.create(i.codec, "r")
            self._cc.extradata = i.extradata or None
            self._cc.thread_count = _threads()
        except (av.error.FFmpegError, ValueError) as e:
            raise Unreadable(f"no decoder for {i.codec}: {e}") from e
        self._file: tuple[str, Any] | None = None
        self._keys: list[fmp4.Sample] = []
        self._times: list[float] = []

    def close(self) -> None:
        if self._file is not None:
            self._file[1].close()
            self._file = None

    def _spent(self) -> int:
        return self.sampling.bytes_read - self.sampling.bytes_open

    def _read(self, s: fmp4.Sample) -> bytes:
        if self._file is None or self._file[0] != s.path:
            self.close()
            self._file = (s.path, open(s.path, "rb"))
        f = self._file[1]
        f.seek(s.offset)
        data = f.read(s.size)
        self.sampling.bytes_read += len(data)
        if len(data) != s.size:
            raise OSError(f"{os.path.basename(s.path)} ends before a sample")
        return data

    def _list(self, segments: list[fmp4.Segment], sync_only: bool) -> list[fmp4.Sample]:
        out: list[fmp4.Sample] = []
        for seg in segments:
            got, read = fmp4.samples(seg, self._init, sync_only)
            self.sampling.bytes_read += read
            out.extend(got)
        return out

    def run(self, sink: Sink) -> Sampling:
        sm = self.sampling
        dur = max(sm.duration_s, self._catalog_s)
        sm.budget = budget_bytes(dur)
        try:
            size = sum(os.path.getsize(s.path) for s in self._segments)
            if dur <= SHORT_S and size <= SEQUENTIAL_MAX_BYTES:
                sm.mode = "package-sequential"
                sm.budget = max(sm.budget, 2 * size)
                sm.planned = max(1, int(sm.duration_s / SEQUENTIAL_STEP_S))
                samples = self._list(self._segments, False)
                self._decode(samples, 0.0, float("inf"), SEQUENTIAL_STEP_S, sink, True)
                return sm
            ts = self._init.timescale
            keys = sorted(self._list(self._segments, True), key=lambda k: k.pts)
            self._keys, self._times = keys, [k.pts / ts for k in keys]
            self._cc.skip_frame = "NONKEY"
            self._keyframes_and_stretches(self._times, sink, None)
        except (fmp4.Unreadable, OSError) as e:
            raise Unreadable(str(e)) from e
        finally:
            self.close()
        return sm

    def _key(self, i: int) -> tuple[float, Any] | None:
        sm, cc, k = self.sampling, self._cc, self._keys[i]
        try:
            cc.flush_buffers()
            pkt = av.Packet(self._read(k))
            pkt.pts, pkt.dts = k.pts, k.dts
            frames = cc.decode(pkt) or cc.decode(None)
        except (av.error.FFmpegError, OSError, ValueError) as e:
            sm.fail(f"a keyframe could not be decoded: {e}")
            return None
        if not frames or frames[0].is_corrupt:
            sm.fail("a keyframe gave no frame" if not frames else "a keyframe decoded broken")
            return None
        return self._times[i], frames[0]

    def _stretch(self, a: float, b: float, sink: Sink) -> None:
        # the segments from the one holding the last keyframe at or before a
        k = bisect.bisect_right(self._times, a) - 1
        start = self._times[k] if k >= 0 else 0.0
        segs = [s for s in self._segments if s.start + s.duration > start - 1e-6 and s.start <= b]
        ts = self._init.timescale
        samples = sorted(self._list(segs, False), key=lambda s: s.dts)
        first = next(
            (i for i, s in enumerate(samples) if s.sync and s.pts / ts >= start - 1e-6), None
        )
        if first is None:
            return
        self._decode(samples[first:], a, b, STRETCH_STEP_S, sink, False)

    def _decode(
        self,
        samples: list[fmp4.Sample],
        a: float,
        b: float,
        step: float,
        sink: Sink,
        counted: bool,
    ) -> None:
        """Decode samples in their decode order, the reference frames, a sample
        every step from a to b."""
        sm, cc = self.sampling, self._cc
        skip = cc.skip_frame
        ts = self._init.timescale
        nxt = a

        def frames() -> Iterator[Any]:
            for s in samples:
                if s.dts / ts > b + 2.0:
                    break
                try:
                    pkt = av.Packet(self._read(s))
                    pkt.pts, pkt.dts = s.pts, s.dts
                    yield from cc.decode(pkt)
                except (av.error.FFmpegError, ValueError) as e:
                    sm.fail(f"a frame could not be decoded: {e}")
            yield from cc.decode(None)

        keyed = False
        try:
            cc.flush_buffers()
            cc.skip_frame = "NONREF"
            for frame in frames():
                keyed = keyed or frame.key_frame
                if not keyed or frame.is_corrupt:
                    continue
                t = (frame.pts or 0) / ts
                if t > b:
                    break
                if t + 1e-6 >= nxt:
                    sm.attempted += 1
                    self._take(frame, t, sink, counted)
                    nxt = t + step
        finally:
            cc.flush_buffers()
            cc.skip_frame = skip


def open_source(path: str, duration_s: float = 0.0) -> Original | Package:
    """The source of the video at path: a package's rendition (an .m3u8),
    else an original."""
    if path.lower().endswith(".m3u8"):
        return Package(path, duration_s)
    return Original(path, duration_s)


def identity(paths: list[str]) -> list[tuple[int, int, int, int] | None]:
    """The size, the modification time, the inode and the device of each
    file: what tells it changed."""
    out: list[tuple[int, int, int, int] | None] = []
    for p in paths:
        try:
            st = os.stat(p)
            out.append((st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev))
        except OSError:
            out.append(None)
    return out
