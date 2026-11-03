"""Where a measurement's samples come from: the frames of a title, decoded
from a file opened once, each labelled with its own presentation time.

The title's bar black is read first (calibrate()): the edges of CALIBRATION
frames spread over it (planes.py). Then the samples.

An original is read so that no two samples are more than TARGET_GAP_S
apart anywhere, its start and its end too:

  * of a container with an index of its keyframes (Matroska's cues, MP4's
    sample table, AVI's index; none further apart than INDEX_GAP_MAX_S),
    every keyframe (at most one each MIN_KEY_SPACING_S: an intra-only file
    is every frame a keyframe), a seek to each and the keyframe decoded
    alone; between two keyframes further apart than TARGET_GAP_S, the group
    of pictures decoded through (its reference frames), the latest frame
    within TARGET_GAP_S of the last sample taken each time;
  * of one without (MPEG-TS, MPEG-PS, a Matroska file without cues or cut
    short), the whole file in its order, each group of pictures held until
    the next keyframe says how long it is: its keyframe alone when it is
    short, decoded through when it is long;
  * a title of at most SHORT_S seconds whose file is at most
    SEQUENTIAL_MAX_BYTES is decoded through, a sample every
    SEQUENTIAL_STEP_S.

Besides, STRETCHES stretches spread over the title, STRETCH_SHARE of it
together (STRETCH_MIN_S to STRETCH_MAX_S each), are decoded through, a
sample every STRETCH_STEP_S: what changes within a shot for less than
TARGET_GAP_S. An original's reads have no budget but twice its size
(PICTURE_ORIGINAL_READ_MB_PER_HOUR sets one).

A package's video rendition (its HLS playlist of fMP4 segments) is read by
its sync samples, each by its byte range and decoded with the init
segment's configuration, no two picked further apart than TARGET_GAP_S
where they allow it, coarse to fine (every 16th first, then every 8th, the
stretches, the rest) within READ_MB_PER_HOUR of the title
(PICTURE_READ_MB_PER_HOUR), at least READ_MIN_MB. An encoder's rung with a
keyframe every 6 s and none at a cut leaves its samples 6 s apart: the
measurement says it can't tell (child.py).

What the samples are is what was decoded: a seek that lands on another
frame than the keyframe it sought is counted (missed), a frame sampled twice
too (duplicates), and the gaps are those between the frames' own times.
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

from . import fmp4, planes

TARGET_GAP_S = 2.9
MIN_KEY_SPACING_S = 0.5
# A keyframe decoded at more than this from the one sought is another.
MISSED_TOLERANCE_S = 0.5
SHORT_S = 300.0
SEQUENTIAL_STEP_S = 1.0
SEQUENTIAL_MAX_BYTES = 512 << 20
STRETCHES = 12
STRETCH_SHARE = 0.02
STRETCH_MIN_S = 4.0
STRETCH_MAX_S = 20.0
STRETCH_STEP_S = 0.5
CALIBRATION = 24
CALIBRATION_MIN = 8
READ_MB_PER_HOUR = 1024
READ_MIN_MB = 96
# The most keyframes a package's title takes: a longer one's are further
# apart.
MAX_SAMPLES = 6000
# A container's index lists the keyframes when none of its entries are
# further apart than this (an MPEG-TS's is a few points its probe found).
INDEX_GAP_MAX_S = 30.0
# A seek reads at most this much to find a keyframe, a group of pictures
# decoded through (of a stretch, of a run of long ones) this much; a group of
# pictures of a file read in its order is held up to GOP_HOLD_MAX before it
# is decoded as it comes.
SEEK_READ_MAX = 48 << 20
THROUGH_READ_MAX = 512 << 20
GOP_HOLD_MAX = 64 << 20
# A package's keyframe passes before the stretches, and after them.
COARSE, FINE = (16, 8), (4, 2, 1)

Sink = Callable[[float, Any, bool], None]


class Unreadable(Exception):
    """The video can't be read."""


class Undecidable(Exception):
    """The video is read, and can't be measured (rotated, none)."""


@dataclass
class Sampling:
    """What a source did: how it sampled (keyframes, stream, sequential, of
    an original, package, package-sequential), the title's duration as its
    file says, what it meant to take, the samples it tried, decoded and
    failed, the keyframes it sought and decoded another frame for
    (missed), the frames it decoded twice (duplicates), the frames read to
    find the bar black (calibration), the bytes it read to open the file
    and in all, the budget and whether it stopped there, and the samples'
    own times."""

    mode: str = ""
    duration_s: float = 0.0
    planned: int = 0
    attempted: int = 0
    decoded: int = 0
    failed: int = 0
    missed: int = 0
    duplicates: int = 0
    stretches: int = 0
    calibration: int = 0
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

    def end_s(self) -> float:
        """Where the title ends: its file's duration, or its last sample."""
        return max(self.duration_s, max(self.times, default=0.0))

    def largest_gap(self) -> tuple[float, float] | None:
        """The longest stretch of the title without a sample, from its start
        to its end, and where it begins."""
        if not self.times:
            return None
        t = sorted(set(self.times))
        best = (t[0], 0.0)
        for a, b in itertools.pairwise(t):
            if b - a > best[0]:
                best = (b - a, a)
        end = self.end_s()
        if end - t[-1] > best[0]:
            best = (end - t[-1], t[-1])
        return best

    def max_gap(self) -> float | None:
        g = self.largest_gap()
        return None if g is None else g[0]


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


def _per_hour(env: str, default: float) -> float:
    try:
        v = float(os.environ.get(env) or default)
    except ValueError:
        return default
    return min(max(v, 0.0), 1 << 20)


def budget_bytes(duration_s: float) -> int:
    """The read budget of a package's title of duration_s seconds."""
    per_hour = max(_per_hour("PICTURE_READ_MB_PER_HOUR", READ_MB_PER_HOUR), 16.0)
    return int(max(READ_MIN_MB, per_hour * duration_s / 3600.0) * (1 << 20))


def original_budget(duration_s: float, size: int) -> int:
    """The read budget of an original of duration_s seconds and size bytes:
    twice its size, or PICTURE_ORIGINAL_READ_MB_PER_HOUR of the title."""
    per_hour = _per_hour("PICTURE_ORIGINAL_READ_MB_PER_HOUR", 0.0)
    if per_hour <= 0:
        return 2 * size + (64 << 20)
    return int(max(4.0, per_hour * duration_s / 3600.0) * (1 << 20))


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


def thin(times: list[float], spacing: float) -> list[int]:
    """The indexes of times (ascending), each at least spacing after the
    last one kept."""
    out: list[int] = []
    for i, t in enumerate(times):
        if not out or t - times[out[-1]] >= spacing:
            out.append(i)
    return out


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
    """The largest gap the keyframes picked of a package this long aim at."""
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


class _Taker:
    """Takes the samples of a run of decoded frames: the first, then each
    time a frame lies more than step after the last sample taken, the
    latest frame within step of it (or that frame, when none is): no two
    samples further apart than step where the frames allow it. A frame
    taken already (by another run over the same stretch) counts as taken."""

    def __init__(self, source: _Source, sink: Sink, counted: bool) -> None:
        self.source, self.sink, self.counted = source, sink, counted
        self.last: float | None = None
        self.prev: tuple[float, Any] | None = None
        self.keyed = False

    def _take(self, t: float, frame: Any) -> None:
        self.source._take(frame, t, self.sink, self.counted)
        self.last = t

    def frame(self, t: float, frame: Any, step: float, always: bool = False) -> None:
        """A decoded frame at t; always (a keyframe of an original's: a shot's
        first frame) takes it whatever the spacing, but MIN_KEY_SPACING_S
        after the last sample at the least."""
        if self.last is None:
            self._take(t, frame)
            return
        if always and t - self.last >= MIN_KEY_SPACING_S:
            p = self.prev
            if t - self.last > step and p is not None and p[0] > self.last:
                self._take(*p)
            self._take(t, frame)
            self.prev = None
            return
        if t - self.last > step:
            p = self.prev
            if p is not None and p[0] > self.last:
                self._take(*p)
            if t - self.last > step:
                self._take(t, frame)
                self.prev = None
                return
        self.prev = (t, frame)

    def close_at(self, t_next: float, step: float) -> None:
        """The run ends where the next sample is due (t_next): take the
        latest frame held when the gap to it would be more than step."""
        p = self.prev
        if (
            p is not None
            and self.last is not None
            and t_next - self.last > step
            and p[0] > self.last
        ):
            self._take(*p)
        self.prev = None


class _Source:
    """What an original and a package share."""

    sampling: Sampling
    video: Video
    blacks: planes.Blacks
    _catalog_s: float

    def __init__(self) -> None:
        self.blacks = {}
        self._seen: set[int] = set()

    def _spent(self) -> int:
        """The bytes read so far, besides opening the file."""
        raise NotImplementedError

    def _over(self) -> bool:
        sm = self.sampling
        if self._spent() >= sm.budget:
            sm.stopped = "budget"
            return True
        return False

    def _calibrated(self, seen: list[dict[str, tuple[float, float]]]) -> None:
        sm = self.sampling
        sm.calibration = len(seen)
        if len(seen) < CALIBRATION_MIN:
            raise Unreadable(
                f"only {len(seen)} of {CALIBRATION} frames spread over the title could be read"
            )
        self.blacks = planes.calibrate(seen)

    def _take(self, frame: Any, t: float, sink: Sink, counted: bool = True) -> bool:
        """Hand a decoded frame at its own time t to sink; False when it is
        taken already (a frame decoded twice)."""
        sm = self.sampling
        key = round(t * 1000)
        if key in self._seen:
            return False
        if frame.width != self.video.width or frame.height != self.video.height:
            if sm.decoded == 0:
                # what the frames are is what the players show
                self.video.width, self.video.height = frame.width, frame.height
            else:
                sm.fail(
                    f"a frame of {frame.width}x{frame.height} in a video of "
                    f"{self.video.width}x{self.video.height}"
                )
                return True
        if frame.rotation:
            raise Undecidable(f"the video is rotated by {frame.rotation % 360} degrees")
        self._seen.add(key)
        sm.decoded += 1
        sm.times.append(t)
        sink(t, frame, counted)
        return True


# ------------------------------------------------------------- originals
class Original(_Source):
    """An original: a file libavformat reads, opened once."""

    def __init__(self, path: str, duration_s: float = 0.0) -> None:
        super().__init__()
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
        """Decode the samples, handing each to sink(time, frame, counted)."""
        sm = self.sampling
        dur = max(sm.duration_s, self._catalog_s)
        size = self._io.size()
        sm.budget = original_budget(dur, size)
        # the index as the container has it, before any seek of the
        # calibration's adds the keyframes it passes to it
        keys = self._index()
        self._calibrate(sm.duration_s or dur)
        if dur <= SHORT_S and size <= SEQUENTIAL_MAX_BYTES:
            sm.mode = "sequential"
            sm.planned = max(1, int(sm.duration_s / SEQUENTIAL_STEP_S))
            self._stream(sink, everything=True)
        else:
            times = [t for t, _ts in keys]
            if len(times) >= 2 and _max_gap(times, sm.duration_s) <= INDEX_GAP_MAX_S:
                sm.mode = "keyframes"
                self._indexed(keys, sink)
            else:
                sm.mode = "stream"
                self._stream(sink, everything=False)
        sm.bytes_read = self._io.bytes
        return sm

    def _calibrate(self, dur: float) -> None:
        seen = []
        for k in range(CALIBRATION):
            frame = self._seek_frame(self._ts((k + 0.5) * dur / CALIBRATION), False, count=False)
            if frame is not None:
                seen.append(planes.frame_edges(frame))
        self._calibrated(seen)

    def _seek_frame(
        self, ts: int, keyframe_only: bool, count: bool = True, after: float | None = None
    ) -> Any | None:
        """The first keyframe decoded at or after a seek to ts (the stream's
        time base), and not before after (seconds: a seek that landed on an
        earlier keyframe reads on); None when there is none (a failure, when
        counted)."""
        sm, v, c = self.sampling, self._v, self._c
        cc = v.codec_context
        why = "no frame at a sample's time"
        try:
            cc.flush_buffers()
            cc.skip_frame = "NONKEY"
            c.seek(ts, stream=v, backward=True, any_frame=False)
            limit = self._io.bytes + SEEK_READ_MAX
            for pkt in c.demux(v):
                if pkt.size == 0:
                    break
                if keyframe_only and not pkt.is_keyframe:
                    continue
                if after is not None:
                    stamps = [x for x in (pkt.pts, pkt.dts) if x is not None]
                    if stamps and self._time(max(stamps)) < after:
                        if self._io.bytes > limit:
                            why = "no keyframe within the read limit of a seek"
                            break
                        continue
                frames = cc.decode(pkt)
                if not frames and (keyframe_only or pkt.is_keyframe):
                    frames = cc.decode(None)
                if frames:
                    if frames[0].is_corrupt or frames[0].pts is None:
                        why = "a keyframe decoded broken"
                        break
                    return frames[0]
                if self._io.bytes > limit:
                    why = "no keyframe within the read limit of a seek"
                    break
        except (av.error.FFmpegError, OSError, ValueError) as e:
            why = f"a sample could not be decoded: {e}"
        if count:
            sm.fail(why)
        return None

    def _indexed(self, keys: list[tuple[float, int]], sink: Sink) -> None:
        """Every keyframe, the long groups of pictures decoded through, the
        stretches; in the order of the title."""
        sm = self.sampling
        end = sm.duration_s
        times = [t for t, _ts in keys]
        picked = thin(times, MIN_KEY_SPACING_S)

        def target(k: int) -> int:
            # A seek lands on the keyframe at or before its time as the
            # container reckons it: an MP4's index holds decoding times and
            # its seeks reckon presentation times, so one to a keyframe's own
            # entry lands on the keyframe before it where frames are
            # reordered. Halfway to the next keyframe lands on this one.
            ts = keys[k][1]
            nxt = keys[k + 1][1] if k + 1 < len(keys) else self._ts(max(end, keys[k][0] + 1.0))
            return ts + max(0, (nxt - ts) // 2)

        plan: list[tuple[float, int, str, int, float]] = []
        for i, k in enumerate(picked):
            t = keys[k][0]
            nxt = keys[picked[i + 1]][0] if i + 1 < len(picked) else end
            if nxt - t <= TARGET_GAP_S:
                plan.append((t, 0, "key", target(k), nxt))
            elif plan and plan[-1][2] == "fill" and abs(plan[-1][4] - t) < 1e-6:
                # long groups of pictures one after the other: one run
                plan[-1] = (*plan[-1][:4], nxt)
            else:
                plan.append((t, 0, "fill", target(k), nxt))
        for a, b in stretches(end):
            plan.append((a, 1, "stretch", self._ts(a), b))
        plan.sort()
        sm.planned = len(plan)
        for t, _order, kind, ts, b in plan:
            if self._over():
                return
            if kind == "key":
                self._key(t, ts, sink)
            elif kind == "fill":
                self._through(t, ts, b, TARGET_GAP_S, sink, True)
            else:
                sm.stretches += 1
                self._through(t, ts, b, STRETCH_STEP_S, sink, False)

    def _key(self, t: float, ts: int, sink: Sink) -> None:
        sm = self.sampling
        sm.attempted += 1
        frame = self._seek_frame(ts, keyframe_only=True, after=t - MISSED_TOLERANCE_S)
        if frame is None:
            return
        got = self._time(frame.pts)
        missed = abs(got - t) > MISSED_TOLERANCE_S
        if missed:
            sm.missed += 1
            if len(sm.errors) < 5:
                sm.errors.append(
                    f"a seek to the keyframe at {t:.1f} s decoded the frame at {got:.1f} s"
                )
        if not self._take(frame, got, sink, True) and missed:
            # landed on a frame another sample took
            sm.duplicates += 1

    def _through(self, a: float, ts: int, b: float, step: float, sink: Sink, counted: bool) -> None:
        """Decode from the keyframe at or before a (ts) up to b, its reference
        frames, the samples no further apart than step."""
        sm, v, c = self.sampling, self._v, self._c
        cc = v.codec_context
        taker = _Taker(self, sink, counted)
        keyed = False
        try:
            cc.flush_buffers()
            cc.skip_frame = "NONREF"
            c.seek(ts, stream=v, backward=True, any_frame=False)
            stop = self._io.bytes + THROUGH_READ_MAX
            for pkt in c.demux(v):
                if pkt.is_keyframe:
                    # the limit is a group of pictures': a run of them
                    # one after the other reads on
                    stop = self._io.bytes + THROUGH_READ_MAX
                try:
                    frames = cc.decode(pkt if pkt.size else None)
                except (av.error.FFmpegError, ValueError) as e:
                    sm.fail(f"a frame could not be decoded: {e}")
                    continue
                done = False
                for frame in frames:
                    # nothing before the first keyframe, nothing broken: a
                    # decoder conceals what it lacks with grey
                    keyed = keyed or frame.key_frame
                    if frame.pts is None or not keyed or frame.is_corrupt:
                        continue
                    t = self._time(frame.pts)
                    if t >= b - 1e-6:
                        done = True
                        break
                    if t + 1e-6 < a:
                        continue
                    sm.attempted += 1
                    taker.frame(t, frame, step, always=counted and frame.key_frame)
                if done or not pkt.size:
                    break
                if self._io.bytes > stop:
                    sm.stopped = sm.stopped or "a group of pictures decoded through read its limit"
                    break
                if self._over():
                    break
        except (av.error.FFmpegError, OSError) as e:
            sm.fail(f"the video could not be read through: {e}")
        finally:
            taker.close_at(b, step)
            cc.flush_buffers()

    def _stream(self, sink: Sink, everything: bool) -> None:
        """Read the file in its order: each group of pictures held until the
        next keyframe says how long it is, then its keyframe decoded alone,
        or the group decoded through when it is long (everything: always)."""
        sm, v, c = self.sampling, self._v, self._c
        windows = stretches(sm.duration_s)
        step = SEQUENTIAL_STEP_S if everything else TARGET_GAP_S
        gop: list[Any] = []
        held = 0
        t0: float | None = None
        live: _Taker | None = None  # a group decoded as it comes (held too long)
        last_key: list[float | None] = [None]

        def pkt_time(pkt: Any) -> float | None:
            ts = pkt.pts if pkt.pts is not None else pkt.dts
            return None if ts is None else self._time(ts)

        def finish(t_next: float) -> None:
            nonlocal gop, held, live
            if live is not None:
                self._drain(live, t_next, step)
                live = None
            elif gop and t0 is not None:
                long = everything or t_next - t0 > TARGET_GAP_S
                stretch = any(a < t_next and b > t0 for a, b in windows)
                if long or stretch:
                    taker = _Taker(self, sink, True)
                    self._decode_gop(gop, taker, step, windows)
                    taker.close_at(t_next, step)
                elif last_key[0] is None or t0 - last_key[0] >= MIN_KEY_SPACING_S:
                    # (an intra-only file: one keyframe each half second)
                    self._decode_key(gop[0], sink)
                    last_key[0] = t0
            gop, held = [], 0

        try:
            # back to the file's start after the calibration's seeks: to a
            # time before any stream's, as a transport stream's search lands
            # past its first keyframe for its video's own start (that
            # keyframe's decoding time is before it)
            c.seek(0)
        except av.error.FFmpegError:
            try:
                c.seek(self._start, stream=v, backward=True)
            except av.error.FFmpegError:
                pass
        try:
            for pkt in c.demux(v):
                if not pkt.size:
                    break
                if self._over():
                    return
                if pkt.is_keyframe:
                    t = pkt_time(pkt)
                    if t is not None:
                        finish(t)
                        t0 = t
                        gop, held = [pkt], pkt.size
                    continue
                if live is not None:
                    self._decode_gop([pkt], live, step, windows, flush=False)
                    continue
                if not gop:
                    continue  # before the first keyframe
                gop.append(pkt)
                held += pkt.size
                if held > GOP_HOLD_MAX:
                    live = _Taker(self, sink, True)
                    self._decode_gop(gop, live, step, windows, flush=False)
                    gop = []
            finish(sm.duration_s)
        except (av.error.FFmpegError, OSError) as e:
            sm.fail(f"the video could not be read through: {e}")

    def _decode_key(self, pkt: Any, sink: Sink) -> None:
        sm, cc = self.sampling, self._v.codec_context
        sm.attempted += 1
        try:
            cc.flush_buffers()
            cc.skip_frame = "NONKEY"
            frames = cc.decode(pkt) or cc.decode(None)
        except (av.error.FFmpegError, ValueError) as e:
            sm.fail(f"a keyframe could not be decoded: {e}")
            return
        if not frames or frames[0].is_corrupt or frames[0].pts is None:
            sm.fail("a keyframe gave no frame" if not frames else "a keyframe decoded broken")
            return
        if not self._take(frames[0], self._time(frames[0].pts), sink, True):
            sm.duplicates += 1

    def _decode_gop(
        self,
        pkts: list[Any],
        taker: _Taker,
        step: float,
        windows: list[tuple[float, float]],
        flush: bool = True,
    ) -> None:
        """Decode a group of pictures' packets through (its reference frames),
        handing its frames to taker as they come (a frame held is a frame's
        memory): a sample every step, every STRETCH_STEP_S within a
        stretch."""
        sm, cc = self.sampling, self._v.codec_context
        if pkts and pkts[0].is_keyframe:
            cc.flush_buffers()
            cc.skip_frame = "NONREF"
            taker.keyed = False

        def hand(frames: list[Any]) -> None:
            for frame in frames:
                # nothing before the first keyframe, nothing broken: a
                # decoder conceals what it lacks with grey
                taker.keyed = taker.keyed or frame.key_frame
                if frame.pts is None or not taker.keyed or frame.is_corrupt:
                    continue
                t = self._time(frame.pts)
                sm.attempted += 1
                inside = any(a <= t <= b for a, b in windows)
                taker.frame(t, frame, STRETCH_STEP_S if inside else step, always=frame.key_frame)

        for pkt in pkts:
            try:
                frames = cc.decode(pkt)
            except (av.error.FFmpegError, ValueError) as e:
                sm.fail(f"a frame could not be decoded: {e}")
                continue
            hand(frames)
        if flush:
            try:
                hand(cc.decode(None))
            except (av.error.FFmpegError, ValueError):
                pass

    def _drain(self, taker: _Taker, t_next: float, step: float) -> None:
        sm, cc = self.sampling, self._v.codec_context
        try:
            for frame in cc.decode(None):
                if frame.pts is None or frame.is_corrupt or not taker.keyed:
                    continue
                sm.attempted += 1
                taker.frame(self._time(frame.pts), frame, step)
        except (av.error.FFmpegError, ValueError):
            pass
        taker.close_at(t_next, step)


# -------------------------------------------------------------- packages
class Package(_Source):
    """A package's video rendition: its HLS media playlist."""

    def __init__(self, playlist: str, duration_s: float = 0.0) -> None:
        super().__init__()
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
            ts = self._init.timescale
            keys = sorted(self._list(self._segments, True), key=lambda k: k.pts)
            self._keys, self._times = keys, [k.pts / ts for k in keys]
            if not keys:
                raise Unreadable("the rendition has no sync sample")
            self._calibrate()
            size = sum(os.path.getsize(s.path) for s in self._segments)
            if dur <= SHORT_S and size <= SEQUENTIAL_MAX_BYTES:
                sm.mode = "package-sequential"
                sm.budget = max(sm.budget, 2 * size)
                sm.planned = max(1, int(sm.duration_s / SEQUENTIAL_STEP_S))
                taker = _Taker(self, sink, True)
                samples = self._list(self._segments, False)
                self._decode(samples, 0.0, float("inf"), SEQUENTIAL_STEP_S, taker)
                taker.close_at(sm.duration_s, SEQUENTIAL_STEP_S)
                return sm
            self._keyframes_and_stretches(sink, dur)
        except (fmp4.Unreadable, OSError) as e:
            raise Unreadable(str(e)) from e
        finally:
            self.close()
        return sm

    def _calibrate(self) -> None:
        n = len(self._keys)
        seen = []
        for k in range(CALIBRATION):
            got = self._key(min(n - 1, int((k + 0.5) * n / CALIBRATION)), count=False)
            if got is not None:
                seen.append(planes.frame_edges(got))
        self._calibrated(seen)

    def _keyframes_and_stretches(self, sink: Sink, dur: float) -> None:
        """The sync samples, coarse to fine, and the stretches."""
        sm = self.sampling
        picks = cover(self._times, gap_for(dur))
        sm.planned = len(picks)
        done: set[int] = set()

        def keys(strides: tuple[int, ...]) -> bool:
            for p in passes(picks, strides, done):
                for i in p:
                    if self._over():
                        return False
                    sm.attempted += 1
                    frame = self._key(i)
                    if frame is not None:
                        # the frame of the sample's own bytes: taken already
                        # by a stretch, it is no other
                        self._take(frame, self._times[i], sink, True)
            return True

        if not keys(COARSE):
            return
        for a, b in stretches(sm.duration_s or dur):
            if self._over():
                return
            sm.stretches += 1
            self._stretch(a, b, sink)
        keys(FINE)

    def _key(self, i: int, count: bool = True) -> Any | None:
        sm, cc, k = self.sampling, self._cc, self._keys[i]
        why = ""
        try:
            cc.flush_buffers()
            cc.skip_frame = "NONKEY"
            pkt = av.Packet(self._read(k))
            pkt.pts, pkt.dts = k.pts, k.dts
            frames = cc.decode(pkt) or cc.decode(None)
            if frames and not frames[0].is_corrupt:
                return frames[0]
            why = "a keyframe gave no frame" if not frames else "a keyframe decoded broken"
        except (av.error.FFmpegError, OSError, ValueError) as e:
            why = f"a keyframe could not be decoded: {e}"
        if count:
            sm.fail(why)
        return None

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
        taker = _Taker(self, sink, False)
        self._decode(samples[first:], a, b, STRETCH_STEP_S, taker)
        taker.close_at(b, STRETCH_STEP_S)

    def _decode(
        self, samples: list[fmp4.Sample], a: float, b: float, step: float, taker: _Taker
    ) -> None:
        """Decode samples in their decode order, the reference frames, handing
        those from a to b to taker."""
        sm, cc = self.sampling, self._cc
        ts = self._init.timescale

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
                if not keyed or frame.is_corrupt or frame.pts is None:
                    continue
                t = frame.pts / ts
                if t > b:
                    break
                if t + 1e-6 < a:
                    continue
                sm.attempted += 1
                taker.frame(t, frame, step)
        finally:
            cc.flush_buffers()


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
