"""Reading a package's video rendition: an HLS media playlist of fragmented
MP4 segments (an init segment, then segments of movie fragments).

A package is sampled by its keyframes: each fragment's track runs (trun)
list its samples, their sizes and whether each is a sync sample (a
keyframe), so a keyframe is read by its byte range alone and decoded with
the init segment's decoder configuration (hvcC, avcC), no demuxer between.
Only what the fragments' headers take is read besides: each segment's box
headers and its moof boxes.
"""

from __future__ import annotations

import math
import os
import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass, replace

# The sample entries a package's video rendition has, and the decoder each
# needs, with the box of its configuration.
_ENTRIES = {
    b"hvc1": ("hevc", b"hvcC"),
    b"hev1": ("hevc", b"hvcC"),
    b"avc1": ("h264", b"avcC"),
    b"avc3": ("h264", b"avcC"),
    b"av01": ("av1", b"av1C"),
    b"vp09": ("vp9", b"vpcC"),
}
# The containers a box lists boxes in, down to a video sample entry.
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"mvex"}
# A sample's flags: sample_is_non_sync_sample.
_NON_SYNC = 0x00010000


class Unreadable(Exception):
    """A playlist, an init segment or a segment that can't be read as a
    package's."""


@dataclass(frozen=True)
class Init:
    """A video rendition's init segment: its decoder and configuration, its
    frame, its timescale, its pixel aspect (1, 1 for square pixels), how
    far its track's matrix turns it (degrees), and the defaults its
    fragments' samples take (trex) where they say none."""

    codec: str
    extradata: bytes
    width: int
    height: int
    timescale: int
    sar: tuple[int, int] = (1, 1)
    rotation: int = 0
    default_duration: int = 0
    default_size: int = 0
    default_flags: int = 0


@dataclass(frozen=True)
class Sample:
    """A sample of a segment: its decode and presentation time (in the
    track's timescale), whether it is a sync sample (a keyframe), and its
    bytes, as offset and size in the segment file."""

    dts: int
    pts: int
    sync: bool
    path: str
    offset: int
    size: int


@dataclass(frozen=True)
class Segment:
    """A playlist's segment: where it starts (seconds), how long it is, its
    file."""

    start: float
    duration: float
    path: str


_EXTINF = re.compile(r"^#EXTINF:\s*([0-9.]+)")
_MAP = re.compile(r'^#EXT-X-MAP:.*URI="([^"]+)"')


def read_playlist(path: str) -> tuple[str, list[Segment]]:
    """The init segment of an HLS media playlist and its segments, as paths
    in the playlist's folder. A playlist of byte ranges (one file), of a URI
    that is a URL or leaves the folder, or without an init segment is
    Unreadable."""
    base = os.path.dirname(os.path.abspath(path))
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f]
    except OSError as e:
        raise Unreadable(f"the playlist can't be read: {e}") from e
    init: str | None = None
    out: list[Segment] = []
    at, pending = 0.0, None
    for ln in lines:
        if not ln:
            continue
        if ln.startswith("#EXT-X-BYTERANGE") or (ln.startswith("#EXT-X-MAP") and "BYTERANGE" in ln):
            raise Unreadable("a playlist of byte ranges")
        mm = _MAP.match(ln)
        if mm:
            init = _beside(base, mm.group(1))
            continue
        mi = _EXTINF.match(ln)
        if mi:
            pending = float(mi.group(1))
            continue
        if ln.startswith("#"):
            continue
        seg = _beside(base, ln)
        if pending is None:
            raise Unreadable("a segment without its duration")
        out.append(Segment(at, pending, seg))
        at += pending
        pending = None
    if init is None or not out:
        raise Unreadable("the playlist names no init segment, or no segment")
    return init, out


def _beside(base: str, uri: str) -> str:
    if "://" in uri or uri.startswith("/") or ".." in uri.split("/"):
        raise Unreadable(f"the playlist names {uri!r}, which is not in its folder")
    return os.path.join(base, uri)


def _boxes(b: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """The boxes of b[start:end]: (type, where its body starts, where it ends)."""
    off = start
    while off + 8 <= end:
        size, kind = struct.unpack(">I4s", b[off : off + 8])
        head = 8
        if size == 1:
            if off + 16 > end:
                return
            size = struct.unpack(">Q", b[off + 8 : off + 16])[0]
            head = 16
        elif size == 0:
            size = end - off
        if size < head or off + size > end:
            return
        yield kind, off + head, off + size
        off += size


def parse_init(data: bytes) -> Init:
    """The video track of an init segment (moov): its sample entry's decoder
    and configuration, its frame and pixel aspect, its media timescale, and
    the samples' defaults of its movie extends box (trex)."""
    found = _find_video(data, 0, len(data))
    if found is None:
        raise Unreadable("the init segment has no video track this reads")
    trex = _find_trex(data, 0, len(data)) or (0, 0, 0)
    return replace(found, default_duration=trex[0], default_size=trex[1], default_flags=trex[2])


def _find_trex(b: bytes, start: int, end: int) -> tuple[int, int, int] | None:
    for kind, body, stop in _boxes(b, start, end):
        if kind in (b"moov", b"mvex"):
            got = _find_trex(b, body, stop)
            if got is not None:
                return got
        elif kind == b"trex" and stop - body >= 24:
            return struct.unpack(">III", b[body + 12 : body + 24])
    return None


def _rotation(b: bytes, body: int, stop: int) -> int:
    """How far a track header's (tkhd) matrix turns its picture, in degrees
    (0, 90, 180, 270)."""
    p = body + 4 + (32 if b[body] == 1 else 20) + 16
    if p + 16 > stop:
        return 0
    a, bb, _u, c = struct.unpack(">iiii", b[p : p + 16])
    return round(math.degrees(math.atan2(bb, a))) % 360 if (a, bb, c) != (0, 0, 0) else 0


def _find_video(
    b: bytes, start: int, end: int, timescale: int = 0, rotation: int = 0
) -> Init | None:
    for kind, body, stop in _boxes(b, start, end):
        if kind == b"tkhd":
            rotation = _rotation(b, body, stop)
        elif kind == b"mdhd":
            version = b[body]
            p = body + 4 + (16 if version == 1 else 8)
            timescale = struct.unpack(">I", b[p : p + 4])[0]
        elif kind in _CONTAINERS:
            # a container's children in their order: an mdia's mdhd (its
            # timescale) comes before its minf
            got = _find_video(b, body, stop, timescale, rotation)
            if got is not None:
                return got
        elif kind == b"stsd":
            for entry, ebody, estop in _boxes(b, body + 8, stop):
                if entry not in _ENTRIES:
                    continue
                codec, config = _ENTRIES[entry]
                width, height = struct.unpack(">HH", b[ebody + 24 : ebody + 28])
                extradata, sar = b"", (1, 1)
                for child, cbody, cstop in _boxes(b, ebody + 78, estop):
                    if child == config:
                        extradata = b[cbody:cstop]
                    elif child == b"pasp" and cstop - cbody >= 8:
                        h, v = struct.unpack(">II", b[cbody : cbody + 8])
                        if h > 0 and v > 0:
                            sar = (h, v)
                return Init(codec, extradata, width, height, timescale or 90000, sar, rotation)
    return None


def samples(segment: Segment, init: Init, sync_only: bool = True) -> tuple[list[Sample], int]:
    """The samples of a segment (its sync samples only, by default) in decode
    order, and the bytes read to find them (its box headers and its moof
    boxes)."""
    out: list[Sample] = []
    read = 0
    try:
        with open(segment.path, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            off = 0
            while off + 8 <= size:
                f.seek(off)
                head = f.read(16)
                read += len(head)
                if len(head) < 8:
                    break
                box, kind = struct.unpack(">I4s", head[:8])
                hlen = 8
                if box == 1:
                    box = struct.unpack(">Q", head[8:16])[0]
                    hlen = 16
                elif box == 0:
                    box = size - off
                if box < hlen or off + box > size:
                    break
                if kind == b"moof":
                    f.seek(off)
                    moof = f.read(box)
                    read += len(moof)
                    out.extend(_fragment(moof, off, segment.path, init, sync_only))
                off += box
    except OSError as e:
        name = os.path.basename(segment.path)
        raise Unreadable(f"the segment {name} can't be read: {e}") from e
    return out, read


def _fragment(moof: bytes, at: int, path: str, init: Init, sync_only: bool) -> list[Sample]:
    """The samples of a movie fragment that starts at the offset at of its
    segment."""
    out: list[Sample] = []
    for kind, body, stop in _boxes(moof, 8, len(moof)):
        if kind != b"traf":
            continue
        base = at
        dur0, size0, flags0 = init.default_duration, init.default_size, init.default_flags
        decode_time = 0
        runs = []
        for k, b, s in _boxes(moof, body, stop):
            flags = int.from_bytes(moof[b + 1 : b + 4], "big")
            if k == b"tfhd":
                p = b + 8
                if flags & 0x1:
                    base = struct.unpack(">Q", moof[p : p + 8])[0]
                    p += 8
                if flags & 0x2:
                    p += 4
                if flags & 0x8:
                    dur0 = struct.unpack(">I", moof[p : p + 4])[0]
                    p += 4
                if flags & 0x10:
                    size0 = struct.unpack(">I", moof[p : p + 4])[0]
                    p += 4
                if flags & 0x20:
                    flags0 = struct.unpack(">I", moof[p : p + 4])[0]
            elif k == b"tfdt":
                wide = moof[b] == 1
                decode_time = struct.unpack(
                    ">Q" if wide else ">I", moof[b + 4 : b + (12 if wide else 8)]
                )[0]
            elif k == b"trun":
                runs.append((b, s, flags))
        t = decode_time
        for b, _s, flags in runs:
            count = struct.unpack(">I", moof[b + 4 : b + 8])[0]
            p = b + 8
            data = 0
            if flags & 0x1:
                data = struct.unpack(">i", moof[p : p + 4])[0]
                p += 4
            first = None
            if flags & 0x4:
                first = struct.unpack(">I", moof[p : p + 4])[0]
                p += 4
            offset = base + data
            for i in range(count):
                dur, size, sflags, cto = dur0, size0, flags0, 0
                if flags & 0x100:
                    dur = struct.unpack(">I", moof[p : p + 4])[0]
                    p += 4
                if flags & 0x200:
                    size = struct.unpack(">I", moof[p : p + 4])[0]
                    p += 4
                if flags & 0x400:
                    sflags = struct.unpack(">I", moof[p : p + 4])[0]
                    p += 4
                elif i == 0 and first is not None:
                    sflags = first
                if flags & 0x800:
                    cto = struct.unpack(">i", moof[p : p + 4])[0]
                    p += 4
                sync = not sflags & _NON_SYNC
                if size > 0 and (sync or not sync_only):
                    out.append(Sample(t, t + cto, sync, path, offset, size))
                offset += size
                t += dur
    return out
