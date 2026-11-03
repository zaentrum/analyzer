"""A decoded frame reduced to what the rules read of it, from its own Y, U
and V planes: no colour conversion of a decoder's or a filter's is in
between, so a deep blue (RGB 0, 0, 100: luma about 7 of 255) is picture by
its chroma wherever it is decoded.

Each plane is cut into SEGMENTS segments along every row and every column,
and each segment's mean is taken, scaled to 8 bits (a 10-bit plane's is
divided by 4). A title's bar black is read first, per side, from a few
frames spread over it (edges(): the luma of each side's outermost line and
how far its chroma lies from neutral; calibrate()). Each frame's lines are
then read against that black, not against the frame's own edge: an open
matte shot whose edges are a flat dark grey is no bar where the title's
bars are black. The rules read the lines of the zone along each side (the
first ZONE of the frame's rows from the top and from the bottom, of its
columns from the left and from the right), outermost first, each as:

  dev_full    how far the line's mean lies from the title's bar black
              (levels)
  dev_center  the same of its middle three fifths, which a logo in a corner
              leaves alone
  dim         the segments that hold anything: their luma more than DIM
              levels from that black, or their chroma more than CHROMA_DIM
              from neutral (a bit a segment)

A side whose edges hold no black (their luma above BLACK_MAX, or their
chroma more than BLACK_CHROMA from neutral, in most of the frames read) has
no bar black: its lines are all picture. A frame shows a picture when a
segment anywhere is CONTENT levels from the darkest bar black, or
CHROMA_LIT from neutral. What is kept of a frame is its zones' lines, a few
bytes each.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np

SEGMENTS = 32
# The share of a frame's height (width) along each side the rules read: a
# 9:16 picture pillarboxed in a 16:9 frame leaves 34 % of it to each side.
ZONE = 0.36
# Levels (8 bits) of luma from the bar black, and of chroma from neutral,
# at which a segment holds anything.
DIM = 3.0
CHROMA_DIM = 4.0
# Levels of luma from the black, or of chroma from neutral, at which a frame
# shows a picture.
CONTENT = 10.0
CHROMA_LIT = 8.0
# A bar black: its luma at most BLACK_MAX, its chroma within BLACK_CHROMA of
# neutral.
BLACK_MAX = 40.0
BLACK_CHROMA = 8.0
# The corners of a line, which a logo may take: its first and its last
# CORNER segments. The rest is its middle.
CORNER = SEGMENTS // 5
NEUTRAL = 128.0

SIDES = ("top", "bottom", "left", "right")
_BITS = (np.uint64(1) << np.arange(SEGMENTS, dtype=np.uint64)).astype(np.uint64)

# A title's bar black per side; None for a side whose edges hold none.
Blacks = dict[str, float | None]


@dataclass
class SideLines:
    """The zone's lines along one side, outermost first, read against the
    title's bar black of that side (black; None: the side has none, and
    the arrays say nothing)."""

    black: float | None
    dev_full: np.ndarray  # float32 (Z,)
    dev_center: np.ndarray  # float32 (Z,)
    dim: np.ndarray  # uint32 (Z,): bit j set when segment j holds anything


@dataclass
class Reduced:
    """A frame as the rules read it: its four sides, and whether it shows a
    picture at all (a black frame, or one faded out, does not)."""

    width: int
    height: int
    sides: dict[str, SideLines]
    shows: bool


def zone(length: int) -> int:
    """How many lines of a side of length lines the rules read."""
    return max(1, min(length // 2, math.ceil(length * ZONE)))


def _edges(n: int, k: int) -> np.ndarray:
    return (np.arange(k + 1, dtype=np.int64) * n) // k


def _segments(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The segment means of a 2-D plane: (rows, SEGMENTS) along each row,
    (columns, SEGMENTS) along each column."""
    h, w = plane.shape
    kx, ky = min(SEGMENTS, w), min(SEGMENTS, h)
    ex, ey = _edges(w, kx), _edges(h, ky)
    rows = np.add.reduceat(plane, ex[:-1], axis=1, dtype=np.uint32) / np.diff(ex)
    cols = (np.add.reduceat(plane, ey[:-1], axis=0, dtype=np.uint32) / np.diff(ey)[:, None]).T
    if kx < SEGMENTS:
        rows = np.repeat(rows, -(-SEGMENTS // kx), axis=1)[:, :SEGMENTS]
    if ky < SEGMENTS:
        cols = np.repeat(cols, -(-SEGMENTS // ky), axis=1)[:, :SEGMENTS]
    return rows.astype(np.float32), cols.astype(np.float32)


def _line_segments(line: np.ndarray) -> np.ndarray:
    """The segment means of one line (a 1-D array)."""
    n = line.shape[0]
    k = min(SEGMENTS, n)
    e = _edges(n, k)
    return np.add.reduceat(line, e[:-1], dtype=np.uint32) / np.diff(e)


def _spread(a: np.ndarray, n: int) -> np.ndarray:
    """A chroma plane's lines (a, its first axis) for each of the n lines of
    the luma plane."""
    return a[np.minimum(np.arange(n) * a.shape[0] // n, a.shape[0] - 1)]


def edges(
    y: np.ndarray, u: np.ndarray, v: np.ndarray, depth: int
) -> dict[str, tuple[float, float]]:
    """Each side's outermost line: its luma, and how far its chroma lies
    from neutral (the farthest of its segments'), scaled to 8 bits."""
    scale = 1.0 / (1 << max(0, depth - 8))
    out: dict[str, tuple[float, float]] = {}
    for name, ly, lu, lv in (
        ("top", y[0], u[0], v[0]),
        ("bottom", y[-1], u[-1], v[-1]),
        ("left", y[:, 0], u[:, 0], v[:, 0]),
        ("right", y[:, -1], u[:, -1], v[:, -1]),
    ):
        luma = float(_line_segments(ly).mean()) * scale
        cu = np.abs(_line_segments(lu) * scale - NEUTRAL)
        cv = np.abs(_line_segments(lv) * scale - NEUTRAL)
        out[name] = (luma, float(max(cu.max(), cv.max())))
    return out


def calibrate(seen: list[dict[str, tuple[float, float]]]) -> Blacks:
    """The title's bar black per side, of the edges of frames spread over it
    (edges()): the median of their outermost lines' luma, when that and the
    median of their chroma are a black's; else None."""
    out: Blacks = {}
    for name in SIDES:
        luma = [e[name][0] for e in seen]
        chroma = [e[name][1] for e in seen]
        if not luma:
            out[name] = None
            continue
        b, c = statistics.median(luma), statistics.median(chroma)
        out[name] = round(b, 2) if b <= BLACK_MAX and c <= BLACK_CHROMA else None
    return out


def reduce_planes(
    y: np.ndarray, u: np.ndarray, v: np.ndarray, depth: int, blacks: Blacks
) -> Reduced:
    """A frame's Y, U and V planes (2-D arrays; the chroma ones subsampled or
    not) of a format of depth bits, reduced against the title's bar blacks
    (see the top of the file)."""
    h, w = y.shape
    scale = np.float32(1.0 / (1 << max(0, depth - 8)))
    yr, yc = (a * scale for a in _segments(y))
    ur, uc = (a * scale for a in _segments(u))
    vr, vc = (a * scale for a in _segments(v))
    ur, vr = _spread(ur, h), _spread(vr, h)
    uc, vc = _spread(uc, w), _spread(vc, w)

    sides: dict[str, SideLines] = {}
    zr, zc = zone(h), zone(w)
    for name, ly, lu, lv, lines in (
        ("top", yr, ur, vr, np.arange(zr)),
        ("bottom", yr, ur, vr, h - 1 - np.arange(zr)),
        ("left", yc, uc, vc, np.arange(zc)),
        ("right", yc, uc, vc, w - 1 - np.arange(zc)),
    ):
        black = blacks.get(name)
        z = len(lines)
        if black is None:
            empty = np.zeros(z, np.float32)
            sides[name] = SideLines(None, empty, empty, np.zeros(z, np.uint32))
            continue
        s, su, sv = ly[lines], lu[lines], lv[lines]
        d = s - np.float32(black)
        chroma = np.maximum(np.abs(su - NEUTRAL), np.abs(sv - NEUTRAL))
        dim = (np.abs(d) > DIM) | (chroma > CHROMA_DIM)
        sides[name] = SideLines(
            black=black,
            dev_full=np.abs(d.mean(axis=1)).astype(np.float32),
            dev_center=np.abs(d[:, CORNER : SEGMENTS - CORNER].mean(axis=1)).astype(np.float32),
            dim=(dim.astype(np.uint64) @ _BITS).astype(np.uint32),
        )
    known = [b for b in blacks.values() if b is not None]
    darkest = min(known) if known else float(min(yr[0].mean(), yr[-1].mean()))
    lit = np.maximum(np.abs(ur - NEUTRAL), np.abs(vr - NEUTRAL))
    shows = bool((np.abs(yr - darkest) > CONTENT).any() or (lit > CHROMA_LIT).any())
    return Reduced(w, h, sides, shows)


def frame_planes(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """A decoded PyAV frame's Y, U and V planes as arrays, and their depth.
    A frame of another format than planar YUV (an RGB one, a gray one, NV12)
    is converted to yuv444p first."""
    fmt = frame.format
    if not (fmt.name.startswith("yuv") and len(frame.planes) >= 3 and not fmt.is_rgb):
        frame = frame.reformat(format="yuv444p")
        fmt = frame.format
    depth = fmt.components[0].bits
    wide = depth > 8
    dtype = np.dtype("<u2") if wide else np.dtype(np.uint8)
    out = []
    for i in range(3):
        p = frame.planes[i]
        per_line = p.line_size // dtype.itemsize
        a = np.frombuffer(p, dtype=dtype, count=per_line * p.height)
        out.append(a.reshape(p.height, per_line)[:, : p.width])
    return out[0], out[1], out[2], depth


def frame_edges(frame) -> dict[str, tuple[float, float]]:
    """A decoded PyAV frame's edges (edges())."""
    y, u, v, depth = frame_planes(frame)
    return edges(y, u, v, depth)


def reduce_frame(frame, blacks: Blacks) -> Reduced:
    """A decoded PyAV frame, reduced against the title's bar blacks."""
    y, u, v, depth = frame_planes(frame)
    return reduce_planes(y, u, v, depth, blacks)
