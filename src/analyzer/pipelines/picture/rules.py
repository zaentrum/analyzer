"""The rules: the picture area of a title from its samples.

Each sample is reduced (planes.py) and added to an Accumulator as it is
decoded; of a sample only counts are kept, so the memory is the same for
ten samples and for ten thousand.

A line of a side's zone is bar in a sample when it lies at the title's bar
black of that side (planes.py, read from frames spread over the title, not
from the sample's own edge): its mean within LEVEL levels of it, and no
segment more than DIM levels of luma, or CHROMA_DIM of chroma, away from it.
A side with no bar black has no bar line. A sample's picture on a side
starts at its first line that is not bar. Then, per side:

  * the edge is the outermost first line of any sample: the picture's union
    over the samples, so a black frame, a dark scene or a fade adds nothing,
    and a wider scene, a subtitle or a caption in the bar widens the area;
  * a logo is left out: a mark in a corner of the bar (not in the middle
    three fifths of the line), in at least LOGO_SHARE of the samples, at the
    same place in the same shape (nothing else is ever in that corner of
    the bar but the mark and what touches it): a mark that moves between
    corners is in neither long enough;
  * the bar's edge is where enough samples show the picture reaching it
    (SUPPORT_MIN, and SUPPORT_SHARE of those that show a picture, within a
    few lines; the samples of a stretch decoded through are no support); a
    union edge up to NEAR lines beyond it, which few samples show (ringing
    at a bright edge, a blocky frame), widens the area to take it in, and
    one further out is another picture than the one most samples show (a
    wider shot): that side keeps the full frame, as one whose edge no
    samples enough show does, and one whose whole zone is bar (a bar that
    wide is not told from a dark picture);
  * the bars of an axis (top and bottom, left and right) are kept only
    together and when their edges are near the same (within SYMMETRY of the
    axis' length, at least SYMMETRY_MIN lines): a lone bar, or a lopsided
    pair, keeps the axis whole.

The edges are rounded outwards to even pixels. Too few samples that show a
picture, and the area says nothing (reason).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .planes import CORNER, SEGMENTS, SIDES, Reduced, zone

# A bar's line: its mean within LEVEL levels (8 bits) of the title's bar
# black of its side.
LEVEL = 2.0
# Samples, and the share of those that show a picture, that must show the
# picture reaching an edge (within SUPPORT_TOLERANCE of the side's length,
# at least 2 lines) for it to be the bar's.
SUPPORT_MIN = 3
SUPPORT_SHARE = 0.03
SUPPORT_TOLERANCE = 0.004
# How far beyond the bar's edge the union may lie and still widen the area:
# NEAR of the side's length, at least NEAR_MIN lines.
NEAR = 0.015
NEAR_MIN = 8
# The bars' edges of an axis are near the same within this share of its
# length, and at least SYMMETRY_MIN lines.
SYMMETRY = 0.006
SYMMETRY_MIN = 4
# A logo: in at least this share of the samples; what touches it (its edge,
# the ringing around it) is within LOGO_MARGIN of the side's length of it
# along the side's lines (at least LOGO_MARGIN_MIN lines), and a segment
# across them.
LOGO_SHARE = 0.8
LOGO_MARGIN = 0.01
LOGO_MARGIN_MIN = 8
# Samples that must show a picture: MIN_INFORMATIVE, and INFORMATIVE_SHARE
# of the samples.
MIN_INFORMATIVE = 6
INFORMATIVE_SHARE = 0.1

_CENTER = np.uint32(sum(1 << j for j in range(CORNER, SEGMENTS - CORNER)))
_BIT = np.arange(SEGMENTS, dtype=np.uint32)
_AXES = (("top", "bottom"), ("left", "right"))


@dataclass(frozen=True)
class Box:
    """A rectangle of the coded frame, in pixels: its left and top edge,
    its width and its height."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class Decision:
    """The area the rules found, or None and why; each side's edge and why;
    how many samples there were and how many showed a picture."""

    area: Box | None
    reason: str | None
    sides: dict[str, Any] = field(default_factory=dict)
    samples: int = 0
    informative: int = 0


def _first(mask: np.ndarray, none: int) -> int:
    """The index of mask's first true element; none when there is none."""
    if not mask.size:
        return none
    i = int(np.argmax(mask))
    return i if mask[i] else none


class _Hist:
    """Per line (the zone's lines, and one more for none), how many samples'
    picture starts there: all of them (the union), and those counted for
    support; and the time of a sample whose picture starts outermost."""

    def __init__(self, z: int) -> None:
        self.all = np.zeros(z + 1, np.int64)
        self.counted = np.zeros(z + 1, np.int64)
        self.at: tuple[int, float | None] = (z, None)

    def add(self, line: int, counted: bool, t: float | None) -> None:
        self.all[line] += 1
        if counted:
            self.counted[line] += 1
        if line < self.at[0]:
            self.at = (line, t)


class _Side:
    """One side's counts over the samples."""

    def __init__(self, name: str, length: int) -> None:
        self.name, self.length = name, length
        self.z = zone(length)
        # by the whole line (strict), and by its middle (center)
        self.strict = _Hist(self.z)
        self.center = _Hist(self.z)
        # Per line and segment, how many samples hold something there.
        self.cells = np.zeros((self.z, SEGMENTS), np.int64)
        self.black: float | None = None

    def add(self, lines: Any, t: float | None, counted: bool) -> None:
        z = self.z
        self.black = lines.black
        if lines.black is None:
            # no bar black: the side's lines are all picture
            self.strict.add(0, counted, t)
            self.center.add(0, counted, t)
            return
        dim = lines.dim
        strict = (dim != 0) | (lines.dev_full > LEVEL)
        center = ((dim & _CENTER) != 0) | (lines.dev_center > LEVEL)
        fs = _first(strict, z)
        fc = _first(center, z)
        self.strict.add(fs, counted, t)
        self.center.add(fc, counted, t)
        # the cells of the lines before the picture (by the middle) only:
        # what a logo is looked for in
        if fc > 0:
            self.cells[:fc] += (dim[:fc, None] >> _BIT) & 1


class Accumulator:
    """The samples' counts, per side, for a frame of width x height."""

    def __init__(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.sides = {
            "top": _Side("top", height),
            "bottom": _Side("bottom", height),
            "left": _Side("left", width),
            "right": _Side("right", width),
        }
        self.samples = 0
        self.counted = 0
        self.informative = 0
        self.counted_informative = 0

    def add(self, r: Reduced, t: float | None = None, counted: bool = True) -> None:
        """Add a sample reduced (planes.py) of a frame this size, taken at t
        (seconds), which the report names. A sample not counted (one of a
        stretch decoded through) widens the union, and is no support of an
        edge."""
        if r.width != self.width or r.height != self.height:
            raise ValueError(
                f"a sample of {r.width}x{r.height} in a title of {self.width}x{self.height}"
            )
        self.samples += 1
        self.counted += counted
        if r.shows:
            self.informative += 1
            self.counted_informative += counted
        for name in SIDES:
            self.sides[name].add(r.sides[name], t, counted)

    def decide(self) -> Decision:
        """The picture area of the samples added, by the rules at the top of
        the file."""
        w, h = self.width, self.height
        n, informative = self.samples, self.informative
        found: dict[str, int] = {}
        bars: dict[str, int] = {}
        sides: dict[str, Any] = {}
        for name in SIDES:
            found[name], bars[name], sides[name] = _edge(
                self.sides[name], n, self.counted_informative
            )
        for a, b in _AXES:
            _pair(found, bars, sides, a, b, h if a == "top" else w)
        sides["black"] = {name: s.black for name, s in self.sides.items()}
        need = max(MIN_INFORMATIVE, math.ceil(INFORMATIVE_SHARE * max(self.counted, 1)))
        if self.counted_informative < need:
            why = (
                f"too few samples show a picture ({self.counted_informative} of "
                f"{self.counted}, {need} needed)"
            )
            return Decision(None, why, sides, n, informative)
        top, bottom = found["top"], h - found["bottom"]
        left, right = found["left"], w - found["right"]
        if bottom - top < h // 8 or right - left < w // 8:
            why = f"the picture found is too small ({right - left}x{bottom - top})"
            return Decision(None, why, sides, n, informative)
        x0, y0 = left - left % 2, top - top % 2
        x1, y1 = min(w, right + right % 2), min(h, bottom + bottom % 2)
        return Decision(Box(x0, y0, x1 - x0, y1 - y0), None, sides, n, informative)


def _edge(s: _Side, n: int, informative: int) -> tuple[int, int, dict[str, Any]]:
    """A side's edge (in lines; 0: none), the bar's edge most samples show
    (0: none), and why."""
    why: dict[str, Any] = {"bar": 0, "support": 0}
    if n == 0:
        why["note"] = "no samples"
        return 0, 0, why
    hist = s.strict
    union = _first(hist.all > 0, s.z)
    center = _first(s.center.all > 0, s.z)
    if center > union:
        margin = max(LOGO_MARGIN_MIN, math.ceil(s.length * LOGO_MARGIN))
        logo = _logo(s.cells[:center], n, margin)
        if logo is not None:
            hist, union = s.center, center
            why["logo"] = logo
    if hist.at[1] is not None:
        why["at"] = round(hist.at[1], 1)
    if union == 0:
        why["note"] = (
            "the title's edges on this side hold no black"
            if s.black is None
            else "the picture reaches the frame's edge"
        )
        return 0, 0, why
    if union >= s.z:
        why["note"] = "no sample shows a picture within the zone: kept whole"
        return 0, 0, why
    tol = max(2, round(s.length * SUPPORT_TOLERANCE))
    need = max(SUPPORT_MIN, math.ceil(SUPPORT_SHARE * informative))
    # support[d]: the samples counted whose picture starts within tol of d
    c = np.concatenate([[0], np.cumsum(hist.counted)])
    d = np.arange(s.z + 1)
    support = c[np.minimum(d + tol + 1, s.z + 1)] - c[d]
    bar = _first(support[: s.z] >= need, -1)
    why["support"] = int(support[union])
    if bar < 0:
        why["note"] = f"no edge shows in {need} samples: kept whole"
        why["edge"] = union
        return 0, 0, why
    # the line most of those samples start at
    bar = max(union, bar + int(np.argmax(hist.counted[bar : bar + tol + 1])))
    near = max(NEAR_MIN, math.ceil(s.length * NEAR))
    if bar - union > near:
        k = int(hist.all[: union + 1].sum())
        why["note"] = (
            f"{k} samples show a picture {bar - union} lines beyond the edge "
            f"{int(support[bar])} show: kept whole"
        )
        why["edge"] = union
        return 0, 0, why
    why["bar"] = union
    why["support"] = int(support[bar])
    notes = ["bar"]
    if bar > union:
        notes.append(f"widened by {bar - union} lines that few samples show")
    if "logo" in why:
        notes.append("a logo left out")
    why["note"] = ", ".join(notes)
    return union, bar, why


def _logo(cells: np.ndarray, n: int, margin: int) -> dict[str, Any] | None:
    """The logo of a bar's lines, when what is ever in them is one: the
    cells in at least LOGO_SHARE of the samples, all in the corners, and
    nothing else but what touches them (within margin lines, a segment
    across). None when it is no logo."""
    mark = cells >= LOGO_SHARE * n
    if not mark.any():
        return None
    middle = np.zeros(SEGMENTS, bool)
    middle[CORNER : SEGMENTS - CORNER] = True
    if ((cells > 0) & middle).any():
        return None
    # the mark and what touches it; nothing else
    rows = mark.any(axis=1)
    near_rows = np.convolve(rows.astype(np.int64), np.ones(2 * margin + 1, np.int64), "same") > 0
    segs = mark.any(axis=0)
    near_segs = segs.copy()
    near_segs[1:] |= segs[:-1]
    near_segs[:-1] |= segs[1:]
    if ((cells > 0) & ~(near_rows[:, None] & near_segs[None, :])).any():
        return None
    lines, used = np.flatnonzero(rows), np.flatnonzero(segs)
    return {
        "lines": [int(lines[0]), int(lines[-1]) + 1],
        "segments": [int(used[0]), int(used[-1]) + 1],
        "share": round(float(cells[mark].min()) / n, 2),
    }


def _pair(
    found: dict[str, int], bars: dict[str, int], sides: dict[str, Any], a: str, b: str, length: int
) -> None:
    """Keep an axis' bars only together and when their edges are near the
    same."""
    if found[a] == 0 and found[b] == 0:
        return
    tol = max(SYMMETRY_MIN, math.ceil(length * SYMMETRY))
    if found[a] == 0 or found[b] == 0:
        note = "a bar on one side only: the axis kept whole"
    elif abs(bars[a] - bars[b]) > tol:
        note = (
            f"the bars' edges differ by {abs(bars[a] - bars[b])} lines, more than {tol}: "
            "the axis kept whole"
        )
    else:
        return
    for s in (a, b):
        if found[s]:
            sides[s]["edge"] = found[s]
        found[s] = 0
        sides[s]["bar"] = 0
        sides[s]["note"] = note
