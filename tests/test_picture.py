"""The picture-area measurement (pipelines/picture): its rules on frames
drawn here as Y, U and V planes, the sampling's plans, a package's playlist
and fragments, and the measurement's own process. The cases a title has to
get right, as small videos ffmpeg makes, are in test_picture_cases.py."""

from __future__ import annotations

import itertools
import math
import struct
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from analyzer.pipelines import picture
from analyzer.pipelines.picture import Box, child, fmp4, planes, rules, sources

W, H = 320, 180
# a 2.39:1 picture letterboxed in this 16:9 frame
TOP, PIC = 24, 132
LETTERBOX = Box(0, TOP, W, PIC)
FULL = Box(0, 0, W, H)


# ------------------------------------------------------------ drawn frames
def yuv(
    *,
    top: int = TOP,
    height: int = PIC,
    left: int = 0,
    width: int = W,
    black: float = 16,
    level: float | None = None,
    seed: int = 0,
    chroma: tuple[float, float] | None = None,
    noise: float = 0.0,
    marks: tuple[tuple[int, int, int, int, float], ...] = (),
    fill: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Y, U and V planes (4:2:0, 8 bits) of a picture over left, top,
    width, height, in bars at black: a texture of 40 to 219 (a flat level),
    its chroma neutral or (u, v); grain of noise levels over everything;
    marks (x, y, w, h, level) drawn on top; fill (y, u, v) the whole frame
    at one colour first."""
    rng = np.random.default_rng(seed)
    y = np.full((H, W), black, np.float64)
    u = np.full((H // 2, W // 2), 128.0)
    v = np.full((H // 2, W // 2), 128.0)
    if fill is not None:
        y[:], u[:], v[:] = fill
    if height and width:
        if level is None:
            rows, cols = np.arange(height)[:, None], np.arange(width)[None, :]
            y[top : top + height, left : left + width] = (
                40 + (rows * 13 + cols * 7 + seed * 31) % 180
            )
        else:
            y[top : top + height, left : left + width] = level
        if chroma is not None:
            cs = np.s_[top // 2 : (top + height) // 2, left // 2 : (left + width) // 2]
            u[cs], v[cs] = chroma
    for x, yy, w, h, lv in marks:
        y[yy : yy + h, x : x + w] = lv
    if noise:
        y += rng.normal(0, noise, y.shape)
        u += rng.normal(0, noise / 2, u.shape)
        v += rng.normal(0, noise / 2, v.shape)
    return tuple(np.clip(np.rint(p), 0, 255).astype(np.uint8) for p in (y, u, v))


# the title's bar black of a frame whose bars are at 16, each side
BLACKS: planes.Blacks = dict.fromkeys(planes.SIDES, 16.0)


def reduced(blacks: planes.Blacks | None = None, **kw) -> planes.Reduced:
    return planes.reduce_planes(*yuv(**kw), 8, BLACKS if blacks is None else blacks)


def calibrated(n: int = 24, **kw) -> planes.Blacks:
    """The bar blacks calibrate() reads off n frames drawn so."""
    return planes.calibrate([planes.edges(*yuv(seed=i, **kw), 8) for i in range(n)])


def decide(*samples: planes.Reduced | tuple[planes.Reduced, bool]) -> rules.Decision:
    acc = rules.Accumulator(W, H)
    for i, s in enumerate(samples):
        r, counted = s if isinstance(s, tuple) else (s, True)
        acc.add(r, float(i), counted)
    return acc.decide()


def letterboxed(n: int = 40, blacks: planes.Blacks | None = None, **kw) -> list[planes.Reduced]:
    return [reduced(blacks, seed=i, **kw) for i in range(n)]


def test_a_letterbox_is_its_picture() -> None:
    d = decide(*letterboxed())
    assert d.area == LETTERBOX
    assert d.sides["top"]["bar"] == TOP and d.sides["bottom"]["bar"] == H - TOP - PIC


def test_a_full_frame_is_the_frame() -> None:
    blacks = calibrated(top=0, height=H)
    assert blacks == dict.fromkeys(planes.SIDES)
    d = decide(*letterboxed(top=0, height=H, blacks=blacks))
    assert d.area == FULL
    assert d.sides["top"]["note"] == "the title's edges on this side hold no black"


def test_black_dark_and_faded_frames_add_nothing() -> None:
    black = [reduced(height=0, seed=i) for i in range(10)]
    faded = [reduced(height=0, black=17, seed=i) for i in range(10)]
    dark = [reduced(level=24, seed=i) for i in range(10)]
    assert decide(*letterboxed(), *black, *faded, *dark).area == LETTERBOX
    # A frame faded to a grey over its bars too is no black there: flat as
    # it is, it is told from the bars' black, as an open matte shot's grey
    # sky is.
    grey = [reduced(height=0, fill=(30, 128, 128), seed=i) for i in range(10)]
    assert decide(*letterboxed(), *grey).area == FULL


def test_a_wider_shot_few_samples_show_keeps_the_axis_whole() -> None:
    # A 1.78 shot of a few seconds in a 2.39 film, sampled twice: the
    # picture is the union, and reaches the frame's edge.
    d = decide(*letterboxed(), *letterboxed(2, top=0, height=H))
    assert d.area == FULL
    assert d.sides["top"]["note"] == "the picture reaches the frame's edge"
    # A 1.85 shot sampled twice: a side whose edge few samples show further
    # out than most show theirs keeps the full frame (more of it may be in
    # what was not sampled).
    d = decide(*letterboxed(), *letterboxed(2, top=8, height=164))
    assert d.area == FULL
    assert "a picture 16 lines beyond the edge 40 show" in d.sides["top"]["note"]


def test_a_wider_shot_many_samples_show_is_the_area() -> None:
    d = decide(*letterboxed(), *letterboxed(8, top=8, height=164))
    assert d.area == Box(0, 8, W, 164)


def test_what_few_samples_show_near_the_edge_widens_it() -> None:
    # Ringing at a bright edge, a blocky frame: a few lines out of the bar
    # in one sample widen the area to take them in.
    ringing = reduced(seed=99, marks=((0, 18, W, 6, 22), (0, 156, W, 6, 22)))
    d = decide(*letterboxed(), ringing)
    assert d.area == Box(0, 18, W, 144)
    assert "widened by" in d.sides["top"]["note"] and "widened by" in d.sides["bottom"]["note"]


def test_a_night_sky_a_few_levels_above_black_is_picture() -> None:
    # A night film: the sky of its picture four levels above the bars'
    # black, a bright skyline low in the frame, never a brighter sky.
    night = [reduced(level=20, seed=i, marks=((0, 110, W, 46, 160),)) for i in range(40)]
    assert decide(*night).area == LETTERBOX


def test_a_line_within_two_levels_of_black_is_bar() -> None:
    # What is within LEVEL of the bars' black in every sample is black:
    # nobody sees it.
    dark = ((0, 24, W, 36, 17), (0, 120, W, 36, 17))
    assert decide(*letterboxed(marks=dark)).area == Box(0, 60, W, 60)
    # Lopsided, as a picture dark at its top only would make it: the axis
    # is kept whole.
    sky = [reduced(level=17, seed=i, marks=((0, 110, W, 46, 160),)) for i in range(40)]
    assert decide(*sky).area == FULL


def test_a_deep_blue_open_matte_is_picture_by_its_chroma() -> None:
    # A shot whose extra rows are a deep blue sea (RGB 0, 0, 100 in BT.709:
    # Y 22, U 172, V 124): its chroma is what tells it from the bars, read
    # from the U and V planes themselves. Even at a luma one level from
    # black.
    for luma in (22, 17):
        blue = [
            reduced(top=0, height=H, seed=i, fill=(luma, 172, 124), marks=((0, TOP, W, PIC, 120),))
            for i in range(8)
        ]
        assert decide(*letterboxed(), *blue).area == FULL, luma


def test_ten_bit_planes_read_as_eight() -> None:
    acc = rules.Accumulator(W, H)
    for i in range(40):
        y, u, v = (p.astype(np.uint16) * 4 for p in yuv(seed=i))
        acc.add(planes.reduce_planes(y, u, v, 10, BLACKS), float(i))
    assert acc.decide().area == LETTERBOX


def test_grain_in_the_bars_is_no_picture() -> None:
    assert decide(*letterboxed(noise=1.5)).area == LETTERBOX


def test_a_static_logo_in_a_corner_of_a_bar_is_left_out() -> None:
    logo = ((280, 6, 28, 10, 230),)
    d = decide(*letterboxed(marks=logo))
    assert d.area == LETTERBOX
    assert d.sides["top"]["logo"]["segments"] == [28, 31] and "logo" in d.sides["top"]["note"]


def test_a_mark_in_the_middle_of_a_bar_is_no_logo() -> None:
    d = decide(*letterboxed(marks=((150, 6, 28, 10, 230),)))
    assert d.area == FULL


def test_a_mark_in_fewer_than_most_samples_is_no_logo() -> None:
    logo = ((280, 6, 28, 10, 230),)
    d = decide(*letterboxed(28, marks=logo), *letterboxed(12))
    assert d.area == FULL


def test_a_mark_that_moves_between_corners_is_no_logo() -> None:
    left, right = ((12, 6, 28, 10, 230),), ((280, 6, 28, 10, 230),)
    d = decide(*letterboxed(20, marks=left), *letterboxed(20, marks=right))
    assert d.area == FULL


def test_a_corner_with_more_than_its_mark_is_no_logo() -> None:
    logo = ((280, 6, 28, 10, 230),)
    other = ((260, 2, 8, 4, 230),)
    d = decide(*letterboxed(36, marks=logo), *letterboxed(4, marks=logo + other))
    assert d.area == FULL


def test_subtitles_in_a_bar_keep_the_axis_whole() -> None:
    glyphs = tuple((120 + 12 * g, 162, 8, 10, 235) for g in range(8))
    d = decide(*letterboxed(24, marks=glyphs), *letterboxed(16))
    assert d.area == FULL
    assert "the axis kept whole" in d.sides["bottom"]["note"]


def test_subtitles_only_a_stretch_shows_keep_the_axis_whole() -> None:
    # The samples of a stretch decoded through widen the union, and are no
    # support of an edge: a subtitle between keyframes keeps the axis whole.
    glyphs = tuple((120 + 12 * g, 162, 8, 10, 235) for g in range(8))
    stretch = [(r, False) for r in letterboxed(8, marks=glyphs)]
    assert decide(*letterboxed(), *stretch).area == FULL


def test_bars_of_another_height_keep_the_axis_whole() -> None:
    assert decide(*letterboxed(top=24, height=116)).area == FULL
    # a row or two is ringing, an encoder's rounding: within SYMMETRY
    assert decide(*letterboxed(top=24, height=130)).area == Box(0, 24, W, 130)


def test_a_bar_on_one_side_only_keeps_the_axis_whole() -> None:
    d = decide(*letterboxed(top=24, height=H - 24))
    assert d.area == FULL
    assert d.sides["top"]["note"] == "a bar on one side only: the axis kept whole"


def test_a_pillarbox_and_a_windowbox() -> None:
    assert decide(*letterboxed(top=0, height=H, left=40, width=240)).area == Box(40, 0, 240, H)
    assert decide(*letterboxed(top=18, height=144, left=64, width=192)).area == Box(
        64, 18, 192, 144
    )


def test_edges_round_outwards_to_even_pixels() -> None:
    assert decide(*letterboxed(top=23, height=133)).area == Box(0, 22, W, 134)


def test_too_few_samples_with_a_picture_say_so() -> None:
    d = decide(*letterboxed(3), *[reduced(height=0, seed=i) for i in range(30)])
    assert d.area is None and d.reason.startswith("too few samples show a picture")


def test_an_edge_that_is_never_black_has_no_bar() -> None:
    blacks = calibrated(black=60)
    assert blacks["top"] is None and blacks["bottom"] is None
    d = decide(*letterboxed(black=60, blacks=blacks))
    assert d.area == FULL
    assert d.sides["top"]["note"] == "the title's edges on this side hold no black"


def test_the_bar_black_is_the_titles() -> None:
    # Most frames' edges are the bars' black: the median of the edges of the
    # frames read; a coloured edge is no black.
    seen = [planes.edges(*yuv(seed=i), 8) for i in range(20)]
    seen += [planes.edges(*yuv(seed=i, top=0, height=H), 8) for i in range(4)]
    assert planes.calibrate(seen)["top"] == 16.0
    blue = [planes.edges(*yuv(seed=i, fill=(22, 172, 124), height=0), 8) for i in range(9)]
    assert planes.calibrate(blue)["top"] is None


def test_a_flat_grey_edge_of_a_wider_shot_is_picture() -> None:
    # An open-matte shot whose top and bottom rows are a flat dark grey (RGB
    # 18: luma 31) in a film whose bars are black (16): those rows lie 15
    # levels from the title's bar black, picture, though they are flat.
    grey = [
        reduced(seed=i, top=0, height=H, fill=(31, 128, 128), marks=((0, TOP, W, PIC, 120),))
        for i in range(8)
    ]
    assert decide(*letterboxed(), *grey).area == FULL
    pillar = [
        reduced(seed=i, top=0, height=H, fill=(31, 128, 128), marks=((40, 0, 240, H, 120),))
        for i in range(8)
    ]
    assert decide(*letterboxed(top=0, height=H, left=40, width=240), *pillar).area == FULL


def test_a_frame_read_in_two_steps_is_read_as_at_once() -> None:
    # Its segment means first, the bar black after: what the calibration's
    # frames are kept as, to be samples once the black is known.
    for kw in ({}, {"noise": 3.0, "seed": 4}, {"chroma": (90, 160), "seed": 2}):
        y, u, v = yuv(**kw)
        segs = planes.segments(y, u, v, 8)
        two, one = planes.edges_of(segs), planes.edges(y, u, v, 8)
        for name in planes.SIDES:
            assert two[name] == pytest.approx(one[name], abs=1e-4)
        once, twice = planes.reduce_planes(y, u, v, 8, BLACKS), planes.reduce_segs(segs, BLACKS)
        assert once.shows == twice.shows
        for name in planes.SIDES:
            a, b = once.sides[name], twice.sides[name]
            assert a.black == b.black and (a.dim == b.dim).all()
            assert (a.dev_full == b.dev_full).all() and (a.dev_center == b.dev_center).all()


def at_edges(*samples: planes.Reduced | tuple[planes.Reduced, bool]) -> list[bool]:
    """Whether the area is the whole frame for good, after each sample."""
    acc, out = rules.Accumulator(W, H), []
    for i, s in enumerate(samples):
        r, counted = s if isinstance(s, tuple) else (s, True)
        acc.add(r, float(i), counted)
        out.append(acc.at_edges())
    return out


def test_a_picture_at_every_edge_decides_the_whole_frame() -> None:
    # The sixth sample that shows a picture decides it, and the area is the
    # whole frame then and after.
    full = letterboxed(10, top=0, height=H)
    assert at_edges(*full) == [False] * 5 + [True] * 5
    assert decide(*full[:6]).area == FULL


def test_a_bar_anywhere_never_decides_early() -> None:
    assert not any(at_edges(*letterboxed()))
    assert not any(at_edges(*letterboxed(20, left=40, width=W - 80, top=0, height=H)))
    # the top's picture at the frame's edge, a bar at the bottom: the axis
    # is kept whole in the end, but not before every sample is in
    high = letterboxed(20, top=0, height=H - 24)
    assert not any(at_edges(*high)) and decide(*high).area == FULL


def test_a_mark_at_a_corner_of_the_edge_decides_nothing() -> None:
    # A logo in a corner of the outermost line is picture by the whole line,
    # not by its middle: the logo rule may still leave it out.
    marks = ((4, 0, 26, 10, 235),)
    assert not any(at_edges(*letterboxed(20, top=10, height=H - 20, marks=marks)))


def test_a_wider_shot_after_the_bars_decides_when_it_comes() -> None:
    samples = [*letterboxed(30), *letterboxed(2, top=0, height=H)]
    assert at_edges(*samples) == [False] * 30 + [True, True]
    assert decide(*samples).area == FULL


def test_samples_of_a_stretch_widen_but_are_not_enough_alone() -> None:
    # A stretch's samples are no support: enough counted samples that show
    # a picture are needed besides.
    stretch = [(r, False) for r in letterboxed(10, top=0, height=H)]
    assert not any(at_edges(*stretch))
    assert at_edges(*stretch, *letterboxed(6, top=0, height=H))[-1]


def test_the_counts_take_the_same_memory_for_any_number_of_samples() -> None:
    big_w, big_h = 3840, 2160
    y = np.full((big_h, big_w), 16, np.uint16) * 4
    y[276:1884] = 400
    u = np.full((big_h // 2, big_w // 2), 512, np.uint16)
    r = planes.reduce_planes(y, u, u.copy(), 10, BLACKS)
    acc = rules.Accumulator(big_w, big_h)
    acc.add(r, 0.0)
    tracemalloc.start()
    for i in range(2000):
        acc.add(r, float(i))
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 4 << 20
    assert acc.decide().area == Box(0, 276, big_w, 1608)


# ----------------------------------------------------------------- sampling
def test_keyframes_are_picked_no_further_apart_than_the_gap() -> None:
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 9.0, 9.4, 30.0]
    picks = sources.cover(times, 2.9)
    # from each pick the latest within the gap, else the next one there is
    assert [times[i] for i in picks] == [0.0, 2.5, 3.0, 9.0, 9.4, 30.0]
    every = [k * 0.5 for k in range(100)]
    got = [every[i] for i in sources.cover(every, 2.9)]
    assert max(b - a for a, b in itertools.pairwise(got)) == 2.5


def test_an_originals_keyframes_are_all_taken_but_those_too_close() -> None:
    times = [0.0, 0.04, 0.5, 1.0, 1.2, 1.6, 9.0]
    assert [times[i] for i in sources.thin(times, 0.5)] == [0.0, 0.5, 1.0, 1.6, 9.0]


class _Collect:
    """A source's _take as a list of the times taken."""

    def __init__(self) -> None:
        self.taken: list[float] = []

    def _take(self, frame, t, sink, counted) -> bool:
        if t in self.taken:
            return False
        self.taken.append(t)
        return True


def test_a_run_decoded_through_leaves_no_two_samples_further_apart_than_its_step() -> None:
    # frames every 0.12 s (the reference frames of a group of pictures) from
    # a keyframe at 10 s to the next at 22.5 s
    src = _Collect()
    taker = sources._Taker(src, None, True)
    t = 10.0
    while t < 22.5:
        taker.frame(round(t, 2), None, 2.9)
        t += 0.12
    taker.close_at(22.5, 2.9)
    got = [*src.taken, 22.5]
    assert got[0] == 10.0
    assert max(b - a for a, b in itertools.pairwise(got)) <= 2.9
    assert len(src.taken) == 5


def test_the_largest_gap_is_between_the_frames_decoded_and_the_ends() -> None:
    sm = sources.Sampling(duration_s=100.0, times=[0.0, 2.0, 4.5, 40.0, 41.0, 43.0, 99.5])
    assert sm.largest_gap() == (56.5, 43.0)
    sm = sources.Sampling(duration_s=100.0, times=[5.0, 7.0, 9.0])
    assert sm.largest_gap() == (91.0, 9.0)
    sm = sources.Sampling(duration_s=100.0, times=[1.0, 3.0, 60.0])
    assert sm.largest_gap() == (57.0, 3.0)
    assert sources.Sampling(duration_s=10.0, times=[1.0, 3.0]).largest_gap() == (7.0, 3.0)


def test_the_gaps_run_from_the_first_frame_decoded_exactly_to_the_videos_last() -> None:
    # The picture starts at 7 s (the frames before lack their keyframe) and
    # ends at 399.5 s, its sound running on to 405 s.
    times = [7.0 + 2.5 * k for k in range(158)]
    sm = sources.Sampling(duration_s=405.0, times=times)
    assert sm.largest_gap() == (7.0, 0.0)
    sm.video_start_s = 7.0
    assert sm.largest_gap() == pytest.approx((5.5, 399.5))
    sm.video_end_s = 399.6
    assert sm.max_gap() == pytest.approx(2.5)
    # a frame before the start (reordered before the first keyframe) is one
    sm.times.insert(0, 6.9)
    assert sm.spans()[0] == (6.9, 6.9)


def test_the_gaps_left_are_decoded_frame_by_frame_up_to_the_cap() -> None:
    keys = [10.0 * k for k in range(41)]

    def key_before(t: float) -> float | None:
        before = [k for k in keys if k <= t + 1e-6]
        return before[-1] if before else None

    def runs(times: list[float]) -> list[tuple[str, float, float]]:
        return sources.to_close(sources.Sampling(duration_s=400.0, times=times), key_before)

    dense = [2.5 * k for k in range(161)]
    assert runs(dense) == []
    # the tail first (it may tell the title ends earlier), then the gaps
    # between, the head last
    times = [t for t in dense if not 101 < t < 108 and 7 <= t <= 390]
    assert runs(times) == [("tail", 390.0, math.inf), ("gap", 100.0, 110.0), ("head", 0.0, 7.5)]
    # more gaps than CLOSE_MAX_GAPS, or one whose run from its keyframe is
    # longer than CLOSE_MAX_SPAN_S, and none are decoded: it stays in doubt
    cuts = [80 * k + 11 for k in range(sources.CLOSE_MAX_GAPS + 1)]
    assert runs([t for t in dense if not any(a < t < a + 5 for a in cuts)]) == []
    assert runs([t for t in dense if not any(a < t < a + 5 for a in cuts[1:])]) == [
        ("gap", 90.0, 97.5), ("gap", 170.0, 177.5), ("gap", 250.0, 257.5), ("gap", 330.0, 337.5),
    ]  # fmt: skip
    assert runs([t for t in dense if not 150 < t < 175]) == [("gap", 150.0, 175.0)]
    assert runs([t for t in dense if not 150 < t < 182]) == []
    nothing_known = sources.to_close(sources.Sampling(duration_s=400.0, times=[50.0, 60.0]),
                                     lambda t: None)  # fmt: skip
    assert nothing_known == []


def test_a_measurement_in_doubt_says_why() -> None:
    dense = [k * 2.5 for k in range(41)]
    assert child.doubts(sources.Sampling(duration_s=100.0, times=dense)) == ([], [])
    codes, why = child.doubts(
        sources.Sampling(duration_s=100.0, times=[k * 6.0 for k in range(17)], missed=2,
                         duplicates=1, stopped="budget", bytes_read=300 << 20)
    )  # fmt: skip
    assert codes == ["budget", "gap", "missed", "duplicates"]
    assert why[1] == "its samples are up to 6.0 s apart (from 0.0 s), more than 3 s"


def test_the_picks_go_coarse_to_fine_each_pass_in_order() -> None:
    picks = list(range(40))
    done: set[int] = set()
    got = list(sources.passes(picks, (16, 8, 4), done))
    assert got == [[0, 16, 32], [8, 24], [4, 12, 20, 28, 36]]
    assert [len(p) for p in sources.passes(picks, (2, 1), done)] == [10, 20]


def test_the_stretches_spread_over_the_title() -> None:
    s = sources.stretches(7200)
    assert len(s) == sources.STRETCHES
    assert all(abs((b - a) - 12.0) < 1e-9 for a, b in s)
    assert s[0][0] == pytest.approx(300 - 6) and s[-1][1] == pytest.approx(6900 + 6)
    assert all(b - a == sources.STRETCH_MIN_S for a, b in sources.stretches(600))


def test_the_budget_follows_the_length(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sources.budget_bytes(7200) == 2 * sources.READ_MB_PER_HOUR << 20
    assert sources.budget_bytes(60) == sources.READ_MIN_MB << 20
    monkeypatch.setenv("PICTURE_READ_MB_PER_HOUR", "256")
    assert sources.budget_bytes(7200) == 512 << 20


# ------------------------------------------------------------------ package
def test_a_playlist_names_its_segments_in_its_folder(tmp_path: Path) -> None:
    (tmp_path / "playlist.m3u8").write_text(
        '#EXTM3U\n#EXT-X-MAP:URI="init.mp4"\n#EXTINF:6.000,\nseg-00000.m4s\n'
        "#EXTINF:4.500,\nseg-00001.m4s\n#EXT-X-ENDLIST\n"
    )
    init, segs = fmp4.read_playlist(str(tmp_path / "playlist.m3u8"))
    assert init == str(tmp_path / "init.mp4")
    assert [(s.start, s.duration, Path(s.path).name) for s in segs] == [
        (0.0, 6.0, "seg-00000.m4s"),
        (6.0, 4.5, "seg-00001.m4s"),
    ]


@pytest.mark.parametrize(
    "line",
    ["#EXT-X-BYTERANGE:1000@0\nall.mp4", "../other/seg.m4s", "https://cdn.test/seg.m4s"],
)
def test_a_playlist_of_ranges_or_of_other_folders_is_not_read(tmp_path: Path, line: str) -> None:
    (tmp_path / "p.m3u8").write_text(f'#EXTM3U\n#EXT-X-MAP:URI="init.mp4"\n#EXTINF:6,\n{line}\n')
    with pytest.raises(fmp4.Unreadable):
        fmp4.read_playlist(str(tmp_path / "p.m3u8"))


def box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), kind) + body


def full(kind: bytes, flags: int, body: bytes, version: int = 0) -> bytes:
    return box(kind, struct.pack(">I", (version << 24) | flags) + body)


def test_an_init_segment_says_its_decoder_its_frame_and_its_turn() -> None:
    hvcc = b"\x01" + b"\x00" * 22
    entry = b"\x00" * 24 + struct.pack(">HH", 1920, 1080) + b"\x00" * 50
    entry += box(b"hvcC", hvcc) + box(b"pasp", struct.pack(">II", 4, 3))
    stsd = full(b"stsd", 0, struct.pack(">I", 1) + box(b"hvc1", entry))
    # a 90 degree turn: a=0, b=1, c=-1, d=0 (16.16)
    matrix = struct.pack(">9i", 0, 1 << 16, 0, -(1 << 16), 0, 0, 0, 0, 1 << 30)
    tkhd = full(b"tkhd", 3, b"\x00" * 20 + b"\x00" * 16 + matrix + struct.pack(">II", 0, 0))
    mdhd = full(b"mdhd", 0, b"\x00" * 8 + struct.pack(">I", 24000) + b"\x00" * 8)
    trak = box(b"trak", tkhd + box(b"mdia", mdhd + box(b"minf", box(b"stbl", stsd))))
    trex = full(b"trex", 0, struct.pack(">IIIII", 1, 1, 1001, 0, 0x00010000))
    init = fmp4.parse_init(box(b"moov", trak + box(b"mvex", trex)))
    assert (init.codec, init.width, init.height, init.timescale) == ("hevc", 1920, 1080, 24000)
    assert init.extradata == hvcc and init.sar == (4, 3) and init.rotation == 90
    assert (init.default_duration, init.default_flags) == (1001, 0x00010000)


def test_a_fragment_lists_its_sync_samples_by_their_bytes(tmp_path: Path) -> None:
    # Three samples of 100, 40 and 60 bytes after the moof; the first is a
    # sync sample (first_sample_flags), the others take the default flags
    # of the track (non-sync), the third has its own flags (sync).
    init = fmp4.Init("h264", b"", 640, 360, 1000, default_duration=500, default_flags=0x00010000)
    tfhd = full(b"tfhd", 0x020000, struct.pack(">I", 1))  # default-base-is-moof
    tfdt = full(b"tfdt", 0, struct.pack(">I", 6000))
    trun_body = struct.pack(">IiI", 3, 0, 0)  # count, data offset (patched), first flags
    trun_body += (
        struct.pack(">II", 100, 0) + struct.pack(">II", 40, 0x00010000) + struct.pack(">II", 60, 0)
    )
    for _ in range(2):
        trun = full(b"trun", 0x1 | 0x4 | 0x200 | 0x400, trun_body)
        moof = box(
            b"moof", full(b"mfhd", 0, struct.pack(">I", 1)) + box(b"traf", tfhd + tfdt + trun)
        )
        trun_body = struct.pack(">IiI", 3, len(moof) + 8, 0) + trun_body[12:]
    seg = tmp_path / "seg.m4s"
    seg.write_bytes(moof + box(b"mdat", b"\x00" * 200))
    got, read = fmp4.samples(fmp4.Segment(6.0, 1.5, str(seg)), init, sync_only=False)
    assert [(s.pts, s.sync, s.offset, s.size) for s in got] == [
        (6000, True, len(moof) + 8, 100),
        (6500, False, len(moof) + 108, 40),
        (7000, True, len(moof) + 148, 60),
    ]
    assert read == len(moof) + 16 + 16  # the box headers and the moof
    keys, _read = fmp4.samples(fmp4.Segment(6.0, 1.5, str(seg)), init)
    assert [s.pts for s in keys] == [6000, 7000]


# ---------------------------------------------------- the measurement itself
def test_a_measurement_that_saw_its_file_change_says_nothing(monkeypatch, tmp_path) -> None:
    calls = {"n": 0}

    def identity(paths: list[str]) -> list:
        calls["n"] += 1
        return [(calls["n"], 0, 0, 0)]

    class Source:
        video = sources.Video("h264", W, H)
        sampling = sources.Sampling(mode="keyframes", duration_s=40)
        blacks = BLACKS

        def run(self, sink):
            for i in range(40):
                sink(float(i), _Frame(i), True)
            return self.sampling

    monkeypatch.setattr(sources, "identity", identity)
    monkeypatch.setattr(sources, "open_source", lambda *_a: Source())
    drawn = {i: reduced(seed=i) for i in range(40)}
    monkeypatch.setattr(planes, "reduce_segs", lambda f, blacks: drawn[f.seed])
    out = child.run(str(tmp_path / "film.mkv"), 600_000)
    assert out["error"] == "the file changed while it was measured" and "area" not in out


class _Frame:
    def __init__(self, seed: int) -> None:
        self.seed = seed


def test_a_measurement_whose_samples_fail_says_why(monkeypatch, tmp_path) -> None:
    class Source:
        video = sources.Video("h264", W, H)
        sampling = sources.Sampling(mode="keyframes", duration_s=40)
        blacks = BLACKS

        def run(self, sink):
            for i in range(40):
                self.sampling.attempted += 1
                if i % 10 == 0:
                    self.sampling.fail("a keyframe decoded broken")
                    continue
                sink(float(i), _Frame(i), True)
            return self.sampling

    monkeypatch.setattr(sources, "identity", lambda paths: [None])
    monkeypatch.setattr(sources, "open_source", lambda *_a: Source())
    drawn = {i: reduced(seed=i) for i in range(40)}
    monkeypatch.setattr(planes, "reduce_segs", lambda f, blacks: drawn[f.seed])
    out = child.run(str(tmp_path / "film.mkv"), 600_000)
    assert out["error"].startswith("4 of 40 samples failed, more than 1") and "area" not in out


def test_the_process_is_stopped_at_its_deadline_and_says_nothing(tmp_path: Path) -> None:
    # A "video" the child can't open fast: a FIFO no one writes to.
    import os

    fifo = tmp_path / "never.mkv"
    os.mkfifo(fifo)
    t0 = time.monotonic()
    m = picture.measure(str(fifo), 1_000, deadline=time.time() + 1.5)
    assert m.interrupted and m.area is None and m.reason == "stopped before it was done"
    assert time.monotonic() - t0 < 10
    stopped = picture.measure(str(fifo), 1_000, stop=lambda: True)
    assert stopped.interrupted


def test_the_process_says_what_it_could_not_read(tmp_path: Path) -> None:
    bad = tmp_path / "not-a-video.mkv"
    bad.write_bytes(b"\x00" * 4096)
    m = picture.measure(str(bad), 60_000)
    assert m.area is None and m.error and m.error.startswith("the file can't be opened")
    assert m.cpu_seconds > 0 and m.max_rss_bytes > 0
    assert m.report()["method"] == picture.METHOD == "picture/5"


def test_the_decoder_is_named(tmp_path: Path) -> None:
    assert child.decoder().startswith("av ")
    assert sys.modules["av"].__version__ in child.decoder()
