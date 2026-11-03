"""The cases a title's picture area has to get right, each a small video
ffmpeg makes here (320x180 mostly, a frame or two a second, a few minutes)
and measured as the analyzer measures it, in its own process.

Each film is a list of shots rendered by one ffmpeg run, with a keyframe at
every cut as an encoder's scene detection puts one, and every ten seconds
besides. Its truth is the union of its picture over every frame: an area
that leaves out any of it cuts the picture, which no case may do. Where the
rules can tell, the area is the truth (rounded outwards to even pixels, and
widened by what an encoder rings at a bright edge); where they can't, the
whole frame. The tests are skipped without an ffmpeg that encodes H.264.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from analyzer.pipelines import picture
from analyzer.pipelines.picture import Box, sources

W, H = 320, 180
# a 2.39:1 picture letterboxed in 16:9, and a 1.85:1 one
TOP, PIC = 24, 132
T185, P185 = 4, 172
T239 = Box(0, TOP, W, PIC)
FULL = Box(0, 0, W, H)
# what an encoder's ringing at a bright edge widens an area by, at most: the
# block (16 lines) the picture's edge cuts through
RING = 8


def _encoders() -> set[str]:
    if shutil.which("ffmpeg") is None:
        return set()
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return {ln.split()[1] for ln in out.stdout.splitlines() if len(ln.split()) > 1}


ENCODERS = _encoders()
pytestmark = pytest.mark.skipif("libx264" not in ENCODERS, reason="no ffmpeg that encodes H.264")


def ff(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True
    )


class Film:
    """Shots, one after the other, of fps frames a second."""

    def __init__(self, fps: int = 2) -> None:
        self.fps = fps
        self.parts: list[str] = []
        self.starts: list[float] = []
        self.t = 0.0

    def _add(self, chain: str, d: float) -> Film:
        self.parts.append(chain)
        self.starts.append(self.t)
        self.t += d
        return self

    def _noise(self) -> str:
        return f"noise=alls=2:allf=t+u:all_seed={len(self.parts) + 1}"

    def wide(self, d: float) -> Film:
        """A 2.39 shot, letterboxed."""
        return self._add(
            f"testsrc2=s={W}x{PIC}:r={self.fps}:d={d},{self._noise()},pad={W}:{H}:0:{TOP}:black", d
        )

    def full(self, d: float) -> Film:
        """A 1.78 shot: the full frame."""
        return self._add(f"testsrc2=s={W}x{H}:r={self.fps}:d={d},{self._noise()}", d)

    def high(self, d: float) -> Film:
        """A shot whose picture reaches the frame's top edge, over a bar of 24
        lines at its bottom."""
        return self._add(
            f"testsrc2=s={W}x{H - 24}:r={self.fps}:d={d},{self._noise()},pad={W}:{H}:0:0:black", d
        )

    def flat(self, d: float) -> Film:
        """A 1.85 shot, letterboxed a little."""
        return self._add(
            f"testsrc2=s={W}x{P185}:r={self.fps}:d={d},{self._noise()},pad={W}:{H}:0:{T185}:black",
            d,
        )

    def night(self, d: float, band: int = 90) -> Film:
        """A 2.39 night shot: a sky a few levels above black, a skyline
        low in the picture, never a brighter sky."""
        n = len(self.parts)
        return self._add(
            f"color=c=0x050505:s={W}x{PIC}:r={self.fps}:d={d}[n{n}];"
            f"testsrc2=s={W}x{PIC - band}:r={self.fps}:d={d},eq=brightness=-0.1[k{n}];"
            f"[n{n}][k{n}]overlay=0:{band},{self._noise()},pad={W}:{H}:0:{TOP}:black",
            d,
        )

    def blue(self, d: float, untagged: bool = False) -> Film:
        """A 1.78 open-matte shot whose extra rows are a deep blue sea, RGB
        0, 0, 100, in BT.709."""
        n = len(self.parts)
        return self._add(
            f"color=c=0x000064:s={W}x{H}:r={self.fps}:d={d},format=rgb24[b{n}];"
            f"testsrc2=s={W}x{PIC}:r={self.fps}:d={d},format=rgb24[p{n}];"
            f"[b{n}][p{n}]overlay=0:{TOP},scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            d,
        )

    def credits(self, d: float, inside: bool = False) -> Film:
        """End credits scrolling up through the whole frame (or through the
        2.39 picture only)."""
        h = PIC if inside else H
        chain = (
            f"color=c=black:s={W}x{h}:r={self.fps}:d={d},format=yuv420p,"
            "geq=lum='if(lt(mod(Y+T*20\\,18)\\,7)*lt(abs(X-160)\\,60)*lt(mod(X\\,5)\\,3)\\,235\\,16)'"
            ":cb=128:cr=128"
        )
        if inside:
            chain += f",pad={W}:{H}:0:{TOP}:black"
        return self._add(chain, d)

    def render(self, path: Path, vf: str = "", enc: list[str] | None = None) -> Path:
        if path.exists():
            return path
        graph = [f"{c},format=yuv420p,setsar=1[s{i}]" for i, c in enumerate(self.parts)]
        tail = "".join(f"[s{i}]" for i in range(len(self.parts)))
        tail += f"concat=n={len(self.parts)}:v=1:a=0" + (f",{vf}" if vf else "") + "[v]"
        enc = enc or [
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-g", str(10 * self.fps), "-keyint_min", str(self.fps), "-pix_fmt", "yuv420p",
        ]  # fmt: skip
        keys = []
        if self.starts:
            keys = ["-force_key_frames", ",".join(f"{t:.3f}" for t in self.starts)]
        ff("-filter_complex", ";".join([*graph, tail]), "-map", "[v]", *enc, *keys, str(path))
        return path


def glyphs(y: int, start: float, end: float, x0: int, n: int) -> str:
    """A line of burned-in text: n glyphs at x0, y, on screen from start to end."""
    on = f"enable='between(t,{start},{end})'"
    return ",".join(
        f"drawbox=x={x0 + g * 9}:y={y}:w=6:h=10:color=white:t=fill:{on}" for g in range(n)
    )


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("picture-cases")


def measure(path: Path, duration_ms: int | None = None) -> picture.Measurement:
    m = picture.measure(str(path), duration_ms)
    assert m.error is None, m.error
    return m


def whole_by(m: picture.Measurement, *spans: tuple[float, float]) -> None:
    """The area is the whole frame, decided early by a sample of one of the
    spans (a shot whose picture reaches every edge)."""
    assert m.area == FULL, (m.reason, m.sides)
    assert m.sampling["stoppedEarly"], m.sampling
    at = m.sampling["stoppedAtS"]
    assert any(a <= at <= b for a, b in spans), (at, spans)


def keeps(m: picture.Measurement, truth: Box) -> None:
    """The area leaves none of the truth out."""
    a = m.area
    assert a is not None, m.reason
    assert a.x <= truth.x and a.y <= truth.y, (a, truth, m.sides)
    assert a.x + a.width >= truth.x + truth.width, (a, truth, m.sides)
    assert a.y + a.height >= truth.y + truth.height, (a, truth, m.sides)


def near(m: picture.Measurement, truth: Box, ring: int = RING) -> None:
    """The area is the truth, widened by an encoder's ringing at most."""
    keeps(m, truth)
    a = m.area
    assert truth.x - a.x <= ring and truth.y - a.y <= ring, (a, truth)
    assert (a.x + a.width) - (truth.x + truth.width) <= ring, (a, truth)
    assert (a.y + a.height) - (truth.y + truth.height) <= ring, (a, truth)


def wide_film(media: Path, name: str, length: float, inserts: dict[float, tuple[str, float]],
              vf: str = "", fps: int = 2) -> Path:  # fmt: skip
    """A 2.39 film of length seconds, with shots of another kind at times."""
    f = Film(fps)
    t = 0.0
    for at, (kind, d) in sorted(inserts.items()):
        if at > t:
            f.wide(at - t)
        getattr(f, kind)(d)
        t = at + d
    if length > t:
        f.wide(length - t)
    return f.render(media / f"{name}.mkv", vf)


# ------------------------------------------------- decided before the end
def test_a_16_9_title_is_decided_whole_early(media: Path) -> None:
    # Its picture reaches every edge in the calibration's keyframes: nothing
    # after can make the area smaller, so they are all it reads.
    f = Film()
    f.full(400)
    path = f.render(media / "full.mkv")
    m = measure(path, 400_000)
    assert m.area == FULL and m.reason is None
    assert m.sampling["stoppedEarly"] and m.sampling["doubt"] == []
    assert m.samples <= 24 and m.sampling["maxGapS"] > 3, m.sampling
    assert m.bytes_read < os.path.getsize(path) / 4


def test_a_letterboxed_title_is_never_decided_early(media: Path) -> None:
    m = measure(wide_film(media, "wide", 400, {}), 400_000)
    assert not m.sampling["stoppedEarly"] and m.sampling["maxGapS"] <= 2.9
    near(m, T239)


def test_a_late_wider_shot_is_sampled_until_seen(media: Path) -> None:
    # No keyframe of the calibration's falls in the 1.78 shot at 380 s: the
    # rest of the title is sampled until it is.
    whole_by(measure(wide_film(media, "late178", 400, {380: ("full", 8)}), 400_000), (380, 388))


def test_one_side_at_the_edge_and_a_bar_on_the_other_is_sampled_whole(media: Path) -> None:
    # The top's picture reaches the frame's edge, the bottom's never does:
    # no early end; all of it is sampled, and the lone bar keeps the axis
    # whole.
    f = Film()
    f.high(400)
    m = measure(f.render(media / "high.mkv"), 400_000)
    assert not m.sampling["stoppedEarly"] and m.sampling["maxGapS"] <= 2.9
    assert m.area == FULL and m.sides["bottom"]["edge"] > 0, m.sides


def test_a_16_9_package_of_an_idr_every_6_s_is_decided_whole(media: Path) -> None:
    # Its sync samples are 6 s apart, too far to tell a crop: its picture
    # reaching every edge tells the whole frame all the same.
    f = Film()
    f.full(400)
    m = measure(package(f.render(media / "full.mkv"), media / "pkg-full", copy=False), 400_000)
    assert m.area == FULL and m.reason is None and m.sampling["stoppedEarly"]


# ---------------------------------------------- a wider picture in places
@pytest.mark.parametrize("length", [8, 3])
def test_a_short_wider_sequence_keeps_the_full_frame(media: Path, length: int) -> None:
    # One 1.78 sequence in a 2.39 film, its own keyframe at its cut.
    path = wide_film(media, f"seq178-{length}", 400, {186: ("full", length)})
    m = measure(path, 400_000)
    assert m.area == FULL and m.sampling["mode"] == "keyframes"


def test_short_wider_shots_keep_the_full_frame(media: Path) -> None:
    path = wide_film(media, "shots-wider", 400, {100 + 100 * k: ("full", 3) for k in range(3)})
    assert measure(path, 400_000).area == FULL


def test_flat_shots_in_a_scope_film_cut_nothing(media: Path) -> None:
    path = wide_film(media, "shots-185", 400, {50 * (k + 1) + 3: ("flat", 6) for k in range(7)})
    m = measure(path, 400_000)
    keeps(m, Box(0, T185, W, P185))
    assert m.area in (Box(0, T185, W, P185), FULL)


def test_a_sequence_past_the_catalogs_length_is_sampled(media: Path) -> None:
    # The catalog says 5 minutes; the file runs 400 s, 1.78 from 300 s.
    path = wide_film(media, "tail178", 400, {300: ("full", 30)})
    m = measure(path, 300_000)
    assert m.area == FULL and m.sampling["durationS"] == pytest.approx(400, abs=1)


def test_a_long_film_samples_every_keyframe(media: Path) -> None:
    # 40 minutes at a frame a second, a 12 s 1.78 sequence in the middle.
    path = wide_film(media, "long-1fps", 2400, {1203: ("full", 12)}, fps=1)
    m = measure(path, 2_400_000)
    assert m.area == FULL and m.samples >= 240


@pytest.mark.parametrize("gop", [4, 10])
def test_a_shot_between_keyframes_far_apart_is_seen(media: Path, gop: int) -> None:
    # A fixed group of pictures, no keyframe at the cuts (as a streaming
    # encode): short 1.78 shots that hold no keyframe. The groups longer than
    # 2.9 s are decoded through.
    f = Film()
    t = 0.0
    for at, d in ((101.0, 3.0), (203.0, 4.0), (307.0, 6.0)):
        f.wide(at - t).full(d)
        t = at + d
    f.wide(400 - t)
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-g", str(2 * gop),
           "-keyint_min", str(2 * gop), "-sc_threshold", "0", "-pix_fmt", "yuv420p"]  # fmt: skip
    path = media / f"nokf{gop}.mkv"
    if not path.exists():
        f.starts = []  # no keyframe forced at the cuts
        f.render(path, enc=enc)
    whole_by(measure(path, 400_000), (101, 104), (203, 207), (307, 313))


@pytest.mark.skipif("libx265" not in ENCODERS, reason="no ffmpeg that encodes HEVC")
def test_an_open_gop_film_with_its_keyframes_far_apart_is_seen(media: Path) -> None:
    f = Film()
    f.wide(101).full(4).wide(102).full(8).wide(185)
    x265 = "keyint=50:min-keyint=2:scenecut=40:open-gop=1:log-level=error"
    enc = ["-c:v", "libx265", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p",
           "-x265-params", x265]  # fmt: skip
    path = media / "opengop.mkv"
    if not path.exists():
        f.starts = []
        f.render(path, enc=enc)
    whole_by(measure(path, 400_000), (101, 105), (207, 215))


# ------------------------------------------------------- dark, grey, deep
def test_a_night_film_keeps_its_dark_sky(media: Path) -> None:
    f = Film()
    for _ in range(4):
        f.night(100)
    near(measure(f.render(media / "night.mkv"), 400_000), T239)


def test_a_night_teaser_keeps_its_dark_sky(media: Path) -> None:
    m = measure(Film().night(110, band=70).render(media / "night-teaser.mkv"), 110_000)
    assert m.sampling["mode"] == "sequential"
    near(m, T239)


def test_grey_bars_are_bars(media: Path) -> None:
    bar = f"w={W}:h={TOP}:color=0x121212:t=fill"
    vf = f"drawbox=x=0:y=0:{bar},drawbox=x=0:y={TOP + PIC}:{bar}"
    near(measure(wide_film(media, "grey-bars", 360, {}, vf=vf), 360_000), T239)


def test_a_deep_blue_open_matte_shot_keeps_the_full_frame(media: Path) -> None:
    path = wide_film(media, "blue", 400, {150: ("blue", 40)})
    assert measure(path, 400_000).area == FULL
    # the same with its colour tags cleared: the chroma is read from the
    # planes, whatever the file says of them
    bare = media / "blue-untagged.mkv"
    if not bare.exists():
        tags = "colour_primaries=2:transfer_characteristics=2:matrix_coefficients=2"
        ff("-i", str(path), "-c", "copy", "-bsf:v", f"h264_metadata={tags}", str(bare))
    assert measure(bare, 400_000).area == FULL


def test_full_range_video_has_its_bars_at_zero(media: Path) -> None:
    f = Film()
    f.wide(360)
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-g", "20",
           "-pix_fmt", "yuvj420p", "-color_range", "pc"]  # fmt: skip
    near(measure(f.render(media / "fullrange.mkv", enc=enc), 360_000), T239)


def test_a_flat_grey_edge_of_a_wider_shot_keeps_the_frame(media: Path) -> None:
    # A 20 s open-matte shot whose top and bottom bands are a flat dark grey
    # (RGB 18: luma 31) in a film whose bars are black: the title's bar
    # black (16) tells them from the bars, flat as they are.
    f = Film()
    n = len(f.parts)
    f.wide(200)
    f._add(
        f"color=c=0x121212:s={W}x{TOP}:r=2:d=20[t{n}];color=c=0x121212:s={W}x{TOP}:r=2:d=20[u{n}];"
        f"testsrc2=s={W}x{PIC}:r=2:d=20[m{n}];[t{n}][m{n}][u{n}]vstack=inputs=3",
        20,
    )
    f.wide(180)
    m = measure(f.render(media / "flatgrey.mkv"), 400_000)
    assert m.area == FULL and m.sampling["blacks"]["top"] == pytest.approx(16, abs=1)


def test_a_flat_grey_edge_of_a_wider_shot_keeps_a_pillarbox_frame(media: Path) -> None:
    f = Film()
    pil = f"pad={W}:{H}:40:0:black"
    f._add(f"testsrc2=s=240x{H}:r=2:d=200,{f._noise()},{pil}", 200)
    n = len(f.parts)
    f._add(
        f"color=c=0x121212:s=40x{H}:r=2:d=20[l{n}];color=c=0x121212:s=40x{H}:r=2:d=20[r{n}];"
        f"testsrc2=s=240x{H}:r=2:d=20[m{n}];[l{n}][m{n}][r{n}]hstack=inputs=3",
        20,
    )
    f._add(f"testsrc2=s=240x{H}:r=2:d=180,{f._noise()},{pil}", 180)
    assert measure(f.render(media / "flatgrey-pillar.mkv"), 400_000).area == FULL


def test_ten_bit_video_is_measured_as_eight(media: Path) -> None:
    f = Film()
    f.wide(360)
    enc = [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-g",
        "20",
        "-pix_fmt",
        "yuv420p10le",
    ]
    path = f.render(media / "tenbit.mkv", vf="format=yuv420p10le", enc=enc)
    near(measure(path, 360_000), T239)


@pytest.mark.skipif("libx265" not in ENCODERS, reason="no ffmpeg that encodes HEVC")
def test_hdr10_video_is_measured(media: Path) -> None:
    f = Film()
    f.wide(200).night(160)
    enc = [
        "-c:v", "libx265", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p10le",
        "-x265-params", "keyint=20:min-keyint=2:log-level=error:colorprim=bt2020:transfer=smpte2084"
        ":colormatrix=bt2020nc",
    ]  # fmt: skip
    near(measure(f.render(media / "hdr10.mkv", vf="format=yuv420p10le", enc=enc), 360_000), T239)


# ------------------------------------------------- what sits in the bars
def test_a_logo_in_a_corner_of_the_bar_is_left_out(media: Path) -> None:
    path = wide_film(
        media, "logo", 400, {}, vf="drawbox=x=284:y=6:w=26:h=10:color=white@0.9:t=fill"
    )
    m = measure(path, 400_000)
    near(m, T239)
    assert "logo" in m.sides["top"]


def test_a_logo_part_of_the_time_keeps_the_frame(media: Path) -> None:
    vf = "drawbox=x=284:y=6:w=26:h=10:color=white@0.9:t=fill:enable='lt(mod(t,100),40)'"
    m = measure(wide_film(media, "logo-40", 400, {}, vf=vf), 400_000)
    keeps(m, Box(0, 6, W, PIC + TOP - 6))
    assert m.area == FULL


def test_a_logo_that_moves_between_corners_is_no_logo(media: Path) -> None:
    # (a drawbox's x is read once: two boxes, each on screen half the time)
    box = "y=6:w=26:h=10:color=white@0.9:t=fill"
    vf = (f"drawbox=x=10:{box}:enable='lt(mod(t,100),50)',"
          f"drawbox=x=284:{box}:enable='gte(mod(t,100),50)'")  # fmt: skip
    m = measure(wide_film(media, "logo-moving", 400, {}, vf=vf), 400_000)
    keeps(m, Box(0, 6, W, PIC + TOP - 6))
    assert m.area == FULL


@pytest.mark.parametrize("y", [6, 160])
def test_one_caption_in_a_bar_between_keyframes_is_seen(media: Path, y: int) -> None:
    # One line of text, 4 s on screen, in a bar, between two keyframes 10 s
    # apart and away from the stretches.
    path = wide_film(media, f"caption-{y}", 400, {}, vf=glyphs(y, 141, 145, 120, 9))
    m = measure(path, 400_000)
    assert m.area == FULL


def test_subtitles_in_the_bar_between_keyframes_keep_the_frame(media: Path) -> None:
    # Short lines in the bottom bar, on screen from 1 s to 7 s of every 10:
    # never on a keyframe; the stretches decoded through find them.
    vf = ",".join(glyphs(162, 10 * k + 1, 10 * k + 7, 120, 9) for k in range(40))
    m = measure(wide_film(media, "subs", 400, {}, vf=vf), 400_000)
    keeps(m, Box(0, TOP, W, 172 - TOP))
    assert m.area == FULL


def test_end_credits_through_the_whole_frame_keep_it(media: Path) -> None:
    f = Film()
    f.wide(340).credits(60)
    assert measure(f.render(media / "credits.mkv"), 400_000).area == FULL


def test_end_credits_inside_the_picture_keep_the_letterbox(media: Path) -> None:
    f = Film()
    f.wide(340).credits(60, inside=True)
    near(measure(f.render(media / "credits-inside.mkv"), 400_000), T239)


def test_a_fade_of_the_whole_frame_cuts_nothing(media: Path) -> None:
    vf = "fade=t=out:st=190:d=6:color=0x0c0c0c,fade=t=in:st=196:d=4:color=0x0c0c0c"
    m = picture.measure(str(wide_film(media, "fade-all", 400, {}, vf=vf)), 400_000)
    assert m.area is None or m.area == FULL or (m.area.y <= TOP and m.area.height >= PIC)


# ------------------------------------------------------- other pictures
def test_a_windowbox_is_its_picture(media: Path) -> None:
    f = Film()
    for _ in range(4):
        f._add(f"testsrc2=s=192x144:r=2:d=90,{f._noise()},pad={W}:{H}:64:18:black", 90)
    near(measure(f.render(media / "windowbox.mkv"), 360_000), Box(64, 18, 192, 144))


def test_a_narrow_picture_in_a_scope_letterbox_is_its_picture(media: Path) -> None:
    f = Film()
    for _ in range(4):
        f._add(f"testsrc2=s=176x{PIC}:r=2:d=90,{f._noise()},pad={W}:{H}:72:{TOP}:black", 90)
    near(measure(f.render(media / "postage.mkv"), 360_000), Box(72, TOP, 176, PIC))


def test_a_trailer_of_mixed_shots_keeps_the_full_frame(media: Path) -> None:
    f = Film(fps=24)
    for _ in range(6):
        f.wide(12).full(3)
    enc = [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-g",
        "48",
        "-pix_fmt",
        "yuv420p",
    ]
    assert measure(f.render(media / "trailer.mkv", enc=enc), 90_000).area == FULL


def test_a_short_clip_is_decoded_through(media: Path) -> None:
    f = Film(fps=24)
    f.wide(30)
    enc = [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-g",
        "240",
        "-pix_fmt",
        "yuv420p",
    ]
    m = measure(f.render(media / "clip.mkv", enc=enc), 30_000)
    assert m.sampling["mode"] == "sequential" and m.samples >= 29
    near(m, T239)


def test_a_still_pillarboxed_is_its_picture(media: Path) -> None:
    still = media / "still.png"
    path = media / "still.mkv"
    if not path.exists():
        ff("-f", "lavfi", "-i", "testsrc2=s=180x180:d=1", "-frames:v", "1", str(still))
        ff("-loop", "1", "-framerate", "1", "-t", "200", "-i", str(still),
           "-vf", f"pad={W}:{H}:70:0:black,format=yuv420p", "-c:v", "libx264",
           "-preset", "ultrafast", "-g", "250", "-tune", "stillimage", str(path))  # fmt: skip
    near(measure(path, 200_000), Box(70, 0, 180, H))


# ----------------------------------------------------- formats and files
def test_an_anamorphic_dvd_says_its_pixels(media: Path) -> None:
    path = media / "dvd.mkv"
    if not path.exists():
        ff("-f", "lavfi", "-i", "testsrc2=s=360x216:r=25:d=360",
           "-vf", "noise=alls=5:allf=t+u,pad=360:288:0:36:black,setsar=64/45",
           "-c:v", "mpeg2video", "-q:v", "4", "-g", "12", str(path))  # fmt: skip
    m = measure(path, 360_000)
    near(m, Box(0, 36, 360, 216))
    assert m.frame == Box(0, 0, 360, 288) and m.video["sar"] == "64:45"
    assert m.sampling["mode"] == "keyframes" and m.video["codec"] == "mpeg2video"


def test_an_interlaced_dvd_program_stream_is_decoded_through(media: Path) -> None:
    path = media / "dvd.mpg"
    if not path.exists():
        ff("-f", "lavfi", "-i", "testsrc2=s=360x216:r=50:d=120",
           "-vf", "noise=alls=5:allf=t+u,pad=360:288:0:36:black,tinterlace=mode=interleave_top,"
           "setsar=64/45,fieldorder=tff", "-c:v", "mpeg2video", "-q:v", "4", "-g", "12",
           "-flags", "+ilme+ildct", "-top", "1", "-f", "vob", str(path))  # fmt: skip
    m = measure(path, 120_000)
    assert m.sampling["mode"] == "sequential"
    near(m, Box(0, 36, 360, 216), ring=8)


def test_an_interlaced_transport_stream_is_read_in_its_order(media: Path) -> None:
    path = media / "broadcast.ts"
    if not path.exists():
        ff("-f", "lavfi", "-i", f"testsrc2=s={W}x{PIC}:r=50:d=330",
           "-vf", f"noise=alls=5:allf=t+u,pad={W}:{H}:0:{TOP}:black,tinterlace=mode=interleave_top",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-flags", "+ildct+ilme",
           "-x264-params", "tff=1", "-g", "50", "-f", "mpegts", str(path))  # fmt: skip
    m = measure(path, 330_000)
    assert m.sampling["mode"] == "stream" and m.sampling["maxGapS"] <= 2.9
    assert m.bytes_read >= os.path.getsize(path)
    # interlaced coding rings further: its blocks are of field pairs
    near(m, T239, ring=16)


def test_a_film_in_a_transport_stream_is_read_in_its_order(media: Path) -> None:
    # An MPEG-TS has no index of its keyframes: the few points its probe
    # finds are none, however far into the file they reach.
    path = media / "wide.ts"
    if not path.exists():
        src = wide_film(media, "wide", 400, {})
        ff("-i", str(src), "-c", "copy", "-f", "mpegts", str(path))
    m = measure(path, 400_000)
    assert m.sampling["mode"] == "stream" and m.sampling["maxGapS"] <= 2.9
    near(m, T239)
    # its 1.78 sequence, as a keyframe of the calibration's shows it
    path = media / "seq178-8.ts"
    if not path.exists():
        src = wide_film(media, "seq178-8", 400, {186: ("full", 8)})
        ff("-i", str(src), "-c", "copy", "-f", "mpegts", str(path))
    whole_by(measure(path, 400_000), (186, 194))


def test_a_transport_stream_with_b_frames_is_read_from_its_first_keyframe(media: Path) -> None:
    # Its first keyframe is decoded before its video's start time: the read
    # in its order starts at it all the same, so its first 10 s are sampled.
    path = media / "wide-bframes.ts"
    if not path.exists():
        f = Film(fps=8)
        f.wide(400)
        enc = ["-c:v", "libx264", "-preset", "veryfast", "-bf", "3", "-crf", "28",
               "-g", "80", "-pix_fmt", "yuv420p"]  # fmt: skip
        src = f.render(media / "wide-bframes.mkv", enc=enc)
        ff("-i", str(src), "-c", "copy", "-f", "mpegts", str(path))
    m = measure(path, 400_000)
    assert m.sampling["mode"] == "stream" and m.sampling["maxGapS"] <= 2.9, m.reason
    near(m, T239)


def test_an_mp4_with_b_frames_seeks_its_own_keyframes(media: Path) -> None:
    # An MP4's index holds decoding times, before its frames' presentation
    # times where they are reordered, and a seek to one lands on the
    # keyframe before it. Its keyframes 2 s apart are each read all the same,
    # none twice.
    path = media / "wide-bframes-gop2.mp4"
    if not path.exists():
        f = Film(fps=8)
        f.wide(400)
        enc = ["-c:v", "libx264", "-preset", "veryfast", "-bf", "3", "-crf", "28",
               "-g", "16", "-pix_fmt", "yuv420p"]  # fmt: skip
        src = f.render(media / "wide-bframes-gop2.mkv", enc=enc)
        ff("-i", str(src), "-c", "copy", str(path))
    m = measure(path, 400_000)
    assert m.sampling["mode"] == "keyframes", m.reason
    assert m.sampling["missed"] == 0 and m.sampling["duplicates"] == 0, m.reason
    assert m.sampling["maxGapS"] <= 2.9
    near(m, T239)


def test_a_rotated_video_cant_be_told(media: Path) -> None:
    flat, path = media / "phone-flat.mp4", media / "phone.mp4"
    if not path.exists():
        ff("-f", "lavfi", "-i", f"testsrc2=s={W}x{PIC}:r=10:d=20",
           "-vf", f"pad={W}:{H}:0:{TOP}:black", "-c:v", "libx264", "-preset", "ultrafast",
           str(flat))  # fmt: skip
        try:
            ff("-display_rotation:v:0", "90", "-i", str(flat), "-c", "copy", str(path))
        except subprocess.CalledProcessError:  # an ffmpeg before 6.0
            ff("-i", str(flat), "-c", "copy", "-metadata:s:v:0", "rotate=90", str(path))
    m = picture.measure(str(path), 20_000)
    assert m.area is None and m.reason == "the video is rotated by 90 degrees"


def test_a_cover_picture_is_no_video(media: Path) -> None:
    cover, path = media / "cover.jpg", media / "song.m4a"
    if not path.exists():
        ff("-f", "lavfi", "-i", "testsrc2=s=200x200:d=1", "-frames:v", "1", str(cover))
        ff("-f", "lavfi", "-i", "sine=frequency=440:duration=30", "-i", str(cover),
           "-map", "0:a", "-map", "1:v", "-c:a", "aac", "-c:v", "copy",
           "-disposition:v:0", "attached_pic", str(path))  # fmt: skip
    m = picture.measure(str(path), 30_000)
    assert m.area is None and m.reason.startswith("the file has no video stream")


def test_fonts_in_a_matroska_file_are_read_once(media: Path) -> None:
    font = media / "font.ttf"
    font.write_bytes(os.urandom(4 << 20))
    plain = wide_film(media, "plain", 400, {186: ("full", 8)})
    path = media / "fonts.mkv"
    if not path.exists():
        ff("-i", str(plain), "-attach", str(font), "-metadata:s:t", "mimetype=font/ttf",
           "-c", "copy", str(path))  # fmt: skip
    m = measure(path, 400_000)
    assert m.area == FULL
    opened, read = m.sampling["bytesOpen"], m.bytes_read
    # the font is read once, as the file is opened; the video's groups of
    # pictures of 10 s are decoded through
    assert 4 << 20 <= opened < 5 << 20
    assert read - opened < 2 * os.path.getsize(plain), (opened, read)


def test_a_damaged_stretch_of_a_file_cant_be_told(media: Path) -> None:
    # The bytes of the 1.78 sequence's packets zeroed: the seeks into it land
    # on the next good keyframe, and its samples are missing, not wrong.
    src = wide_film(media, "seq178-20", 400, {186: ("full", 20)})
    path = media / "damaged.mkv"
    if not path.exists():
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "packet=pts_time,pos,size", "-of", "csv=p=0", str(src)],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        spans = []
        for ln in probe.splitlines():
            t, size, pos = [*ln.split(","), "", "", ""][:3]
            if t and pos and 185 <= float(t) < 207:
                spans.append((int(pos), int(pos) + int(size)))
        data = bytearray(src.read_bytes())
        lo, hi = min(a for a, _b in spans), max(b for _a, b in spans)
        data[lo:hi] = bytes(hi - lo)
        path.write_bytes(bytes(data))
    m = picture.measure(str(path), 400_000)
    assert m.area is None, m.sides
    assert set(m.sampling.get("doubt") or ["error"]) & {"gap", "missed", "duplicates", "error"}


def test_a_file_cut_short_cant_be_told(media: Path) -> None:
    # Matroska writes its cues at its end: cut at 60 %, the file has none,
    # and says it runs 400 s; what is left is read in its order.
    src = wide_film(media, "wide", 400, {})
    path = media / "cut.mkv"
    if not path.exists():
        data = src.read_bytes()
        path.write_bytes(data[: int(len(data) * 0.6)])
    m = picture.measure(str(path), 400_000)
    assert m.area is None and "gap" in (m.sampling.get("doubt") or []), (m.reason, m.error)
    # what is left shows a picture in the whole frame: the area is that,
    # whatever the rest held
    src = wide_film(media, "seq178-8", 400, {186: ("full", 8)})
    path = media / "cut-178.mkv"
    if not path.exists():
        data = src.read_bytes()
        path.write_bytes(data[: int(len(data) * 0.6)])
    whole_by(picture.measure(str(path), 400_000), (186, 194))


def test_a_run_of_long_groups_of_pictures_reads_on(media: Path, monkeypatch) -> None:
    # Every group of pictures of this film is longer than 2.9 s: one run
    # decodes them all through, and its read limit is each group's, so a
    # film far longer than the limit is read to its end.
    path = wide_film(media, "seq178-8", 400, {186: ("full", 8)})
    monkeypatch.setattr(sources, "THROUGH_READ_MAX", path.stat().st_size // 8)
    src = sources.Original(str(path), 400.0)
    try:
        sm = src.run(lambda t, frame, counted: None)
    finally:
        src.close()
    assert sm.mode == "keyframes" and sm.stopped is None
    assert sm.max_gap() <= sources.TARGET_GAP_S and max(sm.times) > 397


def test_a_read_budget_reached_cant_be_told(media: Path, monkeypatch) -> None:
    monkeypatch.setenv("PICTURE_ORIGINAL_READ_MB_PER_HOUR", "16")
    big = media / "budget.mkv"
    if not big.exists():
        ff("-f", "lavfi", "-i", f"testsrc2=s={W}x{PIC}:r=24:d=120", "-vf",
           f"noise=alls=10:allf=t+u,pad={W}:{H}:0:{TOP}:black", "-c:v", "libx264", "-preset",
           "ultrafast", "-crf", "16", "-g", "24", str(big))  # fmt: skip
    assert big.stat().st_size > 8 << 20
    m = picture.measure(str(big), 120_000)
    assert m.area is None and m.sampling["doubt"][0] == "budget", m.reason


@pytest.mark.skipif(shutil.which("mkvmerge") is None, reason="no mkvmerge")
def test_a_file_mkvmerge_made_is_read_as_any(media: Path) -> None:
    src = wide_film(media, "seq178-8", 400, {186: ("full", 8)})
    made, bare = media / "mkvmerge.mkv", media / "mkvmerge-nocues.mkv"
    if not made.exists():
        subprocess.run(["mkvmerge", "-q", "-o", str(made), str(src)], check=True)
        subprocess.run(["mkvmerge", "-q", "--cues", "0:none", "-o", str(bare), str(src)],
                       check=True)  # fmt: skip
    m = measure(made, 400_000)
    assert m.area == FULL and m.sampling["mode"] == "keyframes"
    assert m.bytes_read < 2 * os.path.getsize(made)
    m = measure(bare, 400_000)
    assert m.area == FULL and m.sampling["mode"] == "stream"


def package(src: Path, out: Path, *, copy: bool = True) -> Path:
    """An HLS package of src: its video in 6 s fMP4 segments, copied (its
    keyframes) or encoded (a keyframe at every segment's start, and none
    between)."""
    playlist = out / "playlist.m3u8"
    if playlist.exists():
        return playlist
    out.mkdir(parents=True, exist_ok=True)
    codec = ["-c:v", "copy"] if copy else [
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-g", "12", "-keyint_min", "12",
        "-sc_threshold", "0",
    ]  # fmt: skip
    ff("-i", str(src), *codec, "-f", "hls", "-hls_segment_type", "fmp4", "-hls_time", "6",
       "-hls_playlist_type", "vod", "-hls_fmp4_init_filename", "init.mp4",
       "-hls_segment_filename", str(out / "seg-%05d.m4s"), str(playlist))  # fmt: skip
    return playlist


def test_a_package_is_sampled_by_its_keyframes(media: Path) -> None:
    # A rung copied of a source with a keyframe every 2 s: its sync samples
    # leave no two more than 3 s apart.
    f = Film()
    f.wide(400)
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-g", "4",
           "-pix_fmt", "yuv420p"]  # fmt: skip
    src = f.render(media / "wide-gop2.mkv", enc=enc)
    m = measure(package(src, media / "pkg-copy"), 400_000)
    assert m.sampling["mode"] == "package" and m.measured_on["kind"] == "package"
    assert m.sampling["maxGapS"] <= 2.9 and not m.sampling["stoppedEarly"]
    near(m, T239)


def test_a_package_with_a_keyframe_every_6_s_cant_tell(media: Path) -> None:
    # An encoder's rung: an IDR every 6 s, none at a cut. A shot of 3 s
    # between two of them has no sample: the measurement can't tell.
    vf = "drawbox=x=284:y=6:w=26:h=10:color=white@0.9:t=fill"
    pkg = package(wide_film(media, "logo", 400, {}, vf=vf), media / "pkg-encoded", copy=False)
    m = measure(pkg, 400_000)
    assert m.area is None
    assert m.reason.startswith("can't tell: its samples are up to 6.0 s apart"), m.reason
    assert m.sampling["doubt"] == ["gap"]


def test_a_short_package_is_decoded_through(media: Path) -> None:
    f = Film(fps=24)
    for _ in range(6):
        f.wide(12).full(3)
    enc = [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-g",
        "48",
        "-pix_fmt",
        "yuv420p",
    ]
    src = f.render(media / "trailer.mkv", enc=enc)
    m = measure(package(src, media / "pkg-trailer", copy=False), 90_000)
    assert m.sampling["mode"] == "package-sequential" and m.area == FULL


def test_a_measurement_reports_what_it_measured_and_what_it_cost(media: Path) -> None:
    path = wide_film(media, "wide", 400, {})
    m = measure(path, 400_000)
    r = m.report()
    assert r["method"] == "picture/4" and r["measuredOn"]["kind"] == "original"
    assert r["measuredOn"]["size"] == os.path.getsize(path)
    assert r["bytesRead"] > 0 and r["cpuSeconds"] >= 0 and r["sampling"]["stretches"] == 12
    assert r["decoder"].startswith("av ") and r["video"]["codec"] == "h264"
