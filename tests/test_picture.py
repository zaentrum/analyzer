"""The picture-area measurement (pipelines/picture.py).

The rules are held on frames drawn here and reduced as ffmpeg reduces them
(segment means per row, then per column); the sampling, the package's
segments and ffmpeg's reduction on short videos ffmpeg makes, one for each
case the measurement has to get right. Those are skipped without an ffmpeg
that encodes H.264."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from analyzer.pipelines import picture
from analyzer.pipelines.picture import Box, decide

BLACK = 16


# ------------------------------------------------------------ drawn frames
def canvas(w: int, h: int, level: int = BLACK) -> list[bytearray]:
    return [bytearray([level]) * w for _ in range(h)]


def paint(
    img: list[bytearray], x: int, y: int, w: int, h: int, seed: int = 0, dim: int = 0
) -> None:
    """A textured picture over x, y, w, h: values 40..219 (less dim)."""
    for r in range(y, y + h):
        row = img[r]
        for c in range(x, x + w):
            row[c] = max(BLACK, 40 + (c * 7 + r * 13 + seed * 31) % 180 - dim)


def fill(img: list[bytearray], x: int, y: int, w: int, h: int, level: int) -> None:
    for r in range(y, y + h):
        img[r][x : x + w] = bytes([level]) * w


def reduce(img: list[bytearray], k: int = picture.SEGMENTS) -> bytes:
    """img reduced as ffmpeg's area scale reduces it: k means per row, then
    k per column."""
    h, w = len(img), len(img[0])
    out = bytearray()
    for row in img:
        for j in range(k):
            a, b = j * w // k, (j + 1) * w // k
            out.append(round(sum(row[a:b]) / (b - a)))
    for col in zip(*img, strict=True):
        for j in range(k):
            a, b = j * h // k, (j + 1) * h // k
            out.append(round(sum(col[a:b]) / (b - a)))
    return bytes(out)


W, H = 320, 180


def letterboxed(n: int = 30, top: int = 22, height: int = 134, **kw) -> list[bytes]:
    frames = []
    for i in range(n):
        img = canvas(W, H)
        paint(img, 0, top, W, height, seed=i, **kw)
        frames.append(reduce(img))
    return frames


def test_a_letterbox_is_its_picture() -> None:
    d = decide(letterboxed(), W, H)
    assert d.area == Box(0, 22, 320, 134)
    assert d.sides["top"]["bar"] == 22 and d.sides["bottom"]["bar"] == 24


def test_a_full_frame_is_the_frame() -> None:
    frames = []
    for i in range(30):
        img = canvas(W, H)
        paint(img, 0, 0, W, H, seed=i)
        frames.append(reduce(img))
    d = decide(frames, W, H)
    assert d.area == Box(0, 0, W, H)


def test_dark_and_black_frames_never_shrink_it() -> None:
    # A third of the frames black (a fade), a third so dark that no row of
    # theirs holds picture: the bright third alone sets the edges.
    frames = letterboxed(10)
    frames += [reduce(canvas(W, H)) for _ in range(10)]
    frames += letterboxed(10, dim=200)
    d = decide(frames, W, H)
    assert d.area == Box(0, 22, 320, 134)


def test_a_wider_scene_widens_it() -> None:
    # 2.39 throughout, 1.85 in some of the frames: the union.
    frames = letterboxed(24) + letterboxed(6, top=4, height=172)
    d = decide(frames, W, H)
    assert d.area == Box(0, 4, 320, 172)


def test_an_edge_few_frames_show_keeps_the_frame_whole() -> None:
    # One frame of 29 reaches further out: it may be a scene the samples
    # saw too little of, so the side keeps the full frame.
    frames = letterboxed(29) + letterboxed(1, top=4, height=172)
    d = decide(frames, W, H)
    assert d.area == Box(0, 0, W, H)
    assert "needed" in d.sides["top"]["note"]


def test_noise_on_single_lines_of_a_bar_is_no_picture() -> None:
    frames = []
    for i in range(30):
        img = canvas(W, H)
        paint(img, 0, 22, W, 134, seed=i)
        img[3 + i % 15][40:80] = bytes([BLACK + 60]) * 40  # a bright line, never two at once
        frames.append(reduce(img))
    assert decide(frames, W, H).area == Box(0, 22, 320, 134)


def test_a_subtitle_in_a_bar_keeps_that_side_whole() -> None:
    frames = letterboxed(26)
    for i in range(4):  # a subtitle in the bottom bar, in 4 frames of 30
        img = canvas(W, H)
        paint(img, 0, 22, W, 134, seed=i)
        for g in range(10):
            fill(img, 100 + g * 12, 164, 8, 10, 235)
        frames.append(reduce(img))
    d = decide(frames, W, H)
    assert d.area == Box(0, 22, 320, 158)
    assert d.sides["bottom"]["note"].startswith("the bar is not black")
    assert d.sides["bottom"]["seen"]


def test_a_subtitle_over_a_black_frame_counts_too() -> None:
    frames = letterboxed(29)
    img = canvas(W, H)
    for g in range(10):
        fill(img, 100 + g * 12, 164, 8, 10, 235)
    frames.append(reduce(img))
    assert decide(frames, W, H).area == Box(0, 22, 320, 158)


def test_a_logo_in_a_bar_is_left_out() -> None:
    frames = []
    for i in range(30):
        img = canvas(W, H)
        paint(img, 0, 22, W, 134, seed=i)
        fill(img, 290, 6, 24, 11, 150)  # in every frame, a quarter of the width at most
        frames.append(reduce(img))
    assert decide(frames, W, H).area == Box(0, 22, 320, 134)


def test_a_band_across_a_bar_is_no_logo() -> None:
    # A band across the bar in every frame (a ticker) is too wide for a
    # logo: it is picture, and the area takes it in.
    frames = []
    for i in range(30):
        img = canvas(W, H)
        paint(img, 0, 22, W, 134, seed=i)
        fill(img, 20, 6, 280, 8, 150)
        frames.append(reduce(img))
    assert decide(frames, W, H).area == Box(0, 6, 320, 150)


def test_a_dim_band_in_a_bar_is_no_bar() -> None:
    # Dark picture, not black, where the bar would be (a dim scene of a
    # wider frame): the side keeps the full frame rather than cut it.
    frames = letterboxed(28)
    for i in range(2):
        img = canvas(W, H)
        paint(img, 0, 22, W, 134, seed=i)
        fill(img, 0, 2, W, 18, BLACK + 12)
        frames.append(reduce(img))
    assert decide(frames, W, H).area.y == 0


def test_ringing_at_an_edge_widens_it() -> None:
    frames = []
    for i in range(30):
        img = canvas(W, H)
        paint(img, 0, 22, W, 134, seed=i)
        fill(img, 0, 21, W, 1, BLACK + 9)  # one dim line right outside the edge
        frames.append(reduce(img))
    d = decide(frames, W, H)
    assert d.area == Box(0, 20, 320, 136)  # line 21 taken in, rounded to even
    assert "widened" in d.sides["top"]["note"]


def test_pillarbox_and_windowbox() -> None:
    def framed(x: int, y: int, w: int, h: int) -> list[bytes]:
        out = []
        for i in range(30):
            img = canvas(W, H)
            paint(img, x, y, w, h, seed=i)
            out.append(reduce(img))
        return out

    assert decide(framed(40, 0, 240, 180), W, H).area == Box(40, 0, 240, 180)
    assert decide(framed(40, 40, 240, 100), W, H).area == Box(40, 40, 240, 100)
    # A wide picture in a narrow window: half its columns' segments lie in
    # the bars, which the rows' extent leaves out.
    assert decide(framed(40, 50, 240, 80), W, H).area == Box(40, 50, 240, 80)


def test_edges_round_outwards_to_even_pixels() -> None:
    assert decide(letterboxed(top=23, height=133), W, H).area == Box(0, 22, 320, 134)


def test_too_few_frames_with_a_picture_say_so() -> None:
    frames = letterboxed(3) + [reduce(canvas(W, H)) for _ in range(27)]
    d = decide(frames, W, H)
    assert d.area is None and d.reason.startswith("too few frames")


def test_edges_that_are_never_black_have_no_bars() -> None:
    frames = [reduce(canvas(W, H, level=90)) for _ in range(30)]
    assert decide(frames, W, H).area == Box(0, 0, W, H)


def test_sample_times_spread_over_the_title() -> None:
    assert len(picture.sample_times(3600)) == 360
    assert len(picture.sample_times(60)) == picture.MIN_SAMPLES
    assert len(picture.sample_times(10 * 3600)) == picture.MAX_SAMPLES
    ts = picture.sample_times(100, 10)
    assert ts[0] > 0 and ts[-1] < 100
    assert picture.sample_times(0) == []


# --------------------------------------------------------------- playlists
def test_a_playlist_names_its_segments_in_its_folder(tmp_path: Path) -> None:
    (tmp_path / "playlist.m3u8").write_text(
        '#EXTM3U\n#EXT-X-MAP:URI="init.mp4"\n#EXTINF:6.0,\nseg-00001.m4s\n#EXTINF:5.5,\nseg-00002.m4s\n'
        "#EXT-X-ENDLIST\n"
    )
    init, segs = picture.read_playlist(str(tmp_path / "playlist.m3u8"))
    assert init == str(tmp_path / "init.mp4")
    assert segs == [
        (0.0, str(tmp_path / "seg-00001.m4s"), 6.0),
        (6.0, str(tmp_path / "seg-00002.m4s"), 5.5),
    ]
    assert picture.pick_segments(segs, [1.0, 2.0, 7.0]) == [
        (0.0, str(tmp_path / "seg-00001.m4s")),
        (6.0, str(tmp_path / "seg-00002.m4s")),
    ]


@pytest.mark.parametrize(
    "line", ["#EXT-X-BYTERANGE:1000@0\nall.mp4", "../other/seg.m4s", "https://example.com/seg.m4s"]
)
def test_a_playlist_of_ranges_or_of_other_folders_is_not_read(tmp_path: Path, line: str) -> None:
    (tmp_path / "p.m3u8").write_text(f'#EXTM3U\n#EXT-X-MAP:URI="init.mp4"\n#EXTINF:6.0,\n{line}\n')
    assert picture.read_playlist(str(tmp_path / "p.m3u8")) == (None, [])


def box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), kind) + body


def test_a_fragments_first_frame_ends_where_its_header_says() -> None:
    # tfhd: default-base-is-moof; trun: a data offset, sizes per sample.
    tfhd = box(b"tfhd", struct.pack(">I", 0x020000) + struct.pack(">I", 1))
    trun = box(
        b"trun",
        struct.pack(">I", 0x000201)
        + struct.pack(">I", 2)
        + struct.pack(">i", 0)
        + struct.pack(">I", 5000)
        + struct.pack(">I", 900),
    )
    traf = box(b"traf", tfhd + trun)
    moof_len = 8 + 16 + len(traf)
    # the data offset is from the moof's start to the mdat's payload
    trun = box(
        b"trun",
        struct.pack(">I", 0x000201)
        + struct.pack(">I", 2)
        + struct.pack(">i", moof_len + 8)
        + struct.pack(">I", 5000)
        + struct.pack(">I", 900),
    )
    moof = box(b"moof", box(b"mfhd", struct.pack(">II", 0, 1)) + box(b"traf", tfhd + trun))
    styp = box(b"styp", b"msdh\0\0\0\0")
    seg = styp + moof + struct.pack(">I4s", 8 + 5900, b"mdat")
    assert picture.first_frame_end(seg) == len(styp) + len(moof) + 8 + 5000
    assert picture.first_frame_end(styp) is None


# ------------------------------------------------- videos ffmpeg makes here
def _encoder() -> bool:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return False
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return "libx264" in r.stdout


needs_ffmpeg = pytest.mark.skipif(not _encoder(), reason="needs ffmpeg and ffprobe with libx264")

ENC = [
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-crf",
    "23",
    "-g",
    "10",
    "-keyint_min",
    "10",
    "-sc_threshold",
    "0",
    "-pix_fmt",
    "yuv420p",
]


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True
    )


def _src(w: int, h: int, d: int = 30) -> str:
    return f"testsrc2=size={w}x{h}:rate=10:duration={d}"


def _glyphs(y: int, start: int, end: int) -> str:
    on = f"enable='between(t,{start},{end})'"
    box = "drawbox=x={x}:y={y}:w=8:h=10:color=white:t=fill:{on}"
    return "".join("," + box.format(x=100 + g * 12, y=y, on=on) for g in range(10))


VIDEOS = {
    # a 2.39 picture letterboxed: 320x134 at y=22 in a 320x180 frame
    "letterbox": (
        ["-f", "lavfi", "-i", _src(320, 134), "-vf", "pad=320:180:0:22:black"],
        Box(0, 22, 320, 134),
    ),
    "pillarbox": (
        ["-f", "lavfi", "-i", _src(240, 180), "-vf", "pad=320:180:40:0:black"],
        Box(40, 0, 240, 180),
    ),
    "windowbox": (
        ["-f", "lavfi", "-i", _src(240, 100), "-vf", "pad=320:180:40:40:black"],
        Box(40, 40, 240, 100),
    ),
    # dark from 8 s to 20 s, then a fade to black from 24 s
    "dark": (
        [
            "-f",
            "lavfi",
            "-i",
            _src(320, 134),
            "-vf",
            "eq=brightness=-0.45:enable='between(t,8,20)',fade=t=out:st=24:d=6,pad=320:180:0:22:black",
        ],
        Box(0, 22, 320, 134),
    ),
    # 2.39 for 20 s, then the full 16:9 frame: the largest is the frame
    "aspect": (
        [
            "-f",
            "lavfi",
            "-i",
            _src(320, 134, 20),
            "-f",
            "lavfi",
            "-i",
            _src(320, 180, 10),
            "-filter_complex",
            "[0:v]pad=320:180:0:22:black[a];[a][1:v]concat=n=2:v=1:a=0",
        ],
        Box(0, 0, 320, 180),
    ),
    # 2.39 with 1.85 (320x172 at y=4) from 10 s to 20 s: the union
    "aspect185": (
        [
            "-f",
            "lavfi",
            "-i",
            _src(320, 134, 10),
            "-f",
            "lavfi",
            "-i",
            _src(320, 172, 10),
            "-f",
            "lavfi",
            "-i",
            _src(320, 134, 10),
            "-filter_complex",
            "[0:v]pad=320:180:0:22:black[a];[1:v]pad=320:180:0:4:black[b];"
            "[2:v]pad=320:180:0:22:black[c];[a][b][c]concat=n=3:v=1:a=0",
        ],
        Box(0, 4, 320, 172),
    ),
    # grain over the whole frame, the bars too
    "noise": (
        ["-f", "lavfi", "-i", _src(320, 134), "-vf", "pad=320:180:0:22:black,noise=c0s=14:allf=t"],
        Box(0, 22, 320, 134),
    ),
    # a white subtitle in the bottom bar from 10 s to 16 s: that bar is kept
    "subtitle": (
        [
            "-f",
            "lavfi",
            "-i",
            _src(320, 134),
            "-vf",
            "pad=320:180:0:22:black" + _glyphs(164, 10, 16),
        ],
        Box(0, 22, 320, 158),
    ),
    # a grey logo in the top bar the whole time: left out
    "logo": (
        [
            "-f",
            "lavfi",
            "-i",
            _src(320, 134),
            "-vf",
            "pad=320:180:0:22:black,drawbox=x=290:y=6:w=24:h=11:color=0xb0b0b0@0.7:t=fill",
        ],
        Box(0, 22, 320, 134),
    ),
    "full": (["-f", "lavfi", "-i", _src(320, 180)], Box(0, 0, 320, 180)),
}


@pytest.fixture(scope="module")
def videos(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("videos")
    for name, (args, _want) in VIDEOS.items():
        _ffmpeg(*args, *ENC, str(d / f"{name}.mkv"))
    # the letterbox as a package's video rendition: fMP4 HLS, 3 s segments
    (d / "pkg").mkdir()
    _ffmpeg(
        "-i",
        str(d / "letterbox.mkv"),
        "-c",
        "copy",
        "-f",
        "hls",
        "-hls_segment_type",
        "fmp4",
        "-hls_time",
        "3",
        "-hls_playlist_type",
        "vod",
        "-hls_fmp4_init_filename",
        "init.mp4",
        "-hls_segment_filename",
        str(d / "pkg" / "seg-%05d.m4s"),
        str(d / "pkg" / "playlist.m3u8"),
    )
    return d


@needs_ffmpeg
@pytest.mark.parametrize("name", list(VIDEOS))
def test_a_video_measures_as_drawn(videos: Path, name: str) -> None:
    m = picture.measure(str(videos / f"{name}.mkv"), None, interval_s=1.0)
    assert m.error is None and m.reason is None, (m.error, m.reason, m.sides)
    assert m.frame == Box(0, 0, 320, 180)
    want = VIDEOS[name][1]
    if name == "noise":
        # Grain in a bar may light the guard band at the edge, which the
        # area then takes in: never less than the picture, at most the band
        # more on a side.
        a, guard = m.area, 2 * max(2, round(180 * picture.GUARD))
        assert a.x <= want.x and a.y <= want.y, m.sides
        assert (
            a.x + a.width >= want.x + want.width and a.y + a.height >= want.y + want.height
        ), m.sides
        assert want.y - a.y <= guard and (a.y + a.height) - (want.y + want.height) <= guard, m.sides
    else:
        assert m.area == want, m.sides
    assert m.samples >= picture.MIN_INFORMATIVE and m.cpu_seconds > 0
    body = m.report()
    assert body["frame"] == {"width": 320, "height": 180}
    assert body["method"] == picture.METHOD


@needs_ffmpeg
def test_a_package_is_measured_a_segments_first_frame_at_a_time(videos: Path) -> None:
    m = picture.measure(str(videos / "pkg" / "playlist.m3u8"), None, interval_s=1.0)
    assert m.area == Box(0, 22, 320, 134), (m.error, m.reason, m.sides)
    assert m.samples == 10  # one a segment


@needs_ffmpeg
def test_a_segment_cut_after_its_first_frame_reduces_as_the_whole_one(videos: Path) -> None:
    pkg = videos / "pkg"
    init, segs = picture.read_playlist(str(pkg / "playlist.m3u8"))
    cut = picture._Job(0.0, init=init, segment=segs[3][1]).run("ffmpeg", picture._Cpu())
    whole = picture._reduce(
        "ffmpeg", ["-i", picture._concat(init, segs[3][1])], None, picture._Cpu()
    )
    assert cut is not None and cut == whole
    head = picture._segment_head(init, segs[3][1])
    assert (
        head is not None
        and len(head) < (pkg / "init.mp4").stat().st_size + Path(segs[3][1]).stat().st_size
    )


@needs_ffmpeg
def test_a_full_hd_letterbox_of_a_wide_film(tmp_path: Path) -> None:
    # A 2.39 film in a 1920x1080 frame: its picture 1920x804, 138 lines of
    # bar above and below.
    path = tmp_path / "wide.mkv"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=1920x804:rate=2:duration=12",
        "-vf",
        "pad=1920:1080:0:138:black",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-g",
        "2",
        "-pix_fmt",
        "yuv420p",
        str(path),
    )
    m = picture.measure(str(path), None)
    assert m.area == Box(0, 138, 1920, 804), (m.error, m.reason, m.sides)


@needs_ffmpeg
def test_what_cant_be_read_says_why(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mkv"
    junk.write_bytes(b"\x1a\x45\xdf\xa3 not a video")
    m = picture.measure(str(junk), 60_000)
    assert m.area is None and m.error and not m.interrupted
    assert "error" in m.report()


@needs_ffmpeg
def test_a_measurement_past_its_deadline_says_nothing(videos: Path) -> None:
    m = picture.measure(str(videos / "letterbox.mkv"), None, interval_s=1.0, deadline=0.0)
    assert m.interrupted and m.area is None
    m = picture.measure(str(videos / "letterbox.mkv"), None, interval_s=1.0, stop=lambda: True)
    assert m.interrupted
