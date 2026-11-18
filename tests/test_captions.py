"""The captions read of an original against a made-up ffprobe that answers
as a video it is told of would: what it asks ffprobe, at which priority; the
windows it places (the start, 10 to 90 %, fewer in a short video) and reads,
each on its own; captions present, absent, found in a later window; and every
guard that makes a read one that can't vouch for "none": a window that lands
elsewhere, one that decodes another's frames, negative times, frames without
times (a raw stream), no length, a length its last packet or its bytes belie
(a clock that started again), a window inside a group of pictures read from
the keyframe after it, errors, a timeout, a read stopped; and the ffprobe it
runs - the pipeline's 7.1, named by its version line or by its libraries,
asked again when the binary is replaced, never another release (the image's
4.4.2), nor the PATH's."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pytest
from captions_fake import FFPROBE_442, FFPROBE_715, LIBS_44, LIBS_71, FakeProbe, Video, frames

from analyzer.pipelines import captions


@pytest.fixture
def original(tmp_path: Path) -> Path:
    # as large as a ten-minute video at the made-up ffprobe's bitrate: a
    # sparse file, nothing written
    p = tmp_path / "Film (2024).mkv"
    with p.open("wb") as f:
        f.truncate(600 * 25 * 20_000)
    return p


ALL = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
CAPTIONED = {"frames": frames(48, with_captions=True), "streams": [{"closed_captions": 1}],
             "format": {"duration": "600.000000", "start_time": "0.000000"}}  # fmt: skip


def interval(run: list[str]) -> str:
    return run[run.index("-read_intervals") + 1]


def read(ff: FakeProbe, path: Path, **kw) -> captions.CaptionsRead:
    return captions.read(str(path), ffprobe=str(ff.path), **kw)


def test_captions_present(fake_ffprobe, original: Path) -> None:
    ff = fake_ffprobe(video=Video(captions=[[0, 600]]))
    r = read(ff, original)
    # duration: the length its file gives; span: the picture's as its records
    # state it (its start_time and duration), which the windows were placed over
    assert (r.ok, r.captions, r.start_packet, r.error, r.duration) == (True, True, 0, None, 600.0)
    assert r.span == {"start": 0.0, "end": 600.0}  # the end of its last packet
    assert r.probe == "ffprobe version n7.1.5-12-g1fdbca85aa-20260731"
    assert r.file_size == 600 * 25 * 20_000 and r.path == str(original)
    # every window is read, captions at the start or not
    assert [w.at for w in r.windows] == ALL and all(w.captions for w in r.windows)
    assert r.frames == 6 * 48


def test_captions_told_by_the_stream_s_closed_captions(fake_ffprobe, original) -> None:
    ff = fake_ffprobe().answers({**CAPTIONED, "frames": frames(48, with_captions=False)}, call=2)
    r = read(ff, original)
    assert (r.ok, r.captions) == (True, True) and r.windows[0].captions is True


def test_none_only_when_every_window_decoded_frames_without_captions(fake_ffprobe, original):
    ff = fake_ffprobe()
    r = read(ff, original)
    assert (r.ok, r.captions, r.error) == (True, False, None)
    assert [w.label for w in r.windows] == ["start", "10%", "30%", "50%", "70%", "90%"]
    assert all(w.frames == 48 and not w.error and w.captions is False for w in r.windows)
    # each placed at its share of the picture's span (0 to 600 s), and read
    # from the keyframe before it, its times the frames'
    assert [x for w in r.windows for x in (w.aim, w.start, w.end)] == pytest.approx([
        0.0, 0.0, 1.88, 60.0, 60.0, 61.88, 180.0, 180.0, 181.88,
        300.0, 300.0, 301.88, 420.0, 420.0, 421.88, 540.0, 540.0, 541.88])  # fmt: skip


def test_the_read_s_runs_niced(fake_ffprobe, original) -> None:
    ff = fake_ffprobe()
    read(ff, original)
    # the header probe of every stream; the start's 48 packets of the film's
    # stream; where the file says it ends, to the end or 5 s past the slack;
    # then each window and the packets from its seek to its place
    assert ff.runs[0] == [*captions.header_args(), str(original)]
    assert ff.runs[1] == [*captions.decode_args(48, None, 0), str(original)]
    # where its span ends as its records state it, to a frame step, 0.5 s and 5 s past
    assert ff.runs[2] == [*captions.end_args(600.0, 605.54, 0), str(original)]
    # the first packets' times and keyframes, none decoded (the GOP: the
    # short-picture rule, the landing of a GOP of 0; a whole read's leading
    # pictures); the start's packets to its place
    assert ff.runs[3] == [*captions.keys_args(0), str(original)]
    assert ff.runs[4] == [*captions.list_args(0.0, 0), str(original)]
    assert [interval(run) for run in ff.runs[5:]] == [
        "60.000%+#48", "60.000%60.001", "180.000%+#48", "180.000%180.001",
        "300.000%+#48", "300.000%300.001", "420.000%+#48", "420.000%420.001",
        "540.000%+#48", "540.000%540.001"]  # fmt: skip
    assert ff.niceness is not None and ff.niceness >= captions.NICE


@pytest.mark.parametrize("packets", [24, 96, 10**6, 0, "48", 48.0, None])
def test_a_claim_that_names_another_count_of_packets_is_one_it_can_t_serve(
    fake_ffprobe, original, packets
) -> None:
    # the plan's 48, not configuration: a claim that names 24 would read six
    # windows of 24 frames; the read fails, and runs no ffprobe on the file
    ff = fake_ffprobe()
    r = read(ff, original, packets=packets)
    assert (r.ok, r.captions, r.unreadable, r.windows) == (False, None, True, [])
    assert r.error.endswith("a window, where the plan reads 48: a read it can't serve")
    assert ff.runs == []
    assert read(fake_ffprobe(), original, packets=48).ok


@pytest.mark.parametrize(
    ("span", "offsets"),
    [
        (600.0, [0.0, 60.0, 180.0, 300.0, 420.0, 540.0]),
        (120.0, [0.0, 12.0, 36.0, 60.0, 84.0, 90.0]),  # 90 % at most 30 s short of the end
        (100.0, [0.0, 10.0, 30.0, 50.0, 70.0, 70.0]),  # six, never merged: a short picture
        (8.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # shorter than the tail: all at the start
    ],
)
def test_the_windows_a_span_places(span, offsets) -> None:
    # each at its plan fraction, whatever the tail caps
    assert captions.plan(0.0, span) == pytest.approx(list(zip(ALL, offsets, strict=True)))


def test_captions_that_start_later_are_found_in_a_later_window(fake_ffprobe, original) -> None:
    ff = fake_ffprobe(video=Video(captions=[[100, 600]]))
    r = read(ff, original)
    assert (r.ok, r.captions) == (True, True)
    assert [(w.label, w.captions) for w in r.windows] == [
        ("start", False), ("10%", False), ("30%", True), ("50%", True), ("70%", True),
        ("90%", True)]  # fmt: skip


# ------------------------------------------------------------ the guards


def test_a_window_that_lands_elsewhere_fails_the_read(fake_ffprobe, original) -> None:
    # every seek lands at the start, as on a clock that started again: the
    # windows decode the start's frames, far from where they were placed
    ff = fake_ffprobe(video=Video(lands="start"))
    r = read(ff, original)
    assert (r.ok, r.captions) == (False, None) and not r.unreadable
    assert r.error.startswith("the 10% window: its first frame is at 0.000 s, 60.0 s from where "
                              "it was placed (60.000 s): it landed elsewhere")  # fmt: skip
    assert all(w.error and w.captions is None for w in r.windows[1:])


def test_a_window_may_land_a_group_of_pictures_from_its_place(fake_ffprobe, original) -> None:
    # a keyframe every 14 s: the windows land 4, 12, 6, 0 and 8 s before
    # 60, 180, 300, 420 and 540 s; every 15 s of 620 s, 2, 6, 10, 14 and 3 s
    # before 62, 186, 310, 434 and 558 s: all within max(10 s, the GOP and
    # 0.5 s)
    assert read(fake_ffprobe(video=Video(gop=14)), original).captions is False
    r = read(fake_ffprobe(video=Video(gop=15, length=620)), original)
    assert r.ok and [round(w.aim - w.start) for w in r.windows] == [0, 2, 6, 10, 14, 3]
    # each on the last keyframe before its place, as its packets show
    assert all(w.keyframe_before for w in r.windows)


def test_a_window_on_the_keyframe_before_its_place_may_lie_up_to_30_s_back(
    fake_ffprobe, original
) -> None:
    # a keyframe every 31 s: the 10 % window lands 29 s before 60 s, on the
    # last keyframe before it, which its packets show; every 25 s, the 70 %
    # 20 s before 420 s
    for gop, at, back in ((31, 1, 29), (25, 4, 20)):
        r = read(fake_ffprobe(video=Video(gop=gop)), original)
        w = r.windows[at]
        assert (r.ok, r.captions) == (True, False), r.error
        assert (round(w.aim - w.start), w.keyframe_before, w.error) == (back, True, None)


def test_a_window_more_than_30_s_back_fails(fake_ffprobe, original) -> None:
    # a keyframe every 40 s of 750: the 10 % window lands 35 s before 75 s,
    # on the keyframe before, but too far
    r = read(fake_ffprobe(video=Video(gop=40, length=750)), original)
    assert (r.ok, r.captions) == (False, None)
    assert r.error == ("the 10% window: its first frame is at 40.000 s, 35.0 s from where it was "
                       "placed (75.000 s): it landed elsewhere")  # fmt: skip
    # the 30 % window, 25 s back on the keyframe before, is one
    assert (r.windows[2].error, r.windows[2].keyframe_before) == (None, True)


def test_a_window_whose_seek_passed_a_keyframe_over_fails(fake_ffprobe, original) -> None:
    # seeks that land on the keyframe before the one before their place: the
    # 10 % window 19 s before 69 s, with the keyframe at 60 s between
    r = read(fake_ffprobe(video=Video(gop=10, length=690, lands="skip")), original)
    assert (r.ok, r.captions) == (False, None)
    assert r.error == ("the 10% window: its first frame is at 50.000 s, 19.0 s before where it "
                       "was placed (69.000 s), and its packets don't show it the last keyframe "
                       "before there (a keyframe at 60.000 s lies between): it landed "
                       "elsewhere")  # fmt: skip
    # the 50 % 15 s back: more than max(10 s, the GOP of 10 s and 0.5 s), and
    # not on the keyframe before its place, fails too
    w = r.windows[3]
    assert (round(w.aim - w.start), w.keyframe_before) == (15, False)
    assert w.error.startswith("its first frame is at 330.000 s, 15.0 s before where it was "
                              "placed (345.000 s), and its packets don't show it")  # fmt: skip


def test_a_window_that_starts_after_its_place_is_none_before_it(fake_ffprobe, original) -> None:
    # a transport stream's seeks land where they aim, and its windows begin
    # at the keyframe after: up to 15 s, and no keyframe before
    r = read(fake_ffprobe(video=Video(gop=6.1, lands="after")), original)
    assert r.ok and not any(w.keyframe_before for w in r.windows[1:])


def test_a_window_that_decodes_another_s_frames_fails() -> None:
    # a window that lands on the start's frames (1 s in) decodes them again
    start = captions.Window(0.0, 0.0, frames=48, captions=False, start=0.0, end=1.88)
    look = captions._Look([{"closed_captions": 0}],
                          [{"pts_time": f"{1 + i / 25:.6f}"} for i in range(48)],
                          "", {})  # fmt: skip
    w = captions._window(0.1, 6.0, look, 48, [start], (False, ""))
    assert w.error == "it decoded frames the start window decoded (1.000-2.880 s)"


def test_a_short_video_is_read_whole(fake_ffprobe, original) -> None:
    # a minute, a keyframe every 10 s: its windows can't lie apart, and the
    # whole picture is read, every packet decoded (the short-picture rule)
    r = read(fake_ffprobe(video=Video(length=60.0, gop=10.0)), original)
    assert (r.ok, r.captions) == (True, False)
    [w] = r.windows
    got = (w.whole, w.label, w.frames, w.packets, w.start, w.end)
    assert got == (True, "whole", 1500, 1500, 0.0, 59.96)


def test_negative_times_fail_the_read(fake_ffprobe, original) -> None:
    r = read(fake_ffprobe(video=Video(start_time=-5.0)), original)
    assert (r.ok, r.captions) == (False, None)
    assert r.error == "its frames' presentation times are negative (-5.000 s)"
    assert r.windows[0].captions is None


def test_frames_without_times_fail_the_read_a_raw_stream(fake_ffprobe, original) -> None:
    r = read(fake_ffprobe(video=Video(pts=False)), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error == "its frames have no presentation times (a raw stream)"


def test_a_file_that_gives_no_length_fails_the_read(fake_ffprobe, original) -> None:
    r = read(fake_ffprobe(video=Video(says=0)), original)
    assert (r.ok, r.unreadable) == (False, True)
    assert r.error == "its file gives no length to place the windows by"


def test_a_length_its_last_packet_belies_fails_the_read(fake_ffprobe, original) -> None:
    # a header that says 30 s of 300: from 30 s on, packets run on past
    # 35.54 s, where the listing stops (a frame step, 0.5 s and 5 s); the last
    # one listed, at 35.52 s, ends a frame step later
    r = read(fake_ffprobe(video=Video(length=300.0, says=30.0)), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error == ("its picture ends at 30.000 s as its records state it (its picture's "
                       "start_time and duration), its last video packet read there ends at "
                       "35.560 s: more than a frame step (0.040 s) and 0.5 s apart")  # fmt: skip
    # one that says 9 s more than it has: more than a frame step and 0.5 s
    assert read(fake_ffprobe(video=Video(length=600.0, says=609.0)), original).captions is None
    # a last group of pictures of 60 s, 1,500 packets, is listed to its end
    ff = fake_ffprobe(video=Video(gop=60.0))
    assert not read(ff, original).unreadable and interval(ff.runs[2]) == "600.000%605.540"


def test_a_length_its_bytes_belie_fails_the_read_a_clock_that_started_again(
    fake_ffprobe, original
) -> None:
    # a clock that started again, read as one that wrapped: 26 hours of a
    # file whose bytes last 10 minutes at the bitrate of its first frames
    r = read(fake_ffprobe(video=Video(length=600.0, says=94_743.0)), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error == ("its file says it lasts 94743 s, more than 10 times the 600 s its "
                       "300000000 bytes last at the bitrate of its first frames")  # fmt: skip
    # a first frame smaller than the rest makes the bytes last longer: no doubt
    assert read(fake_ffprobe(video=Video(packet_bytes=2_000)), original).captions is False


def test_no_packet_where_the_file_says_it_ends_fails_the_read(fake_ffprobe, original) -> None:
    ff = fake_ffprobe().answers("", call=3)
    r = read(ff, original)
    assert (r.ok, r.unreadable) == (False, True)
    assert r.error == ("no video packet read where its picture ends as its records state it "
                       "(600.000 s, its picture's start_time and duration)")  # fmt: skip


# ------------------------------------------------------------ the closed lists

FILM = {"index": 0, "codec_type": "video", "codec_name": "h264", "height": 1080}


def streams(*more: dict, film: dict = FILM) -> list[dict]:
    return [film, *({"index": i + 1, **m} for i, m in enumerate(more))]


@pytest.mark.parametrize(
    "more",
    [
        {"codec_type": "subtitle", "codec_name": "eia_608", "codec_tag_string": "c608"},
        {"codec_type": "subtitle", "codec_tag_string": "c708"},
        {"codec_type": "data", "codec_tag_string": "c708"},
    ],
)
def test_a_stream_of_captions_of_its_own_is_captions(fake_ffprobe, original, more) -> None:
    # the picture carries none: the stream does
    r = read(fake_ffprobe(video=Video(streams=streams(more))), original)
    assert (r.ok, r.captions, r.caption_streams, r.off_list) == (True, True, 1, [])
    assert not any(w.captions for w in r.windows)


@pytest.mark.parametrize(
    "more",
    [
        {"codec_type": "audio", "codec_name": "eac3"},
        {"codec_type": "subtitle", "codec_name": "subrip"},
        {"codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"},
        {"codec_type": "subtitle", "codec_name": "dvb_teletext"},
        {"codec_type": "attachment", "codec_name": "ttf"},
        {"codec_type": "data", "codec_tag_string": "tmcd"},
        {"codec_type": "data", "codec_name": "unknown", "codec_tag_string": "tmcd"},
        {"codec_type": "data", "codec_name": "scte_35"},
        {"codec_type": "data", "codec_name": "timed_id3"},
        # attached pictures: Matroska's (as ffprobe shows them, and by its
        # tags alone) and MP4's cover, which has no mimetype
        {"codec_type": "video", "codec_name": "png", "disposition": {"attached_pic": 1},
         "tags": {"filename": "cover_land.png", "mimetype": "image/png"}},
        {"codec_type": "video", "codec_name": "mjpeg",
         "tags": {"filename": "poster.jpg", "mimetype": "image/jpeg"}},
        {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}},
    ],
)
def test_the_streams_on_the_list(fake_ffprobe, original, more) -> None:
    r = read(fake_ffprobe(video=Video(streams=streams(more))), original)
    assert (r.ok, r.captions, r.caption_streams, r.off_list) == (True, False, 0, [])


@pytest.mark.parametrize("container", captions.CONTAINERS)
def test_the_containers_on_the_list(fake_ffprobe, original, container: str) -> None:
    r = read(fake_ffprobe(video=Video(format_name=container)), original)
    assert (r.ok, r.captions, r.off_list) == (True, False, [])


@pytest.mark.parametrize("codec", captions.VIDEO_CODECS)
def test_the_video_codecs_on_the_list(fake_ffprobe, original, codec: str) -> None:
    film = {**FILM, "codec_name": codec}
    r = read(fake_ffprobe(video=Video(streams=[film])), original)
    if codec in captions.SCANNED_CODECS:  # never "none": its windows can't tell
        assert (r.ok, r.captions, r.codec, r.off_list) == (False, None, codec, [])
    else:
        assert (r.ok, r.captions, r.codec, r.off_list) == (True, False, codec, [])


@pytest.mark.parametrize(
    ("video", "off"),
    [
        (Video(format_name="flv"), ["container:flv"]),
        (Video(format_name="mxf"), ["container:mxf"]),
        (Video(streams=[{**FILM, "codec_name": "vc1"}]), ["codec:vc1"]),
        (Video(streams=[{**FILM, "codec_name": "mpeg4"}]), ["codec:mpeg4"]),
        (Video(streams=streams({"codec_type": "data", "codec_name": "bin_data"})),
         ["stream:data:bin_data"]),
        (Video(streams=streams({"codec_type": "data"})), ["stream:data:none"]),
        (Video(streams=streams({"codec_type": "subtitle", "codec_tag_string": "c609"})),
         ["stream:subtitle:c609"]),
        # a track of captions in Matroska: no codec ffmpeg names
        (Video(streams=streams({"codec_type": "subtitle", "codec_name": "unknown",
                                "codec_tag_string": "[0][0][0][0]"})),
         ["stream:subtitle:none"]),
        (Video(streams=streams({"codec_type": "video", "codec_name": "h264"})),
         ["video-streams:2"]),
        (Video(streams=[{**FILM, "height": 486}]), ["picture:vbi-lines"]),
        (Video(format_name="h264"), ["container:h264"]),  # a raw stream
        # an image no container marks as attached is a video stream
        (Video(streams=streams({"codec_type": "video", "codec_name": "mjpeg"})),
         ["video-streams:2", "codec:mjpeg"]),
        (Video(streams=streams({"codec_type": "video", "codec_name": "png",
                                "tags": {"title": "poster.jpg"}})),
         ["video-streams:2", "codec:png"]),
        # marked, but no image, or with a length of its own
        (Video(streams=streams({"codec_type": "video", "codec_name": "mjpeg",
                                "disposition": {"attached_pic": 1},
                                "tags": {"filename": "poster.jpg",
                                         "mimetype": "application/octet-stream"}})),
         ["video-streams:2", "codec:mjpeg"]),
        (Video(streams=streams({"codec_type": "video", "codec_name": "mjpeg",
                                "disposition": {"attached_pic": 1},
                                "tags": {"filename": "poster.jpg", "mimetype": "image/jpeg",
                                         "DURATION": "01:30:00.000000000"}})),
         ["video-streams:2", "codec:mjpeg"]),
        (Video(streams=streams({"codec_type": "video", "codec_name": "mjpeg",
                                "disposition": {"attached_pic": 1}, "nb_frames": "2400"})),
         ["video-streams:2", "codec:mjpeg"]),
        (Video(streams=streams({"codec_type": "video", "codec_name": "h264",
                                "disposition": {"attached_pic": 1}})),
         ["video-streams:2"]),
    ],
)
def test_what_is_off_the_lists_is_not_known(fake_ffprobe, original, video, off) -> None:
    # off a list, on any axis: reported, and the read is not known; nothing
    # is decoded
    ff = fake_ffprobe(video=video)
    r = read(ff, original)
    assert (r.ok, r.captions, r.unreadable, r.off_list) == (False, None, True, off)
    assert r.error == "off the closed lists: " + ", ".join(off)
    assert len(ff.runs) == 1


# An MP4's chapter track: on the list only where all five hold (chapter_track)
MP4 = "mov,mp4,m4a,3gp,3g2,mj2"
CHAPTERS = {
    "codec_type": "data", "codec_name": "bin_data", "codec_tag_string": "text", "nb_frames": "3",
}  # fmt: skip


def without(key: str) -> dict:
    return {k: v for k, v in CHAPTERS.items() if k != key}


def test_the_chapter_track_of_an_mp4_is_on_the_list(fake_ffprobe, original) -> None:
    video = Video(format_name=MP4, chapters=3, streams=streams(CHAPTERS))
    r = read(fake_ffprobe(video=video), original)
    assert (r.ok, r.captions, r.off_list) == (True, False, [])


# each condition failing alone, and the condition that holds it: taken out, the stream is on it
OFF = ["stream:data:bin_data"]
CHAPTER_FAILS = [
    ("container", Video(format_name="matroska,webm", chapters=3,
                        streams=streams(CHAPTERS)), OFF),
    ("stream", Video(format_name=MP4, chapters=3,
                     streams=streams({**CHAPTERS, "codec_tag_string": "tx3g"})), OFF),
    ("stream", Video(format_name=MP4, chapters=3,
                     streams=streams(without("codec_tag_string"))), OFF),
    # no chapters, and a track of no samples (the count holds, 0 = 0): the chapters alone fail
    ("has", Video(format_name=MP4, chapters=0,
                  streams=streams({**CHAPTERS, "nb_frames": "0"})), OFF),
    ("count", Video(format_name=MP4, chapters=3,
                    streams=streams({**CHAPTERS, "nb_frames": "4"})), OFF),
    ("count", Video(format_name=MP4, chapters=3, streams=streams(without("nb_frames"))), OFF),
    ("only", Video(format_name=MP4, chapters=3,
                   streams=streams(CHAPTERS, CHAPTERS)), OFF * 2),
]  # fmt: skip


@pytest.mark.parametrize(("condition", "video", "off"), CHAPTER_FAILS)
def test_a_chapter_track_short_of_one_condition_is_off_the_list(
    fake_ffprobe, original, condition, video, off
) -> None:
    r = read(fake_ffprobe(video=video), original)
    assert (r.ok, r.captions, r.unreadable, r.off_list) == (False, None, True, off)


@pytest.mark.parametrize(("condition", "video", "off"), CHAPTER_FAILS)
def test_each_chapter_condition_is_what_holds_it(
    fake_ffprobe, original, monkeypatch, condition, video, off
) -> None:
    # that condition taken out, the same file's stream is on the list: the test above fails then
    if condition == "stream":
        real = captions._chapter_stream
        monkeypatch.setattr(
            captions, "_chapter_stream", lambda st: real({**st, "codec_tag_string": "text"})
        )
    else:
        monkeypatch.setattr(captions, f"_chapter_{condition}", lambda *a: True)
    r = read(fake_ffprobe(video=video), original)
    assert (r.ok, r.off_list) == (True, [])


def test_attached_pictures_before_the_film_are_passed_over(fake_ffprobe, original) -> None:
    # a poster and a landscape cover attached to a Matroska file, and one
    # whose file name says a font: the mimetype and the marks decide
    def attached(i: int, codec: str, name: str, mime: str) -> dict:
        return {"index": i, "codec_type": "video", "codec_name": codec, "duration": "600.0",
                "disposition": {"attached_pic": 1}, "tags": {"filename": name, "mimetype": mime}}

    video = Video(streams=[attached(0, "mjpeg", "poster.jpg", "image/jpeg"),
                           attached(1, "png", "cover_land.png", "image/png"),
                           {**FILM, "index": 2},
                           attached(3, "mjpeg", "font.ttf", "image/jpeg")])  # fmt: skip
    ff = fake_ffprobe(video=video)
    r = read(ff, original)
    assert (r.ok, r.captions, r.codec) == (True, False, "h264")
    # every run after the header's reads the film's stream
    assert all(run[run.index("-select_streams") + 1] == "2" for run in ff.runs[1:])


def test_a_window_counts_its_frames_in_fields_and_those_that_repeat_one(
    fake_ffprobe, original
) -> None:
    # progressive for 100 s, then in fields; soft pulldown in the first 30 s
    ff = fake_ffprobe(video=Video(interlaced=[[100, 600]], repeats=[[0, 40]]))
    r = read(ff, original)
    assert [(w.label, w.interlaced, w.repeats) for w in r.windows] == [
        ("start", 0, 48), ("10%", 0, 0), ("30%", 48, 0), ("50%", 48, 0), ("70%", 48, 0),
        ("90%", 48, 0)]  # fmt: skip
    assert r.captions is False  # fields are no captions: the catalog holds them on its own


def test_hevc_s_packets_are_scanned_for_captions_its_decoder_does_not_export(
    fake_ffprobe, original
) -> None:
    # a suffix SEI: the decoded frames show none, the packets' bytes do
    hevc = Video(streams=[{**FILM, "codec_name": "hevc"}], bytes_captions=[[0, 600]])
    ff = fake_ffprobe(video=hevc)
    r = read(ff, original)
    assert (r.ok, r.captions) == (True, True) and all(w.captions for w in r.windows)
    assert any("-show_data" in run for run in ff.runs)
    # H.264 has no suffix SEI: nothing is scanned
    ff = fake_ffprobe(video=Video(bytes_captions=[[0, 600]]))
    r = read(ff, original)
    assert r.captions is False and not any("-show_data" in run for run in ff.runs)
    # and HEVC whose frames show captions is not scanned either
    ff = fake_ffprobe(video=Video(streams=[{**FILM, "codec_name": "hevc"}], captions=[[0, 600]]))
    read(ff, original)
    assert not any("-show_data" in run for run in ff.runs)


def test_a_window_that_decodes_fewer_than_half_its_packets_fails(fake_ffprobe, original) -> None:
    # 10 frames of 48 at a keyframe, and none after it to read from
    ff = fake_ffprobe().answers({**CAPTIONED, "frames": frames(10, with_captions=False, start=60.0),
                                 "streams": [{"closed_captions": 0, "avg_frame_rate": "25/1"}]},
                                call=6)  # fmt: skip
    r = read(ff, original)
    assert (r.ok, r.captions) == (False, None)
    assert r.error == ("the 10% window: it decoded 10 frames, fewer than 24 (half its 48 "
                       "packets, 24 at least), from the keyframe (video packet 0)")  # fmt: skip


# ------------------------------------------- before a keyframe


def test_a_video_cut_inside_a_group_of_pictures_is_read_from_its_first_keyframe(
    fake_ffprobe, original
) -> None:
    # its picture states it starts at 0.000 s: a span whose start lies 3.96 s
    # from its first frame fails the read (a frame step and 0.5 s)
    film = {**FILM, "start_time": "3.960000", "duration": "596.040000"}
    ff = fake_ffprobe(video=Video(first_key=3.96, gop=4.0, streams=[film]))
    r = read(ff, original)
    w = r.windows[0]
    assert (w.start_packet, w.frames, w.start, w.error) == (99, 48, 3.96, None)
    first, keys, again, end = ff.runs[1:5]
    assert first == [*captions.decode_args(48), str(original)]
    # the keyframe looked for among the packets, none decoded; then the
    # packets up to it and 48 from it decoded, counted from the start
    assert keys == [*captions.key_args(), str(original)] and "-show_frames" not in keys
    assert again == [*captions.decode_args(147), str(original)]
    assert end == [*captions.end_args(600.0, 605.54), str(original)]
    assert r.captions is False
    r = read(fake_ffprobe(video=Video(first_key=3.96, gop=4.0)), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error.startswith("its picture starts at 0.000 s as its records state it")


def test_a_window_inside_a_group_of_pictures_is_read_from_the_keyframe_after(
    fake_ffprobe, original
) -> None:
    # a transport stream's seek lands where it aims, inside a group of 4 s:
    # from the same point, the packets up to the next keyframe and 48 after
    ff = fake_ffprobe(video=Video(lands="after", gop=4.0, first_key=0.0))
    r = read(ff, original)
    assert r.captions is False
    w = r.windows[1]  # 60 s is a keyframe; 180 s too: every 4 s from 0
    assert (w.start, w.error) == (60.0, None)
    # groups of 6.1 s from 0: a keyframe 1 s after the 10 % point (60 s): 23
    # of the 48 packets decode, fewer than half, and the window is read from
    # the keyframe on; 3 s after the 30 % point (180 s): none of the 48 does,
    # and the keyframe is looked for
    r = read(fake_ffprobe(video=Video(lands="after", gop=6.1, first_key=0.0)), original)
    ten, thirty = r.windows[1], r.windows[2]
    assert r.captions is False
    assert (ten.frames, ten.start_packet, ten.start) == (48, 25, pytest.approx(61.0))
    assert (thirty.frames, thirty.start_packet, thirty.start) == (48, 75, pytest.approx(183.0))


def test_a_first_frame_far_from_the_start_its_records_state_fails_the_read(
    fake_ffprobe, original
) -> None:
    # the file's clock starts at 0, its video's first keyframe 5 s in, its
    # records state its picture from 0 to 600 s: its first frame is 5 s from
    # that start, more than a frame step and 0.5 s
    r = read(fake_ffprobe(video=Video(first_key=5.0)), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error == ("its picture starts at 0.000 s as its records state it (its picture's "
                       "start_time and duration), its first frame decoded is at 5.000 s: more "
                       "than a frame step (0.040 s) and 0.5 s apart")  # fmt: skip


def test_without_a_keyframe_in_reach_the_read_fails(fake_ffprobe, original) -> None:
    # the first keyframe 60 s in: past the 1200 packets looked through
    ff = fake_ffprobe(video=Video(first_key=60.0, gop=4.0))
    r = read(ff, original)
    assert (r.ok, r.captions) == (False, None)
    assert r.error == "no keyframe among the first 1200 video packets" and not r.transient
    assert r.windows[0].frames == 0
    # its start window placed no frame: it says no span (a span of none is
    # no report the catalog takes), and goes on the wire as its windows
    from analyzer.katalog import _captions_read

    body = _captions_read(r)
    assert r.span is None and "span" not in body and len(body["captionWindows"]) == 6
    assert body["captionWindows"][0]["frames"] == 0 and r.gop == 0.0


# ------------------------------------------------------- failed runs


@pytest.mark.parametrize(
    ("rc", "err", "transient"),
    [
        (1, "Film (2024).mkv: Invalid data found when processing input", False),
        (1, "[matroska,webm @ 0x1] EBML header parsing failed", False),
        (1, "Film (2024).mkv: Input/output error", True),
        (1, "Film (2024).mkv: Stale file handle", True),
        (234, "", False),
    ],
)
def test_an_error_is_a_failed_read_never_no_captions(
    fake_ffprobe, original, rc: int, err: str, transient: bool
) -> None:
    ff = fake_ffprobe().answers("{}", err=err, rc=rc)
    r = read(ff, original)
    assert (r.ok, r.captions, r.frames, r.unreadable) == (False, None, 0, True)
    assert r.error.startswith(f"ffprobe exited {rc}") and (err in r.error)
    assert r.transient is transient


def test_a_window_s_error_fails_the_read_naming_it(fake_ffprobe, original) -> None:
    ff = fake_ffprobe().answers("{}", err="Film (2024).mkv: Input/output error", rc=1, call=8)
    r = read(ff, original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, False)
    assert r.error == "the 30% window: ffprobe exited 1: Film (2024).mkv: Input/output error"
    assert r.transient is True
    # the windows after it are read all the same
    assert [w.label for w in r.windows] == ["start", "10%", "30%", "50%", "70%", "90%"]
    assert (r.windows[2].captions, r.windows[2].frames) == (None, 0)


def test_a_timeout_is_a_failed_read_never_no_captions(fake_ffprobe, original) -> None:
    ff = fake_ffprobe().answers(sleep=30)
    t0 = time.monotonic()
    r = read(ff, original, timeout=1.0)
    assert time.monotonic() - t0 < 10  # ended, not waited for
    assert (r.ok, r.captions, r.timed_out, r.transient) == (False, None, True, True)
    assert r.error == "ffprobe did not answer within 1 s"


def test_the_timeout_is_the_whole_read_s(fake_ffprobe, original) -> None:
    ff = fake_ffprobe().answers(sleep=30, call=4)  # a window's run
    t0 = time.monotonic()
    r = read(ff, original, timeout=3.0)
    assert time.monotonic() - t0 < 12 and r.timed_out and r.captions is None


def test_a_read_stopped_with_its_worker_failed(fake_ffprobe, original) -> None:
    ff = fake_ffprobe().answers(sleep=30)
    r = read(ff, original, stop=lambda: True)
    assert (r.ok, r.captions, r.interrupted, r.transient) == (False, None, True, True)


@pytest.mark.parametrize(
    ("answer", "err", "starts", "transient"),
    [
        ("not json", "", "ffprobe's output is no JSON", False),
        ("[]", "", "ffprobe's output is no JSON object", False),
        # an audio file: no video stream to carry them
        ({"format": {"format_name": "matroska,webm", "duration": "60"},
          "streams": [{"index": 0, "codec_type": "audio", "codec_name": "aac"}]}, "",
         "no video stream to read", False),
    ],
)
def test_a_read_that_cant_tell_fails(fake_ffprobe, original, answer, err, starts, transient):
    ff = fake_ffprobe().answers(answer, err=err)
    r = read(ff, original)
    assert r.ok is False and r.captions is None
    assert r.error.startswith(starts) and r.transient is transient


def test_no_captions_while_ffprobe_said_a_storage_error_fails(fake_ffprobe, original) -> None:
    plain = {**CAPTIONED, "frames": frames(48, with_captions=False),
             "streams": [{"closed_captions": 0, "avg_frame_rate": "25/1"}]}  # fmt: skip
    ff = fake_ffprobe().answers(plain, err="Read error at pos. 4096", call=2)
    r = read(ff, original)
    assert (r.ok, r.captions) == (False, None)
    assert r.error.startswith("ffprobe found no captions and said") and r.transient


def test_captions_found_stand_whatever_ffprobe_says_besides(fake_ffprobe, original) -> None:
    ff = fake_ffprobe().answers(CAPTIONED, err="[h264 @ 0x1] co located POCs unavailable",
                                call=2)  # fmt: skip
    r = read(ff, original)
    assert r.ok and r.captions is True and "co located POCs unavailable" in r.note


def test_no_file_to_read_fails_without_running_ffprobe(fake_ffprobe, tmp_path: Path) -> None:
    ff = fake_ffprobe()
    r = read(ff, tmp_path / "gone.mkv")
    assert (r.ok, r.captions, r.transient) == (False, None, True)
    assert r.error.startswith("no file to read") and ff.runs == []


# ------------------------------------------------------------ its ffprobe


def test_the_image_s_4_4_2_is_refused(fake_ffprobe, original) -> None:
    ff = fake_ffprobe(FFPROBE_442)
    with pytest.raises(captions.WrongProbe, match=r"4\.4\.2.*not the pipeline's 7\.1\.x"):
        captions.read(str(original), ffprobe=str(ff.path))
    assert ff.runs == []  # it never read the file


@pytest.mark.parametrize(
    ("version", "release", "ok"),
    [
        (FFPROBE_715, "7.1.5", True),
        ("ffprobe version 7.1 Copyright (c) 2007-2024\n" + LIBS_71, "7.1", True),
        ("ffprobe version 7.1.1-1ubuntu1 Copyright\n" + LIBS_71, "7.1.1", True),
        # a build named by its commit, of the 7.1 branch's libraries
        ("ffprobe version 1fdbca8 Copyright (c) 2007-2026\n" + LIBS_71, "7.1", True),
        (FFPROBE_442, "4.4.2", False),
        ("ffprobe version 4.4.2 Copyright\n" + LIBS_71, "4.4.2", False),  # its line rules
        ("ffprobe version 1a2b3c4 Copyright\n" + LIBS_44, None, False),
        ("ffprobe version n7.0.2 Copyright\n", "7.0.2", False),
        ("ffprobe version 9.0.2 Copyright\nlibavutil      61.  1.102 / 61.  1.102\n", "9.0.2",
         False),
        ("ffprobe version N-117000-g0123456789-20260731 Copyright\n", None, False),
        ("ffmpeg version n7.1.5-12-g1fdbca85aa-20260731 Copyright\n" + LIBS_71, None, False),
        ("", None, False),
    ],
)
def test_which_ffprobe_is_the_pipeline_s(version: str, release, ok: bool) -> None:
    p = captions.parse_version("/x/ffprobe", version)
    assert p.release == release and p.ok is ok
    assert (p.refusal() is None) is ok


def test_a_build_named_by_its_commit_says_what_its_libraries_are() -> None:
    p = captions.parse_version("/x/ffprobe", "ffprobe version 1fdbca8 Copyright\n" + LIBS_71)
    assert p.name == (
        "7.1 by its libraries (ffprobe version 1fdbca8: libavutil 59.39.100, "
        "libavcodec 61.19.101, libavformat 61.7.103)"
    )
    assert captions.parse_version("/x/ffprobe", FFPROBE_715).name == (
        "ffprobe version n7.1.5-12-g1fdbca85aa-20260731"
    )
    assert captions.parse_version("/x/ffprobe", "").name is None


# katalog-manager's reading of a report's "ffprobe" (internal/captions,
# ffprobeRE): a release's major and minor, of the version or its whole line
CATALOG_FFPROBE = re.compile(
    r"^(?:(?i:ffprobe)\s+(?i:version)\s+)?n?(\d+)\.(\d+)(?:\.\d+)*(?:[-+~_ ].*)?$"
)


@pytest.mark.parametrize(
    "version",
    [
        FFPROBE_715,
        "ffprobe version 7.1 Copyright\n" + LIBS_71,
        "ffprobe version 7.1.1-1ubuntu1 Copyright\n" + LIBS_71,
        "ffprobe version 1fdbca8 Copyright (c) 2007-2026\n" + LIBS_71,
    ],
)
def test_every_ffprobe_a_read_runs_is_named_as_the_catalog_takes_it(version: str) -> None:
    p = captions.parse_version("/x/ffprobe", version)
    m = CATALOG_FFPROBE.match(p.name or "")
    assert p.ok and m and (m.group(1), m.group(2)) == ("7", "1"), p.name


def test_an_ffprobe_replaced_in_place_is_asked_again(fake_ffprobe, original) -> None:
    ff = fake_ffprobe().answers(CAPTIONED)
    assert captions.probe(str(ff.path)).ok
    # what the binary answers changes, the binary does not: its verdict stands
    (ff.dir / "version.txt").write_text(FFPROBE_442)
    assert captions.probe(str(ff.path)).ok
    # another binary put where it was (the image's 4.4.2, say), no restart:
    # asked again, and refused
    other = FakeProbe(ff.dir.parent / "other", FFPROBE_442)
    os.replace(other.path, ff.path)
    assert captions.probe(str(ff.path)).release == "4.4.2"
    with pytest.raises(captions.WrongProbe, match=r"4\.4\.2"):
        captions.read(str(original), ffprobe=str(ff.path))


def test_no_ffprobe_where_it_is_named(tmp_path: Path) -> None:
    captions.probe.cache_clear()
    p = captions.probe(str(tmp_path / "none" / "ffprobe"))
    assert not p.ok and "no ffprobe to run at" in p.refusal()
    plain = tmp_path / "ffprobe"
    plain.write_text("#!/bin/sh\n")  # not executable
    assert not captions.probe(str(plain)).ok


def test_its_ffprobe_is_captions_ffprobe_or_its_own_folder_never_the_path(
    monkeypatch, fake_ffprobe, original
) -> None:
    monkeypatch.delenv("CAPTIONS_FFPROBE", raising=False)
    assert captions.ffprobe_path() == "/opt/ffmpeg-7.1/bin/ffprobe" == captions.DEFAULT_FFPROBE
    monkeypatch.setenv("CAPTIONS_FFPROBE", "/elsewhere/ffprobe")
    assert captions.ffprobe_path() == "/elsewhere/ffprobe"
    # a 7.1 ffprobe on the PATH is never taken for the one it names
    on_path = fake_ffprobe().answers(CAPTIONED)
    monkeypatch.setenv("PATH", f"{on_path.dir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(captions.WrongProbe, match="no ffprobe to run at /elsewhere/ffprobe"):
        captions.read(str(original))
    assert on_path.runs == []


@pytest.mark.parametrize(("value", "s"), [(None, 1800.0), ("30", 30.0), ("1", 5.0), ("x", 1800.0)])
def test_how_long_a_read_may_take(monkeypatch, value, s: float) -> None:
    if value is None:
        monkeypatch.delenv("CAPTIONS_READ_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("CAPTIONS_READ_TIMEOUT_S", value)
    assert captions.read_timeout() == s
