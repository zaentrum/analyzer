"""The captions read's rules that make "none" true only where it is, one
test a rule: anything ffprobe says on stderr in a read that found no captions
(MUST 1); a picture whose decoder doesn't export every caption it may carry
never "none", and a packet scan that fails closed (MUST 2); the bounds of
every ffprobe (MUST 3); the file the same after the read as before it; a
frame placed in time; a window that starts after the one before; attached
pictures of the cover codecs only; the chapter track's tag and count exactly;
and the windows placed over the picture's own presentation span."""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import pytest
from captions_fake import FakeProbe, Video, frames

from analyzer.pipelines import captions

FILM = {"index": 0, "codec_type": "video", "codec_name": "h264", "height": 1080}
HEVC = {**FILM, "codec_name": "hevc"}
SAID = "[h264 @ 0x1] error while decoding MB 12 7, bytestream -5\n"


@pytest.fixture
def original(tmp_path: Path) -> Path:
    p = tmp_path / "Film (2024).mkv"
    with p.open("wb") as f:
        f.truncate(600 * 25 * 20_000)
    return p


def read(ff: FakeProbe, path: Path, **kw) -> captions.CaptionsRead:
    return captions.read(str(path), ffprobe=str(ff.path), **kw)


# -------------------------------------------- MUST 1: anything it says


@pytest.mark.parametrize("kind", ["decode", "list"])
def test_a_line_on_stderr_in_a_window_s_run_fails_the_window(fake_ffprobe, original, kind):
    r = read(fake_ffprobe(video=Video(said={kind: SAID})), original)
    assert (r.ok, r.captions) == (False, None)
    assert "ffprobe found no captions and said: [h264 @ 0x1] error while decoding" in r.error


def test_a_line_on_stderr_in_a_window_s_keyframe_scan_fails_the_window(fake_ffprobe, original):
    # seeks that land where they aim, in groups of 6.1 s: the 10 % window
    # looks for its keyframe among the packets from its point
    r = read(fake_ffprobe(video=Video(lands="after", gop=6.1, said={"keys-at": SAID})), original)
    assert (r.ok, r.captions) == (False, None)
    assert r.windows[1].error.startswith("ffprobe found no captions and said: [h264")


def test_a_line_on_stderr_in_the_keyframes_listing_fails_the_read(fake_ffprobe, original) -> None:
    # the listing of the first packets' keyframes decides the short-picture rule
    r = read(fake_ffprobe(video=Video(said={"keys": SAID})), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error.startswith("the listing of its keyframes said: [h264")


@pytest.mark.parametrize(
    ("kind", "words"),
    [pytest.param("header", "its header probe said: [h264", id="header"),
     pytest.param("end", "the listing of its end said: [h264", id="end")],
)  # fmt: skip
def test_a_line_on_stderr_in_the_header_or_the_end_fails_the_read(
    fake_ffprobe, original, kind, words
) -> None:
    r = read(fake_ffprobe(video=Video(said={kind: SAID})), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error.startswith(words)


def test_a_line_on_stderr_in_the_packet_scan_fails_the_window(fake_ffprobe, original) -> None:
    said = "[matroska,webm @ 0x1] Read error at pos. 123456: Input/output error\n"
    r = read(fake_ffprobe(video=Video(streams=[HEVC], said={"scan": said})), original)
    assert r.captions is None
    assert all(w.error.startswith("its packet scan failed: its packet scan said: [matroska")
               for w in r.windows)  # fmt: skip
    assert all(w.transient for w in r.windows)  # the storage's


# ------------------------------------------------ MUST 2: HEVC, the scan


def test_hevc_is_never_none(fake_ffprobe, original) -> None:
    # its windows as read, its captions not known until its packets' signature
    r = read(fake_ffprobe(video=Video(streams=[HEVC])), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, False)
    assert (r.captions_known, r.reason) == (False, "hevc: needs the packets' signature")
    assert all(not w.error for w in r.windows) and len(r.windows) == 6


def test_hevc_with_captions_in_its_packets_says_captions(fake_ffprobe, original) -> None:
    r = read(fake_ffprobe(video=Video(streams=[HEVC], bytes_captions=[[290, 310]])), original)
    assert (r.ok, r.captions) == (True, True)
    assert [w.label for w in r.windows if w.captions] == ["50%"]


@pytest.mark.parametrize(
    ("dump", "words"),
    [
        pytest.param("empty", "has 0 of the 48 packets asked for", id="empty"),
        pytest.param("cut", "ends inside a packet (cut short)", id="cut"),
        pytest.param("short", "has 47 of the 48 packets asked for", id="short"),
        pytest.param(
            "nodata", "has a packet without its bytes (0 bytes of its size 27)", id="nodata"
        ),
        pytest.param("junk", "has a line outside a packet ('packets:')", id="junk"),
    ],
)
def test_a_packet_scan_that_can_t_vouch_for_its_packets_fails(
    fake_ffprobe, original, dump, words
) -> None:
    r = read(fake_ffprobe(video=Video(streams=[HEVC], dump=dump)), original)
    assert r.captions is None
    want = f"its packet scan failed: its packet scan's dump {words}"
    assert all(w.error == want for w in r.windows)


def test_the_packet_scan_reads_no_more_than_its_bounds(fake_ffprobe, original, monkeypatch) -> None:
    monkeypatch.setattr(captions, "SCAN_PACKETS_MAX", 47)
    r = read(fake_ffprobe(video=Video(streams=[HEVC])), original)
    assert all(w.error == "its packet scan failed: its packet scan would read 48 packets, more "
               "than 47" for w in r.windows)  # fmt: skip
    monkeypatch.setattr(captions, "SCAN_PACKETS_MAX", 1000)
    monkeypatch.setattr(captions, "SCAN_BYTES_MAX", 1000)  # 27 bytes a packet: 37 packets
    r = read(fake_ffprobe(video=Video(streams=[HEVC])), original)
    assert all(w.error == "its packet scan failed: its packet scan's dump runs past 1000 bytes "
               "of packets" for w in r.windows)  # fmt: skip


def test_a_hevc_read_goes_on_the_wire_as_its_windows_with_captions_not_known(
    fake_ffprobe, original
) -> None:
    from analyzer.katalog import _captions_read

    body = _captions_read(read(fake_ffprobe(video=Video(streams=[HEVC])), original))
    assert len(body["captionWindows"]) == 6 and "error" not in body
    assert (body["captionsKnown"], body["reason"]) == (False, "hevc: needs the packets' signature")
    assert all(w["captions"] is False for w in body["captionWindows"])
    # captions found in a window go as found, nothing more said
    found = read(fake_ffprobe(video=Video(streams=[HEVC], bytes_captions=[[290, 310]])), original)
    body = _captions_read(found)
    assert "captionsKnown" not in body and any(w["captions"] for w in body["captionWindows"])


# ------------------------------------------------------- MUST 3: bounds


@pytest.mark.parametrize(
    "video",
    [pytest.param(Video(), id="windows"), pytest.param(Video(length=90.0), id="whole"),
     pytest.param(Video(streams=[HEVC]), id="hevc-scanned"),
     pytest.param(Video(lands="after", gop=6.1), id="keyframe-scans")],
)  # fmt: skip
def test_every_run_states_one_thread(fake_ffprobe, original, video) -> None:
    # the plan's bounds, on every run: the header probe, the listings, the
    # end's, every decode, a whole read and a packet scan alike
    ff = fake_ffprobe(video=video)
    read(ff, original)
    assert len(ff.runs) > 5
    assert all("-threads" in run and run[run.index("-threads") + 1] == "1" for run in ff.runs)
    assert any("-show_frames" in run for run in ff.runs)


def test_every_ffprobe_runs_bounded(fake_ffprobe, original) -> None:
    ff = fake_ffprobe()
    read(ff, original)
    limits = json.loads((ff.dir / "limits").read_text())
    assert limits["cpu"] == [600, 600]
    if sys.platform != "darwin":  # macOS can't lower an address space
        assert limits["as"] == [1280 << 20, 1280 << 20]


def test_a_run_that_hits_a_bound_fails_its_window(fake_ffprobe, original) -> None:
    # the 10 % window's decode killed as the CPU bound kills it (SIGXCPU)
    ff = fake_ffprobe().answers(call=6, signal=signal.SIGXCPU)
    r = read(ff, original)
    assert (r.ok, r.captions) == (False, None)
    assert r.windows[1].error == f"ffprobe exited -{int(signal.SIGXCPU)}"


# ------------------------------------------------------ S1 to S5, and the plan


def test_a_file_that_changes_while_it_is_read_fails_the_read(fake_ffprobe, original) -> None:
    r = read(fake_ffprobe(video=Video(touch="end")), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error == "the file changed while it was read (its inode, its time or its size)"


def test_a_frame_the_read_can_t_place_in_time_fails_its_window(fake_ffprobe, original) -> None:
    # 47 of the 10 % window's 48 frames without a time (the first has one)
    r = read(fake_ffprobe(video=Video(untimed=[[60.03, 62.0]])), original)
    assert (r.ok, r.captions) == (False, None)
    assert r.windows[1].error == ("47 of its 48 frames have no presentation time: it can't place "
                                  "them")  # fmt: skip


def test_a_window_that_doesn_t_start_after_the_one_before_fails() -> None:
    before = [captions.Window(0.3, 30.0, frames=48, captions=False, start=30.0, end=31.88)]
    frames = [{"pts_time": f"{25 + i / 25:.6f}"} for i in range(48)]
    look = captions._Look([{"closed_captions": 0}], frames, "", {})
    w = captions._window(0.5, 49.98, look, 48, before, (True, ""))
    assert w.error == ("it doesn't start after the 30% window (25.000 s, that one ends at "
                       "31.880 s)")  # fmt: skip


@pytest.mark.parametrize(
    "second",
    [
        {"codec_name": "h264", "tags": {"filename": "c.jpg", "mimetype": "image/jpeg"}},
        {"codec_name": "hevc", "disposition": {"attached_pic": 1},
         "tags": {"filename": "c.png", "mimetype": "image/png"}},
    ],
)  # fmt: skip
def test_a_second_film_marked_as_an_image_is_a_second_film(fake_ffprobe, original, second) -> None:
    streams = [FILM, {"index": 1, "codec_type": "video", **second}]
    r = read(fake_ffprobe(video=Video(streams=streams)), original)
    assert (r.ok, r.captions, r.off_list) == (False, None, ["video-streams:2"])


MP4 = "mov,mp4,m4a,3gp,3g2,mj2"
CHAPTERS = {"index": 1, "codec_type": "data", "codec_name": "bin_data",
            "codec_tag_string": "text", "nb_frames": "3"}  # fmt: skip


@pytest.mark.parametrize(
    "change",
    [pytest.param({"codec_tag_string": "TEXT"}, id="tag-TEXT"),
     pytest.param({"codec_tag_string": "Text"}, id="tag-Text"),
     pytest.param({"nb_frames": " 3 "}, id="count-spaces"),
     pytest.param({"nb_frames": "+3"}, id="count-plus"),
     pytest.param({"nb_frames": "3\n"}, id="count-newline"),
     pytest.param({"nb_frames": 3}, id="count-number"),
     pytest.param({"nb_frames": "\u0663"}, id="count-arabic-indic")],
)  # fmt: skip
def test_the_chapter_track_s_tag_and_count_are_read_exactly(fake_ffprobe, original, change):
    video = Video(format_name=MP4, chapters=3, streams=[FILM, {**CHAPTERS, **change}])
    r = read(fake_ffprobe(video=video), original)
    assert r.off_list == ["stream:data:bin_data"]


def test_the_windows_are_placed_over_the_span_its_records_state(fake_ffprobe, original) -> None:
    # its picture states it starts 5 s in and lasts 595 s: the span the
    # windows are placed over, which the read reports beside its file's length
    film = {**FILM, "start_time": "5.000000", "duration": "595.000000"}
    r = read(fake_ffprobe(video=Video(first_key=5.0, streams=[film])), original)
    assert (r.ok, r.span, r.duration) == (True, {"start": 5.0, "end": 600.0}, 600.0)
    want = [5.0] + [5.0 + f * 595.0 for f in captions.WINDOWS]
    assert [w.aim for w in r.windows] == pytest.approx(want)


@pytest.mark.parametrize(
    ("duration_ms", "ok", "words"),
    [pytest.param(600_000, True, "", id="record"),
     pytest.param(None, True, "", id="no-record-the-format-s"),  # the format's 600 s
     pytest.param(601_000, False, "its picture ends at 601.000 s as its records state",
                  id="record-1s-long"),
     pytest.param(599_400, False, "its picture ends at 599.400 s as its records state",
                  id="record-0.6s-short"),
     pytest.param(599_600, True, "", id="record-0.4s-short")],
)  # fmt: skip
def test_a_picture_of_no_duration_of_its_own_takes_its_record_s(
    fake_ffprobe, original, duration_ms, ok, words
) -> None:
    # a Matroska film states no duration of its own: its record's durationMs,
    # its end held to its last packet within a frame step and 0.5 s
    film = {**FILM, "duration": None}
    r = read(fake_ffprobe(video=Video(streams=[film])), original, duration_ms=duration_ms)
    assert r.ok is ok and words in (r.error or ""), r.error


def test_a_group_of_pictures_of_0_never_reads_none_from_windows_off_their_places(
    fake_ffprobe, original
) -> None:
    # one keyframe in the first packets listed (a GOP of 0): every seek lands on
    # it, and the windows that read the start's frames fail
    r = read(fake_ffprobe(video=Video(gop=10_000.0)), original)
    assert (r.ok, r.captions) == (False, None)
    assert len(r.windows) == 6 and all(w.error for w in r.windows[1:])


@pytest.mark.parametrize(
    ("span", "offsets"),
    [
        pytest.param(600.0, [0.0, 60.0, 180.0, 300.0, 420.0, 540.0], id="600"),
        pytest.param(100.0, [0.0, 10.0, 30.0, 50.0, 70.0, 70.0], id="100"),  # 70 s at most
        pytest.param(120.0, [0.0, 12.0, 36.0, 60.0, 84.0, 90.0], id="120"),
        pytest.param(40.0, [0.0, 4.0, 10.0, 10.0, 10.0, 10.0], id="40"),
        pytest.param(8.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], id="8"),  # never before the start
    ],
)
def test_the_tail_rule_is_the_plan_s(span, offsets) -> None:
    # six places, each its plan fraction; the tail caps where they lie, no more
    want = list(zip((0.0, *captions.WINDOWS), offsets, strict=True))
    assert captions.plan(0.0, span) == pytest.approx(want)


def test_a_tail_capped_window_says_its_plan_fraction_on_the_wire(fake_ffprobe, original) -> None:
    from analyzer.katalog import _captions_read

    # 130 s: its 90 % window at 100 s (30 s short of the end), six windows read
    r = read(fake_ffprobe(video=Video(length=130.0)), original)
    body = _captions_read(r)
    assert [w["at"] for w in body["captionWindows"]] == [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    assert [w.aim for w in r.windows][-1] == 100.0 and body["span"] == {"start": 0.0, "end": 130.0}


def test_a_short_picture_whose_places_coincide_is_read_whole_never_merged(fake_ffprobe, original):
    # 100 s: its 70 and 90 % places meet at 70 s - never one window of the two
    r = read(fake_ffprobe(video=Video(length=100.0)), original)
    assert [w.whole for w in r.windows] == [True]


# ------------------------------------------------ the short-picture rule


@pytest.mark.parametrize(
    ("span", "gop", "short"),
    [(89.96, 2.0, True), (119.96, 2.0, False), (119.96, 10.0, True), (599.96, 2.0, False),
     (599.96, 30.0, False), (299.96, 30.0, True), (20.0, 2.0, True)],
)  # fmt: skip
def test_which_picture_is_short(span: float, gop: float, short: bool) -> None:
    assert captions.plan_short(0.0, span, gop, 0.04) is short


def test_a_short_picture_is_read_whole(fake_ffprobe, original) -> None:
    r = read(fake_ffprobe(video=Video(length=90.0)), original)
    assert (r.ok, r.captions) == (True, False)
    got = [(w.label, w.whole, w.frames, w.packets) for w in r.windows]
    assert got == [("whole", True, 2250, 2250)]


def test_a_short_picture_s_captions_between_the_places_are_found(fake_ffprobe, original) -> None:
    # captions at 30-35 s alone: no window of the plan would fall on them
    r = read(fake_ffprobe(video=Video(length=90.0, captions=[[30, 35]])), original)
    assert (r.ok, r.captions) == (True, True) and r.windows[0].whole


def _whole_answer(n: int, packets: int, start: float = 0.0) -> dict:
    return {"frames": frames(n, with_captions=False, start=start),
            "streams": [{"closed_captions": 0, "nb_read_packets": str(packets)}]}  # fmt: skip


@pytest.mark.parametrize(
    ("answer", "words"),
    [
        pytest.param(_whole_answer(2200, 2250), "decoded 2200 frames of 2250 packets",
                     id="fewer-frames"),
        # fewer packets than its span needs (90 s at 0.04 s: 2223)
        pytest.param(_whole_answer(2000, 2000), "fewer than its span", id="short-end"),
        pytest.param(_whole_answer(2200, 2200, start=2.0), "fewer than its span", id="late-start"),
        pytest.param(_whole_answer(2240, 2240, start=0.6), "don't reach the picture's start",
                     id="late-start-enough-packets"),
    ],
)  # fmt: skip
def test_a_whole_read_that_can_t_stand_for_the_windows_fails(
    fake_ffprobe, original, answer, words
) -> None:
    ff = fake_ffprobe(video=Video(length=90.0)).answers(answer, call=6)  # the whole read
    r = read(ff, original)
    assert (r.ok, r.captions) == (False, None)
    assert words in r.windows[0].error


@pytest.mark.parametrize(
    "video",
    [pytest.param(Video(length=90.0, untimed=[[40, 41]]), id="untimed"),
     pytest.param(Video(length=90.0, said={"decode": SAID}), id="stderr")],
)  # fmt: skip
def test_a_whole_read_with_a_frame_out_of_time_or_a_line_on_stderr_fails(
    fake_ffprobe, original, video
) -> None:
    r = read(fake_ffprobe(video=video), original)
    assert (r.ok, r.captions) == (False, None) and r.windows[0].whole and r.windows[0].error


def test_a_whole_read_that_hits_a_bound_fails(fake_ffprobe, original) -> None:
    ff = fake_ffprobe(video=Video(length=90.0)).answers(call=6, signal=signal.SIGXCPU)
    r = read(ff, original)
    assert (r.ok, r.captions) == (False, None)
    assert r.windows[0].error == f"ffprobe exited -{int(signal.SIGXCPU)}"


def test_a_short_hevc_picture_read_whole_is_never_none(fake_ffprobe, original) -> None:
    r = read(fake_ffprobe(video=Video(length=90.0, streams=[HEVC])), original)
    assert (r.ok, r.captions, r.captions_known) == (False, None, False) and r.windows[0].whole


def test_a_whole_read_goes_on_the_wire_marked_whole() -> None:
    from analyzer.katalog import _caption_window

    w = captions.Window(0.0, 0.0, frames=2250, captions=False, start=0.0, end=89.96, whole=True,
                        packets=2253, leading=3)  # fmt: skip
    sent = _caption_window(w)
    assert (sent["whole"], sent["packets"], sent["leading"]) == (True, 2253, 3)


@pytest.mark.parametrize(
    ("packets", "leading", "ok", "words"),
    [pytest.param(2266, 16, True, "", id="16-leading"),
     # more than 16 (or none listed): the keyframes' listing gives None
     pytest.param(2267, None, False, "its leading pictures can't be counted",
                  id="17-leading-none-counted"),
     pytest.param(2251, 0, False, "its whole read decoded 2250 frames of 2251 packets",
                  id="a-packet-not-decoded-none-leading"),
     pytest.param(2250, 0, True, "", id="every-packet"),
     pytest.param(2249, 0, False, "its whole read decoded 2250 frames of 2249 packets",
                  id="more-frames-than-packets")],
)  # fmt: skip
def test_a_whole_read_is_complete_by_the_plan(packets, leading, ok, words) -> None:
    # packets - leading <= frames <= packets, leading at most 16 (the plan's
    # wholeLeadingMax): 2250 frames of a picture of 0-90 s
    look = captions._Look([{"nb_read_packets": str(packets)}],
                          frames(2250, with_captions=False), "", {})  # fmt: skip
    w = captions._whole(look, 0.0, 90.0, leading, 0.54, 0.04)
    assert (w.error is None) is ok, w.error
    if ok:
        assert w.leading == leading
    else:
        assert w.error.startswith(words)


@pytest.mark.parametrize(
    ("video", "ok", "leading"),
    [pytest.param(Video(length=90.0), True, 0, id="from-its-first-keyframe"),
     # 10 packets before its first keyframe at 25 a second, which no decoder
     # outputs: the plan's leading pictures, counted from the keyframes' listing
     pytest.param(Video(length=90.0, first_key=0.4), True, 10, id="10-leading"),
     # 16 at 50 a second (0.32 s: its start within a frame step and 0.5 s)
     pytest.param(Video(length=90.0, fps=50.0, first_key=0.32), True, 16, id="16-leading"),
     # 17: more than the plan's wholeLeadingMax, the whole read is not complete
     pytest.param(Video(length=90.0, fps=50.0, first_key=0.34), False, None, id="17-leading")],
)  # fmt: skip
def test_a_whole_read_counts_its_leading_pictures_as_the_plan_does(
    fake_ffprobe, original, video, ok, leading
) -> None:
    # the plan's wholeComplete: packets - leading <= frames <= packets, leading
    # the packets before the first keyframe (and the first group's packets
    # timed before it), at most 16, as the keyframes' listing shows them
    r = read(fake_ffprobe(video=video), original)
    [w] = r.windows
    assert w.whole and (r.ok, r.captions) == ((True, False) if ok else (False, None)), r.error
    if ok:
        assert (w.leading, w.packets - w.frames) == (leading, leading)
    else:
        assert w.error.startswith("its leading pictures can't be counted")


def test_a_packet_s_side_data_in_the_dump_is_passed_over(fake_ffprobe, original) -> None:
    # a transport stream's packets carry side data, dumped as a section of their own
    video = Video(streams=[HEVC], dump="side", bytes_captions=[[290, 310]])
    assert (read(fake_ffprobe(video=video), original).captions) is True
    r = read(fake_ffprobe(video=Video(streams=[HEVC], dump="sideopen")), original)
    assert r.captions is None
    want = "its packet scan failed: its packet scan's dump has a section inside a packet's side"
    assert all(w.error.startswith(want) for w in r.windows)


@pytest.mark.parametrize(
    "streams",
    [pytest.param([], id="no-streams"),
     pytest.param([{"index": 0, "codec_type": "audio", "codec_name": "aac"}], id="no-video")],
)  # fmt: skip
def test_a_header_probe_that_lists_no_film_is_never_none(fake_ffprobe, original, streams):
    # (the read's own header probe decides: it keeps no stored probe of its own)
    r = read(fake_ffprobe(video=Video(streams=streams)), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error == "no video stream to read"


def test_each_ffprobe_run_has_its_own_bound(fake_ffprobe, original, monkeypatch) -> None:
    # the plan's 180 s of wall time a read, made 1 s here: a window's decode of
    # 2 s fails the read; runs of 0.3 s each, more than 1 s in all, don't
    monkeypatch.setattr(captions, "RUN_TIMEOUT_S", 1.0)
    r = read(fake_ffprobe().answers(call=6, sleep=2.0), original)
    assert (r.ok, r.captions, r.timed_out) == (False, None, True)
    assert r.error == "ffprobe did not answer within 1 s (a read's bound)"
    r = read(fake_ffprobe(video=Video(delay=0.3)), original)  # its runs 0.3 s each
    assert r.ok and r.seconds > 1.0


def test_a_group_of_pictures_of_0_counts_no_keyframe_before(fake_ffprobe, original) -> None:
    r = read(fake_ffprobe(video=Video(gop=10_000.0)), original)
    assert not any(w.keyframe_before for w in r.windows)
    assert all(w.error for w in r.windows[1:])


# ------------------------------------ the plan aligned: the frame step


@pytest.mark.parametrize(
    ("rates", "step", "whence"),
    [pytest.param({"avg_frame_rate": "24000/1001", "r_frame_rate": "24000/1001"}, 1001 / 24000,
                  "avg_frame_rate", id="avg"),
     pytest.param({"avg_frame_rate": "50/1", "r_frame_rate": "25/1"}, 0.02, "avg_frame_rate",
                  id="avg-before-r"),
     pytest.param({"avg_frame_rate": "0/0", "r_frame_rate": "30/1"}, 1 / 30, "r_frame_rate",
                  id="avg-0/0-r"),
     # r_frame_rate only where avg_frame_rate is 0/0: none stated, no plan
     pytest.param({"r_frame_rate": "30/1"}, None, None, id="no-avg-r"),
     pytest.param({"avg_frame_rate": "25.0/1", "r_frame_rate": "25/1"}, None, None,
                  id="an-avg-not-a-ratio"),
     pytest.param({"avg_frame_rate": "0/1", "r_frame_rate": "25/1"}, None,
                  "avg_frame_rate below one frame a second", id="avg-0/1"),
     # digits alone: spaces make it no ratio, and r_frame_rate stands in for 0/0 alone
     pytest.param({"avg_frame_rate": " 25 / 1 ", "r_frame_rate": "25/1"}, None, None,
                  id="spaces"),
     pytest.param({"avg_frame_rate": "0/0", "r_frame_rate": "0/0"}, None, None, id="no-rate"),
     pytest.param({}, None, None, id="none-listed"),
     pytest.param({"avg_frame_rate": "-25/1", "r_frame_rate": "25/0"}, None, None,
                  id="not-a-rate"),
     pytest.param({"avg_frame_rate": "25.5/1", "r_frame_rate": 25}, None, None,
                  id="not-whole-numbers")],
)  # fmt: skip
def test_the_frame_step_is_the_probe_s_rate_alone(rates, step, whence) -> None:
    # what the catalog knows without reading: avg, r where avg is 0/0; none
    # stated, no step (and no plan)
    got = captions.frame_step({**FILM, **rates})
    assert got == ((None, whence) if step is None else (pytest.approx(step), whence))


def test_the_read_s_frame_step_is_its_header_s_rate(fake_ffprobe, original) -> None:
    # frames 25 a second, a header that says 50; one that states no rate has
    # no plan: the read fails, never a step it measured
    for rates, step, whence in (
        ({"avg_frame_rate": "50/1"}, 0.02, "avg_frame_rate"),
        ({"avg_frame_rate": "0/0", "r_frame_rate": "50/1"}, 0.02, "r_frame_rate"),
    ):
        r = read(fake_ffprobe(video=Video(streams=[{**FILM, **rates}])), original)
        assert (r.ok, r.step, r.step_from) == (True, pytest.approx(step), whence), r.error
    none = {**FILM, "avg_frame_rate": "0/0", "r_frame_rate": "0/0"}
    r = read(fake_ffprobe(video=Video(streams=[none])), original)
    assert (r.ok, r.captions, r.unreadable, r.step) == (False, None, True, None)
    assert r.error == (
        "its header probe states no frame rate of its picture (avg_frame_rate, r_frame_rate): "
        "no plan, its windows can't be placed"
    )


# ------------------------- the plan aligned: the start and the length


@pytest.mark.parametrize(
    ("film", "fmt", "ms", "want"),
    [pytest.param({"start_time": "1.000000"}, {"start_time": "0.000000"}, 300_000,
                  (1.0, 301.0, "its record's durationMs"), id="the-picture-s-start-first"),
     pytest.param({}, {"start_time": "0.500000"}, 300_000, (0.5, 300.5, "its record's durationMs"),
                  id="the-format-s-start"),
     pytest.param({}, {}, 300_000, None, id="no-start-no-plan"),
     pytest.param({"start_time": "0.000000", "duration": "250.000000"}, {"duration": "300.000000"},
                  300_000, (0.0, 250.0, "its picture's start_time and duration"),
                  id="the-picture-s-duration-first"),
     pytest.param({"start_time": "0.000000", "duration": "0.000000"}, {}, 300_000,
                  (0.0, 300.0, "its record's durationMs"), id="a-duration-of-0"),
     pytest.param({"start_time": "0.000000"}, {"duration": "300.000000"}, "299000",
                  (0.0, 300.0, "its format's duration"), id="durationMs-a-string"),
     pytest.param({"start_time": "0.000000"}, {"duration": "300.000000"}, True,
                  (0.0, 300.0, "its format's duration"), id="durationMs-a-boolean"),
     pytest.param({"start_time": "0.000000", "tags": {"DURATION": "00:05:00.000000000"}}, {}, None,
                  None, id="a-duration-tag-alone")],
)  # fmt: skip
def test_the_span_is_the_plan_s_in_its_order(film, fmt, ms, want) -> None:
    # the picture's start_time, else the format's, else none (never 0); its
    # duration, else the record's durationMs (a number), else the format's;
    # no tag read for a length
    assert captions.picture_span({"format": fmt}, {**FILM, **film}, ms) == want


def test_a_picture_whose_start_none_states_is_not_read(fake_ffprobe, original) -> None:
    film = {**FILM, "start_time": None}
    r = read(fake_ffprobe(video=Video(streams=[film], format_start=False)), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error.startswith("its records state no start of its picture")
    # the format's start where the picture states none
    r = read(fake_ffprobe(video=Video(streams=[film])), original)
    assert (r.ok, r.span["start"]) == (True, 0.0), r.error


def test_the_probe_s_frame_step_holds_the_span_s_ends(fake_ffprobe, original) -> None:
    # its first keyframe 0.56 s in: within a step of 0.1 s (a header of 10 a
    # second) and 0.5 s, not within the 0.04 s of a header of 25
    ten = {**FILM, "avg_frame_rate": "10/1", "r_frame_rate": "10/1"}
    r = read(fake_ffprobe(video=Video(first_key=0.56, streams=[ten])), original)
    assert (r.ok, r.step, r.windows[0].start) == (True, pytest.approx(0.1), 0.56), r.error
    r = read(fake_ffprobe(video=Video(first_key=0.56)), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error.startswith("its picture starts at 0.000 s as its records state it")


# ------------------------------- the plan aligned: both ends of the span


@pytest.mark.parametrize(
    ("first_key", "ok"), [(0.0, True), (0.52, True), (0.56, False), (5.0, False)]
)
def test_the_span_s_start_is_held_as_its_end_is(fake_ffprobe, original, first_key, ok) -> None:
    # its records state its picture from 0: its first frame within a frame
    # step (0.04 s) and 0.5 s of it, else the read fails
    r = read(fake_ffprobe(video=Video(first_key=first_key)), original)
    if ok:
        assert (r.ok, r.span["start"]) == (True, first_key), r.error
    else:
        assert (r.ok, r.captions, r.unreadable) == (False, None, True)
        assert r.error == (
            "its picture starts at 0.000 s as its records state it (its picture's start_time "
            f"and duration), its first frame decoded is at {first_key:.3f} s: more than a frame "
            "step (0.040 s) and 0.5 s apart"
        )


@pytest.mark.parametrize(
    ("first", "last", "ok"),
    [(0.52, 89.96, True), (0.56, 89.96, False), (0.0, 89.48, True), (0.0, 89.44, False)],
)
def test_a_whole_read_reaches_the_span_within_the_plan_s_tolerance(first, last, ok) -> None:
    # a short picture of 0-90 s at 25 a second: its whole read's frames from
    # within a frame step and 0.5 s of its start to within them of its end
    # (the packets before its first frame lead its first keyframe: they count
    # as its packets, and as its leading ones)
    n, lead = round((last - first) * 25) + 1, round(first * 25)
    look = captions._Look([{"nb_read_packets": str(n + lead)}],
                          frames(n, with_captions=False, start=first), "", {})  # fmt: skip
    w = captions._whole(look, 0.0, 90.0, lead, 0.04 + captions.END_SLACK_S, 0.04)
    assert (w.error is None) is ok, w.error
    if not ok:
        want = f"its frames ({first:.3f}-{last:.3f} s) don't reach the picture's"
        assert w.error.startswith(want)


# -------------------------------------- the plan aligned: the place slack


@pytest.mark.parametrize(("gop", "slack"), [(2.0, 10.0), (9.5, 10.0), (12.0, 12.5), (0.0, 0.54)])
def test_the_place_slack_is_the_plan_s(gop, slack) -> None:
    # max(10 s, the GOP and 0.5 s); of a GOP of 0 a frame step and 0.5 s
    assert captions.place_slack(gop, 0.04) == pytest.approx(slack)


def test_a_window_further_from_its_place_than_its_slack_fails() -> None:
    # a GOP of 2 s: 9 s after its place is within 10 s, 12 s isn't (the
    # earlier 15 s took it); 12 s before it without keyframeBefore isn't either
    slack = captions.place_slack(2.0, 0.04)

    def at(t: float) -> captions._Look:
        decoded = frames(48, with_captions=False, start=t)
        return captions._Look([{"closed_captions": 0}], decoded, "", {})

    assert captions._window(0.1, 60.0, at(69.0), 48, [], (False, ""), slack).error is None
    w = captions._window(0.1, 60.0, at(72.0), 48, [], (False, ""), slack)
    assert w.error == ("its first frame is at 72.000 s, 12.0 s from where it was placed "
                       "(60.000 s): it landed elsewhere")  # fmt: skip
    w = captions._window(0.1, 60.0, at(48.0), 48, [], (False, "no keyframe"), slack)
    assert w.error.startswith("its first frame is at 48.000 s, 12.0 s before where it was placed")


# ------------------------------------------- the closed lists' cover rule

COVER_MARKS = {"filename": "cover.jpg", "mimetype": "image/jpeg"}


@pytest.mark.parametrize(
    ("st", "cover"),
    [pytest.param({"codec_name": "mjpeg", "disposition": {"attached_pic": 1}}, True,
                  id="the-disposition"),
     pytest.param({"codec_name": "mjpeg", "tags": COVER_MARKS}, True, id="both-tags"),
     pytest.param({"codec_name": "mjpeg", "tags": {"filename": "cover.jpg"}}, False,
                  id="a-filename-alone"),
     pytest.param({"codec_name": "mjpeg", "tags": {"mimetype": "image/jpeg"}}, False,
                  id="a-mimetype-alone"),
     pytest.param({"codec_name": "webp", "tags": {**COVER_MARKS, "mimetype": "image/webp"}}, False,
                  id="webp-its-codec-always-tested"),
     pytest.param({"codec_name": "h264", "disposition": {"attached_pic": 1}}, False,
                  id="h264-marked"),
     pytest.param({"codec_name": "png", "tags": {**COVER_MARKS, "mimetype": "font/ttf"}}, False,
                  id="a-mimetype-not-an-image"),
     pytest.param({"codec_name": "png", "tags": {**COVER_MARKS, "mimetype": " IMAGE/PNG "}}, True,
                  id="its-mimetype-in-capitals"),
     pytest.param({"codec_name": "png", "tags": {"filename": "", "mimetype": "image/png"}}, True,
                  id="an-empty-filename-tag"),
     pytest.param({"codec_name": "bmp", "tags": {"filename": "a.bmp", "mimetype": ""}}, False,
                  id="an-empty-mimetype-no-disposition"),
     pytest.param({"codec_name": "mjpeg", "disposition": {"attached_pic": 1}, "nb_frames": "1"},
                  True, id="one-frame"),
     pytest.param({"codec_name": "mjpeg", "disposition": {"attached_pic": 1}, "nb_frames": "2"},
                  False, id="two-frames"),
     pytest.param({"codec_name": "mjpeg", "disposition": {"attached_pic": 1}, "nb_frames": "5",
                   "tags": {"NUMBER_OF_FRAMES": "1"}}, False, id="five-frames-a-tag-of-one"),
     # a count there that can't be read is no cover's
     pytest.param({"codec_name": "mjpeg", "disposition": {"attached_pic": 1}, "nb_frames": " 5"},
                  False, id="a-count-not-in-digits"),
     pytest.param({"codec_name": "mjpeg", "disposition": {"attached_pic": 1}, "nb_frames": "2.0"},
                  False, id="a-count-of-2.0"),
     pytest.param({"codec_name": "mjpeg", "tags": {**COVER_MARKS, "DURATION": "00:00:05.0"}},
                  False, id="a-duration-tag"),
     pytest.param({"codec_name": "mjpeg", "tags": {**COVER_MARKS, "DURATION-eng": "00:00:05.0"}},
                  False, id="a-duration-tag-of-a-language")],
)  # fmt: skip
def test_the_cover_rule_is_the_closed_lists_one(st, cover) -> None:
    # marked (the disposition, or both tags, never one alone), of a cover's
    # codec always (its mimetype, where it has one, image/*), at most one
    # frame, no length of its own
    assert captions.attached_picture({"codec_type": "video", **st}) is cover


# ------------------- the rulings every side carries (the plan's second look)


@pytest.mark.parametrize(
    ("nb", "cover"),
    [pytest.param("1e3", False, id="1e3"), pytest.param(" 900", False, id="space-900"),
     pytest.param("+900", False, id="plus-900"), pytest.param("0x384", False, id="hex"),
     pytest.param(["900"], False, id="a-list"), pytest.param(" 1", False, id="space-1"),
     pytest.param(True, False, id="a-boolean"), pytest.param(float("nan"), False, id="nan"),
     pytest.param(None, False, id="null"), pytest.param("-1", False, id="minus-1"),
     pytest.param("inf", False, id="inf"), pytest.param(" 5", False, id="space-5"),
     pytest.param("1", True, id="1"), pytest.param(1, True, id="the-number-1")],
)  # fmt: skip
def test_a_cover_s_frame_count_it_can_t_read_means_no_cover(nb, cover) -> None:
    st = {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}}
    assert captions.attached_picture({**st, "nb_frames": nb}) is cover


@pytest.mark.parametrize(
    ("tags", "cover"),
    [pytest.param({"NUMBER_OF_FRAMES-eng": "5"}, False, id="5-a-language-s"),
     pytest.param({"NUMBER_OF_FRAMES-eng": "1"}, True, id="1-a-language-s"),
     pytest.param({"NUMBER_OF_FRAMES-eng": "0x1"}, False, id="0x1-a-language-s")],
)  # fmt: skip
def test_a_count_tag_is_matched_by_its_start(tags, cover) -> None:
    st = {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}}
    assert captions.attached_picture({**st, "tags": tags}) is cover


@pytest.mark.parametrize(
    ("mark", "cover"),
    [pytest.param(1, True, id="one"), pytest.param(True, True, id="true"),
     pytest.param(1.0, False, id="one-float"), pytest.param("1", False, id="text-1"),
     pytest.param("0", False, id="text-0"), pytest.param(0, False, id="zero"),
     pytest.param(2, False, id="two")],
)  # fmt: skip
def test_a_disposition_s_mark_is_the_number_1_or_true(mark, cover) -> None:
    st = {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": mark}}
    assert captions.attached_picture(st) is cover


@pytest.mark.parametrize(
    ("rates", "words"),
    [pytest.param({"avg_frame_rate": "1/2", "r_frame_rate": "25/1"},
                  "avg_frame_rate below one frame a second", id="avg-1/2"),
     pytest.param({"avg_frame_rate": "0/0", "r_frame_rate": "999/1000"},
                  "r_frame_rate below one frame a second", id="r-999/1000")],
)  # fmt: skip
def test_a_frame_rate_below_one_a_second_gives_no_plan(rates, words) -> None:
    assert captions.frame_step({**FILM, **rates}) == (None, words)
    assert captions.frame_step({**FILM, "avg_frame_rate": "1/1"}) == (1.0, "avg_frame_rate")


def test_a_read_of_a_frame_rate_below_one_a_second_fails(fake_ffprobe, original) -> None:
    r = read(fake_ffprobe(video=Video(streams=[{**FILM, "avg_frame_rate": "1/2"}])), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert "avg_frame_rate below one frame a second: no plan" in r.error


def test_a_group_of_pictures_longer_than_its_listing_is_not_known(
    fake_ffprobe, original, monkeypatch
) -> None:
    # a GOP longer than 1200 frame steps (its packets' times further apart
    # than its stated rate): not known
    monkeypatch.setattr(captions._Runner, "keys", lambda self, step, a, b: (1201 * step, 0))
    r = read(fake_ffprobe(), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert "longer than 1200 frame steps" in r.error
    monkeypatch.setattr(captions._Runner, "keys", lambda self, step, a, b: (1200 * step, 0))
    assert "longer than 1200" not in (read(fake_ffprobe(), original).error or "")


def test_a_whole_read_needs_about_its_span_s_packets() -> None:
    # packets x the frame step at least its span less a frame step and 0.5 s:
    # 90 s at 0.04 s, 2237 packets at least
    assert captions.whole_packets_needed(0.0, 90.0, 0.04) == pytest.approx(2236.5)
    assert captions.whole_packets_enough(2237, 0.0, 90.0, 0.04)
    assert not captions.whole_packets_enough(2236, 0.0, 90.0, 0.04)
    shy = captions._Look([{"nb_read_packets": "2236"}], frames(2236, with_captions=False), "", {})
    w = captions._whole(shy, 0.0, 90.0, 0, 0.54, 0.04)
    assert w.error.startswith("its whole read has 2236 packets, fewer than its span")


def test_a_number_that_is_not_finite_is_not_stated() -> None:
    assert [captions._float(v) for v in ("nan", "inf", "-inf", "1.5")] == [None, None, None, 1.5]
    assert captions._pts({"pts_time": "nan", "best_effort_timestamp_time": "2.0"}) == 2.0
    for text in ('{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
        with pytest.raises(json.JSONDecodeError):
            captions.load_json(text)


def test_an_ffprobe_answer_holding_nan_fails_its_read(fake_ffprobe, original) -> None:
    # the start's decode answers NaN for a frame's time: no JSON it takes
    ff = fake_ffprobe().answers('{"frames": [{"pts_time": NaN}], "streams": []}', call=2)
    r = read(ff, original)
    assert (r.ok, r.captions) == (False, None)
    assert "NaN is no number it states" in (r.error or "") + " ".join(
        w.error or "" for w in r.windows
    )


# --------------- the second look: a verdict rests on runs checked for what they must be


def _call_of(fake_ffprobe, original: Path, entries: str, video: Video | None = None) -> int:
    """Which run of a read of the made-up video lists entries (its number, from 1)."""
    ff = fake_ffprobe(video=video or Video())
    read(ff, original)
    return next(n for n, run in enumerate(ff.runs, 1)
                if run[run.index("-show_entries") + 1] == entries)  # fmt: skip


@pytest.mark.parametrize(
    "answer",
    [pytest.param("", id="empty"), pytest.param('{"frames": [', id="no-listing"),
     pytest.param("K__\n___\n", id="flags-without-times"),
     pytest.param("".join(f"{i * 0.04:.6f},{'K__' if i % 50 == 0 else '___'}\n"
                          for i in range(600)), id="600-of-1200-short-of-its-end")],
)  # fmt: skip
def test_a_keyframes_listing_that_is_no_listing_fails_the_read(fake_ffprobe, original, answer):
    # empty, no listing, or fewer packets than asked that don't reach the
    # picture's end (exit 0, nothing on stderr): not known, never a gop of 0
    keys = _call_of(fake_ffprobe, original, "packet=pts_time,flags")
    r = read(fake_ffprobe().answers(answer, call=keys), original)
    assert (r.ok, r.captions, r.unreadable, r.gop) == (False, None, True, None)
    assert r.error.startswith("the listing of its keyframes ")


def test_a_short_picture_s_listing_of_every_packet_is_its_listing(fake_ffprobe, original) -> None:
    # 20 s at 25 a second: 500 packets, fewer than 1200, reaching its end
    ff = fake_ffprobe(video=Video(length=20.0))
    r = read(ff, original)
    assert (r.ok, r.captions, r.gop) == (True, False, pytest.approx(2.0)) and r.windows[0].whole


def test_a_window_s_keyframe_scan_that_is_no_listing_fails_the_window(fake_ffprobe, original):
    # seeks that land where they aim, in groups of 6.1 s: the 10 % window
    # looks for its keyframe among the packets from its point, and is
    # answered something that is no listing of flags
    video = Video(lands="after", gop=6.1)
    ff = fake_ffprobe(video=video)
    read(ff, original)
    scans = [n for n, run in enumerate(ff.runs, 1) if run[run.index("-show_entries") + 1]
             == "packet=flags"]  # fmt: skip
    r = read(fake_ffprobe(video=video).answers("K__\nnot a flag\n", call=scans[0]), original)
    assert (r.ok, r.captions) == (False, None)
    w = r.windows[1]
    assert w.error.startswith("no keyframe among the first 1200 video packets")
    assert "its keyframes' listing is no listing of packets" in w.said


@pytest.mark.parametrize(
    "answer",
    [pytest.param("600.000000,600.000000,0.040000\nnot\n", id="junk"),
     pytest.param('{"packets": []}', id="no-listing")],
)  # fmt: skip
def test_an_end_s_listing_that_is_no_listing_fails_the_read(fake_ffprobe, original, answer):
    end = _call_of(fake_ffprobe, original, "packet=pts_time,dts_time,duration_time")
    r = read(fake_ffprobe().answers(answer, call=end), original)
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error.startswith("the listing of its end is no listing of packets")


def test_a_listing_from_a_seek_that_is_no_listing_is_no_proof() -> None:
    # keyframeBefore only lets a window lie further back: of a listing that
    # is no listing, it is never granted
    class Run(captions._Runner):
        def __init__(self) -> None:
            self.stream = 0

        def _run(self, args):  # type: ignore[override]
            return "58.000000,58.000000,K__\nnot a packet\n61.000000,61.000000,___\n", ""

    look = captions._Look([{}], frames(48, with_captions=False, start=58.0), "", {})
    assert Run().keyframe_before(60.0, look, 13.0) == (
        False, "its packets' listing is no listing of packets")


def test_a_csv_line_s_trailing_empty_fields_are_none_of_its_own() -> None:
    # 7.1 writes a transport stream's packet with an empty field for its side
    # data: "1.480000,K__,"
    assert captions._KEYS_LINE.fullmatch(captions._fields("1.480000,K__,\n"))
    assert captions._END_LINE.fullmatch(captions._fields("6.880000,6.720000,0.040000,"))
    assert captions._LIST_LINE.fullmatch(captions._fields("1.920000,N/A,K__"))
    assert not captions._KEYS_LINE.fullmatch(captions._fields("1.480000,K__,x"))


# ------------- the lead's ruling of 11-18, as amended: what the verdict rests on


def test_a_short_picture_s_replaced_start_window_is_not_what_its_none_rests_on(
    fake_ffprobe, original
) -> None:
    # 90 s: its start window's decode (run 2) says something on stderr, and
    # fails; it only measured the span's start, and is replaced by the whole
    # read, which decodes the same first packets and says nothing: "none"
    # rests on the whole run alone
    start = {"frames": frames(48, with_captions=False), "streams": [{"closed_captions": 0}]}
    ff = fake_ffprobe(video=Video(length=90.0)).answers(start, err=SAID, call=2)
    r = read(ff, original)
    assert (r.ok, r.captions) == (True, False), r.error
    [w] = r.windows
    assert w.whole and w.error is None and w.said == ""
    assert "error while decoding" not in (r.note or "")


def test_a_whole_run_that_said_anything_is_not_known(fake_ffprobe, original) -> None:
    # the whole run (run 6) answers every frame, and a line on stderr
    args = captions.whole_args()
    whole = _call_of(fake_ffprobe, original, args[args.index("-show_entries") + 1],
                     Video(length=90.0))  # fmt: skip
    ff = fake_ffprobe(video=Video(length=90.0)).answers(_whole_answer(2250, 2250), err=SAID,
                                                        call=whole)  # fmt: skip
    r = read(ff, original)
    assert (r.ok, r.captions) == (False, None) and r.windows[0].whole
    assert r.windows[0].error.startswith("ffprobe found no captions and said: [h264 @ 0x1]")
