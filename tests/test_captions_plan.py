"""The captions read's window plan held against the plan's reference: the
caption-window plan, the same on every side that reads captions. The
reference is data the read is checked against, not configuration: the read
keeps its constants in its own code and never reads a plan file; only this
test does, from beside it (never a path the environment gives), pinned by its
sha256: a copy that differs from the pinned digest fails this test, not a
read.

The reference is the plan file vendored beside this test (PLAN_FILE, from the
library's schemas, library/v2/caption-windows.json, pinned by PLAN_SHA256):
the start, plus 10, 30, 50, 70 and 90 % of the picture's own presentation
span, each at most the tail short of the end; the slowest frame rate that
gives a plan (minRate) and how a frame step is read (words.frameStep); which
stream is the picture (words.picture, the closed lists' cover rule); the
window's packets and least frames; the end tolerance; where a window may lie
from its place; the gop a read reports and how long it may be (gopPackets);
the short-picture rule and when its whole read is complete
(words.wholeComplete); the bounds every run reads under. Every key of the
file is one this test knows: a plan with another fails it."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import pytest
from captions_fake import FakeProbe, Video, frames

from analyzer.katalog import _captions_read
from analyzer.pipelines import captions

PLAN_FILE = "caption-windows.json"  # the vendored reference, beside this file
PLAN_SHA256 = "02087f1b50f7c6854f8e42ce2526063606657c58aef9942d151005279aa03b87"
# the plan's keys, and its words', every one of them read below
KEYS = {"schema", "note", "start", "at", "minRate", "endTolerance", "tailS", "packets",
        "minFrames", "packetsShare", "gopPackets", "placeSlack", "shortReach", "wholeLeadingMax",
        "bounds", "words", "rule"}  # fmt: skip
WORDS = {"picture", "end", "frameStep", "wholeComplete"}
FILM = {"index": 0, "codec_type": "video", "codec_name": "h264", "height": 1080}


def reference(here: Path | None = None) -> dict:
    """The vendored plan, its digest checked: a copy that differs fails."""
    data = ((here or Path(__file__).parent) / PLAN_FILE).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PLAN_SHA256, "not the pinned plan"
    plan = json.loads(data)
    assert plan["schema"] == "zaentrum.library.caption-windows/1"
    return {**plan, "at": ((0.0,) if plan["start"] else ()) + tuple(plan["at"]),
            "tail_s": float(plan["tailS"])}  # fmt: skip


@pytest.fixture
def original(tmp_path: Path) -> Path:
    p = tmp_path / "Film (2024).mkv"
    with p.open("wb") as f:
        f.truncate(600 * 25 * 20_000)
    return p


def test_a_copy_of_the_plan_that_differs_fails(tmp_path) -> None:
    ours = (Path(__file__).parent / PLAN_FILE).read_bytes()
    assert b'"tailS": 30' in ours
    (tmp_path / PLAN_FILE).write_bytes(ours.replace(b'"tailS": 30', b'"tailS": 31'))
    with pytest.raises(AssertionError, match="not the pinned plan"):
        reference(tmp_path)


def test_every_key_of_the_plan_is_one_the_read_knows() -> None:
    plan = json.loads((Path(__file__).parent / PLAN_FILE).read_bytes())
    assert set(plan) == KEYS and set(plan["words"]) == WORDS
    reference()  # and the digest is the pinned one


def test_the_read_keeps_the_plan_s_values() -> None:
    ref = reference()
    assert captions.MIN_RATE == ref["minRate"]
    assert captions.PACKETS == ref["packets"] == ref["shortReach"]["packets"]
    share = ref["packets"] * ref["packetsShare"]
    assert captions.MIN_WINDOW_FRAMES == max(ref["minFrames"], share)
    assert captions.KEY_SCAN_PACKETS == ref["gopPackets"]
    tolerance = ref["endTolerance"]
    assert (tolerance["frameSteps"], captions.END_SLACK_S) == (1, tolerance["s"])
    slack = ref["placeSlack"]
    ours = (captions.PLACE_SLACK_S, captions.PLACE_MARGIN_S, captions.BACK_SLACK_S)
    assert ours == (slack["s"], slack["gopPlus"], slack["backS"])
    reach = ref["shortReach"]
    assert reach["gop"] is True and captions.REACH_MARGIN_S == reach["s"]
    assert captions.WHOLE_LEADING_MAX == ref["wholeLeadingMax"]
    b = ref["bounds"]
    ours = (captions.THREADS, captions.LIMIT_AS_BYTES, captions.LIMIT_CPU_S, captions.RUN_TIMEOUT_S)
    assert ours == (b["threads"], b["addressSpaceMiB"] << 20, b["cpuS"], b["timeoutS"])


def test_the_read_s_place_slack_and_short_reach_are_the_plan_s() -> None:
    ref = reference()
    slack, tolerance = ref["placeSlack"], ref["endTolerance"]
    for gop in (0.0, 2.0, 9.5, 12.0, 25.0):
        if gop:
            want = max(slack["s"], gop + slack["gopPlus"])
        else:
            want = tolerance["frameSteps"] * 0.04 + tolerance["s"]
        assert captions.place_slack(gop, 0.04) == pytest.approx(want)
    # the short reach: two places closer than packets x frameStep + gop + 0.5 s
    for span, gop in ((100.0, 2.0), (120.0, 2.0), (150.0, 10.0), (300.0, 2.0)):
        places = sorted(place for _at, place in captions.plan(0.0, span))
        reach = ref["shortReach"]["packets"] * 0.04 + gop + ref["shortReach"]["s"]
        want = any(b - a < reach for a, b in pairwise(places))
        assert captions.plan_short(0.0, span, gop, 0.04) == want


def test_the_read_s_places_are_the_plan_s() -> None:
    ref = reference()
    assert (0.0, *captions.WINDOWS) == tuple(ref["at"])
    assert captions.TAIL_S == ref["tail_s"]


@pytest.mark.parametrize("span", [599.96, 7200.0, 300.0, 120.0, 100.0])
def test_the_read_places_each_window_as_the_plan_does(span: float) -> None:
    ref = reference()
    start = 12.5
    end = start + span
    places = [start if at == 0 else max(start, min(start + at * span, end - ref["tail_s"]))
              for at in ref["at"]]  # fmt: skip
    want = list(zip(ref["at"], places, strict=True))
    assert captions.plan(start, end) == want  # six, at their fractions, never merged, to the bit


def test_places_exactly_a_reach_apart_are_not_closer() -> None:
    # 320 s at a step of 1/16 s, a GOP of 28.5 s: places 32 s apart, a reach
    # of 48 x 1/16 + 28.5 + 0.5 = 32 s: not closer, read in windows; a GOP of
    # 28.6 s reaches further: read whole
    assert captions.plan_short(0.0, 320.0, 28.5, 0.0625) is False
    assert captions.plan_short(0.0, 320.0, 28.6, 0.0625) is True


@pytest.mark.parametrize(
    ("start", "length", "gop_frames", "short"),
    [pytest.param(1.4, 150.0, 300, True, id="24fps-start-1.4-150s-gop-300"),
     pytest.param(0.0, 150.0, 300, False, id="24fps-start-0-150s-gop-300"),
     pytest.param(1.4, 150.001, 300, False, id="24fps-start-1.4-150.001s-gop-300")],
)  # fmt: skip
def test_the_short_rule_is_decided_over_the_places_themselves(start, length, gop_frames, short):
    # avg_frame_rate 24/1, start_time 1.400000, duration 150.000000, a gop of
    # 300 frames: its places 1.4 and 16.4 lie 14.999999999999998 s apart, a
    # reach of 15 s - read whole, as the record logic's plan_short decides it
    # over plan_places (an offset from the start, 15.000000000000002, would
    # have said six windows)
    step = 1 / 24
    end = start + length
    gop = gop_frames * step
    assert captions.plan_short(start, end, gop, step) is short
    places = sorted(p for _at, p in captions.plan(start, end))
    reach = 48 * step + gop + 0.5
    assert any(b - a < reach for a, b in pairwise(places)) is short


# ------------------------------------------- minRate and words.frameStep


def test_a_rate_below_the_plan_s_min_rate_gives_no_plan() -> None:
    # minRate frames a second gives a plan, one below it none (0/1 among them)
    rate = reference()["minRate"]
    assert rate == 1  # (the read's words say "one frame a second")
    step = captions.frame_step({**FILM, "avg_frame_rate": f"{rate * 1000}/1000"})
    assert step == (pytest.approx(1 / rate), "avg_frame_rate")
    for slow in (f"{rate * 1000 - 1}/1000", "0/1"):
        step, why = captions.frame_step({**FILM, "avg_frame_rate": slow, "r_frame_rate": "25/1"})
        assert (step, why) == (None, "avg_frame_rate below one frame a second"), slow
    # r_frame_rate is held to it as well, where it stands in for 0/0
    assert captions.frame_step({**FILM, "avg_frame_rate": "0/0", "r_frame_rate": "1/2"})[0] is None
    # a step too large for a float is below any rate
    assert captions.frame_step({**FILM, "avg_frame_rate": "1/" + "9" * 400})[0] is None


@pytest.mark.parametrize(
    "avg",
    [pytest.param(" 25/1", id="a-space-before"), pytest.param("25/1 ", id="a-space-after"),
     pytest.param("25 / 1", id="spaces-about-the-slash"), pytest.param("+25/1", id="a-sign"),
     pytest.param("25.0/1", id="a-point"), pytest.param("25", id="no-slash"),
     pytest.param("25/1\n", id="a-newline"), pytest.param("٢٥/1", id="arabic-indic"),
     pytest.param(25, id="a-number"), pytest.param(None, id="null"),
     # more digits than Python 3.11 reads of a text, whatever this one reads
     pytest.param("9" * 4301 + "/1", id="more-digits-than-python-reads"),
     pytest.param("25/" + "1" * 4301, id="a-den-of-more-digits-than-python-reads"),
     # a rate past any a float holds: a step of 0 is no step
     pytest.param("1" + "0" * 400 + "/1", id="a-rate-past-a-float")],
)  # fmt: skip
def test_an_avg_frame_rate_that_is_no_ratio_of_digits_gives_no_plan_and_no_fallback(avg) -> None:
    # the plan's frameStep: num/den of digits alone; r_frame_rate stands in
    # for the text 0/0 alone, never for a malformed avg_frame_rate
    rates = {"avg_frame_rate": avg, "r_frame_rate": "25/1"}
    assert captions.frame_step({**FILM, **rates}) == (None, None)
    assert captions.frame_step({**FILM, "r_frame_rate": "25/1"}) == (None, None)  # missing


@pytest.mark.parametrize(
    ("avg", "step", "whence"),
    [pytest.param("0/0", 0.04, "r_frame_rate", id="exactly-0/0"),
     pytest.param("00/0", None, None, id="00/0-is-not-the-text-0/0"),
     pytest.param("0/00", None, None, id="0/00-is-not-the-text-0/0"),
     pytest.param(" 0/0", None, None, id="a-space-and-0/0")],
)  # fmt: skip
def test_r_frame_rate_stands_in_where_avg_frame_rate_is_exactly_0_0(avg, step, whence) -> None:
    got = captions.frame_step({**FILM, "avg_frame_rate": avg, "r_frame_rate": "25/1"})
    assert got == ((None, None) if step is None else (pytest.approx(step), whence))


def test_a_read_of_a_malformed_avg_frame_rate_fails_without_a_plan(fake_ffprobe, original) -> None:
    film = {**FILM, "avg_frame_rate": "25 / 1", "r_frame_rate": "25/1"}
    r = captions.read(str(original), ffprobe=str(fake_ffprobe(video=Video(streams=[film])).path))
    assert (r.ok, r.captions, r.unreadable, r.step) == (False, None, True, None)
    assert r.error.startswith("its header probe states no frame rate of its picture")


# -------------------------------------------------- words.picture: covers

COVER = {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}}


@pytest.mark.parametrize(
    ("st", "cover"),
    [pytest.param(COVER, True, id="attached_pic-1"),
     pytest.param({**COVER, "disposition": {"attached_pic": True}}, True, id="attached_pic-true"),
     pytest.param({**COVER, "disposition": {"attached_pic": 1.0}}, False, id="attached_pic-1.0"),
     pytest.param({**COVER, "disposition": {"attached_pic": "1"}}, False, id="attached_pic-text"),
     pytest.param({**COVER, "nb_frames": "1"}, True, id="nb_frames-1"),
     pytest.param({**COVER, "nb_frames": 1}, True, id="nb_frames-the-number-1"),
     pytest.param({**COVER, "nb_frames": "1.0"}, False, id="nb_frames-1.0-unreadable"),
     pytest.param({**COVER, "nb_frames": 1.0}, False, id="nb_frames-the-number-1.0-unreadable"),
     pytest.param({**COVER, "nb_frames": "01"}, True, id="nb_frames-01"),
     pytest.param({**COVER, "nb_frames": "2"}, False, id="nb_frames-2"),
     pytest.param({**COVER, "nb_frames": "0" * 4300 + "1"}, False,
                  id="nb_frames-of-more-digits-than-python-reads"),
     pytest.param({**COVER, "nb_frames": "0" * 4299 + "1"}, True, id="nb_frames-of-4300-digits"),
     pytest.param({**COVER, "tags": {"NUMBER_OF_FRAMES-eng": "1"}}, True, id="a-count-tag-of-1"),
     pytest.param({**COVER, "tags": {"NUMBER_OF_FRAMES-eng": "1.0"}}, False,
                  id="a-count-tag-unreadable"),
     pytest.param({**COVER, "tags": {"number_of_frames": "2"}}, False, id="a-count-tag-of-2"),
     pytest.param({**COVER, "tags": {"DURATION-eng": "00:00:00.04"}}, False,
                  id="a-tag-that-starts-DURATION"),
     pytest.param({**COVER, "tags": {"mimetype": "image/png", "MIMETYPE": "font/ttf"}}, False,
                  id="tags-that-clash-by-case"),
     pytest.param({**COVER, "tags": {"mimetype": "image/png", "MIMETYPE": "image/png"}}, True,
                  id="tags-the-same-in-two-cases"),
     pytest.param({**COVER, "tags": {"mimetype": ""}}, False, id="an-empty-mimetype")],
)  # fmt: skip
def test_the_cover_rule_is_the_plan_s_words(st, cover) -> None:
    # words.picture: the attached_pic disposition as the number 1 or true;
    # every count in digits alone, one there that can't be read no cover's;
    # tags that start NUMBER_OF_FRAMES and DURATION
    assert "digits alone" in reference()["words"]["picture"]
    assert captions.attached_picture(st) is cover


# ------------------------------------------------------ gop and gopPackets


def test_a_read_reports_its_gop(fake_ffprobe, original) -> None:
    # a keyframe every 2 s: the read's gop is 2 s, and it goes on the wire
    r = captions.read(str(original), ffprobe=str(fake_ffprobe(video=Video(gop=2.0)).path))
    assert (r.ok, r.captions, r.gop) == (True, False, pytest.approx(2.0))
    assert _captions_read(r)["gop"] == pytest.approx(2.0)
    # one keyframe among the first packets: a gop of 0, said as 0
    r = captions.read(str(original), ffprobe=str(fake_ffprobe(video=Video(gop=10_000.0)).path))
    assert r.gop == 0.0 and _captions_read(r)["gop"] == 0.0


def test_a_gop_above_gop_packets_frame_steps_is_not_known(fake_ffprobe, original) -> None:
    # a listing of more lines than the gopPackets asked for (keyframes 1250
    # packets apart, longer than gopPackets x frameStep) is no listing of the
    # first gopPackets packets: the read is not known, and says no gop
    n = reference()["gopPackets"]
    listing = "\n".join(f"{i * 0.04:.6f},{'K__' if i in (0, n + 50) else '___'}"
                         for i in range(n + 60))  # fmt: skip
    keys = _kinds(fake_ffprobe(), original).index("keys") + 1  # which run lists them
    ff = fake_ffprobe().answers(listing, call=keys)
    r = captions.read(str(original), ffprobe=str(ff.path))
    assert (r.ok, r.captions, r.unreadable) == (False, None, True)
    assert r.error == (
        f"the listing of its keyframes has {n + 60} packets: not the first {n}, "
        "nor every packet of its picture"
    )
    assert "gop" not in _captions_read(r)


def _kinds(ff: FakeProbe, original: Path) -> list[str]:
    """The kinds of the runs a read of the made-up video makes, in order."""
    captions.read(str(original), ffprobe=str(ff.path))
    out = []
    for run in ff.runs:
        entries = run[run.index("-show_entries") + 1]
        out.append("header" if entries.startswith("format=") else
                   "keys" if entries == "packet=pts_time,flags" else "other")  # fmt: skip
    return out


# --------------------------------------------------- words.wholeComplete


def test_a_whole_read_has_every_picture_it_decoded_timed() -> None:
    # every picture it decoded has a time: one without fails the whole read
    assert "every picture it decoded has a time" in reference()["words"]["wholeComplete"]
    decoded = frames(2250, with_captions=False)
    del decoded[1000]["pts_time"]
    look = captions._Look([{"nb_read_packets": "2250"}], decoded, "", {})
    w = captions._whole(look, 0.0, 90.0, 0, 0.54, 0.04)
    assert w.error == "1 of its 2250 frames have no presentation time: it can't place them"


def test_a_whole_read_s_packets_cover_the_plan_s_span_within_the_tolerance() -> None:
    # packets x frameStep >= end - start - endTolerance: 90 s at 0.04 s and a
    # tolerance of 0.54 s, 2237 packets at least
    ref = reference()
    assert "packets x frameStep >= end - start - endTolerance" in ref["words"]["wholeComplete"]
    tolerance = ref["endTolerance"]["frameSteps"] * 0.04 + ref["endTolerance"]["s"]
    least = next(n for n in range(2200, 2300) if n * 0.04 >= 90.0 - tolerance)
    assert least == 2237
    assert captions.whole_packets_enough(least, 0.0, 90.0, 0.04)
    assert not captions.whole_packets_enough(least - 1, 0.0, 90.0, 0.04)


# --------------------------------------------------- no number that isn't


def test_nan_and_infinity_are_no_number_stated() -> None:
    assert "A number that is NaN or Infinity is no number stated." in reference()["rule"]
    for text in ('{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}'):
        with pytest.raises(json.JSONDecodeError):
            captions.load_json(text)
    assert [captions._float(v) for v in ("nan", "inf", "-Infinity")] == [None, None, None]
    assert captions.picture_span({"format": {"duration": "nan"}},
                                 {**FILM, "start_time": "0.000000", "duration": "inf"},
                                 float("nan")) is None  # fmt: skip
