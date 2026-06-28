"""Unit tests for the subtitle text-pattern detectors. The ffmpeg/ffprobe
calls aren't tested here — they're covered by live runs in the cluster.
We test only the pure functions: SRT parsing and pattern matching."""

from __future__ import annotations

from analyzer.pipelines.subtitles import Cue, _parse_srt, _parse_ts


def test_parse_ts_handles_comma_and_dot() -> None:
    assert _parse_ts("00:00:00,000") == 0
    assert _parse_ts("00:01:23,456") == 83_456
    assert _parse_ts("01:00:00,000") == 3_600_000
    assert _parse_ts("00:01:23.456") == 83_456
    assert _parse_ts("garbage") is None


def test_parse_srt_two_cues() -> None:
    srt = (
        "1\n"
        "00:00:05,000 --> 00:00:08,000\n"
        "Hello world.\n"
        "\n"
        "2\n"
        "00:00:10,500 --> 00:00:12,000\n"
        "Second line.\n"
    )
    cues = _parse_srt(srt)
    assert len(cues) == 2
    assert cues[0] == Cue(start_ms=5_000, end_ms=8_000, text="Hello world.")
    assert cues[1].start_ms == 10_500
    assert cues[1].text == "Second line."


def test_parse_srt_strips_ass_tags() -> None:
    srt = (
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "{\\an8}Top line\n"
    )
    cues = _parse_srt(srt)
    assert cues[0].text == "Top line"
