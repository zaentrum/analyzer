"""Tests for the keyframe extractor's content-window selection (pure logic;
the ffmpeg sampling/encode paths need a real video and are exercised in the
pipeline integration, not here)."""

from __future__ import annotations

from analyzer.pipelines import keyframe


def _win(duration_ms, segments):
    return keyframe._content_window(duration_ms, segments)


def test_no_segments_uses_5_to_95_band():
    lo, hi = _win(3_600_000, [])
    assert round(lo) == 180
    assert round(hi) == 3420


def test_intro_and_credits_bound_the_window():
    segs = [
        {"kind": "intro", "startMs": 0, "endMs": 300_000},
        {"kind": "credits", "startMs": 3_000_000, "endMs": 3_600_000},
    ]
    lo, hi = _win(3_600_000, segs)
    assert round(lo) == 300  # after the intro
    assert round(hi) == 3000  # before the credits


def test_collapsed_window_falls_back_to_middle():
    # An intro that eats almost the whole runtime collapses the window, so we
    # fall back to the safe 15%-85% middle rather than an empty range.
    segs = [{"kind": "intro", "startMs": 0, "endMs": 95_000}]
    lo, hi = _win(100_000, segs)
    assert round(lo) == 15
    assert round(hi) == 85


def test_zero_or_missing_duration_returns_none():
    assert _win(0, []) is None
    assert _win(None, []) is None
