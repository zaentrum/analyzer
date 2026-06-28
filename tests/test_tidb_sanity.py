"""Tests for the TIDB sanity-check predicate. Runs without a network
call — the function operates on parsed segment dicts only."""

from __future__ import annotations

from analyzer.pipelines.tidb import _remap_kind, sanity_check

# A nominal 22-minute episode duration (1_320_000 ms). Used as the
# reference length for sitcoms like Ghosts in our smoketest data.
DUR_22_MIN = 22 * 60 * 1000


def _seg(kind: str, start: int, end: int) -> dict:
    return {"kind": kind, "startMs": start, "endMs": end,
            "source": "tidb", "confidence": 0.98, "label": None}


def test_empty_input_passes() -> None:
    ok, reason = sanity_check([], DUR_22_MIN)
    assert ok and reason == "empty"


def test_missing_duration_skips_check() -> None:
    # A duration we can't trust → trust the upstream and let the player
    # clamp; this is the right behaviour for the long tail of items
    # where TMDB never returned a runtime.
    weird = [_seg("intro", 30_000, 90_000)]
    ok, _ = sanity_check(weird, None)
    assert ok
    ok, _ = sanity_check(weird, 0)
    assert ok


def test_typical_sitcom_intro_credits_pass() -> None:
    # Intro at 2:00-2:05, credits in the last minute — matches the
    # Ghosts smoketest data we hand-corrected earlier.
    segs = [
        _seg("intro",   120_000, 125_000),
        _seg("credits", DUR_22_MIN - 60_000, DUR_22_MIN),
    ]
    ok, reason = sanity_check(segs, DUR_22_MIN)
    assert ok, reason


def test_segment_past_file_length_fails() -> None:
    # TIDB submission against the extended cut: end_ms past the
    # broadcast cut's duration. Fall back to ML.
    segs = [_seg("credits", DUR_22_MIN - 30_000, DUR_22_MIN + 5 * 60 * 1000)]
    ok, reason = sanity_check(segs, DUR_22_MIN)
    assert not ok
    assert "past" in reason


def test_zero_length_segment_fails() -> None:
    segs = [_seg("intro", 30_000, 30_000)]
    ok, reason = sanity_check(segs, DUR_22_MIN)
    assert not ok
    assert "invalid" in reason


def test_credits_at_start_passes_after_remap() -> None:
    # TIDB submitters routinely tag the opening title sequence as
    # 'credits' rather than 'intro'. After `_remap_kind` runs at parse
    # time we expect them as 'intro', and sanity-check accepts that.
    segs = [_seg("intro", 30_000, 60_000)]
    ok, reason = sanity_check(segs, DUR_22_MIN)
    assert ok, reason


def test_recap_up_to_10min_passes() -> None:
    # Long-form drama recap stretches near our new 10-min ceiling.
    segs = [_seg("recap", 0, 9 * 60 * 1000)]
    ok, reason = sanity_check(segs, DUR_22_MIN)
    assert ok, reason


def test_recap_past_10min_fails() -> None:
    segs = [_seg("recap", 0, 11 * 60 * 1000)]
    ok, reason = sanity_check(segs, DUR_22_MIN)
    assert not ok
    assert "recap" in reason


def test_one_bad_segment_invalidates_whole_set() -> None:
    # Conservative: even if 3 of 4 segments are fine, one bad one means
    # the TIDB submission is suspect and the whole response is rejected.
    # The bad row here goes past the end of the file — a clear wrong-cut
    # signal that survives the relaxed (post-remap) sanity rules.
    segs = [
        _seg("intro",   30_000, 90_000),
        _seg("recap",   90_000, 180_000),
        _seg("credits", DUR_22_MIN - 60_000, DUR_22_MIN),
        # one bogus row — extends 5 min past the file's actual end:
        _seg("credits", DUR_22_MIN - 30_000, DUR_22_MIN + 5 * 60 * 1000),
    ]
    ok, reason = sanity_check(segs, DUR_22_MIN)
    assert not ok
    assert "past" in reason


def test_feature_length_film_credits_pass() -> None:
    # 2-hour film, credits at 1:54 → 2:00 — fine.
    dur = 2 * 60 * 60 * 1000  # 7_200_000 ms
    segs = [_seg("credits", dur - 6 * 60 * 1000, dur)]
    ok, reason = sanity_check(segs, dur)
    assert ok, reason


# --- kind remap (TIDB submitter style normalisation) -----------------

def test_remap_credits_at_start_becomes_intro() -> None:
    # 25 s into a 22-min sitcom — that's the title sequence.
    assert _remap_kind("credits", 25_000, DUR_22_MIN) == "intro"


def test_remap_credits_at_end_stays_credits() -> None:
    assert _remap_kind("credits", DUR_22_MIN - 30_000, DUR_22_MIN) == "credits"


def test_remap_intro_at_end_becomes_credits() -> None:
    # 21 min into a 22-min file — really a mid/post-credits tag.
    assert _remap_kind("intro", 21 * 60 * 1000, DUR_22_MIN) == "credits"


def test_remap_intro_at_start_stays_intro() -> None:
    assert _remap_kind("intro", 30_000, DUR_22_MIN) == "intro"


def test_remap_passthrough_when_no_duration() -> None:
    # Without a duration we can't reason about position; trust the tag.
    assert _remap_kind("credits", 25_000, None) == "credits"
    assert _remap_kind("intro",   25_000, 0) == "intro"


def test_remap_other_kinds_passthrough() -> None:
    # recap / preview are rarely mistagged; leave them alone so a bad
    # remap doesn't create new false positives.
    assert _remap_kind("recap",   25_000, DUR_22_MIN) == "recap"
    assert _remap_kind("preview", 25_000, DUR_22_MIN) == "preview"
