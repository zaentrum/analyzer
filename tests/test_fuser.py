"""Tiny unit test for the fuser merge logic — the only piece of analyzer
code that runs without ffmpeg/CUDA being installed."""

from __future__ import annotations

import pytest

from analyzer import worker
from analyzer.katalog import ClaimedItem
from analyzer.pipelines.fuser import AGREEMENT_BONUS, merge


def test_merges_overlapping_same_kind_with_agreement_bonus() -> None:
    bf = {"kind": "credits", "startMs": 5400_000, "endMs": 5500_000,
          "source": "blackframe", "confidence": 0.70}
    sl = {"kind": "credits", "startMs": 5450_000, "endMs": 5600_000,
          "source": "silence", "confidence": 0.40}
    result = merge([[bf], [sl]])
    assert len(result) == 1
    m = result[0]
    assert m["startMs"] == 5400_000
    assert m["endMs"] == 5600_000
    # Higher-priority source wins; confidence gets the bonus.
    assert m["source"] == "blackframe"
    assert m["confidence"] == 0.70 + AGREEMENT_BONUS


def test_does_not_merge_different_kinds() -> None:
    intro = {"kind": "intro", "startMs": 0, "endMs": 90_000, "source": "silence",
             "confidence": 0.4}
    credits = {"kind": "credits", "startMs": 5400_000, "endMs": 5500_000,
               "source": "blackframe", "confidence": 0.7}
    result = merge([[intro], [credits]])
    assert len(result) == 2


def test_canonical_keeps_highest_priority_per_kind() -> None:
    # Two overlapping credits detections + one non-overlapping silence
    # spike near the tail. The canonical step should pick the higher-
    # priority blackframe one and drop the silence-only candidate that
    # didn't overlap (so it never merged in).
    bf = {"kind": "credits", "startMs": 5400_000, "endMs": 5600_000,
          "source": "blackframe", "confidence": 0.70}
    sl_overlap = {"kind": "credits", "startMs": 5500_000, "endMs": 5650_000,
                  "source": "silence", "confidence": 0.30}
    sl_far = {"kind": "credits", "startMs": 6100_000, "endMs": 6200_000,
              "source": "silence", "confidence": 0.30}
    result = merge([[bf], [sl_overlap, sl_far]])
    assert len(result) == 1
    assert result[0]["source"] == "blackframe"


def test_preview_kind_round_trips() -> None:
    # `preview` is the TIDB nomenclature for the post-credits
    # next-episode teaser. Make sure the fuser carries it through;
    # before migration 018 the kind didn't exist and the controller
    # whitelist would have rejected it.
    p = {"kind": "preview", "startMs": 1380_000, "endMs": 1410_000,
         "source": "tidb", "confidence": 0.98}
    result = merge([[p]])
    assert len(result) == 1
    assert result[0]["kind"] == "preview"


# ------------------------------------------ an intro or a recap is bounded
EPISODE_MS = 1_360_000  # 22:40
CREDITS_AT = 1_323_000  # 22:03, where the end credits' chapter starts


def _seg(kind: str, start: int, end: int, source: str, confidence: float,
         label: str | None = None) -> dict:
    return {"kind": kind, "startMs": start, "endMs": end, "source": source,
            "confidence": confidence, "label": label}


def _chapter(kind: str, start: int, end: int, title: str) -> dict:
    return _seg(kind, start, end, "chapter", 1.00, title)


def _pause(start: int, end: int) -> dict:
    """A window the silence pipeline finds in the head of a file."""
    return _seg("intro", start, end, "silence", 0.40, f"silence {(end - start) / 1000:.1f}s")


def _of(kind: str, segments: list[dict]) -> list[tuple[int, int, str]]:
    return [(s["startMs"], s["endMs"], s["source"]) for s in segments if s["kind"] == kind]


# A file with coarse chapters: "Main Titles" at 0:00, and the next chapter
# mark at the end credits.
MAIN_TITLES = _chapter("intro", 0, CREDITS_AT, "Main Titles")
END_CREDITS = _chapter("credits", CREDITS_AT, EPISODE_MS, "End Credits")


@pytest.mark.parametrize("others", [[], [_pause(0, 1_800)]])
def test_a_coarse_intro_chapter_alone_gives_no_intro(others: list[dict]) -> None:
    # The chapter's range is the whole episode, and nothing else says where
    # the intro ends: a pause in the first seconds is too soon to.
    result = merge([[MAIN_TITLES, END_CREDITS], others], duration_ms=EPISODE_MS)
    assert _of("intro", result) == []
    assert _of("credits", result) == [(CREDITS_AT, EPISODE_MS, "chapter")]


@pytest.mark.parametrize("others", [
    [_pause(36_500, 38_000), _pause(71_000, 72_500)],  # the first pause after it
    [_seg("intro", 1_000, 38_000, "chromaprint", 0.80, "chromaprint x3")],
])
def test_a_coarse_intro_chapter_ends_where_another_pipeline_ends_it(others: list[dict]) -> None:
    result = merge([[MAIN_TITLES, END_CREDITS], others], duration_ms=EPISODE_MS)
    assert _of("intro", result) == [(0, 38_000, "chapter")]
    assert _of("credits", result) == [(CREDITS_AT, EPISODE_MS, "chapter")]


def test_a_recap_chapter_then_a_coarse_intro_chapter() -> None:
    # "Recap" 0:00-0:50, then "Main Titles" until the end credits at 22:46.
    # The recap chapter is a recap's length and stays as it is; the intro
    # ends at the pause after the titles, 1:30.
    episode, credits_at = 1_410_000, 1_366_000
    chapters = [_chapter("recap", 0, 50_000, "Recap"),
                _chapter("intro", 50_000, credits_at, "Main Titles"),
                _chapter("credits", credits_at, episode, "End Credits")]
    pauses = [_pause(48_500, 51_500), _pause(88_000, 90_000)]
    result = merge([chapters, pauses], duration_ms=episode)
    assert _of("recap", result) == [(0, 50_000, "chapter")]
    # The pause between the recap and the titles overlaps the chapter's
    # start, and joins the intro as overlapping candidates do.
    assert _of("intro", result) == [(48_500, 90_000, "chapter")]
    assert _of("credits", result) == [(credits_at, episode, "chapter")]


def test_an_intro_chapter_of_an_intros_length_is_kept_as_it_is() -> None:
    intro = _chapter("intro", 5_000, 45_000, "Intro")
    result = merge([[intro], [_pause(120_000, 122_000)]], duration_ms=EPISODE_MS)
    assert [s for s in result if s["kind"] == "intro"] == [intro]


@pytest.mark.parametrize(("others", "intro"), [
    ([], []),
    # The next candidate is the intro, as it was found.
    ([_seg("intro", 0, 42_000, "subtitle", 0.55, "no dialogue until 42s")],
     [(0, 42_000, "subtitle")]),
])
def test_an_over_long_chromaprint_intro_is_dropped(others: list[dict], intro: list) -> None:
    # A match across the siblings that runs most of the head window (9:50)
    # is no title sequence, and it would stretch every candidate it overlaps.
    theme = _seg("intro", 0, 590_000, "chromaprint", 0.90, "chromaprint x3")
    result = merge([[theme], others, [_pause(41_000, 42_000)]], duration_ms=EPISODE_MS)
    assert _of("intro", result) == intro


def test_no_merge_makes_an_intro_too_long() -> None:
    # Each is an intro's length, but merged they would run 3:20: they stay
    # apart, and the higher-priority one is the intro.
    quiet = _seg("intro", 0, 170_000, "subtitle", 0.55, "no dialogue until 170s")
    theme = _seg("intro", 150_000, 200_000, "chromaprint", 0.80, "chromaprint x2")
    result = merge([[quiet], [theme]], duration_ms=EPISODE_MS)
    assert _of("intro", result) == [(150_000, 200_000, "chromaprint")]


@pytest.mark.parametrize(("duration_ms", "intro"), [
    (360_000, []),                           # a 6-minute episode: 1:30 at most
    (None, [(0, 100_000, "chromaprint")]),   # no runtime: SPAN_MS alone
])
def test_an_intro_is_at_most_a_quarter_of_the_runtime(duration_ms: int | None,
                                                      intro: list) -> None:
    theme = _seg("intro", 0, 100_000, "chromaprint", 0.80, "chromaprint x2")
    assert _of("intro", merge([[theme]], duration_ms=duration_ms)) == intro


def test_a_coarse_intro_chapter_gives_way_where_a_pipeline_places_the_intro() -> None:
    # The titles follow a cold open: the theme the siblings share is at
    # 2:30-2:55, not at the chapter's start. Ending the chapter at the first
    # pause would make the cold open's first 38 s the intro.
    theme = _seg("intro", 150_000, 175_000, "chromaprint", 0.80, "chromaprint x3")
    result = merge([[MAIN_TITLES, END_CREDITS], [theme], [_pause(36_500, 38_000)]],
                   duration_ms=EPISODE_MS)
    assert _of("intro", result) == [(150_000, 175_000, "chromaprint")]


@pytest.mark.parametrize(("others", "recap"), [
    ([], []),
    ([_seg("recap", 3_000, 93_000, "subtitle", 0.90)], [(0, 93_000, "chapter")]),
])
def test_a_coarse_recap_chapter_ends_where_another_pipeline_ends_it(others: list[dict],
                                                                    recap: list) -> None:
    chapter = _chapter("recap", 0, CREDITS_AT, "Recap")
    result = merge([[chapter, END_CREDITS], others], duration_ms=EPISODE_MS)
    assert _of("recap", result) == recap


def test_the_worker_bounds_the_segments_by_the_items_runtime() -> None:
    item = ClaimedItem(id="i", type="episode", title="Clip", year=None,
                       duration_ms=360_000, path="clip.mkv")
    theme = _seg("intro", 0, 100_000, "chromaprint", 0.80, "chromaprint x2")
    assert worker._fuse(item, [[theme]], []) == []
