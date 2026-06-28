"""Tiny unit test for the fuser merge logic — the only piece of analyzer
code that runs without ffmpeg/CUDA being installed."""

from __future__ import annotations

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
