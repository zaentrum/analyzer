"""Combine signals from individual detectors into the final segment list
that gets uploaded to katalog. Operates on the TIDB-aligned skippable
kinds (intro / recap / credits / preview); file-internal chapter atoms
go straight to ItemChapters and skip this stage entirely — see
migration 018 for the split rationale.

Responsibility:
  Merge overlapping segments of the same kind. blackframe + silence
  often overlap at the credit boundary; we keep one segment whose
  confidence is the *max* of the inputs (boosted slightly when two
  independent sources agree).

  Keep intros and recaps a plausible length (SPAN_MS, MAX_RUNTIME_SHARE).
  A candidate too long to be one is dropped before it can stretch the
  candidates it overlaps, no merge makes one longer than that, and the
  segment a kind ends up with is one of a plausible length: the next
  candidate when the best one isn't. Nothing is trimmed to fit. A
  chapter's candidate is where its kind starts: too long, it takes its end
  from another pipeline's candidate near that start, or is dropped too
  (see _bounded).
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

AGREEMENT_BONUS = 0.10
SOURCE_PRIORITY = {
    # `tidb` is community-curated, verified-by-humans data straight
    # from theintrodb.org. Always outranks our local detectors and
    # outranks `manual` too — if a contributor has already labelled
    # the boundary, take it. (Operators can still override by deleting
    # the tidb row and adding a manual one.)
    "tidb": 10,
    "manual": 6,
    # `chapter` here is the ffprobe chapter detector emitting a labelled
    # intro/credits/recap derived from a chapter atom title — the atoms
    # themselves no longer flow through this fuser. The labelled cases
    # are high-signal (the encoder explicitly named the boundary) and
    # rank just under manual. The title is; the range, which runs to the
    # next chapter mark, isn't always (see _bounded).
    "chapter": 5,
    "chromaprint": 4,
    "whisper": 3,
    "subtitle": 3,
    "blackframe": 2,
    "transnet": 2,
    "silence": 1,
}

# The shortest and the longest an intro and a recap can be, in ms. The
# longest title sequences run about two minutes, and the pipelines already
# draw the line at 3: the subtitle pipeline takes a longer stretch without
# dialogue for a cold open, not an intro, and chromaprint a match that
# starts later for a recurring scene, not the title. A recap varies more
# (a season opener's runs a few minutes), so it gets up to 5. Under 10 s a
# skip button is gone before it can be pressed, and what a detector finds
# that short is a pause or a sting, not a segment.
SPAN_MS = {
    "intro": (10_000, 180_000),
    "recap": (10_000, 300_000),
}
# Nor is either more than a quarter of the runtime, which is the tighter
# bound in a short episode (under 12 min for an intro, under 20 min for a
# recap). Without a runtime, SPAN_MS alone bounds them.
MAX_RUNTIME_SHARE = 0.25
# Silence and black-frame windows mark a boundary, where a segment may
# start or end, rather than a segment of their own.
BOUNDARY_SOURCES = ("silence", "blackframe")
# A segment that starts within this of a chapter's start agrees with the
# chapter on where its kind starts: a detector's start can be off by a few
# seconds (a logo card, a fade-in).
NEAR_START_MS = 10_000


def merge(signals: list[list[dict]], duration_ms: int | None = None) -> list[dict]:
    """The pipelines' signals fused into the segments to upload: at most one
    per skippable kind. `duration_ms`, the item's runtime when it is known,
    bounds intros and recaps further (MAX_RUNTIME_SHARE)."""
    flat: list[dict] = _bounded([s for group in signals for s in group], duration_ms)
    flat.sort(key=lambda s: (s["kind"], s["startMs"]))
    merged: list[dict] = []
    for seg in flat:
        prev = merged[-1] if merged else None
        same_kind = prev is not None and prev["kind"] == seg["kind"]
        overlap   = prev is not None and seg["startMs"] < prev["endMs"]
        # A merge that would make an intro or a recap too long to be one
        # isn't made: the two stay apart, and the pick below takes one.
        if same_kind and overlap and _fits(prev, seg, duration_ms):
            prev["endMs"] = max(prev["endMs"], seg["endMs"])
            # Two different sources agreeing on the same kind range is stronger
            # evidence than either alone. Pick the higher-priority source as
            # the canonical one and bump confidence.
            if seg["source"] != prev["source"]:
                if SOURCE_PRIORITY.get(seg["source"], 0) > SOURCE_PRIORITY.get(prev["source"], 0):
                    prev["source"] = seg["source"]
                boosted = max(prev["confidence"], seg["confidence"]) + AGREEMENT_BONUS
                prev["confidence"] = min(0.99, boosted)
            else:
                prev["confidence"] = max(prev["confidence"], seg["confidence"])
            if seg.get("label") and not prev.get("label"):
                prev["label"] = seg["label"]
            continue
        merged.append(dict(seg))

    # Collapse intro / credits / recap / preview to a single canonical
    # segment per kind. Multiple non-overlapping silence/blackframe
    # windows in the tail were each emitting their own "credits" tick on
    # the scrub bar — visually distracting and not what the user wants.
    # Keep the segment from the highest-priority source per kind (so a
    # chromaprint credit roll beats incidental silence detections),
    # using the longest segment from that source if multiple exist. An
    # intro or a recap must be a plausible length (SPAN_MS) to be kept:
    # a lone pause the silence pipeline found is no intro.
    canonical_kinds = ("intro", "recap", "credits", "preview")
    canonical = [s for s in merged if s["kind"] not in canonical_kinds]
    for kind in canonical_kinds:
        candidates = [s for s in merged if s["kind"] == kind and _plausible(s, duration_ms)]
        if not candidates:
            continue
        candidates.sort(
            key=lambda s: (
                -SOURCE_PRIORITY.get(s["source"], 0),
                -(s["endMs"] - s["startMs"]),
            )
        )
        canonical.append(candidates[0])
    merged = canonical

    return sorted(merged, key=lambda s: s["startMs"])


def _span(kind: str, duration_ms: int | None) -> tuple[int, int] | None:
    """The shortest and the longest a segment of `kind` can be, in ms, or
    None for a kind without bounds (credits, preview)."""
    span = SPAN_MS.get(kind)
    if span is None:
        return None
    shortest, longest = span
    if duration_ms is not None and duration_ms > 0:
        longest = min(longest, int(duration_ms * MAX_RUNTIME_SHARE))
    return shortest, longest


def _too_long(seg: dict, duration_ms: int | None) -> bool:
    span = _span(seg["kind"], duration_ms)
    return span is not None and seg["endMs"] - seg["startMs"] > span[1]


def _plausible(seg: dict, duration_ms: int | None) -> bool:
    span = _span(seg["kind"], duration_ms)
    return span is None or span[0] <= seg["endMs"] - seg["startMs"] <= span[1]


def _fits(prev: dict, seg: dict, duration_ms: int | None) -> bool:
    """Whether `prev` merged with `seg` (which starts no earlier) is still
    short enough for its kind."""
    span = _span(prev["kind"], duration_ms)
    return span is None or max(prev["endMs"], seg["endMs"]) - prev["startMs"] <= span[1]


def _bounded(flat: list[dict], duration_ms: int | None) -> list[dict]:
    """`flat` without the intros and recaps too long to be one, which would
    otherwise stretch every candidate they overlap.

    A chapter titled as an intro or a recap says where that segment
    starts, but its range runs to the next chapter mark: in a file with
    coarse chapters, the end credits. Too long, it is kept as that start,
    with the end of another pipeline's candidate near it (_chapter_end),
    and dropped when there is none."""
    fitting = [s for s in flat if not _too_long(s, duration_ms)]
    out: list[dict] = []
    for seg in flat:
        if not _too_long(seg, duration_ms):
            out.append(seg)
            continue
        end = _chapter_end(seg, fitting, duration_ms) if seg["source"] == "chapter" else None
        if end is None:
            log.info("fuser.too_long", kind=seg["kind"], source=seg["source"],
                     start_ms=seg["startMs"], end_ms=seg["endMs"])
            continue
        log.info("fuser.chapter_end", kind=seg["kind"], label=seg.get("label"),
                 start_ms=seg["startMs"], end_ms=end["endMs"], end_source=end["source"])
        out.append({**seg, "endMs": end["endMs"], "confidence": end["confidence"]})
    return out


def _chapter_end(chapter: dict, candidates: list[dict], duration_ms: int | None) -> dict | None:
    """The candidate whose end ends the intro or recap `chapter` starts, or
    None. Of the other pipelines' candidates of its kind within reach of
    the chapter's start (where it could still run, SPAN_MS):

      * a segment that starts near the chapter (NEAR_START_MS) and ends a
        plausible length after it, from the highest-priority source, and
        the earliest end of those. When segments are within reach but
        none is such, they put the segment where the chapter doesn't,
        and the chapter gives way to them;
      * when there is no segment within reach, the first boundary window
        (BOUNDARY_SOURCES) to end a plausible length after the start: the
        pause after an intro, where the episode starts."""
    kind, start = chapter["kind"], chapter["startMs"]
    span = _span(kind, duration_ms)
    if span is None:
        return None
    earliest, latest = start + span[0], start + span[1]
    reach = [c for c in candidates
             if c["kind"] == kind and c["source"] != "chapter"
             and c["startMs"] < latest and c["endMs"] > start]
    segments = [c for c in reach if c["source"] not in BOUNDARY_SOURCES]
    if segments:
        ends = [c for c in segments
                if abs(c["startMs"] - start) <= NEAR_START_MS
                and earliest <= c["endMs"] <= latest]
        return min(ends, key=lambda c: (-SOURCE_PRIORITY.get(c["source"], 0), c["endMs"]),
                   default=None)
    pauses = [c for c in reach if earliest <= c["endMs"] <= latest]
    return min(pauses, key=lambda c: c["endMs"], default=None)
