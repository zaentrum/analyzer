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
"""

from __future__ import annotations

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
    # rank just under manual.
    "chapter": 5,
    "chromaprint": 4,
    "whisper": 3,
    "subtitle": 3,
    "blackframe": 2,
    "transnet": 2,
    "silence": 1,
}


def merge(signals: list[list[dict]]) -> list[dict]:
    flat: list[dict] = [s for group in signals for s in group]
    flat.sort(key=lambda s: (s["kind"], s["startMs"]))
    merged: list[dict] = []
    for seg in flat:
        prev = merged[-1] if merged else None
        same_kind = prev is not None and prev["kind"] == seg["kind"]
        overlap   = prev is not None and seg["startMs"] < prev["endMs"]
        if same_kind and overlap:
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
    # using the longest segment from that source if multiple exist.
    canonical_kinds = ("intro", "recap", "credits", "preview")
    canonical = [s for s in merged if s["kind"] not in canonical_kinds]
    for kind in canonical_kinds:
        candidates = [s for s in merged if s["kind"] == kind]
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
