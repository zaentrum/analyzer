"""Extract chapter boundaries from the media container via ffprobe.

Returns two parallel outputs:

  * `segments` — chapters whose title matches one of the TIDB skippable
    kinds (intro / recap / credits). These ride the normal fuser flow
    and end up in com_nalet_katalog_mediasegments alongside detections
    from the other pipelines.
  * `chapters` — every chapter atom unchanged, including the unlabeled
    "Act 1" / "Cold Open" markers that don't map to a TIDB kind. These
    bypass the fuser and go straight to com_nalet_katalog_itemchapters
    via PUT /api/chapters. They're descriptive metadata, not skippable
    moments, so mixing them with TIDB-aligned segments was always a
    nomenclature mismatch — see migration 018.

ffprobe is the highest-confidence signal we have: chapters were set by
the encoder and require zero analysis, so when they exist the player
can render a full timeline and the analyzer doesn't have to fall back
to acoustic / visual detectors for structure."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

CREDIT_TITLE_PATTERNS = (
    re.compile(r"\bend credits?\b", re.IGNORECASE),
    re.compile(r"\bclosing credits?\b", re.IGNORECASE),
    re.compile(r"\bcredits?\b", re.IGNORECASE),
)
RECAP_TITLE_PATTERNS = (
    re.compile(r"\brecap\b", re.IGNORECASE),
    re.compile(r"\bpreviously on\b", re.IGNORECASE),
)
INTRO_TITLE_PATTERNS = (
    re.compile(r"\b(?:opening|title)\s+sequence\b", re.IGNORECASE),
    re.compile(r"\bintro(?:duction)?\b", re.IGNORECASE),
    re.compile(r"\bmain titles?\b", re.IGNORECASE),
)


@dataclass
class ChapterResult:
    """Output of the chapter pipeline: two lists posted to two endpoints."""
    segments: list[dict]   # TIDB-kind matches, flow through the fuser
    chapters: list[dict]   # every chapter atom, posted to /api/chapters


def detect(path: str) -> ChapterResult:
    """Read chapter atoms from `path` via ffprobe and split them."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_chapters", path],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning("chapters.ffprobe_failed", path=path, error=str(e)[:300])
        return ChapterResult(segments=[], chapters=[])

    try:
        raw_chapters = json.loads(result.stdout).get("chapters", [])
    except json.JSONDecodeError:
        log.warning("chapters.json_decode_failed", path=path)
        return ChapterResult(segments=[], chapters=[])

    segments: list[dict] = []
    chapters: list[dict] = []
    ordinal = 0
    for ch in raw_chapters:
        start = _to_ms(ch.get("start_time"))
        end = _to_ms(ch.get("end_time"))
        if start is None or end is None or end <= start:
            continue
        ordinal += 1
        title = (ch.get("tags") or {}).get("title", "")
        # Every atom lands in the chapters list; ordinal preserves
        # source-file ordering so the UI can render a contiguous bar
        # without re-sorting by startMs.
        chapters.append(
            {
                "startMs": start,
                "endMs": end,
                "title": (title[:120] or None),
                "ordinal": ordinal,
            }
        )
        # When the title matches a TIDB kind, also surface it as a
        # high-confidence segment so the fuser can use it (and the
        # player's skip-intro / skip-credits buttons fire on it).
        kind = _classify_chapter(title)
        if kind is not None:
            segments.append(
                {
                    "kind": kind,
                    "startMs": start,
                    "endMs": end,
                    "source": "chapter",
                    "confidence": 1.00,
                    "label": (title[:120] or None),
                }
            )
    log.info(
        "chapters.detected",
        path=path,
        count=len(chapters),
        labeled_as_segments=len(segments),
    )
    return ChapterResult(segments=segments, chapters=chapters)


def _classify_chapter(title: str) -> str | None:
    """Return a TIDB-aligned kind, or None when the chapter is purely
    structural (no skip-intro / skip-credits hint)."""
    if not title:
        return None
    if any(p.search(title) for p in CREDIT_TITLE_PATTERNS):
        return "credits"
    if any(p.search(title) for p in RECAP_TITLE_PATTERNS):
        return "recap"
    if any(p.search(title) for p in INTRO_TITLE_PATTERNS):
        return "intro"
    return None


def _to_ms(v: str | None) -> int | None:
    if v is None:
        return None
    try:
        return int(float(v) * 1000)
    except (TypeError, ValueError):
        return None
