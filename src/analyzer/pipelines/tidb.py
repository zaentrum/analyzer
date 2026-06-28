"""TIDB (theintrodb.org) pipeline.

For shows and movies TIDB carries, this returns community-curated
intro / recap / credits / preview segments straight from the API. The
fuser ranks these above the locally-derived detectors (chromaprint,
blackframe, silence) so they win whenever they exist — see
`fuser.SOURCE_PRIORITY`.

Why a network call lives in the analyzer pipeline at all:

  * It's cheap: one HTTP round-trip per item, well under a second.
  * It's incremental on the *catalog* side: even after the per-file
    pipelines have run, a re-claim picks up new TIDB submissions for
    free.
  * 404 is the common case for niche / older shows; we treat it as
    "no data, skip" and the rest of the pipelines pick up the slack.

Rate-limiting: TIDB caps anon callers at 30 req/10 s on /media,
500/day with an API key (`TIDB_API_KEY`). The analyzer claims 2
items/cycle by default; comfortable margins all round. On 429 the
function returns an empty list and the next claim retries — we don't
back off here because the fuser is fine without TIDB data on this
specific item.
"""

from __future__ import annotations

import os

import httpx
import structlog

log = structlog.get_logger(__name__)

API_BASE = "https://api.theintrodb.org/v2"
USER_AGENT = "chino-katalog-analyzer/0.2"

# Tags TIDB returns directly map to the kinds the analyzer + Fiori
# already understand, with one rename: TIDB calls the post-credits
# next-episode teaser "preview"; we keep the same name on our side.
SEGMENT_KINDS = ("intro", "recap", "credits", "preview")
TIDB_SOURCE = "tidb"
# Pinned confidence for TIDB rows. Anything community-curated is far
# more trustworthy than the local detectors (chromaprint ≤ 0.95,
# blackframe ≤ 0.7), but we don't claim 1.0 — leave room for a
# `manual` override that's even higher.
TIDB_CONFIDENCE = 0.98


def _enabled() -> bool:
    # Easy off-switch for ops without re-deploying. Defaults to on
    # because the only failure mode is the API being unreachable, which
    # we already handle as a no-op.
    return os.environ.get("TIDB_ENABLED", "true").lower() in ("1", "true", "yes")


def detect(
    *,
    tmdb_id: str | int | None,
    season: int | None = None,
    episode: int | None = None,
    duration_ms: int | None = None,
    media_type: str = "tv",
    timeout_seconds: float = 6.0,
) -> list[dict]:
    """Return MediaSegment dicts for `tmdb_id` (+season/episode for TV).

    Returns [] when:
      * `tmdb_id` is None / empty,
      * `media_type == "tv"` and either `season` or `episode` is None,
      * TIDB returns 404 (no submissions for this media),
      * TIDB returns 429 (rate-limited — log and move on),
      * the request times out / the connection is refused.

    In every case the analyzer simply falls back to the other detectors.
    """
    if not _enabled() or not tmdb_id:
        return []
    if media_type == "tv" and (season is None or episode is None):
        return []

    params: dict[str, str | int] = {"tmdb_id": int(tmdb_id)}
    if media_type == "tv":
        params["season"] = int(season)  # type: ignore[arg-type]
        params["episode"] = int(episode)  # type: ignore[arg-type]

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    api_key = os.environ.get("TIDB_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
            resp = client.get(f"{API_BASE}/media", params=params)
    except (httpx.HTTPError, OSError) as exc:
        log.warning("tidb.fetch_failed", tmdb_id=tmdb_id, error=str(exc)[:200])
        return []

    if resp.status_code == 404:
        return []
    if resp.status_code == 429:
        log.info(
            "tidb.rate_limited",
            tmdb_id=tmdb_id,
            reset=resp.headers.get("X-RateLimit-Reset")
            or resp.headers.get("X-UsageLimit-Reset"),
        )
        return []
    if resp.status_code >= 400:
        log.warning("tidb.bad_status", tmdb_id=tmdb_id, status=resp.status_code)
        return []

    try:
        body = resp.json()
    except ValueError:
        return []

    out: list[dict] = []
    for kind in SEGMENT_KINDS:
        for entry in body.get(kind) or []:
            start_ms = entry.get("start_ms")
            end_ms = entry.get("end_ms")
            # Resolve TIDB's "open" boundaries with the item's duration:
            # intro/recap may have null start (= 0); credits/preview may
            # have null end (= end of media).
            if start_ms is None:
                start_ms = 0
            if end_ms is None:
                end_ms = duration_ms or 0
            if end_ms <= start_ms:
                continue
            mapped = _remap_kind(kind, int(start_ms), duration_ms)
            out.append(
                {
                    "kind": mapped,
                    "startMs": int(start_ms),
                    "endMs": int(end_ms),
                    "confidence": TIDB_CONFIDENCE,
                    "source": TIDB_SOURCE,
                    "label": None,
                }
            )
    return out


# Position threshold for the kind remap below. Anything in the first
# 30% of the file is "head territory" (intro / opening titles / recap);
# anything in the last 30% is "tail territory" (credits / preview).
_HEAD_FRAC = 0.30
_TAIL_FRAC = 0.70


def _remap_kind(kind: str, start_ms: int, duration_ms: int | None) -> str:
    """Reconcile TIDB submitter style with our nomenclature.

    TIDB is community-curated; different submitters tag the same kind
    of boundary inconsistently:

      * Many shows have their opening title sequence tagged
        `kind=credits` (because it's literally the show's opening
        credits) when we want to treat it as `kind=intro` (skippable
        head-of-episode music). When start position is in the file's
        first 30%, remap.
      * Conversely, mid/post-credits stingers sometimes get tagged
        `kind=intro` for the next chapter (especially in serial dramas
        where every episode opens with a recap-of-previous teaser);
        when start is past 70% of the file, that's an end-of-episode
        marker — remap to credits.

    Without a duration we can't reason about position, so leave the
    tag alone.
    """
    if duration_ms is None or duration_ms <= 0:
        return kind
    if kind == "credits" and start_ms < duration_ms * _HEAD_FRAC:
        return "intro"
    if kind == "intro" and start_ms > duration_ms * _TAIL_FRAC:
        return "credits"
    return kind


# --- Sanity check -----------------------------------------------------
# TIDB is community-curated and almost always correct, but we still see
# the occasional submission against the wrong cut of an episode (extended
# vs broadcast, US vs international cut, S01E13 ↔ S01E13.5 numbering
# drift). When the segment windows don't fit our actual file, we'd
# rather fall back to the ML detectors than ship bad data — so the
# tidb_first worker checks each returned set with the predicates below
# before committing it. A single failing window invalidates the whole
# TIDB response for the item; the per_file pass then gets to retry from
# scratch.
#
# Thresholds are deliberately loose: TIDB submissions are usually within
# a few seconds of the truth, so anything off by *minutes* is the
# evidence of a wrong-cut mismatch we want to catch.

# Tolerance on the file's end: TIDB end_ms may legitimately equal the
# file duration. Anything past 105% of duration is a different cut and
# the whole TIDB response is treated as wrong-cut, not just that row.
_TAIL_TOLERANCE = 1.05

# Per-kind duration bounds. The maxes are generous because real-world
# content varies wildly: Marvel-style credits run 10-15 min, sitcom
# intros are 5 s, a long-form drama recap can stretch to 8 min. These
# only catch obviously absurd values (44-minute credits, zero-length
# windows) — wrong-cut detection is handled by _TAIL_TOLERANCE.
#
# Position-in-file rules used to live here too but were causing more
# false positives than wrong-cut catches: TIDB submitters routinely
# tag opening title sequences as `kind=credits` rather than
# `kind=intro`, which is a nomenclature drift, not a data error. The
# canonical fix is `_remap_kind` above; this function is now purely
# about catching submissions that don't fit *our* file at all.
_INTRO_MIN_DUR_MS    = 1_000          # 1 s
_INTRO_MAX_DUR_MS    = 5 * 60 * 1000   # 5 min
_RECAP_MIN_DUR_MS    = 1_000
_RECAP_MAX_DUR_MS    = 10 * 60 * 1000  # 10 min — long-form drama recaps
_CREDITS_MIN_DUR_MS    = 1_000
_CREDITS_MAX_DUR_MS    = 20 * 60 * 1000  # 20 min — Marvel mid-credit stingers
_PREVIEW_MIN_DUR_MS    = 1_000
_PREVIEW_MAX_DUR_MS    = 5 * 60 * 1000

_DUR_BOUNDS = {
    "intro":   (_INTRO_MIN_DUR_MS,   _INTRO_MAX_DUR_MS),
    "recap":   (_RECAP_MIN_DUR_MS,   _RECAP_MAX_DUR_MS),
    "credits": (_CREDITS_MIN_DUR_MS, _CREDITS_MAX_DUR_MS),
    "preview": (_PREVIEW_MIN_DUR_MS, _PREVIEW_MAX_DUR_MS),
}


def sanity_check(segments: list[dict], duration_ms: int | None) -> tuple[bool, str]:
    """Return (ok, reason). When `ok` is False, callers should reject
    the whole TIDB response and fall back to the ML pipelines."""
    if not segments:
        return True, "empty"
    if duration_ms is None or duration_ms <= 0:
        # Without a duration we can't bounds-check anything; trust the
        # endpoint and let the player clamp on playback.
        return True, "no_duration_skipped_check"

    tail = int(duration_ms * _TAIL_TOLERANCE)
    for s in segments:
        kind = s.get("kind")
        start = int(s.get("startMs") or 0)
        end   = int(s.get("endMs")   or 0)
        if start < 0 or end <= start:
            return False, f"{kind}: invalid window [{start},{end})"
        if end > tail:
            return False, (
                f"{kind}: end {end} past 105% of duration {duration_ms}"
            )
        dur = end - start
        bounds = _DUR_BOUNDS.get(kind)
        if bounds is not None:
            lo, hi = bounds
            if dur < lo or dur > hi:
                return False, f"{kind} duration {dur} outside [{lo},{hi}]"
        # Unknown kinds slip through — the Java validator will reject
        # them, and we'd rather log "unexpected kind 'foo'" once than
        # eat a sanity false-positive here.
    return True, "ok"
