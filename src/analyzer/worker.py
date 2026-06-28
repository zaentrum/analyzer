"""Worker loop. One python process = one worker. Replicas scale linearly
because POST /api/analyze/claim is race-free (Postgres SKIP LOCKED)."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import structlog

from .katalog import ClaimedItem, KatalogClient
from .pipelines import blackframe, chapters, chromaprint, fuser, silence, subtitles, tidb

# Segment kinds that don't make sense for a given media type. Movies are
# standalone — they have no recurring theme tune or "Previously on…"
# preamble, so chromaprint / recap heuristics are noise for them; only
# the credits roll at the tail is useful. Series episodes legitimately
# carry intro + recap + credits.
SUPPRESSED_KINDS = {
    "movie": {"intro", "recap"},
}

log = structlog.get_logger(__name__)


@dataclass
class AnalyzeResult:
    """Output of `analyze_one`: segments + chapters posted to two endpoints."""
    segments: list[dict]
    chapters: list[dict]


def analyze_one(item: ClaimedItem, client: KatalogClient | None = None) -> AnalyzeResult:
    """Run every per-file pipeline against `item.path` and return both
    the fused skippable-segments list (TIDB vocabulary) and the raw
    chapter atoms list. Designed to be exception-safe: an individual
    pipeline raising returns an empty contribution but doesn't kill
    the worker.

    Series episodes additionally get a chromaprint pass that fingerprints
    the file's head + tail and asks katalog for sibling episodes of the
    same season; the longest fingerprint match across siblings localises
    the recurring intro / credit-roll. Movies skip that pass since they
    have no siblings."""
    if not os.path.exists(item.path):
        log.warning("analyze.missing_file", item_id=item.id, path=item.path)
        raise FileNotFoundError(item.path)

    # Chapters runs in a separate path: ffprobe extracts the file's
    # structural chapter atoms (Cold Open / Act 1 / Tag) and also lifts
    # any title-labelled intro / credits / recap chapters into the
    # fused segments stream. The atoms themselves are written to the
    # sibling ItemChapters entity via PUT /api/chapters — see
    # migration 018 for the split rationale.
    chapter_result = chapters.detect(item.path)
    chapter_segments = chapter_result.segments
    chapter_atoms    = chapter_result.chapters

    # TIDB runs FIRST among the network/per-file detectors. It's a cheap
    # (~0.5 s) call that, on a hit, gives us segments at confidence 0.98
    # — strictly better than anything the local detectors can produce.
    # We still run the local pipelines below; the fuser ranks them under
    # TIDB and only ships whichever the decider picks per kind. This
    # keeps the analyzer correct for the long tail of niche shows TIDB
    # doesn't carry.
    pipelines: list[tuple[str, Any]] = [
        (
            "tidb",
            lambda: tidb.detect(
                tmdb_id=item.tmdb_id,
                season=item.season_number,
                episode=item.episode_number,
                duration_ms=item.duration_ms,
                media_type="movie" if item.type == "movie" else "tv",
            ),
        ),
        # NOTE: pipeline names here are singular to match the
        # ItemProcessingSteps `step` column whitelist on the katalog
        # side. The Python modules are still chapters / subtitles
        # (plural) — only the labels change. The `chapter` step is
        # bookkept here even though we already invoked the chapters
        # detector above, so the Fiori checklist reflects whether the
        # ffprobe pass ran for this item.
        ("chapter",    lambda: chapter_segments),
        ("subtitle",   lambda: subtitles.detect(item.path, item.duration_ms)),
        ("blackframe", lambda: blackframe.detect(item.path, item.duration_ms)),
        ("silence",    lambda: silence.detect(item.path, item.duration_ms)),
    ]
    if item.type == "episode" and client is not None:
        siblings = []
        try:
            siblings = client.siblings(item.id, limit=5)
        except Exception as e:
            log.warning("chromaprint.siblings_failed", item_id=item.id, error=str(e)[:200])
        if siblings:
            sibling_paths = [s.path for s in siblings if os.path.exists(s.path)]
            pipelines.append((
                "chromaprint",
                lambda: chromaprint.detect(item.path, item.duration_ms, sibling_paths),
            ))

    # Skip pipelines whose step has already been answered by a previous
    # pass — `done` (someone wrote the result) or `not_applicable` (the
    # tidb_first short-circuit told us TIDB already covers this item).
    # Saves us a full ML cycle on every TIDB-handled item; on a fresh
    # ingest the map is empty so nothing is skipped.
    pre_existing_status: dict[str, str] = {}
    if client is not None:
        pre_existing_status = client.get_steps(item.id)
    SHORT_CIRCUIT = {"done", "not_applicable"}

    signals: list[list[dict]] = []
    for name, run in pipelines:
        prior = pre_existing_status.get(name)
        if prior in SHORT_CIRCUIT:
            log.info(
                "pipeline.short_circuit",
                pipeline=name,
                item_id=item.id,
                prior_status=prior,
            )
            signals.append([])
            continue
        # Step bookkeeping: flip to in_progress before, then done/failed/
        # skipped after. Cheap network call; the worker's main hot path
        # is ffmpeg, not these PUTs.
        if client is not None:
            client.upsert_step(item.id, name, "in_progress")
        try:
            t0 = time.monotonic()
            sig = run()
            signals.append(sig)
            log.info(
                "pipeline.done",
                pipeline=name,
                item_id=item.id,
                seconds=round(time.monotonic() - t0, 2),
                count=len(sig),
            )
            if client is not None:
                # "skipped" instead of "done" when the pipeline ran cleanly
                # but produced zero rows AND the input data was missing — eg.
                # tidb 404 (no tmdb match), chromaprint no siblings, subtitles
                # no SRT. Distinguishing skipped from done makes the Fiori
                # checklist read truthfully. The chapter step is marked
                # "done" when the file had any chapter atoms at all, even
                # if none of them were title-classified as intro/credits/
                # recap, since the atoms themselves are useful.
                if name == "chapter":
                    step_status = "done" if chapter_atoms else "skipped"
                else:
                    step_status = "done" if sig else "skipped"
                client.upsert_step(item.id, name, step_status)
        except Exception as e:
            log.exception("pipeline.failed", pipeline=name, item_id=item.id, error=str(e)[:300])
            signals.append([])
            if client is not None:
                client.upsert_step(item.id, name, "failed", error=str(e)[:300])
    fused = fuser.merge(signals)
    # Drop segments that don't apply to this media type (e.g. intro on a
    # movie is always noise — movies aren't a recurring series).
    drop_kinds = SUPPRESSED_KINDS.get(item.type, set())
    if drop_kinds:
        fused = [s for s in fused if s["kind"] not in drop_kinds]
    return AnalyzeResult(segments=fused, chapters=chapter_atoms)


def run_worker(client: KatalogClient, batch_size: int, idle_sleep: float, error_sleep: float,
               stop: threading.Event) -> None:
    """Blocking loop. Exits when `stop` is set (or process is killed)."""
    while not stop.is_set():
        try:
            batch = client.claim("per_file", batch_size)
        except Exception as e:
            log.exception("worker.claim_failed", error=str(e)[:300])
            stop.wait(error_sleep)
            continue

        if not batch:
            stop.wait(idle_sleep)
            continue

        for item in batch:
            if stop.is_set():
                break
            t0 = time.monotonic()
            try:
                result = analyze_one(item, client=client)
                client.upload_segments(item.id, result.segments)
                client.upload_chapters(item.id, result.chapters)
                log.info(
                    "worker.item_done",
                    item_id=item.id,
                    title=item.title,
                    segments=len(result.segments),
                    chapters=len(result.chapters),
                    seconds=round(time.monotonic() - t0, 2),
                )
            except FileNotFoundError as e:
                client.fail(item.id, f"file missing: {e}")
            except Exception as e:
                log.exception("worker.item_failed", item_id=item.id, error=str(e)[:300])
                # Don't fail the item on transient errors — let it stay
                # in_progress; the janitor will reset stuck rows. For now,
                # mark failed so we don't infinitely retry the same bug.
                try:
                    client.fail(item.id, str(e)[:500])
                except Exception:
                    log.exception("worker.fail_report_failed", item_id=item.id)


# --- Stage-1 TIDB-only sweep -----------------------------------------
# Runs alongside the per_file worker but does *only* the TIDB network
# call. On a hit + sanity-OK, it marks the per_file ML steps as
# not_applicable so the slow GPU loop doesn't claim the same item — and
# uploads TIDB's segments straight away so the player has data within
# seconds of ingest. On a miss or sanity failure, it leaves every ML
# step pending and the existing per_file pass picks it up.
#
# Why a separate worker:
#   1. TIDB is rate-limited to 30 req / 10s anonymously; the per_file
#      pace (1-2 items/min) doesn't even hit that ceiling, but a
#      dedicated sweep can.
#   2. The catalog has thousands of items where TIDB has the answer
#      and we're currently doing 60 s of ffmpeg work to discover that.
#      Decoupling cuts the time-to-first-data from days to hours.
#   3. It also makes the per_file worker free to focus on items TIDB
#      has nothing for — the long tail of niche shows / movies.


# ML steps the tidb_first pass short-circuits when TIDB has authoritative
# data. `chapter` is intentionally NOT in this list — chapter atoms feed
# the ItemChapters entity, which is orthogonal to skippable segments and
# still useful even when TIDB knows the intro/credits boundaries.
TIDB_FIRST_SHORT_CIRCUIT_STEPS = ("blackframe", "silence", "subtitle", "chromaprint")


def run_tidb_first_worker(
    client: KatalogClient,
    batch_size: int,
    idle_sleep: float,
    error_sleep: float,
    claim_interval: float,
    stop: threading.Event,
) -> None:
    """TIDB-only sweep. Same lifecycle contract as `run_worker`: blocks
    until `stop` is set. Throttles to `claim_interval` seconds between
    individual TIDB lookups so anonymous-tier rate limits (30 req / 10 s)
    stay comfortably under the ceiling."""
    while not stop.is_set():
        try:
            batch = client.claim("tidb_first", batch_size)
        except Exception as e:
            log.exception("tidb_first.claim_failed", error=str(e)[:300])
            stop.wait(error_sleep)
            continue

        if not batch:
            stop.wait(idle_sleep)
            continue

        for item in batch:
            if stop.is_set():
                break
            t0 = time.monotonic()
            try:
                # Refresh tidb=in_progress as a heartbeat. The claim
                # endpoint flipped it on dequeue, but if processing
                # within this worker takes a while we want the timestamp
                # to keep moving (a janitor sweep would otherwise mark
                # us stuck).
                client.upsert_step(item.id, "tidb", "in_progress")

                segments = tidb.detect(
                    tmdb_id=item.tmdb_id,
                    season=item.season_number,
                    episode=item.episode_number,
                    duration_ms=item.duration_ms,
                    media_type="movie" if item.type == "movie" else "tv",
                )
                # Suppress kinds that don't make sense for this media type
                # (intro/recap on a standalone movie). Same SUPPRESSED_KINDS
                # mapping the per_file fuser uses.
                drop = SUPPRESSED_KINDS.get(item.type, set())
                if drop:
                    segments = [s for s in segments if s["kind"] not in drop]

                if not segments:
                    # TIDB had nothing (404 / empty / dropped by media-type
                    # filter). Leave every ML step pending so the per_file
                    # pass picks the item up later.
                    client.upsert_step(item.id, "tidb", "skipped",
                                       error="tidb: no submissions for this media")
                    log.info("tidb_first.miss", item_id=item.id, title=item.title,
                             seconds=round(time.monotonic() - t0, 2))
                    continue

                ok, reason = tidb.sanity_check(segments, item.duration_ms)
                if not ok:
                    client.upsert_step(item.id, "tidb", "failed",
                                       error=f"sanity_check: {reason}")
                    log.info("tidb_first.sanity_failed", item_id=item.id,
                             title=item.title, reason=reason,
                             seconds=round(time.monotonic() - t0, 2))
                    continue

                # TIDB win: upload the segments (full replace), mark tidb
                # done, short-circuit the per_file ML steps. We do NOT
                # upload chapters here — those need ffprobe on the file,
                # which this worker doesn't have. The chapter step stays
                # pending; the per_file pass will pick it up and the
                # short-circuit prevents the other detectors from running
                # alongside.
                client.upload_segments(item.id, segments)
                client.upsert_step(item.id, "tidb", "done",
                                   details=f"segments={len(segments)} sanity=ok")
                client.mark_steps_not_applicable(
                    item.id,
                    list(TIDB_FIRST_SHORT_CIRCUIT_STEPS),
                    reason="tidb_first hit, ML detectors redundant",
                )
                log.info("tidb_first.hit", item_id=item.id, title=item.title,
                         segments=len(segments),
                         seconds=round(time.monotonic() - t0, 2))
            except Exception as e:
                log.exception("tidb_first.item_failed", item_id=item.id,
                              error=str(e)[:300])
                try:
                    client.upsert_step(item.id, "tidb", "failed",
                                       error=str(e)[:500])
                except Exception:
                    log.exception("tidb_first.fail_report_failed",
                                  item_id=item.id)

            stop.wait(claim_interval)
