"""Event-driven analyzer worker.

Each process runs ONE Kafka consumer in the `analyzer-workers` group:
CONSUME `stube.catalog.item.enriched` -> run the per-file pipeline
(`analyze_one`, which itself runs tidb + chapters + subtitle + blackframe
+ silence + chromaprint and writes segments/chapters + step statuses)
-> PRODUCE `stube.catalog.item.analyzed` for the transcoder. Replicas
scale linearly because the consumer group hands each partition to exactly
one worker; per-item ordering is preserved by keying every event on the
item id.

Idempotency: offsets are committed only AFTER the item is fully processed
AND the next event is produced, so a crash mid-work reprocesses. Rework is
safe because the katalog (item_id, step) unique index + the per-pipeline
step-status short-circuit make a second pass a no-op."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import structlog
from confluent_kafka import Consumer, Producer

from . import kafka
from .katalog import ClaimedItem, KatalogClient
from .pipelines import blackframe, chapters, chromaprint, fuser, keyframe, silence, subtitles, tidb

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

    # Self-thumbnail fallback: for an item TMDB/fanart had no image for, extract a
    # representative keyframe (avoiding intro/credits + black frames, from the
    # segments just fused) and submit it as artwork. Best-effort — never fails the
    # analyze pass.
    if client is not None:
        _maybe_extract_keyframe(item, fused, client)

    return AnalyzeResult(segments=fused, chapters=chapter_atoms)


def _maybe_extract_keyframe(item: ClaimedItem, segments: list[dict], client: KatalogClient) -> None:
    """Fill a missing poster/backdrop from a video keyframe. Episodes get it as
    their BACKDROP (poster stays the series image, so each episode shows a unique
    still); a poster-less movie/series gets it as poster+backdrop. Off via
    KEYFRAME_ARTWORK=false."""
    if os.environ.get("KEYFRAME_ARTWORK", "true").lower() == "false":
        return
    if item.type == "episode":
        kinds = [] if item.has_own_backdrop else ["backdrop"]
    else:  # movie / series
        kinds = [] if (item.has_own_poster or item.has_own_backdrop) else ["poster", "backdrop"]
    if not kinds:
        return
    try:
        jpeg = keyframe.extract(item.path, item.duration_ms, segments)
    except Exception as e:  # noqa: BLE001 - best-effort, must not fail analyze
        log.warning("keyframe.extract_failed", item_id=item.id, error=str(e)[:200])
        return
    if not jpeg:
        log.info("keyframe.no_usable_frame", item_id=item.id)
        return
    for kind in kinds:
        try:
            client.put_artwork(item.id, kind, jpeg, "image/jpeg")
            log.info("keyframe.uploaded", item_id=item.id, kind=kind, bytes=len(jpeg))
        except Exception as e:  # noqa: BLE001
            log.warning("keyframe.upload_failed", item_id=item.id, kind=kind, error=str(e)[:200])


# --- Event-driven consumer -------------------------------------------
# Steps whose terminal status means "this item's analysis pass already
# ran". These are exactly the pipelines analyze_one bookkeeps. When ALL
# of them are already in a terminal state we skip the (expensive) rework
# but STILL produce the analyzed event so the chain isn't stuck. chromaprint
# is intentionally NOT in the guard set: it only exists for multi-episode
# series and legitimately never appears for movies / single-episode items,
# so requiring it would wedge those items forever.
ANALYZER_STEPS = ("tidb", "chapter", "subtitle", "blackframe", "silence")
TERMINAL_STATUSES = {"done", "skipped", "not_applicable", "failed"}


def _already_analyzed(steps: dict[str, str]) -> bool:
    """True when every analyzer step has a terminal status recorded — the
    item was analyzed in a prior pass (e.g. a crash after producing the
    event but before committing the offset). We still re-emit the next
    event so the chain progresses, but skip the ffmpeg work."""
    if not steps:
        return False
    return all(steps.get(name) in TERMINAL_STATUSES for name in ANALYZER_STEPS)


def _process_item(item: ClaimedItem, client: KatalogClient) -> None:
    """Run the per-item analysis and write segments + chapters. Raises on
    an unrecoverable per-item error so the caller can mark it failed."""
    result = analyze_one(item, client=client)
    client.upload_segments(item.id, result.segments)
    client.upload_chapters(item.id, result.chapters)
    log.info(
        "worker.item_done",
        item_id=item.id,
        title=item.title,
        segments=len(result.segments),
        chapters=len(result.chapters),
    )


def run_event_consumer(
    client: KatalogClient,
    brokers: str,
    group_id: str,
    consume_topic: str,
    produce_topic: str,
    security_protocol: str,
    produce_step: str,
    error_sleep: float,
    stop: threading.Event,
) -> None:
    """Blocking Kafka consume->analyze->produce loop. Exits when `stop`
    is set (or the process is killed).

    Per-message flow (replaces the old claim-poll loop):
      1. Parse the value -> itemId. Malformed messages are logged +
         committed + skipped (poison messages must not wedge a partition).
      2. client.get_item(itemId) -> full detail. None/404 => log + commit
         + skip.
      3. Idempotency guard: if every analyzer step is already terminal,
         skip the work but STILL produce the analyzed event, then commit.
      4. Run analyze_one UNCHANGED (all katalog step/segment/chapter writes
         are preserved — they are the state the Activity monitor reads).
      5. Success: produce the analyzed event + flush, THEN commit the
         offset. Failure: mark the item failed, then commit (avoid a
         poison-loop). The offset is only committed once we're done, so a
         crash mid-work reprocesses.
    """
    consumer: Consumer | None = None
    producer: Producer | None = None
    try:
        producer = kafka.build_producer(brokers, security_protocol)
        consumer = kafka.build_consumer(brokers, group_id, security_protocol)
        consumer.subscribe([consume_topic])
        log.info(
            "consumer.started",
            brokers=brokers,
            group_id=group_id,
            consume_topic=consume_topic,
            produce_topic=produce_topic,
        )

        while not stop.is_set():
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            err = msg.error()
            if err is not None:
                if kafka.is_partition_eof(err):
                    continue
                log.warning("consumer.poll_error", error=str(err))
                stop.wait(error_sleep)
                continue

            _handle_message(
                consumer, producer, client, msg, produce_topic, produce_step
            )
    except Exception as e:
        # A failure here is fatal to the consumer thread; main's /readyz
        # goes red once the thread dies so k8s restarts the pod.
        log.exception("consumer.fatal", error=str(e)[:300])
        raise
    finally:
        if producer is not None:
            try:
                producer.flush(10)
            except Exception:
                log.warning("consumer.producer_flush_failed")
        if consumer is not None:
            consumer.close()
        log.info("consumer.stopped")


def _handle_message(
    consumer: Consumer,
    producer: Producer,
    client: KatalogClient,
    msg: Any,
    produce_topic: str,
    produce_step: str,
) -> None:
    """Process one consumed message end-to-end, committing its offset
    exactly once when done (success, skip, or handled failure). Errors
    are contained so one bad item can't kill the loop."""
    item_id = kafka.parse_item_id(msg.value())
    if item_id is None:
        log.warning("consumer.malformed", value=str(msg.value())[:200])
        consumer.commit(message=msg)
        return

    t0 = time.monotonic()
    try:
        item = client.get_item(item_id)
        if item is None:
            log.warning("consumer.item_missing", item_id=item_id)
            consumer.commit(message=msg)
            return

        # Idempotency guard: skip the expensive rework if the item was
        # already fully analyzed, but still emit the next event so the
        # pipeline advances.
        if _already_analyzed(client.get_steps(item.id)):
            log.info("consumer.already_analyzed", item_id=item.id, title=item.title)
        else:
            try:
                _process_item(item, client)
            except FileNotFoundError as e:
                client.fail(item.id, f"file missing: {e}")
                consumer.commit(message=msg)
                return

        # Produce the next-stage event, flush, THEN commit. If we crash
        # between produce and commit the item reprocesses harmlessly.
        _emit_and_commit(
            consumer, producer, msg, item, produce_topic, produce_step
        )
        log.info(
            "consumer.item_complete",
            item_id=item.id,
            seconds=round(time.monotonic() - t0, 2),
        )
    except Exception as e:
        # Unrecoverable per-item error: mark failed and commit so we don't
        # spin on a poison item. The katalog step rows already carry the
        # per-pipeline failures analyze_one recorded.
        log.exception("consumer.item_failed", item_id=item_id, error=str(e)[:300])
        try:
            client.fail(item_id, str(e)[:500])
        except Exception:
            log.exception("consumer.fail_report_failed", item_id=item_id)
        consumer.commit(message=msg)


def _emit_and_commit(
    consumer: Consumer,
    producer: Producer,
    msg: Any,
    item: ClaimedItem,
    produce_topic: str,
    produce_step: str,
) -> None:
    """Produce the analyzed event (keyed by itemId), flush the producer,
    then commit the consumed offset. Ordering matters: the next stage
    must see the event before we forget we processed this item."""
    event = kafka.build_event(
        item.id,
        step=produce_step,
        status="done",
        type_=item.type,
        source="analyzer",
    )
    kafka.produce_event(producer, produce_topic, event)
    producer.flush()
    consumer.commit(message=msg)
