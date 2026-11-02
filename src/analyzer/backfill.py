"""The picture-area backfill: the titles the catalog has no picture area of.

The catalog hands them out at its own pace (POST /api/picture/claim): its
settings say whether it does, when (never in its quiet hours), how many at
once and how many an hour, and it hands out a title's package where it has
one (the video rendition's playlist), else its original. This loop asks,
measures what it is handed (pipelines/picture.py), reports each title, and
when the catalog hands out nothing waits as long as it says. A claim's
deadline (the quiet hours' start, or the end of the claim's lease) stops a
measurement still running then: its title is given back unmeasured and
handed out again later. Everything it needs to resume lives in the catalog,
so a restart of either side loses at most the titles in hand, which the
catalog hands out again once their lease ends.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

import structlog

from .katalog import KatalogClient
from .pipelines import picture

log = structlog.get_logger(__name__)

# How long to wait when the catalog has no backfill (an older one), or can't
# be asked.
NO_BACKFILL_WAIT_S = 600.0
# Bounds of a wait the catalog asks for when it hands out nothing.
WAIT_MIN_S, WAIT_MAX_S = 30.0, 3600.0


def run(client: KatalogClient, stop: threading.Event, worker: str) -> None:
    """Measure what the catalog hands out until stop is set."""
    log.info("backfill.started", worker=worker)
    while not stop.is_set():
        try:
            wait = once(client, stop, worker)
        except Exception as e:  # the loop outlives any one title
            log.exception("backfill.failed", error=str(e)[:300])
            wait = WAIT_MIN_S
        if wait > 0:
            stop.wait(wait)
    log.info("backfill.stopped", worker=worker)


def once(client: KatalogClient, stop: threading.Event, worker: str) -> float:
    """One claim: its titles measured and reported. Returns how long to wait
    before the next one (0: at once)."""
    claim = client.claim_picture_areas(worker)
    if claim is None:
        return NO_BACKFILL_WAIT_S
    jobs: list[dict[str, Any]] = claim.get("items") or []
    if not jobs:
        try:
            wait = float(claim.get("retryAfterSeconds") or WAIT_MAX_S)
        except (TypeError, ValueError):
            wait = WAIT_MAX_S
        log.info("backfill.idle", why=claim.get("idle"), wait_s=round(wait))
        return min(max(wait, WAIT_MIN_S), WAIT_MAX_S)
    deadline = _time(claim.get("deadline"))
    for job in jobs:
        if stop.is_set():
            break
        measure_one(client, job, deadline, stop)
    return 0.0


def measure_one(
    client: KatalogClient, job: dict[str, Any], deadline: float | None, stop: threading.Event
) -> None:
    """Measure one title the catalog handed out and report it: given back
    unmeasured when its deadline passes or the worker stops."""
    kind = "extras" if job.get("kind") == "extra" else "items"
    item_id, path = str(job.get("id") or ""), str(job.get("path") or "")
    if not item_id or not path:
        log.warning("backfill.malformed", job=str(job)[:300])
        return
    m = picture.measure(path, job.get("durationMs"), deadline=deadline, stop=stop.is_set)
    if m.interrupted:
        client.put_picture_area(kind, item_id, {"released": True, "path": path})
        log.info("backfill.released", item_id=item_id, kind=kind)
        return
    client.put_picture_area(kind, item_id, m.report())


def _time(v: Any) -> float | None:
    """An RFC 3339 time as seconds since the epoch; None when there is none."""
    if not isinstance(v, str) or not v:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
