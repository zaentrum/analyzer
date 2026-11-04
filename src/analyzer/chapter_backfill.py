"""The chapter backfill: the thread that reads the chapters of the titles the
catalog hands out, off the event consumer's.

The catalog hands out (POST /api/chapters/claim) the titles that miss their
chapters: an original whose record says it has chapters, none of them in the
catalog, and no read of that original that settled it — a title never
analyzed, one whose read failed, one an older analyzer read. The worker reads
each original's header (pipelines.chapters.read: ffprobe's chapters, nothing
decoded) and reports what it found with the claim's token (PUT
/api/chapters/items/{id}): the chapters with the read, a read whose header
holds none, or why it failed. Which titles, how many an hour and when is the
catalog's (its settings chapters.backfill*: off by default, never from 17:00
to 22:00 UTC by default), so this thread only asks: when the catalog hands
out nothing it waits as long as the catalog says. A read is started only
before its claim's deadline (the window's end, or the lease's), and never runs
past it: one the deadline or the worker's stop cuts short gives its title
back, which is no failure of the title. Everything the backfill needs to
resume lives in the catalog, so a restart loses at most the title in hand,
which the catalog hands out again once its lease ends.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any

import structlog

from .katalog import ClaimTimedOut, KatalogClient
from .pipelines import chapters

log = structlog.get_logger(__name__)

# How long to wait when the catalog has no chapter backfill (an older one),
# or can't be asked; the bounds of a wait the catalog asks for when it hands
# out nothing (the least also the wait after a claim it did not answer in
# time: it is there, and slow); the wait after a claim whose titles the
# catalog settled from their source records alone, and the least wait between
# two claims.
NO_BACKFILL_WAIT_S = 600.0
WAIT_MIN_S, WAIT_MAX_S = 30.0, 3600.0
SETTLED_WAIT_S = 1.0
CLAIM_GAP_S = 1.0
# How long a claim's answer is waited for: well past the catalog's own bound
# of a claim (20 s, whatever its storage does).
CLAIM_TIMEOUT_S = 60.0
# A read is started only this long before its claim's deadline at the
# latest: a header takes seconds.
MIN_READ_S = 10.0


class ChapterBackfill:
    """The chapter backfill of one analyzer: run() is its thread."""

    def __init__(self, client: KatalogClient, stop: threading.Event, worker: str) -> None:
        self.client, self.stop, self.worker = client, stop, worker

    def run(self) -> None:
        """Ask, read and report until stop is set."""
        log.info("chapters.backfill_started", worker=self.worker)
        wait = 0.0
        while not self.stop.wait(wait):
            try:
                wait = self.claim_once()
            except Exception as e:  # the worker outlives any one title
                log.exception("chapters.backfill_failed", error=str(e)[:300])
                wait = WAIT_MIN_S
        log.info("chapters.backfill_stopped", worker=self.worker)

    def claim_once(self) -> float:
        """One claim: its titles read and reported. Returns how long to wait
        before the next one (0: at once)."""
        try:
            claim = self.client.claim_chapters(self.worker, limit=1, timeout=CLAIM_TIMEOUT_S)
        except ClaimTimedOut:
            # not "can't be asked": the catalog is slow (its storage, most
            # often), and is asked again soon
            return WAIT_MIN_S
        if claim is None:
            return NO_BACKFILL_WAIT_S
        jobs: list[dict[str, Any]] = claim.get("items") or []
        if not jobs:
            if _int(claim.get("settled")) > 0:
                # the catalog settled titles from their records: more may wait
                log.info("chapters.backfill_settled", settled=claim.get("settled"))
                return SETTLED_WAIT_S
            try:
                wait = float(claim.get("retryAfterSeconds") or WAIT_MAX_S)
            except (TypeError, ValueError):
                wait = WAIT_MAX_S
            log.info("chapters.backfill_idle", why=claim.get("idle"), wait_s=round(wait))
            return min(max(wait, WAIT_MIN_S), WAIT_MAX_S)
        deadline = _time(claim.get("deadline"))
        for job in jobs:
            self.read_job(job, deadline)
        # never a claim straight after a claim: a title the catalog hands out
        # again at once costs a second at least
        return CLAIM_GAP_S

    def read_job(self, job: dict[str, Any], deadline: float | None) -> None:
        """Read one title the catalog handed out and report it with its
        claim's token; give it back unread when its deadline is too near, or
        the worker stops."""
        item_id, path = str(job.get("id") or ""), str(job.get("path") or "")
        token = str(job.get("token") or "")
        if not item_id or not path:
            log.warning("chapters.backfill_malformed", job=str(job)[:300])
            return
        limit = chapters.read_timeout()
        cut = False
        if deadline is not None:
            left = deadline - time.time()
            if left < MIN_READ_S:
                self._release(item_id, token, "its deadline is too near")
                return
            if left < limit:
                limit, cut = left, True
        if self.stop.is_set():
            self._release(item_id, token, "the worker stops")
            return
        duration = job.get("durationMs")
        r = chapters.read(
            path,
            duration_ms=duration if isinstance(duration, int) else None,
            timeout=limit,
            stop=self.stop.is_set,
        )
        if r.interrupted or (r.timed_out and cut):
            self._release(item_id, token, r.error or "")
            return
        source = str(job.get("sourceId") or "") or None
        try:
            if r.ok:
                self.client.put_chapters(item_id, r.chapters, read=r.report(source), token=token)
            else:
                self.client.report_chapters_failed(item_id, r.report(source), token=token)
        except Exception as e:  # its lease ends, and the catalog hands it out again
            log.warning("chapters.backfill_report_failed", item_id=item_id, error=str(e)[:300])
            return
        log.info(
            "chapters.backfill_read",
            item_id=item_id,
            ok=r.ok,
            found=r.found,
            chapters=len(r.chapters),
            seconds=round(r.seconds, 2),
        )

    def _release(self, item_id: str, token: str, why: str) -> None:
        self.client.release_chapters(item_id, token)
        log.info("chapters.backfill_released", item_id=item_id, why=why)


def enabled() -> bool:
    """Whether this analyzer asks the catalog for titles to read
    (CHAPTER_BACKFILL, true unless false): the catalog decides whether it
    hands any out."""
    return os.environ.get("CHAPTER_BACKFILL", "true").lower() != "false"


def _int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _time(v: Any) -> float | None:
    """An RFC 3339 time as seconds since the epoch; None when there is none."""
    if not isinstance(v, str) or not v:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
