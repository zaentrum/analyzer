"""The picture worker: the one thread that measures picture areas
(pipelines/picture), off the event consumer's.

Two kinds of title come to it:

  * the analyze pass's: once the consumer has sent a movie's or an
    episode's analyzed event it hands the title over (submit, which never
    waits) and goes on with the next message. The worker measures it
    unless the catalog has an area of what the title plays now, of this
    measurement's method (GET /api/picture/items/{id}), and reports it
    (PUT, without a token). These go first, PASS_DEADLINE_S at most each
    (PICTURE_PASS_DEADLINE_S); QUEUE_MAX of them wait at most, and a title
    that finds the queue full is left to the backfill.
  * the backfill's: while none of those waits, it asks the catalog for the
    titles without a picture area of what they play now (POST
    /api/picture/claim, naming its method), measures each and reports it
    with its claim's token. A measurement still running at its claim's
    deadline (the quiet hours' start, or the lease's end) gives its title
    back ({"released": true}), but the last run the catalog allows a title
    (the job's last: its runs before had their whole lease and did not
    finish either): that one reports the title too costly to measure, an
    unknown of what it read (its frame and video from a probe of the file,
    sampling.doubt "cost"). When the catalog hands out nothing it waits as
    long as the catalog says, or until a pass's title comes.

A measurement runs in its own process at a lower priority (measure.py); the
worker stops it at its deadline, and when the worker stops: a claim's title
is given back then. Everything the backfill needs to resume lives in the
catalog, so a restart loses at most the titles in hand, which the catalog
hands out again.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from .katalog import ClaimedItem, KatalogClient
from .pipelines import picture

log = structlog.get_logger(__name__)

# The analyze pass's titles waiting at most, and how long one's measurement
# runs at most: as the catalog's lease, the whole measurement of a 3-hour 4K
# remux with room to spare.
QUEUE_MAX = 256
PASS_DEADLINE_S = 6 * 3600.0
# How long to wait when the catalog has no backfill (an older one), or can't
# be asked; the bounds of a wait the catalog asks for when it hands out
# nothing.
NO_BACKFILL_WAIT_S = 600.0
WAIT_MIN_S, WAIT_MAX_S = 30.0, 3600.0
# How long a probe of a title too costly to measure runs at most.
PROBE_S = 120.0


@dataclass(frozen=True)
class PassTitle:
    """A title the analyze pass handed over: a movie's or an episode's
    original."""

    item_id: str
    path: str
    duration_ms: int | None


class PictureWorker:
    """The picture worker of one analyzer: run() is its thread, submit()
    hands it the analyze pass's titles. passes and backfill say which kind
    of title it measures (PICTURE_AREA, PICTURE_BACKFILL)."""

    def __init__(
        self,
        client: KatalogClient,
        stop: threading.Event,
        worker: str,
        *,
        passes: bool = True,
        backfill: bool = True,
    ) -> None:
        self.client, self.stop, self.worker = client, stop, worker
        self.passes, self.backfill = passes, backfill
        self._queue: queue.Queue[PassTitle] = queue.Queue(maxsize=QUEUE_MAX)

    def submit(self, item: ClaimedItem) -> bool:
        """Hand over a title of the analyze pass, at once: False when it is
        no movie or episode, the passes' titles are off, or the queue is
        full (the backfill measures it later)."""
        if not self.passes or item.type not in ("movie", "episode"):
            return False
        try:
            self._queue.put_nowait(PassTitle(item.id, item.path, item.duration_ms))
        except queue.Full:
            log.warning("picture.queue_full", item_id=item.id, waiting=QUEUE_MAX)
            return False
        return True

    def run(self) -> None:
        """Measure what comes until stop is set."""
        log.info(
            "picture.worker_started", worker=self.worker, passes=self.passes, backfill=self.backfill
        )
        wait = 0.0
        while not self.stop.is_set():
            title = self._next(wait)
            if self.stop.is_set():
                break
            try:
                if title is not None:
                    self.measure_pass(title)
                    wait = 0.0
                elif self.backfill:
                    wait = self.claim_once()
                else:
                    wait = WAIT_MAX_S
            except Exception as e:  # the worker outlives any one title
                log.exception("picture.worker_failed", error=str(e)[:300])
                wait = WAIT_MIN_S
        log.info("picture.worker_stopped", worker=self.worker)

    def _next(self, wait: float) -> PassTitle | None:
        """The next title of the analyze pass, waiting up to wait seconds
        for one (and no longer than stop is unset)."""
        end = time.monotonic() + wait
        while True:
            left = end - time.monotonic()
            try:
                if left <= 0:
                    return self._queue.get_nowait()
                return self._queue.get(timeout=min(left, 1.0))
            except queue.Empty:
                if left <= 1.0 or self.stop.is_set():
                    return None

    def measure_pass(self, title: PassTitle) -> None:
        """Measure a title of the analyze pass and report it, unless the
        catalog has an area of what it plays now of this method."""
        have = self.client.picture_area("items", title.item_id)
        if (
            have
            and have.get("current")
            and have.get("state") in ("measured", "unknown")
            and have.get("method") == picture.METHOD
        ):
            log.info("picture.current", item_id=title.item_id, state=have.get("state"))
            return
        m = picture.measure(
            title.path,
            title.duration_ms,
            deadline=time.time() + _pass_deadline(),
            stop=self.stop.is_set,
        )
        if m.interrupted:
            # the backfill hands it out later
            log.info("picture.pass_stopped", item_id=title.item_id)
            return
        self.client.put_picture_area("items", title.item_id, m.report())

    def claim_once(self) -> float:
        """One claim: its titles measured and reported. Returns how long to
        wait before the next one (0: at once)."""
        claim = self.client.claim_picture_areas(self.worker, method=picture.METHOD)
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
            self.measure_job(job, deadline)
        return 0.0

    def measure_job(self, job: dict[str, Any], deadline: float | None) -> None:
        """Measure one title the catalog handed out and report it with its
        claim's token: given back unmeasured when its deadline passes or
        the worker stops."""
        kind = "extras" if job.get("kind") == "extra" else "items"
        item_id, path = str(job.get("id") or ""), str(job.get("path") or "")
        token = str(job.get("token") or "")
        if not item_id or not path:
            log.warning("backfill.malformed", job=str(job)[:300])
            return
        m = None
        if not self.stop.is_set():
            m = picture.measure(
                path, job.get("durationMs"), deadline=deadline, stop=self.stop.is_set
            )
        if m is not None and m.interrupted and job.get("last") and not self.stop.is_set():
            body = self._too_costly(path, job, m)
            if body is not None:
                body["token"] = token
                self.client.put_picture_area(kind, item_id, body)
                log.info("backfill.too_costly", item_id=item_id, kind=kind, s=round(m.seconds))
                return
        if m is None or m.interrupted:
            self.client.put_picture_area(
                kind, item_id, {"released": True, "path": path, "token": token}
            )
            log.info("backfill.released", item_id=item_id, kind=kind)
            return
        body = m.report()
        body["token"] = token
        self.client.put_picture_area(kind, item_id, body)

    def _too_costly(
        self, path: str, job: dict[str, Any], m: picture.Measurement
    ) -> dict[str, Any] | None:
        """The report of a title too costly to measure: what a probe of its
        file says it is, the measurement's time and CPU, and why. None when
        the worker stops meanwhile (the title is given back)."""
        p = picture.measure(
            path, job.get("durationMs"), deadline=time.time() + PROBE_S, stop=self.stop.is_set,
            probe=True,
        )  # fmt: skip
        if p.interrupted:
            return None
        runs = int(job.get("costlyRuns") or 0) + 1
        p.seconds, p.cpu_seconds = m.seconds, m.cpu_seconds
        p.sampling = {**p.sampling, "doubt": ["cost"], "costlyRuns": runs}
        if not p.error:
            p.reason = (
                f"can't tell: too costly to measure: {runs} runs of {round(m.seconds / 60)} min "
                "each did not finish"
            )
        return p.report()


def _pass_deadline() -> float:
    try:
        v = float(os.environ.get("PICTURE_PASS_DEADLINE_S") or PASS_DEADLINE_S)
    except ValueError:
        return PASS_DEADLINE_S
    return min(max(v, 60.0), 24 * 3600.0)


def _time(v: Any) -> float | None:
    """An RFC 3339 time as seconds since the epoch; None when there is none."""
    if not isinstance(v, str) or not v:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
