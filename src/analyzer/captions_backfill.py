"""The captions backfill: the thread that reads whether the originals the
catalog hands out carry closed captions, off the event consumer's.

An original that carries closed captions (CEA-608/708 as A53 SEI), or whose
captions are not known, is not retired: its source record comes from a probe
that decodes no frame, and can't tell. The catalog never opens a media file,
so it hands out (POST /api/captions/claim) the originals on disk no read
settled, and the worker reads each (pipelines.captions.read: 48 video packets
decoded in each of its windows - the picture's start, from its first
keyframe, and 10, 30, 50, 70 and 90 % of its own span in - with the
pipeline's ffprobe 7.1, niced and bounded) and reports it with the claim's
token (PUT /api/captions/sources/{sourceId}): window by window, with the
length its file gives and the span the windows were placed over, or why the
read failed.
Only "none" has to be true: a read that can't vouch for its windows is a
failed read, never "no captions", and the catalog keeps the original while it
is. The paths and the JSON are katalog.py's (KatalogClient.claim_captions).

The backfill refuses to start with any other ffprobe than the 7.1 release
(CAPTIONS_FFPROBE, /opt/ffmpeg-7.1/bin/ffprobe in the image; never the PATH's,
the image's 4.4.2): it logs why and asks for nothing.

Which originals, how many an hour and when is the catalog's, so this thread
only asks, as the chapter backfill does: when the catalog hands out nothing it
waits as long as the catalog says. A read is started only before its claim's
deadline (the window's end, or the lease's), and never runs past it: one the
deadline or the worker's stop cuts short gives its original back, which is no
failure of it. Everything the backfill needs to resume lives in the catalog,
so a restart loses at most the original in hand, which the catalog hands out
again once its lease ends.
"""

from __future__ import annotations

import os
import threading
import time

import structlog

from .katalog import CaptionsJob, ClaimTimedOut, KatalogClient
from .pipelines import captions

log = structlog.get_logger(__name__)

# How long to wait when the catalog has no captions backfill (an older one),
# or can't be asked; the bounds of a wait the catalog asks for when it hands
# out nothing (the least also the wait after a claim it did not answer in
# time: it is there, and slow), and the least wait between two claims.
NO_BACKFILL_WAIT_S = 600.0
WAIT_MIN_S, WAIT_MAX_S = 30.0, 3600.0
CLAIM_GAP_S = 1.0
# How long a claim's answer is waited for: well past the catalog's own bound
# of a claim.
CLAIM_TIMEOUT_S = 60.0
# A read is started only this long before its claim's deadline at the
# latest: 48 packets take seconds.
MIN_READ_S = 10.0
# The catalog's answer to a report it can't take (400): that body will never
# be taken, so it is never sent again as it is - the read is reported as a
# failed one in its place, at once, with the same claim's token.
REFUSED = 400
# The catalog's answer to a report it failed to keep (500): the read is
# kept, and sent again when the catalog hands the same file out again, never
# the file read again; how many such reads are kept at most (the oldest go
# first).
UNKEPT = 500
REFUSED_KEPT = 256


class CaptionsBackfill:
    """The captions backfill of one analyzer: run() is its thread. ffprobe
    is the ffprobe it reads with (captions.ffprobe_path() when None)."""

    def __init__(
        self,
        client: KatalogClient,
        stop: threading.Event,
        worker: str,
        *,
        ffprobe: str | None = None,
    ) -> None:
        self.client, self.stop, self.worker = client, stop, worker
        self.ffprobe = ffprobe or captions.ffprobe_path()
        # reads the catalog failed to keep (UNKEPT), by source: the file's
        # identity when it was read, and the read
        self.refused: dict[str, tuple[tuple[int, int, int] | None, captions.CaptionsRead]] = {}

    def refusal(self) -> str | None:
        """Why this backfill does not start: its ffprobe is not the 7.1
        release. None when it starts."""
        return captions.probe(self.ffprobe).refusal()

    def run(self) -> None:
        """Ask, read and report until stop is set; nothing at all with an
        ffprobe other than the 7.1 release."""
        why = self.refusal()
        if why:
            log.error("captions.backfill_refused", worker=self.worker, ffprobe=self.ffprobe,
                      why=why)
            return
        log.info(
            "captions.backfill_started",
            worker=self.worker,
            ffprobe=self.ffprobe,
            probe=captions.probe(self.ffprobe).name,
        )
        wait = 0.0
        while not self.stop.wait(wait):
            try:
                wait = self.claim_once()
            except captions.WrongProbe as e:  # it changed under the worker
                log.error("captions.backfill_refused", worker=self.worker, ffprobe=self.ffprobe,
                          why=str(e))
                return
            except Exception as e:  # the worker outlives any one original
                log.exception("captions.backfill_failed", error=str(e)[:300])
                wait = WAIT_MIN_S
        log.info("captions.backfill_stopped", worker=self.worker)

    def claim_once(self) -> float:
        """One claim: its originals read and reported. Returns how long to
        wait before the next one (0: at once)."""
        try:
            claim = self.client.claim_captions(self.worker, limit=1, timeout=CLAIM_TIMEOUT_S)
        except ClaimTimedOut:
            # not "can't be asked": the catalog is slow, and is asked again soon
            return WAIT_MIN_S
        if claim is None:
            return NO_BACKFILL_WAIT_S
        if not claim.jobs:
            if claim.malformed:
                return CLAIM_GAP_S
            wait = claim.retry_after_s or WAIT_MAX_S
            log.info("captions.backfill_idle", why=claim.idle, wait_s=round(wait))
            return min(max(wait, WAIT_MIN_S), WAIT_MAX_S)
        for job in claim.jobs:
            self.read_job(job, claim.deadline)
        # never a claim straight after a claim: an original the catalog hands
        # out again at once costs a second at least
        return CLAIM_GAP_S

    def read_job(self, job: CaptionsJob, deadline: float | None) -> None:
        """Read one original the catalog handed out (again, where it read it
        already: the newer read replaces the older) and report it with the
        claim's token; give it back unread when its deadline is too near, or
        cuts the read short, or the worker stops, and when the read can't
        name its ffprobe."""
        kept = self.refused.get(job.source_id)
        r: captions.CaptionsRead | None
        if kept is not None and kept[0] is not None and kept[0] == _identity(job.path):
            # the catalog failed to keep this read of this very file: sent
            # again, the file not read again
            r = kept[1]
            log.info("captions.backfill_refused_read_again", source_id=job.source_id)
        else:
            try:
                r = self._read(job, deadline)
            except captions.WrongProbe:
                self._release(job, "its ffprobe is not the pipeline's 7.1")
                raise
            except Exception as e:  # a read that broke is reported failed, never left
                log.exception("captions.backfill_read_broke", source_id=job.source_id,
                              error=str(e)[:300])  # fmt: skip
                name = captions.probe(self.ffprobe).name
                if not name:
                    self._release(job, "its read broke, and its ffprobe can't be named")
                    return
                r = captions.CaptionsRead(
                    path=job.path, ok=False, probe=name, unreadable=True,
                    error=f"the read broke: {type(e).__name__}: {e}"[:300],
                )  # fmt: skip
        if r is None:
            return  # given back
        if not r.probe:
            # no result without the ffprobe that read it
            log.error("captions.backfill_unnamed_ffprobe", source_id=job.source_id,
                      ffprobe=self.ffprobe)  # fmt: skip
            self._release(job, "its read can't name its ffprobe")
            return
        for w in r.windows:
            if w.error:
                log.warning("captions.backfill_window_failed", source_id=job.source_id,
                            window=w.label, why=w.error)  # fmt: skip
        status = self.client.report_captions(job, r)
        if status == REFUSED:
            # a body the catalog can't take is no report of the read: never
            # sent again as it is, it is reported as a failed read instead
            self.refused.pop(job.source_id, None)
            log.warning("captions.backfill_report_refused", source_id=job.source_id, status=status)
            r = refused_read(r)
            status = self.client.report_captions(job, r)
            if status is None or status >= 400:
                # not even that is taken: its lease ends, and the catalog
                # hands it out again to be read again
                log.warning("captions.backfill_report_failed", source_id=job.source_id,
                            status=status)  # fmt: skip
                return
        if status == UNKEPT:
            # kept: the catalog hands the original out again once its lease
            # ends, and this read is what is sent then (the same file)
            self._keep_refused(job, r)
            log.warning("captions.backfill_report_unkept", source_id=job.source_id, status=status)
            return
        if status is None or status >= 400:
            # its lease ends, and the catalog hands it out again
            log.warning("captions.backfill_report_failed", source_id=job.source_id, status=status)
            return
        self.refused.pop(job.source_id, None)
        log.info(
            "captions.backfill_read",
            source_id=job.source_id,
            item_id=job.item_id,
            captions=r.captions,
            windows=len(r.windows),
            frames=r.frames,
            error=r.error,
            seconds=round(r.seconds, 2),
        )

    def _read(self, job: CaptionsJob, deadline: float | None) -> captions.CaptionsRead | None:
        """A read of job's original, no longer than its deadline; None when
        the original is given back instead (its deadline too near, or cutting
        the read short; the worker stopping)."""
        limit = captions.read_timeout()
        cut = False
        if deadline is not None:
            left = deadline - time.time()
            if left < MIN_READ_S:
                self._release(job, "its deadline is too near")
                return None
            if left < limit:
                limit, cut = left, True
        if self.stop.is_set():
            self._release(job, "the worker stops")
            return None
        r = captions.read(
            job.path,
            # the plan's count; a claim that names another is failed by the read
            packets=captions.PACKETS if job.packets is None else job.packets,
            timeout=limit,
            stop=self.stop.is_set,
            ffprobe=self.ffprobe,
            duration_ms=job.duration_ms,
        )
        if r.interrupted or (r.timed_out and cut):
            self._release(job, r.error or "")
            return None
        return r

    def _release(self, job: CaptionsJob, why: str) -> None:
        self.client.release_captions(job)
        log.info("captions.backfill_released", source_id=job.source_id, why=why)

    def _keep_refused(self, job: CaptionsJob, r: captions.CaptionsRead) -> None:
        """Keep a read the catalog failed to keep (UNKEPT), with its file's
        identity now (a file that is another then is read again),
        REFUSED_KEPT at most."""
        if job.source_id not in self.refused and len(self.refused) >= REFUSED_KEPT:
            self.refused.pop(next(iter(self.refused)))
        self.refused[job.source_id] = (_identity(job.path), r)


def refused_read(r: captions.CaptionsRead) -> captions.CaptionsRead:
    """The failed read reported in place of a read whose report the catalog
    refused (400): what it can't take is a read that says nothing, and is not
    known - only the ffprobe that read it and the size of the file read, which
    the catalog checks for themselves."""
    return captions.CaptionsRead(
        path=r.path, ok=False, probe=r.probe, unreadable=True, file_size=r.file_size,
        error="the catalog refused its report (400): a report it can't take is a failed read",
    )  # fmt: skip


def _identity(path: str) -> tuple[int, int, int] | None:
    """A file's inode, time and size; None when it can't be told."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_ino, st.st_mtime_ns, st.st_size


def enabled() -> bool:
    """Whether this analyzer asks the catalog for originals to read
    (CAPTIONS_BACKFILL, true unless false): the catalog decides whether it
    hands any out."""
    return os.environ.get("CAPTIONS_BACKFILL", "true").lower() != "false"
