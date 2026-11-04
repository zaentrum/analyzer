"""Read a file's chapters from its header via ffprobe.

`read` is what the catalog keeps of a title's chapters: the header of the
original, read with ffprobe and nothing decoded (-nofind_stream_info: the
container's own chapter marks, its format's duration and its streams'
durations, as the header gives them). It says whether the header could be
read at all — a read that failed is no file without chapters — how many
chapter marks it held, and the chapters of them that have a length.

`detect` is the analyze pass's: the same read, split two ways.

  * `segments` — chapters whose title matches one of the TIDB skippable
    kinds (intro / recap / credits). These ride the normal fuser flow
    and end up in com_nalet_katalog_mediasegments alongside detections
    from the other pipelines. An intro or recap chapter says where that
    segment starts; its range runs to the next chapter mark, which in a
    file with coarse chapters is the end credits. The fuser takes the
    range only when it is a plausible length (fuser.SPAN_MS), and
    otherwise ends the segment where another pipeline does near that
    start, or drops it.
  * `chapters` — every chapter with a length, unchanged, including the
    unlabeled "Act 1" / "Cold Open" markers that don't map to a TIDB kind.
    These bypass the fuser and go straight to com_nalet_katalog_itemchapters
    via PUT /api/chapters, with what the read was. They're descriptive
    metadata, not skippable moments, so mixing them with TIDB-aligned
    segments was always a nomenclature mismatch — see migration 018.

A mark the header gives no end (Matroska written without ChapterTimeEnd, an
MP4's Nero chapters) ends where the next mark starts, the last where the
file does (its format's duration, else its longest stream's, else the
catalog's length of the title): what ffprobe works out itself once it
decodes. A mark that still has no length is left out, and counted.

ffprobe is the highest-confidence signal we have: chapters were set by
the encoder and require zero analysis, so when they exist the player
can render a full timeline and the analyzer doesn't have to fall back
to acoustic / visual detectors for structure."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

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

# The header only: the container's chapters, its duration and its streams'
# durations, with no stream probed or decoded.
FFPROBE_ARGS = (
    "-v",
    "error",
    "-nofind_stream_info",
    "-print_format",
    "json",
    "-show_chapters",
    "-show_entries",
    "format=duration,start_time:stream=duration",
)
# How long a read may take: a header on a storage that answers slowly (an
# MKV's attachments are read with it) takes seconds, a stalled one longer.
# CHAPTERS_READ_TIMEOUT_S overrides it.
READ_TIMEOUT_S = 120.0
# The most characters a chapter's title keeps (the catalog's column).
TITLE_MAX = 120


@dataclass
class ChapterRead:
    """What a read of a file's header found: whether it could be read (ok;
    error says why not), the chapter marks it held (found), and those with a
    length (chapters, in the file's order, each with its ordinal)."""

    path: str
    ok: bool
    chapters: list[dict] = field(default_factory=list)
    found: int = 0
    error: str | None = None
    note: str | None = None
    seconds: float = 0.0
    probe: str | None = None
    file_size: int | None = None
    # the read ran out of its time (timed_out), or was stopped (interrupted)
    timed_out: bool = False
    interrupted: bool = False

    def report(self) -> dict[str, Any]:
        """The read as the catalog keeps it (PUT /api/chapters, "read")."""
        out: dict[str, Any] = {"path": self.path, "seconds": round(self.seconds, 3)}
        if self.probe:
            out["probe"] = self.probe
        if self.file_size is not None:
            out["fileSize"] = self.file_size
        if self.ok:
            out["found"] = self.found
            if self.note:
                out["note"] = self.note
        else:
            out["error"] = self.error or "the header could not be read"
        return out


@dataclass
class ChapterResult:
    """Output of the chapter pipeline: two lists posted to two endpoints,
    and the read they came of (None: a result made by hand, read as its
    chapters say)."""

    segments: list[dict]  # TIDB-kind matches, flow through the fuser
    chapters: list[dict]  # every chapter with a length, posted to /api/chapters
    read: ChapterRead | None = None

    @property
    def ok(self) -> bool:
        """Whether the header could be read (its chapters are what it holds)."""
        return self.read is None or self.read.ok

    @property
    def error(self) -> str | None:
        return None if self.read is None else self.read.error

    def report(self, path: str) -> dict[str, Any]:
        """The read as the catalog keeps it."""
        if self.read is not None:
            return self.read.report()
        return {"path": path, "found": len(self.chapters)}


def read(
    path: str,
    *,
    duration_ms: int | None = None,
    timeout: float | None = None,
    stop: Callable[[], bool] | None = None,
) -> ChapterRead:
    """Read the chapters of `path`'s header. duration_ms is the catalog's
    length of the title: the end of a last mark the header gives none, when
    the header says no length of the file either. A read that runs longer
    than timeout (READ_TIMEOUT_S), or while stop() says to stop, is ended:
    it failed (timed_out, or interrupted), saying so."""
    t0 = time.monotonic()
    done = ChapterRead(path=path, ok=False, probe=probe_version())

    def failed(error: str) -> ChapterRead:
        done.error, done.seconds = error[:500], time.monotonic() - t0
        log.warning("chapters.read_failed", path=path, error=done.error)
        return done

    try:
        done.file_size = os.stat(path).st_size
    except OSError as e:
        return failed(f"no file to read: {e.strerror or e}")
    limit = read_timeout() if timeout is None else timeout
    try:
        # its own session: a read that is ended takes whatever it started
        proc = subprocess.Popen(
            ["ffprobe", *FFPROBE_ARGS, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as e:
        return failed(f"ffprobe could not run: {e}")
    end = t0 + limit
    while True:
        try:
            out, err = proc.communicate(timeout=min(1.0, max(end - time.monotonic(), 0.01)))
            break
        except subprocess.TimeoutExpired:
            if stop is not None and stop():
                done.interrupted = True
            elif time.monotonic() >= end:
                done.timed_out = True
            else:
                continue
            _kill(proc)
            if done.interrupted:
                return failed("the read was stopped")
            return failed(f"ffprobe did not answer within {limit:g} s")
    if proc.returncode != 0:
        why = " ".join((err or "").split())[-300:]
        return failed(f"ffprobe exited {proc.returncode}" + (f": {why}" if why else ""))
    try:
        probed = json.loads(out or "{}")
    except json.JSONDecodeError as e:
        return failed(f"ffprobe's output is no JSON: {e}")
    if not isinstance(probed, dict):
        return failed("ffprobe's output is no JSON object")

    raw = probed.get("chapters") or []
    if not isinstance(raw, list):
        raw = []
    done.ok, done.found = True, len(raw)
    done.chapters = marks(raw, _end_of(probed, duration_ms))
    left = done.found - len(done.chapters)
    if left:
        done.note = f"{left} of {done.found} chapter marks have no length, left out"
    done.seconds = time.monotonic() - t0
    log.info(
        "chapters.read",
        path=path,
        found=done.found,
        chapters=len(done.chapters),
        seconds=round(done.seconds, 2),
    )
    return done


def _kill(proc: subprocess.Popen) -> None:
    """End a read: its process group killed, its pipes closed."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        log.warning("chapters.read_unkilled", pid=proc.pid)


def detect(path: str, duration_ms: int | None = None) -> ChapterResult:
    """Read the chapters of `path`'s header and split them (see the module)."""
    r = read(path, duration_ms=duration_ms)
    segments: list[dict] = []
    for ch in r.chapters:
        # When the title matches a TIDB kind, also surface it as a
        # high-confidence segment so the fuser can use it (and the
        # player's skip-intro / skip-credits buttons fire on it). The
        # range is the chapter's, as the file has it: the fuser judges
        # whether an intro or a recap can be that long.
        kind = _classify_chapter(ch["title"] or "")
        if kind is not None:
            segments.append(
                {
                    "kind": kind,
                    "startMs": ch["startMs"],
                    "endMs": ch["endMs"],
                    "source": "chapter",
                    "confidence": 1.00,
                    "label": ch["title"],
                }
            )
    log.info(
        "chapters.detected",
        path=path,
        ok=r.ok,
        count=len(r.chapters),
        labeled_as_segments=len(segments),
    )
    return ChapterResult(segments=segments, chapters=r.chapters, read=r)


def marks(raw: list[Any], end_ms: int | None) -> list[dict]:
    """The chapters of ffprobe's chapter marks with a length, in their order,
    each numbered (ordinal). A mark without an end ends where the next later
    mark starts, else at end_ms (the file's end); a start before 0 is 0."""
    starts: list[int | None] = [
        _to_ms(ch.get("start_time")) if isinstance(ch, dict) else None for ch in raw
    ]
    chapters: list[dict] = []
    for i, ch in enumerate(raw):
        start = starts[i]
        if start is None:
            continue
        end = _to_ms(ch.get("end_time"))
        if end is None:
            later = [s for j, s in enumerate(starts) if j != i and s is not None and s > start]
            if end_ms is not None:
                later.append(end_ms)
            end = min(later) if later else None
        start = max(start, 0)
        if end is None or end <= start:
            continue
        chapters.append(
            {
                "startMs": start,
                "endMs": end,
                "title": _title((ch.get("tags") or {}).get("title")),
                # ordinal preserves source-file ordering so the UI can render
                # a contiguous bar without re-sorting by startMs
                "ordinal": len(chapters) + 1,
            }
        )
    return chapters


def _end_of(probed: dict, duration_ms: int | None) -> int | None:
    """Where the file ends, in ms: its format's duration (from its start),
    else its longest stream's, else the catalog's length of the title."""
    fmt = probed.get("format") or {}
    duration = _to_ms(fmt.get("duration"))
    if duration is not None and duration > 0:
        return duration + max(_to_ms(fmt.get("start_time")) or 0, 0)
    longest = max(
        (
            _to_ms(s.get("duration")) or 0
            for s in probed.get("streams") or []
            if isinstance(s, dict)
        ),
        default=0,
    )
    if longest > 0:
        return longest
    return duration_ms if duration_ms and duration_ms > 0 else None


def _title(t: Any) -> str | None:
    """A chapter's title as the catalog keeps it: no control characters, at
    most TITLE_MAX characters, None when nothing is left."""
    if not isinstance(t, str):
        return None
    kept = (c for c in t if c in "\t\n\r" or (c >= " " and c != "\x7f"))
    s = "".join(" " if c in "\t\n\r" else c for c in kept)
    s = s.strip()[:TITLE_MAX].strip()
    return s or None


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


def _to_ms(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(float(v) * 1000)
    except (TypeError, ValueError, OverflowError):
        return None


def read_timeout() -> float:
    """How long a read may take: READ_TIMEOUT_S, or CHAPTERS_READ_TIMEOUT_S
    (5 s to an hour)."""
    try:
        v = float(os.environ.get("CHAPTERS_READ_TIMEOUT_S") or READ_TIMEOUT_S)
    except ValueError:
        return READ_TIMEOUT_S
    return min(max(v, 5.0), 3600.0)


@lru_cache(maxsize=1)
def probe_version() -> str | None:
    """ffprobe's version line ("ffprobe version 4.4.2-0ubuntu0.22.04.1 ..."),
    the reader the catalog keeps with a read; None when it can't say."""
    try:
        r = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = (r.stdout or "").splitlines()[0] if r.returncode == 0 and r.stdout else ""
    return " ".join(line.split()[:3])[:200] or None
