"""measure(): a measurement in its own process (child.py), at a lower
priority, stopped at its deadline or when the worker stops; its CPU and its
peak memory are read when it is reaped."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from .rules import Box

log = structlog.get_logger(__name__)

# The measurement's version, kept with every area.
METHOD = "picture/4"
# A measurement without a deadline of its own stops after this long.
MAX_SECONDS = 3600.0
# How often the parent looks at its deadline, and at stop.
POLL_S = 0.2
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class Measurement:
    """What a measurement found of the video at path.

    frame is its coded size (a Box at 0, 0), area the picture's within it:
    the frame itself when there are no bars, None when the measurement can't
    tell (reason says why) or failed (error says why). sides says, per side
    (top, bottom, left, right), where its edge is and why. samples are the
    frames decoded, informative those that show a picture; sampling how
    they were taken (their largest gap, the bytes read); measured_on what
    the area is of (the original or the package, its size and time);
    seconds the time it took, cpu_seconds the CPU and max_rss_bytes the peak
    memory of its process."""

    path: str
    frame: Box | None = None
    area: Box | None = None
    reason: str | None = None
    error: str | None = None
    samples: int = 0
    informative: int = 0
    sides: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    video: dict[str, Any] = field(default_factory=dict)
    measured_on: dict[str, Any] = field(default_factory=dict)
    decoder: str = ""
    seconds: float = 0.0
    cpu_seconds: float = 0.0
    max_rss_bytes: int = 0
    # The measurement was stopped (its deadline passed, the worker stops):
    # it says nothing, and the title is measured again later.
    interrupted: bool = False

    @property
    def full(self) -> bool:
        """The picture fills the frame: there are no bars."""
        return self.area is not None and self.area == self.frame

    @property
    def duration_s(self) -> float:
        return float(self.sampling.get("durationS") or 0.0)

    @property
    def bytes_read(self) -> int:
        return int(self.sampling.get("bytesRead") or 0)

    def report(self) -> dict[str, Any]:
        """The catalog's body for it (PUT /api/picture/items/{id})."""
        frame = area = None
        if self.frame is not None:
            frame = {"width": self.frame.width, "height": self.frame.height}
        if self.area is not None:
            a = self.area
            area = {"x": a.x, "y": a.y, "width": a.width, "height": a.height}
        body: dict[str, Any] = {
            "path": self.path,
            "method": METHOD,
            "frame": frame,
            "area": area,
            "samples": self.samples,
            "informative": self.informative,
            "durationMs": round(self.duration_s * 1000),
            "seconds": round(self.seconds, 1),
            "cpuSeconds": round(self.cpu_seconds, 1),
            "bytesRead": self.bytes_read,
            "measuredOn": self.measured_on,
            "sampling": self.sampling,
            "video": self.video,
            "decoder": self.decoder,
            "sides": self.sides,
        }
        if self.reason:
            body["reason"] = self.reason[:500]
        if self.error:
            body["error"] = self.error[:500]
        return body


def measure(
    path: str,
    duration_ms: int | None = None,
    *,
    deadline: float | None = None,
    stop: Callable[[], bool] | None = None,
    probe: bool = False,
) -> Measurement:
    """Measure the picture area of the video at path: a file libavformat
    reads (an original), or a package's video rendition, its HLS media
    playlist (.m3u8). deadline (time.time()) and stop end it early: the
    result is interrupted and says nothing. Never raises for what the file
    holds: a video that can't be read says why in error. probe only opens
    the file: its frame, its video and what it is, and no area."""
    t0 = time.monotonic()
    m = Measurement(path=path)
    if deadline is None:
        deadline = time.time() + MAX_SECONDS
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        sys.executable,
        "-m",
        "analyzer.pipelines.picture.child",
        path,
        str(int(duration_ms or 0)),
        *(["--probe"] if probe else []),
    ]
    try:
        p = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
    except OSError as e:
        m.error = f"the measurement could not start: {e}"
        return m
    out: list[bytes] = []
    err: list[bytes] = []
    readers = [
        threading.Thread(target=_drain, args=(p.stdout, out, 1 << 22), daemon=True),
        threading.Thread(target=_drain, args=(p.stderr, err, 1 << 16), daemon=True),
    ]
    for r in readers:
        r.start()
    status, usage = _wait(p, deadline, stop)
    for r in readers:
        r.join(timeout=5)
    m.seconds = time.monotonic() - t0
    if usage is not None:
        m.cpu_seconds = usage.ru_utime + usage.ru_stime
        m.max_rss_bytes = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    if status is None:
        m.interrupted, m.reason = True, "stopped before it was done"
    else:
        _read(m, b"".join(out), status, b"".join(err))
    log.info(
        "picture.measured",
        path=path,
        frame=None if m.frame is None else f"{m.frame.width}x{m.frame.height}",
        area=None if m.area is None else f"{m.area.width}x{m.area.height}+{m.area.x}+{m.area.y}",
        full=m.full,
        reason=m.reason,
        error=m.error,
        samples=m.samples,
        informative=m.informative,
        mode=m.sampling.get("mode"),
        max_gap_s=m.sampling.get("maxGapS"),
        bytes_read=m.bytes_read,
        seconds=round(m.seconds, 1),
        cpu_seconds=round(m.cpu_seconds, 1),
        max_rss_mb=round(m.max_rss_bytes / 2**20),
        interrupted=m.interrupted,
    )
    return m


def _drain(stream: Any, into: list[bytes], keep: int) -> None:
    """Read stream to its end, keeping its last keep bytes."""
    size = 0
    try:
        while chunk := stream.read(65536):
            into.append(chunk)
            size += len(chunk)
            while size > keep and len(into) > 1:
                size -= len(into.pop(0))
    except (OSError, ValueError):
        pass


def _wait(
    p: subprocess.Popen, deadline: float, stop: Callable[[], bool] | None
) -> tuple[int | None, Any]:
    """Wait for the child, killing it at the deadline or at stop: its exit
    status (None when it was killed so) and its resource usage."""
    killed = False
    while True:
        try:
            pid, status, usage = os.wait4(p.pid, os.WNOHANG)
        except ChildProcessError:
            return None if killed else 1, None
        if pid:
            p.returncode = os.waitstatus_to_exitcode(status)
            return (None if killed else p.returncode), usage
        if not killed and (time.time() >= deadline or (stop is not None and stop())):
            p.kill()
            killed = True
            continue
        time.sleep(POLL_S)


def _read(m: Measurement, stdout: bytes, status: int, stderr: bytes) -> None:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    try:
        body = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        body = None
    if not isinstance(body, dict):
        tail = stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        m.error = f"the measurement ended without a result (exit {status}): " + " | ".join(tail)
        return
    f, a = body.get("frame"), body.get("area")
    if isinstance(f, dict):
        m.frame = Box(0, 0, int(f["width"]), int(f["height"]))
    if isinstance(a, dict):
        m.area = Box(int(a["x"]), int(a["y"]), int(a["width"]), int(a["height"]))
    m.reason, m.error = body.get("reason"), body.get("error")
    m.samples, m.informative = int(body.get("samples") or 0), int(body.get("informative") or 0)
    m.sides = body.get("sides") or {}
    m.sampling = body.get("sampling") or {}
    m.video = body.get("video") or {}
    m.measured_on = body.get("measuredOn") or {}
    m.decoder = body.get("decoder") or ""
    if m.error:
        m.area = None
