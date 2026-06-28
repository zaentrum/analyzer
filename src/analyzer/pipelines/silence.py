"""Silence detection via ffmpeg's `silencedetect` filter. We scan two
windows: the first 10% (looking for intro-end / first dialogue) and the
last 15% (corroborating signal for the blackframe credit boundary).
Running on the whole file is wasteful and not what intro-skipper does
either."""

from __future__ import annotations

import re
import subprocess

import structlog

log = structlog.get_logger(__name__)

# ffmpeg writes pairs of lines:
#   [silencedetect @ 0x...] silence_start: 5398.7
#   [silencedetect @ 0x...] silence_end: 5402.3 | silence_duration: 3.6
SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<t>[0-9.]+)")
SILENCE_END_RE = re.compile(
    r"silence_end:\s*(?P<t>[0-9.]+)\s*\|\s*silence_duration:\s*(?P<d>[0-9.]+)"
)

# A silence window worth flagging. Shorter than this is usually just a pause
# between lines of dialogue.
MIN_SILENCE_SECONDS = 1.5
# -30 dBFS is intro-skipper's default; -50 misses music tails, -20 is too eager.
NOISE_THRESHOLD_DB = -30


def detect(path: str, duration_ms: int | None) -> list[dict]:
    if duration_ms is None or duration_ms < 60_000:
        return []

    duration_s = duration_ms / 1000.0
    windows = [
        ("intro", 0.0, min(duration_s * 0.10, 600.0)),
        ("credits", duration_s * 0.85, duration_s),
    ]
    out: list[dict] = []
    for kind, start, end in windows:
        scan = _scan_window(path, start, end)
        for s_start, s_end in scan:
            if s_end <= s_start:
                continue
            out.append(
                {
                    "kind": kind,
                    "startMs": int(s_start * 1000),
                    "endMs": int(s_end * 1000),
                    "source": "silence",
                    "confidence": 0.40,
                    "label": f"silence {s_end - s_start:.1f}s",
                }
            )
    log.info("silence.detected", path=path, count=len(out))
    return out


def _scan_window(path: str, start_s: float, end_s: float) -> list[tuple[float, float]]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-ss", f"{start_s:.2f}",
        "-to", f"{end_s:.2f}",
        "-i", path,
        "-af", f"silencedetect=noise={NOISE_THRESHOLD_DB}dB:d={MIN_SILENCE_SECONDS}",
        "-vn",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        log.warning("silence.timeout", path=path)
        return []

    pending_start: float | None = None
    out: list[tuple[float, float]] = []
    for line in result.stderr.splitlines():
        m_start = SILENCE_START_RE.search(line)
        if m_start:
            pending_start = start_s + float(m_start.group("t"))
            continue
        m_end = SILENCE_END_RE.search(line)
        if m_end and pending_start is not None:
            abs_end = start_s + float(m_end.group("t"))
            out.append((pending_start, abs_end))
            pending_start = None
    return out
