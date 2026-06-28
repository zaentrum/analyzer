"""Black-frame detection via ffmpeg's `blackdetect` filter. We only scan the
last 15% of the runtime because that's where the credit-roll boundary
lives — running blackdetect on the full file is wasteful (a 2h movie has
many random dark scenes) and slow.

The first long black region after that 85% mark is treated as the credit
start. Confidence is 0.7 unless we also see a tail-of-file silence
window overlapping it (the fuser combines signals)."""

from __future__ import annotations

import re
import subprocess

import structlog

log = structlog.get_logger(__name__)

# ffmpeg writes lines like:
#   [blackdetect @ 0x...] black_start:5398.71 black_end:5402.34 black_duration:3.63
BLACK_RE = re.compile(
    r"black_start:(?P<start>[0-9.]+)\s+black_end:(?P<end>[0-9.]+)\s+black_duration:(?P<dur>[0-9.]+)"
)

# A genuine credit-roll boundary tends to be at least 2 seconds of black.
MIN_BLACK_SECONDS = 2.0
# Skip the scan if we don't have a duration to seek into.
TAIL_FRACTION = 0.15


def detect(path: str, duration_ms: int | None) -> list[dict]:
    if duration_ms is None or duration_ms < 60_000:
        # Too short to bother — most shorts have no credits.
        return []

    start_seconds = max(0.0, (duration_ms / 1000.0) * (1 - TAIL_FRACTION))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-ss", f"{start_seconds:.2f}",
        "-i", path,
        "-vf", f"blackdetect=d={MIN_BLACK_SECONDS}:pic_th=0.98",
        "-an",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log.warning("blackframe.timeout", path=path)
        return []

    # blackdetect writes to stderr.
    segments: list[dict] = []
    for m in BLACK_RE.finditer(result.stderr):
        rel_start = float(m.group("start"))
        rel_end = float(m.group("end"))
        abs_start_ms = int((start_seconds + rel_start) * 1000)
        abs_end_ms = int((start_seconds + rel_end) * 1000)
        if abs_end_ms <= abs_start_ms:
            continue
        segments.append(
            {
                "kind": "credits",
                "startMs": abs_start_ms,
                "endMs": abs_end_ms,
                "source": "blackframe",
                "confidence": 0.70,
                "label": f"black {float(m.group('dur')):.1f}s",
            }
        )
    log.info("blackframe.detected", path=path, count=len(segments))
    return segments
