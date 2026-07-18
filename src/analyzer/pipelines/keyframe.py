"""Keyframe artwork extractor.

For an item that TMDB and fanart.tv had no image for, pick a representative frame
from the video itself and use it as poster/backdrop. Selection:

  * stay in the CONTENT window — after the detected `intro` and before the
    `credits` / trailing black boundary (from the fused segments); fall back to
    the middle 15%-85% when those aren't known;
  * reject BLACK / FLAT frames — each candidate is downscaled to 16x16 grayscale
    and scored by pixel variability (population stddev), preferring the most
    visually detailed non-dark frame;

No numpy/PIL: the stats are computed in pure Python over the 256 sampled pixels.
All ffmpeg calls are best-effort and time-bounded; any failure yields None.
"""

from __future__ import annotations

import math
import subprocess

import structlog

log = structlog.get_logger(__name__)

_SAMPLE = 16  # downscale to _SAMPLE x _SAMPLE grayscale for stats
_FINAL_WIDTH = 1280  # output jpeg width (height keeps aspect)
_BLACK_MEAN_FLOOR = 16.0  # 0..255 luma; below => ~black frame
_FLAT_STD_FLOOR = 12.0  # below => flat/solid frame (little detail)
# Candidate positions across the content window (fractions).
_CANDIDATES = (0.15, 0.28, 0.40, 0.50, 0.60, 0.72, 0.85)


def extract(path: str, duration_ms: int | None, segments: list[dict]) -> bytes | None:
    """Return JPEG bytes of the best keyframe, or None if none is usable."""
    if not duration_ms or duration_ms <= 0:
        # Unmatched items often have no stored duration (TMDB never set a
        # runtime); probe the file directly so extraction doesn't depend on it.
        duration_ms = _probe_duration_ms(path)
    window = _content_window(duration_ms, segments)
    if window is None:
        return None
    lo, hi = window

    scored: list[tuple[float, float, float]] = []  # (stddev, mean, ts_seconds)
    for frac in _CANDIDATES:
        ts = lo + (hi - lo) * frac
        stats = _frame_stats(path, ts)
        if stats is None:
            continue
        mean, std = stats
        scored.append((std, mean, ts))

    if not scored:
        return None
    # Prefer non-dark, non-flat frames; among those the most variable. If every
    # candidate trips a floor (e.g. a very dark film), fall back to the single
    # most-variable frame rather than give up.
    good = [s for s in scored if s[1] >= _BLACK_MEAN_FLOOR and s[0] >= _FLAT_STD_FLOOR]
    std, mean, ts = max(good or scored, key=lambda s: s[0])
    log.info("keyframe.picked", ts=round(ts, 1), stddev=round(std, 1), mean=round(mean, 1))
    return _extract_jpeg(path, ts)


def _content_window(duration_ms: int | None, segments: list[dict]) -> tuple[float, float] | None:
    """The [start, end] seconds to sample within — after any intro, before the
    credits/black boundary; the safe middle when those are unknown."""
    dur = (duration_ms or 0) / 1000.0
    if dur <= 0:
        return None
    intro_end = 0.0
    credits_start = dur
    for s in segments or []:
        kind = s.get("kind")
        try:
            start = float(s.get("startMs", 0)) / 1000.0
            end = float(s.get("endMs", 0)) / 1000.0
        except (TypeError, ValueError):
            continue
        if kind == "intro":
            intro_end = max(intro_end, end)
        elif kind in ("credits", "blackframe", "preview") and 0 < start < credits_start:
            credits_start = start
    lo = max(intro_end, dur * 0.05)
    hi = min(credits_start, dur * 0.95)
    if hi - lo < 1.0:  # window collapsed -> safe middle
        lo, hi = dur * 0.15, dur * 0.85
    if hi <= lo:
        lo, hi = 0.0, dur
    return lo, hi


def _probe_duration_ms(path: str) -> int | None:
    """ffprobe the container duration in ms; None on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        secs = float(r.stdout.strip())
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None
    return int(secs * 1000) if secs > 0 else None


def _frame_stats(path: str, ts: float) -> tuple[float, float] | None:
    """(mean, stddev) of a _SAMPLE x _SAMPLE grayscale frame at ts; None on error."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-ss", f"{ts:.2f}", "-i", path,
        "-frames:v", "1",
        "-vf", f"scale={_SAMPLE}:{_SAMPLE},format=gray",
        "-f", "rawvideo", "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return None
    px = r.stdout
    n = _SAMPLE * _SAMPLE
    if len(px) < n:
        return None
    px = px[:n]
    mean = sum(px) / n
    var = sum((p - mean) ** 2 for p in px) / n
    return mean, math.sqrt(var)


def _extract_jpeg(path: str, ts: float) -> bytes | None:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-ss", f"{ts:.2f}", "-i", path,
        "-frames:v", "1",
        "-vf", f"scale={_FINAL_WIDTH}:-2",
        "-q:v", "3", "-f", "mjpeg", "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return r.stdout or None
