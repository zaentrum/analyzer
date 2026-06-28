"""Extract the first text-based subtitle stream embedded in the media
container, parse it as SRT, and match each cue against a small
vocabulary of intro/credits/recap markers.

Three signals come out of this pipeline:
  * **Recap** — "Previously on…" / "Last time on…" / "In our last…"
    appearing in a cue within the first 5 minutes. High confidence:
    the phrase is unambiguous.
  * **Credits** — "Directed by…" / "Music by…" / "Produced by…" /
    "Starring…" appearing in a cue within the last 15 minutes. High
    confidence: shows + films use these as the literal credit roll.
  * **Intro** — the first dialogue cue is more than 30s into the
    file (and within the first 3 minutes). Whatever's before it is
    likely the intro / opening titles. Lower confidence — quiet
    cold opens trigger this too.

Embedded subtitle extraction shells out to ffmpeg with `-map 0:s:<idx>`
for the first text-based subtitle stream. PGS / DVD bitmap subs are
ignored because they'd require OCR (left to a later pass).
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)


@dataclass
class Cue:
    start_ms: int
    end_ms: int
    text: str


TEXT_SUBTITLE_CODECS = {
    "subrip", "srt", "ass", "ssa", "webvtt", "mov_text"
}

RECAP_RE = re.compile(
    r"\b("
    r"previously on"
    r"|last time on"
    r"|last episode"
    r"|in our last"
    r"|in the last episode"
    r"|previous(?:ly)? on this"
    r")\b",
    re.IGNORECASE,
)
CREDIT_RE = re.compile(
    r"\b("
    r"directed by"
    r"|written by"
    r"|produced by"
    r"|executive producer"
    r"|music by"
    r"|cinematography by"
    r"|edited by"
    r"|story by"
    r"|screenplay by"
    r"|starring"
    r")\b",
    re.IGNORECASE,
)

RECAP_WINDOW_MS = 5 * 60 * 1000
RECAP_DEFAULT_DURATION_MS = 90 * 1000
CREDITS_WINDOW_FROM_END_MS = 15 * 60 * 1000
INTRO_MIN_GAP_MS = 30 * 1000
INTRO_MAX_GAP_MS = 3 * 60 * 1000


def detect(path: str, duration_ms: int | None) -> list[dict]:
    cues = _extract_cues(path)
    if not cues:
        return []

    segments: list[dict] = []

    # ------------------------------------------------------------- recap
    for cue in cues:
        if cue.start_ms > RECAP_WINDOW_MS:
            break
        if RECAP_RE.search(cue.text):
            end = cue.start_ms + RECAP_DEFAULT_DURATION_MS
            if duration_ms is not None:
                end = min(end, duration_ms)
            segments.append({
                "kind": "recap",
                "startMs": cue.start_ms,
                "endMs": end,
                "source": "subtitle",
                "confidence": 0.90,
                "label": _label(cue.text),
            })
            break

    # ----------------------------------------------------------- credits
    if duration_ms is not None and duration_ms > CREDITS_WINDOW_FROM_END_MS:
        cutoff = duration_ms - CREDITS_WINDOW_FROM_END_MS
        for cue in cues:
            if cue.start_ms < cutoff:
                continue
            if CREDIT_RE.search(cue.text):
                segments.append({
                    "kind": "credits",
                    "startMs": cue.start_ms,
                    "endMs": duration_ms,
                    "source": "subtitle",
                    "confidence": 0.90,
                    "label": _label(cue.text),
                })
                break

    # ------------------------------------------------------------- intro
    # First dialogue cue more than 30s in, but within first 3 min.
    # Below 30s is too short to be a useful intro skip; above 3 min is
    # more likely a cold-open / silent scene than a music intro.
    first = cues[0]
    if INTRO_MIN_GAP_MS < first.start_ms <= INTRO_MAX_GAP_MS:
        segments.append({
            "kind": "intro",
            "startMs": 0,
            "endMs": first.start_ms,
            "source": "subtitle",
            "confidence": 0.55,
            "label": f"no dialogue until {first.start_ms // 1000}s",
        })

    log.info("subtitles.detected", path=path, segments=len(segments), cues=len(cues))
    return segments


# ---------------------------------------------------------------- helpers
def _extract_cues(path: str) -> list[Cue]:
    stream_idx = _first_text_subtitle_stream(path)
    if stream_idx is None:
        return []
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-map", f"0:{stream_idx}",
             "-f", "srt", "-"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        log.warning("subtitles.extract_timeout", path=path)
        return []
    if result.returncode != 0:
        log.warning(
            "subtitles.extract_failed",
            path=path,
            stderr=result.stderr[:300],
        )
        return []
    return _parse_srt(result.stdout)


def _first_text_subtitle_stream(path: str) -> int | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-select_streams", "s", path],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return None
    # Prefer English / undefined-language tracks if any exist.
    candidates = [s for s in streams if s.get("codec_name") in TEXT_SUBTITLE_CODECS]
    if not candidates:
        return None
    eng = [s for s in candidates if (s.get("tags") or {}).get("language", "").lower()
           in ("eng", "und", "")]
    chosen = (eng or candidates)[0]
    return chosen.get("index")


def _parse_srt(text: str) -> list[Cue]:
    cues: list[Cue] = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        timing = lines[1]
        if " --> " not in timing:
            continue
        start_str, end_str = timing.split(" --> ", 1)
        start = _parse_ts(start_str.strip())
        end = _parse_ts(end_str.strip())
        if start is None or end is None or end <= start:
            continue
        body = " ".join(lines[2:]).strip()
        # ASS-style { } tags pollute matching; strip them.
        body = re.sub(r"\{[^}]*\}", "", body)
        if body:
            cues.append(Cue(start_ms=start, end_ms=end, text=body))
    return cues


def _parse_ts(ts: str) -> int | None:
    # SRT: "HH:MM:SS,mmm" or "HH:MM:SS.mmm".
    try:
        if "," in ts:
            hms, ms = ts.split(",", 1)
        elif "." in ts:
            hms, ms = ts.split(".", 1)
        else:
            return None
        h, m, s = hms.split(":")
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms[:3].ljust(3, "0"))
    except (ValueError, AttributeError):
        return None


def _label(text: str) -> str:
    snippet = text.strip().replace("\n", " ")[:100]
    return f"text: {snippet}"
