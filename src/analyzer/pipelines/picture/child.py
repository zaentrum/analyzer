"""The measurement's own process: python -m analyzer.pipelines.picture.child
PATH [DURATION_MS] measures the video at path and writes the result, one
line of JSON, to stdout. The parent (measure.py) runs it at a lower
priority, kills it at its deadline, and reads its CPU and its memory when it
reaps it; whatever the decoder does, it can't hold the worker up.

A measurement never says an area while it is in doubt: it says it can't
tell (reason; sampling.doubt names why) when its samples are more than
MAX_GAP_S apart anywhere in the title, when the read budget stopped it,
when a seek decoded another frame than the keyframe it sought, or when a
frame was decoded twice where another was due. The players then fit the
whole frame, as without an area.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import av

from . import planes, rules, sources

NICE = 10
# Samples that may fail, as a share of those tried (and at least one), for
# a measurement still to say anything.
FAIL_SHARE = 0.03
# No two samples further apart than this, the title's start and end too.
MAX_GAP_S = 3.0


def run(path: str, duration_ms: int | None = None) -> dict[str, Any]:
    """Measure the video at path (an original, or a package's rendition);
    the result as measure.py reads it."""
    t0 = time.monotonic()
    out: dict[str, Any] = {"path": path, "decoder": decoder()}
    catalog_s = (duration_ms or 0) / 1000.0
    try:
        src = sources.open_source(path, catalog_s)
    except sources.Undecidable as e:
        out["reason"] = str(e)
        return out
    except sources.Unreadable as e:
        out["error"] = str(e)
        return out
    files = getattr(src, "files", [path])
    before = sources.identity(files)
    v = src.video
    out["frame"] = {"width": v.width, "height": v.height}
    out["video"] = {
        "codec": v.codec,
        "sar": f"{v.sar[0]}:{v.sar[1]}",
        "container": v.container,
        "durationS": round(v.duration_s, 3),
    }
    out["measuredOn"] = _measured_on(src, before)
    acc: rules.Accumulator | None = None

    def sink(t: float, frame: Any, counted: bool) -> None:
        nonlocal acc
        r = planes.reduce_frame(frame, src.blacks)
        if acc is None:
            acc = rules.Accumulator(r.width, r.height)
        acc.add(r, t, counted)

    sm: sources.Sampling | None = None
    try:
        sm = src.run(sink)
    except sources.Undecidable as e:
        out["reason"] = str(e)
    except sources.Unreadable as e:
        out["error"] = str(e)
    finally:
        if isinstance(src, sources.Original):
            src.close()
    sm = sm or src.sampling
    out["sampling"] = _sampling(sm)
    out["sampling"]["blacks"] = dict(src.blacks)
    out["frame"] = {"width": src.video.width, "height": src.video.height}
    out["seconds"] = round(time.monotonic() - t0, 2)
    if "reason" in out or "error" in out:
        return out
    if sources.identity(files) != before:
        out["error"] = "the file changed while it was measured"
        return out
    allowed = max(1, int(FAIL_SHARE * sm.attempted))
    if sm.failed > allowed:
        out["error"] = (
            f"{sm.failed} of {sm.attempted} samples failed, more than {allowed}: "
            + "; ".join(sm.errors[:2])
        )
        return out
    if acc is None:
        out["error"] = f"none of {sm.attempted} samples could be decoded"
        return out
    d = acc.decide()
    out["samples"], out["informative"] = d.samples, d.informative
    out["sides"] = d.sides
    out["reason"] = d.reason
    codes, why = doubts(sm)
    out["sampling"]["doubt"] = codes
    if codes:
        out["reason"] = "can't tell: " + "; ".join(why)
        return out
    if d.area is not None:
        a = d.area
        out["area"] = {"x": a.x, "y": a.y, "width": a.width, "height": a.height}
    return out


def doubts(sm: sources.Sampling) -> tuple[list[str], list[str]]:
    """What leaves a measurement in doubt: codes (gap, budget, missed,
    duplicates) and why."""
    codes: list[str] = []
    why: list[str] = []
    if sm.stopped:
        codes.append("budget")
        why.append(f"the reads stopped ({sm.stopped}) after {sm.bytes_read / 2**20:.0f} MiB")
    gap = sm.largest_gap()
    if gap is None:
        codes.append("gap")
        why.append("no sample was taken")
    elif gap[0] > MAX_GAP_S:
        codes.append("gap")
        why.append(
            f"its samples are up to {gap[0]:.1f} s apart (from {gap[1]:.1f} s), "
            f"more than {MAX_GAP_S:g} s"
        )
    if sm.missed:
        codes.append("missed")
        why.append(f"{sm.missed} seeks decoded another frame than the keyframe they sought")
    if sm.duplicates:
        codes.append("duplicates")
        why.append(f"{sm.duplicates} samples decoded a frame taken already")
    return codes, why


def decoder() -> str:
    """The decoder's versions: PyAV's, and the FFmpeg libraries it has."""
    lav = av.library_versions.get("libavcodec")
    lavc = ".".join(map(str, lav)) if lav else "?"
    ff = getattr(av, "ffmpeg_version_info", "?")
    return f"av {av.__version__} (FFmpeg {ff}, libavcodec {lavc})"


def _measured_on(src: Any, ident: list[Any]) -> dict[str, Any]:
    """What the measurement is of: the original or the package, its size
    and its time (the playlist's for a package, with the rendition's frame)."""
    first = ident[0] if ident else None
    kind = "package" if isinstance(src, sources.Package) else "original"
    out: dict[str, Any] = {"kind": kind}
    if first is not None:
        out["size"] = first[0]
        out["mtime"] = round(first[1] / 1e9, 3)
    if kind == "package":
        out["bytes"] = sum(i[0] for i in ident[1:] if i is not None)
    return out


def _sampling(sm: sources.Sampling) -> dict[str, Any]:
    gap = sm.max_gap()
    return {
        "mode": sm.mode,
        "durationS": round(sm.duration_s, 3),
        "planned": sm.planned,
        "attempted": sm.attempted,
        "decoded": sm.decoded,
        "failed": sm.failed,
        "missed": sm.missed,
        "duplicates": sm.duplicates,
        "stretches": sm.stretches,
        "calibration": sm.calibration,
        "maxGapS": None if gap is None else round(gap, 2),
        "bytesOpen": sm.bytes_open,
        "bytesRead": sm.bytes_read,
        "budgetBytes": sm.budget,
        "stopped": sm.stopped,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: python -m analyzer.pipelines.picture.child PATH [DURATION_MS]", file=sys.stderr
        )
        return 2
    try:
        os.nice(int(os.environ.get("PICTURE_NICE") or NICE))
    except (OSError, ValueError):
        pass
    duration_ms = int(argv[2]) if len(argv) > 2 and argv[2].isdigit() else None
    try:
        out = run(argv[1], duration_ms)
    except Exception as e:  # the parent reads what failed
        out = {"path": argv[1], "error": f"{type(e).__name__}: {e}"[:500]}
    sys.stdout.write(json.dumps(out, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
