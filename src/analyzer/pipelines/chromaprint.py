"""Cross-episode chromaprint matching — the only reliable way to detect
the recurring theme-song intro on shows whose intro doesn't end on a hard
black fade (e.g. How I Met Your Mother).

Algorithm
---------
For the target episode plus N sibling episodes of the same season we run
`fpcalc -raw` on a head window (first INTRO_SCAN_SECONDS) and on a tail
window (last CREDITS_SCAN_SECONDS). fpcalc emits one 32-bit fingerprint
roughly every 0.124 seconds. We compute the popcount of XOR between
sliding windows in the target's fingerprint and each sibling's
fingerprint; runs of low Hamming-distance pairs identify a recurring
audio sequence (= intro or credits jingle).

This is the same primitive that audio-fingerprint intro-skippers use.
We don't ship TransNet / Whisper here yet — pure audio fingerprint
matching has been sufficient on test material and is dramatically cheaper
to compute (no GPU, finishes in seconds per episode)."""

from __future__ import annotations

import json
import subprocess

import structlog

log = structlog.get_logger(__name__)

# How far in we'll scan from each end. The intro is almost always inside
# the first 10 minutes; credits in the last 5. Wider windows cost time
# without finding more matches.
INTRO_SCAN_SECONDS = 600
CREDITS_SCAN_SECONDS = 300

# Tile size for the sliding-window cross-correlation, in fingerprint
# frames. fpcalc frames at ~0.124s, so 30 frames ≈ 3.7s. Below this the
# false-positive rate climbs sharply — random scenes will match by
# chance.
WINDOW_FRAMES = 30
# Each fingerprint is a 32-bit int, so 32 bits per frame. Up to
# MAX_BIT_DIFF bits may differ within the window for it to count as a
# match. Typical audio-fingerprint intro-skippers use ~5/32; we're a
# little stricter to favour precision over recall.
MAX_BIT_DIFF_PER_FRAME = 6

# Minimum run length (in frames) of consecutive matching windows for the
# detector to emit a segment. 60 frames ≈ 7.5s — shorter than that is
# usually a scene transition or a stinger, not an intro.
MIN_RUN_FRAMES = 60

# Gap-tolerance (in fingerprint frames) for fusing adjacent matching
# runs into a single segment. fpcalc misses a frame or two whenever the
# intro audio has a sub-second silence (logo cards) or a hard cut. Fuse
# runs separated by up to MAX_GAP_FRAMES so the resulting segment
# covers the full intro / credits jingle, not just the cleanest
# sub-stretch.
MAX_GAP_FRAMES = 24  # ~3 seconds

# How many siblings to require a match against before trusting the
# range. With 5 siblings we want at least 2 confirmations so a single
# coincidentally-similar episode doesn't pollute the result.
MIN_SIBLING_AGREEMENT = 2

# Maximum siblings to compare against — beyond ~6 the marginal value
# drops while the runtime cost climbs linearly.
MAX_SIBLINGS = 6

# How far past the start of the head window we believe a genuine intro
# can still start. HIMYM has cold opens of 30-120s and the intro
# follows; over INTRO_LATEST_START a "match" is almost certainly a
# recurring scene tag, not the title sequence.
INTRO_LATEST_START_S = 180

# fpcalc emits ~8 fingerprints per second by default. Module-level
# constant rather than re-deriving it inside _frames_to_ms.
FPCALC_FPS = 8.0


def detect(path: str, duration_ms: int | None, sibling_paths: list[str]) -> list[dict]:
    """Return intro / credits segments inferred from cross-episode
    fingerprint agreement. Returns an empty list when not enough
    siblings exist or fpcalc isn't installed (handled in worker)."""
    if duration_ms is None or duration_ms < 60_000:
        return []
    if not sibling_paths:
        return []
    siblings = sibling_paths[:MAX_SIBLINGS]
    duration_s = duration_ms / 1000.0

    out: list[dict] = []

    # ---------- intro ----------
    # The HIMYM-style title theme lives in the first ~3 minutes after
    # an optional cold open. Score every matching run we find against
    # each sibling, then pick the run that starts EARLIEST (after a
    # small-tie tolerance) — taking the longest run picks up the
    # recurring laugh-track tail at ~2 minutes in instead.
    head_fps = _fingerprint(path, 0, min(INTRO_SCAN_SECONDS, duration_s))
    if head_fps:
        all_runs: list[list[tuple[int, int]]] = []
        for sib in siblings:
            sib_fps = _fingerprint(sib, 0, INTRO_SCAN_SECONDS)
            if not sib_fps:
                continue
            runs = _all_runs(head_fps, sib_fps)
            if runs:
                all_runs.append(runs)
        latest = int(INTRO_LATEST_START_S * FPCALC_FPS)
        intro_range = _earliest_consensus(all_runs, latest_start_frames=latest)
        if intro_range is not None:
            start_ms, end_ms = _frames_to_ms(intro_range, base_seconds=0)
            out.append(
                {
                    "kind": "intro",
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "source": "chromaprint",
                    "confidence": min(0.95, 0.6 + 0.1 * len(all_runs)),
                    "label": f"chromaprint x{len(all_runs)}",
                }
            )

    # ---------- credits ----------
    # Pick the run that starts EARLIEST in the tail window — the
    # credit-roll theme is typically the first recurring music after
    # the closing scene, and the segment should run from there to EOF
    # (unless an end-of-credits stinger is detected, which we leave to
    # a follow-up pass; emitting to EOF is the user-correct default).
    tail_start = max(0.0, duration_s - CREDITS_SCAN_SECONDS)
    tail_fps = _fingerprint(path, tail_start, duration_s)
    if tail_fps:
        all_runs = []
        for sib in siblings:
            sib_dur = _file_duration(sib)
            sib_tail = _fingerprint(sib, max(0.0, sib_dur - CREDITS_SCAN_SECONDS), None)
            if not sib_tail:
                continue
            runs = _all_runs(tail_fps, sib_tail)
            if runs:
                all_runs.append(runs)
        credits_range = _earliest_consensus(all_runs)
        if credits_range is not None:
            start_ms, _ = _frames_to_ms(credits_range, base_seconds=tail_start)
            # Extend to end-of-file. A film-style post-credit scene
            # would cut this short, but for a TV show the credit roll
            # universally runs to the closing slate.
            end_ms = int(duration_s * 1000)
            out.append(
                {
                    "kind": "credits",
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "source": "chromaprint",
                    "confidence": min(0.95, 0.6 + 0.1 * len(all_runs)),
                    "label": f"chromaprint x{len(all_runs)}",
                }
            )

    log.info("chromaprint.detected", path=path, segments=len(out), siblings=len(siblings))
    return out



# ---------------------------------------------------------------------------


def _file_duration(path: str) -> float:
    """ffprobe a single file for duration in seconds. Used only on
    sibling paths where we don't already have the value from katalog."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def _fingerprint(path: str, start_s: float, end_s: float | None) -> list[int]:
    """Compute a chromaprint fingerprint for [start_s, end_s] of `path`.

    fpcalc on Debian/Ubuntu is built against a minimal libavformat that
    can't read MKV containers (which is most of the catalogue). Pipe
    ffmpeg's WAV output into fpcalc -raw -json -length N - instead so
    fpcalc only sees PCM. Empty list on any failure (no binary, decode
    error, JSON parse failure) — the caller treats that as 'no signal'."""
    length = (end_s - start_s) if (end_s is not None and end_s > start_s) else 600.0
    ff_cmd = [
        "ffmpeg",
        "-hide_banner", "-nostats", "-loglevel", "error",
        "-ss", f"{start_s:.2f}",
        "-t",  f"{length:.2f}",
        "-i", path,
        "-vn",
        "-ac", "2",          # downmix matches fpcalc's expected layout
        "-ar", "22050",
        "-f", "wav",
        "pipe:1",
    ]
    fp_cmd = ["fpcalc", "-raw", "-json", "-length", f"{length:.2f}", "-"]
    try:
        ff = subprocess.Popen(
            ff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        fp = subprocess.Popen(
            fp_cmd, stdin=ff.stdout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        # Close our handle to ffmpeg's stdout so it gets SIGPIPE when fpcalc exits.
        if ff.stdout is not None:
            ff.stdout.close()
        try:
            out, err = fp.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            fp.kill()
            ff.kill()
            log.warning("chromaprint.fpcalc_timeout", path=path)
            return []
        ff.wait(timeout=30)
    except FileNotFoundError as e:
        log.warning("chromaprint.binary_missing", missing=str(e))
        return []
    if fp.returncode != 0:
        msg = (err.decode("utf-8", "replace") if err else "")[:200]
        log.warning("chromaprint.fpcalc_error", path=path, stderr=msg)
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    raw = data.get("fingerprint")
    if isinstance(raw, list):
        return [int(v) & 0xFFFFFFFF for v in raw]
    return []


def _all_runs(target: list[int], sibling: list[int]) -> list[tuple[int, int]]:
    """Find every run of matching WINDOW_FRAMES-tiles between target and
    sibling, with adjacent runs fused if they're separated by at most
    MAX_GAP_FRAMES (e.g. a single off-window for a logo card or a hard
    cut). Returns [(start, end), …] in TARGET frame units. Empty list
    if nothing crosses MIN_RUN_FRAMES after fusion."""
    raw = _raw_runs(target, sibling)
    if not raw:
        return []
    # Sort by target start and fuse adjacent runs whose gap is small.
    raw.sort(key=lambda r: r[0])
    fused: list[tuple[int, int]] = [raw[0]]
    for s, e in raw[1:]:
        ps, pe = fused[-1]
        if s - pe <= MAX_GAP_FRAMES:
            fused[-1] = (ps, max(pe, e))
        else:
            fused.append((s, e))
    return [r for r in fused if (r[1] - r[0]) >= MIN_RUN_FRAMES]


def _raw_runs(target: list[int], sibling: list[int]) -> list[tuple[int, int]]:
    """Locate every WINDOW_FRAMES tile in target that has a sibling
    match under the bit-diff threshold, then extend each anchor forward
    while the rolling match holds. Returns the raw (unfused) run list
    in target frame units."""
    if len(target) < WINDOW_FRAMES or len(sibling) < WINDOW_FRAMES:
        return []
    sib_len = len(sibling)
    tgt_len = len(target)
    threshold_bits = WINDOW_FRAMES * MAX_BIT_DIFF_PER_FRAME

    runs: list[tuple[int, int]] = []
    i = 0
    while i <= tgt_len - WINDOW_FRAMES:
        matched_j: int | None = None
        for j in range(0, sib_len - WINDOW_FRAMES + 1, WINDOW_FRAMES // 6 or 1):
            score = 0
            ok = True
            for k in range(WINDOW_FRAMES):
                score += _popcount32(target[i + k] ^ sibling[j + k])
                if score > threshold_bits * 1.5:
                    ok = False
                    break
            if ok and score <= threshold_bits:
                matched_j = j
                break

        if matched_j is not None:
            # Extend the run forward while consecutive WINDOW_FRAMES
            # tiles keep matching. Stops as soon as one tile goes over
            # threshold; the caller fuses neighbouring runs separated
            # by short gaps via MAX_GAP_FRAMES.
            run_start = i
            ii = i + WINDOW_FRAMES
            jj = matched_j + WINDOW_FRAMES
            while ii < tgt_len and jj < sib_len:
                tile = min(WINDOW_FRAMES, tgt_len - ii, sib_len - jj)
                s = 0
                for k in range(tile):
                    s += _popcount32(target[ii + k] ^ sibling[jj + k])
                if tile > 0 and s > (threshold_bits * tile / WINDOW_FRAMES):
                    break
                ii += tile
                jj += tile
            runs.append((run_start, ii))
            i = ii
        else:
            i += WINDOW_FRAMES // 4 or 1
    return runs


def _earliest_consensus(
    per_sibling_runs: list[list[tuple[int, int]]],
    latest_start_frames: int | None = None,
) -> tuple[int, int] | None:
    """Pick the earliest-starting run that ≥ MIN_SIBLING_AGREEMENT
    siblings independently produced. Each sibling contributes its own
    list of matching runs; we cluster runs by start frame (within a
    WINDOW_FRAMES tolerance) and report the cluster whose representative
    start is smallest — that's the run that overlaps in time across
    siblings, not the unrelated late laugh-track match. For an intro
    we additionally cap by latest_start_frames; runs starting after
    that are almost certainly recurring scene tags, not the title."""
    if len(per_sibling_runs) < MIN_SIBLING_AGREEMENT:
        return None
    # Flatten with the sibling index so duplicates within one sibling
    # don't inflate the agreement count.
    flat = [(start, end, sib_idx)
            for sib_idx, runs in enumerate(per_sibling_runs)
            for (start, end) in runs]
    if latest_start_frames is not None:
        flat = [r for r in flat if r[0] <= latest_start_frames]
    if not flat:
        return None
    flat.sort(key=lambda r: r[0])
    # Greedy cluster on start frame.
    best_cluster: list[tuple[int, int, int]] = []
    cur: list[tuple[int, int, int]] = []
    for r in flat:
        if not cur or r[0] - cur[-1][0] <= WINDOW_FRAMES * 2:
            cur.append(r)
        else:
            if _distinct_siblings(cur) >= MIN_SIBLING_AGREEMENT and (
                not best_cluster or cur[0][0] < best_cluster[0][0]
            ):
                best_cluster = cur
                break  # earliest qualifying cluster wins
            cur = [r]
    if not best_cluster and _distinct_siblings(cur) >= MIN_SIBLING_AGREEMENT:
        best_cluster = cur
    if not best_cluster:
        return None
    starts = sorted(r[0] for r in best_cluster)
    ends = sorted(r[1] for r in best_cluster)
    mid = len(best_cluster) // 2
    return starts[mid], ends[mid]


def _distinct_siblings(cluster: list[tuple[int, int, int]]) -> int:
    return len({r[2] for r in cluster})


def _frames_to_ms(rng: tuple[int, int], base_seconds: float) -> tuple[int, int]:
    """Convert a (start_frame, end_frame) range in fpcalc frame units to
    a wall-clock ms range, anchored at base_seconds."""
    start_ms = int((base_seconds + rng[0] / FPCALC_FPS) * 1000)
    end_ms = int((base_seconds + rng[1] / FPCALC_FPS) * 1000)
    return start_ms, end_ms


_POPCOUNT_TABLE = bytes(bin(i).count("1") for i in range(256))


def _popcount32(x: int) -> int:
    """Hamming weight of a 32-bit int. Pure-python for portability —
    swap for numpy at scale if profiling shows it's the hot spot."""
    x = x & 0xFFFFFFFF
    return (
        _POPCOUNT_TABLE[x & 0xFF]
        + _POPCOUNT_TABLE[(x >> 8) & 0xFF]
        + _POPCOUNT_TABLE[(x >> 16) & 0xFF]
        + _POPCOUNT_TABLE[(x >> 24) & 0xFF]
    )


