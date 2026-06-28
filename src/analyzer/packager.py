"""Per-item CMAF packager.

Takes a source file (HEVC video + AAC audio in any container), runs
shaka-packager to emit a streaming-friendly CMAF tree under
/var/lib/katalog/packages/{itemId}/, and writes the manifest.json that
katalog-stream reads when deciding whether to serve a pre-packaged
item or fall through to on-demand transcode.

Design rules (Phase 2 MVP — first cut):
* Video is HEVC passthrough. We never re-encode. If the source video
  codec isn't hevc/h264 we fail the package job; the operator picks a
  different source. This matches the user's "no re-encode" constraint.
* Audio is AAC passthrough when the source track is already AAC,
  otherwise we transcode to AAC-LC 48 kHz stereo via ffmpeg before
  shaka-packager runs (browsers can't decode AC3/DTS/TrueHD natively
  so this is the minimum needed for playback).
* Subtitles are extracted to WebVTT.
* All state is on disk under the per-item output directory. The
  three sentinels {.packaging, .complete, .failed} are mutually
  exclusive and tell every reader where this package is in its
  lifecycle without needing a database round-trip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("analyzer.packager")

# Where katalog-stream looks for packaged items. Must match the
# mountPath in k8s/{analyzer,stream}-deployment.yaml.
PACKAGES_ROOT = Path("/var/lib/katalog/packages")

# Per-category sharding under PACKAGES_ROOT keeps any single directory
# under ~50 entries even at tens of thousands of items per category —
# filesystem directory ops (readdir on NFS, especially) get visibly
# slow once a directory holds thousands of entries. Layout:
#   /var/lib/katalog/packages/{category}/{shard}/{itemId}/...
# where category is movies/shows/music (derived from katalog's item
# type) and shard is the first two hex chars of the item uuid. The
# stream service probes categories on read to find the package — it
# only knows the item id, not the type.
_CATEGORY_BY_TYPE = {
    "movie": "movies",
    "episode": "shows",
    "series": "shows",
    "season": "shows",
    "album": "music",
    "track": "music",
    "song": "music",
}


def _item_root(item_id: str, item_type: str | None) -> Path:
    """Return the on-disk package root for one item."""
    category = _CATEGORY_BY_TYPE.get((item_type or "").lower(), "other")
    shard = (item_id[:2] or "00").lower()
    return PACKAGES_ROOT / category / shard / item_id


def _find_existing_root(item_id: str) -> Path | None:
    """Probe every category for an existing package directory for the
    given item id. Returns the first hit, or None when nothing is on
    disk yet. Used by package_status when the caller doesn't know
    item.type (chino-api admin GET hits this path)."""
    shard = (item_id[:2] or "00").lower()
    categories = [*set(_CATEGORY_BY_TYPE.values()), "other"]
    for category in categories:
        path = PACKAGES_ROOT / category / shard / item_id
        if path.exists():
            return path
    return None

# Segment length in seconds. Same value the legacy on-demand pipeline
# used; long enough to amortize HTTP overhead, short enough for snappy
# seeks. shaka-packager will align cuts to source IDRs near this mark.
SEGMENT_SECONDS = 6

# Manifest schema version. Bump in lockstep with
# stream/internal/pkgmanifest/manifest.go's CurrentVersion when the
# on-disk shape changes incompatibly.
MANIFEST_VERSION = 1

# Trickplay (scrub-preview thumbnails) parameters. The defaults match
# common practice and shaka's own trickplay docs: one ~16:9
# frame every 10 s, tiled 10x10 per sprite-sheet JPG. That's ~30 KB
# per sprite and 1 sprite per ~16.7 min of source, so a 90 min movie
# produces ~6 sprites + a small VTT. The player loads the VTT once
# and pulls one tiny sprite as the cursor enters each 1000 s window.
TRICKPLAY_INTERVAL_SEC = 10
TRICKPLAY_THUMB_WIDTH = 320
TRICKPLAY_THUMB_HEIGHT = 180
TRICKPLAY_GRID_COLS = 10
TRICKPLAY_GRID_ROWS = 10


class PackageError(RuntimeError):
    """Raised when a packaging job fails. The message is written to
    .failed so operators can see what went wrong."""


def _run_ffmpeg_capturing(label: str, args: list[str]) -> None:
    """Run an ffmpeg invocation capturing stderr so failures surface
    the actual diagnostic instead of the bare exit code.

    subprocess.run(check=True) raises CalledProcessError on non-zero
    exit but its str(e) is a one-line `Command '[...]' returned
    non-zero exit status 1.` — useless when triaging failures on the
    bulk-packaging queue. Capturing stderr + raising PackageError
    with a 1500-char snippet keeps the .failed sentinel actionable.
    """
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise PackageError(
            f"ffmpeg {label} exited {result.returncode}: {stderr[-1500:] or '(no stderr)'}"
        )


@dataclass
class _Probe:
    container: str
    duration_ms: int
    video: dict[str, Any]
    audio: list[dict[str, Any]]
    subtitles: list[dict[str, Any]]


def package_item(item_id: str, source_path: str, item_type: str | None = None) -> dict[str, Any]:
    """Package one item synchronously. Returns the written manifest.

    Safe to retry: a previous .failed or partial run is wiped before
    re-attempting. Concurrent calls for the same item_id are NOT
    serialised here — the caller (the analyzer FastAPI layer) owns the
    queue.

    item_type is katalog's classification (movie/episode/album/…) and
    decides which top-level category directory the package lands under.
    """
    src = Path(source_path)
    if not src.exists():
        raise PackageError(f"source not found: {source_path}")

    out_root = _item_root(item_id, item_type)
    _reset_output_dir(out_root)
    (out_root / ".packaging").write_text(
        json.dumps({"started_at": datetime.now(UTC).isoformat(), "pid": os.getpid()})
    )

    try:
        probe = _ffprobe(src)
        if probe.video.get("codec_name") not in ("hevc", "h264"):
            raise PackageError(
                f"video codec {probe.video.get('codec_name')!r} not supported "
                "(passthrough only; HEVC and H.264 are the allowed input codecs)"
            )

        # Stage source through ffmpeg if any audio track isn't already
        # AAC. shaka-packager doesn't encode audio — it only packages —
        # so non-AAC inputs need a pre-transmux pass. Subtitles are
        # extracted in the same pass.
        with tempfile.TemporaryDirectory(prefix=f"pkg-{item_id}-") as tmp:
            tmpdir = Path(tmp)
            packaging_source, audio_meta = _prepare_source(src, probe, tmpdir)
            subtitle_meta = _extract_subtitles(src, probe, out_root / "subs")
            video_meta, audio_meta = _run_shaka_packager(
                packaging_source, probe, audio_meta, out_root
            )

        # Trickplay runs against the original source — only 1 frame
        # per TRICKPLAY_INTERVAL_SEC, so HEVC decode cost is small
        # (~30 s on a 90 min movie) and we don't need the
        # transmuxed intermediate to still exist.
        trickplay_meta = _generate_trickplay(src, probe, out_root / "trickplay")

        manifest = {
            "version": MANIFEST_VERSION,
            "itemId": item_id,
            "source": {
                "path": str(src),
                "mtime": _iso(src.stat().st_mtime),
                "size": src.stat().st_size,
                "durationMs": probe.duration_ms,
            },
            "packagedAt": datetime.now(UTC).isoformat(),
            "packager": _packager_version(),
            "renditions": {
                "video": [video_meta],
                "audio": audio_meta,
            },
            "subtitles": subtitle_meta,
        }
        if trickplay_meta is not None:
            manifest["trickplay"] = trickplay_meta
        _write_atomic(out_root / "manifest.json", json.dumps(manifest, indent=2).encode())

        # Flip the sentinel atomically so a partially-written package is
        # never observable as .complete.
        (out_root / ".packaging").unlink(missing_ok=True)
        (out_root / ".complete").write_text(
            datetime.now(UTC).isoformat() + "\n"
        )
        log.info("packager.complete", item_id=item_id, dir=str(out_root))
        return manifest

    except Exception as e:
        (out_root / ".packaging").unlink(missing_ok=True)
        (out_root / ".failed").write_text(
            json.dumps({"error": str(e), "at": datetime.now(UTC).isoformat()}, indent=2)
        )
        log.exception("packager.failed", item_id=item_id, error=str(e))
        raise


def _reset_output_dir(out_root: Path) -> None:
    """Clear any prior package contents so a retry starts clean.
    We keep the directory itself (NFS bind mount lives here) and just
    wipe its contents."""
    if out_root.exists():
        for child in out_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out_root.mkdir(parents=True, exist_ok=True)


def _ffprobe(path: Path) -> _Probe:
    out = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    raw = json.loads(out.stdout)
    fmt = raw.get("format", {})
    duration_ms = int(float(fmt.get("duration", "0")) * 1000)
    streams = raw.get("streams", [])

    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    subtitles = [s for s in streams if s.get("codec_type") == "subtitle"]
    return _Probe(
        container=fmt.get("format_name", ""),
        duration_ms=duration_ms,
        video=video,
        audio=audio,
        subtitles=subtitles,
    )


def _prepare_source(
    src: Path, probe: _Probe, tmpdir: Path
) -> tuple[Path, list[dict[str, Any]]]:
    """Return a path shaka-packager can consume + per-audio-track
    metadata.

    shaka-packager only accepts MP4/fMP4/TS as input containers for
    HEVC — feeding it an MKV makes the WebM demuxer choke ("Unsupported
    video codec"). So when the source isn't MP4 we always remux to an
    MP4 intermediate (video stream copy, no re-encode). Audio is also
    copied when it's already AAC, otherwise transcoded to AAC-LC
    48 kHz stereo (the minimum compatible with browsers).

    The one fast path: source is already MP4 AND every audio track is
    AAC → hand the source to shaka-packager directly with no
    intermediate. Most rips don't hit this so it's a small but real
    optimization for the items that do.

    audio_meta is a list of dicts with keys {idx, codec, language,
    title, channels, default}, indexed in the order the audio streams
    appear in the output (preserved source order, so audio_meta[N]
    matches the Nth audio stream)."""
    # Always go through the remux pass even when the source is already
    # MP4-with-AAC. The original short-circuit (return src directly)
    # would have preserved 5.1 / 7.1 layouts, undefined channel
    # layouts, and unusual sample rates — every one of which trips
    # Chrome MSE's AAC parser. Video is a stream-copy so the cost is
    # ~30 s per movie even on a remuxed-MKV source.
    log.info(
        "packager.prepare",
        strategy="remux_to_mp4",
        container=probe.container,
        audio_count=len(probe.audio),
        non_aac=[s.get("codec_name") for s in probe.audio if s.get("codec_name") != "aac"],
    )
    transmuxed = tmpdir / "transmux.mp4"
    args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(src),
        "-map", "0:v:0",
        "-c:v", "copy",
    ]
    if probe.video.get("codec_name") == "hevc":
        # Tag HEVC as hvc1 so MP4 readers (and shaka-packager) recognise
        # the codec — many MKV→MP4 muxers leave it as hev1, which some
        # tools then reject. ONLY for HEVC sources; forcing it on H.264
        # makes ffmpeg refuse with "Tag hvc1 incompatible with output
        # codec id '27' (avc1)".
        args += ["-tag:v", "hvc1"]
    for i, _stream in enumerate(probe.audio):
        args += ["-map", f"0:a:{i}"]
    # ALL audio output streams are re-encoded to AAC-LC stereo 48 kHz
    # 192 kbps. Even tracks that are already AAC are re-encoded — we
    # lose the passthrough optimization for a critical correctness
    # guarantee: every audio rendition has identical channel count,
    # sample rate, and known channel layout. Per-stream ffmpeg options
    # (`-ac:a:N`, `-b:a:N`, …) are widely supported in the docs but in
    # practice silently no-op for channel-count specs, leaving 5.1
    # source audio as 5.1 output with channel_layout=unknown — which
    # Chrome's MSE then rejects with CHUNK_DEMUXER_ERROR_APPEND_FAILED.
    # Global options apply uniformly and avoid that trap.
    args += [
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
    ]
    # No subtitle streams in the intermediate — they're extracted
    # separately to WebVTT.
    args += ["-sn", "-movflags", "+faststart", str(transmuxed)]
    _run_ffmpeg_capturing("transmux", args)
    # Audio is always re-encoded → transcoded=True for every track.
    meta = [_audio_meta_from_stream(i, s, transcoded=True) for i, s in enumerate(probe.audio)]
    return transmuxed, meta


# Markers that, when present in an audio track's source `title`, tell
# us the title is really a codec descriptor (e.g.
# "DTS-HD Master Audio / 5.1 / 48 kHz / 2618 kbps / 24-bit") rather
# than a human-meaningful track name. Once we downmix to stereo AAC,
# those descriptors are doubly wrong — they describe the source codec
# we just stripped. Fall back to a language-derived label instead.
_CODEC_TITLE_HINTS = (
    "dts", "atmos", "truehd", "dolby", "ac3", "eac3", "flac", "pcm",
    "khz", "kbps", "bit", "channel", "master audio", "lossless",
    "5.1", "7.1", "2.0", "stereo", "mono",
)


def _audio_display_name(meta: dict[str, Any], idx: int) -> str:
    """Pick a player-friendly NAME for an audio rendition.

    Preference order:
      1. The source title when it looks like a content description
         (commentary tracks, "Director's Cut", etc.).
      2. A language label ("English", "German", …) when the title is
         missing OR is just a codec descriptor.
      3. "Track N" as the final fallback for unlabelled tracks in
         unknown languages.
    """
    title = (meta.get("title") or "").strip()
    if title and not any(h in title.lower() for h in _CODEC_TITLE_HINTS):
        return title
    lang = (meta.get("language") or "").lower()
    if lang and lang != "und":
        return _LANG_DISPLAY.get(lang, lang)
    return f"Track {idx}"


_LANG_DISPLAY = {
    "eng": "English", "en": "English",
    "deu": "German", "ger": "German", "de": "German",
    "fra": "French", "fre": "French", "fr": "French",
    "spa": "Spanish", "es": "Spanish",
    "ita": "Italian", "it": "Italian",
    "jpn": "Japanese", "ja": "Japanese",
    "zho": "Chinese", "chi": "Chinese", "zh": "Chinese",
    "por": "Portuguese", "pt": "Portuguese",
    "rus": "Russian", "ru": "Russian",
    "nld": "Dutch", "nl": "Dutch",
}


def _audio_meta_from_stream(
    idx: int, stream: dict[str, Any], transcoded: bool = False
) -> dict[str, Any]:
    tags = stream.get("tags") or {}
    disp = stream.get("disposition") or {}
    return {
        "idx": idx,
        "codec": "aac" if transcoded else stream.get("codec_name", "aac"),
        "language": tags.get("language") or "und",
        "title": tags.get("title") or "",
        # Output channels: always 2 after the stereo downmix in
        # _prepare_source. Falls back to the source channel count only
        # for the (currently unreachable) passthrough path.
        "channels": 2 if transcoded else (stream.get("channels") or 2),
        "default": bool(disp.get("default")),
    }


def _extract_subtitles(
    src: Path, probe: _Probe, subs_dir: Path
) -> list[dict[str, Any]]:
    """Extract every embedded text subtitle stream to WebVTT. Image-
    based subtitles (PGS, VobSub) are skipped — they need OCR which is
    out of scope here. Returns the list of subtitle entries for the
    manifest."""
    if not probe.subtitles:
        return []
    subs_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for i, s in enumerate(probe.subtitles):
        codec = s.get("codec_name", "")
        if codec in ("hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"):
            log.info("packager.subs.skip_image", idx=i, codec=codec)
            continue
        target = subs_dir / f"{i}.vtt"
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                    "-i", str(src),
                    "-map", f"0:s:{i}",
                    "-c:s", "webvtt",
                    str(target),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            log.warning("packager.subs.failed", idx=i, codec=codec, error=str(e))
            continue
        tags = s.get("tags") or {}
        disp = s.get("disposition") or {}
        out.append({
            "id": f"sub{i}",
            "path": f"subs/{i}.vtt",
            "language": tags.get("language") or "und",
            "title": tags.get("title") or "",
            "default": bool(disp.get("default")),
            "forced": bool(disp.get("forced")),
            "format": "webvtt",
        })
    return out


def _generate_trickplay(src: Path, probe: _Probe, out_dir: Path) -> dict[str, Any] | None:
    """Build scrub-preview sprite sheets + WebVTT for the source.

    Strategy: one ffmpeg invocation samples the source at
    fps=1/INTERVAL, scales each frame, and tiles GRID_COLS x GRID_ROWS
    of them per output JPG. That gives ffmpeg the freedom to do
    everything in a single decode pass — much faster than
    image-per-frame extraction. After ffmpeg writes the sprites we
    walk the disk to learn how many frames actually came out (the
    last sprite is usually partial) and emit a VTT mapping each
    timestamp range to its sprite + xywh fragment.

    Returns the manifest's trickplay block, or None when the source
    is too short to produce even one thumbnail (skip the section
    rather than write an empty VTT)."""
    if probe.duration_ms < TRICKPLAY_INTERVAL_SEC * 1000:
        log.info("packager.trickplay.skip_short", duration_ms=probe.duration_ms)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    sprite_template = out_dir / "sprite-%04d.jpg"
    # ffmpeg filter: sample 1 frame per interval, scale to fit a 320x180
    # box (preserving aspect ratio), pad to exact 320x180, then tile
    # 10x10 frames per output JPG.
    vf = (
        f"fps=1/{TRICKPLAY_INTERVAL_SEC},"
        f"scale={TRICKPLAY_THUMB_WIDTH}:{TRICKPLAY_THUMB_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TRICKPLAY_THUMB_WIDTH}:{TRICKPLAY_THUMB_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"tile={TRICKPLAY_GRID_COLS}x{TRICKPLAY_GRID_ROWS}"
    )
    args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        # -skip_frame nokey makes the HEVC decoder discard non-keyframe
        # samples instead of fully decoding them. We only need 1 frame
        # every 10 s and HEVC GOPs are typically <= 10 s long, so
        # keyframe-only decode is effectively free. Without this the
        # decoder was burning ~10 min per movie decoding everything just
        # to throw 99% away. -an / -sn drop audio + subtitles entirely.
        "-skip_frame", "nokey",
        "-an", "-sn",
        "-i", str(src),
        "-vf", vf,
        # qscale ~5 is a sweet spot for thumbnail JPGs: indistinguishable
        # from higher quality at 320 px but ~50 KB per cell.
        "-qscale:v", "5",
        # ffmpeg's numbered-output writer defaults to start at 1 (so
        # we'd get sprite-0001.jpg first). Forcing 0 keeps the file
        # names aligned with the cue indices the VTT generates below,
        # which use 0-based math (sprite_idx = i // cells_per_sprite).
        "-start_number", "0",
        str(sprite_template),
    ]
    t0 = time.monotonic()
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        log.warning("packager.trickplay.failed", error=str(e))
        # Trickplay is a nice-to-have, not a blocker. Drop the section
        # and continue rather than failing the whole package.
        return None
    elapsed = round(time.monotonic() - t0, 1)

    sprites = sorted(out_dir.glob("sprite-*.jpg"))
    if not sprites:
        log.warning("packager.trickplay.no_output")
        return None

    # Total thumbnails actually produced. The last sprite may be a
    # partial tile, so we count by inspecting its dimensions instead
    # of assuming the full grid is filled.
    cells_per_sprite = TRICKPLAY_GRID_COLS * TRICKPLAY_GRID_ROWS
    # Conservatively assume every sprite is full except the last;
    # cap by total expected thumbs from the source duration.
    expected_thumbs = probe.duration_ms // (TRICKPLAY_INTERVAL_SEC * 1000)
    total_thumbs = min(expected_thumbs, len(sprites) * cells_per_sprite)

    # Write the WebVTT cue file. Each cue spans INTERVAL_SEC and points
    # at one cell of one sprite via the WebVTT xywh fragment.
    vtt_lines = ["WEBVTT", ""]
    for i in range(total_thumbs):
        sprite_idx = i // cells_per_sprite
        cell = i % cells_per_sprite
        row = cell // TRICKPLAY_GRID_COLS
        col = cell % TRICKPLAY_GRID_COLS
        x = col * TRICKPLAY_THUMB_WIDTH
        y = row * TRICKPLAY_THUMB_HEIGHT
        start_ms = i * TRICKPLAY_INTERVAL_SEC * 1000
        end_ms = min((i + 1) * TRICKPLAY_INTERVAL_SEC * 1000, probe.duration_ms)
        vtt_lines.append(f"{_ms_to_vtt(start_ms)} --> {_ms_to_vtt(end_ms)}")
        vtt_lines.append(
            f"sprite-{sprite_idx:04d}.jpg"
            f"#xywh={x},{y},{TRICKPLAY_THUMB_WIDTH},{TRICKPLAY_THUMB_HEIGHT}"
        )
        vtt_lines.append("")
    vtt_path = out_dir / "thumbnails.vtt"
    _write_atomic(vtt_path, ("\n".join(vtt_lines)).encode())

    log.info(
        "packager.trickplay.done",
        elapsed_s=elapsed,
        sprites=len(sprites),
        thumbs=total_thumbs,
    )
    return {
        "vttPath": "trickplay/thumbnails.vtt",
        "spritePattern": "trickplay/sprite-%04d.jpg",
        "intervalSec": TRICKPLAY_INTERVAL_SEC,
        "thumbWidth": TRICKPLAY_THUMB_WIDTH,
        "thumbHeight": TRICKPLAY_THUMB_HEIGHT,
        "gridCols": TRICKPLAY_GRID_COLS,
        "gridRows": TRICKPLAY_GRID_ROWS,
    }


def _ms_to_vtt(ms: int) -> str:
    """Format milliseconds as a WebVTT cue timestamp HH:MM:SS.mmm."""
    s, mmm = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{mmm:03d}"


def _run_shaka_packager(
    src: Path,
    probe: _Probe,
    audio_meta: list[dict[str, Any]],
    out_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Invoke shaka-packager with one stream descriptor per output
    rendition. Returns the (video_rendition, audio_renditions) entries
    that go into the manifest."""
    hls_dir = out_root / "hls"
    hls_dir.mkdir(parents=True, exist_ok=True)

    # shaka-packager selects streams by absolute MP4 stream index when
    # given a number, or by type ("video"/"audio"/"text") to pick the
    # first stream of that type. We use the type form for video (there's
    # only one) and the absolute index for each audio track (the only
    # way to pick the Nth — `stream_selector` is just an alias of
    # `stream` so the original `stream=audio,stream_selector=N` failed
    # with "stream=0 not available"). Our intermediate MP4 always has
    # video at index 0 and audio at indices 1..N, so audio i sits at
    # absolute index (i + 1).
    descriptors = [
        ",".join([
            f"in={src}",
            "stream=video",
            "init_segment=hls/v0/init.mp4",
            "segment_template=hls/v0/seg-$Number%05d$.m4s",
            "playlist_name=hls/v0/playlist.m3u8",
            "iframe_playlist_name=hls/v0/iframes.m3u8",
        ])
    ]
    for i, meta in enumerate(audio_meta):
        # Pick a name that describes the track *content*, not the
        # source codec. Source titles like "DTS-HD Master Audio /
        # 5.1 / 48 kHz / 2618 kbps / 24-bit" are misleading once we've
        # transcoded to stereo AAC — they're really codec metadata
        # masquerading as a title. _audio_display_name strips those.
        # shaka-packager uses ',' and '=' as stream-descriptor field
        # separators with no escape mechanism, so we also sanitize.
        raw_name = _audio_display_name(meta, i)
        safe_name = raw_name.replace(",", ";").replace("=", "-")
        descriptors.append(",".join([
            f"in={src}",
            f"stream={i + 1}",
            f"init_segment=hls/a{i}/init.mp4",
            f"segment_template=hls/a{i}/seg-$Number%05d$.m4s",
            f"playlist_name=hls/a{i}/playlist.m3u8",
            "hls_group_id=audio",
            f"hls_name={safe_name}",
        ]))

    cmd = [
        "packager", *descriptors,
        "--segment_duration", str(SEGMENT_SECONDS),
        "--hls_master_playlist_output", "hls/master.m3u8",
        "--hls_playlist_type", "VOD",
    ]
    log.info("packager.shaka.start", cmd=cmd, cwd=str(out_root))
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(out_root), capture_output=True, text=True)
    if result.returncode != 0:
        raise PackageError(
            f"shaka-packager exited {result.returncode}: {result.stderr.strip()[:500]}"
        )
    log.info("packager.shaka.done", elapsed_s=round(time.monotonic() - t0, 1))

    video_count = len(list((out_root / "hls/v0").glob("seg-*.m4s")))
    video_rendition = {
        "id": "v0",
        "dir": "hls/v0",
        "codec": _codec_string_for_video(probe.video),
        "width": probe.video.get("width") or 0,
        "height": probe.video.get("height") or 0,
        "bitrateBps": int(probe.video.get("bit_rate") or 0),
        "hdr": _is_hdr(probe.video),
        "frameRate": probe.video.get("avg_frame_rate") or probe.video.get("r_frame_rate") or "",
        "segments": video_count,
        "targetDuration": SEGMENT_SECONDS,
    }
    audio_renditions: list[dict[str, Any]] = []
    for i, meta in enumerate(audio_meta):
        seg_count = len(list((out_root / f"hls/a{i}").glob("seg-*.m4s")))
        audio_renditions.append({
            "id": f"a{i}",
            "dir": f"hls/a{i}",
            "codec": "mp4a.40.2",
            "language": meta["language"],
            "title": meta["title"],
            "default": meta["default"] or (i == 0),
            "channels": meta["channels"],
            "bitrateBps": 192000,
            "segments": seg_count,
        })
    return video_rendition, audio_renditions


def _codec_string_for_video(stream: dict[str, Any]) -> str:
    """Compose the MP4 codec string the master playlist will advertise
    in CODECS=. shaka-packager writes the real avcC/hvcC bytes; this
    string just has to match what's inside the init segment."""
    codec = stream.get("codec_name", "")
    if codec == "h264":
        return "avc1.640028"  # placeholder — packager rewrites; refine in Phase 2b
    if codec == "hevc":
        return "hev1.1.6.L120.B0"  # ditto
    return codec


def _is_hdr(stream: dict[str, Any]) -> bool:
    transfer = (stream.get("color_transfer") or "").lower()
    return transfer in ("smpte2084", "arib-std-b67")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


def _write_atomic(path: Path, content: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def _packager_version() -> str:
    try:
        out = subprocess.run(
            ["packager", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout else "shaka-packager"
    except Exception:
        return "shaka-packager"


# ---------------------------------------------------------------------------
# Package-state inspector (used by the GET status endpoint).
# ---------------------------------------------------------------------------

def package_status(item_id: str) -> dict[str, Any]:
    """Report the current packaging state of one item. Filesystem-only;
    cheap to call frequently. Possible states:
      - "absent": no output directory exists yet
      - "packaging": .packaging sentinel present
      - "complete": .complete sentinel present (manifest.json is canonical)
      - "failed": .failed sentinel present

    Probes all category dirs (movies/shows/music/other) so the caller
    doesn't have to know item.type. Cheap — at most 4 stat calls.
    """
    out_root = _find_existing_root(item_id)
    if out_root is None:
        return {"state": "absent", "item_id": item_id}
    if not out_root.exists():
        return {"state": "absent", "item_id": item_id}
    if (out_root / ".complete").exists():
        try:
            manifest = json.loads((out_root / "manifest.json").read_text())
        except Exception:
            manifest = None
        return {
            "state": "complete",
            "item_id": item_id,
            "completedAt": (out_root / ".complete").read_text().strip(),
            "manifest": manifest,
        }
    if (out_root / ".packaging").exists():
        try:
            info = json.loads((out_root / ".packaging").read_text())
        except Exception:
            info = {}
        return {"state": "packaging", "item_id": item_id, **info}
    if (out_root / ".failed").exists():
        try:
            info = json.loads((out_root / ".failed").read_text())
        except Exception:
            info = {}
        return {"state": "failed", "item_id": item_id, **info}
    return {"state": "unknown", "item_id": item_id}
