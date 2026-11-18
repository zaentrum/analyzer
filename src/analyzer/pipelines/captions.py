"""Read whether an original's video carries closed captions, window by
window, and say "none" only where that is certain.

Closed captions (EIA/CEA-608/708) ride in a video stream itself: an A53
cc_data SEI (ITU-T T.35, "GA94") with its pictures, as broadcast and disc
sources carry them. Nothing in a container says so, and a probe that decodes
no frame can't tell: ffprobe gives a stream's closed_captions only once it has
decoded a frame of it. So `read` decodes PACKETS video packets in each of
several windows of the video, about two seconds of a film each, and nothing
else of the file, and looks for what the decoder exports with each frame
whose SEI carries them (the side data "ATSC A53 Part 4 Closed Captions"), or
the stream's closed_captions=1, which 7.1 then gives.

The windows: the video's start, read from its first keyframe, and 10, 30,
50, 70 and 90 % of the length its file gives (WINDOWS), each at most TAIL_S
(30 s), or half the video, short of its end; those that fall on one another
are one (plan: the catalog requires the same set of the length a read
reports). Each is read from the keyframe nearest its point: the one before
it where the container's index says so (Matroska, MP4), else the first after
it (a transport stream). A window that decodes no frame of its first PACKETS
packets (a file cut from a stream starts inside a group of pictures, a seek
can land inside one) looks for the keyframe among the KEY_SCAN_PACKETS video
packets from the same point, none of them decoded, and decodes the packets
up to it and PACKETS after it, counted from that point. Every window is read
and kept on its own: where it was placed (at), the presentation times of the
first and last frame it decoded (start, end), how many, and whether one
showed captions.

Only "none" has to be true, so a read that can't vouch for its windows fails,
and a failed read is unknown, never "none". A read fails when the file
isn't there, ffprobe can't open it, finds no video stream in it, does not
answer in time (CAPTIONS_READ_TIMEOUT_S, 120 s, for the whole read) or is
stopped; when the file gives no length, or frames without presentation times
(a raw stream); and when the length it gives can't be the video's: its last
video packet lies more than DURATION_SLACK_S (10 s), or DURATION_SLACK (2 %)
of it, from where that length ends, or its bytes, at the bitrate of its first
frames, last less than a tenth of it (BITRATE_FACTOR: a clock that started
again, read as one that wrapped). A window fails when ffprobe exits with an
error or prints no JSON, when no keyframe is among its packets, when it
decoded no frame, when its first frame lies more than AIM_SLACK_S (15 s) from
where it was placed - up to BACK_SLACK_S (30 s) before it where the packets
from its seek show no keyframe between its first frame and there (its
keyframe_before: the seek landed on the last keyframe before its place, in a
long group of pictures) - when its frames' times are negative, when it
decodes frames another window decoded, or when it showed no captions while
ffprobe said an error of the storage (`Read error`, `Input/output error`: it
may have cut it short). A failure of the storage or of the worker is
transient: it says nothing of the file.

The ffprobe is not the image's. The image's own ffmpeg and ffprobe (Ubuntu
22.04's 4.4.2, on the PATH) read the chapters and run every other pipeline;
this read runs only the pipeline's ffprobe 7.1, the static build of the same
release the transcoder and chino-stream run (CAPTIONS_FFPROBE,
/opt/ffmpeg-7.1/bin/ffprobe in the image), and never the PATH's. `probe` says
which ffprobe is there (`ffprobe -version`), and a read refuses one that is
not of the 7.1 release (WrongProbe): what it read is reported with its
version.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import structlog

from .chapters import STORAGE_ERRORS

log = structlog.get_logger(__name__)

# The ffprobe this read runs where CAPTIONS_FFPROBE names none: the image's
# copy of the pipeline's 7.1 build. Never the PATH's (the image's 4.4.2).
DEFAULT_FFPROBE = "/opt/ffmpeg-7.1/bin/ffprobe"
# The release this read runs, and refuses any other.
RELEASE = "7.1"
# The side data a decoded frame exports of its A53 closed captions.
SIDE_DATA = "ATSC A53 Part 4 Closed Captions"
# The video packets decoded in a window: about two seconds of a film (the
# packager's, and the catalog's, CLOSED_CAPTION_PACKETS); a claim may name
# another count, up to PACKETS_MAX.
PACKETS = 48
PACKETS_MAX = 1000
# How far a keyframe is looked for, in video packets, none decoded: 48 s at
# 25 frames a second, 20 s at 60 - past the longest group of pictures of a
# stream a capture is cut from.
KEY_SCAN_PACKETS = 1200
# How long a read may take, all of it: six windows of a 4K remux on a storage
# that answers slowly take seconds, a stalled one longer.
# CAPTIONS_READ_TIMEOUT_S overrides it.
READ_TIMEOUT_S = 120.0
# The niceness a read runs at, where there is `nice`.
NICE = 10

# The windows after the start, as fractions of the video's length, and how
# far a window stays short of the video's end (or half of it, the less).
WINDOWS = (0.1, 0.3, 0.5, 0.7, 0.9)
TAIL_S = 30.0
# Two windows' places closer than this are one.
AT_SLACK = 1e-3
# How far a window's first decoded frame may lie from where it was placed
# (the start's at the file's start, every other one at the start's first
# frame and its share of the length, as the catalog places them): more than
# a group of pictures a source keeps (x264's own reach 10 s), so a window
# further off landed elsewhere. The catalog's WindowPlaceSlack. Before its
# place a window may lie up to BACK_SLACK_S where the packets from its seek,
# none decoded, show no keyframe between its first frame and its place: the
# seek landed on the last keyframe before it (keyframe_before), in a group
# of pictures that long; a keyframe between them is one the seek passed over.
AIM_SLACK_S = 15.0
BACK_SLACK_S = 30.0
# How far after the first decoded frame a window's keyframe may lie (the
# leading pictures of an open group of pictures come before it), and how
# near its place the packets listed from its seek must reach.
LEAD_S = 1.0
REACH_S = 1.0
# How far past where the file says it ends the length's check lists packets:
# past the slack, so a packet there shows the length short.
END_MARGIN_S = 5.0
# How many frames a window decodes at least to be one: half its packets, and
# never fewer than the catalog's MinWindowFrames.
MIN_WINDOW_FRAMES = 24
# How far the video's last packet may lie from where the length its file
# gives ends: the larger of DURATION_SLACK_S and DURATION_SLACK of it.
DURATION_SLACK_S = 10.0
DURATION_SLACK = 0.02
# A file whose bytes, at the bitrate of its first frames, last less than a
# BITRATE_FACTOR-th of the length it gives can't be that long (a clock that
# started again, read as one that wrapped). One way only: a dark first frame
# is small, and makes the bytes last longer.
BITRATE_FACTOR = 10.0

# The closed lists: what the read knows how to see, each entry for a reason
# the README's tables give. A file that meets anything off them is read as
# not known, saying what (off_list: "container:flv", "codec:vc1",
# "stream:data:bin_data", "video-streams:2", "picture:vbi-lines").
# Containers whose streams the header probe lists all of, caption tracks too.
CONTAINERS = ("matroska,webm", "mov,mp4,m4a,3gp,3g2,mj2", "mpegts", "mpeg", "avi")
# Video codecs whose 7.1 decoder exports the closed captions a picture
# carries (A53 side data: H.264 and HEVC SEI, MPEG-1/2 picture user data,
# AV1 ITU-T T.35 metadata through libdav1d); those whose packets are scanned
# for the A53 payload besides (HEVC's suffix SEI, which 7.1 doesn't export).
VIDEO_CODECS = ("h264", "hevc", "mpeg1video", "mpeg2video", "av1")
SCANNED_CODECS = ("hevc",)
# A video stream the container marks as an attached file - Matroska's
# attachment (its filename and mimetype tags) or the attached_pic disposition
# (ffmpeg's mark of Matroska's image attachments and MP4's cover, covr) - is
# a still picture, no film, when it is an image and has no length of its own:
# skipped, whatever its file name. An image: its mimetype is image/*; an MP4
# cover has no mimetype, the type of its data (JPEG, PNG, BMP) comes as its
# codec, one of COVER_CODECS. No length of its own: the container counts no
# more than one frame of it (nb_frames, the NUMBER_OF_FRAMES tag) and gives it
# no DURATION tag; ffprobe gives a stream the file's duration where the
# container gives none, so its duration is no mark. A video stream of an
# image that isn't so marked is a video stream like any other, off the codec
# list.
COVER_CODECS = ("mjpeg", "png", "bmp")
# Streams of closed captions of their own: counted (caption_streams).
CAPTION_CODECS = ("eia_608",)
CAPTION_TAGS = ("c608", "c708")
# Subtitle streams: the subtitle rule holds them, they are no closed captions.
SUBTITLE_CODECS = (
    "subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text", "microdvd", "hdmv_pgs_subtitle",
    "dvd_subtitle", "dvb_subtitle", "dvb_teletext",
)  # fmt: skip
# Data streams that carry no captions: timecode, splice markers, ID3, EPG.
DATA_CODECS = ("scte_35", "timed_id3", "epg")
DATA_TAGS = ("tmcd",)
# And an MP4's chapter track (chapter_track), by this rule, the same words in
# the analyzer, the record logic (schemas) and the waves' check: a data stream
# is the file's chapter track, and on the list, only where all of these hold in
# the stored probe: the container is the MP4 family (mov,mp4,m4a,3gp,3g2,mj2);
# the stream is codec_type data, codec_name bin_data, codec_tag_string text;
# the file has chapters; the stream's sample count (nb_frames) equals the
# number of chapters; and it is the file's only such stream. A probe that lacks
# the tag or the count leaves it off the list. A package loses no chapters by
# it: they come from the catalog's marks (read from the original's record or
# header by the chapter backfill), the build hands them to the packager, and
# the version keeps them.
CHAPTER_CONTAINER = "mov,mp4,m4a,3gp,3g2,mj2"
CHAPTER_CODEC, CHAPTER_TAG = "bin_data", "text"
# A picture this tall keeps an analog capture's vertical blanking lines,
# line 21's captions among them, in the picture: they are not read.
VBI_HEIGHTS = (486, 512, 608)
# An A53 cc_data payload's first bytes: ITU-T T.35 "GA94", user_data_type_code 3.
A53_BYTES = b"GA94\x03"


def header_args() -> list[str]:
    """ffprobe's arguments to list every stream of the file and its format,
    nothing decoded but what ffprobe probes."""
    return [
        "-v", "error", "-show_entries",
        "format=format_name,duration,start_time:stream=index,codec_type,codec_name,"
        "codec_tag_string,width,height,field_order,nb_frames:stream_disposition=attached_pic:"
        "stream_tags:chapter=id",
        "-of", "json",
    ]  # fmt: skip


def decode_args(packets: int, at: float | None = None, stream: int = 0) -> list[str]:
    """ffprobe's arguments to decode `packets` packets of the stream (by its
    index) from the time at (the file's start when None), and show what
    tells their captions and fields, their times and sizes."""
    start = "" if at is None else f"{at:.3f}"
    return [
        "-v", "error", "-select_streams", str(stream), "-read_intervals", f"{start}%+#{packets}",
        "-show_frames", "-show_entries",
        "frame=pts_time,best_effort_timestamp_time,pkt_size,interlaced_frame,repeat_pict"
        ":frame_side_data=side_data_type:stream=closed_captions,avg_frame_rate,r_frame_rate",
        "-of", "json",
    ]  # fmt: skip


def key_args(at: float | None = None, stream: int = 0) -> list[str]:
    """ffprobe's arguments to list the flags of KEY_SCAN_PACKETS packets of
    the stream from the time at (the file's start when None), none decoded
    ("K__" a keyframe's)."""
    start = "" if at is None else f"{at:.3f}"
    return [
        "-v", "error", "-select_streams", str(stream), "-read_intervals",
        f"{start}%+#{KEY_SCAN_PACKETS}", "-show_entries", "packet=flags", "-of", "csv=p=0",
    ]  # fmt: skip


def end_args(at: float, until: float, stream: int = 0) -> list[str]:
    """ffprobe's arguments to list the times of the stream's packets from the
    time at (where the file's length says it ends: the last keyframe before
    it) to the end of the file, or to the time until where it goes on past
    it, none decoded: their presentation times, and their decoding times for
    a container that gives no other (AVI). No count of packets bounds it: a
    last group of pictures is listed whole, however long."""
    return [
        "-v", "error", "-select_streams", str(stream), "-read_intervals",
        f"{at:.3f}%{until:.3f}", "-show_entries", "packet=pts_time,dts_time", "-of", "csv=p=0",
    ]  # fmt: skip


def list_args(at: float, stream: int = 0) -> list[str]:
    """ffprobe's arguments to list the stream's packets from the time at,
    seeking as a window's decode does, up to it, none decoded: their times
    and flags ("K__" a keyframe's)."""
    return [
        "-v", "error", "-select_streams", str(stream), "-read_intervals",
        f"{at:.3f}%{at + 0.001:.3f}", "-show_entries", "packet=pts_time,dts_time,flags",
        "-of", "csv=p=0",
    ]  # fmt: skip


def scan_args(packets: int, at: float | None = None, stream: int = 0) -> list[str]:
    """ffprobe's arguments to print the bytes of `packets` packets of the
    stream from the time at, none decoded."""
    start = "" if at is None else f"{at:.3f}"
    return [
        "-v", "error", "-select_streams", str(stream), "-read_intervals", f"{start}%+#{packets}",
        "-show_entries", "packet=data", "-show_data", "-of", "json",
    ]  # fmt: skip


# A release in ffprobe's version line: "n7.1.5-12-g1fdbca85aa-20260731" (a
# static build of the 7.1 branch), "4.4.2-0ubuntu0.22.04.1", "7.1"; not a
# build named by its commit ("1fdbca8") or a master build ("N-117000-g…").
_RELEASE = re.compile(r"n?(\d+)\.(\d+)(?:\.(\d+))?(?=$|[^\d.])")
_LIBRARY = re.compile(r"^(lib\w+)\s+(\d+)\.\s*(\d+)\.\s*(\d+)\s*/", re.MULTILINE)
# The libraries of a release branch, which a build named by its commit says
# it is of (libavutil, libavcodec and libavformat, major and minor).
BRANCH_LIBRARIES = {
    "7.1": {"libavutil": (59, 39), "libavcodec": (61, 19), "libavformat": (61, 7)},
}


class WrongProbe(Exception):
    """The ffprobe this read would run is not the pipeline's 7.1."""


@dataclass(frozen=True)
class Probe:
    """An ffprobe as `ffprobe -version` says it is: its version line (line),
    the release it names (release: "7.1.5", or "7.1" for a build named by
    its commit whose libraries are the branch's), its libraries' versions,
    and why it could not be asked (error)."""

    path: str
    line: str | None = None
    release: str | None = None
    libraries: dict[str, str] = field(default_factory=dict)
    by_libraries: bool = False
    error: str | None = None

    @property
    def branch(self) -> str | None:
        """The release branch, "7.1" of "7.1.5"; None when it can't tell."""
        if not self.release:
            return None
        return ".".join(self.release.split(".")[:2])

    @property
    def ok(self) -> bool:
        """Whether this is the ffprobe the read runs: the 7.1 release."""
        return self.error is None and self.branch == RELEASE

    @property
    def name(self) -> str | None:
        """What a read says read it, as the catalog takes it (a release
        first): the version line ("ffprobe version n7.1.5-12-g1fdbca85aa-
        20260731"), or, where the line names no release, the branch its
        libraries say with the line and the libraries ("7.1 by its libraries
        (ffprobe version 1fdbca8: libavutil 59.39.100, ...)"). None when it
        can't be named."""
        if not self.line or not self.release:
            return None
        if not self.by_libraries:
            return self.line[:200]
        libs = ", ".join(
            f"{lib} {self.libraries[lib]}"
            for lib in ("libavutil", "libavcodec", "libavformat")
            if lib in self.libraries
        )
        return f"{self.release} by its libraries ({self.line}: {libs})"[:200]

    def refusal(self) -> str | None:
        """Why a read refuses this ffprobe; None when it runs it."""
        if self.error:
            return f"{self.error} (CAPTIONS_FFPROBE names {self.path})"
        if self.ok:
            return None
        found = self.release or "a release it does not name"
        return (
            f"{self.path} is ffprobe {found} ({self.line or 'no version line'}), "
            f"not the pipeline's {RELEASE}.x: the captions are read with {RELEASE} only"
        )


def ffprobe_path() -> str:
    """The ffprobe this read runs: CAPTIONS_FFPROBE, else DEFAULT_FFPROBE.
    Never the PATH's."""
    return os.environ.get("CAPTIONS_FFPROBE") or DEFAULT_FFPROBE


def parse_version(path: str, text: str) -> Probe:
    """The Probe of `ffprobe -version`'s answer `text`."""
    first = text.splitlines()[0] if text.strip() else ""
    words = first.split()
    line = " ".join(words[:3])[:200] or None
    libraries = {
        m.group(1): f"{m.group(2)}.{m.group(3)}.{m.group(4)}" for m in _LIBRARY.finditer(text)
    }
    if len(words) < 3 or words[:2] != ["ffprobe", "version"]:
        return Probe(path, line=line, libraries=libraries, error="it does not say it is ffprobe")
    m = _RELEASE.match(words[2])
    if m:
        release = ".".join(g for g in m.groups() if g is not None)
        return Probe(path, line=line, release=release, libraries=libraries)
    for branch, want in BRANCH_LIBRARIES.items():
        have = {lib: tuple(int(x) for x in v.split(".")[:2]) for lib, v in libraries.items()}
        if all(have.get(lib) == mm for lib, mm in want.items()):
            return Probe(path, line=line, release=branch, libraries=libraries, by_libraries=True)
    return Probe(path, line=line, libraries=libraries)


def probe(path: str) -> Probe:
    """What the ffprobe at path is, by `ffprobe -version`: asked again
    whenever the file there is another (its inode, its time or its size
    changed), so a binary replaced in place is never taken for the one
    before it."""
    try:
        st = os.stat(path)
    except OSError:
        return Probe(path, error=f"no ffprobe to run at {path}")
    return _probe(path, st.st_ino, st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=8)
def _probe(path: str, _inode: int, _mtime_ns: int, _size: int) -> Probe:
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        return Probe(path, error=f"no ffprobe to run at {path}")
    try:
        r = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return Probe(path, error=f"{path} -version did not run: {e}")
    if r.returncode != 0:
        return Probe(path, error=f"{path} -version exited {r.returncode}")
    return parse_version(path, r.stdout or "")


# what the tests clear between made-up ffprobes
probe.cache_clear = _probe.cache_clear  # type: ignore[attr-defined]


def require(path: str | None = None) -> Probe:
    """The ffprobe at path (ffprobe_path()) when it is the 7.1 release;
    raises WrongProbe, saying why, when it is not."""
    p = probe(path or ffprobe_path())
    why = p.refusal()
    if why:
        raise WrongProbe(why)
    return p


@dataclass
class Window:
    """What one window of a read decoded: where it was placed (at: the
    fraction of the video's length, 0 its start; aim: the time it aimed at),
    its frames, the presentation times of the first and the last of them
    (start, end: the file's own seconds, of the frames decoded, not of the
    point the window aimed at), whether one showed captions (captions: None
    when the window failed), from which packet after its point it read
    (start_packet: the keyframe, where its first packets decoded none), or
    why it failed (error; transient for the storage or the worker)."""

    at: float
    aim: float | None = None
    frames: int = 0
    captions: bool | None = None
    start: float | None = None
    end: float | None = None
    start_packet: int = 0
    error: str | None = None
    transient: bool = False
    said: str = ""  # what ffprobe said on stderr
    # its frames in fields (interlaced_frame), and those that repeat a field
    # (repeat_pict: soft pulldown)
    interlaced: int = 0
    repeats: int = 0
    # whether the packets from its seek show no keyframe between its first
    # frame and its place (the keyframe it was read from is the last before
    # it); False for one that starts after its place
    keyframe_before: bool = False

    @property
    def label(self) -> str:
        return "start" if self.at < AT_SLACK else f"{round(self.at * 100, 1):g}%"

    def fail(self, why: str, transient: bool = False) -> None:
        self.error, self.transient, self.captions = why[:300], transient, None

    def __str__(self) -> str:
        if self.error:
            return f"{self.label}: failed"
        key = f" from packet {self.start_packet}" if self.start_packet else ""
        span = f" {self.start:.3f}-{self.end:.3f} s" if self.start is not None else ""
        return f"{self.label}: {self.frames} frames{key}{span}" + (
            ", captions" if self.captions else ""
        )


@dataclass
class CaptionsRead:
    """What a read of an original found: whether it could tell (ok; error
    says why not), whether its video carries closed captions (captions), how
    many video frames its windows decoded (frames), what each window did
    (windows), the length the windows were placed by (duration), from which
    video packet the start window read (start_packet: the first keyframe,
    where the first packets decoded none), and the ffprobe that read it
    (probe: its version). unreadable: the read failed as a whole, not one of
    its windows (the file, its length, its clock)."""

    path: str
    ok: bool
    captions: bool | None = None
    frames: int = 0
    start_packet: int = 0
    windows: list[Window] = field(default_factory=list)
    duration: float | None = None
    # what the header probe found: streams of captions of their own, the
    # film's codec, and what is off the closed lists (None: not probed)
    caption_streams: int | None = None
    codec: str | None = None
    off_list: list[str] = field(default_factory=list)
    error: str | None = None
    note: str | None = None
    seconds: float = 0.0
    probe: str | None = None
    file_size: int | None = None
    unreadable: bool = False
    # the read ran out of its time (timed_out), or was stopped (interrupted);
    # transient: it failed for the storage or the worker, which says nothing
    # of the file
    timed_out: bool = False
    interrupted: bool = False
    transient: bool = False


class _Failed(Exception):
    """A run, or the read, failed: why, and whether for the storage or the
    worker."""

    def __init__(
        self, why: str, transient: bool, *, timed_out: bool = False, interrupted: bool = False
    ) -> None:
        super().__init__(why)
        self.transient, self.timed_out, self.interrupted = transient, timed_out, interrupted


@dataclass
class _Look:
    """What a window's runs gave: the video stream, the frames decoded, what
    ffprobe said, the file's format (its length), the packet after the
    window's point the decoded ones started from, and why none decoded."""

    streams: list[dict]
    frames: list[dict]
    why: str
    fmt: dict
    start_packet: int = 0
    error: str | None = None
    transient: bool = False
    scanned: bool = False  # its packets' bytes held an A53 payload


def plan(duration: float | None) -> list[float]:
    """The windows a read of a video of the length duration places: the
    start (0), and WINDOWS each at most TAIL_S, or half the video, short of
    its end, those that fall on one another one - the set the catalog
    requires of a read that reports that length."""
    limit = 1.0 - min(TAIL_S / duration, 0.5) if duration and duration > 0 else 1.0
    out = [0.0]
    for f in WINDOWS:
        at = min(f, limit)
        if not any(abs(a - at) < AT_SLACK for a in out):
            out.append(at)
    return out


def read(
    path: str,
    *,
    packets: int = PACKETS,
    timeout: float | None = None,
    stop: Callable[[], bool] | None = None,
    ffprobe: str | None = None,
) -> CaptionsRead:
    """Read whether the video of the original at `path` carries closed
    captions, window by window, `packets` video packets decoded in each,
    with the ffprobe at ffprobe (ffprobe_path()), which must be the 7.1
    release (WrongProbe). A read that runs longer than timeout
    (read_timeout(), the whole read), or while stop() says to stop, is
    ended: it failed (timed_out, or interrupted), saying so."""
    p = require(ffprobe)
    packets = min(max(int(packets), 1), PACKETS_MAX)
    t0 = time.monotonic()
    limit = read_timeout() if timeout is None else timeout
    done = CaptionsRead(path=path, ok=False, probe=p.name)
    try:
        try:
            done.file_size = os.stat(path).st_size
        except OSError as e:
            raise _Failed(f"no file to read: {e.strerror or e}", True) from e
        run = _Runner(p, path, t0 + limit, limit, stop)
        head = run.header()  # a file ffprobe can't open fails here
        film = _closed_lists(head, done)
        if done.off_list:
            raise _Failed("off the closed lists: " + ", ".join(done.off_list), False)
        if film is None:
            raise _Failed("no video stream to read", False)
        run.stream, run.codec = int(film["index"]), done.codec
        first = run.look(None, packets)
        if not first.streams:
            said = f": {first.why}" if first.why else ""
            raise _Failed(f"no video stream to read{said}", bool(STORAGE_ERRORS.search(first.why)))
        fmt = head.get("format") if isinstance(head.get("format"), dict) else {}
        origin = _float(fmt.get("start_time")) or 0.0
        length = _float(fmt.get("duration"))
        if first.frames and all(_pts(f) is None for f in first.frames):
            raise _Failed("its frames have no presentation times (a raw stream)", False)
        if not length or length <= 0:
            raise _Failed("its file gives no length to place the windows by", False)
        done.duration = length
        _check_length(run, first, origin, length, done.file_size)
        place = origin
        for at in plan(length):
            aim = place + at * length
            look = first if at == 0 else run.look(aim, packets, tolerate=True)
            before = run.keyframe_before(aim, look)
            w = _window(at, aim, look, packets, done.windows, before)
            done.windows.append(w)
            if at == 0 and w.start is not None and not w.error:
                # the catalog places each window from the start's first frame
                place = w.start
    except _Failed as e:
        done.error, done.transient, done.unreadable = str(e)[:500], e.transient, True
        done.timed_out, done.interrupted = e.timed_out, e.interrupted
        done.frames = sum(w.frames for w in done.windows)
        done.seconds = time.monotonic() - t0
        log.warning("captions.read_failed", path=path, error=done.error, transient=e.transient)
        return done
    done.frames = sum(w.frames for w in done.windows)
    done.start_packet = done.windows[0].start_packet
    done.seconds = time.monotonic() - t0
    said = next((w.said for w in done.windows if w.said), "")
    done.note = ("windows: " + "; ".join(str(w) for w in done.windows)
                 + (f"; ffprobe said: {said}" if said else ""))[:500]  # fmt: skip
    failed = [w for w in done.windows if w.error]
    if any(w.captions for w in done.windows) or done.caption_streams:
        done.ok, done.captions = True, True
    elif failed:
        w = failed[0]
        done.error = (w.error if w.at < AT_SLACK else f"the {w.label} window: {w.error}")[:500]
        done.transient = w.transient
        log.warning("captions.read_failed", path=path, error=done.error, transient=w.transient)
        return done
    else:
        done.ok, done.captions = True, False
    log.info(
        "captions.read",
        path=path,
        captions=done.captions,
        frames=done.frames,
        windows=len(done.windows),
        with_captions=sum(1 for w in done.windows if w.captions),
        start_packet=done.start_packet,
        seconds=round(done.seconds, 2),
    )
    return done


def _closed_lists(head: dict, done: CaptionsRead) -> dict | None:
    """Hold the header probe's format and streams to the closed lists: set
    done's caption_streams, codec and off_list, and give the film's stream
    (the one video stream that is no attached picture), None when there is
    none."""
    fmt = head.get("format") if isinstance(head.get("format"), dict) else {}
    off: list[str] = []
    name = str(fmt.get("format_name") or "none")
    if name not in CONTAINERS:
        off.append(f"container:{name}")
    films, captioned = [], 0
    for st in head.get("streams") or []:
        if not isinstance(st, dict):
            continue
        kind, codec = str(st.get("codec_type") or "none"), str(st.get("codec_name") or "none")
        tag = str(st.get("codec_tag_string") or "").lower()
        if codec == "unknown":  # a codec ffmpeg has no name of
            codec = "none"
        what = codec if codec != "none" or not tag or tag.startswith("[") else tag
        if codec in CAPTION_CODECS or tag in CAPTION_TAGS:
            captioned += 1
        elif kind == "video":
            if not attached_picture(st):
                films.append(st)
        elif kind in ("audio", "attachment"):
            continue
        elif kind == "subtitle" and codec in SUBTITLE_CODECS:
            continue
        elif kind == "data" and (
            codec in DATA_CODECS or tag in DATA_TAGS or chapter_track(st, head)
        ):
            continue
        else:
            off.append(f"stream:{kind}:{what}")
    if len(films) > 1:
        off.append(f"video-streams:{len(films)}")
    for st in films:
        codec = str(st.get("codec_name") or "none")
        if codec not in VIDEO_CODECS and f"codec:{codec}" not in off:
            off.append(f"codec:{codec}")
    if any(st.get("height") in VBI_HEIGHTS for st in films):
        off.append("picture:vbi-lines")
    film = films[0] if len(films) == 1 else None
    if film is not None:
        done.codec = str(film.get("codec_name") or "none")
    done.caption_streams, done.off_list = captioned, off
    return film


def _chapter_stream(st: dict) -> bool:
    """codec_type data, codec_name bin_data, codec_tag_string text."""
    return (
        st.get("codec_type") == "data"
        and st.get("codec_name") == CHAPTER_CODEC
        and str(st.get("codec_tag_string") or "").lower() == CHAPTER_TAG
    )


def _chapter_container(head: dict) -> bool:
    """The container is the MP4 family (mov,mp4,m4a,3gp,3g2,mj2)."""
    fmt = head.get("format") if isinstance(head.get("format"), dict) else {}
    return fmt.get("format_name") == CHAPTER_CONTAINER


def _chapter_marks(head: dict) -> int:
    """The number of the file's chapters (0: it has none)."""
    chapters = head.get("chapters")
    return len(chapters) if isinstance(chapters, list) else 0


def _chapter_has(head: dict) -> bool:
    """The file has chapters."""
    return _chapter_marks(head) > 0


def _chapter_count(st: dict, head: dict) -> bool:
    """The stream's sample count (nb_frames) equals the number of chapters; a
    probe that lacks the count leaves it off the list."""
    count = st.get("nb_frames")
    try:
        return (
            int(count) == _chapter_marks(head)
            if count is not None and not isinstance(count, bool)
            else False
        )
    except (TypeError, ValueError):
        return False


def _chapter_only(head: dict) -> bool:
    """It is the file's only such stream."""
    return (
        sum(1 for x in head.get("streams") or [] if isinstance(x, dict) and _chapter_stream(x)) == 1
    )


def chapter_track(st: dict, head: dict) -> bool:
    """Whether a data stream is the file's chapter track, and on the list: only
    where all of these hold in the stored probe: the container is the MP4
    family (mov,mp4,m4a,3gp,3g2,mj2); the stream is codec_type data, codec_name
    bin_data, codec_tag_string text; the file has chapters; the stream's
    sample count (nb_frames) equals the number of chapters; and it is the
    file's only such stream. A probe that lacks the tag or the count leaves it
    off the list."""
    return (
        _chapter_container(head)
        and _chapter_stream(st)
        and _chapter_has(head)
        and _chapter_count(st, head)
        and _chapter_only(head)
    )


def attached_picture(st: dict) -> bool:
    """Whether a video stream of the header probe is a still picture the
    container holds as an attached file (COVER_CODECS' rule): marked as one,
    an image, and of no length of its own."""
    tags = {str(k).lower(): str(v) for k, v in (st.get("tags") or {}).items()}
    pic = bool((st.get("disposition") or {}).get("attached_pic"))
    marked = pic or ("filename" in tags and "mimetype" in tags)
    mime = tags.get("mimetype", "").strip().lower()
    image = mime.startswith("image/") if mime else pic and st.get("codec_name") in COVER_CODECS
    counted = (_float(st.get("nb_frames")) or 0, _float(tags.get("number_of_frames")) or 0)
    length = max(counted) > 1 or any(k.startswith("duration") for k in tags)
    return marked and image and not length


def _check_length(
    run: _Runner, first: _Look, origin: float, length: float, size: int | None
) -> None:
    """Fail the read (_Failed) when the length the file gives can't be its
    video's: its last video packet lies further from where that length ends
    than the slack, or its bytes at the bitrate of its first frames last
    less than a BITRATE_FACTOR-th of it."""
    rate = _bytes_per_second(first)
    if rate and size and length > BITRATE_FACTOR * (size / rate):
        raise _Failed(
            f"its file says it lasts {length:.0f} s, more than {BITRATE_FACTOR:g} times the "
            f"{size / rate:.0f} s its {size} bytes last at the bitrate of its first frames",
            False,
        )
    end = origin + length
    slack = max(DURATION_SLACK_S, DURATION_SLACK * length)
    last = run.last_pts(end, end + slack + END_MARGIN_S)
    if last is None:
        raise _Failed(f"no video packet read where its file says it ends ({end:.3f} s)", False)
    if abs(last - end) > slack:
        raise _Failed(
            f"its file says it ends at {end:.3f} s, its last video packet read there is at "
            f"{last:.3f} s",
            False,
        )


def _window(
    at: float,
    aim: float,
    look: _Look,
    packets: int,
    before: list[Window],
    keyed: tuple[bool, str] = (False, ""),
) -> Window:
    """The Window of what a window's runs gave: its frames, their times and
    captions, or why it can't vouch for them (no frame decoded; none showed
    captions while ffprobe said an error of the storage; its first frame far
    from its aim, unless before it and keyed: its keyframe the last before
    its aim, or why not; times below 0; frames another window decoded): a
    failed window shows no captions either way (None)."""
    said = f": {look.why}" if look.why else ""
    w = Window(at, aim, frames=len(look.frames), start_packet=look.start_packet, said=look.why)
    times = [t for t in (_pts(f) for f in look.frames) if t is not None]
    if times:
        w.start, w.end = round(min(times), 3), round(max(times), 3)
    shown = look.scanned or _shows(look)
    w.interlaced = sum(1 for f in look.frames if f.get("interlaced_frame") == 1)
    w.repeats = sum(1 for f in look.frames if (_float(f.get("repeat_pict")) or 0) > 0)
    storage = bool(STORAGE_ERRORS.search(look.why))
    w.captions = shown
    w.keyframe_before, unkeyed = keyed
    back = None if w.start is None else aim - w.start
    if look.error:
        w.fail(look.error, look.transient)
    elif not look.streams:
        w.fail(f"no video stream to read{said}", storage)
    elif not look.frames:
        w.fail(
            f"no video frame decoded of {packets} packets from the keyframe "
            f"(video packet {look.start_packet}){said}",
            storage,
        )
    elif w.start is None or w.end is None:
        w.fail("its frames have no presentation times")
    elif w.start < 0:
        w.fail(f"its frames' presentation times are negative ({w.start:.3f} s)")
    elif back is not None and (-back > AIM_SLACK_S or back > BACK_SLACK_S):
        w.fail(
            f"its first frame is at {w.start:.3f} s, {abs(back):.1f} s from where it "
            f"was placed ({aim:.3f} s): it landed elsewhere"
        )
    elif back is not None and back > AIM_SLACK_S and not w.keyframe_before:
        w.fail(
            f"its first frame is at {w.start:.3f} s, {back:.1f} s before where it was placed "
            f"({aim:.3f} s), and its packets don't show it the last keyframe before there "
            f"({unkeyed}): it landed elsewhere"
        )
    elif other := _overlapped(w, before):
        w.fail(f"it decoded frames the {other.label} window decoded ({w.start:.3f}-{w.end:.3f} s)")
    elif not shown and storage:
        # the storage may have cut the window short of the frames that carry them
        w.fail(f"ffprobe found no captions and said{said}", True)
    return w


# A line of ffprobe's dump of a packet's bytes: its offset, then up to 16
# bytes in hex in pairs, padded to 41 columns, then the same as text.
_HEX_LINE = re.compile(r"^[0-9a-f]{8}: (.{41})", re.MULTILINE)


def _shows(look: _Look) -> bool:
    """Whether a frame of look shows captions (their side data), or its
    stream says so."""
    return any(
        isinstance(sd, dict) and sd.get("side_data_type") == SIDE_DATA
        for f in look.frames
        for sd in f.get("side_data_list") or []
    ) or any(s.get("closed_captions") == 1 for s in look.streams)


def _overlapped(w: Window, before: list[Window]) -> Window | None:
    """The window of before that decoded frames of w's span, if one did."""
    for o in before:
        if o.error or o.start is None or o.end is None or w.start is None or w.end is None:
            continue
        if w.start <= o.end and o.start <= w.end:
            return o
    return None


def _pts(frame: dict) -> float | None:
    """A decoded frame's presentation time, in the file's seconds."""
    for k in ("pts_time", "best_effort_timestamp_time"):
        v = _float(frame.get(k))
        if v is not None:
            return v
    return None


def _bytes_per_second(look: _Look) -> float | None:
    """The video's bitrate in bytes a second, of the frames a window decoded:
    their bytes over the time they span (a frame more), else, of frames
    without times, at the stream's own frame rate; None when they don't say.
    A header's average frame rate is not taken: a remux can say 600 of 25."""
    sizes = [_float(f.get("pkt_size")) for f in look.frames]
    sizes = [x for x in sizes if x and x > 0]
    if not sizes:
        return None
    times = sorted(t for t in (_pts(f) for f in look.frames) if t is not None)
    if len(times) > 1 and times[-1] > times[0]:
        step = (times[-1] - times[0]) / (len(times) - 1)
        return sum(sizes) / (times[-1] - times[0] + step)
    fps = next((r for s in look.streams if (r := _rate(s.get("r_frame_rate")))), None)
    return sum(sizes) / len(sizes) * fps if fps else None


def _float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _rate(v: object) -> float | None:
    """A frame rate as ffprobe writes it ("25/1"); None for none ("0/0")."""
    if not isinstance(v, str) or "/" not in v:
        return None
    n, d = (_float(x) for x in v.split("/", 1))
    return n / d if n and d else None


class _Runner:
    """The ffprobe runs of one read, niced, within its deadline (end)."""

    def __init__(
        self, p: Probe, path: str, end: float, limit: float, stop: Callable[[], bool] | None
    ) -> None:
        self.p, self.path, self.end, self.limit, self.stop = p, path, end, limit, stop
        self.stream, self.codec = 0, None  # the film's stream, by its index, and codec

    def header(self) -> dict:
        """The header probe: the file's format and every stream."""
        out, _why = self._run(header_args())
        try:
            probed = json.loads(out or "{}")
        except json.JSONDecodeError as e:
            raise _Failed(f"ffprobe's output is no JSON: {e}", False) from e
        if not isinstance(probed, dict):
            raise _Failed("ffprobe's output is no JSON object", False)
        return probed

    def look(self, at: float | None, packets: int, *, tolerate: bool = False) -> _Look:
        """A window from the time at (the file's start when None): its
        first `packets` packets decoded, else, where they decode fewer than
        half as many frames, the packets up to the first keyframe among
        KEY_SCAN_PACKETS from that point and `packets` after it; a window
        still short of half is no window. An HEVC window that shows no
        captions has its packets' bytes scanned for the A53 payload (a
        suffix SEI, which 7.1 doesn't export). tolerate: a run that fails is
        the window's failure (error), not the read's - but the read's timeout
        or stop, which end it."""
        try:
            got, count = self.decode(at, packets), packets
            need = max(MIN_WINDOW_FRAMES, packets / 2)
            if got.streams and len(got.frames) < need:
                key = self.first_keyframe(at)
                if key is None:
                    where = "" if at is None else " from the window's point"
                    got.error = (
                        f"no keyframe among the first {KEY_SCAN_PACKETS} video packets{where}"
                    )
                    return got
                if key:  # at 0 the first packets were read from it already
                    count = key + packets
                    again = self.decode(at, count)
                    again.fmt, again.start_packet = again.fmt or got.fmt, key
                    got = again
                if got.frames and len(got.frames) < need:
                    got.error = (
                        f"it decoded {len(got.frames)} frames, fewer than {need:g} (half its "
                        f"{packets} packets, {MIN_WINDOW_FRAMES} at least), from the keyframe "
                        f"(video packet {got.start_packet})"
                    )
                    return got
            if self.codec in SCANNED_CODECS and got.frames and not _shows(got):
                got.scanned = self.scan(at, count)
            return got
        except _Failed as e:
            if not tolerate or e.timed_out or e.interrupted:
                raise
            return _Look([], [], "", {}, error=str(e), transient=e.transient)

    def decode(self, at: float | None, packets: int) -> _Look:
        out, why = self._run(decode_args(packets, at, self.stream))
        try:
            probed = json.loads(out or "{}")
        except json.JSONDecodeError as e:
            raise _Failed(f"ffprobe's output is no JSON: {e}", False) from e
        if not isinstance(probed, dict):
            raise _Failed("ffprobe's output is no JSON object", False)
        streams = [s for s in probed.get("streams") or [] if isinstance(s, dict)]
        frames = [f for f in probed.get("frames") or [] if isinstance(f, dict)]
        fmt = probed.get("format") if isinstance(probed.get("format"), dict) else {}
        return _Look(streams, frames, why, fmt)

    def first_keyframe(self, at: float | None) -> int | None:
        """The index of the first keyframe among KEY_SCAN_PACKETS video
        packets from the time at; None when none is."""
        out, _why = self._run(key_args(at, self.stream))
        for i, line in enumerate(x for x in (out or "").splitlines() if x.strip()):
            if line.strip().startswith("K"):
                return i
        return None

    def keyframe_before(self, aim: float, look: _Look) -> tuple[bool, str]:
        """Whether the packets from the window's seek to its aim, none
        decoded, show the keyframe it was read from (at its first decoded
        frame, or up to LEAD_S after it) as the last one before its aim: no
        other keyframe between it and there, and the packets reaching it;
        else False, and why. False for a window that failed, and for one that
        starts after its aim. A listing that fails is no proof; the read's
        time and stop end it all the same."""
        times = [t for t in (_pts(f) for f in look.frames) if t is not None]
        if look.error or not times:
            return False, "it decoded no frame"
        start = round(min(times), 3)
        if start > aim + 1e-3:
            return False, "it starts after its place"
        try:
            out, _why = self._run(list_args(aim, self.stream))
        except _Failed as e:
            if e.timed_out or e.interrupted:
                raise
            return False, f"its packets could not be listed: {e}"
        packets = []
        for line in (out or "").splitlines():
            *values, flags = line.strip().split(",") or [""]
            t = next((v for v in (_float(x) for x in values) if v is not None), None)
            if t is not None:
                packets.append((t, flags.startswith("K")))
        keys = [t for t, key in packets if key]
        own = next((t for t in sorted(keys) if t >= start - 1e-3), None)
        if own is None or own > start + LEAD_S:
            return False, "the packets from its seek show no keyframe where its frames begin"
        if max(t for t, _key in packets) < aim - REACH_S:
            return False, "the packets from its seek end short of its place"
        passed = [t for t in keys if own + 1e-3 < t <= aim + 1e-3]
        if passed:
            return False, f"a keyframe at {passed[0]:.3f} s lies between"
        return True, ""

    def last_pts(self, at: float, until: float) -> float | None:
        """The latest presentation time of the video packets read from the
        time at to the end of the file, or to the time until; None when none
        was read."""
        out, _why = self._run(end_args(at, until, self.stream))
        times = []
        for line in (out or "").splitlines():
            # a packet's presentation time, else its decoding time
            values = (_float(x) for x in line.strip().split(","))
            t = next((v for v in values if v is not None), None)
            if t is not None:
                times.append(t)
        return max(times) if times else None

    def scan(self, at: float | None, packets: int) -> bool:
        """Whether the bytes of `packets` packets from the time at hold an
        A53 payload (A53_BYTES), none decoded."""
        out, _why = self._run(scan_args(packets, at, self.stream))
        try:
            probed = json.loads(out or "{}")
        except json.JSONDecodeError:
            return False
        for pk in probed.get("packets") or [] if isinstance(probed, dict) else []:
            hexes = _HEX_LINE.findall(str(pk.get("data") or ""))
            try:
                data = bytes.fromhex("".join(hexes))  # its spaces ignored
            except ValueError:
                continue
            if A53_BYTES in data:
                return True
        return False

    def _run(self, args: list[str]) -> tuple[str, str]:
        """ffprobe's stdout and stderr (folded) of args on the file; _Failed
        when it can't run, exits with an error, outlives the read's deadline
        or is stopped."""
        cmd = [self.p.path, *args, self.path]
        nice = shutil.which("nice")
        if nice:
            cmd = [nice, "-n", str(NICE), *cmd]
        try:
            # its own session: a read that is ended takes whatever it started
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
        except OSError as e:
            raise _Failed(f"ffprobe could not run: {e}", True) from e
        while True:
            try:
                out, err = proc.communicate(
                    timeout=min(1.0, max(self.end - time.monotonic(), 0.01))
                )
                break
            except subprocess.TimeoutExpired:
                if self.stop is not None and self.stop():
                    _kill(proc)
                    raise _Failed("the read was stopped", True, interrupted=True) from None
                if time.monotonic() >= self.end:
                    _kill(proc)
                    raise _Failed(
                        f"ffprobe did not answer within {self.limit:g} s", True, timed_out=True
                    ) from None
        why = " ".join((err or "").split())[-300:]
        if proc.returncode != 0:
            raise _Failed(
                f"ffprobe exited {proc.returncode}" + (f": {why}" if why else ""),
                bool(STORAGE_ERRORS.search(why)),
            )
        return out, why


def _kill(proc: subprocess.Popen) -> None:
    """End a read: its process group killed, its pipes closed."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        log.warning("captions.read_unkilled", pid=proc.pid)


def read_timeout() -> float:
    """How long a read may take: READ_TIMEOUT_S, or CAPTIONS_READ_TIMEOUT_S
    (5 s to an hour)."""
    try:
        v = float(os.environ.get("CAPTIONS_READ_TIMEOUT_S") or READ_TIMEOUT_S)
    except ValueError:
        return READ_TIMEOUT_S
    return min(max(v, 5.0), 3600.0)
