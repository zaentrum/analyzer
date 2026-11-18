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

The windows (the caption-window plan, the same on every side that reads
captions): the picture's start, read from its first keyframe, and 10, 30,
50, 70 and 90 % of the picture's presentation span as its records state it
(WINDOWS, picture_span: its start_time, else the format's, else none and no
plan, plus its duration, else the record's container.durationMs that the
claim hands out, else the format's duration; start + at x (end - start)),
the start window never capped,
each at most TAIL_S (30 s) short of its end: six windows, never merged,
each saying its plan fraction, a capped one too (plan; the read reports the
span it measured, {start, end}, and its gop, beside the length its file
gives, which the catalog holds to its record). Where those windows can't lie
apart (plan_short, over the places themselves as the record logic decides
it: two closer than a window reaches, or on one another), the picture is read
whole instead, every packet decoded, and that whole read stands for them;
"none" needs it complete (_whole: every packet decoded but its leading
pictures, counted from the keyframes' listing, at most WHOLE_LEADING_MAX).
Each window
is read from the keyframe nearest its point: the one before it where the
container's index says so (Matroska, MP4), else the first after it (a
transport stream). A window that decodes fewer frames than
half its first PACKETS packets (MIN_WINDOW_FRAMES, 24, at least: a file cut
from a stream starts inside a group of pictures, a seek can land inside one)
looks for the keyframe among the KEY_SCAN_PACKETS video packets from the same
point, none of them decoded, and decodes the packets up to it and PACKETS
after it, counted from that point. Every window is read and kept on its own:
where it was placed (at), the presentation times of the first and last frame
it decoded (start, end), how many, and whether one showed captions.

Only "none" has to be true, so a read that can't vouch for its windows fails,
and a failed read is unknown, never "none". A verdict rests only on runs
whose output was checked for what it must be, and on none that said anything
on stderr: every run whose output feeds it (a short picture's start window,
replaced by its whole read, feeds it not; the whole run does). A read
fails when its claim names another count of packets than the plan's, when the file
isn't there, ffprobe can't open it, finds no video stream in it, does not
answer in time (RUN_TIMEOUT_S, 180 s, of each run: a window's, a listing's,
the end's, a short picture's whole read; CAPTIONS_READ_TIMEOUT_S, 1800 s, of
the read as a whole) or is
stopped; when the file gives no length, or frames without presentation times
(a raw stream); when its records state no length of its picture; when its
picture's span as its records state it can't be the video's: the end of its
last video packet lies more than a frame step and END_SLACK_S (0.5 s) from
where that span ends, or its first frame decoded from where it starts (the
plan's end tolerance, at both ends: the frame step the header probe's
avg_frame_rate gives, else its r_frame_rate only where that is exactly 0/0,
each num/den in digits alone; a probe that states no such rate, or one below
MIN_RATE, one frame a second, gives no plan, and the read fails); when its
group of pictures (the longest among its first KEY_SCAN_PACKETS packets,
which the read reports as its gop) is longer than those packets last at the
frame step; or its bytes, at the
bitrate of its first frames, last less than a
tenth of the length the file gives (BITRATE_FACTOR: a clock that started
again, read as one that wrapped); when its header probe, its keyframes'
listing or its end's listing says anything on stderr, or a listing is no
listing of packets (empty, a line of anything else, or more packets or fewer
than it asked for, but where they reach the picture's end); when its start
window places no frame (it says no span); and when the file is another after the read than
before it (its inode, its time or its size changed). A window fails when
ffprobe exits with an error, prints no JSON, or says anything on stderr in
any of the window's runs (its decode, its keyframe's, its listing, its packet
scan) while it showed no captions; when no keyframe is among its packets,
when it decoded fewer frames than that half, or a frame it can't place in
time; when its first frame lies further from where it was placed than
place_slack (max(10 s, the GOP and 0.5 s) either way; of a GOP of 0 a frame
step and 0.5 s) - up to BACK_SLACK_S (30 s) before it where the packets from its
seek show no keyframe between its first frame and there (its
keyframe_before: the seek landed on the last keyframe before its place, in a
long group of pictures) - when its frames' times are negative, when it
doesn't start after the window before it, or decodes frames another window
decoded. A failure of the storage (`Read error`, `Input/output error`) or of
the worker is transient: it says nothing of the file.

A picture of a codec whose decoder doesn't export every caption it may carry
(SCANNED_CODECS: HEVC, whose suffix SEI 7.1 doesn't export) is never read as
"none": its windows' packets are scanned for the A53 payload, and say
captions where it is there, but a read of one that shows none is not known:
its windows go to the catalog as read, with captionsKnown false and
HEVC_REASON, and the catalog holds it until the record's captionSignature
releases it. The scan reads ffprobe's dump packet by packet (no more than
SCAN_PACKETS_MAX packets and SCAN_BYTES_MAX of their bytes, the window fails
past them), and fails its window on any line on stderr, fewer packets than it
asked for, a packet whose bytes aren't its size, and output that isn't the
dump.

Every ffprobe a read runs is bounded: one decoding thread (-threads 1, in
every run's arguments: a listing that decodes no frame of its own still
opens the decoders ffprobe probes the streams with), and
its process's address space (LIMIT_AS_BYTES, 1,280 MiB) and CPU time
(LIMIT_CPU_S, 600 s), set before ffprobe starts (RLIMIT_AS, RLIMIT_CPU; a
system that can't lower the address space, macOS, runs without it). A run
that hits one fails, as any run that exits with an error does.

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
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise

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
# plan's packets, and the catalog's CLOSED_CAPTION_PACKETS). The plan's, not
# configuration: a claim that names another count is one a read can't serve,
# and it fails.
PACKETS = 48
# How far a keyframe is looked for, in video packets, none decoded: 48 s at
# 25 frames a second, 20 s at 60 - past the longest group of pictures of a
# stream a capture is cut from.
KEY_SCAN_PACKETS = 1200
# How long each ffprobe run of a read may take, in wall time (the plan's
# bounds: a window's, a listing's, the end's, a short picture's whole read
# alike), and the read as a whole, all its runs: CAPTIONS_READ_TIMEOUT_S
# overrides the second, never the first.
RUN_TIMEOUT_S = 180.0
READ_TIMEOUT_S = 1800.0
# The niceness a read runs at, where there is `nice`.
NICE = 10
# The bounds of every ffprobe a read runs: its decoding threads, and its
# process's address space and CPU seconds (RLIMIT_AS, RLIMIT_CPU).
THREADS = 1
LIMIT_AS_BYTES = 1280 << 20
LIMIT_CPU_S = 600
# The packet scan's bounds: the packets and the bytes of them it reads at
# most. ffprobe's dump of a packet is read and let go packet by packet, so
# the worker holds one packet's dump at a time; a window past either bound
# fails (a transport stream's window up to its keyframe can run to hundreds
# of megabytes of a 4K film).
SCAN_PACKETS_MAX = 1000
SCAN_BYTES_MAX = 128 << 20

# The slowest frame rate, in frames a second, a header may state of its
# picture and give a plan (the plan's minRate): a slower one would stretch a
# frame step's tolerance past any window.
MIN_RATE = 1
# The most digits a number written in a probe has (a frame rate's, a count's)
# for the plan's rules to read it: what Python 3.11's int() reads of a text
# (sys.int_info.default_max_str_digits), which the record logic's readers and
# the catalog's hold to. Stated here, as the image's Python 3.11 (Ubuntu
# 22.04's 3.11.0~rc1) has no such limit of its own.
INT_DIGITS = 4300
# The windows after the start, as fractions of the picture's own
# presentation span as its records state it (its end held to the end of its
# last video packet), and how far a window stays short of its end: the
# caption-window plan.
WINDOWS = (0.1, 0.3, 0.5, 0.7, 0.9)
TAIL_S = 30.0
# The short-picture rule: where the plan's windows can't be disjoint - two of
# its places closer than a window reaches (its PACKETS at the picture's frame
# step, from a keyframe up to a group of pictures before its place, and
# REACH_MARGIN), or two that fall on one another - the read is the whole
# picture, every packet decoded; that whole read stands for the windows, and
# "none" needs it complete: every packet but those before the first keyframe
# decoded, every frame placed in time, from within the plan's end tolerance (a
# frame step and END_SLACK_S) of the picture's start to within it of its end.
REACH_MARGIN_S = 0.5
# How many of a whole read's packets may lead its first keyframe (the
# plan's wholeLeadingMax): the leading pictures a decoder may leave out.
WHOLE_LEADING_MAX = 16
# Two windows' places closer than this are one.
AT_SLACK = 1e-3
# How far a window's first decoded frame may lie from where it was placed,
# either way (the plan's placeSlack, place_slack): PLACE_SLACK_S (x264's own
# reach of a group of pictures, 10 s), or the group of pictures the first
# packets showed and PLACE_MARGIN_S where that is longer, so a window further
# off landed elsewhere; of a GOP of 0 (fewer than two keyframes among them)
# only at its own place, within a frame step and END_SLACK_S. Before its
# place a window may lie up to BACK_SLACK_S where the packets from its seek,
# none decoded, show no keyframe between its first frame and its place: the
# seek landed on the last keyframe before it (keyframe_before), in a group
# of pictures that long; a keyframe between them is one the seek passed over.
PLACE_SLACK_S = 10.0
PLACE_MARGIN_S = 0.5
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
# The plan's end tolerance: how far the span measured - its first frame, the
# end of its last video packet - may lie from the picture's span as its
# records state it, at its start and at its end: a frame step and END_SLACK_S.
END_SLACK_S = 0.5
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
# What a read of such a picture that found no captions says on the wire: its
# windows as read, captionsKnown false and this reason; the catalog holds the
# original until the record's captionSignature releases it.
HEVC_REASON = "hevc: needs the packets' signature"
# A video stream the container marks as an attached file - Matroska's
# attachment (its filename and mimetype tags) or the attached_pic disposition
# (ffmpeg's mark of Matroska's image attachments and MP4's cover, covr) - is
# a still picture, no film, when it is an image and has no length of its own:
# skipped, whatever its file name. An image: its codec is one of
# COVER_CODECS (JPEG, PNG, BMP), and its mimetype, where it has one, image/*
# (an MP4 cover has none); a stream of any other codec is a video stream, a
# second film, whatever its marks say. No length of its own: the container counts no
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
# And an MP4's chapter track (chapter_track), by this rule: a data stream is
# the file's chapter track, and on the list, only where all of these hold in
# the stored probe: the container is the MP4 family (mov,mp4,m4a,3gp,3g2,mj2);
# the stream is codec_type data, codec_name bin_data, codec_tag_string text
# (exactly: not TEXT, not Text); the file has chapters; the stream's sample
# count (nb_frames, ffprobe's: a number written in digits, nothing around
# them) equals the number of chapters; and it is the file's only such stream.
# A probe that lacks the tag or the count leaves it off the list. A package loses no chapters by
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


# Every run's arguments state its decoding threads (THREADS, the plan's
# bounds): a run that decodes no frame of its own still has the decoders that
# ffprobe's probing of the streams opens.


def header_args() -> list[str]:
    """ffprobe's arguments to list every stream of the file and its format,
    nothing decoded but what ffprobe probes."""
    return [
        "-v", "error", "-threads", str(THREADS), "-show_entries",
        "format=format_name,duration,start_time:stream=index,codec_type,codec_name,"
        "codec_tag_string,width,height,field_order,nb_frames,start_time,duration,"
        "avg_frame_rate,r_frame_rate:"
        "stream_disposition=attached_pic:"
        "stream_tags:chapter=id",
        "-of", "json",
    ]  # fmt: skip


def decode_args(packets: int, at: float | None = None, stream: int = 0) -> list[str]:
    """ffprobe's arguments to decode `packets` packets of the stream (by its
    index) from the time at (the file's start when None), and show what
    tells their captions and fields, their times and sizes."""
    start = "" if at is None else f"{at:.3f}"
    return [
        "-v", "error", "-threads", str(THREADS), "-select_streams", str(stream),
        "-read_intervals", f"{start}%+#{packets}", "-show_frames", "-show_entries",
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
        "-v", "error", "-threads", str(THREADS), "-select_streams", str(stream),
        "-read_intervals", f"{start}%+#{KEY_SCAN_PACKETS}", "-show_entries", "packet=flags",
        "-of", "csv=p=0",
    ]  # fmt: skip


def keys_args(stream: int = 0) -> list[str]:
    """ffprobe's arguments to list the times and flags ("K__" a keyframe's)
    of the stream's first KEY_SCAN_PACKETS packets from the file's start, in
    decode order, none decoded: its groups of pictures and its leading
    pictures (the plan's gop, and wholeComplete's leading)."""
    return [
        "-v", "error", "-threads", str(THREADS), "-select_streams", str(stream),
        "-read_intervals", f"%+#{KEY_SCAN_PACKETS}", "-show_entries", "packet=pts_time,flags",
        "-of", "csv=p=0",
    ]  # fmt: skip


def end_args(at: float, until: float, stream: int = 0) -> list[str]:
    """ffprobe's arguments to list the times of the stream's packets from the
    time at (where the file's length says it ends: the last keyframe before
    it) to the end of the file, or to the time until where it goes on past
    it, none decoded: their presentation times, and their decoding times for
    a container that gives no other (AVI). No count of packets bounds it: a
    last group of pictures is listed whole, however long."""
    return [
        "-v", "error", "-threads", str(THREADS), "-select_streams", str(stream),
        "-read_intervals", f"{at:.3f}%{until:.3f}", "-show_entries",
        "packet=pts_time,dts_time,duration_time", "-of", "csv=p=0",
    ]  # fmt: skip


def whole_args(stream: int = 0) -> list[str]:
    """ffprobe's arguments to decode every packet of the stream (a short
    picture's whole read), counting them, and show what a window's decode
    shows of each frame."""
    return [
        "-v", "error", "-threads", str(THREADS), "-select_streams", str(stream), "-count_packets",
        "-show_frames", "-show_entries",
        "frame=pts_time,best_effort_timestamp_time,pkt_size,interlaced_frame,repeat_pict"
        ":frame_side_data=side_data_type:stream=closed_captions,nb_read_packets",
        "-of", "json",
    ]  # fmt: skip


def list_args(at: float, stream: int = 0) -> list[str]:
    """ffprobe's arguments to list the stream's packets from the time at,
    seeking as a window's decode does, up to it, none decoded: their times
    and flags ("K__" a keyframe's)."""
    return [
        "-v", "error", "-threads", str(THREADS), "-select_streams", str(stream),
        "-read_intervals", f"{at:.3f}%{at + 0.001:.3f}", "-show_entries",
        "packet=pts_time,dts_time,flags", "-of", "csv=p=0",
    ]  # fmt: skip


def scan_args(packets: int, at: float | None = None, stream: int = 0) -> list[str]:
    """ffprobe's arguments to print the size and the bytes of `packets`
    packets of the stream from the time at, none decoded, in its default
    format: a section a packet ([PACKET] ... [/PACKET]), its bytes a hex dump,
    line by line (_Dump reads it as it comes)."""
    start = "" if at is None else f"{at:.3f}"
    return [
        "-v", "error", "-threads", str(THREADS), "-select_streams", str(stream),
        "-read_intervals", f"{start}%+#{packets}", "-show_entries", "packet=size,data",
        "-show_data", "-of", "default",
    ]  # fmt: skip


# Each ffprobe a read runs is started through this, run by the analyzer's
# own Python (-I -S: nothing of the environment's): its address space and
# CPU seconds bounded (RLIMIT_AS, RLIMIT_CPU), then the command run in its
# place. A system that can't lower the address space (macOS) runs without
# that bound; any other failure to set one fails the run.
_LIMITED = (
    "import os, resource, sys\n"
    "space, cpu = int(sys.argv[1]), int(sys.argv[2])\n"
    "resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))\n"
    "try:\n"
    "    resource.setrlimit(resource.RLIMIT_AS, (space, space))\n"
    "except (ValueError, OSError):\n"
    "    if sys.platform != 'darwin':\n"
    "        raise\n"
    "os.execvp(sys.argv[3], sys.argv[3:])\n"
)


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
    # a short picture's whole read (the short-picture rule): every packet of
    # it decoded, how many packets it has
    whole: bool = False
    packets: int = 0
    # of those, how many lead its first keyframe (the packets before it)
    leading: int = 0

    @property
    def label(self) -> str:
        if self.whole:
            return "whole"
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
    (windows), the length its file gives (duration: the catalog holds it to
    its record's; it places no window), the picture's span the windows were
    placed over, as it was measured (span: {"start", "end"}, the first frame
    of its start and the end of its last video packet), the longest group of
    pictures among its first packets (gop, seconds), from which video packet
    the start window read (start_packet: the first keyframe, where the first
    packets decoded none), and the ffprobe that read it (probe: its
    version). unreadable: the read
    failed as a whole, not one of its windows (the file, its length, its
    clock, a file that changed while it was read), or can't say "none" of
    its picture at all (SCANNED_CODECS)."""

    path: str
    ok: bool
    captions: bool | None = None
    frames: int = 0
    start_packet: int = 0
    windows: list[Window] = field(default_factory=list)
    duration: float | None = None
    span: dict | None = None
    # the longest group of pictures among its picture's first KEY_SCAN_PACKETS
    # packets, seconds at the frame step (0: fewer than two keyframes): the
    # plan's gop, which a read reports (None: not listed)
    gop: float | None = None
    # the frame step it read by, and what gave it (frame_step; not on the wire)
    step: float | None = None
    step_from: str | None = None
    # whether the windows can tell captions at all (False: HEVC that showed
    # none, captions_known false on the wire, with its reason)
    captions_known: bool = True
    reason: str | None = None
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


def plan(start: float, end: float) -> list[tuple[float, float]]:
    """The plan's six windows over a picture whose span is start to end
    (seconds of its own clock): [(at, place)], each at its own fraction (at:
    0 the start, then WINDOWS; a window the tail caps still says its
    fraction), placed as the record logic's plan_places places it, in the
    same arithmetic: the start at start itself, never capped; each other at
    start + at x (end - start), at most TAIL_S short of end, never before
    start. Always six, never merged: places that coincide, or lie closer than
    a window reaches, make it a short picture, read whole (plan_short)."""
    out = [(0.0, start)]
    for at in WINDOWS:
        out.append((at, max(start, min(start + at * (end - start), end - TAIL_S))))
    return out


def plan_short(start: float, end: float, gop: float, step: float) -> bool:
    """Whether the plan's windows over a picture of the span start to end
    can't be disjoint (the short-picture rule, as the record logic's
    plan_short decides it, over the places themselves): two of its places
    fall on one another, or lie closer than a window reaches - its PACKETS at
    the frame step, from a keyframe up to a group of pictures (gop) before
    its place, and REACH_MARGIN_S. Such a picture is read whole."""
    places = sorted(p for _at, p in plan(start, end))
    reach = PACKETS * step + gop + REACH_MARGIN_S
    return any(b - a < reach for a, b in pairwise(places))  # closer than a window reaches


# A frame rate as the plan reads one (its frameStep): "num/den" of digits
# alone, no space, sign or point.
_RATE = re.compile(r"([0-9]+)/([0-9]+)")


def frame_step(film: dict) -> tuple[float | None, str | None]:
    """(step, from) of the picture's frame step (the plan's frameStep), as
    the record logic's frame_step_of reads it: one over the header probe's
    avg_frame_rate of its film, or its r_frame_rate only where
    avg_frame_rate is exactly the text "0/0", each a ratio "num/den" of
    digits alone (no space, sign or point), both above 0, divided as Python
    divides - what the catalog knows without reading. (None, None) where the
    rate it takes is no such ratio (an avg_frame_rate that is missing or of
    another form among them: r_frame_rate stands in for "0/0" alone), and
    (None, why) where it is below MIN_RATE frames a second (0/1 among them):
    there is no plan, and the read fails (the catalog can't know a step a
    read would measure)."""
    key = "r_frame_rate" if film.get("avg_frame_rate") == "0/0" else "avg_frame_rate"
    rate = film.get(key)
    m = _RATE.fullmatch(rate) if isinstance(rate, str) else None
    if m is None or max(len(m.group(1)), len(m.group(2))) > INT_DIGITS:
        return None, None
    num, den = int(m.group(1)), int(m.group(2))
    if den <= 0:
        return None, None
    slow = (None, f"{key} below one frame a second")  # MIN_RATE
    if num <= 0:
        return slow
    try:
        step = den / num
    except OverflowError:  # a step too large for a float: slower than any rate
        return slow
    if step <= 0:  # a rate past any a float holds: no step at all
        return None, None
    return (step, key) if step <= 1 / MIN_RATE else slow


def place_slack(gop: float, step: float) -> float:
    """How far a window's first frame may lie from its place, either way (the
    plan's placeSlack): max(PLACE_SLACK_S, the GOP and PLACE_MARGIN_S); of a
    GOP of 0, its own place alone (a frame step and END_SLACK_S)."""
    return max(PLACE_SLACK_S, gop + PLACE_MARGIN_S) if gop else step + END_SLACK_S


def read(
    path: str,
    *,
    packets: int = PACKETS,
    timeout: float | None = None,
    stop: Callable[[], bool] | None = None,
    ffprobe: str | None = None,
    duration_ms: int | float | None = None,
) -> CaptionsRead:
    """Read whether the video of the original at `path` carries closed
    captions, window by window, PACKETS video packets decoded in each (the
    plan's count: packets is what the claim names, and a read of another
    count fails, as one it can't serve), with the ffprobe at ffprobe
    (ffprobe_path()), which must be the 7.1 release (WrongProbe), over the
    picture's span as its records state it (picture_span: duration_ms is its
    source record's container.durationMs, which the claim hands out). A read
    that runs longer than timeout (read_timeout(), the whole read), or while
    stop() says to stop, is ended: it failed (timed_out, or interrupted),
    saying so."""
    p = require(ffprobe)
    t0 = time.monotonic()
    limit = read_timeout() if timeout is None else timeout
    done = CaptionsRead(path=path, ok=False, probe=p.name)
    try:
        if type(packets) is not int or packets != PACKETS:
            raise _Failed(
                f"its claim names {str(packets)[:20]} video packets a window, where the plan "
                f"reads {PACKETS}: a read it can't serve",
                False,
            )
        try:
            before = os.stat(path)
        except OSError as e:
            raise _Failed(f"no file to read: {e.strerror or e}", True) from e
        done.file_size = before.st_size
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
        length = _float(fmt.get("duration"))
        if first.frames and all(_pts(f) is None for f in first.frames):
            raise _Failed("its frames have no presentation times (a raw stream)", False)
        if not length or length <= 0:
            raise _Failed("its file gives no length to place the windows by", False)
        done.duration = length
        claim = picture_span(head, film, duration_ms)
        if claim is None:
            fmt0 = head.get("format") if isinstance(head.get("format"), dict) else {}
            stated = _seconds(film.get("start_time")) is not None or (
                _seconds(fmt0.get("start_time")) is not None
            )
            raise _Failed(
                (
                    "its records state no length of its picture (its picture's duration, its "
                    "record's container.durationMs, its format's duration)"
                    if stated
                    else "its records state no start of its picture (its picture's start_time, "
                    "its format's)"
                )
                + ": no plan, its windows can't be placed",
                False,
            )
        place, end, whose = claim
        if end <= place:
            raise _Failed("its records state a picture that ends at or before it starts", False)
        step, stated = frame_step(film)
        if step is None:
            raise _Failed(
                (
                    f"its header probe states a frame rate of its picture {stated}"
                    if stated
                    else "its header probe states no frame rate of its picture (avg_frame_rate, "
                    "r_frame_rate)"
                )
                + ": no plan, its windows can't be placed",
                False,
            )
        done.step_from = stated
        done.step = step
        last = _check_length(run, first, length, end, step, whose, done.file_size)
        # the plan's gop and a whole read's leading pictures, of a listing of
        # the first packets held to what it must be (keys)
        gop, leading = run.keys(step, place, end)
        done.gop = gop
        if gop > KEY_SCAN_PACKETS * step:
            # longer than the packets it was listed from: not known
            raise _Failed(
                f"its group of pictures ({gop:.3f} s) is longer than {KEY_SCAN_PACKETS} frame "
                f"steps ({KEY_SCAN_PACKETS * step:.3f} s): not known",
                False,
            )
        # max(10 s, the GOP and 0.5 s) either way; of a GOP of 0 a window
        # counts only at its own place, and no keyframe before it counts
        slack = place_slack(gop, step)
        # the start window, then the others over the picture's span as its
        # records state it, its end held to the video's last packet
        start = _window(
            0.0, place, first, packets, [], run.keyframe_before(place, first, gop), slack
        )
        done.windows.append(start)
        # the span's start held as its end is: the start's first frame within
        # a frame step and END_SLACK_S of the start its records state
        if start.start is not None and abs(start.start - place) > step + END_SLACK_S:
            raise _Failed(
                f"its picture starts at {place:.3f} s as its records state it ({whose}), its "
                f"first frame decoded is at {start.start:.3f} s: more than a frame step "
                f"({step:.3f} s) and {END_SLACK_S:g} s apart",
                False,
            )
        # the span measured (the plan's report shape): the first frame of its
        # start, the end of its last video packet - only where the start
        # window placed a frame: a read that measured no start says no span
        # (and is no read the catalog takes as one), never a span of none
        if start.start is not None:
            done.span = {"start": start.start, "end": last}
        if plan_short(place, end, gop, step):
            # the short-picture rule: the whole picture, every packet decoded,
            # stands for the windows. The verdict rests on the runs whose
            # output feeds it: here the whole read's (and what it says on
            # stderr fails it, as any run's does). The start window's decode
            # only measured the span's start; its window is replaced, so what
            # it said, or why it failed, is not what "none" rests on: the
            # whole run decodes the same first packets, and anything it says
            # of them fails the whole read.
            whole = _whole(run.whole(), place, end, leading, step + END_SLACK_S, step)
            done.windows = [whole]
        for at, aim in [] if done.windows[0].whole else plan(place, end)[1:]:
            look = run.look(aim, packets, tolerate=True)
            keyed = run.keyframe_before(aim, look, gop)
            done.windows.append(_window(at, aim, look, packets, done.windows, keyed, slack))
        try:
            after = os.stat(path)
        except OSError as e:
            raise _Failed(f"the file is gone after its read: {e.strerror or e}", True) from e
        if (after.st_ino, after.st_mtime_ns, after.st_size) != (
            before.st_ino,
            before.st_mtime_ns,
            before.st_size,
        ):
            raise _Failed(
                "the file changed while it was read (its inode, its time or its size)", True
            )
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
    elif done.codec in SCANNED_CODECS:
        # never "none" of a picture whose decoder doesn't export every caption
        # it may carry: its windows sent as read, its captions not known until
        # its packets' signature is on record (HEVC_REASON)
        done.captions_known, done.reason = False, HEVC_REASON
        done.error = (
            f"its picture is {done.codec}, whose decoder doesn't export a caption a "
            "suffix SEI carries, and whose packets aren't read whole here: its "
            "windows can't tell none"
            + (f" (and the {failed[0].label} window: {failed[0].error})" if failed else "")
        )[:500]
        log.warning("captions.read_not_known", path=path, reason=done.reason, error=done.error)
        return done
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
    """codec_type data, codec_name bin_data, codec_tag_string text, each
    exactly."""
    return (
        st.get("codec_type") == "data"
        and st.get("codec_name") == CHAPTER_CODEC
        and st.get("codec_tag_string") == CHAPTER_TAG
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
    count = st.get("nb_frames")  # ffprobe's: a string of digits, nothing else
    return isinstance(count, str) and _digits(count) == _chapter_marks(head)


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
    """Whether a video stream of the header probe is a cover, a still picture
    the container holds as an attached file, by the closed lists' cover rule
    (the plan's words.picture, as the record logic's attached_picture reads
    it) - all of these: marked as one (the attached_pic disposition as the
    number 1 or true, or both a filename and a mimetype tag, never one
    alone); an image of a cover's codec (COVER_CODECS, always tested), its
    mimetype, where it has one, image/* (never in the codec's stead), and
    without one the attached_pic disposition (an MP4 cover); of at most one
    frame by every count it is given (nb_frames, a tag that starts
    NUMBER_OF_FRAMES), each in digits alone (_digits) - a count that is
    there but can't be read is no cover's -; and of no length of its own (no
    tag that starts DURATION). Tags are read without case; two keys that
    differ by it alone and disagree make it no image, so no cover."""
    raw = st.get("tags") if isinstance(st.get("tags"), dict) else {}
    tags = {str(k).lower(): v for k, v in raw.items()}
    clash = any(tags[str(k).lower()] != v for k, v in raw.items())
    disposition = st.get("disposition") if isinstance(st.get("disposition"), dict) else {}
    pic = _mark(disposition.get("attached_pic"))
    marked = pic or ("filename" in tags and "mimetype" in tags)
    mime = str(tags.get("mimetype") or "").strip().lower()
    image = (
        not clash
        and st.get("codec_name") in COVER_CODECS  # with a mimetype or without
        and (mime.startswith("image/") if "mimetype" in tags else pic)
    )
    # its frame counts: nb_frames and every NUMBER_OF_FRAMES... tag (by its
    # start, as the DURATION tags are); one there that can't be read is no
    # cover's
    counts = [st["nb_frames"]] if "nb_frames" in st else []
    counts += [v for k, v in tags.items() if k.startswith("number_of_frames")]
    one_frame = all(n is not None and n <= 1 for n in map(_digits, counts))
    length = any(k.startswith("duration") for k in tags)
    return bool(marked and image and one_frame and not length)


def _mark(v: object) -> bool:
    """A disposition's mark: the number 1 or the boolean true, nothing else
    (1.0, "1" and "0" are no mark)."""
    return v is True or (type(v) is int and v == 1)


def _digits(v: object) -> int | None:
    """A count as a probe states it - a whole number from 0, or the same
    written in digits alone - as an int, else None: "1e3", " 900", "+900",
    "0x384", "1_0", "1.0", 1.0, ["900"] and more than INT_DIGITS digits are
    no counts (the record logic's gate_digits)."""
    if isinstance(v, int) and not isinstance(v, bool):
        return v if 0 <= v < 10**INT_DIGITS else None
    if not isinstance(v, str) or len(v) > INT_DIGITS or _DIGITS.fullmatch(v) is None:
        return None
    return int(v)


_DIGITS = re.compile(r"[0-9]+")


def picture_span(
    head: dict, film: dict, duration_ms: int | float | None
) -> tuple[float, float, str] | None:
    """(start, end, whose) of the picture's presentation span as its records
    state it, in the plan's order (plan_of, as the catalog's PictureSpan reads
    it): its start the picture's start_time, else the format's, else none -
    never a start of 0 none states; its length the picture's duration (above
    0), else the record's container.durationMs (the claim's durationMs, a
    number above 0), else the format's duration; no tag (a Matroska
    DURATION) is read for one. None where they state no start or no
    length."""
    fmt = head.get("format") if isinstance(head.get("format"), dict) else {}
    start = _seconds(film.get("start_time"))
    if start is None:
        start = _seconds(fmt.get("start_time"))
    if start is None:
        return None
    length = _seconds(film.get("duration"))
    if length is not None and length > 0:
        return start, start + length, "its picture's start_time and duration"
    if isinstance(duration_ms, int | float) and not isinstance(duration_ms, bool):
        ms = _seconds(duration_ms)
        if ms is not None and ms > 0:
            return start, start + ms / 1000, "its record's durationMs"
    length = _seconds(fmt.get("duration"))  # only where the record gives none
    if length is not None and length > 0:
        return start, start + length, "its format's duration"
    return None


def _seconds(v: object) -> float | None:
    """A time as ffprobe or a record writes it (a string or a number, never
    a boolean), None where it is none."""
    if isinstance(v, bool):
        return None
    t = _float(v)
    return t if t is not None and math.isfinite(t) else None


def _check_length(
    run: _Runner,
    first: _Look,
    length: float,
    end: float,
    step: float,
    whose: str,
    size: int | None,
) -> float:
    """The end of the video's last packet (its presentation time and its
    duration), listed from where the picture's span as its records state it
    ends (end); fail the read (_Failed) when that end can't be its video's:
    its last video packet ends further from it than a frame step and
    END_SLACK_S, or the file's bytes at
    the bitrate of its first frames last less than a BITRATE_FACTOR-th of
    the length the file gives."""
    rate = _bytes_per_second(first)
    if rate and size and length > BITRATE_FACTOR * (size / rate):
        raise _Failed(
            f"its file says it lasts {length:.0f} s, more than {BITRATE_FACTOR:g} times the "
            f"{size / rate:.0f} s its {size} bytes last at the bitrate of its first frames",
            False,
        )
    slack = step + END_SLACK_S
    last = run.last_pts(end, end + slack + END_MARGIN_S)
    if last is None:
        raise _Failed(
            f"no video packet read where its picture ends as its records state it ({end:.3f} s, "
            f"{whose})",
            False,
        )
    if abs(last - end) > slack:
        raise _Failed(
            f"its picture ends at {end:.3f} s as its records state it ({whose}), its last video "
            f"packet read there ends at {last:.3f} s: more than a frame step ({step:.3f} s) and "
            f"{END_SLACK_S:g} s apart",
            False,
        )
    return last


def _window(
    at: float,
    aim: float,
    look: _Look,
    packets: int,
    before: list[Window],
    keyed: tuple[bool, str] = (False, ""),
    slack: float = PLACE_SLACK_S,
) -> Window:
    """The Window of what a window's runs gave: its frames, their times and
    captions, or why it can't vouch for them (no frame decoded; a frame it
    can't place in time; none showed captions while ffprobe said anything on
    stderr in any of its runs; its first frame far from its aim, unless
    before it and keyed: its keyframe the last before its aim, or why not;
    times below 0; a start not after the window before it; frames another
    window decoded): a failed window shows no captions either way (None).
    slack is how far its first frame may lie from its aim, either way
    (place_slack: max(PLACE_SLACK_S, the GOP and PLACE_MARGIN_S); of a GOP of
    0 a frame step and END_SLACK_S, its own place alone)."""
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
    elif len(times) < len(look.frames):
        w.fail(
            f"{len(look.frames) - len(times)} of its {len(look.frames)} frames have no "
            "presentation time: it can't place them"
        )
    elif w.start < 0:
        w.fail(f"its frames' presentation times are negative ({w.start:.3f} s)")
    elif back is not None and (-back > slack or back > BACK_SLACK_S):
        w.fail(
            f"its first frame is at {w.start:.3f} s, {abs(back):.1f} s from where it "
            f"was placed ({aim:.3f} s): it landed elsewhere"
        )
    elif back is not None and back > slack and not w.keyframe_before:
        w.fail(
            f"its first frame is at {w.start:.3f} s, {back:.1f} s before where it was placed "
            f"({aim:.3f} s), and its packets don't show it the last keyframe before there "
            f"({unkeyed}): it landed elsewhere"
        )
    elif other := _overlapped(w, before):
        w.fail(f"it decoded frames the {other.label} window decoded ({w.start:.3f}-{w.end:.3f} s)")
    elif (prev := _previous(before)) is not None and w.start <= prev.end:  # type: ignore[operator]
        w.fail(
            f"it doesn't start after the {prev.label} window ({w.start:.3f} s, that one ends at "
            f"{prev.end:.3f} s)"
        )
    elif not shown and look.why:
        # anything ffprobe said in a run of a window that shows no captions:
        # it may have cut it short of the frames that carry them
        w.fail(f"ffprobe found no captions and said{said}", storage)
    return w


def whole_packets_enough(packets: int, start: float, end: float, step: float) -> bool:
    """Whether a whole read of the picture over [start, end] has packets
    enough: packets x the frame step at least its span less the end
    tolerance (a frame step and END_SLACK_S)."""
    return packets * step >= (end - start) - (step + END_SLACK_S)


def whole_packets_needed(start: float, end: float, step: float) -> float:
    """The fewest packets whole_packets_enough takes, for its words."""
    return ((end - start) - (step + END_SLACK_S)) / step


def _whole(
    look: _Look, start: float, end: float, leading: int | None, reach: float, step: float
) -> Window:
    """The Window of a short picture's whole read (whole_args): its frames,
    their times and captions, or why it can't stand for the windows: its run
    failed or said anything on stderr while it showed no captions, a frame it
    can't place in time, packets too few for the picture's span, leading
    pictures that can't be counted (leading None: the keyframes' listing
    shows none, no time of the first keyframe, or more than
    WHOLE_LEADING_MAX), fewer frames than its packets less the leading ones
    or more than its packets (the plan's wholeComplete), or frames that don't
    reach from within reach (the plan's end tolerance: a frame step and
    END_SLACK_S) of the picture's start to within it of its end."""
    said = f": {look.why}" if look.why else ""
    w = Window(0.0, start, frames=len(look.frames), said=look.why, whole=True,
               leading=leading or 0)  # fmt: skip
    counted = [_float(s.get("nb_read_packets")) for s in look.streams]
    w.packets = int(next((n for n in counted if n is not None), 0))
    times = [t for t in (_pts(f) for f in look.frames) if t is not None]
    if times:
        w.start, w.end = round(min(times), 3), round(max(times), 3)
    shown = _shows(look)
    w.interlaced = sum(1 for f in look.frames if f.get("interlaced_frame") == 1)
    w.repeats = sum(1 for f in look.frames if (_float(f.get("repeat_pict")) or 0) > 0)
    w.captions = shown
    storage = bool(STORAGE_ERRORS.search(look.why))
    if look.error:
        w.fail(look.error, look.transient)
    elif not look.frames or w.start is None or w.end is None:
        w.fail(f"its whole read decoded no frame it can place{said}", storage)
    elif len(times) < len(look.frames):
        w.fail(
            f"{len(look.frames) - len(times)} of its {len(look.frames)} frames have no "
            "presentation time: it can't place them"
        )
    elif not whole_packets_enough(w.packets, start, end, step):
        w.fail(
            f"its whole read has {w.packets} packets, fewer than its span "
            f"({end - start:.3f} s) at its frame step needs "
            f"({whole_packets_needed(start, end, step):.0f})"
        )
    elif leading is None:
        w.fail(
            "its leading pictures can't be counted: its keyframes' listing shows no keyframe, "
            f"no time of the first, or more than {WHOLE_LEADING_MAX} of them"
        )
    elif w.packets <= 0 or not w.packets - leading <= len(look.frames) <= w.packets:
        w.fail(
            f"its whole read decoded {len(look.frames)} frames of {w.packets} packets "
            f"({leading} before the first keyframe)"
        )
    elif w.start > start + reach or w.end < end - reach:
        w.fail(
            f"its frames ({w.start:.3f}-{w.end:.3f} s) don't reach the picture's start and end "
            f"({start:.3f}-{end:.3f} s)"
        )
    elif not shown and look.why:
        w.fail(f"ffprobe found no captions and said{said}", storage)
    return w


def _previous(before: list[Window]) -> Window | None:
    """The last window of before that decoded frames it could place."""
    for o in reversed(before):
        if not o.error and o.start is not None and o.end is not None:
            return o
    return None


# A line of ffprobe's dump of a packet's bytes: its offset, then up to 16
# bytes in hex in pairs, padded to 41 columns, then the same as text.
_HEX_LINE = re.compile(r"^[0-9a-f]{8}: (.{41})", re.MULTILINE)
# A packet's flags as ffprobe 7.1 writes them (K a keyframe, D discard, C
# corrupt, _ none), a time as it writes one (or N/A), and the lines of the
# listings a read holds to them: of the keyframes (keys_args: time, flags), of
# the packets from a seek to its place (list_args: time, decoding time,
# flags) and of the end (end_args: time, decoding time, duration). A line
# may end in empty fields: ffprobe's csv writer gives a transport stream's
# packet one for its side data ("1.480000,K__,"); _fields strips them.
_FLAGS = re.compile(r"[K_][D_][C_]")
_TIME = r"(-?[0-9]+(?:\.[0-9]+)?|N/A)"
_KEYS_LINE = re.compile(rf"{_TIME},([K_][D_][C_])")
_LIST_LINE = re.compile(rf"{_TIME},{_TIME},([K_][D_][C_])")
_END_LINE = re.compile(rf"{_TIME},{_TIME},{_TIME}")


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
    """A number as ffprobe writes one; None where it is none, or isn't finite
    (NaN and Infinity count as not stated)."""
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _no_constant(constant: str) -> None:
    raise json.JSONDecodeError(f"{constant} is no number it states", constant, 0)


def load_json(text: str | bytes) -> object:
    """JSON as the read loads it: NaN, Infinity and -Infinity refused (a
    JSONDecodeError), never taken for a number."""
    return json.loads(text, parse_constant=_no_constant)


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
        """The header probe: the file's format and every stream. Anything it
        says on stderr fails the read."""
        out, why = self._run(header_args())
        if why:
            raise _Failed(f"its header probe said: {why}", bool(STORAGE_ERRORS.search(why)))
        try:
            probed = load_json(out or "{}")
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
            said = [got.why]  # what every run of the window says on stderr
            need = max(MIN_WINDOW_FRAMES, packets / 2)
            if got.streams and len(got.frames) < need:
                key, keyed = self.first_keyframe(at)
                said.append(keyed)
                if key is None:
                    where = "" if at is None else " from the window's point"
                    got.error = (
                        f"no keyframe among the first {KEY_SCAN_PACKETS} video packets{where}"
                    )
                    got.why = _said(said)
                    return got
                if key:  # at 0 the first packets were read from it already
                    count = key + packets
                    again = self.decode(at, count)
                    again.fmt, again.start_packet = again.fmt or got.fmt, key
                    said.append(again.why)
                    got = again
                if got.frames and len(got.frames) < need:
                    got.error = (
                        f"it decoded {len(got.frames)} frames, fewer than {need:g} (half its "
                        f"{packets} packets, {MIN_WINDOW_FRAMES} at least), from the keyframe "
                        f"(video packet {got.start_packet})"
                    )
                    got.why = _said(said)
                    return got
            got.why = _said(said)
            if self.codec in SCANNED_CODECS and got.frames and not _shows(got):
                try:
                    got.scanned = self.scan(at, count)
                except _Failed as e:  # the window's, whichever window it is
                    if e.timed_out or e.interrupted:
                        raise
                    got.error, got.transient = f"its packet scan failed: {e}"[:300], e.transient
            return got
        except _Failed as e:
            if not tolerate or e.timed_out or e.interrupted:
                raise
            return _Look([], [], "", {}, error=str(e), transient=e.transient)

    def decode(self, at: float | None, packets: int) -> _Look:
        out, why = self._run(decode_args(packets, at, self.stream))
        try:
            probed = load_json(out or "{}")
        except json.JSONDecodeError as e:
            raise _Failed(f"ffprobe's output is no JSON: {e}", False) from e
        if not isinstance(probed, dict):
            raise _Failed("ffprobe's output is no JSON object", False)
        streams = [s for s in probed.get("streams") or [] if isinstance(s, dict)]
        frames = [f for f in probed.get("frames") or [] if isinstance(f, dict)]
        fmt = probed.get("format") if isinstance(probed.get("format"), dict) else {}
        return _Look(streams, frames, why, fmt)

    def keys(self, step: float, start: float, end: float) -> tuple[float, int | None]:
        """(gop, leading) of the listing of the first KEY_SCAN_PACKETS video
        packets, none decoded, in decode order with their times and flags
        (keys_args), as the record logic's picture_keys reads it: gop the
        longest run between two keyframes among them, in seconds at the frame
        step (0 where they hold fewer than two keyframes), the plan's gop,
        which the read reports; leading the stream's own leading pictures
        (the plan's wholeComplete: the packets before its first keyframe and
        the first group's packets timed before that keyframe), None where no
        keyframe is listed, its time is not, or they are more than
        WHOLE_LEADING_MAX. A verdict rests only on a listing held to what it
        must be: a line of a packet's time (or N/A) and its flags each, and
        KEY_SCAN_PACKETS of them, or fewer only where they reach the end of
        the picture's span (start to end) at the frame step, as a whole
        read's packets must; anything else (an empty listing, one that is no
        listing, one of more lines than asked for), and anything it says on
        stderr, fails the read: the short-picture rule, the place slack and
        a whole read's completeness are decided by it."""
        out, why = self._run(keys_args(self.stream))
        if why:
            raise _Failed(
                f"the listing of its keyframes said: {why}", bool(STORAGE_ERRORS.search(why))
            )
        listed: list[tuple[float | None, bool]] = []
        for line in (out or "").splitlines():
            if not line.strip():
                continue
            m = _KEYS_LINE.fullmatch(_fields(line))
            if m is None:
                said = line.strip()[:40]
                raise _Failed(
                    f"the listing of its keyframes is no listing of packets ({said!r})", False
                )
            listed.append((_float(m.group(1)), m.group(2).startswith("K")))
        n = len(listed)
        if n > KEY_SCAN_PACKETS or (
            n < KEY_SCAN_PACKETS and not whole_packets_enough(n, start, end, step)
        ):
            raise _Failed(
                f"the listing of its keyframes has {n} packets: not the first {KEY_SCAN_PACKETS}, "
                "nor every packet of its picture",
                False,
            )
        keys = [i for i, (_t, key) in enumerate(listed) if key]
        gop = max((b - a for a, b in pairwise(keys)), default=0) * step
        if not keys or listed[keys[0]][0] is None:
            return gop, None
        first, at = keys[0], listed[keys[0]][0]
        stop = keys[1] if len(keys) > 1 else n
        leading = first + sum(1 for t, _key in listed[first + 1 : stop] if t is not None and t < at)
        return gop, leading if leading <= WHOLE_LEADING_MAX else None

    def whole(self) -> _Look:
        """A short picture's whole read: every packet of the film decoded
        (whole_args), counted. A run that fails is the whole read's failure,
        not the read's - but the read's timeout or stop, which end it."""
        try:
            out, why = self._run(whole_args(self.stream))
        except _Failed as e:
            if e.timed_out or e.interrupted:
                raise
            return _Look([], [], "", {}, error=str(e), transient=e.transient)
        try:
            probed = load_json(out or "{}")
        except json.JSONDecodeError as e:
            return _Look([], [], why, {}, error=f"ffprobe's output is no JSON: {e}")
        if not isinstance(probed, dict):
            return _Look([], [], why, {}, error="ffprobe's output is no JSON object")
        streams = [s for s in probed.get("streams") or [] if isinstance(s, dict)]
        frames = [f for f in probed.get("frames") or [] if isinstance(f, dict)]
        return _Look(streams, frames, why, {})

    def first_keyframe(self, at: float | None) -> tuple[int | None, str]:
        """The index of the first keyframe among KEY_SCAN_PACKETS video
        packets from the time at (None when none is), and what the run said
        on stderr."""
        out, why = self._run(key_args(at, self.stream))
        lines = [_fields(x) for x in (out or "").splitlines() if x.strip()]
        if not all(_FLAGS.fullmatch(x) for x in lines):
            # no listing of packets' flags: no keyframe it can vouch for
            return None, _said([why, "its keyframes' listing is no listing of packets"])
        for i, line in enumerate(lines):
            if line.startswith("K"):
                return i, why
        return None, why

    def keyframe_before(self, aim: float, look: _Look, gop: float = 1.0) -> tuple[bool, str]:
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
        if not gop:
            return False, "a GOP of 0: no keyframe before counts"
        start = round(min(times), 3)
        if start > aim + 1e-3:
            return False, "it starts after its place"
        try:
            out, why = self._run(list_args(aim, self.stream))
        except _Failed as e:
            if e.timed_out or e.interrupted:
                raise
            return False, f"its packets could not be listed: {e}"
        look.why = _said([look.why, why])  # the window's, as its decode's
        packets = []
        for line in (out or "").splitlines():
            if not line.strip():
                continue
            m = _LIST_LINE.fullmatch(_fields(line))
            if m is None:  # no listing it can hold to: no proof
                return False, "its packets' listing is no listing of packets"
            t = next((v for v in (_float(m.group(1)), _float(m.group(2))) if v is not None), None)
            if t is not None:
                packets.append((t, m.group(3).startswith("K")))
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
        """The end of the last of the video packets read from the time at to
        the end of the file, or to the time until (its presentation time,
        else its decoding time, and its duration where it says one); None
        when none was read. A listing that is no listing of packets (a line
        of other than a time, a decoding time and a duration each, or N/A)
        fails the read, as anything it says on stderr does."""
        out, why = self._run(end_args(at, until, self.stream))
        if why:
            raise _Failed(f"the listing of its end said: {why}", bool(STORAGE_ERRORS.search(why)))
        ends = []
        for line in (out or "").splitlines():
            if not line.strip():
                continue
            m = _END_LINE.fullmatch(_fields(line))
            if m is None:
                raise _Failed(
                    f"the listing of its end is no listing of packets ({line.strip()[:40]!r})",
                    False,
                )
            pts, dts, dur = m.groups()
            t = _float(pts) if _float(pts) is not None else _float(dts)
            if t is not None:
                ends.append(t + (_float(dur) or 0.0))
        return round(max(ends), 6) if ends else None

    def scan(self, at: float | None, packets: int) -> bool:
        """Whether the bytes of `packets` packets from the time at hold an
        A53 payload (A53_BYTES), none decoded: True at the first packet that
        does, whatever comes after it. Read packet by packet as ffprobe
        dumps them (_Dump); _Failed for a scan that can't vouch it read them
        all whole: more than SCAN_PACKETS_MAX packets asked, anything said on
        stderr, fewer packets than asked, a packet whose bytes aren't its
        size, output that isn't the dump, past SCAN_BYTES_MAX bytes."""
        if packets > SCAN_PACKETS_MAX:
            raise _Failed(
                f"its packet scan would read {packets} packets, more than {SCAN_PACKETS_MAX}", False
            )
        dump = _Dump(packets)
        why = self._stream(scan_args(packets, at, self.stream), dump.feed)
        if dump.found:
            return True
        if why:
            raise _Failed(f"its packet scan said: {why}", bool(STORAGE_ERRORS.search(why)))
        dump.end()
        return False

    def _late(self, run_end: float) -> str:
        """Why a run was ended for its time: its own bound, or the read's."""
        if run_end <= self.end:
            return f"ffprobe did not answer within {RUN_TIMEOUT_S:g} s (a read's bound)"
        return f"ffprobe did not answer within {self.limit:g} s"

    def _start(self, args: list[str]) -> subprocess.Popen:
        """ffprobe on the file with args, niced and bounded (_LIMITED), in a
        session of its own: a read that is ended takes whatever it started."""
        cmd = [self.p.path, *args, self.path]
        nice = shutil.which("nice")
        if nice:
            cmd = [nice, "-n", str(NICE), *cmd]
        cmd = [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _LIMITED,
            str(LIMIT_AS_BYTES),
            str(LIMIT_CPU_S),
            *cmd,
        ]
        try:
            return subprocess.Popen(
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

    def _stream(self, args: list[str], feed: Callable[[str], bool]) -> str:
        """Run args, handing each line of its stdout to feed as it comes
        (feed returns True to end the run there, or raises _Failed, which
        ends it and is the run's): its stderr (folded). _Failed as _run for a
        run that exits with an error (not one feed ended), outlives the
        read's deadline or is stopped."""
        proc = self._start(args)
        run_end = time.monotonic() + RUN_TIMEOUT_S
        ended, failed, err = threading.Event(), [], []

        def out() -> None:
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    if feed(line):
                        ended.set()
                        break
            except _Failed as e:
                failed.append(e)
                ended.set()

        def errors() -> None:
            err.append(proc.stderr.read())  # type: ignore[union-attr]

        readers = [
            threading.Thread(target=out, daemon=True),
            threading.Thread(target=errors, daemon=True),
        ]
        for t in readers:
            t.start()
        try:
            while True:
                try:
                    proc.wait(
                        timeout=0.2
                        if ended.is_set()
                        else min(1.0, max(min(self.end, run_end) - time.monotonic(), 0.01))
                    )
                    break
                except subprocess.TimeoutExpired:
                    if ended.is_set():
                        break
                    if self.stop is not None and self.stop():
                        raise _Failed("the read was stopped", True, interrupted=True) from None
                    if time.monotonic() >= min(self.end, run_end):
                        raise _Failed(self._late(run_end), True, timed_out=True) from None
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning("captions.read_unkilled", pid=proc.pid)
            for t in readers:
                t.join(timeout=10)
        if failed:
            raise failed[0]
        why = " ".join("".join(err).split())[-300:]
        if not ended.is_set() and proc.returncode != 0:
            raise _Failed(
                f"ffprobe exited {proc.returncode}" + (f": {why}" if why else ""),
                bool(STORAGE_ERRORS.search(why)),
            )
        return why

    def _run(self, args: list[str]) -> tuple[str, str]:
        """ffprobe's stdout and stderr (folded) of args on the file; _Failed
        when it can't run, exits with an error (a bound it hit among them:
        killed), outlives the read's deadline or is stopped."""
        proc = self._start(args)
        run_end = time.monotonic() + RUN_TIMEOUT_S
        while True:
            try:
                out, err = proc.communicate(
                    timeout=min(1.0, max(min(self.end, run_end) - time.monotonic(), 0.01))
                )
                break
            except subprocess.TimeoutExpired:
                if self.stop is not None and self.stop():
                    _kill(proc)
                    raise _Failed("the read was stopped", True, interrupted=True) from None
                if time.monotonic() >= min(self.end, run_end):
                    _kill(proc)
                    raise _Failed(self._late(run_end), True, timed_out=True) from None
        why = " ".join((err or "").split())[-300:]
        if proc.returncode != 0:
            raise _Failed(
                f"ffprobe exited {proc.returncode}" + (f": {why}" if why else ""),
                bool(STORAGE_ERRORS.search(why)),
            )
        return out, why


def _fields(line: str) -> str:
    """A line of a csv listing without its trailing empty fields."""
    return line.strip().rstrip(",")


def _said(whys: list[str]) -> str:
    """What the runs of a window said on stderr, each once, in their order."""
    return "; ".join(dict.fromkeys(w for w in whys if w))[-300:]


class _Dump:
    """ffprobe's dump of a window's packets (scan_args: -show_entries
    packet=size,data -show_data -of default), read line by line as it comes,
    one packet held at a time: whether a packet's bytes hold an A53 payload
    (found), and _Failed for a dump that can't be the packets asked for, whole
    (a line outside a packet's section, one of none of its fields, a section
    opened inside another or left open, a packet whose bytes aren't its
    size, more packets than asked, past SCAN_BYTES_MAX bytes, fewer packets
    than asked: end)."""

    def __init__(self, asked: int) -> None:
        self.asked, self.packets, self.bytes, self.found = asked, 0, 0, False
        self.size: int | None = None
        self.data: bytearray | None = None
        self.side = False  # inside a packet's side data (a transport stream's), passed over

    def _bad(self, why: str) -> _Failed:
        return _Failed(f"its packet scan's dump {why}", False)

    def feed(self, line: str) -> bool:
        s = line.rstrip("\r\n")
        if s == "[PACKET]":
            if self.data is not None:
                raise self._bad("opens a packet inside another")
            self.data, self.size = bytearray(), None
            return False
        if self.data is None:
            raise self._bad(f"has a line outside a packet ({s[:40]!r})")
        if s == "[SIDE_DATA]":
            if self.side:
                raise self._bad("opens a packet's side data inside another")
            self.side = True
            return False
        if s == "[/SIDE_DATA]":
            if not self.side:
                raise self._bad("closes side data it didn't open")
            self.side = False
            return False
        if self.side:
            if s.startswith("[") and s.endswith("]"):
                raise self._bad(f"has a section inside a packet's side data ({s[:40]!r})")
            return False  # its side data's own lines: no part of the packet's bytes
        if s == "[/PACKET]":
            if self.size is None or self.size <= 0 or len(self.data) != self.size:
                raise self._bad(
                    f"has a packet without its bytes ({len(self.data)} bytes of its size "
                    f"{self.size})"
                )
            self.packets, self.bytes = self.packets + 1, self.bytes + self.size
            if A53_BYTES in self.data:
                self.found = True
                return True
            self.data = None
            if self.packets > self.asked:
                raise self._bad(f"has more packets than the {self.asked} asked for")
            if self.bytes > SCAN_BYTES_MAX:
                raise self._bad(f"runs past {SCAN_BYTES_MAX} bytes of packets")
            return False
        if s.startswith("size="):
            try:
                self.size = int(s[5:])
            except ValueError:
                raise self._bad(f"has a size that is no number ({s[:40]!r})") from None
            return False
        if s in ("data=", ""):  # (the writer ends a packet's dump with an empty line)
            return False
        m = _HEX_LINE.match(s)
        if not m:
            raise self._bad(f"has a line that is no part of a packet ({s[:40]!r})")
        try:
            self.data += bytes.fromhex(m.group(1))  # its spaces ignored
        except ValueError:
            raise self._bad(f"has a line of bytes that aren't hex ({s[:40]!r})") from None
        if self.size is not None and len(self.data) > self.size:
            raise self._bad(f"has a packet of more bytes than its size ({self.size})")
        if self.bytes + len(self.data) > SCAN_BYTES_MAX:
            raise self._bad(f"runs past {SCAN_BYTES_MAX} bytes of packets")
        return False

    def end(self) -> None:
        """_Failed for a dump that ended short: inside a packet, or of fewer
        packets than asked."""
        if self.found:
            return
        if self.data is not None:
            raise self._bad("ends inside a packet (cut short)")
        if self.packets < self.asked:
            raise self._bad(f"has {self.packets} of the {self.asked} packets asked for")


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
