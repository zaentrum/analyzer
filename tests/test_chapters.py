"""A file's chapters, read from its header as the catalog keeps them: from
small files ffmpeg (and mkvmerge, where it is installed) makes here with
chapter marks, read with the ffprobe on the PATH; and the rules a read
follows of ffprobe's marks, a header that holds none, one that can't be
read, one that answers too late. The tests that read files are skipped
without ffmpeg."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

from analyzer.pipelines import chapters

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="no ffmpeg"
)

# four marks: an intro, an act, a mark of no length, the end credits
METADATA = """;FFMETADATA1
[CHAPTER]
TIMEBASE=1/1000
START=0
END=10000
title=Intro
[CHAPTER]
TIMEBASE=1/1000
START=10000
END=20000
title=Act 1
[CHAPTER]
TIMEBASE=1/1000
START=15000
END=15000
title=Marker
[CHAPTER]
TIMEBASE=1/1000
START=20000
END=30000
title=End Credits
"""


def ff(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True
    )


def video(path: Path, metadata: str | None = None, seconds: int = 30) -> Path:
    """A small video of `seconds`, with the chapters of an ffmetadata file."""
    inputs = ["-f", "lavfi", "-i", f"color=c=black:s=64x36:r=5:d={seconds}"]
    maps = ["-map", "0"]
    if metadata is not None:
        meta = path.with_suffix(".txt")
        meta.write_text(metadata)
        inputs += ["-i", str(meta)]
        maps += ["-map_chapters", "1"]
    ff(*inputs, *maps, "-c:v", "mpeg4", str(path))
    return path


@needs_ffmpeg
def test_the_marks_with_a_length_are_the_chapters(tmp_path: Path) -> None:
    # Matroska keeps the mark of no length as it is: it is left out
    path = video(tmp_path / "film.mkv", METADATA)
    r = chapters.read(str(path))
    assert r.ok and r.error is None and r.found == 4
    assert [(c["startMs"], c["endMs"], c["title"], c["ordinal"]) for c in r.chapters] == [
        (0, 10_000, "Intro", 1),
        (10_000, 20_000, "Act 1", 2),
        (20_000, 30_000, "End Credits", 3),
    ]
    assert r.note == "1 of 4 chapter marks have no length, left out"
    assert r.file_size == path.stat().st_size and r.probe and r.probe.startswith("ffprobe version")
    report = r.report()
    assert report["found"] == 4 and report["fileSize"] == r.file_size and "error" not in report
    assert report["note"] == r.note


@needs_ffmpeg
def test_the_chapters_of_an_mp4(tmp_path: Path) -> None:
    # an MP4's chapter track runs each mark to the next one's start
    r = chapters.read(str(video(tmp_path / "film.mp4", METADATA)))
    assert r.ok and r.found == 4 and r.note is None
    assert [(c["startMs"], c["endMs"], c["title"]) for c in r.chapters] == [
        (0, 10_000, "Intro"),
        (10_000, 15_000, "Act 1"),
        (15_000, 20_000, "Marker"),
        (20_000, 30_000, "End Credits"),
    ]


@needs_ffmpeg
def test_the_pass_takes_an_intro_and_the_end_credits_as_segments(tmp_path: Path) -> None:
    res = chapters.detect(str(video(tmp_path / "film.mkv", METADATA)))
    assert res.ok and len(res.chapters) == 3
    assert [(s["kind"], s["startMs"], s["endMs"], s["source"]) for s in res.segments] == [
        ("intro", 0, 10_000, "chapter"),
        ("credits", 20_000, 30_000, "chapter"),
    ]
    assert res.report("ignored")["found"] == 4


@needs_ffmpeg
def test_a_header_without_chapters_was_read(tmp_path: Path) -> None:
    r = chapters.read(str(video(tmp_path / "bare.mkv")))
    assert r.ok and r.found == 0 and r.chapters == [] and r.note is None
    assert r.report()["found"] == 0


@pytest.mark.skipif(
    shutil.which("mkvmerge") is None or shutil.which("ffmpeg") is None, reason="no mkvmerge"
)
def test_marks_without_an_end_end_where_the_next_starts_and_the_last_where_the_file_does(
    tmp_path: Path,
) -> None:
    # mkvmerge writes a simple chapter file's marks with their starts alone;
    # ffprobe gives them no end unless it decodes
    plain = video(tmp_path / "plain.mkv", seconds=12)
    marks = tmp_path / "marks.txt"
    marks.write_text(
        "CHAPTER01=00:00:00.000\nCHAPTER01NAME=Opening\n"
        "CHAPTER02=00:00:04.000\nCHAPTER02NAME=Act 1\n"
        "CHAPTER03=00:00:08.000\nCHAPTER03NAME=End Credits\n"
    )
    path = tmp_path / "merged.mkv"
    subprocess.run(
        ["mkvmerge", "-q", "-o", str(path), "--chapters", str(marks), str(plain)], check=True
    )
    r = chapters.read(str(path))
    assert r.ok and r.found == 3
    ends = [c["endMs"] for c in r.chapters]
    assert ends[:2] == [4_000, 8_000]
    assert 12_000 <= ends[2] <= 12_500  # the header's duration of the file


def test_a_file_that_is_not_there_was_not_read(tmp_path: Path) -> None:
    r = chapters.read(str(tmp_path / "gone.mkv"))
    assert not r.ok and r.error and r.error.startswith("no file to read")
    assert r.report()["error"] == r.error and "found" not in r.report()


@needs_ffmpeg
def test_a_file_ffprobe_cant_read_was_not_read(tmp_path: Path) -> None:
    path = tmp_path / "broken.mkv"
    path.write_bytes(b"\x1a\x45\xdf\xa3 not a header")
    r = chapters.read(str(path))
    assert not r.ok and r.error and r.error.startswith("ffprobe exited ")
    res = chapters.detect(str(path))
    assert not res.ok and res.chapters == [] and res.segments == []
    assert res.error and res.error.startswith("ffprobe exited ")


def fake_ffprobe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    """An ffprobe on the PATH that runs body (its version answered at once)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "ffprobe"
    exe.write_text(
        '#!/bin/sh\nif [ "$1" = "-version" ]; then echo "ffprobe version fake"; exit 0; fi\n'
        + body
        + "\n"
    )
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    chapters.probe_version.cache_clear()


def test_a_read_that_answers_too_late_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_ffprobe(tmp_path, monkeypatch, "sleep 30")
    path = tmp_path / "film.mkv"
    path.write_bytes(b"a film")
    t0 = time.monotonic()
    r = chapters.read(str(path), timeout=1)
    assert time.monotonic() - t0 < 10
    assert not r.ok and r.timed_out and not r.interrupted
    assert r.error == "ffprobe did not answer within 1 s" and r.probe == "ffprobe version fake"
    chapters.probe_version.cache_clear()


def test_a_read_that_is_stopped_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ffprobe(tmp_path, monkeypatch, "sleep 30")
    path = tmp_path / "film.mkv"
    path.write_bytes(b"a film")
    until = time.monotonic() + 0.5
    r = chapters.read(str(path), timeout=60, stop=lambda: time.monotonic() > until)
    assert not r.ok and r.interrupted and not r.timed_out and r.error == "the read was stopped"
    chapters.probe_version.cache_clear()


def test_the_read_reads_the_header_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # what ffprobe is asked: the chapters, the format's and the streams'
    # durations, no stream probed (nothing decoded)
    fake_ffprobe(
        tmp_path, monkeypatch, 'echo "$@" > "$(dirname "$0")/args"; echo "{\\"chapters\\": []}"'
    )
    path = tmp_path / "film.mkv"
    path.write_bytes(b"a film")
    assert chapters.read(str(path)).ok
    args = (tmp_path / "bin" / "args").read_text().split()
    assert "-nofind_stream_info" in args and "-show_chapters" in args
    assert args[args.index("-show_entries") + 1] == "format=duration,start_time:stream=duration"
    assert not {"-show_frames", "-show_packets", "-count_frames", "-count_packets"} & set(args)
    chapters.probe_version.cache_clear()


# -------------------------------------------------- the rules of the marks
def test_a_mark_without_an_end_ends_where_the_next_later_one_starts() -> None:
    raw = [{"start_time": "0.000000"}, {"start_time": "300.5"}, {"start_time": "1200"}]
    got = chapters.marks(raw, 1_500_000)
    assert [(c["startMs"], c["endMs"]) for c in got] == [
        (0, 300_500),
        (300_500, 1_200_000),
        (1_200_000, 1_500_000),
    ]
    # without the file's end the last has no length
    assert [c["endMs"] for c in chapters.marks(raw, None)] == [300_500, 1_200_000]


def test_marks_out_of_order_end_at_the_next_later_start() -> None:
    raw = [{"start_time": "600"}, {"start_time": "0"}, {"start_time": "300"}]
    got = chapters.marks(raw, 900_000)
    assert [(c["startMs"], c["endMs"], c["ordinal"]) for c in got] == [
        (600_000, 900_000, 1),
        (0, 300_000, 2),
        (300_000, 600_000, 3),
    ]


def test_a_mark_is_kept_as_the_catalog_takes_it() -> None:
    long = "x" * 130
    raw = [
        {"start_time": "-0.5", "end_time": "10", "tags": {"title": "Cold\x00 Open\n"}},
        {"start_time": "10", "end_time": "10", "tags": {"title": "no length"}},
        {"start_time": "20", "end_time": "15"},
        {"start_time": "N/A", "end_time": "30"},
        {"start_time": "30", "end_time": "40", "tags": {"title": long}},
        {"start_time": "40", "end_time": "50", "tags": {"title": "   "}},
        "not a mark",
    ]
    got = chapters.marks(raw, None)
    assert [(c["startMs"], c["endMs"], c["title"]) for c in got] == [
        (0, 10_000, "Cold Open"),
        (30_000, 40_000, "x" * 120),
        (40_000, 50_000, None),
    ]


def test_where_the_file_ends() -> None:
    assert chapters._end_of({"format": {"duration": "100.0", "start_time": "0.5"}}, None) == 100_500
    assert (
        chapters._end_of(
            {"format": {}, "streams": [{"duration": "90"}, {"duration": "95.5"}]}, None
        )
        == 95_500
    )
    assert chapters._end_of({"format": {}, "streams": [{}]}, 80_000) == 80_000
    assert chapters._end_of({}, None) is None


def test_a_result_made_by_hand_was_read() -> None:
    res = chapters.ChapterResult(segments=[], chapters=[{"startMs": 0, "endMs": 1, "title": None}])
    assert res.ok and res.error is None and res.report("/m.mkv") == {"path": "/m.mkv", "found": 1}


def test_the_read_timeout() -> None:
    assert chapters.read_timeout() == chapters.READ_TIMEOUT_S


@pytest.mark.parametrize(
    "value,want", [("30", 30.0), ("1", 5.0), ("99999", 3600.0), ("soon", 120.0)]
)
def test_the_read_timeout_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, value: str, want: float
) -> None:
    monkeypatch.setenv("CHAPTERS_READ_TIMEOUT_S", value)
    assert chapters.read_timeout() == want
