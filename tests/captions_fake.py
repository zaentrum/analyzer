"""A made-up ffprobe for the captions tests: a script that says it is the
release its version text names, and answers a read as a video it is told of
would (the fixture fake_ffprobe, conftest.py).

The video (Video, written beside the script as video.json): its length and
start time, its frame rate and group of pictures, where a seek lands (on the
keyframe before its point, as Matroska's index has it; on the one after it,
as a transport stream reads on; or at the start whatever it seeks, as on a
clock that started again), which spans of it carry captions, whether its
frames have times, and the length its file says it has. A run's own answer
(answers(..., call=N), or of every run: answers(...)) replaces what the video
would answer: an error, no JSON, a timeout."""

from __future__ import annotations

import json
import stat
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from analyzer.pipelines import captions

LIBS_71 = """libavutil      59. 39.100 / 59. 39.100
libavcodec     61. 19.101 / 61. 19.101
libavformat    61.  7.103 / 61.  7.103
libavdevice    61.  3.100 / 61.  3.100
libavfilter    10.  5.100 / 10.  5.100
libswscale      8.  3.100 /  8.  3.100
libswresample   5.  3.100 /  5.  3.100
libpostproc    58.  3.100 / 58.  3.100
"""
LIBS_44 = """libavutil      56. 70.100 / 56. 70.100
libavcodec     58.134.100 / 58.134.100
libavformat    58. 76.100 / 58. 76.100
libavdevice    58. 13.100 / 58. 13.100
libavfilter     7.110.100 /  7.110.100
libswscale      5.  9.100 /  5.  9.100
libswresample   3.  9.100 /  3.  9.100
libpostproc    55.  9.100 / 55.  9.100
"""
# The pipeline's: BtbN's static build of n7.1.5, as the image carries it.
FFPROBE_715 = (
    "ffprobe version n7.1.5-12-g1fdbca85aa-20260731 Copyright (c) 2007-2026 the FFmpeg developers\n"
    "built with gcc\nconfiguration: --enable-gpl\n" + LIBS_71
)
# The image's own: Ubuntu 22.04's.
FFPROBE_442 = (
    "ffprobe version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2007-2021 the FFmpeg developers\n"
    "built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)\nconfiguration: --prefix=/usr\n" + LIBS_44
)


@dataclass
class Video:
    """What the made-up ffprobe answers of: a video of `length` seconds from
    `start_time`, `fps` frames a second, a keyframe every `gop` seconds from
    `first_key` on (a file cut inside a group of pictures starts before its
    first), its seeks landing `lands` ("before" its point, "after" it,
    "start": wherever they aim, or "skip": on the keyframe before the one
    before it), the spans [from, to) that carry captions,
    whether its frames have times (`pts`), the length its file says
    (`says`, else its own), `packet_bytes` a frame."""

    length: float = 600.0
    start_time: float = 0.0
    fps: float = 25.0
    gop: float = 2.0
    first_key: float = 0.0
    lands: str = "before"
    captions: list[list[float]] = field(default_factory=list)
    pts: bool = True
    says: float | None = None
    packet_bytes: int = 20_000
    # what the header probe lists: the container and every stream
    format_name: str = "matroska,webm"
    streams: list[dict] = field(default_factory=lambda: [
        {"index": 0, "codec_type": "video", "codec_name": "h264", "height": 1080}])
    # spans whose frames are in fields, whose frames repeat a field, and
    # whose packets' bytes hold an A53 payload the decoder doesn't export
    interlaced: list[list[float]] = field(default_factory=list)
    repeats: list[list[float]] = field(default_factory=list)
    bytes_captions: list[list[float]] = field(default_factory=list)
    # the file's chapters, as the header probe lists them (-show_entries chapter=id)
    chapters: int = 0
    # what it says on stderr, by the kind of run ("header", "decode", "keys",
    # "list", "end", "scan"; "keys-at": a keyframe scan from a window's point
    # alone), answering all the same, exit 0
    said: dict[str, str] = field(default_factory=dict)
    # spans whose decoded frames have no time
    untimed: list[list[float]] = field(default_factory=list)
    # how its packet scan's dump goes wrong: "empty" (no packet), "cut" (cut
    # inside a packet), "nodata" (packets without their bytes), "short" (a
    # packet fewer than asked), "junk" (a line that is no part of it); "side"
    # gives each packet its side data, as a transport stream's has, and
    # "sideopen" leaves that section open
    dump: str = ""
    # the kind of run during which the file read is changed (a byte added)
    touch: str = ""
    # whether the header probe's format states its start_time
    format_start: bool = True
    # how long each run takes before it answers (seconds)
    delay: float = 0.0


SCRIPT = '''#!{python} -I
import json, os, sys
d = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
if args[:1] == ["-version"]:
    sys.stdout.write(open(os.path.join(d, "version.txt")).read())
    sys.exit(0)
calls = os.path.join(d, "calls")
n = int(open(calls).read()) + 1 if os.path.exists(calls) else 1
open(calls, "w").write(str(n))
open(os.path.join(d, f"args-{{n}}"), "w").write("\\n".join(args) + "\\n")
open(os.path.join(d, "nice"), "w").write(str(os.nice(0)))
import resource
limits = {{"as": resource.getrlimit(resource.RLIMIT_AS),
          "cpu": resource.getrlimit(resource.RLIMIT_CPU)}}
open(os.path.join(d, "limits"), "w").write(json.dumps(limits))
own = next((p for p in (os.path.join(d, f"{{n}}.json"), os.path.join(d, "every.json"))
            if os.path.exists(p)), None)
if own:
    a = json.load(open(own))
    if a.get("sleep"):
        import time; time.sleep(a["sleep"])
    sys.stderr.write(a.get("stderr", ""))
    sys.stdout.write(a.get("stdout", ""))
    sys.stdout.flush()
    if a.get("signal"):
        import signal; os.kill(os.getpid(), a["signal"])
    sys.exit(a.get("rc", 0))
v = json.load(open(os.path.join(d, "video.json")))
if v.get("delay"):
    import time; time.sleep(v["delay"])
entries = args[args.index("-show_entries") + 1] if "-show_entries" in args else ""
kind = ("header" if entries.startswith("format=") else "scan" if "-show_data" in args else
        "list" if entries == "packet=pts_time,dts_time,flags" else
        "keys" if entries in ("packet=flags", "packet=pts_time,flags") else
        "end" if entries.startswith("packet=pts_time") else "decode")
sys.stderr.write(v.get("said", {{}}).get(kind, ""))
if kind == "keys" and not args[args.index("-read_intervals") + 1].startswith("%"):
    sys.stderr.write(v.get("said", {{}}).get("keys-at", ""))  # a window's own keyframe scan
if v.get("touch") == kind:
    with open(args[-1], "ab") as f:
        f.write(b"x")
if kind == "header":  # the header probe
    fmt = {{"format_name": v["format_name"]}}
    if v["format_start"]:
        fmt["start_time"] = f"{{v['start_time']:.6f}}"
    says = v["says"] if v["says"] is not None else v["length"]
    if says:
        fmt["duration"] = f"{{says:.6f}}"
    # the film states its own start and duration (as an MP4's track does),
    # unless the test lists them itself, or says it has none ("duration": null)
    streams = []
    for st in v["streams"]:
        st = dict(st)
        if st.get("codec_type") == "video":  # (its duration is no mark of an attached picture)
            st.setdefault("start_time", f"{{v['start_time']:.6f}}")
            # its frame rate, as ffprobe states it, unless the test states its own
            from fractions import Fraction
            rate = Fraction(v["fps"]).limit_denominator(1001)
            st.setdefault("avg_frame_rate", f"{{rate.numerator}}/{{rate.denominator}}")
            st.setdefault("r_frame_rate", f"{{rate.numerator}}/{{rate.denominator}}")
            st.setdefault("duration", f"{{says:.6f}}" if says else None)
            for k in ("start_time", "duration"):  # a field the test says it hasn't (None)
                if st[k] is None:
                    del st[k]
        streams.append(st)
    head = {{"format": fmt, "streams": streams}}
    if v.get("chapters"):
        head["chapters"] = [{{"id": i}} for i in range(v["chapters"])]
    json.dump(head, sys.stdout)
    sys.exit(0)
iv = args[args.index("-read_intervals") + 1] if "-read_intervals" in args else "%"  # a whole read
# from a point: so many packets ("+#48"), to a time ("615.000"), or to the end
point, rest = iv.split("%", 1)
count, bound = 10**9, None
if rest.startswith("+#"):
    count = int(rest[2:])
elif rest:
    bound = float(rest)
first, step, gop = v["start_time"], 1.0 / v["fps"], v["gop"]
end = v["start_time"] + v["length"]
keys = [k for k in (v["start_time"] + v["first_key"] + i * gop
                    for i in range(int((v["length"] - v["first_key"]) / gop) + 1)) if k < end]
listing = entries == "packet=pts_time,dts_time,flags"
ends = entries.startswith("packet=pts_time") and not listing and entries != "packet=pts_time,flags"
def land(t):
    # the end's packets are read where the file ends, whatever the clock
    if t is None or (v["lands"] == "start" and not ends):
        return first
    t = min(max(t, first), end)
    if v["lands"] == "after" and not ends:
        return t
    before = [k for k in keys if k <= t] or [first]
    if v["lands"] == "skip" and not ends and len(before) > 1:
        return before[-2]  # a seek that passes the keyframe before its point over
    return before[-1]
at = land(float(point) if point else None)
times = [at + i * step for i in range(int(round((end - at) * v["fps"])))]
times = [t for t in times if bound is None or t < bound][:count]
if ends:  # each packet's time, and its duration where the listing asks for it
    dur = f",{{step:.6f}}" if "duration_time" in entries else ""
    print("\\n".join(f"{{t:.6f}},{{t:.6f}}{{dur}}" for t in times))
    sys.exit(0)
iskey = [any(abs(t - k) < step / 2 for k in keys) for t in times]
if listing:  # the packets' times and flags, none decoded
    flag = {{True: "K__", False: "___"}}
    print("\\n".join(f"{{t:.6f}},{{t:.6f}},{{flag[k]}}" for t, k in zip(times, iskey)))
    sys.exit(0)
def within(spans, t):
    return any(a <= t < b for a, b in spans)
def dump(b):  # a packet as ffprobe's default writer dumps it (-show_data)
    out = ["[PACKET]\\n", f"size={{len(b)}}\\n"]
    if v["dump"] in ("side", "sideopen"):  # a transport stream's packet side data
        out += ["[SIDE_DATA]\\n", "side_data_type=MPEGTS Stream ID\\n", "id=224\\n"]
        out += [] if v["dump"] == "sideopen" else ["[/SIDE_DATA]\\n"]
    out.append("data=\\n")
    for o in range(0, len(b), 16) if v["dump"] != "nodata" else ():
        row = b[o:o + 16]
        hx = "".join(f"{{x:02x}}" + (" " if i & 1 else "") for i, x in enumerate(row))
        text = "".join(chr(x) if 32 <= x < 127 else "." for x in row)
        out.append(f"{{o:08x}}: " + hx.ljust(41) + text + "\\n")
    return "".join(out) + "\\n[/PACKET]\\n"  # an empty line ends the dump, as ffprobe's does
if "-show_data" in args:  # the packets' bytes
    # a suffix SEI whose A53 payload crosses a line of the dump; a slice
    blob = dump(bytes.fromhex("00000001 500104 40 b50031 000000 4741393403 c1fffc8080"))
    zero = dump(bytes.fromhex("00000001 2601af") + bytes(20))
    text = "".join(blob if within(v["bytes_captions"], t) else zero for t in times)
    if v["dump"] == "empty":
        text = ""
    elif v["dump"] == "cut":
        text = text[:text.rindex("[/PACKET]")]
    elif v["dump"] == "short":
        text = text[:text.rindex("[PACKET]")]
    elif v["dump"] == "junk":
        text = "packets:\\n" + text
    sys.stdout.write(text)
    sys.exit(0)
if entries == "packet=pts_time,flags":  # the keyframes' listing: times and flags
    print("\\n".join(f"{{t:.6f}},{{'K__' if k else '___'}}" for t, k in zip(times, iskey)))
    sys.exit(0)
if "packet=flags" in " ".join(args):
    print("\\n".join("K__" if k else "___" for k in iskey))
    sys.exit(0)
out, seen = [], False
for t, k in zip(times, iskey):
    seen = seen or k
    if not seen:
        continue  # nothing decodes before a keyframe
    f = {{"pkt_size": str(v["packet_bytes"]), "interlaced_frame": int(within(v["interlaced"], t)),
         "repeat_pict": int(within(v["repeats"], t))}}
    if v["pts"] and not within(v["untimed"], t):
        f["pts_time"] = f"{{t:.6f}}"
    if any(a <= t < b for a, b in v["captions"]):
        f["side_data_list"] = [{{"side_data_type": "ATSC A53 Part 4 Closed Captions"}}]
    out.append(f)
fmt = {{"start_time": f"{{v['start_time']:.6f}}"}}
says = v["says"] if v["says"] is not None else v["length"]
if says:
    fmt["duration"] = f"{{says:.6f}}"
stream = {{"closed_captions": 0, "avg_frame_rate": f"{{int(v['fps'])}}/1"}}
if "-count_packets" in args:
    stream["nb_read_packets"] = str(len(times))
json.dump({{"frames": out, "streams": [stream], "format": fmt}}, sys.stdout)
'''


class FakeProbe:
    """A made-up ffprobe in a folder of its own: path is the script."""

    def __init__(self, folder: Path, version: str, video: Video | None = None) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.dir = folder
        self.path = folder / "ffprobe"
        self.path.write_text(SCRIPT.format(python=sys.executable))
        self.path.chmod(self.path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        (folder / "version.txt").write_text(version)
        self.video(video or Video())

    def video(self, video: Video) -> FakeProbe:
        """The video it answers of."""
        (self.dir / "video.json").write_text(json.dumps(asdict(video)))
        return self

    def answers(
        self,
        out: dict[str, Any] | str = "",
        *,
        err: str = "",
        rc: int = 0,
        sleep: float | None = None,
        call: int | None = None,
        signal: int | None = None,
    ) -> FakeProbe:
        """What it answers a run (the call-th, else every run) in place of
        the video: out on stdout (JSON of a dict), err on stderr, rc its exit
        code, after sleep seconds; killed by signal, where one is named."""
        name = f"{call}.json" if call else "every.json"
        body = {"stdout": out if isinstance(out, str) else json.dumps(out), "stderr": err,
                "rc": rc, "sleep": sleep, "signal": signal}  # fmt: skip
        (self.dir / name).write_text(json.dumps(body))
        return self

    @property
    def runs(self) -> list[list[str]]:
        """The arguments of every run but -version, in their order."""
        out, n = [], 1
        while (self.dir / f"args-{n}").exists():
            out.append((self.dir / f"args-{n}").read_text().splitlines())
            n += 1
        return out

    @property
    def niceness(self) -> int | None:
        try:
            return int((self.dir / "nice").read_text().strip())
        except (OSError, ValueError):
            return None


def frames(n: int, *, with_captions: bool, start: float = 0.0, fps: float = 25.0) -> list[dict]:
    """n decoded frames as ffprobe 7.1 shows them, from start at fps, each
    with the A53 side data or none."""
    sd = [{"side_data_type": captions.SIDE_DATA}] if with_captions else []
    return [{"pts_time": f"{start + i / fps:.6f}", "pkt_size": "20000",
             **({"side_data_list": sd} if sd else {})} for i in range(n)]  # fmt: skip

