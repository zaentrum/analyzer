# analyzer

Python content-analysis pipelines for the zaentrum platform. Consumes
`stube.catalog.item.enriched` from Kafka (group `analyzer-workers`),
runs a set of per-file detectors — tidb, chapters, silence, blackframe,
subtitles, and cross-episode audio-fingerprint (chromaprint)
intro/credits detection — writes their segments, chapters and step
statuses to the catalog API, and emits `stube.catalog.item.analyzed`
for the transcoder. It measures each title's picture area, and the
catalog's backfill of the titles that have none; it reads the chapters
of the titles that miss theirs, as the catalog's chapter backfill hands
them out. It also carries a per-item CMAF packager.

## Picture area

Many sources keep a full frame with a wider picture letterboxed inside
it, black bars baked into the video (a 1920x1080 frame whose picture is
1920x804), or a narrower one pillarboxed. The players fit the picture
area to the screen, not the frame. The analyzer measures it once and
reports it to the catalog (`PUT /api/picture/items/{id}`, an extra's
`/api/picture/extras/{id}`), in pixels with the frame it measured, and
what it measured (`measuredOn`: the original and its size, or the
package); the package stays as it is.

The area is the union of the picture over the whole title: never smaller
than the picture of a scene sampled, so a film whose aspect changes keeps
the largest, and a bar counts only where it is black throughout. Cutting
the picture is far worse than missing a crop: a measurement in doubt says
it can't tell, and the players fit the whole frame.

A title's bar black is read first: the edges of 24 frames spread over it,
the median of their outermost lines' luma, per side, when that is a black
(at most 40, its chroma neutral). Each frame is then read against it, not
against its own edge: a flat dark grey edge of an open-matte shot is no
bar where the title's bars are black.

Those 24 frames, keyframes spread evenly over the title, are its first
samples. As soon as every side's picture reaches the frame's edge in a
sample (its outermost line is picture by the whole line and by its middle,
and enough samples show a picture), no sample after can make the area
other than the whole frame, as the union only grows, and the measurement
stops there: `sampling.stoppedEarly`, `stoppedAtS`, with the samples it took
and their largest gap as they are. A 16:9 picture in a 16:9 frame, or a
frame cut to its picture, is so told by a few keyframes. An area that crops
anything is told only by all the samples below, with every rule and doubt
of them.

What is sampled (`pipelines/picture/sources.py`), from the file opened
once (PyAV), each sample labelled with its frame's own time:

- of an original with an index of its keyframes (Matroska's cues, MP4's
  sample table), every keyframe (one every half second at most: an
  intra-only file is every frame a keyframe), a seek to each and the
  keyframe decoded alone, so every shot that starts with one, as an
  encoder puts one at most cuts, holds its first frame; the groups of
  pictures longer than 2.9 s, from one keyframe to the next, decoded
  through (their reference frames), so no two samples are more than 2.9 s
  apart anywhere;
- of an original without one (MPEG-TS, MPEG-PS, a Matroska file without
  cues or cut short), the whole file in its order, each group of pictures
  held until the next keyframe says how long it is: its keyframe alone
  when it is short, decoded through when it is long;
- of a package's video rendition (its HLS playlist of fMP4 segments), its
  fragments' sync samples, no two picked more than 2.9 s apart where they
  allow it, each read by its byte range and decoded with the init
  segment's configuration, coarse to fine within a budget of 1 GiB an hour
  of film (`PICTURE_READ_MB_PER_HOUR`);
- twelve stretches spread over the title, 2 % of it together (4 to 20 s
  each), decoded through, a sample every half second: what changes within
  a shot for less than 3 s, a subtitle burned into a bar;
- a title of five minutes or less (512 MiB at most) is decoded through, a
  sample a second.

What the samples leave more than 3 s apart is then decoded again, frame by
frame: the title's tail, from its last sample to the end of the file; its
head, before its first sample; and up to four gaps between, each at most
30 s of the title from the keyframe its run starts at, as one long group of
pictures leaves (`sampling.closed`: the runs, their frames, bytes and CPU).
A run decodes the reference frames only until it nears the gap (none of the
gap's frames refers to those it skips), then every frame. A title with more
such gaps, or a longer one, is sampled far apart all through (a package's
encoded rung): it is not decoded again, and stays in doubt.

The title's picture starts at its first keyframe (`sampling.videoStartS`,
when that is not its first frame): a file copied from inside a group of
pictures has frames before it that lack their reference frames, which no
decoder makes exactly and no player shows; a frame before it that the
decoder makes all the same (from a keyframe its container does not mark) is
a sample. The picture ends at the end the file says, or at its video's last
frame (`videoEndS`) when the file runs on past that to its end, its sound
longer than its picture: its tail, read to the end of the file, says where
every stream's data ends (`dataEndS`). A file whose every stream stops more
than 3 s short of the end it says is cut short, and keeps that end.

An original's reads have no budget but twice its size; one can be set
(`PICTURE_ORIGINAL_READ_MB_PER_HOUR`).

A frame is read from its own Y, U and V planes (`planes.py`), 32 segment
means per line scaled to 8 bits, so its chroma counts (a deep blue sky of
RGB 0, 0, 100 is picture) whatever the file's colour tags say and
whatever ffmpeg is installed. The rules (`rules.py`) keep counts, not
frames: the same memory for ten samples or six thousand.

- A line is bar in a sample when its mean lies within 2 levels of the
  title's bar black of its side and no segment of it is 3 levels (its
  chroma 4) away from that; a side whose edges hold no black has no bar.
- The edge is the outermost picture line of any sample (the union): a
  dark scene or a black frame adds nothing, a wider shot, a subtitle or a
  caption in the bar widens it.
- Where enough samples show the bar's edge (3, and 3 % of those that
  show a picture), what fewer show up to 1.5 % of the side further out
  (ringing at a bright edge, a blocky frame) widens the area; further out
  it is another picture than most samples show, and that side keeps the
  full frame. A stretch's samples widen the union and support no edge.
- A logo is left out: a mark in a corner of the bar, never in its middle,
  in at least 80 % of the samples, in the same place and the same shape (a
  mark that moves between corners is in neither long enough).
- The bars of an axis are kept only together and near the same (0.6 % of
  the side, 4 lines at least): a lone bar or a lopsided pair keeps the
  axis whole.

The edges are rounded outwards to even pixels. A measurement that did not
stop early with the whole frame says it can't tell (`reason`, and
`sampling.doubt`) when its samples are more than 3 s apart anywhere in the
title, from where its picture starts to where it ends, once its gaps were
decoded frame by frame (`gap`), when
the read budget stopped it (`budget`), when a seek decoded another frame
than the keyframe it sought (`missed`), or when one landed on a frame
another sample took (`duplicates`); also with too few samples that show a
picture, a rotated video, or one with no video stream. A video that can't
be read, more than 3 % of the samples failing, or a file that changes
while it is read fails (`error`).

A measurement runs in its own process (`child.py`) at nice 10, stopped at
its deadline or when the worker stops (`measure.py`), which reads its CPU
and its peak memory when it reaps it. The picture worker (`pictures.py`)
is one thread: the event consumer hands it each movie or episode once its
`analyzed` event is sent, and goes on. The worker measures those first,
six hours each at most (`PICTURE_PASS_DEADLINE_S`, as the catalog's lease),
unless the catalog has an area of what the title plays now of this method
(`GET /api/picture/items/{id}`); while none waits, it claims the
backfill's titles (`POST /api/picture/claim`, naming its method), at the
catalog's pace and outside its quiet hours, the original while there is
one, and reports each with its claim's token. A measurement still running
at the claim's deadline, or when the worker stops, gives its title back
(`{"released": true}`), but on the last run the catalog allows a title
(`last`: its runs before had their whole lease and did not finish either)
it reports the title too costly to measure, an unknown of what a probe of
the file says it is (`sampling.doubt` `cost`), so a deletion wave does not
keep its original for ever. `PICTURE_AREA=false` leaves the pass's titles
out, `PICTURE_BACKFILL=false` the backfill, and `PICTURE_THREADS` (2) is
the decoder's threads, of frames and slices.

What it costs, measured on synthetic 20-minute titles on a shared
workstation at background priority, so on its efficiency cores, one decoder
thread: the CPU is an upper bound, not a speed; the bytes are what any
machine reads. A title whose picture fills its frame stops after the
calibration's keyframes, 24 decoded, of which the sixth that shows a
picture decides:

| the title, its picture the whole frame | samples | read | CPU | peak memory |
|---|---|---|---|---|
| 1080p H.264 at 8 Mbit/s with 5.1 AAC, Matroska | 6 | 7.2 MiB | 3.2 s | 104 MiB |
| the same in MP4 | 6 | 8.0 MiB | 3.3 s | 107 MiB |
| the same in MPEG-TS | 6 | 74 MiB | 3.4 s | 105 MiB |
| 4K HEVC 10-bit at 46 Mbit/s, a keyframe a second, Matroska | 6 | 23 MiB | 5.1 s | 349 MiB |
| the same in MP4 | 6 | 24 MiB | 12 s | 347 MiB |
| the same in MPEG-TS | 6 | 202 MiB | 15 s | 350 MiB |

A transport stream's seeks search through it, so they read more. A title
that crops anything is measured whole. The film below is 1080p at 24 fps,
a 2.39 picture with heavy grain, H.264 at 8 Mbit/s with B-frames, shots of
2 to 14 s with a keyframe at each cut, 5.1 AAC:

| the title | sampled | samples | largest gap | read (to open) | read an hour | CPU an hour | peak memory |
|---|---|---|---|---|---|---|---|
| the film in Matroska with a 20 MiB font | keyframes, long groups decoded through | 650 | 2.9 s | 1225 MiB (21) | 3.6 GiB | 1159 s | 210 MiB |
| the same, as mkvmerge writes it | keyframes, long groups decoded through | 650 | 2.9 s | 1225 MiB (21) | 3.6 GiB | 1138 s | 210 MiB |
| the film in MP4 | keyframes, long groups decoded through | 650 | 2.9 s | 1197 MiB (1.3) | 3.5 GiB | 1112 s | 95 MiB |
| the film in MPEG-TS | in its order, long groups decoded through | 667 | 2.9 s | 1312 MiB (1.3) | 3.8 GiB | 1086 s | 109 MiB |
| its package: an encoded rung, H.264 at 5 Mbit/s, an IDR every 6 s | sync samples: can't tell | 302 | 6.0 s | 84 MiB | 251 MiB | 107 s | 87 MiB |
| 6 min of 4K HEVC 10-bit at 46 Mbit/s, a keyframe a second | keyframes | 408 | 1.0 s | 690 MiB (1.1) | 6.7 GiB | 3624 s | 286 MiB |
| the film in HEVC at 3.5 Mbit/s, x265's own keyframes (at cuts, 10 s at most) | keyframes, long groups decoded through | 607 | 2.9 s | 584 MiB (0.06) | 1.7 GiB | 754 s | 122 MiB |

An original whose groups of pictures are longer than 2.9 s, a film whose
shots are, is read whole and its reference frames decoded. A disc's remux
has a keyframe a second or so and is read by its keyframes: at 4K (HEVC,
10 bits) 0.7, 0.9 and 1.4 MiB a keyframe at 31, 46 and 62 Mbit/s, each
decoded in 0.34 to 0.41 s and reduced in 0.11 s at that priority, so an
hour of such a film reads 2.4 to 5 GiB of keyframes (a sixth to a fifth of
it) in 27 to 31 min of CPU, its stretches (2 % of it) some 0.4 GiB and
5 min more; the six-minute title above spends half of its reads and CPU on
its stretches.

Two decoder threads take 0.55 to 0.79 of one's time, for no more than 3 %
more CPU (at that priority, in a quiet moment of the workstation): HEVC's
share each frame's rows, 1.6 times as fast through a 1080p group of
pictures and 1.8 times through a 4K remux's keyframes; the others' share
frames, 1.5 times as fast through a 1080p H.264 group. A measurement has
to end within its deadline, the pass's (`PICTURE_PASS_DEADLINE_S`) or the
claim's (the catalog's lease), both six hours: the whole measurement of a
3-hour 4K remux is some 6,400 s of one such core's CPU, an hour of two,
which leaves room for the analysis the worker shares its pod with at a
higher priority. A report says what it took (`seconds`, `cpuSeconds`).

Decoding the head or the tail again costs a title one group of pictures
(its reference frames up to the gap, every frame from a second before it)
and the sound's packets to the end of the file, whatever the title's
length; measured at background priority as above (the CPU an upper bound),
with the decoder's two threads, on letterboxed titles whose sound runs 7 s
past their picture:

| the title | decoded again | frames | read | CPU | of the measurement's CPU |
|---|---|---|---|---|---|
| 6 min of 1080p HEVC at 3.5 Mbit/s, x265's own keyframes, a last group of 7 s | the tail | 114 | 9.8 MiB | 2.4 s | 2.4 % |
| the same in H.264 at 8 Mbit/s, a keyframe at each cut and every 10 s | the tail | 45 | 5.4 MiB | 0.9 s | 0.7 % |
| 3 min of 4K HEVC 10-bit at 46 Mbit/s, a keyframe a second | the tail | 24 | 13 MiB | 4.9 s | 1.9 % |
| the 1080p HEVC title copied from 1 s into a group of 10 s: 8.75 s before its first keyframe | the head and the tail | 324 | 14 MiB | 2.3 s | 3.5 % |

Of a two-hour film that is a tenth of a per cent or less. Four gaps of 30 s,
the most decoded again between a title's samples, are 2,880 frames at
24 fps: some 60 s of CPU at 1080p, some 10 min of a 4K remux's.

What stays unseen: what changes within a shot for less than 3 s, between
two samples and outside the stretches (a caption shown once in a bar). A
package's encoded rung can't tell, nor can an original the budget stopped,
a file cut short (every stream stopping more than 3 s before the end it
says), or one whose samples leave more than four gaps, or one longer than
30 s from its keyframe: the players fit their whole frame.

## Chapters

A title's chapters live in the catalog, not in its package, which carries
none: the original is the only place they are read from. The analyzer
reads them from the original's header (`pipelines/chapters.py`): ffprobe
with `-nofind_stream_info`, the container's chapter marks, its format's
duration and its streams' durations, and nothing probed or decoded — a
header takes seconds even of a remux, most of it an MKV's attachments.

A mark the header gives no end (Matroska written without
`ChapterTimeEnd`, as mkvmerge writes a simple chapter file; an MP4's Nero
chapters) ends where the next later mark starts, the last where the file
does (the format's duration, else its longest stream's, else the
catalog's length of the title): what ffprobe works out itself once it
decodes. A start before 0 is 0. A mark that still has no length is left
out, and counted: a read says how many marks the header held (`found`)
and which it left out (`note`). A title keeps no control character and
at most 120 characters.

A read that fails is no file without chapters. ffprobe that exits with an
error, prints no JSON or does not answer within 2 minutes
(`CHAPTERS_READ_TIMEOUT_S`, `120`), a file that is not there, and ffprobe
that says an error and finds no chapter (ffprobe 4.4 exits 0 with `Read
error at pos. …` and no chapter when the storage fails it on the chapters'
element) fail the read: the `chapter` step fails (the catalog retries it),
the catalog is told (`PUT /api/chapters/items/{id}` with `"read": {"error",
"transient"}`) and the title's chapters stay as they are. A failure of the
storage or of the worker (no file, no answer in time, a read stopped, an
error such as `Input/output error`, `Read error`, `Stale file handle`) is
`transient`: it says nothing of the file, and the catalog counts it as no
failure of the title; one of the file itself (`Invalid data found`, `moov
atom not found`) counts. A read that ran writes the chapters, the whole set
in place of the title's, with what the read was (`"read": {"path",
"sourceId", "fileSize", "found", "note", "probe", "seconds"}`), before the
step says it is done — `skipped` for a header that holds no chapter with a
length — so a step that says so means the catalog holds what the header
holds.

A header that holds no chapter is not taken on its word: a file cut short
before its chapters' element reads as one without chapters, and nothing in
ffprobe's answer says so. The catalog (its migration 055) never lets a read
that found none delete chapters, holds it against the original's source
record (the library's probe of the file, which names its chapters), and
writes the record's chapters where the record holds some; it settles a
title as having none only where the record agrees.

The pass of a new title reads the chapters first among its detectors; a
pass that finds the `chapter` step finished leaves the chapters alone.
The catalog's chapter backfill reads the titles that miss theirs (an
original of the title's that no read settled): it settles those it can
from their source records itself, and hands out the rest, whose records
can't say. A thread of its own (`chapter_backfill.py`, its own client)
asks for them (`POST /api/chapters/claim`, a title at a time), reads each
original's header and reports the read with the claim's token and the
original it names (`sourceId`). Which titles, how many an hour and when are
the catalog's to say (`chapters.backfill*`, off by default, never from 17:00
to 22:00 UTC by default): while it hands out nothing the thread waits as
long as the catalog says (30 s to an hour; 10 minutes for a catalog without
the backfill, or one that fails the claim), a second after a claim whose
titles the catalog settled from their records, and a second at least
between two claims. A claim's answer is waited for 60 s, well past the
catalog's own bound of a claim (20 s, whatever its storage does); a claim
not answered by then is no sign that the catalog can't be asked, only that
it is slow, and is asked again after 30 s. It starts a read
only up to 10 s before its claim's deadline, reads no longer than the
deadline, and gives the title back unread when the deadline cuts a read
short or the analyzer stops (`{"released": true}`), which is no failure of
the title. `CHAPTER_BACKFILL=false` asks for none.

## Intro and recap bounds

An intro is 10 s to 3 min long and a recap 10 s to 5 min, neither more
than a quarter of the runtime: the fuser drops a candidate outside that
rather than trim it, and never merges candidates into a longer one. A
chapter titled as an intro or a recap marks where it starts, so one
that runs longer (a file with coarse chapters, whose next mark is the
end credits) ends where another detector's candidate near that start
ends, or gives no segment.

## Redelivered and retried events

Each detector is a step: `tidb`, `chapter`, `subtitle`, `blackframe`,
`silence`, and `chromaprint` for a series episode with siblings. The
offset is committed only after the item is processed and `analyzed` is
sent, so a crash reprocesses the event. The work runs on the Kafka poll
thread, and the consumer's `max.poll.interval.ms` is 24 h (librdkafka's
maximum), so the broker never hands an item in progress to another
replica; a rebalance (a replica joining or leaving) waits until every
busy replica has finished its item.

- An event for an item whose steps are all terminal (`done`, `skipped`,
  `not_applicable`, `failed`; `chromaprint` only for an item that has the
  step) runs nothing and only passes the chain on.
- Otherwise a pass runs the steps that aren't `done` or `not_applicable`.
- The catalog retries failed or silent steps by sending `enriched` again,
  marked `"status": "retry"`. A retry runs only the steps that aren't
  finished (`done`, `skipped`, `not_applicable`), then passes the chain
  on. One whose steps have all finished since — a long run the catalog's
  reaper took for dead that reported done after all — is acked with one
  log line (`consumer.retry_already_finished`) and nothing else.
- The catalog replaces an item's whole segment set. A pass that skipped
  `done` steps leaves it alone when its own steps found nothing, and
  otherwise runs the `done` steps' detectors again, their steps
  untouched, so that it writes the whole set. If one of them can't find
  its segments again (an error, or tidb answering an outage with
  nothing), the set is left as it is.
- The `chapter` step writes the chapters before it says it is done or
  skipped, and fails when the header can't be read (see
  [Chapters](#chapters)). A pass that skips a finished `chapter` step
  writes no chapters, so a header it could not read now never empties
  the catalog's. A `done` one gives its share of the segments of the
  header read now, else of the chapters the catalog keeps (`GET
  /api/chapters/items/{id}`), else none: the segments the other steps
  found are written all the same.
- What the `chapter` step found alone (a retry of it) leaves the segments
  as they are: it is no reason to run the `done` detectors again, each a
  decode of the whole file.
- A `chromaprint` step waiting for a run that the pass can't make (no
  sibling episodes now, or the catalog can't list them) is answered
  `skipped` or `failed` rather than left waiting.
- While a step runs, the analyzer reports it `in_progress` again every
  10 minutes (`STEP_HEARTBEAT_SECONDS`, `600`), so the catalog's reaper —
  which takes a step silent for longer than its timeout, 2 h for the
  analyzer's, for a dead run and retries it — never reclaims a live one.
  A step shorter than that reports only its start and its end.

## Layout

```
src/analyzer/main.py              # entry point (uvicorn /healthz + /readyz side-car to worker loop)
src/analyzer/worker.py            # claim/process loop
src/analyzer/pictures.py          # the picture worker: the analyze pass's titles, then the backfill's
src/analyzer/chapter_backfill.py  # the chapter backfill: the chapters of the titles that miss theirs
src/analyzer/config.py            # env-driven config
src/analyzer/katalog.py           # catalog API client (claim, upsert step, fail, picture areas, chapters)
src/analyzer/packager.py          # per-item CMAF packager (shaka-packager, HEVC passthrough)
src/analyzer/pipelines/           # chapters, silence, blackframe, subtitles, chromaprint, tidb, fuser
src/analyzer/pipelines/picture/   # the picture area: sources, planes, rules, its own process
k8s/                              # Deployment, Service, ServiceAccount, ServiceMonitor, GrafanaDashboard
Dockerfile
```

## Local development

```bash
uv sync
uv run pytest
```

The picture-area tests hold the rules on frames drawn as planes
(`tests/test_picture.py`), and the cases a title has to get right on small
videos ffmpeg makes (`tests/test_picture_cases.py`: short wider shots, a
night sky, grey bars, 10-bit and HDR10, a deep blue open matte, a logo,
subtitles in a bar between keyframes, credits, fades, a windowbox, a
trailer, a still, a length past the catalog's, an anamorphic DVD,
interlaced MPEG-TS, a rotated video, fonts in Matroska, packages); those
are skipped without an ffmpeg that encodes H.264, and take about a minute.

The chapter tests read small files ffmpeg makes with chapter marks
(Matroska and MP4, and one mkvmerge writes where it is installed) with the
ffprobe on the PATH, and hold the rules of the marks, a header that can't
be read, what ffprobe's errors say of a read and a read that answers too
late (`tests/test_chapters.py`); the pass's chapter step and the backfill
run against a fake catalog (`tests/test_redelivery.py`,
`tests/test_chapter_backfill.py`). Run them with the ffprobe the image
carries too (Ubuntu 22.04's 4.4): its header reads differ from newer ones'.

## Build the container

```bash
docker build -t zaentrum/analyzer .
```

Build and push the image to your own registry, then apply the `k8s/`
manifests and update the image reference for your environment.

## License

[MPL-2.0](LICENSE).
