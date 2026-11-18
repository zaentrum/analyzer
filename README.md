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
them out, and whether the originals carry closed captions, as the
catalog's captions backfill hands them out. It also
carries a per-item CMAF packager.

## Which ffprobe is called for what

The image carries two ffprobes, and each is called for its own work only:

| ffprobe | where | called for |
|---|---|---|
| the image's own, Ubuntu 22.04's 4.4.2 (with its ffmpeg) | on the PATH, `/usr/bin/ffprobe` | the chapters, silence, black frames, keyframes, subtitles, chromaprint's durations, the packager: everything but the captions |
| the pipeline's 7.1.5, BtbN's static build of the 7.1 branch (the archive the transcoder and chino-stream pin), only its `ffprobe` | `/opt/ffmpeg-7.1/bin/ffprobe`, named by `CAPTIONS_FFPROBE`, not on the PATH | the closed captions, and nothing else |

The picture area calls neither: it decodes with PyAV's own FFmpeg
libraries. The image's build checks the 7.1.5 archive's sha256 and fails
unless `/opt/ffmpeg-7.1/bin/ffprobe` says 7.1 and the PATH's is still the
image's 4.4 at `/usr/bin`. The captions backfill refuses to start with any
ffprobe but one of the 7.1 release, the image's 4.4.2 above all, and never
looks on the PATH (see [Closed captions](#closed-captions)).

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

## Closed captions

Closed captions (CEA-608/708) ride in a video stream itself, as broadcast
and disc sources carry them: an A53 cc_data payload (ATSC A/53 Part 4) in
its pictures' SEI or user data. A QuickTime or MP4 file can hold them in a
track of their own (`c608`, `c708`) as well. Nothing else in a container
says so, and a probe that decodes no frame can't tell, so the source
records say none of every original. An original that carries them, or
whose captions are not known, is not retired; the catalog never opens a
media file itself, so the analyzer reads them (`pipelines/captions.py`) as
the catalog's captions backfill hands the originals out, and reports what
it found. Only "none" lets an original go, so only "none" has to be true: a
read that can't vouch for its windows fails, a file with anything the read
doesn't know how to see is not read (the closed lists below), and a failed
read is not known, which keeps the original.

A read first lists the file's container and every stream, nothing decoded
(the header probe), and holds them to the closed lists: a file with
anything off them is reported with what (`offList`) and not read further.
A track of captions of its own is counted (`captionStreams`), and the
original has captions. Attached pictures are passed over; the one video
stream left is the film, and every later probe of the read selects it
alone.

The read then decodes 48 video packets, about two seconds of a film, in
each of its windows, and nothing else of the file, with the pipeline's
ffprobe 7.1 at nice 10 and one decoding thread: a frame whose SEI or user
data carries captions has the side data `ATSC A53 Part 4 Closed Captions`,
and 7.1 gives the stream's `closed_captions=1` once it decoded one. 7.1's
HEVC decoder exports them only from a prefix SEI: an HEVC window none of
whose frames shows captions has the bytes of its packets read as well, none
decoded, for the A53 payload (`GA94`, then type 3), where a suffix SEI holds
it - and since those are the window's packets alone, never the whole film,
a read of an HEVC film that shows no captions is never "none": its windows
go to the catalog as read, with `captionsKnown: false` and `reason: "hevc:
needs the packets' signature"`, and the catalog holds the original until the
record's `captionSignature` releases it; it says captions where a window
found them, as found. The
windows follow the caption-window plan, the same on every side that reads
captions: the picture's start, read from its first keyframe, and 10, 30,
50, 70 and 90 % of the picture's presentation span as its records state it
- its start the picture's `start_time`, else the format's (where neither
states one, there is no plan, and the read fails); its end that start plus
the picture's `duration`, else its source record's `container.durationMs`
(the claim's `durationMs`, a number), else the format's duration (no tag is
read for one); start + at x (end -
start) - the start window never capped, each other at most 30 s short of the
end, none capped at half the picture: six windows, never merged, each saying
its plan fraction (`at`: 0, then 0.1 ... 0.9) where the tail caps where it
lies - a 2-minute picture's 90 % window lies at 90 s. Where those windows
can't lie apart - two of its places
closer than a window reaches (its 48 packets at the picture's frame step,
from a keyframe up to a group of pictures before its place, and 0.5 s), or
on one another; a picture under about two minutes, with groups of pictures
of a few seconds - the picture is read whole instead, every packet decoded
(the short-picture rule, decided over the places themselves - start +
at x (end - start), at most 30 s short of the end - in the same arithmetic as
the library's record logic, so a picture that starts late is short or not
alike on both sides; the groups of pictures from a listing of its first
1200 packets, none decoded). That whole read stands for the windows, and
"none" needs it complete (the plan's `wholeComplete`): every packet decoded
but the stream's own leading pictures - the packets before its first
keyframe in decode order and the first group's packets timed before that
keyframe, counted from the same listing of its first packets, at most 16
(the plan's `wholeLeadingMax`; more, or a listing that shows no keyframe or
no time of it, and it is not complete) - no more frames than packets, every
frame placed in time, packets enough for its span at its frame step, from
within a frame step and 0.5 s of the picture's start to within them of its
end (the plan's end tolerance), and nothing said on stderr by the whole run;
it runs under the same
bounds as a window, and a bound it hits fails it. The verdict rests on the
runs whose output feeds it, each checked for what it must be, and an error
printed by any of them means not known. Of a short picture that is the
whole run: its start window, which only measured where its span starts and
is replaced by the whole read, feeds no verdict - what it printed, or why it
failed, is not what "none" rests on, as the whole run decodes the same first
packets, and anything it prints of them fails it. It goes on the wire as
one window at 0 with `whole: true`, its `packets` and its `leading` ones. The read reports the span
it measured (`span`: `{"start", "end"}`, the first frame of its start and the
end of its last video packet) and its group of pictures (`gop`: the longest
run between two keyframes among its first 1200 packets, in seconds at its
frame step, 0 where they hold fewer than two keyframes; one longer than
those 1200 frame steps fails the read) beside the length its file gives
(`duration`), which the catalog holds to its record; each end of the span
must lie within a frame step and 0.5 s of the span its records state (the
plan's end tolerance). The
frame step is the one the catalog knows without reading: one over the header
probe's `avg_frame_rate` of the film, or its `r_frame_rate` only where the
average is exactly the text `0/0`, each `num/den` in digits alone (no space,
sign or point), both above 0, neither of more than 4300 digits (what Python
3.11 reads of a text, and so the catalog: the image's 3.11.0~rc1 reads more,
so the read holds to it itself; a count of a cover's frames too). An
`avg_frame_rate` that is missing or of
another form gives no plan, with no fallback to `r_frame_rate`, and so does a
rate below one frame a second (the plan's `minRate`; `0/1` among them): the
read fails, as the catalog can't know a step a read would measure. A picture whose records state it starts late, or ends
before its file (an MP4's track does), is sampled where it is; one whose
records give the file's length (a Matroska film states none of its own)
and that ends early fails the read (below). Each window is placed at its
share of the span from its start, and read from the keyframe nearest its
point: the one before it where the container's index says so (Matroska,
MP4), the first after it where it doesn't (a transport stream). A window
that decodes fewer than 24 frames (half its 48 packets, and the least the
catalog takes of a window) - a file cut from a stream starts inside a
group of pictures, and a seek can land inside one - looks for the keyframe
among the 1200 video packets from the same point, none decoded, and
decodes the packets up to it and 48 after it, counted from that point.
Every window is read and reported on its own, captions found or not.

A window also counts its frames that the decoder flags as interlaced
(`interlaced_frame`) and those that repeat a field (`repeat_pict`, soft
pulldown). The flag sees pictures *coded* as fields (field pictures, H.264's
PAFF and MBAFF, MPEG-2 frames flagged interlaced), not pictures woven into
frames; only an examination of the pictures (ffmpeg's `idet`) sees those,
and this read makes none. A reader must not take the flag for the
examination: a count above 0 shows fields, 0 shows nothing of fields woven
into frames (1080i HEVC woven into frames decodes with the flag 0 on every
frame).

A read fails as a whole (`{"error", "ffprobe"}` on the wire) when:

| | |
|---|---|
| the closed lists | its container, a stream, its video codec or its picture is off them, or it has more than one video stream that is no attached picture: `off the closed lists: codec:vc1` |
| the file | isn't there, ffprobe can't open it, finds no video stream in it |
| its frames | have no presentation times: a raw elementary stream is never read (its container, `h264` or `hevc`, is off the list besides) |
| its length | the file gives none to place the windows by |
| its span and its last packet | its records state no start of its picture (neither its `start_time` nor the format's), no length (neither its `duration`, nor the claim's `durationMs`, nor the format's duration), or its header probe no frame rate (no `avg_frame_rate` of `num/den` in digits, nor an `r_frame_rate` of that form where the average is `0/0`): no plan; or the end of the last video packet (its presentation time, else its decoding time: AVI gives none, and its duration where it says one) listed from where the span its records state ends, none decoded, to a frame step, 0.5 s and 5 s past it (no count of packets bounds it: a last group of pictures is listed whole), ends more than a frame step and 0.5 s from there (a header that says a tenth of the film, a picture that ends before its file where its records give the file's length) |
| the plan's own limits | a stated frame rate below one frame a second (no plan; `r_frame_rate` stands in only where `avg_frame_rate` is exactly `0/0`); a group of pictures longer than 1200 frame steps (longer than the packets it was listed from; the read says its `gop` all the same); a short picture's whole read whose packets × its frame step fall short of its span less a frame step and 0.5 s, or one with a frame it can't place in time |
| a number that isn't finite | NaN or Infinity, in an answer of ffprobe or of the catalog: refused, never taken for a number; none is ever sent |
| its span and its first frame | the first frame its start decoded lies more than a frame step and 0.5 s from where its records state its picture starts (a file cut inside a group of pictures, whose first pictures don't decode; a start time that isn't its picture's) |
| its length and its bytes | its bytes, at the bitrate of the frames its start decoded (their bytes over the time they span), last less than a tenth of it (a clock that started again, read as one that wrapped: 26 hours of a five-minute capture) |
| what ffprobe says | its header probe, its keyframes' listing or its end's listing says anything on stderr |
| what a listing lists | its keyframes' listing (a time and the flags of each packet) or its end's (a time, a decoding time and a duration) is no listing of packets: empty, a line of anything else, or - the keyframes' - more packets than the 1200 asked for, or fewer where they don't reach the picture's end at its frame step; a verdict rests only on runs whose output was checked for what it must be |
| its claim | names another count of packets a window than the plan's 48: a read it can't serve, reported as failed (the catalog hands out 48) |
| the file | is another after the read than before it: its inode, its time or its size changed (it was replaced or written to while it was read) |
| the time | an ffprobe run does not answer within 180 s (every run, the plan's bound: a window's, a listing's, the end's, a short picture's whole read), or the read as a whole within 30 minutes (`CAPTIONS_READ_TIMEOUT_S`, `1800`), or the read is stopped |

Where a window lands is held to where it was placed, not to a distance
alone. The packets from its seek up to its place are listed, none decoded:
a window whose keyframe (its first decoded frame's, or one up to 1 s after
it, which an open group of pictures' leading pictures come before) is the
last one before its place, the listing reaching there, landed where a
seek lands in a group of pictures that long (`keyframeBefore`). A window may
lie max(10 s, the group of pictures the first 1200 packets show and 0.5 s)
from its place either way, and up to 30 s before it with `keyframeBefore`; a
keyframe between it and its place is one its seek passed over. Of a group of
pictures of 0 (fewer than two keyframes among those packets) a window counts
only at its own place, within a frame step and 0.5 s, and never with
`keyframeBefore`.

A window fails (`frames` 0; `start`, `end`, `captions`, `interlacedFrames`,
`repeatFrames` and `keyframeBefore` null; and its `error`) when ffprobe
exits with an error (a bound it hit among them, below) or prints no JSON,
when no keyframe is among its packets, when it decoded no frame, or fewer
than 24 from the keyframe on, or a frame it can't place in time (no
presentation time), when its first frame lies more than that after where it
was placed, more than that before it without `keyframeBefore`, or more than
30 s before it (it landed elsewhere), when its
frames' times are negative, when it doesn't start after the window before
it, when it decodes frames another window decoded, when its packet scan
can't vouch for its packets (below), or when it showed no captions while
ffprobe said anything on stderr in any of its runs - its decode, the
keyframe's listing, the listing from its seek to its place, its packet scan
(a storage's error, `Read error` or `Input/output error`, is transient: it
says nothing of the file). The windows after it are read all the same.

The packet scan reads ffprobe's dump of the window's packets (their `size`
and their bytes, the default writer's sections; a packet's side data, as a
transport stream's packets carry, is passed over) as it comes, one packet
held at a time, and fails the window on any line on stderr, on fewer
packets than it asked for, on a packet whose bytes aren't its size (a
packet without its data among them), on a section left open (a dump cut
short) and on any line that is no part of the dump; it reads 1000 packets
at most (`SCAN_PACKETS_MAX`; a window that would read more fails) and
128 MiB of their bytes (`SCAN_BYTES_MAX`; one past it fails).

Every ffprobe a read runs is bounded: one decoding thread (`-threads 1`,
stated in every run's arguments, the header probe's and the listings' too:
ffprobe opens decoders to probe the streams even where it decodes no frame),
1,280 MiB of address space (`RLIMIT_AS`), 600 s of CPU (`RLIMIT_CPU`) and
180 s of wall time (each run: a window's, a listing's, a short picture's
whole read alike; the plan's bounds, the same on every side),
set by a few lines of the analyzer's own Python before ffprobe starts (a
system that can't lower an address space, macOS, runs without that bound).
A run that hits one is killed, or refused memory and fails, and its window,
or the read, fails as any run that exits with an error does. The worker's
own memory doesn't grow with a window: it holds one packet's dump at a time.
Measured, its peak resident memory rose 4 MiB above the 27 MiB it held
before, for a scan of 48 packets of 4K HEVC 10-bit at 41 Mbit/s and for one
run on to its 128 MiB bound alike. A short picture's whole read holds the
list of its frames ffprobe gives (not measured here). A short video, whose
windows would lie close together or on one another, is read whole (above).

The backfill (`captions_backfill.py`, a thread and a client of its own)
asks for the originals to read (`POST /api/captions/claim`, one at a time:
`{"sourceId", "itemId", "path", "packets", "token", "durationMs"}`, the
last its source record's length; `packets` must be the plan's 48, else the
read fails as one it can't serve), reads each and
reports it with the token (`PUT /api/captions/sources/{sourceId}`:
`{"token", "original": {"captionWindows", "duration", "span", "gop",
"sizeBytes", "ffprobe", "captionStreams", "offList"}}`, and `captionsKnown: false` with its
`reason` where the windows can't tell captions at all: HEVC that showed none). `captionWindows` holds every
window the read placed, `[{"at", "start", "end", "frames", "captions",
"interlacedFrames", "repeatFrames", "keyframeBefore"}]`: `at` the plan's own fraction (0
its start, then 0.1 ... 0.9, a tail-capped window too), `start` and `end` the presentation seconds of
its first and last decoded frame on the file's own timeline (of the frames
decoded, not of the point aimed at), `frames` what it decoded, `captions`
whether one of them showed captions, `interlacedFrames` and `repeatFrames`
how many of them the decoder flagged as coded in fields and as repeating a
field, `keyframeBefore` whether the packets from its seek to its place show
its keyframe the last before its place, none between (false for one that
starts after it, and of a GOP of 0);
`duration` the length its file gives (the catalog holds it to its record's;
it places no window); `span` `{"start", "end"}`, the span the read
measured; `gop` its group of pictures (above), by which the catalog holds
where each window may lie (max(10 s, `gop` + 0.5 s) from its place; of a
`gop` of 0, or none said, a frame step and 0.5 s) and which picture is read
whole: a read that says none, of windows a group of pictures off their
places, is no read the catalog takes; `sizeBytes` the
size of the file as the read found it (the catalog takes a read of the
file it handed out alone); `captionStreams` how many tracks of captions of
their own the file has; `offList` what of it is off the closed lists (`[]`
for none). A read that fails sends `{"error", "ffprobe"}`, with `sizeBytes`
when it found the file, `captionStreams` and `offList` when the header
probe listed it, and `gop` when it listed the first packets' keyframes. A
failed window's `error` is the analyzer's own word of why; the catalog's
window doesn't read it. The catalog holds the length a read says to its own
record's (within 2 s or 1 % of it, the more): the read has no record to
hold it to, and a read of another length is the catalog's to refuse. Whether the original has captions is the catalog's to
work out of the windows and the tracks. Each window that failed is logged
(`captions.backfill_window_failed`: the original, the window, why).
`ffprobe` is the version line as `ffprobe -version` gives it (`ffprobe
version n7.1.5-12-g1fdbca85aa-20260731` in the image), or, for a build
named by its commit, `7.1 by its libraries (ffprobe version 1fdbca8:
libavutil 59.39.100, …)`: the catalog takes reads of 7.1 alone. A read
whose ffprobe can't be named is no result: the original is given back. An
original the catalog read already may come again, and is read again.
Its pace is the catalog's, as the chapter backfill's: while it hands out
nothing the thread waits as long as the catalog says (30 s to an hour; 10
minutes for a catalog without the backfill, or one that fails the claim), a
second at least between two claims; a claim's answer is waited for 60 s. A
read is started only up to 10 s before its claim's deadline and runs no
longer than it; one the deadline cuts short, or the analyzer's stop, gives
the original back (`{"released": true}`). A report the catalog can't take
(400) is never sent again as it is: it is reported at once, with the same
claim's token, as a failed read in its place (`{"error": "the catalog
refused its report (400): …", "ffprobe", "sizeBytes"}`), and the original
is read again when the catalog hands it out again. A read the catalog failed
to keep (500) is kept, with the
file's inode, time and size: when the catalog hands the same original out
again, and the file is still that file, the kept read is sent again and the
file not read again (256 such reads at most, the oldest let go first). A
read that names no span - its start window placed no frame, so it measured
none - says none at all: the catalog takes it, as a read that says nothing
of its picture. A
read that breaks (an exception of the worker's own) is reported as a failed
read, never left without a report. A claim item that can't be read (no
`sourceId` or `path`) is logged by its keys alone, never its token or its
path. The paths and the JSON are all in `katalog.py` (`claim_captions`,
`put_captions`).

The plan's values and rules are constants of the code; nothing reads a plan
at run time. Its reference, the library schemas' `library/v2/caption-windows.json`,
is vendored as `tests/caption-windows.json`, pinned by its sha256, and
`tests/test_captions_plan.py` holds the read to it: every key of it known,
the numbers the read's own, and each of its words (`minRate`, the frame
step's form, the cover rule, the `gop` reported and its bound, a whole
read's completeness, no NaN or Infinity taken for a number) tested. What
the backfill sends is held to `tests/captions-reports.json`
(`tests/test_captions_wire.py`): for each kind of read - windows that show
none and captions, a long group of pictures, a picture that starts late, a
short picture read whole, HEVC not known and HEVC with captions, a failed
window, reads that failed as a whole - the body of its PUT as sent, the
stored probe, record length and file size the catalog holds it to, and what
the analyzer means by it (`none`, `captions`, `unscanned` or `not known`).
A test of the catalog's can take those bodies through its own handler.
`CAPTIONS_REPORTS_WRITE=1` writes the fixture again.

### The closed lists

The read knows how to see captions and fields only where the tables below
say: every way a file carries them that the read has an answer for, how the
read sees each, and those it does not see. What it does not see is not
known, and the original is kept: the read keeps to three lists in
`pipelines/captions.py`, and a file with anything off them is not read.

How closed captions are carried:

| carried as | in | how the read sees it | ffprobe |
|---|---|---|---|
| A53 cc_data in a prefix SEI (ITU-T T.35, `GA94`) | H.264, HEVC | the decoded frame's side data `ATSC A53 Part 4 Closed Captions` | 7.1 |
| A53 cc_data in a suffix SEI | HEVC, in Matroska and in a transport stream | the bytes of the window's packets, for `GA94` and type 3, where no frame shows captions (7.1's decoder exports none) | 7.1 |
| A53 cc_data, SCTE-20 or DVD line-21 data in the picture user data | MPEG-2, MPEG-1 | the decoded frame's side data: 7.1's decoder gives all three as A53 | 7.1 |
| A53 cc_data in ITU-T T.35 metadata | AV1 | the decoded frame's side data, through libdav1d (the image's build decodes AV1 with it; the tests' clip is read where the ffprobe has it, as CI's) | 7.1 |
| a track of captions of their own, `c608` or `c708` | QuickTime, MP4 | the header probe: codec `eia_608`, or the tag; counted in `captionStreams` | 7.1 |
| a track of captions in Matroska | Matroska | not seen: ffmpeg maps no codec to it; off the list (`stream:subtitle:none`), not known and kept | - |
| a private data stream | MPEG transport and program streams | not seen: off the list (`stream:data:<codec>`), not known and kept | - |
| user data of other codecs | VC-1, MPEG-4 Part 2, and every other codec | not seen: off the codec list (`codec:vc1`), not known and kept | - |
| line 21 in the picture, an analog capture that kept its blanking lines | pictures of 486, 512 or 608 lines | not seen: off the list (`picture:vbi-lines`), not known and kept | - |
| teletext, DVB subtitles | transport streams | subtitle streams, on the list: the catalog's subtitle rule holds them; they are no closed captions | 7.1 |
| a caption file beside the video (SCC, MCC, …) | - | not seen: not in the file, and this read says nothing of it | - |
| anything else | any container, codec or stream | off the lists, not known and kept | - |

How a stream is carried in fields:

| in fields as | in | how the read sees it | ffprobe |
|---|---|---|---|
| field pictures (PAFF), MBAFF frames, the picture timing SEI's `pic_struct` of fields | H.264 | the decoded frame's `interlaced_frame`, counted in `interlacedFrames` | 7.1 |
| frames flagged interlaced (`progressive_frame` 0), field pictures | MPEG-2 | `interlaced_frame`, counted in `interlacedFrames` | 7.1 |
| field pictures (the picture timing SEI's `pic_struct` of a field) | HEVC | `interlaced_frame`, counted in `interlacedFrames` | 7.1 |
| a repeated field (`repeat_first_field`; H.264's `pic_struct` of three fields), soft pulldown | MPEG-2, H.264 | the decoded frame's `repeat_pict`, counted in `repeatFrames` | 7.1 |
| frame doubling or tripling (`pic_struct` 7, 8) | HEVC | not seen: 7.1's decoder gives no `repeat_pict` | - |
| woven pictures with no flag: fields woven into frames coded as progressive, hard telecine | any codec | not seen: only an examination of the pictures (`idet`) sees them | - |
| a container's mark (Matroska's FlagInterlaced and FieldOrder, MP4's `fiel`) | Matroska, MP4 | not read here: the header's `field_order`, which the catalog has from its own probe | - |
| fields in part of the video (progressive first, the field order changing) | any codec | seen only where a window falls | 7.1 |
| any codec off the list (VC-1, MPEG-4 Part 2, DV, ProRes, …) | - | not read: off the codec list, not known and kept | - |

The lists, by ffprobe's names:

| list | on it |
|---|---|
| containers (`CONTAINERS`, the format's `format_name`) | `matroska,webm`, `mov,mp4,m4a,3gp,3g2,mj2`, `mpegts`, `mpeg`, `avi` |
| video codecs (`VIDEO_CODECS`, the film's `codec_name`) | `h264`, `hevc`, `mpeg1video`, `mpeg2video`, `av1`; of them `hevc`'s packets are read for the A53 payload as well (`SCANNED_CODECS`) |
| streams of captions (any kind) | codec `eia_608` (`CAPTION_CODECS`), tag `c608` or `c708` (`CAPTION_TAGS`): counted in `captionStreams` |
| video streams | the film, one; and attached pictures, passed over (below) |
| audio, attachment streams | any |
| subtitle streams (`SUBTITLE_CODECS`) | `subrip`, `srt`, `ass`, `ssa`, `webvtt`, `mov_text`, `text`, `microdvd`, `hdmv_pgs_subtitle`, `dvd_subtitle`, `dvb_subtitle`, `dvb_teletext` |
| data streams | codec `scte_35`, `timed_id3`, `epg` (`DATA_CODECS`); tag `tmcd` (`DATA_TAGS`: a QuickTime timecode track, of no codec ffprobe names); an MP4's chapter track, by the rule below (`chapter_track`) |

A data stream is the file's chapter track, and on the list, only where all of
these hold in the stored probe: the container is the MP4 family
(mov,mp4,m4a,3gp,3g2,mj2); the stream is codec_type data, codec_name bin_data,
codec_tag_string text; the file has chapters; the stream's sample count
(nb_frames) equals the number of chapters; and it is the file's only such
stream. A probe that lacks the tag or the count leaves it off the list. The
tag is compared exactly (`TEXT` or `Text` is off the list), and the count
is taken only as ffprobe writes it, a number in digits and nothing around
them (`" 3 "`, `"+3"`, a JSON number are off it). Here
the stored probe is the read's own header probe, which lists the chapters
(`chapter=id`). Any other `bin_data` stream stays off it (`stream:data:bin_data`).
The track is safe to pass: a package's chapters don't come from it but from the
catalog's marks, read from the original's record or header by the chapter
backfill, handed to the packager by the build (its `chapters` and
`chaptersFrom`), and kept by the version.

An attached picture is a video stream that the container marks as an
attached file - by Matroska's `filename` and `mimetype` tags of an
attachment (both, never one alone), or by the `attached_pic` disposition
(ffmpeg's mark of Matroska's image attachments and of an MP4 cover, `covr`;
its mark is the number 1 or `true`, nothing else: `1.0`, `"1"` and `"0"`
are no mark) - that is an
image and has no length of its own: its codec is one of the cover codecs,
`mjpeg`, `png` or `bmp`, and its `mimetype`, where it has one (an empty one
too), `image/*`
(an MP4 cover has none: the type of its data comes as its codec; a stream
of any other codec is a second film, whatever its marks say), and the
container counts no more than one frame of it by every count it gives
(`nb_frames`, and every tag that starts `NUMBER_OF_FRAMES`, as
`NUMBER_OF_FRAMES-eng`), each in digits alone - a count that is there but
can't be read - `1e3`, ` 900`, `+900`, `0x384`, `1.0`, a list - makes it no
attached picture - and gives it no tag that starts `DURATION`. Tags are
read without case; two that differ by it alone and disagree (`mimetype` and
`MIMETYPE`) make it no image. This is the plan's `words.picture`, rule for
rule as the library's record logic reads a stored probe, so the film a read
takes is the one the catalog places its plan on. ffprobe gives a
stream the file's duration where the container gives none, so the stream's
`duration` is no mark. Its file name never decides: `poster.jpg` and
`cover_land.png` are attached pictures by their marks, and a JPEG as a
video stream of its own, unmarked, is a video stream like any other.

`offList` names each find in the order the header probe gives them: the
container, each stream, the count of video streams, each film's codec, its
picture:

| name | when |
|---|---|
| `container:<format_name>` | the container is off the list (`container:flv`; `container:h264`, a raw stream) |
| `stream:<codec_type>:<codec>` | a stream is off the list; its `codec_name`, else (none, or `unknown`) its tag, else `none` (`stream:data:bin_data`, `stream:subtitle:c609`, `stream:subtitle:none`) |
| `video-streams:<n>` | more than one video stream that is no attached picture (`video-streams:2`) |
| `codec:<codec_name>` | a video stream's codec is off the list (`codec:vc1`, `codec:mjpeg`) |
| `picture:vbi-lines` | a video stream of 486, 512 or 608 lines |

What a read of an original costs, measured on synthetic films on a shared
workstation, ffprobe at nice 10 with one decoding thread and every bound
set, from a local disk with the files in its cache, so the time is ffprobe's
CPU: an upper bound of the decoding, not of a storage's answer. A window reads some 2 s of the video
(2 MB at 8 Mbit/s, 10 MB at 40 Mbit/s) and the container's index, and
lists its packets from its seek to its place, none decoded (at most a group
of pictures of them: some 0.05 s a window at 1080p); the header probe reads
the file's header; the length's check at the end lists the last group of
pictures' packets, none decoded:

| the original | the start alone | the whole read: its end and 6 windows |
|---|---|---|
| 10 min of 1080p H.264 at 8 Mbit/s, x264's own keyframes, Matroska | 0.13 s | 0.85 s |
| 5 min of 4K HEVC 10-bit at 41 Mbit/s, a keyframe a second, Matroska, no captions | 2.1 s | 15 s, its windows' packets scanned too (not known: HEVC is never "none") |
| 110 s of that 1080p H.264, a short picture | 0.13 s | 3.0 s, read whole: its 2,638 packets decoded |

An HEVC window that shows no captions has its packets' bytes read for a
suffix SEI: at 41 Mbit/s ffprobe's dump of them takes some 0.7 s, most of
what HEVC without captions costs above H.264's rate of decoding. HEVC whose
frames show captions is not read so. A transport stream whose seeks land
inside long groups of pictures decodes up to the next keyframe in each
window, and costs more: up to some 6 s at 1080p with groups of 10 s, and an
HEVC one has those packets' bytes read too. Every read reads all its
windows, captions or not, well within the 180 s each of its runs may take.

The backfill refuses to start, logging `captions.backfill_refused` with
why, unless the ffprobe `CAPTIONS_FFPROBE` names (`/opt/ffmpeg-7.1/bin/ffprobe`
when unset) says it is of the 7.1 release: by its version line
(`ffprobe version n7.1.5-…`), or, for a build named by its commit, by its
libraries (libavutil 59.39, libavcodec 61.19, libavformat 61.7). The
image's 4.4.2 on the PATH is never used, and an ffprobe of 7.1 on the PATH
is no stand-in for one missing where `CAPTIONS_FFPROBE` points.
`CAPTIONS_BACKFILL=false` asks for none.

### Known

What the plan does not read: nothing after its 90 % window, and nothing in
the last 30 s of the picture (an end title's captions); and captions only
between the windows, of which each reads some 2 s (a film captioned only in
a scene no window falls on reads as none). A short picture, whose windows
can't lie apart, is read whole instead, so none of this holds for it; the
whole read costs a decode of the whole picture, and one that runs out of
its 180 s is not known.

Two limits fail closed: the read fails, and the original is kept.

- A group of pictures longer than 30 s, where a window's place falls more
  than 30 s after its keyframe, fails that window; a source keeps its
  groups of pictures far shorter.
- A picture whose groups of pictures are longer than its first 1200
  packets showed can have windows that lie on one another all the same:
  such a read fails, and is kept.

Two limits of what the field counts show, which a reader of them must keep
apart from an examination of the pictures:

- Fields woven into frames coded as progressive, and hard telecine, show
  no flag: `interlacedFrames` 0 says nothing of them.
- HEVC's frame doubling and tripling give no `repeat_pict` in 7.1:
  `repeatFrames` is 0 for them.

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
src/analyzer/captions_backfill.py # the captions backfill: the closed captions of the originals
src/analyzer/config.py            # env-driven config
src/analyzer/katalog.py           # catalog API client (claim, upsert step, fail, picture areas, chapters, captions)
src/analyzer/packager.py          # per-item CMAF packager (shaka-packager, HEVC passthrough)
src/analyzer/pipelines/           # chapters, captions, silence, blackframe, subtitles, chromaprint, tidb, fuser
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

The captions tests hold the read against a made-up ffprobe (a script that
says it is the release it is told and answers as a video it is told of
would: its length, clock, frame rate and groups of pictures, where its seeks
land, where it carries captions; `tests/captions_fake.py`): captions
present, absent, found in a later window, every guard (a window that lands
elsewhere, one on another's frames, negative times, a raw stream, no length,
a length its last packet or its bytes belie), a window inside a group of
pictures, one that decodes fewer than half its packets, windows on the
keyframe before their place up to 30 s back and one whose seek passed a
keyframe over, a last group of pictures listed whole, errors, a timeout,
a binary replaced in place, a 4.4.2 that must be refused; every entry of
the closed lists and one off them on each axis, attached pictures and
images no container marks, tracks of captions, frames in fields and
repeated ones, HEVC's packets read for a suffix SEI
(`tests/test_captions.py`); and the backfill and its client
against a fake catalog and on the wire, every body held to the catalog's
checks (`tests/test_captions_backfill.py`). `tests/test_captions_real.py`
reads originals ffmpeg makes - five minutes of H.264 with an A53 caption SEI
before every picture in Matroska, MP4 and MPEG-TS, the same without,
captions from 100 s on, HEVC libx265 encoded with `-a53cc 1` and without,
the same HEVC with its A53 SEI moved after the picture (a suffix SEI), a
clip cut inside a group of pictures; an original for each entry of the
closed lists a clip can be made of (QuickTime with a timecode track, AVI,
MPEG-2 with A53 user data in a program and a transport stream, MPEG-1,
subtitles and a font attached, a `c608` track, an MP4 cover, a poster and a
landscape cover attached to Matroska, H.264 and HEVC coded in fields and
MPEG-2 in fields, AV1 where the ffmpeg encodes it) and one off them on each
axis (FLV, MPEG-4 Part 2, a data stream, two films, a picture with its
blanking lines, a JPEG as a video stream); groups of pictures of 15, 25 and
40 s, and one of 10 s whose index holds every fourth keyframe alone - and
the reads that must fail: a
transport stream whose clock starts again, a header that says a tenth of
the length, a raw stream, a short clip, a file that isn't a video, one
without video. It runs when `CAPTIONS_FFPROBE` names an ffprobe 7.1, with
an ffmpeg 7.1 that has libx264 and libx265 beside it (or
`CAPTIONS_TEST_FFMPEG`), and is skipped otherwise; CI fetches the image's
7.1.5 archive for it. `CAPTIONS_NOKEY` may name a file of your own that
starts before its first keyframe, to read too.

## Build the container

```bash
docker build -t zaentrum/analyzer .
```

Build and push the image to your own registry, then apply the `k8s/`
manifests and update the image reference for your environment.

## License

[MPL-2.0](LICENSE).
