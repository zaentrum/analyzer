# analyzer

Python content-analysis pipelines for the zaentrum platform. Consumes
`stube.catalog.item.enriched` from Kafka (group `analyzer-workers`),
runs a set of per-file detectors — tidb, chapters, silence, blackframe,
subtitles, and cross-episode audio-fingerprint (chromaprint)
intro/credits detection — writes their segments, chapters and step
statuses to the catalog API, and emits `stube.catalog.item.analyzed`
for the transcoder. It measures each title's picture area, and the
catalog's backfill of the titles that have none. It also carries a
per-item CMAF packager.

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

The edges are rounded outwards to even pixels. A measurement says it
can't tell (`reason`, and `sampling.doubt`) when its samples are more than
3 s apart anywhere in the title, from its start to its end (`gap`), when
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
30 minutes each at most (`PICTURE_PASS_DEADLINE_S`), unless the catalog
has an area of what the title plays now of this method (`GET
/api/picture/items/{id}`); while none waits, it claims the backfill's
titles (`POST /api/picture/claim`, naming its method), at the catalog's
pace and outside its quiet hours, the original while there is one, and
reports each with its claim's token. A measurement still running at the
claim's deadline, or when the worker stops, gives its title back
(`{"released": true}`). `PICTURE_AREA=false` leaves the pass's titles out,
`PICTURE_BACKFILL=false` the backfill, and `PICTURE_THREADS` (1) is the
decoder's threads.

What it costs, measured on synthetic titles on a shared workstation at
background priority, so on its efficiency cores: the CPU is an upper
bound, not a speed; the bytes are what any machine reads. The film is 20
minutes of 1080p at 24 fps, a 2.39 picture with heavy grain, H.264 at
8 Mbit/s with B-frames, shots of 2 to 14 s with a keyframe at each cut,
5.1 AAC.

| the title | sampled | samples | largest gap | read (to open) | read an hour | CPU an hour | peak memory |
|---|---|---|---|---|---|---|---|
| the film in Matroska with a 20 MiB font | keyframes, long groups decoded through | 650 | 2.9 s | 1225 MiB (21) | 3.6 GiB | 1159 s | 210 MiB |
| the same, as mkvmerge writes it | keyframes, long groups decoded through | 650 | 2.9 s | 1225 MiB (21) | 3.6 GiB | 1138 s | 210 MiB |
| the film in MP4 | keyframes, long groups decoded through | 650 | 2.9 s | 1197 MiB (1.3) | 3.5 GiB | 1112 s | 95 MiB |
| the film in MPEG-TS | in its order, long groups decoded through | 667 | 2.9 s | 1312 MiB (1.3) | 3.8 GiB | 1086 s | 109 MiB |
| its package: an encoded rung, H.264 at 5 Mbit/s, an IDR every 6 s | sync samples: can't tell | 302 | 6.0 s | 84 MiB | 251 MiB | 107 s | 87 MiB |
| 6 min of 4K HEVC 10-bit at 46 Mbit/s, a keyframe a second | keyframes | 408 | 1.0 s | 690 MiB (1.1) | 6.7 GiB | 3624 s | 286 MiB |

An original whose groups of pictures are longer than 2.9 s, a film whose
shots are, is read whole and its reference frames decoded. A disc's remux
has a keyframe a second or so and is read by its keyframes: at 4K (HEVC,
10 bits) 0.7, 0.9 and 1.4 MiB a keyframe at 31, 46 and 62 Mbit/s, each
decoded in 0.34 to 0.41 s and reduced in 0.11 s at that priority, so an
hour of such a film reads 2.4 to 5 GiB of keyframes (a sixth to a fifth of
it) in 27 to 31 min of CPU, its stretches (2 % of it) some 0.4 GiB and
5 min more; the six-minute title above spends half of its reads and CPU on
its stretches. A measurement has to end within its deadline, the pass's
(`PICTURE_PASS_DEADLINE_S`) or the claim's (the catalog's lease); its
report says what it took (`seconds`, `cpuSeconds`).

What stays unseen: what changes within a shot for less than 3 s, between
two samples and outside the stretches (a caption shown once in a bar). A
package's encoded rung can't tell, nor can an original the budget stopped,
nor a file whose duration runs on more than 3 s past its video's last
frame (its audio longer, the file cut short): the players fit their whole
frame.

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
src/analyzer/config.py            # env-driven config
src/analyzer/katalog.py           # catalog API client (claim, upsert step, fail, picture areas)
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

## Build the container

```bash
docker build -t zaentrum/analyzer .
```

Build and push the image to your own registry, then apply the `k8s/`
manifests and update the image reference for your environment.

## License

[MPL-2.0](LICENSE).
