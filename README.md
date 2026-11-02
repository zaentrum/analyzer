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
`/api/picture/extras/{id}`), in pixels with the frame it measured; the
package stays as it is.

The area is the union of the picture over the whole title: never smaller
than the picture of a scene sampled, so a film whose aspect changes keeps
the largest, and a bar counts only where it is black throughout. Cutting
the picture is far worse than missing a crop, so a side whose edge is in
doubt keeps the full frame. `pipelines/picture.py` samples a frame every
10 s of the title (24 to 720 of them), each decoded on its own: of an
original, the first frame ffmpeg can show after seeking there; of a
package's video rendition (its HLS playlist), a segment's first frame,
its init segment and the segment cut right after that frame, so a sample
reads one frame. ffmpeg reduces a frame to 32 segment means per row and
per column, and per side:

- a row (a column) is picture when half its segments are 10 levels above
  black, black being the frame's outermost lines; the edge is the
  outermost picture row of any sample, so dark scenes and fades never
  shrink the area;
- the edge is kept only where at least 3 samples, and 3 % of those with
  a picture, reach it; else that side keeps the full frame;
- anything 8 levels above black on two adjacent lines of the bar, beyond
  a guard band at the edge, in any sample, a black one too, keeps that
  side whole: subtitles or captions burned into the bar, a dim picture,
  specks; noise in a bar averages out in a segment;
- a logo is left out: a mark in the bar in at least half the samples,
  spanning at most a quarter of the side;
- what lies in the guard band (ringing at a bright edge) widens the area.

The edges are rounded outwards to even pixels. Too few samples with a
picture, or a rotated video, can't be told (`reason`); a video that
can't be read fails (`error`). The analyze pass measures a movie or an
episode's original once its `analyzed` event is sent, unless the catalog
has an area of what the title plays now (`GET /api/picture/items/{id}`);
it is no step and never fails the item. A retry measures nothing.

The backfill (`backfill.py`) is a thread that asks the catalog for the
titles that have no area (`POST /api/picture/claim`), at the catalog's
pace and outside its quiet hours, measures what it is handed (the
package where there is one, else the original), and reports each. A
measurement still running at the claim's deadline gives its title back
(`{"released": true}`), and the catalog hands it out again later.

`PICTURE_AREA=false` turns off the analyze pass's measurement,
`PICTURE_BACKFILL=false` the backfill, `PICTURE_WORKERS` (default `2`)
is how many samples are decoded at once, and `PICTURE_INTERVAL_S`
(default `10`, 1 to 120) how far apart they are. Each report says what it
cost (`seconds`, `cpuSeconds`, `durationMs` of the video). Measured on a
shared workstation at background priority (an upper bound, not a speed),
with a 10-bit HEVC source and grain at 11 Mbit/s: about 0.23 s of CPU a
sample of a package, 0.48 s of an original (it decodes the two frames
after the keyframe, which the decoder needs to give it out), so about
80 s and 175 s of CPU an hour of film; a package's sample reads about a
quarter of a megabyte.

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
src/analyzer/backfill.py          # the picture-area backfill: the titles the catalog hands out
src/analyzer/config.py            # env-driven config
src/analyzer/katalog.py           # catalog API client (claim, upsert step, fail, picture areas)
src/analyzer/packager.py          # per-item CMAF packager (shaka-packager, HEVC passthrough)
src/analyzer/pipelines/           # chapters, silence, blackframe, subtitles, chromaprint, tidb, fuser, picture
k8s/                              # Deployment, Service, ServiceAccount, ServiceMonitor, GrafanaDashboard
Dockerfile
```

## Local development

```bash
uv sync
uv run pytest
```

The picture-area tests make short videos with ffmpeg (`tests/test_picture.py`:
a letterbox, a pillarbox, a windowbox, a dark scene and a fade, an aspect
change, grain in the bars, a subtitle and a logo in a bar, a full frame,
and a package of fMP4 segments); they are skipped without an ffmpeg that
encodes H.264.

## Build the container

```bash
docker build -t zaentrum/analyzer .
```

Build and push the image to your own registry, then apply the `k8s/`
manifests and update the image reference for your environment.

## License

[MPL-2.0](LICENSE).
