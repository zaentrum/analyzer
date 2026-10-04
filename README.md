# analyzer

Python content-analysis pipelines for the zaentrum platform. Consumes
`stube.catalog.item.enriched` from Kafka (group `analyzer-workers`),
runs a set of per-file detectors — tidb, chapters, silence, blackframe,
subtitles, and cross-episode audio-fingerprint (chromaprint)
intro/credits detection — writes their segments, chapters and step
statuses to the catalog API, and emits `stube.catalog.item.analyzed`
for the transcoder. It also carries a per-item CMAF packager.

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
src/analyzer/config.py            # env-driven config
src/analyzer/katalog.py           # catalog API client (claim, upsert step, fail)
src/analyzer/packager.py          # per-item CMAF packager (shaka-packager, HEVC passthrough)
src/analyzer/pipelines/           # chapters, silence, blackframe, subtitles, chromaprint, tidb, fuser
k8s/                              # Deployment, Service, ServiceAccount, ServiceMonitor, GrafanaDashboard
Dockerfile
```

## Local development

```bash
uv sync
uv run pytest
```

## Build the container

```bash
docker build -t zaentrum/analyzer .
```

Build and push the image to your own registry, then apply the `k8s/`
manifests and update the image reference for your environment.

## License

[MPL-2.0](LICENSE).
