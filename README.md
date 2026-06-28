# analyzer

Python content-analysis pipelines for the zaentrum platform. Consumes
analyze tasks from Kafka (`stube.processing.task.analyze.*`) and runs a
set of per-file detectors — chapters, silence, blackframe, subtitles,
and cross-episode audio-fingerprint (chromaprint) intro/credits
detection — plus a per-item CMAF packager.

## Status

Scaffold. Code lands in Phase 2 of the migration plan. The Kafka
consumer wiring is in progress; the worker currently still polls the
catalog HTTP API.

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
