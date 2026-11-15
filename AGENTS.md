# analyzer

The analyzer is the first worker of the platform's media pipeline, a Python service. It
consumes `stube.catalog.item.enriched` from Kafka (consumer group `analyzer-workers`), runs
its detectors on the title's original (`chapter`, `tidb`, `subtitle`, `blackframe`,
`silence`, and `chromaprint` across a series' episodes), writes the segments, chapters and
step statuses to the catalog API (katalog-manager), and sends `stube.catalog.item.analyzed`
on to the transcoder. Beside that it measures each title's picture area and reads the
chapters of titles that miss theirs, as the catalog's backfills hand them out. The
[README](README.md) has the rules of each.

How a change is proposed and what a pull request has to pass is in the
[guide to contributing](https://github.com/zaentrum/.github/blob/main/CONTRIBUTING.md);
what may be built is in [building from the specification](https://github.com/zaentrum/zaentrum/blob/main/AGENTS.md).

## Build

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -e .
python -m compileall src
```

Python 3.11 or later (`requires-python` in `pyproject.toml`; `build.yml` builds with 3.11).
At run time the detectors call `ffmpeg` and `ffprobe` from the PATH, `chromaprint` also
`fpcalc`. The image:

```sh
docker build -t zaentrum/analyzer .
```

It is based on `nvidia/cuda` (12.3 runtime, Ubuntu 22.04) and adds Ubuntu's ffmpeg,
`libchromaprint-tools` and shaka-packager v3.4.2; `--build-arg BASE=<mirror>/` takes the
base image from a mirror. `image.yml` builds and pushes it on `main`, on a tag `vX.Y.Z` and
when started by hand.

## Test

`build.yml` installs `ffmpeg` and `mkvtoolnix` from apt, then runs:

```sh
pip install -e ".[dev]"
pytest -q
```

- `tests/test_picture_cases.py` measures small videos that ffmpeg makes here. It skips
  without an ffmpeg that encodes H.264 (libx264), its HEVC cases without libx265.
- `tests/test_chapters.py` reads files with chapter marks through the `ffprobe` on the PATH:
  those cases skip without ffmpeg and ffprobe, the mkvmerge cases (there and in the picture
  cases) without `mkvmerge`.
- The rest needs nothing installed. The catalog is a fake behind an httpx mock transport and
  the broker a fake (`tests/test_redelivery.py`, `tests/test_pictures.py`,
  `tests/test_chapter_backfill.py`).
- The image's ffprobe is Ubuntu 22.04's 4.4, whose header reads differ from newer ones'; run
  the chapter tests with it too.

## Run it

The analyzer runs inside the platform: the operator's chart starts it when the `Zaentrum`
resource sets `features.pipeline` ([self-hosting](https://github.com/zaentrum/zaentrum/blob/main/docs/self-hosting.md)),
as on the reference demo ([how it is deployed](https://github.com/zaentrum/zaentrum/blob/main/docs/reference-demo.md)).
It needs the catalog API, an OIDC client's id and secret (`OIDC_CLIENT_ID`,
`OIDC_CLIENT_SECRET`, for the client-credentials grant; the platform's chart gives analyzer,
packager and transcoder one shared client), a Kafka broker and
the library share at `/var/lib/katalog`. Started by hand against those:

```sh
export KATALOG_API_URL=http://localhost:8081 OIDC_TOKEN_URL=https://idp.example.org/token \
  OIDC_CLIENT_ID=analyzer OIDC_CLIENT_SECRET=... KAFKA_BROKERS=localhost:9092
python -m analyzer.main
```

`KATALOG_API_URL` and the three `OIDC_` variables are required; `config.py` has the defaults
of the others: `KAFKA_GROUP_ID`, `CONSUME_TOPIC`, `PRODUCE_TOPIC`, `KAFKA_SECURITY_PROTOCOL`
(`PLAINTEXT`; a mounted `KAFKA_CERT_DIR` switches to mTLS). `PICTURE_AREA`,
`PICTURE_BACKFILL`, `CHAPTER_BACKFILL` and `TIDB_ENABLED` set to `false` turn each off. Port
8080 serves `/healthz`, `/readyz` (whether the consumer thread is alive) and
`/api/package/{item_id}`. Without a broker or a catalog it starts all the same, answers both
probes and logs each failed connection. No mock server or recorded answers ship; the tests'
fake catalog and broker stand in for them.

## What the tests leave unchecked

- The detectors that decode a whole file never run on a file in a test: the pass tests
  replace them with recorders. `blackframe`, `silence` and `chromaprint` (`fpcalc`) have no
  test of their own; `tests/test_subtitles.py` and `tests/test_keyframe.py` hold only the
  parsing and the pure logic, not the ffmpeg runs.
- `tidb` calls an external API; its tests answer through a mock transport, never the network.
- The per-item CMAF packager (`packager.py`, shaka-packager, `/api/package/{item_id}`) and the
  HTTP server in `main.py` have no test.
- No test talks to a real broker or catalog, or fetches an OIDC token.
- No test runs on a GPU or in the image, which only `image.yml` builds, on `main`, on tags
  and when started by hand.
  `ENABLE_WHISPER` and the `WHISPER_*` settings are read, but no pipeline uses them.
