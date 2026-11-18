# Base-image registry prefix. Empty default = public Docker Hub; a private
# deploy mirror passes --build-arg BASE=registry.example/library/ .
ARG BASE=
# CUDA-capable base for faster-whisper + future TransNetV2 scene-detection.
# Ubuntu 22.04 + CUDA 12.3 runtime is the smallest set that ships cuDNN 9
# (which CTranslate2 in faster-whisper 1.1 dynamically loads). Use the
# "runtime" variant — we don't need nvcc, only the runtime libs.
FROM ${BASE}nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Which ffprobe is called for what:
#   * the image's own ffmpeg and ffprobe, Ubuntu 22.04's 4.4.2 from the apt
#     line below, on the PATH (/usr/bin): the chapters, silence, black frames,
#     keyframes, subtitles, chromaprint's durations and the packager - every
#     pipeline but the two below;
#   * the picture area decodes with PyAV's own FFmpeg libraries (the av
#     wheel's), and calls no ffprobe;
#   * the captions backfill alone calls the pipeline's ffprobe 7.1.5, installed
#     further down at /opt/ffmpeg-7.1/bin/ffprobe and named by CAPTIONS_FFPROBE:
#     not on the PATH, called by nothing else, and refused by the backfill
#     when it isn't 7.1.
# ffmpeg / ffprobe drive every CPU pipeline (chapter extraction, silencedetect,
# blackdetect). libchromaprint-tools provides `fpcalc`, used by the chromaprint
# pipeline to detect recurring intro / credits across episodes of a series.
# python3.11 + venv keeps the runtime libs separate from CUDA. xz-utils
# unpacks the captions ffprobe's archive.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      ffmpeg \
      libchromaprint-tools \
      ca-certificates \
      curl \
      xz-utils \
      && rm -rf /var/lib/apt/lists/*

# shaka-packager: Google's CMAF/HLS/DASH packager, used by Netflix/Disney+
# under the hood. Single static binary from the official GitHub release.
# Used by the new analyzer.packager module to remux source files (HEVC
# passthrough, no re-encode) into the streaming-friendly tree under
# /var/lib/katalog/packages. Pinned to a released version so package
# output stays reproducible.
ARG SHAKA_PACKAGER_VERSION=v3.4.2
RUN curl -fsSL -o /usr/local/bin/packager \
      "https://github.com/shaka-project/shaka-packager/releases/download/${SHAKA_PACKAGER_VERSION}/packager-linux-x64" \
    && chmod +x /usr/local/bin/packager \
    && /usr/local/bin/packager --version

# The captions ffprobe - NOT the image's own. The captions backfill reads
# whether an original's video carries closed captions (A53 SEI) by decoding
# frames in windows across it, with the release of ffprobe the pipeline runs: BtbN's
# static GPL build of the ffmpeg 7.1 branch (n7.1.5), the same archive the
# transcoder and chino-stream images pin. The archive is checked against its
# sha256 (always: an override of the URL names its sha256 too), and only its
# ffprobe is taken, to a folder of its own that is not on the PATH. The build
# fails unless it says 7.1, and unless the PATH's ffprobe is still the
# image's 4.4 at /usr/bin: neither is taken for the other.
ARG CAPTIONS_FFPROBE_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-07-31-14-10/ffmpeg-n7.1.5-12-g1fdbca85aa-linux64-gpl-7.1.tar.xz
ARG CAPTIONS_FFPROBE_SHA256=c1e6caf48923dd8e6bc5e54d51ba70c321175b8162ae9c414c392990e72f0e79
RUN curl -fsSL "${CAPTIONS_FFPROBE_URL}" -o /tmp/ffmpeg-7.1.tar.xz \
    && echo "${CAPTIONS_FFPROBE_SHA256}  /tmp/ffmpeg-7.1.tar.xz" | sha256sum -c - \
    && mkdir -p /tmp/ffmpeg-7.1 /opt/ffmpeg-7.1/bin \
    && tar -xJf /tmp/ffmpeg-7.1.tar.xz -C /tmp/ffmpeg-7.1 --strip-components=1 \
         --wildcards '*/bin/ffprobe' \
    && install -m 0755 /tmp/ffmpeg-7.1/bin/ffprobe /opt/ffmpeg-7.1/bin/ffprobe \
    && rm -rf /tmp/ffmpeg-7.1 /tmp/ffmpeg-7.1.tar.xz \
    && /opt/ffmpeg-7.1/bin/ffprobe -version | head -n 1 \
    && /opt/ffmpeg-7.1/bin/ffprobe -version | head -n 1 | grep -q '^ffprobe version n7\.1\.' \
    && [ "$(command -v ffprobe)" = /usr/bin/ffprobe ] \
    && ffprobe -version | head -n 1 \
    && ffprobe -version | head -n 1 | grep -q '^ffprobe version 4\.4\.'
ENV CAPTIONS_FFPROBE=/opt/ffmpeg-7.1/bin/ffprobe

# Single venv at /opt/venv. Avoids the "system python" warnings and gives us
# a clean PATH for CMD.
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip setuptools wheel

# Install runtime dependencies first as a standalone layer so wheel cache
# survives source-only edits. The package itself is installed after the
# source is in place — setuptools.packages.find points at src/ so the
# directory must exist before `pip install -e .` runs.
WORKDIR /app
RUN pip install httpx==0.28.1 structlog==25.4.0 fastapi==0.118.0 \
                "uvicorn[standard]==0.32.0" faster-whisper==1.1.0 \
                confluent-kafka==2.5.3 av==18.1.0 numpy==2.4.6

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps -e .

# Run as a non-root user. The pod sets a high random uid via the OpenShift
# SCC; 0 here makes that uid take ownership of /app on container start.
RUN chown -R 0:0 /app && chmod -R g=u /app

EXPOSE 8080
CMD ["python", "-m", "analyzer.main"]
