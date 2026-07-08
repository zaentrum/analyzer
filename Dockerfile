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

# ffmpeg / ffprobe drive every CPU pipeline (chapter extraction, silencedetect,
# blackdetect). libchromaprint-tools provides `fpcalc`, used by the chromaprint
# pipeline to detect recurring intro / credits across episodes of a series.
# python3.11 + venv keeps the runtime libs separate from CUDA.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      ffmpeg \
      libchromaprint-tools \
      ca-certificates \
      curl \
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
                confluent-kafka==2.5.3

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps -e .

# Run as a non-root user. The pod sets a high random uid via the OpenShift
# SCC; 0 here makes that uid take ownership of /app on container start.
RUN chown -R 0:0 /app && chmod -R g=u /app

EXPOSE 8080
CMD ["python", "-m", "analyzer.main"]
