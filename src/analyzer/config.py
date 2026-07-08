"""Runtime configuration. Everything comes from env vars; defaults are
chosen so the analyzer is safe-by-default in dev (no destructive ops without
KATALOG_API_URL set)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    katalog_api_url: str
    oidc_token_url: str
    oidc_client_id: str
    oidc_client_secret: str
    # --- Kafka event-consumer settings -------------------------------
    # The analyzer is a PURE Kafka consumer/producer: it CONSUMEs the
    # enriched-item topic, runs the per-file pipeline, then PRODUCEs the
    # analyzed-item topic. Broker list is comma-separated (e.g.
    # "kafka:9092"). The bundled demo broker is PLAINTEXT — no TLS, no
    # certs — so security.protocol defaults to PLAINTEXT and is only an
    # env override for deployments that front the broker with SASL/TLS.
    kafka_brokers: str = "kafka:9092"
    kafka_group_id: str = "analyzer-workers"
    consume_topic: str = "stube.catalog.item.enriched"
    produce_topic: str = "stube.catalog.item.analyzed"
    kafka_security_protocol: str = "PLAINTEXT"
    # The step name carried on the PRODUCEd next-stage event. The
    # transcoder keys its work off this; keep it aligned with the
    # transcoder's expectation.
    produce_step: str = "transcode"
    error_sleep_seconds: float = 30.0
    # Whisper config (used by the GPU pipeline; the CPU-only pipelines ignore
    # these). large-v3 is overkill for credit-text detection; medium gives
    # 95%+ accuracy at half the VRAM.
    whisper_model: str = "medium"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    # When the worker is allowed to actually do GPU inference. Off by default
    # so the very first deploy can prove the CPU pipelines work before we
    # commit GPU time to it.
    enable_whisper: bool = False

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            katalog_api_url=_require("KATALOG_API_URL"),
            oidc_token_url=_require("OIDC_TOKEN_URL"),
            oidc_client_id=_require("OIDC_CLIENT_ID"),
            oidc_client_secret=_require("OIDC_CLIENT_SECRET"),
            kafka_brokers=os.environ.get("KAFKA_BROKERS", "kafka:9092"),
            kafka_group_id=os.environ.get("KAFKA_GROUP_ID", "analyzer-workers"),
            consume_topic=os.environ.get("CONSUME_TOPIC", "stube.catalog.item.enriched"),
            produce_topic=os.environ.get("PRODUCE_TOPIC", "stube.catalog.item.analyzed"),
            kafka_security_protocol=os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
            produce_step=os.environ.get("PRODUCE_STEP", "transcode"),
            error_sleep_seconds=float(os.environ.get("ERROR_SLEEP_SECONDS", "30")),
            whisper_model=os.environ.get("WHISPER_MODEL", "medium"),
            whisper_device=os.environ.get("WHISPER_DEVICE", "cuda"),
            whisper_compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "float16"),
            enable_whisper=os.environ.get("ENABLE_WHISPER", "false").lower() == "true",
        )


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"required env var {key} is empty/unset")
    return val
