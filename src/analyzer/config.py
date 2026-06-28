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
    # Per-worker tunables. The HTTP claim endpoint clamps to [1, 32] anyway.
    claim_batch_size: int = 2
    idle_sleep_seconds: float = 15.0
    error_sleep_seconds: float = 30.0
    # tidb_first sweep: TIDB-only worker that runs alongside the per_file
    # loop and short-circuits the ML pipelines when TIDB already has
    # data. Defaults are sized to stay comfortably under TIDB's anonymous
    # rate limit (30 req / 10 s).
    tidb_first_enabled: bool = True
    tidb_first_batch_size: int = 4
    tidb_first_idle_sleep_seconds: float = 30.0
    tidb_first_claim_interval_seconds: float = 0.5
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
            claim_batch_size=int(os.environ.get("CLAIM_BATCH_SIZE", "2")),
            idle_sleep_seconds=float(os.environ.get("IDLE_SLEEP_SECONDS", "15")),
            error_sleep_seconds=float(os.environ.get("ERROR_SLEEP_SECONDS", "30")),
            tidb_first_enabled=os.environ.get("TIDB_FIRST_ENABLED", "true").lower() == "true",
            tidb_first_batch_size=int(os.environ.get("TIDB_FIRST_BATCH_SIZE", "4")),
            tidb_first_idle_sleep_seconds=float(
                os.environ.get("TIDB_FIRST_IDLE_SLEEP_SECONDS", "30")),
            tidb_first_claim_interval_seconds=float(
                os.environ.get("TIDB_FIRST_CLAIM_INTERVAL_SECONDS", "0.5")),
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
