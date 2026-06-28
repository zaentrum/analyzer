"""Entry point. One process runs:
  - the worker loop (thread)
  - a tiny FastAPI server for /healthz and /readyz, so k8s probes work.
The worker is the actual job; the HTTP server is just kubelet plumbing."""

from __future__ import annotations

import logging
import queue
import signal
import sys
import threading

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException

from .config import Config
from .katalog import KatalogClient
from .packager import package_item, package_status
from .worker import run_tidb_first_worker, run_worker


def _configure_logging() -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )


def main() -> int:
    _configure_logging()
    log = structlog.get_logger("analyzer.main")
    cfg = Config.from_env()
    log.info("analyzer.start", katalog=cfg.katalog_api_url, whisper=cfg.enable_whisper)

    client = KatalogClient(
        base_url=cfg.katalog_api_url,
        token_url=cfg.oidc_token_url,
        client_id=cfg.oidc_client_id,
        client_secret=cfg.oidc_client_secret,
    )

    stop = threading.Event()

    def _handle_sigterm(signum: int, _frame: object) -> None:
        log.info("analyzer.signal", signum=signum)
        stop.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    worker_thread = threading.Thread(
        target=run_worker,
        kwargs={
            "client": client,
            "batch_size": cfg.claim_batch_size,
            "idle_sleep": cfg.idle_sleep_seconds,
            "error_sleep": cfg.error_sleep_seconds,
            "stop": stop,
        },
        daemon=True,
        name="analyzer-worker",
    )
    worker_thread.start()

    # Stage-1 TIDB-only sweep runs in a sibling thread so it doesn't
    # block (or get blocked by) the slow per_file ML loop. Sharing the
    # KatalogClient is safe — httpx.Client is documented as thread-safe
    # for concurrent requests.
    tidb_first_thread: threading.Thread | None = None
    if cfg.tidb_first_enabled:
        tidb_first_thread = threading.Thread(
            target=run_tidb_first_worker,
            kwargs={
                "client": client,
                "batch_size": cfg.tidb_first_batch_size,
                "idle_sleep": cfg.tidb_first_idle_sleep_seconds,
                "error_sleep": cfg.error_sleep_seconds,
                "claim_interval": cfg.tidb_first_claim_interval_seconds,
                "stop": stop,
            },
            daemon=True,
            name="analyzer-tidb-first",
        )
        tidb_first_thread.start()
        log.info("analyzer.tidb_first.started",
                 batch_size=cfg.tidb_first_batch_size,
                 claim_interval=cfg.tidb_first_claim_interval_seconds)

    # Per-item packaging queue. A single consumer thread runs one
    # shaka-packager invocation at a time — packaging is CPU-bound and
    # we don't want N parallel jobs starving the analyzer's other
    # pipelines. The FastAPI POST endpoint just enqueues; the worker
    # picks up and runs to completion. Items already on the queue are
    # deduplicated by item_id.
    pkg_queue: queue.Queue[tuple[str, str, str | None]] = queue.Queue()
    pkg_enqueued: set[str] = set()
    pkg_lock = threading.Lock()

    def _packaging_consumer() -> None:
        plog = structlog.get_logger("analyzer.packager.consumer")
        while not stop.is_set():
            try:
                item_id, src, item_type = pkg_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                plog.info("packager.dequeue", item_id=item_id, source=src, type=item_type)
                package_item(item_id, src, item_type=item_type)
            except Exception as e:
                # package_item already wrote .failed; just log.
                plog.exception("packager.run.failed", item_id=item_id, error=str(e))
            finally:
                with pkg_lock:
                    pkg_enqueued.discard(item_id)
                pkg_queue.task_done()

    pkg_thread = threading.Thread(
        target=_packaging_consumer,
        daemon=True,
        name="analyzer-packager",
    )
    pkg_thread.start()

    app = FastAPI()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/readyz")
    def readyz() -> dict:
        # Per_file thread must be alive; tidb_first is optional.
        return {"ok": worker_thread.is_alive()}

    @app.post("/api/package/{item_id}")
    def enqueue_package(item_id: str) -> dict:
        """Enqueue a packaging job. The source path is resolved here
        (analyzer is the only service that knows on-disk file paths;
        chino-api never has to learn them). Idempotent: re-posting
        while already queued or running is a no-op."""
        item = client.get_item(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="item not found or has no primary asset")
        with pkg_lock:
            if item_id in pkg_enqueued:
                return {"state": "queued", "item_id": item_id, "alreadyEnqueued": True}
            pkg_enqueued.add(item_id)
        pkg_queue.put((item_id, item.path, item.type))
        return {
            "state": "queued",
            "item_id": item_id,
            "source": item.path,
            "type": item.type,
            "queueDepth": pkg_queue.qsize(),
        }

    @app.get("/api/package/{item_id}")
    def get_package_status(item_id: str) -> dict:
        st = package_status(item_id)
        with pkg_lock:
            st["queued"] = item_id in pkg_enqueued
        return st

    uvicorn.run(app, host="0.0.0.0", port=8080, log_config=None)
    stop.set()
    client.close()
    worker_thread.join(timeout=10)
    if tidb_first_thread is not None:
        tidb_first_thread.join(timeout=10)
    pkg_thread.join(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
