"""Kafka plumbing for the event-driven analyzer.

The analyzer is a pure event consumer: it reads item ids off
`stube.catalog.item.enriched`, runs the per-file pipeline, then emits
`stube.catalog.item.analyzed` for the transcoder. This module owns the
confluent-kafka Consumer/Producer wiring and the shared event-envelope
schema so all three workers (analyzer / transcoder / packager) + the Go
hub stay byte-compatible.

Contract (must match across every worker + the hub):

* BROKER: comma-separated `KAFKA_BROKERS` (e.g. "kafka:9092"). The
  bundled demo broker is PLAINTEXT — no TLS, no certs. `security.protocol`
  defaults to PLAINTEXT, env-overridable via `KAFKA_SECURITY_PROTOCOL`.
* CONSUMER: group.id per-worker, enable.auto.commit=false,
  auto.offset.reset=earliest. The offset is committed by the CALLER only
  after the item is fully processed AND the next event is produced, so a
  crash mid-work reprocesses (idempotent on the katalog side).
* PRODUCER: acks=all, key = itemId (utf-8) for per-item ordering.
  flush() is called before the caller commits the consumed offset.

EVENT ENVELOPE (JSON value):
  {"eventId": <uuid4 hex>, "itemId": <str>, "type": <str>, "step": <str>,
   "status": <str>, "occurredAt": <RFC3339>, "source": <str>}
Consumers only REQUIRE `itemId`; every other field is tolerated/ignored.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from confluent_kafka import Consumer, KafkaError, Producer

log = structlog.get_logger("analyzer.kafka")


def _security_conf(security_protocol: str) -> dict[str, str]:
    """Kafka security settings. When KAFKA_CERT_DIR points at a mounted
    mTLS secret (user.crt/user.key + the CLUSTER CA's ca.crt — the shared
    Strimzi profile), it wins over `security_protocol`: a mounted cert dir
    IS the operator's way of saying "this broker speaks mTLS"."""
    cert_dir = os.environ.get("KAFKA_CERT_DIR", "").strip()
    if cert_dir and os.path.isdir(cert_dir):
        return {
            "security.protocol": "SSL",
            "ssl.ca.location": os.path.join(cert_dir, "ca.crt"),
            "ssl.certificate.location": os.path.join(cert_dir, "user.crt"),
            "ssl.key.location": os.path.join(cert_dir, "user.key"),
        }
    return {"security.protocol": security_protocol}


def now_rfc3339() -> str:
    """Current UTC time as an RFC3339 / ISO-8601 string with offset."""
    return datetime.now(UTC).isoformat()


def parse_item_id(raw: bytes | str | None) -> str | None:
    """Parse a message value into its `itemId`. Returns None when the
    payload is missing, non-JSON, not an object, or has no non-empty
    `itemId` — the caller logs a warning and commits/skips those so a
    single poison message doesn't wedge the partition."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    item_id = obj.get("itemId")
    if item_id is None:
        return None
    item_id = str(item_id).strip()
    return item_id or None


def build_event(
    item_id: str,
    *,
    step: str,
    status: str = "done",
    type_: str | None = None,
    source: str = "analyzer",
) -> dict[str, Any]:
    """Build the next-stage event envelope. `eventId` is a fresh uuid4
    hex and `occurredAt` is the current UTC RFC3339 time."""
    event: dict[str, Any] = {
        "eventId": uuid.uuid4().hex,
        "itemId": item_id,
        "step": step,
        "status": status,
        "occurredAt": now_rfc3339(),
        "source": source,
    }
    if type_ is not None:
        event["type"] = type_
    return event


def build_consumer(brokers: str, group_id: str, security_protocol: str) -> Consumer:
    """Construct the manual-commit, earliest-reset consumer used by the
    event loop. Offsets are committed explicitly by the caller."""
    return Consumer(
        {
            "bootstrap.servers": brokers,
            "group.id": group_id,
            **_security_conf(security_protocol),
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            # Keep the broker from evicting us mid-analysis: a per-file
            # pass (ffmpeg + optional whisper) can run for minutes.
            "max.poll.interval.ms": 1_800_000,
        }
    )


def build_producer(brokers: str, security_protocol: str) -> Producer:
    """Construct the acks=all producer used to emit the next-stage
    event. Keyed by itemId at produce-time for per-item ordering."""
    return Producer(
        {
            "bootstrap.servers": brokers,
            **_security_conf(security_protocol),
            "acks": "all",
            "enable.idempotence": True,
        }
    )


def produce_event(producer: Producer, topic: str, event: dict[str, Any]) -> None:
    """Produce one envelope keyed by itemId (utf-8) so all events for an
    item land on the same partition (per-item ordering). The caller must
    flush() before committing the consumed offset."""
    producer.produce(
        topic,
        key=event["itemId"].encode("utf-8"),
        value=json.dumps(event).encode("utf-8"),
    )


def is_partition_eof(err: KafkaError | None) -> bool:
    """True when a poll error is the benign end-of-partition signal
    rather than a real error."""
    return err is not None and err.code() == KafkaError._PARTITION_EOF
