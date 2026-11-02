"""The picture-area backfill (backfill.py) against a fake catalog: what a
claim hands out is measured and reported, nothing handed out waits as long
as the catalog says, and a measurement past its deadline gives its title
back. The measurement itself is replaced by a recorder."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime

import httpx
import pytest

from analyzer import backfill
from analyzer.katalog import KatalogClient
from analyzer.pipelines import picture

BASE = "http://catalog.test"
FILM = "5b0e1c2d-0000-4000-8000-000000000001"
EXTRA = "5b0e1c2d-0000-4000-8000-000000000002"


class Catalog:
    """POST /api/picture/claim answers the claims queued, one per call (an
    empty claim once they are used up); every PUT is recorded."""

    def __init__(self, claims: list[dict] | None, status: int = 200) -> None:
        self.claims = list(claims or [])
        self.status = status
        self.asked: list[dict] = []
        self.reports: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/token":
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300})
        body = json.loads(request.content or b"null")
        if request.method == "POST" and path == "/api/picture/claim":
            self.asked.append(body)
            if self.status != 200:
                return httpx.Response(self.status, text="no backfill")
            if not self.claims:
                return httpx.Response(
                    200, json={"items": [], "idle": "every title has one", "retryAfterSeconds": 900}
                )
            return httpx.Response(200, json=self.claims.pop(0))
        if request.method == "PUT" and path.startswith("/api/picture/"):
            self.reports.append((path, body))
            return httpx.Response(200, json={})
        return httpx.Response(404)


def client(catalog: Catalog) -> KatalogClient:
    c = KatalogClient(BASE, f"{BASE}/token", "worker", "not-a-secret")
    c._http = httpx.Client(transport=httpx.MockTransport(catalog))
    return c


@pytest.fixture
def measured(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int | None, float | None]]:
    """What the backfill measured: each path, its duration and deadline."""
    calls: list[tuple[str, int | None, float | None]] = []

    def measure(
        path: str,
        duration_ms: int | None = None,
        *,
        deadline: float | None = None,
        **_kwargs: object,
    ) -> picture.Measurement:
        calls.append((path, duration_ms, deadline))
        if deadline is not None and deadline <= 0:
            return picture.Measurement(
                path=path, interrupted=True, reason="stopped before it was done"
            )
        return picture.Measurement(
            path=path,
            frame=picture.Box(0, 0, 1920, 1080),
            area=picture.Box(0, 138, 1920, 804),
            samples=300,
            informative=290,
        )

    monkeypatch.setattr(picture, "measure", measure)
    return calls


def claim(*jobs: dict, deadline: str = "2026-11-03T17:00:00Z") -> dict:
    return {"items": list(jobs), "deadline": deadline}


FILM_JOB = {
    "kind": "item",
    "id": FILM,
    "itemId": FILM,
    "type": "movie",
    "path": "/library/movies/5b/film/versions/v1/hls/v0/playlist.m3u8",
    "durationMs": 7_200_000,
}
EXTRA_JOB = {
    "kind": "extra",
    "id": EXTRA,
    "itemId": FILM,
    "type": "extra",
    "path": "/library/movies/5b/film/extras/x1/hls/v0/playlist.m3u8",
    "durationMs": 120_000,
}


def test_a_claims_titles_are_measured_and_reported(measured) -> None:
    catalog = Catalog([claim(FILM_JOB, EXTRA_JOB)])
    wait = backfill.once(client(catalog), threading.Event(), "analyzer-0")
    assert wait == 0  # titles were handed out: ask again at once
    assert catalog.asked == [{"worker": "analyzer-0", "limit": 1}]
    deadline = datetime(2026, 11, 3, 17, 0, tzinfo=UTC).timestamp()
    assert measured == [
        (FILM_JOB["path"], 7_200_000, deadline),
        (EXTRA_JOB["path"], 120_000, deadline),
    ]
    assert [p for p, _b in catalog.reports] == [
        f"/api/picture/items/{FILM}",
        f"/api/picture/extras/{EXTRA}",
    ]
    body = catalog.reports[0][1]
    assert body["area"] == {"x": 0, "y": 138, "width": 1920, "height": 804}
    assert body["frame"] == {"width": 1920, "height": 1080} and body["path"] == FILM_JOB["path"]


def test_a_measurement_past_its_deadline_gives_its_title_back(measured) -> None:
    catalog = Catalog([claim(FILM_JOB, deadline="1970-01-01T00:00:00Z")])
    backfill.once(client(catalog), threading.Event(), "analyzer-0")
    assert catalog.reports == [
        (f"/api/picture/items/{FILM}", {"released": True, "path": FILM_JOB["path"]})
    ]


@pytest.mark.parametrize(
    ("after", "wait"), [(900, 900.0), (5, backfill.WAIT_MIN_S), (86_400, backfill.WAIT_MAX_S)]
)
def test_nothing_handed_out_waits_as_long_as_the_catalog_says(measured, after, wait) -> None:
    catalog = Catalog([{"items": [], "idle": "inside the quiet hours", "retryAfterSeconds": after}])
    assert backfill.once(client(catalog), threading.Event(), "analyzer-0") == wait
    assert measured == [] and catalog.reports == []


@pytest.mark.parametrize("status", [404, 503])
def test_a_catalog_without_a_backfill_is_asked_again_later(measured, status) -> None:
    catalog = Catalog(None, status=status)
    assert (
        backfill.once(client(catalog), threading.Event(), "analyzer-0")
        == backfill.NO_BACKFILL_WAIT_S
    )


def test_a_malformed_title_is_skipped(measured) -> None:
    catalog = Catalog([claim({"kind": "item", "id": FILM})])
    backfill.once(client(catalog), threading.Event(), "analyzer-0")
    assert measured == [] and catalog.reports == []


def test_the_loop_ends_when_the_worker_stops(measured) -> None:
    catalog = Catalog([claim(FILM_JOB)])
    stop = threading.Event()
    t = threading.Thread(target=backfill.run, args=(client(catalog), stop, "analyzer-0"))
    t.start()
    # The claim is measured, then the empty one makes it wait: stop wakes it.
    for _ in range(200):
        if len(catalog.asked) >= 2:
            break
        threading.Event().wait(0.01)
    stop.set()
    t.join(timeout=5)
    assert not t.is_alive()
    assert [p for p, _b in catalog.reports] == [f"/api/picture/items/{FILM}"]
