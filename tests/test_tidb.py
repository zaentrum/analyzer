"""Pipeline-level tests for the TIDB fetcher and its interaction with
the fuser. No network access — all calls go through httpx.MockTransport.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import httpx
import pytest

from analyzer.pipelines import fuser, tidb


@contextmanager
def _patched_httpx_client(handler):
    """Stand in for httpx.Client so we can answer requests deterministically."""
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    tidb.httpx.Client = _factory  # type: ignore[attr-defined]
    try:
        yield
    finally:
        tidb.httpx.Client = real_client  # type: ignore[attr-defined]


def _ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "tmdb_id": 1396,
            "type": "tv",
            "season": 1,
            "episode": 1,
            "intro": [{"start_ms": 228694, "end_ms": 245250}],
            "credits": [{"start_ms": 3431000, "end_ms": None}],
        },
    )


def test_detect_returns_tidb_source_and_resolves_null_end() -> None:
    with _patched_httpx_client(_ok):
        segs = tidb.detect(tmdb_id=1396, season=1, episode=1, duration_ms=3_500_000)
    assert len(segs) == 2
    intro = next(s for s in segs if s["kind"] == "intro")
    assert intro["source"] == "tidb"
    assert intro["startMs"] == 228694
    assert intro["endMs"] == 245250
    credits = next(s for s in segs if s["kind"] == "credits")
    # Null end_ms must be resolved to the item duration so the fuser /
    # decider downstream don't have to reason about open intervals.
    assert credits["endMs"] == 3_500_000


def test_detect_returns_empty_on_404() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "media not found"})

    with _patched_httpx_client(handler):
        assert tidb.detect(tmdb_id=1100, season=1, episode=1) == []


def test_detect_returns_empty_on_429() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"error": "rate limited"},
            headers={"X-RateLimit-Reset": "30"},
        )

    with _patched_httpx_client(handler):
        assert tidb.detect(tmdb_id=2316, season=1, episode=1) == []


def test_detect_skips_when_tmdb_id_missing() -> None:
    assert tidb.detect(tmdb_id=None, season=1, episode=1) == []
    assert tidb.detect(tmdb_id="", season=1, episode=1) == []


def test_detect_skips_episode_without_coords() -> None:
    # season/episode are required for TV. Without them we don't call out.
    assert tidb.detect(tmdb_id=1396, media_type="tv") == []
    assert tidb.detect(tmdb_id=1396, season=1, media_type="tv") == []


def test_detect_movie_works_with_just_tmdb_id() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "tmdb_id": 12345,
                "type": "movie",
                "intro": [{"start_ms": None, "end_ms": 23_000}],
            },
        )

    with _patched_httpx_client(handler):
        segs = tidb.detect(tmdb_id=12345, media_type="movie", duration_ms=7_200_000)
    assert len(segs) == 1
    assert segs[0]["startMs"] == 0  # null start resolved to 0
    assert segs[0]["endMs"] == 23_000


def test_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIDB_ENABLED", "false")
    # `_ok` would otherwise return data; the disabled flag wins.
    with _patched_httpx_client(_ok):
        assert tidb.detect(tmdb_id=1396, season=1, episode=1) == []


def test_fuser_picks_tidb_over_chromaprint() -> None:
    tidb_intro = {
        "kind": "intro", "startMs": 0, "endMs": 31_000,
        "source": "tidb", "confidence": 0.98,
    }
    cp_intro = {
        "kind": "intro", "startMs": 2_000, "endMs": 33_000,
        "source": "chromaprint", "confidence": 0.85,
    }
    result = fuser.merge([[tidb_intro], [cp_intro]])
    intro = next(s for s in result if s["kind"] == "intro")
    # The merge collapses overlap; tidb wins as the canonical source
    # because it ranks higher in fuser.SOURCE_PRIORITY.
    assert intro["source"] == "tidb"


def test_api_key_header_promoted_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(404, json={"error": "media not found"})

    monkeypatch.setenv("TIDB_API_KEY", "abc.shh")
    # Make sure the env var didn't leak from a prior test.
    assert os.environ["TIDB_API_KEY"] == "abc.shh"
    with _patched_httpx_client(handler):
        tidb.detect(tmdb_id=1396, season=1, episode=1)
    assert captured["auth"] == "Bearer abc.shh"
