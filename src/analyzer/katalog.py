"""HTTP client for the katalog Spring app: claim work, upload segments,
fail items. JWT token is cached and re-fetched on 401."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

# Refresh tokens a bit before they expire; Keycloak default is 300s for
# client_credentials, giving us a 30s safety margin.
TOKEN_REFRESH_LEAD_SECONDS = 30


@dataclass
class ClaimedItem:
    id: str
    type: str
    title: str
    year: int | None
    duration_ms: int | None
    path: str
    # Season + episode coords + TMDB IDs flow through from the claim
    # payload so the TIDB pipeline can ask
    # `GET /v2/media?tmdb_id=…&season=…&episode=…`. Movies carry their
    # own TMDB ID; episodes inherit it from the parent series. Either
    # may be None — TIDB pipeline skips the call when there's nothing
    # to ask for.
    season_number: int | None = None
    episode_number: int | None = None
    series_tmdb_id: str | None = None
    movie_tmdb_id: str | None = None

    @property
    def tmdb_id(self) -> str | None:
        """Pick the TMDB ID applicable to this item's media type."""
        if self.type == "movie":
            return self.movie_tmdb_id
        if self.type == "episode":
            return self.series_tmdb_id
        return None


class KatalogClient:
    def __init__(
        self,
        base_url: str,
        token_url: str,
        client_id: str,
        client_secret: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = httpx.Client(timeout=timeout_seconds)
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def close(self) -> None:
        self._http.close()

    # ---------------------------------------------------------------- auth
    def _ensure_token(self) -> str:
        if self._token and time.time() < self._token_expires_at:
            return self._token
        resp = self._http.post(
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        ttl = int(body.get("expires_in", 60))
        self._token_expires_at = time.time() + ttl - TOKEN_REFRESH_LEAD_SECONDS
        log.debug("oidc.token_refreshed", expires_in=ttl)
        return self._token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._ensure_token()}"}

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self._base}{path}"
        for attempt in range(2):
            resp = self._http.request(method, url, headers=self._headers(), **kwargs)
            if resp.status_code == 401 and attempt == 0:
                # Token revoked / rotated; force-refresh and retry once.
                self._token = None
                self._token_expires_at = 0
                continue
            return resp
        return resp  # type: ignore[return-value]

    # ------------------------------------------------------------- claims
    def claim(self, pass_kind: str = "per_file", limit: int = 2) -> list[ClaimedItem]:
        resp = self._request(
            "POST",
            f"/api/analyze/claim?pass={pass_kind}&limit={limit}",
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [
            ClaimedItem(
                id=it["id"],
                type=it["type"],
                title=it.get("title") or "",
                year=it.get("year"),
                duration_ms=it.get("durationMs"),
                path=it["path"],
                season_number=it.get("seasonNumber"),
                episode_number=it.get("episodeNumber"),
                series_tmdb_id=it.get("seriesTmdbId"),
                movie_tmdb_id=it.get("movieTmdbId"),
            )
            for it in items
        ]

    # ---------------------------------------------------------- uploads
    def upload_segments(self, item_id: str, segments: list[dict[str, Any]]) -> None:
        resp = self._request(
            "PUT",
            f"/api/segments/items/{item_id}",
            json={"segments": segments},
        )
        if resp.status_code >= 400:
            log.error(
                "segments.upload_failed",
                item_id=item_id,
                status=resp.status_code,
                body=resp.text[:500],
            )
            resp.raise_for_status()

    def get_steps(self, item_id: str) -> dict[str, str]:
        """Return the current status of every analyzer step on `item_id`.
        Used by the per_file worker to skip pipelines whose step is
        already done or marked not_applicable by the tidb_first pass."""
        try:
            resp = self._request(
                "GET",
                f"/api/analyze/items/{item_id}/steps",
            )
            if resp.status_code >= 400:
                log.warning(
                    "steps.get_failed",
                    item_id=item_id,
                    status=resp.status_code,
                    body=resp.text[:200],
                )
                return {}
            body = resp.json()
            steps = body.get("steps") or {}
            return {str(k): str(v) for k, v in steps.items()}
        except Exception as e:
            log.warning("steps.get_exception", item_id=item_id, error=str(e)[:200])
            return {}

    def mark_steps_not_applicable(
        self, item_id: str, steps: list[str], reason: str | None = None
    ) -> None:
        """Bulk-mark steps as not_applicable. Best-effort; failures are
        logged and swallowed so a short-circuit hiccup doesn't crash
        the worker — the at-worst-case is the per_file pass running on
        an item TIDB already handled, which is wasteful but correct."""
        if not steps:
            return
        try:
            resp = self._request(
                "POST",
                f"/api/analyze/items/{item_id}/steps/skip",
                json={"steps": steps, "reason": reason or ""},
            )
            if resp.status_code >= 400:
                log.warning(
                    "steps.skip_failed",
                    item_id=item_id,
                    steps=steps,
                    status=resp.status_code,
                    body=resp.text[:200],
                )
        except Exception as e:
            log.warning(
                "steps.skip_exception",
                item_id=item_id,
                steps=steps,
                error=str(e)[:200],
            )

    def upload_chapters(self, item_id: str, chapters: list[dict[str, Any]]) -> None:
        """Replace the chapter set for an item. Chapters are file-internal
        structural markers (Cold Open / Act 1) extracted by ffprobe and
        live in com_nalet_katalog_itemchapters — separate from the TIDB-
        aligned MediaSegments. See migration 018 for the split."""
        resp = self._request(
            "PUT",
            f"/api/chapters/items/{item_id}",
            json={"chapters": chapters},
        )
        if resp.status_code >= 400:
            log.error(
                "chapters.upload_failed",
                item_id=item_id,
                status=resp.status_code,
                body=resp.text[:500],
            )
            resp.raise_for_status()

    def get_item(self, item_id: str) -> ClaimedItem | None:
        """Fetch one item with its primary playback path. Returns None
        when the item is unknown or has no primary asset. Used by the
        packager flow to resolve the source file from an admin-issued
        item id (the admin endpoint never trusts a client-supplied
        path)."""
        resp = self._request("GET", f"/api/analyze/items/{item_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        it = resp.json()
        return ClaimedItem(
            id=it["id"],
            type=it["type"],
            title=it.get("title") or "",
            year=it.get("year"),
            duration_ms=it.get("durationMs"),
            path=it["path"],
        )

    def siblings(self, item_id: str, limit: int = 5) -> list[ClaimedItem]:
        """Return up to N sibling episodes of the same series + season,
        with primary playback paths attached. Used by the chromaprint
        pipeline to find a recurring intro / credit-roll across the
        episodes that share a theme tune. Empty list when the item has
        no siblings (movie, only-episode show, etc.)."""
        resp = self._request(
            "GET",
            f"/api/analyze/items/{item_id}/siblings?limit={limit}",
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [
            ClaimedItem(
                id=it["id"],
                type=it["type"],
                title=it.get("title") or "",
                year=it.get("year"),
                duration_ms=it.get("durationMs"),
                path=it["path"],
                season_number=it.get("seasonNumber"),
                episode_number=it.get("episodeNumber"),
                series_tmdb_id=it.get("seriesTmdbId"),
                movie_tmdb_id=it.get("movieTmdbId"),
            )
            for it in items
        ]

    def upsert_step(
        self,
        item_id: str,
        step: str,
        status: str,
        *,
        error: str | None = None,
        details: str | None = None,
    ) -> None:
        """Upsert one processing-step row on the katalog side.

        Best-effort: failures are logged and swallowed so a flaky step
        bookkeeping call doesn't crash the worker mid-analysis. The
        underlying katalog endpoint is idempotent (ON CONFLICT (item_id,
        step) DO UPDATE …), so re-tries on retry are safe.
        """
        body: dict[str, Any] = {"status": status}
        if error is not None:
            body["error"] = error[:500]
        if details is not None:
            body["details"] = details
        try:
            resp = self._request(
                "PUT",
                f"/api/analyze/items/{item_id}/steps/{step}",
                json=body,
            )
            if resp.status_code >= 400:
                log.warning(
                    "step.upsert_failed",
                    item_id=item_id,
                    step=step,
                    status=status,
                    http=resp.status_code,
                    body=resp.text[:300],
                )
        except Exception as e:
            log.warning("step.upsert_exception", item_id=item_id, step=step, error=str(e)[:200])

    def fail(self, item_id: str, reason: str) -> None:
        resp = self._request(
            "POST",
            f"/api/analyze/items/{item_id}/fail",
            json={"reason": reason},
        )
        if resp.status_code >= 400:
            log.warning(
                "fail.report_failed",
                item_id=item_id,
                status=resp.status_code,
                body=resp.text[:300],
            )
