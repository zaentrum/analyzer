"""HTTP client for the katalog Spring app: fetch item detail, upload
segments/chapters (with what each read of a header was), bookkeep steps,
fail items, and claim the backfills' titles. JWT token is cached and
re-fetched on 401.

Work is no longer claimed over HTTP — the analyzer is a pure Kafka
consumer (see worker.run_event_consumer). This client only performs the
per-item reads + writes the pipeline needs once an event has told it
which item to process."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx
import structlog

if TYPE_CHECKING:
    from .pipelines.captions import CaptionsRead

log = structlog.get_logger(__name__)

# Refresh tokens a bit before they expire; Keycloak default is 300s for
# client_credentials, giving us a 30s safety margin.
TOKEN_REFRESH_LEAD_SECONDS = 30


class ClaimTimedOut(Exception):
    """The catalog did not answer a claim in time: it is there, and slow."""


@dataclass(frozen=True)
class CaptionsJob:
    """An original the captions backfill handed out: the original
    (source_id, what its PUT names) and its title (item_id), where it lies
    (path), how many video packets a window decodes as the claim names it
    (packets, any value it gives; None: it names none - the read reads the
    plan's 48, and fails a claim that names another), the claim's token, and
    the length its source record says (duration_ms, its
    container.durationMs: the read's span falls back on it)."""

    source_id: str
    path: str
    token: str
    item_id: str | None = None
    packets: Any = None
    duration_ms: int | None = None


@dataclass(frozen=True)
class CaptionsClaim:
    """The catalog's answer to a captions claim: the originals to read and
    the claim's deadline, or none and why (idle), with how long to wait
    (retry_after_s). malformed counts the originals handed out without a
    sourceId or a path, passed over."""

    jobs: list[CaptionsJob] = field(default_factory=list)
    deadline: float | None = None
    idle: str | None = None
    retry_after_s: float | None = None
    malformed: int = 0


@dataclass
class ClaimedItem:
    id: str
    type: str
    title: str
    year: int | None
    duration_ms: int | None
    path: str
    # Season + episode coords + TMDB IDs flow through from the item
    # detail so the TIDB pipeline can ask
    # `GET /v2/media?tmdb_id=…&season=…&episode=…`. Movies carry their
    # own TMDB ID; episodes inherit it from the parent series. Either
    # may be None — TIDB pipeline skips the call when there's nothing
    # to ask for.
    season_number: int | None = None
    episode_number: int | None = None
    series_tmdb_id: str | None = None
    movie_tmdb_id: str | None = None
    series_title: str | None = None
    # Whether the item has its OWN poster/backdrop (ignoring the series
    # fallback), so the keyframe pipeline only fills genuine gaps.
    has_own_poster: bool = False
    has_own_backdrop: bool = False

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
        extra = kwargs.pop("extra_headers", None)
        for attempt in range(2):
            headers = self._headers()
            if extra:
                headers.update(extra)
            resp = self._http.request(method, url, headers=headers, **kwargs)
            if resp.status_code == 401 and attempt == 0:
                # Token revoked / rotated; force-refresh and retry once.
                self._token = None
                self._token_expires_at = 0
                continue
            return resp
        return resp  # type: ignore[return-value]

    # -------------------------------------------------------- item detail
    @staticmethod
    def _item_from_json(it: dict[str, Any]) -> ClaimedItem:
        """Parse a katalog item-detail JSON object into ClaimedItem.
        Shared by get_item + siblings; tolerates missing optional keys
        so a partial payload never crashes the parse (only `id`, `type`,
        and `path` are load-bearing for the pipeline)."""
        return ClaimedItem(
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
            series_title=it.get("seriesTitle"),
            has_own_poster=bool(it.get("hasOwnPoster")),
            has_own_backdrop=bool(it.get("hasOwnBackdrop")),
        )

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

    def put_artwork(
        self, item_id: str, kind: str, data: bytes, content_type: str = "image/jpeg"
    ) -> None:
        """Upload a self-extracted keyframe as the item's poster/backdrop."""
        resp = self._request(
            "PUT",
            f"/api/artwork/{item_id}/{kind}",
            content=data,
            extra_headers={"Content-Type": content_type},
        )
        if resp.status_code >= 400:
            log.error(
                "artwork.upload_failed",
                item_id=item_id,
                kind=kind,
                status=resp.status_code,
                body=resp.text[:300],
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

    def put_chapters(
        self,
        item_id: str,
        chapters: list[dict[str, Any]],
        *,
        read: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> dict[str, Any] | None:
        """Replace the chapter set for an item with what a read of its
        original found, and say what the read was (`read`: chapters.ChapterRead
        .report()), which the catalog keeps with them (migration 055); token
        is the backfill's claim the read was for. Chapters are file-internal
        structural markers (Cold Open / Act 1) read by ffprobe and live in
        com_nalet_katalog_itemchapters — separate from the TIDB-aligned
        MediaSegments. See migration 018 for the split. Raises when the
        catalog did not take them; answers the title's read as it is now
        (None from a catalog that keeps no reads)."""
        body: dict[str, Any] = {"chapters": chapters}
        if read is not None:
            body["read"] = read
        if token:
            body["token"] = token
        resp = self._request("PUT", f"/api/chapters/items/{item_id}", json=body)
        if resp.status_code >= 400:
            log.error(
                "chapters.upload_failed",
                item_id=item_id,
                status=resp.status_code,
                body=resp.text[:500],
            )
            resp.raise_for_status()
        try:
            return resp.json().get("read")
        except (ValueError, AttributeError):
            return None

    def get_chapters(self, item_id: str) -> list[dict[str, Any]] | None:
        """The chapters the catalog keeps of an item (GET
        /api/chapters/items/{id}): [{"startMs", "endMs", "title", "ordinal"}].
        None when it can't say (an older catalog, or a failure)."""
        try:
            resp = self._request("GET", f"/api/chapters/items/{item_id}")
        except Exception as e:
            log.warning("chapters.get_exception", item_id=item_id, error=str(e)[:200])
            return None
        if resp.status_code != 200:
            return None
        try:
            chs = resp.json().get("chapters")
        except (ValueError, AttributeError):
            return None
        return chs if isinstance(chs, list) else None

    def report_chapters_failed(
        self, item_id: str, read: dict[str, Any], *, token: str | None = None
    ) -> bool:
        """Say that a read of an item's original failed (`read` with its
        "error"): the catalog leaves its chapters as they are and counts the
        failure. Best-effort: False when the catalog did not take it (one
        older than migration 055 keeps no reads)."""
        body: dict[str, Any] = {"read": read}
        if token:
            body["token"] = token
        try:
            resp = self._request("PUT", f"/api/chapters/items/{item_id}", json=body)
        except Exception as e:
            log.warning("chapters.report_exception", item_id=item_id, error=str(e)[:200])
            return False
        if resp.status_code >= 400:
            log.warning("chapters.report_failed", item_id=item_id, status=resp.status_code,
                        body=resp.text[:300])
            return False
        return True

    def release_chapters(self, item_id: str, token: str) -> bool:
        """Give a title the chapter backfill handed out back unread.
        Best-effort."""
        try:
            resp = self._request("PUT", f"/api/chapters/items/{item_id}",
                                 json={"released": True, "token": token})
        except Exception as e:
            log.warning("chapters.release_exception", item_id=item_id, error=str(e)[:200])
            return False
        return resp.status_code < 400

    def claim_chapters(
        self, worker: str, limit: int = 1, timeout: float | None = None
    ) -> dict[str, Any] | None:
        """The chapter backfill's next titles to read (POST
        /api/chapters/claim): {"items": [{"id", "type", "path", "sourceId",
        "durationMs", "token"}], "deadline"}, or no items and "idle" with
        "retryAfterSeconds". None when the catalog has no chapter backfill
        (404, an older one; 503, its migration missing) or fails. timeout is
        how long its answer is waited for (the client's own when None): a
        claim not answered by then raises ClaimTimedOut."""
        try:
            resp = self._request(
                "POST",
                "/api/chapters/claim",
                json={"worker": worker, "limit": limit},
                **({} if timeout is None else {"timeout": timeout}),
            )
        except httpx.TimeoutException as e:
            log.warning("chapters.claim_timeout", error=str(e)[:200], timeout_s=timeout)
            raise ClaimTimedOut(str(e)) from e
        except Exception as e:
            log.warning("chapters.claim_exception", error=str(e)[:200])
            return None
        if resp.status_code != 200:
            if resp.status_code != 404:
                log.warning("chapters.claim_failed", status=resp.status_code, body=resp.text[:300])
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    # ---------------------------------------------------- closed captions
    # The captions backfill's side of the catalog, all of it here: the paths,
    # the claim's answer and the body of a PUT, as katalog-manager's handler
    # takes them (internal/rest/captions.go; the body's types, captions.Report
    # and FileReport, in internal/captions, its windows library.CaptionWindow;
    # its README's "Closed captions"):
    #
    #   POST /api/captions/claim {"worker", "limit"}
    #     200 {"items": [{"sourceId", "itemId", "path", "packets", "token",
    #                     "durationMs"}],
    #          "deadline"}
    #         or {"items": [], "idle", "retryAfterSeconds"}
    #     404 no captions backfill (an older catalog), 503 its migration missing
    #   PUT /api/captions/sources/{sourceId}
    #     {"token", "original": READ}                    what the read found
    #     {"token", "released": true}                    given back unread
    #   READ = {"captionWindows", "duration", "span", "gop", "ffprobe",
    #           "sizeBytes", "captionStreams", "offList", "captionsKnown"?,
    #           "reason"?}                               its windows
    #       "captionsKnown": false and "reason" ("hevc: needs the packets'
    #       signature") of a picture whose windows can't tell captions: HEVC
    #       that showed none (the gate holds it until the record's
    #       captionSignature releases it)
    #       or {"error", "ffprobe", "sizeBytes"?, "captionStreams"?, "offList"?,
    #           "gop"?}                                  a file it could not read
    #       "sizeBytes" the size of the file read (where it was found),
    #       "captionStreams" its streams of closed captions of their own and
    #       "offList" what is off the closed lists (where the header probe ran),
    #       "gop" the longest group of pictures among its picture's first 1200
    #       packets, in seconds at its frame step (0: fewer than two keyframes;
    #       where they were listed)
    #   captionWindows = [{"at", "start", "end", "frames", "captions",
    #       "interlacedFrames", "repeatFrames", "keyframeBefore"}], every
    #       window the read placed: "at" the plan's own fraction (0 its
    #       start, then 0.1 ... 0.9: a window the tail caps, placed at most
    #       30 s short of the end, still says its fraction; six, never
    #       merged, or one whole read: at 0, "whole": true, its "packets"
    #       and its "leading" ones, before its first keyframe), "start" and "end" the
    #       presentation seconds of its first and last decoded frame, on the
    #       file's own timeline, "frames" how many it decoded, "captions"
    #       whether one showed them; a window that failed decoded 0 frames,
    #       its times, captions and counts null, and says why ("error", which
    #       the catalog's window doesn't read: it says why that window failed
    #       of its own): it is never one without captions
    #   "duration" is the length, in seconds, its file gives: the catalog
    #       holds it to its record's, and it places no window; "span"
    #       {"start", "end"} the picture's span as the read measured it: the
    #       first frame of its start, the end of its last video packet (the
    #       windows are placed over the span its records state: its start_time
    #       and duration, else the claim's durationMs, else the format's
    #       duration; the span measured within a frame step and 0.5 s of it at
    #       both ends); "gop" sets how far from its place a window may lie
    #       (max(10 s, gop + 0.5 s); of a gop of 0, or none said, a frame step
    #       and 0.5 s) and which picture is read whole
    #
    # "ffprobe" is the ffprobe that read it, its version line ("ffprobe
    # version n7.1.5-12-g1fdbca85aa-20260731"): the catalog takes 7.1 alone.
    # Whether the original has captions, and how many frames were decoded, is
    # the catalog's to work out of the windows. A read that failed as a whole
    # (no file, no video, no length, frames without times, a length that
    # can't be the video's, a file that changed under it, out of time,
    # stopped) is a file it could not read. A read whose ffprobe can't be
    # named is no report: put_captions refuses it, and the backfill gives the
    # original back. The catalog (FileReport.check) answers 409 for a
    # token that is neither the claim's nor the one whose lease ended last,
    # and refuses (400, nothing changed) a report that gives a key of its own
    # twice, a read that names no ffprobe or one that is not 7.1, a negative
    # captionStreams or sizeBytes, one of more than 64 windows, a window that
    # carries the catalog's own "unreadable", a span that is not {start, end}
    # of numbers, end after start, a gop that is no number from 0, a read
    # with neither windows nor an error, a duration not above 0, and a read
    # of frames that doesn't say sizeBytes. Anything else it keeps: a window
    # that is no window (a value of another type, an "at" outside 0 to 1, an
    # end before its start, captions in no frame) as a failed window, said and
    # never left out; a read none of whose windows decoded a frame, or whose
    # windows lie out of their places, miss one or decoded too few frames, as
    # a failed read, read again.
    def claim_captions(
        self, worker: str, limit: int = 1, timeout: float | None = None
    ) -> CaptionsClaim | None:
        """The captions backfill's next originals to read (POST
        /api/captions/claim). None when the catalog has no captions backfill
        (404, an older one; 503, its migration missing) or fails. timeout is
        how long its answer is waited for (the client's own when None): a
        claim not answered by then raises ClaimTimedOut."""
        try:
            resp = self._request(
                "POST",
                "/api/captions/claim",
                json={"worker": worker, "limit": limit},
                **({} if timeout is None else {"timeout": timeout}),
            )
        except httpx.TimeoutException as e:
            log.warning("captions.claim_timeout", error=str(e)[:200], timeout_s=timeout)
            raise ClaimTimedOut(str(e)) from e
        except Exception as e:
            log.warning("captions.claim_exception", error=str(e)[:200])
            return None
        if resp.status_code != 200:
            if resp.status_code != 404:
                log.warning("captions.claim_failed", status=resp.status_code, body=resp.text[:300])
            return None
        try:
            # NaN and Infinity refused: no number the catalog states
            body = json.loads(resp.content, parse_constant=_no_constant)
        except ValueError:
            return None
        if not isinstance(body, dict):
            return None
        return _captions_claim(body)

    def put_captions(self, job: CaptionsJob, original: CaptionsRead) -> bool:
        """Report what the read of a handed-out original found, or why it
        failed, with the claim's token (PUT /api/captions/sources/{sourceId}).
        Best-effort: False when the catalog did not take it (its lease ends,
        and the catalog hands the original out again). Raises ValueError,
        sending nothing, for a read that names no ffprobe."""
        return _taken(self.report_captions(job, original))

    def report_captions(self, job: CaptionsJob, original: CaptionsRead) -> int | None:
        """put_captions, answering the catalog's status (None: it could not
        be asked)."""
        return self._put_captions(job, {"token": job.token, "original": _captions_read(original)})

    def release_captions(self, job: CaptionsJob) -> bool:
        """Give an original the captions backfill handed out back unread.
        Best-effort."""
        return _taken(self._put_captions(job, {"token": job.token, "released": True}))

    def _put_captions(self, job: CaptionsJob, body: dict[str, Any]) -> int | None:
        try:
            # never a NaN or an Infinity on the wire
            content = json.dumps(body, allow_nan=False)
        except ValueError as e:
            log.warning("captions.put_not_finite", source_id=job.source_id, error=str(e)[:200])
            return None
        try:
            resp = self._request(
                "PUT",
                f"/api/captions/sources/{job.source_id}",
                content=content,
                extra_headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            log.warning("captions.put_exception", source_id=job.source_id, error=str(e)[:200])
            return None
        if resp.status_code >= 400:
            log.warning("captions.put_failed", source_id=job.source_id, status=resp.status_code,
                        body=resp.text[:300])
        return resp.status_code

    # ------------------------------------------------------ picture areas
    def picture_area(self, kind: str, item_id: str) -> dict[str, Any] | None:
        """The catalog's picture area of an item (kind "items") or an extra
        ("extras"), and whether it is of what the title plays now
        ("current"); None when it has none, or the catalog can't say (an
        older one without the endpoint answers 404 too)."""
        try:
            resp = self._request("GET", f"/api/picture/{kind}/{item_id}")
        except Exception as e:
            log.warning("picture.get_exception", item_id=item_id, error=str(e)[:200])
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    def put_picture_area(self, kind: str, item_id: str, body: dict[str, Any]) -> bool:
        """Report a measurement of an item's or an extra's picture area (the
        analyzer's picture.Measurement.report(), or {"released": true} for a
        claim given back unmeasured; with the claim's token when it is the
        backfill's). Best-effort: False when the catalog did not take it."""
        try:
            resp = self._request("PUT", f"/api/picture/{kind}/{item_id}", json=body)
        except Exception as e:
            log.warning("picture.put_exception", item_id=item_id, error=str(e)[:200])
            return False
        if resp.status_code >= 400:
            log.warning("picture.put_failed", item_id=item_id, status=resp.status_code,
                        body=resp.text[:300])
            return False
        return True

    def claim_picture_areas(
        self, worker: str, limit: int = 1, method: str | None = None
    ) -> dict[str, Any] | None:
        """The backfill's next titles to measure (POST /api/picture/claim):
        {"items": [{"kind", "id", "itemId", "type", "path", "durationMs",
        "token"}], "deadline"}, or no items and "idle" with
        "retryAfterSeconds". method names the analyzer's measurement: the
        catalog measures again an area of another. None when the catalog
        has no backfill (404, an older one) or fails."""
        body: dict[str, Any] = {"worker": worker, "limit": limit}
        if method:
            body["method"] = method
        try:
            resp = self._request("POST", "/api/picture/claim", json=body)
        except Exception as e:
            log.warning("picture.claim_exception", error=str(e)[:200])
            return None
        if resp.status_code != 200:
            if resp.status_code != 404:
                log.warning("picture.claim_failed", status=resp.status_code, body=resp.text[:300])
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    def get_item(self, item_id: str) -> ClaimedItem | None:
        """Fetch one item's FULL analyze detail with its primary playback
        path. Returns None when the item is unknown or has no primary
        asset. The endpoint returns
        {id,type,title,year,durationMs,path,seasonNumber,episodeNumber,
        seriesTitle,seriesTmdbId,movieTmdbId} — all of which we parse so
        the TIDB / chromaprint pipelines have the season/episode coords +
        TMDB ids they need. Used by both the Kafka event consumer (to
        resolve the item behind an `itemId`) and the packager flow."""
        resp = self._request("GET", f"/api/analyze/items/{item_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return self._item_from_json(resp.json())

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
        return [self._item_from_json(it) for it in items]

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


def _taken(status: int | None) -> bool:
    """Whether the catalog took a PUT that it answered status."""
    return status is not None and status < 400


def _captions_read(r: CaptionsRead) -> dict[str, Any]:
    """A read as a captions PUT carries it: its windows, the length its
    file gives and the picture's span they were placed over, or why it could
    not read the file; and the ffprobe that read it (ValueError when it
    names none)."""
    if not r.probe:
        raise ValueError("a read names the ffprobe that read it")
    # the size of the file read, where it was found (the catalog takes a read
    # of the file it handed out alone); what the header probe found, where it
    # ran: streams of captions of their own, and what is off the closed lists
    head: dict[str, Any] = {}
    if r.file_size is not None:
        head["sizeBytes"] = r.file_size
    if r.caption_streams is not None:
        head |= {"captionStreams": r.caption_streams, "offList": list(r.off_list)}
    # the longest group of pictures among its picture's first packets, where
    # the read listed them (the plan's gop: where its windows may lie, and
    # which picture is read whole, are of it), a failed read's too
    if r.gop is not None:
        head["gop"] = r.gop
    if r.unreadable or not r.windows or not r.duration or r.timed_out or r.interrupted:
        return {"error": r.error or "the captions could not be read", "ffprobe": r.probe, **head}
    # a picture whose windows can't tell captions at all (HEVC that showed
    # none): its windows as read, captionsKnown false and why
    known = {} if r.captions_known else {"captionsKnown": False, "reason": r.reason}
    return {"captionWindows": [_caption_window(w) for w in r.windows], "duration": r.duration,
            **({"span": r.span} if r.span is not None else {}),
            "ffprobe": r.probe, **head, **known}  # fmt: skip


def _no_constant(constant: str) -> None:
    """NaN, Infinity and -Infinity in an answer: refused, no number it states."""
    raise ValueError(f"{constant} is no number the catalog states")


def _caption_window(w: Any) -> dict[str, Any]:
    """A window of a read as a captions PUT carries it: one that failed
    decoded no frame, and shows no time and no captions, never "none"."""
    if w.error or not w.frames or w.start is None or w.end is None:
        return {"at": w.at, "start": None, "end": None, "frames": 0, "captions": None,
                "interlacedFrames": None, "repeatFrames": None, "keyframeBefore": None,
                "error": w.error or "the window decoded no frame"}  # fmt: skip
    whole = (
        {"whole": True, "packets": w.packets, "leading": w.leading}
        if getattr(w, "whole", False)
        else {}
    )
    return {"at": w.at, "start": w.start, "end": w.end, "frames": w.frames,
            "captions": bool(w.captions), "interlacedFrames": w.interlaced,
            "repeatFrames": w.repeats, "keyframeBefore": bool(w.keyframe_before),
            **whole}  # fmt: skip


def _captions_claim(body: dict[str, Any]) -> CaptionsClaim:
    """The CaptionsClaim of a captions claim's answer: the originals with a
    sourceId and a path, the rest counted and logged."""
    jobs: list[CaptionsJob] = []
    malformed = 0
    for it in body.get("items") or []:
        if not isinstance(it, dict) or not it.get("sourceId") or not it.get("path"):
            malformed += 1
            # its keys alone: an item carries the claim's token and a path
            keys = sorted(map(str, it)) if isinstance(it, dict) else type(it).__name__
            log.warning("captions.claim_malformed", keys=keys)
            continue
        # as the claim names it: the read reads the plan's count, and a claim
        # that names another is one it can't serve (it fails, said so)
        packets = it.get("packets")
        ms = it.get("durationMs")
        duration_ms = ms if type(ms) in (int, float) and math.isfinite(ms) and ms > 0 else None
        jobs.append(
            CaptionsJob(
                source_id=str(it["sourceId"]),
                path=str(it["path"]),
                token=str(it.get("token") or ""),
                item_id=str(it.get("itemId") or "") or None,
                packets=packets,
                duration_ms=duration_ms,
            )
        )
    retry: float | None
    try:
        retry = float(body["retryAfterSeconds"]) if body.get("retryAfterSeconds") else None
    except (TypeError, ValueError):
        retry = None
    idle = body.get("idle")
    return CaptionsClaim(
        jobs=jobs,
        deadline=_rfc3339(body.get("deadline")),
        idle=str(idle) if idle else None,
        retry_after_s=retry,
        malformed=malformed,
    )


def _rfc3339(v: Any) -> float | None:
    """An RFC 3339 time as seconds since the epoch; None when there is none."""
    if not isinstance(v, str) or not v:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
