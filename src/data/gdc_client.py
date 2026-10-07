"""Small, auditable client for official NCI GDC REST endpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils import md5_file, utc_now


LOGGER = logging.getLogger(__name__)


class GDCError(RuntimeError):
    """Raised when a GDC request or integrity check fails."""


class GDCClient:
    """HTTP client with bounded retries and checksum-verified downloads."""

    def __init__(
        self,
        base_url: str = "https://api.gdc.cancer.gov",
        timeout_seconds: int = 120,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({"User-Agent": "nisq-vqc-breast-cancer/0.1"})

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def get_json(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        response = self.session.get(self._url(endpoint), timeout=self.timeout_seconds, **kwargs)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise GDCError(f"GDC GET failed ({response.status_code}) for {response.url}") from exc
        try:
            return response.json()
        except requests.JSONDecodeError as exc:
            raise GDCError(f"GDC returned non-JSON content for {response.url}") from exc

    def post_json(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            self._url(endpoint), json=dict(payload), timeout=self.timeout_seconds
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = response.text[:1000]
            raise GDCError(
                f"GDC POST failed ({response.status_code}) for {response.url}: {message}"
            ) from exc
        try:
            result = response.json()
        except requests.JSONDecodeError as exc:
            raise GDCError(f"GDC returned non-JSON content for {response.url}") from exc
        warnings = result.get("warnings") or {}
        if warnings:
            LOGGER.warning("GDC warnings from %s: %s", endpoint, warnings)
        return result

    def status(self, endpoint: str = "/status") -> dict[str, Any]:
        result = self.get_json(endpoint)
        if result.get("status") != "OK":
            raise GDCError(f"GDC status is not OK: {result}")
        return result

    def search_all(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
        page_size: int = 2000,
    ) -> dict[str, Any]:
        """Retrieve all search hits while preserving the response envelope."""

        offset = 0
        all_hits: list[dict[str, Any]] = []
        first_response: dict[str, Any] | None = None
        while True:
            page_payload = dict(payload)
            page_payload.update({"from": offset, "size": page_size})
            response = self.post_json(endpoint, page_payload)
            first_response = first_response or response
            data = response.get("data", {})
            hits = data.get("hits", [])
            all_hits.extend(hits)
            total = int(data.get("pagination", {}).get("total", len(all_hits)))
            LOGGER.info("Fetched %d/%d records from %s", len(all_hits), total, endpoint)
            if not hits or len(all_hits) >= total:
                break
            offset += len(hits)

        assert first_response is not None
        all_hits.sort(key=lambda hit: str(hit.get("file_id") or hit.get("case_id") or hit.get("id") or ""))
        first_response["data"]["hits"] = all_hits
        first_response["data"]["pagination"].update(
            {"count": len(all_hits), "from": 0, "size": page_size, "page": 1}
        )
        return first_response

    def download_file(
        self,
        file_id: str,
        destination: Path,
        expected_md5: str,
        expected_size: int,
        endpoint: str = "/data",
        chunk_size: int = 1024 * 1024,
    ) -> dict[str, Any]:
        """Download to a temporary file, validate, then atomically publish it.

        Existing files are never overwritten. They are reused only when size and
        GDC MD5 both match.
        """

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            self._validate_file(destination, expected_md5, expected_size)
            return {
                "file_id": file_id,
                "path": str(destination),
                "downloaded_at": None,
                "reused": True,
                "size": destination.stat().st_size,
                "md5": expected_md5,
                "endpoint": f"{self.base_url}/{endpoint.lstrip('/')}/{file_id}",
            }

        temporary = destination.with_suffix(destination.suffix + ".part")
        if temporary.exists():
            raise FileExistsError(
                f"Partial download already exists: {temporary}. Inspect and remove it explicitly."
            )

        url = f"{self._url(endpoint)}/{file_id}"
        LOGGER.info("Downloading %s -> %s", file_id, destination)
        try:
            with self.session.get(url, stream=True, timeout=self.timeout_seconds) as response:
                response.raise_for_status()
                with temporary.open("xb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            self._validate_file(temporary, expected_md5, expected_size)
            os.replace(temporary, destination)
        except Exception:
            # Keep a partial file for diagnosis; it is never accepted as raw data.
            raise

        return {
            "file_id": file_id,
            "path": str(destination),
            "downloaded_at": utc_now(),
            "reused": False,
            "size": destination.stat().st_size,
            "md5": expected_md5,
            "endpoint": url,
        }

    @staticmethod
    def _validate_file(path: Path, expected_md5: str, expected_size: int) -> None:
        actual_size = path.stat().st_size
        if actual_size != int(expected_size):
            raise GDCError(
                f"Size mismatch for {path}: expected {expected_size}, got {actual_size}"
            )
        actual_md5 = md5_file(path)
        if actual_md5.lower() != expected_md5.lower():
            raise GDCError(
                f"MD5 mismatch for {path}: expected {expected_md5}, got {actual_md5}"
            )


def chunks(values: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
