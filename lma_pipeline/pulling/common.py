#!/usr/bin/env python3
"""Resumable, policy-aware corpus collector.

The collector deliberately keeps collection separate from text extraction.  It stores
downloaded source files unchanged and stores crawled HTTP responses as immutable
compressed JSONL records.  Its SQLite ledger is the authority for what still needs
to be collected, so rerunning this file after an interruption is safe.

Only the Python standard library is required.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import email.utils
import hashlib
import os
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


USER_AGENT = "LMA-Corpus-Collector/1.0"
MIN_FREE_BYTES = 1024**3
TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
CHECKPOINT_BYTES = 16 * 1024 * 1024
PROGRESS_INTERVAL_SECONDS = 0.5


class CollectorError(RuntimeError):
    """A collection error that should be written to the job ledger."""


class TransientError(CollectorError):
    """An error that may succeed on a later collector run."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclasses.dataclass(frozen=True)
class Response:
    url: str
    status: int
    headers: dict[str, str]
    body: bytes


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def unix_now() -> float:
    return time.time()


def canonical_url(url: str) -> str:
    """Normalize a URL enough to make sitemap duplicates share one ledger job."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise CollectorError(f"unsupported URL: {url}")
    host = parsed.hostname.lower() if parsed.hostname else ""
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    path = parsed.path or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def safe_filename(url: str, fallback: str = "asset") -> str:
    name = Path(urlsplit(url).path).name
    return name if name and name not in {".", ".."} else fallback


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    with contextlib.suppress(ValueError):
        return max(0.0, float(value))
    with contextlib.suppress(TypeError, ValueError):
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - unix_now())
    return None


def fsync_directory(directory: Path) -> None:
    """Persist a rename on POSIX; Windows cannot open directories for fsync."""
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def human_size(value: int | None) -> str:
    """Format a byte count for a compact terminal progress display."""
    if value is None:
        return "?"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{amount:.1f} TiB"


def default_data_root() -> Path:
    configured = os.environ.get("LMA_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    colab = Path("/content/drive/MyDrive/lma_data")
    if colab.exists():
        return colab
    raise CollectorError(
        "No data root selected. Set LMA_DATA_ROOT or mount Google Drive at "
        "/content/drive/MyDrive and use --data-root."
    )


def ensure_space(data_root: Path, minimum: int = MIN_FREE_BYTES) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(data_root).free
    if free < minimum:
        raise CollectorError(
            f"{data_root} has {free / 1024**3:.1f} GiB free; at least "
            f"{minimum / 1024**3:.1f} GiB is required before collecting."
        )
