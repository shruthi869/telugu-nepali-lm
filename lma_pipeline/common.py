"""Shared preprocessing constants, durable I/O, and progress reporting."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

try:
    import xxhash as _xxhash  # type: ignore
except ImportError:  # Developer/test fallback; Colab installs the pinned wheel.
    _xxhash = None


MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODEL_NAME = "fastText lid.176.ftz"
MODEL_VERSION = "lid.176.ftz (official frozen model)"
DEFAULT_SEED = "lma-preprocess-v1"
SCRIPT_RANGES = {"tel": (0x0C00, 0x0C7F), "nep": (0x0900, 0x097F)}
LANGUAGE_CODES = {"tel": "te", "nep": "ne"}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
ID_RE = re.compile(r"(?<!\d)\d[\d -]{10,14}\d(?!\d)")
SENTENCE_RE = re.compile(r"[^.!?।॥\n]+[.!?।॥]?", re.UNICODE)
MINHASH_SIZE = 128
LSH_BANDS = 16
LSH_ROWS = 8
DEFAULT_MAX_SHINGLES = 5_000
# A giant shared LSH bucket is boilerplate, not a useful set of candidate pairs.
# It is already caught by quality filtering; refusing to expand it prevents an
# accidental O(n²) explosion if a source contains pathological boilerplate.
MAX_LSH_BUCKET_DOCUMENTS = 500
MAX_CANDIDATE_PAIRS = 5_000_000
MAX_CACHED_VERIFICATION_SHINGLES = 200_000
LSH_PARTITIONS = 128
SIGNATURE_BATCH_SIZE = 256
SIGNATURE_CHECKPOINT_DOCUMENTS = 50_000
SIGNATURE_PROGRESS_SCHEMA = 1
SIGNATURE_CHUNK_MAGIC = b"LMA-LSH1"
PARTITIONED_LSH_ALGORITHM = "xxhash64-densified-one-permutation-partitioned-lsh-v2"


class PreprocessError(RuntimeError):
    """A clear, user-actionable preprocessing error."""


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_data_root() -> Path:
    configured = os.environ.get("LMA_DATA_ROOT")
    if configured:
        return Path(configured).expanduser()
    colab = Path("/content/drive/MyDrive/lma_data")
    if colab.exists():
        return colab
    raise PreprocessError("Set LMA_DATA_ROOT, pass --data-root, or mount Google Drive at /content/drive/MyDrive.")


def language_name(language: str) -> str:
    if language not in LANGUAGE_CODES:
        raise PreprocessError(f"unsupported language {language!r}; expected tel or nep")
    return language


class ProgressReporter:
    """Line-oriented progress that remains visible in Colab and survives restarts."""

    def __init__(self, root: Path, run_id: str, every_records: int = 10_000, every_seconds: float = 30.0):
        self.path = root / "audit" / run_id / "preprocess_progress.log"
        self._persistent_enabled = True
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._persistent_enabled = False
            print(f"[{utcnow()}] progress-log warning: {exc}; console progress remains active", file=sys.stderr, flush=True)
        self.every_records = every_records
        self.every_seconds = every_seconds
        self._last_count: dict[tuple[str, str], int] = {}
        self._last_time: dict[tuple[str, str], float] = {}

    def emit(self, message: str) -> None:
        line = f"[{utcnow()}] {message}"
        print(line, flush=True)
        # Console output is the live progress channel.  The Drive copy is useful
        # after a disconnect, but must never terminate a multi-hour run if the
        # FUSE mount momentarily refuses a flush.  Output shards and SQLite state
        # keep their own atomic/durable write rules.
        if not self._persistent_enabled:
            return
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
        except OSError as exc:
            self._persistent_enabled = False
            print(f"[{utcnow()}] progress-log warning: {exc}; continuing without persistent log entry", file=sys.stderr, flush=True)

    def tick(self, stage: str, label: str, records: int, extra: str = "") -> None:
        key = (stage, label)
        now = time.monotonic()
        if records - self._last_count.get(key, 0) < self.every_records and now - self._last_time.get(key, 0.0) < self.every_seconds:
            return
        self._last_count[key] = records
        self._last_time[key] = now
        self.emit(f"{stage}: {label}: {records:,} records{extra}")


def fsync_directory(directory: Path) -> None:
    """Persist a rename on POSIX; Windows cannot open directories for fsync."""
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_gzip_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write a deterministic, durable gzip JSONL shard and then atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    count = 0
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                for record in records:
                    zipped.write((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
                    count += 1
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def atomic_write_text(path: Path, value: str) -> None:
    """Durably replace a small text file without leaving a disk-filling temp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_concatenate(path: Path, parts: Iterable[Path]) -> None:
    """Durably publish the bytewise concatenation of already-durable parts.

    Concatenated gzip members are valid gzip streams, so this lets long-running
    stages checkpoint independent gzip JSONL blocks without recompressing them
    when the authoritative shard is finalized.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def valid_gzip_jsonl(path: Path) -> bool:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for _ in handle:
                pass
        return True
    except (OSError, UnicodeError):
        return False


def read_jsonl_gzip(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_clean_shards(
    accepted_path: Path, rejected_path: Path, records: Iterable[dict[str, Any]], clean_one: Callable[[dict[str, Any]], tuple[dict[str, Any] | None, dict[str, Any] | None]]
) -> int:
    return atomic_clean_pairs(accepted_path, rejected_path, (clean_one(record) for record in records))


def atomic_clean_pairs(
    accepted_path: Path,
    rejected_path: Path,
    pairs: Iterable[tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> int:
    """Stream a candidate shard to accepted/rejected durable outputs together."""
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_tmp = accepted_path.with_name(accepted_path.name + f".{uuid.uuid4().hex}.tmp")
    rejected_tmp = rejected_path.with_name(rejected_path.name + f".{uuid.uuid4().hex}.tmp")
    accepted_count = 0
    try:
        with accepted_tmp.open("wb") as accepted_raw, rejected_tmp.open("wb") as rejected_raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=accepted_raw, mtime=0) as accepted_gz, gzip.GzipFile(filename="", mode="wb", fileobj=rejected_raw, mtime=0) as rejected_gz:
                for accepted, rejected in pairs:
                    if accepted is not None:
                        accepted_gz.write((json.dumps(accepted, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
                        accepted_count += 1
                    if rejected is not None:
                        rejected_gz.write((json.dumps(rejected, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            accepted_raw.flush()
            rejected_raw.flush()
            os.fsync(accepted_raw.fileno())
            os.fsync(rejected_raw.fileno())
        os.replace(accepted_tmp, accepted_path)
        os.replace(rejected_tmp, rejected_path)
        for directory in {accepted_path.parent, rejected_path.parent}:
            fsync_directory(directory)
    except Exception:
        accepted_tmp.unlink(missing_ok=True)
        rejected_tmp.unlink(missing_ok=True)
        raise
    return accepted_count
