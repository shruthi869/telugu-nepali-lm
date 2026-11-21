"""Shared paths, hashing, durable I/O, and progress helpers for Phase 2."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable


LANGUAGES = ("tel", "nep")
LANGUAGE_NAMES = {"tel": "Telugu", "nep": "Nepali"}
MODEL_LABELS = {"tel": "H", "nep": "L"}
DEFAULT_RUN_ID = "draft-v1"
SPLITS = ("train", "validation", "test")

Progress = Callable[[str], None]


class LMError(RuntimeError):
    """A clear, user-actionable language-model pipeline error."""


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def emit(message: str) -> None:
    print(f"[{utcnow()}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def tokenizer_dir(data_root: Path, run_id: str, language: str) -> Path:
    return data_root / "tokenizers" / run_id / language


def tokenizer_model_path(data_root: Path, run_id: str, language: str) -> Path:
    return tokenizer_dir(data_root, run_id, language) / "tokenizer.model"


def manifest_path(data_root: Path, run_id: str, language: str) -> Path:
    return data_root / "audit" / run_id / "manifests" / f"{language}.json"


def tokenized_dir(data_root: Path, run_id: str, language: str) -> Path:
    return data_root / "tokenized" / run_id / language


def packed_bin_path(data_root: Path, run_id: str, language: str, split: str) -> Path:
    return tokenized_dir(data_root, run_id, language) / f"{split}.bin"


def packed_meta_path(data_root: Path, run_id: str, language: str) -> Path:
    return tokenized_dir(data_root, run_id, language) / "meta.json"


NOPOS_SLUG = "nopos"
POSITIONAL_NONE = "none"
POSITIONAL_LEARNED = "learned_absolute"


def positional_ablation(positional_embeddings: str) -> str | None:
    """Subdirectory for the optional no-position ablation. None means the standard run."""
    return NOPOS_SLUG if positional_embeddings == POSITIONAL_NONE else None


def model_dir(data_root: Path, run_id: str, language: str, ablation: str | None = None) -> Path:
    base = data_root / "models" / run_id / language
    return base / ablation if ablation else base


def checkpoint_dir(data_root: Path, run_id: str, language: str, ablation: str | None = None) -> Path:
    return model_dir(data_root, run_id, language, ablation=ablation) / "checkpoints"


def log_path(data_root: Path, run_id: str, language: str, ablation: str | None = None) -> Path:
    return model_dir(data_root, run_id, language, ablation=ablation) / "logs" / "train.jsonl"


def language_config_path(language: str, root: Path | None = None, *, filename: str = "model.json") -> Path:
    return (root or repository_root()) / language / "configs" / filename


def local_work_root(run_id: str, language: str, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit / run_id / language
    if Path("/content").is_dir():
        return Path("/content/lma_lm_work") / run_id / language
    return Path(tempfile.gettempdir()) / "lma_lm_work" / run_id / language


def copy_to_local_ssd(source: Path, destination: Path, progress: Progress = emit) -> Path:
    """Copy a packed bin onto Colab local SSD when that is faster than Drive FUSE."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        progress(f"pack-cache: reusing local copy {destination}")
        return destination
    progress(f"pack-cache: copying {source} -> {destination}")
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def require_language(language: str) -> str:
    if language not in LANGUAGES:
        raise LMError(f"unsupported language {language!r}; expected tel or nep")
    return language
