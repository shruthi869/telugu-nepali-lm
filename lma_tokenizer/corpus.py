"""Streaming corpus access and deterministic tokenizer-training samples."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator

from .common import atomic_write_json, sha256_file, sha256_json


Progress = Callable[[str], None]


def iter_split_records(data_root: Path, manifest: dict[str, Any], split: str) -> Iterator[dict[str, Any]]:
    """Stream final JSONL.gz documents from one manifest split."""
    for relative in manifest["outputs"][split]:
        path = data_root / relative
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSONL in {path}:{line_number}: {exc}") from exc
                if isinstance(record, dict) and isinstance(record.get("cleaned_text"), str):
                    yield record


def text_lines(text: str, max_characters: int) -> Iterator[str]:
    """Produce bounded one-line SentencePiece inputs without changing characters."""
    remaining = " ".join(text.split())
    while remaining:
        if len(remaining) <= max_characters:
            yield remaining
            return
        boundary = remaining.rfind(" ", 0, max_characters + 1)
        if boundary < max_characters // 2:
            boundary = max_characters
        piece = remaining[:boundary].strip()
        if piece:
            yield piece
        remaining = remaining[boundary:].strip()


def _selected(identifier: str, probability: float, seed: str) -> bool:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < probability


def _measure_train_whitespace_tokens(
    data_root: Path,
    run_id: str,
    language: str,
    manifest: dict[str, Any],
    directory: Path,
    progress: Progress,
) -> dict[str, Any]:
    """Measure and cache the count used by tokenizer sample selection.

    Preprocessing's historical ``whitespace_tokens`` manifest field is based on
    Unicode word-regex fragments. Indic combining marks can make that substantially
    larger than ``str.split()`` words, which are the denominator used by the
    tokenizer sample and fertility metrics.  Measuring once avoids that mismatch
    and is durable across Colab restarts.
    """
    payload = {
        "schema_version": 1,
        "manifest_sha256": manifest["_manifest_sha256"],
        "language": language,
        "count_definition": "len(cleaned_text.split())",
    }
    fingerprint = sha256_json(payload)
    path = directory / f"train-whitespace-count-{fingerprint[:16]}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                progress(
                    f"tokenizer-sample: {language}: reusing measured train population of "
                    f"{cached['whitespace_tokens']:,} whitespace tokens"
                )
                return cached
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    documents = whitespace_tokens = 0
    for documents, record in enumerate(iter_split_records(data_root, manifest, "train"), 1):
        whitespace_tokens += len(record["cleaned_text"].split())
        if documents == 1 or documents % 50_000 == 0:
            progress(
                f"tokenizer-sample-calibrate: {language}: {documents:,} documents; "
                f"{whitespace_tokens:,} whitespace tokens"
            )
    measured = {
        **payload,
        "fingerprint": fingerprint,
        "documents": documents,
        "whitespace_tokens": whitespace_tokens,
    }
    atomic_write_json(path, measured)
    progress(
        f"tokenizer-sample-calibrate: {language}: complete; {documents:,} documents; "
        f"{whitespace_tokens:,} whitespace tokens"
    )
    return measured


def build_training_sample(
    data_root: Path,
    run_id: str,
    language: str,
    manifest: dict[str, Any],
    config: dict[str, Any],
    progress: Progress = print,
) -> tuple[Path, dict[str, Any]]:
    """Build or reuse a deterministic, compressed sample of the final train split."""
    target_tokens = int(config["training_sample_whitespace_tokens"])
    directory = data_root / "audit" / run_id / "tokenizer" / language
    directory.mkdir(parents=True, exist_ok=True)
    measured = _measure_train_whitespace_tokens(
        data_root, run_id, language, manifest, directory, progress
    )
    train_tokens = int(measured["whitespace_tokens"])
    if train_tokens < target_tokens * 0.80:
        raise RuntimeError(
            f"complete {language} train split contains only {train_tokens:,} whitespace tokens; "
            f"cannot build the requested {target_tokens:,}-token tokenizer sample"
        )
    # A 5% cushion makes an under-80% sample vanishingly unlikely on a large
    # corpus. Tiny fixtures/corpora are copied whole to avoid document-level
    # Bernoulli variance overwhelming the requested token count.
    probability = (
        1.0
        if int(measured["documents"]) < 1_000
        else min(1.0, target_tokens * 1.05 / max(train_tokens, 1))
    )
    fingerprint_payload = {
        "schema_version": 2,
        "manifest_sha256": manifest["_manifest_sha256"],
        "language": language,
        "selection_seed": f"{run_id}:sentencepiece-training-sample-v1",
        "probability": probability,
        "measured_train_documents": int(measured["documents"]),
        "measured_train_whitespace_tokens": train_tokens,
        "max_training_line_characters": int(config["max_training_line_characters"]),
    }
    fingerprint = sha256_json(fingerprint_payload)
    sample_path = directory / f"training-sample-{fingerprint[:16]}.txt.gz"
    metadata_path = sample_path.with_suffix(".json")
    if sample_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") == fingerprint and metadata.get("sha256") == sha256_file(sample_path):
                progress(
                    f"tokenizer-sample: {language}: reusing {metadata['whitespace_tokens']:,} tokens "
                    f"from {sample_path.relative_to(data_root)}"
                )
                return sample_path, metadata
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{sample_path.name}.", suffix=".tmp", dir=directory)
    temporary = Path(temporary_name)
    documents = lines = whitespace_tokens = characters = 0
    seed = fingerprint_payload["selection_seed"]
    max_characters = int(config["max_training_line_characters"])
    try:
        with os.fdopen(descriptor, "wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
                for seen, record in enumerate(iter_split_records(data_root, manifest, "train"), 1):
                    identifier = str(record.get("document_id", seen))
                    if not _selected(identifier, probability, seed):
                        continue
                    documents += 1
                    for line in text_lines(record["cleaned_text"], max_characters):
                        compressed.write(line.encode("utf-8") + b"\n")
                        lines += 1
                        whitespace_tokens += len(line.split())
                        characters += len(line)
                    if documents == 1 or documents % 10_000 == 0:
                        progress(
                            f"tokenizer-sample: {language}: {seen:,} documents scanned; "
                            f"{documents:,} selected; {whitespace_tokens:,} tokens written"
                        )
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(temporary, sample_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    metadata = {
        **fingerprint_payload,
        "fingerprint": fingerprint,
        "sample_path": str(sample_path.relative_to(data_root)),
        "sha256": sha256_file(sample_path),
        "documents": documents,
        "lines": lines,
        "whitespace_tokens": whitespace_tokens,
        "characters": characters,
        "target_whitespace_tokens": target_tokens,
    }
    if whitespace_tokens < target_tokens * 0.80:
        sample_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"deterministic {language} tokenizer sample produced only {whitespace_tokens:,} tokens; "
            f"expected at least {int(target_tokens * 0.80):,}"
        )
    atomic_write_json(metadata_path, metadata)
    progress(
        f"tokenizer-sample: {language}: complete; {whitespace_tokens:,} tokens in "
        f"{lines:,} bounded lines"
    )
    return sample_path, metadata


def materialize_sample(sample_path: Path, local_root: Path, expected_sha256: str) -> Path:
    """Inflate the persistent Drive sample once onto fast local disk."""
    local_root.mkdir(parents=True, exist_ok=True)
    target = local_root / f"{sample_path.stem}-{expected_sha256[:12]}.txt"
    if target.exists() and target.stat().st_size > 0:
        return target
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with gzip.open(sample_path, "rb") as source, temporary.open("wb") as output:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            output.write(block)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)
    return target
