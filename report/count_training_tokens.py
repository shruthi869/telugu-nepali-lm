#!/usr/bin/env python3
"""Exactly count selected-tokenizer tokens in each final training split.

The preprocessing manifest's historical ``whitespace_tokens`` field is based on
``re.compile(r"\\w+")`` and is not the SentencePiece token count required by the
project brief.  This read-only scan encodes the final training split with the
selected language-specific tokenizer.  It checkpoints after every gzip shard, so
an interrupted Colab session repeats at most one shard.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def count_shard(processor: Any, path: Path, batch_size: int, threads: int = 1) -> dict[str, Any]:
    documents = sentencepiece_tokens = utf8_bytes = characters = whitespace_words = 0
    batch: list[str] = []
    batch_sources: list[tuple[str, str]] = []
    by_source: dict[str, dict[str, int | str]] = {}

    def consume() -> None:
        nonlocal sentencepiece_tokens
        if not batch:
            return
        encoded = processor.encode(batch, out_type=int, num_threads=threads)
        for token_ids, (source_id, category) in zip(encoded, batch_sources):
            token_count = len(token_ids)
            sentencepiece_tokens += token_count
            source = by_source.setdefault(source_id, {
                "course_collection_category": category,
                "documents": 0,
                "sentencepiece_tokens": 0,
            })
            source["documents"] = int(source["documents"]) + 1
            source["sentencepiece_tokens"] = int(source["sentencepiece_tokens"]) + token_count
        batch.clear()
        batch_sources.clear()

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record["cleaned_text"]
            documents += 1
            characters += len(text)
            utf8_bytes += len(text.encode("utf-8"))
            whitespace_words += len(text.split())
            batch.append(text)
            batch_sources.append((
                str(record.get("source_id", "unknown")),
                str(record.get("course_collection_category", "downloaded")),
            ))
            if len(batch) >= batch_size:
                consume()
    consume()
    return {
        "documents": documents,
        "sentencepiece_tokens": sentencepiece_tokens,
        "characters": characters,
        "utf8_bytes": utf8_bytes,
        "whitespace_words": whitespace_words,
        "sources": by_source,
    }


def count_language(data_root: Path, run_id: str, language: str, batch_size: int, minimum_tokens: int = 525_000_000, threads: int = 1) -> dict[str, Any]:
    try:
        import sentencepiece as sentencepiece
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required; install requirements-colab.txt") from exc

    manifest_path = data_root / "audit" / run_id / "manifests" / f"{language}.json"
    tokenizer_manifest_path = (
        data_root / "tokenizers" / run_id / language / "tokenizer_manifest.json"
    )
    manifest = read_json(manifest_path)
    tokenizer_manifest = read_json(tokenizer_manifest_path)
    model_path = data_root / tokenizer_manifest["model_path"]
    if not model_path.is_file():
        raise FileNotFoundError(f"missing selected tokenizer: {model_path}")
    model_sha256 = sha256_file(model_path)
    if model_sha256 != tokenizer_manifest.get("model_sha256"):
        raise RuntimeError(f"selected {language} tokenizer hash does not match its manifest")

    train_shards = list(manifest.get("outputs", {}).get("train", []))
    if not train_shards:
        raise RuntimeError(f"{language} preprocessing manifest has no train shards")
    shard_hashes: dict[str, str] = {}
    for relative in train_shards:
        path = data_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty train shard: {path}")
        shard_hashes[relative] = sha256_file(path)
    manifest_sha256 = sha256_file(manifest_path)
    fingerprint_payload = {
        "schema_version": 2,
        "language": language,
        "preprocessing_manifest_sha256": manifest_sha256,
        "tokenizer_model_sha256": model_sha256,
        "train_shards": train_shards,
        "train_shard_sha256": shard_hashes,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    checkpoint_path = (
        data_root / "audit" / run_id / "exact-token-counts" / f"{language}.json"
    )
    completed: dict[str, dict[str, int]] = {}
    if checkpoint_path.is_file():
        previous = read_json(checkpoint_path)
        if previous.get("fingerprint") == fingerprint:
            completed = {
                path: value for path, value in dict(previous.get("completed_shards", {})).items()
                if isinstance(value, dict) and value.get("shard_sha256")
            }
            print(
                f"[{language}] resuming with {len(completed):,}/{len(train_shards):,} shards complete",
                flush=True,
            )

    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    for index, relative in enumerate(train_shards, 1):
        if relative in completed:
            print(f"[{language}] shard {index}/{len(train_shards)} already counted", flush=True)
            continue
        path = data_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty train shard: {path}")
        print(f"[{language}] counting shard {index}/{len(train_shards)}: {path.name}", flush=True)
        completed[relative] = {
            "shard_sha256": shard_hashes[relative],
            **count_shard(processor, path, batch_size, threads),
        }
        totals = {
            key: sum(value[key] for value in completed.values())
            for key in (
                "documents",
                "sentencepiece_tokens",
                "characters",
                "utf8_bytes",
                "whitespace_words",
            )
        }
        sources: dict[str, dict[str, int | str]] = {}
        for shard in completed.values():
            for source_id, values in shard.get("sources", {}).items():
                target = sources.setdefault(source_id, {
                    "course_collection_category": values["course_collection_category"],
                    "documents": 0,
                    "sentencepiece_tokens": 0,
                })
                if target["course_collection_category"] != values["course_collection_category"]:
                    raise RuntimeError(f"source {source_id} has conflicting course categories")
                target["documents"] = int(target["documents"]) + int(values["documents"])
                target["sentencepiece_tokens"] = int(target["sentencepiece_tokens"]) + int(values["sentencepiece_tokens"])
        checkpoint = {
            **fingerprint_payload,
            "fingerprint": fingerprint,
            "status": "complete" if len(completed) == len(train_shards) else "in_progress",
            "completed_shards": completed,
            "completed_shard_count": len(completed),
            "total_shard_count": len(train_shards),
            "totals": totals,
            "per_source": sources,
            "manual_sentencepiece_tokens": sum(
                int(value["sentencepiece_tokens"])
                for value in sources.values()
                if value["course_collection_category"] == "manual"
            ),
            "minimum_sentencepiece_tokens": minimum_tokens,
            "minimum_sentencepiece_tokens_met": totals["sentencepiece_tokens"] >= minimum_tokens,
        }
        atomic_json(checkpoint_path, checkpoint)
        print(
            f"[{language}] durable: {totals['documents']:,} documents; "
            f"{totals['sentencepiece_tokens']:,} SentencePiece tokens",
            flush=True,
        )

    result = read_json(checkpoint_path)
    if result.get("status") != "complete":
        raise RuntimeError(f"{language} counter ended without a complete checkpoint")
    source_total = sum(int(value["sentencepiece_tokens"]) for value in result.get("per_source", {}).values())
    if source_total != int(result["totals"]["sentencepiece_tokens"]):
        raise RuntimeError(f"{language} per-source token totals do not sum to the final total")
    result["minimum_sentencepiece_tokens"] = minimum_tokens
    result["minimum_sentencepiece_tokens_met"] = int(result["totals"]["sentencepiece_tokens"]) >= minimum_tokens
    if not result["minimum_sentencepiece_tokens_met"]:
        atomic_json(checkpoint_path, result)
        raise RuntimeError(
            f"{language} has {result['totals']['sentencepiece_tokens']:,} exact tokens; "
            f"requires at least {minimum_tokens:,}"
        )
    print(
        f"[{language}] COMPLETE: {result['totals']['sentencepiece_tokens']:,} training tokens",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-id", default="draft-v1")
    parser.add_argument("--language", choices=("all", "tel", "nep"), default="all")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--threads", type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="SentencePiece encoder threads; languages remain sequential",
    )
    parser.add_argument("--minimum-tokens", type=int, default=525_000_000)
    args = parser.parse_args()

    languages = ("tel", "nep") if args.language == "all" else (args.language,)
    if args.minimum_tokens < 1 or args.batch_size < 1 or args.threads < 1:
        raise SystemExit("--minimum-tokens, --batch-size, and --threads must be positive")
    # Train/verify one language at a time: this avoids competing SQLite, Drive,
    # and local-disk traffic in the overnight Colab recovery run.
    results: dict[str, Any] = {}
    for language in languages:
        results[language] = count_language(
            args.data_root.resolve(), args.run_id, language,
            args.batch_size, args.minimum_tokens, args.threads,
        )

    print("\nEXACT FINAL TRAINING TOKEN COUNTS", flush=True)
    for language in languages:
        totals = results[language]["totals"]
        print(
            f"{language.upper()}: {totals['sentencepiece_tokens']:,} SentencePiece tokens; "
            f"{totals['whitespace_words']:,} whitespace words; "
            f"{totals['documents']:,} documents",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
