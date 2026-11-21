#!/usr/bin/env python3
"""Export the corpus and tokenizer metrics from Drive-backed manifests.

This script is read-only with respect to the corpus.  It does not rerun collection,
cleaning, deduplication, assembly, or tokenizer training.  The resulting JSON is
small enough to keep in Git with the Phase 1 report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LANGUAGES = ("tel", "nep")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required Phase 1 artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def source_token_counts(manifest: dict[str, Any]) -> dict[str, int]:
    """Recover integer train-token counts from manifest fractions."""
    train_tokens = int(manifest["statistics"]["train"]["whitespace_tokens"])
    return {
        source_id: round(float(fraction) * train_tokens)
        for source_id, fraction in manifest.get("train_source_token_fractions", {}).items()
    }


def compact_preprocessing(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "run_id": manifest.get("run_id"),
        "language": manifest.get("language"),
        "created_at": manifest.get("created_at"),
        "training_status": manifest.get("training_status"),
        "statistics": manifest.get("statistics", {}),
        "source_train_tokens": source_token_counts(manifest),
        "manual_train_tokens": manifest.get("manual_train_tokens"),
        "manual_train_token_fraction": manifest.get("manual_train_token_fraction"),
        "manual_fraction_requirement_met": manifest.get("manual_fraction_requirement_met"),
        "manual_fraction_requirement_waived": manifest.get("manual_fraction_requirement_waived"),
        "minimum_train_tokens": manifest.get("minimum_train_tokens"),
        "minimum_train_tokens_met": manifest.get("minimum_train_tokens_met"),
        "sampling": manifest.get("sampling", {}),
        "deduplication": manifest.get("deduplication", {}),
        "rejection_counts": manifest.get("rejection_counts", {}),
        "output_shards": {
            split: len(paths)
            for split, paths in manifest.get("outputs", {}).items()
        },
        "language_id_model": manifest.get("language_id_model", {}),
        "registry_sha256": manifest.get("registry_sha256"),
    }


def compact_tokenizer(manifest: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for candidate in manifest.get("candidates", []):
        candidates.append(
            {
                "vocabulary_size": candidate.get("actual_vocabulary_size", candidate.get("vocabulary_size")),
                "metrics": candidate.get("metrics", {}),
            }
        )
    selected = dict(manifest.get("selected_metrics", {}))
    # Full frequency output is large and unnecessary for the report.  Preserve
    # the top 20 pieces plus all aggregate frequency measures.
    selected["top_tokens"] = selected.get("top_tokens", [])[:20]
    selected["tokenization_examples"] = selected.get("tokenization_examples", [])[:3]
    selected["byte_fallback_examples"] = selected.get("byte_fallback_examples", [])[:3]
    selected.pop("unknown_examples", None)
    return {
        "schema_version": manifest.get("schema_version"),
        "language": manifest.get("language"),
        "created_at": manifest.get("created_at"),
        "selected_vocabulary_size": manifest.get("selected_vocabulary_size"),
        "config": manifest.get("config", {}),
        "preprocessing_gate": manifest.get("preprocessing_gate", {}),
        "training_sample": manifest.get("training_sample", {}),
        "selected_metrics": selected,
        "candidates": candidates,
        "model_path": manifest.get("model_path"),
        "vocab_path": manifest.get("vocab_path"),
        "model_sha256": manifest.get("model_sha256"),
        "vocab_sha256": manifest.get("vocab_sha256"),
    }


def export(data_root: Path, run_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "languages": {},
    }
    for language in LANGUAGES:
        preprocessing = read_json(
            data_root / "audit" / run_id / "manifests" / f"{language}.json"
        )
        tokenizer = read_json(
            data_root / "tokenizers" / run_id / language / "tokenizer_manifest.json"
        )
        result["languages"][language] = {
            "preprocessing": compact_preprocessing(preprocessing),
            "tokenizer": compact_tokenizer(tokenizer),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-id", default="draft-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = export(args.data_root.resolve(), args.run_id)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output.resolve()}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
