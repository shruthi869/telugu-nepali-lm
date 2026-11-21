"""Strict gate between preprocessing and tokenizer training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import sha256_file


class PreprocessingNotComplete(RuntimeError):
    """Raised when tokenizer training would consume stale or incomplete data."""


def _safe_output(data_root: Path, relative: str) -> Path:
    root = data_root.resolve()
    output = (root / relative).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise PreprocessingNotComplete(f"manifest output escapes data root: {relative}") from exc
    return output


def validate_preprocessing(
    data_root: Path,
    run_id: str,
    language: str,
    registry_path: Path,
    minimum_train_tokens: int,
    allow_low_manual_fraction: bool = False,
) -> dict[str, Any]:
    """Require a complete, current manifest and all three split outputs.

    ``allow_low_manual_fraction`` is an explicit provenance-preserving override:
    it permits a corpus that is draft only because it misses the manual-token
    target. It never bypasses the minimum-token, registry, or output checks.
    """
    manifest_path = data_root / "audit" / run_id / "manifests" / f"{language}.json"
    if not manifest_path.exists():
        raise PreprocessingNotComplete(f"missing preprocessing manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessingNotComplete(f"cannot read preprocessing manifest {manifest_path}: {exc}") from exc
    errors: list[str] = []
    if manifest.get("run_id") != run_id or manifest.get("language") != language:
        errors.append("manifest run/language does not match the tokenizer request")
    if manifest.get("training_status") != "final" and not allow_low_manual_fraction:
        errors.append("training_status is not final")
    if not manifest.get("minimum_train_tokens_met"):
        errors.append("preprocessing minimum-token requirement is not met")
    if not manifest.get("manual_fraction_requirement_met") and not allow_low_manual_fraction:
        errors.append("the 20% Wiki/manual-token requirement is not met")
    if manifest.get("manual_fraction_requirement_waived") and not allow_low_manual_fraction:
        errors.append("the manual-token requirement was waived rather than satisfied")
    if manifest.get("registry_sha256") != sha256_file(registry_path):
        errors.append("manifest was built with a different source registry; reassemble preprocessing")
    train_tokens = int(manifest.get("statistics", {}).get("train", {}).get("whitespace_tokens", 0))
    if train_tokens < minimum_train_tokens:
        errors.append(f"train split has {train_tokens:,} tokens; requires {minimum_train_tokens:,}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        errors.append("manifest has no split outputs")
    else:
        for split in ("train", "validation", "test"):
            values = outputs.get(split)
            if not isinstance(values, list) or not values:
                errors.append(f"manifest has no {split} shards")
                continue
            missing = [value for value in values if not _safe_output(data_root, value).is_file()]
            if missing:
                errors.append(f"{len(missing)} {split} output shard(s) are missing")
    if errors:
        raise PreprocessingNotComplete("; ".join(errors))
    manifest["_manifest_path"] = str(manifest_path)
    manifest["_manifest_sha256"] = sha256_file(manifest_path)
    manifest["_tokenizer_manual_fraction_override"] = bool(
        allow_low_manual_fraction and not manifest.get("manual_fraction_requirement_met")
    )
    return manifest
