"""Small, resumable yield reports for policy-approved manual sources."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import TOKEN_RE, atomic_write_text, read_jsonl_gzip, utcnow


def _policy_totals(root: Path, run_id: str, language: str, source_id: str) -> dict[str, Any]:
    directory = root / "audit" / run_id / "manual-policy" / language / source_id
    totals: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for path in sorted(directory.glob("*.json")) if directory.exists() else ():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("accepted_by_item_policy", "rejected_by_item_policy", "malformed_or_empty"):
            totals[key] += int(value.get(key, 0))
        reasons.update({str(key): int(count) for key, count in value.get("rejection_reasons", {}).items()})
    return {
        "accepted_by_item_policy": totals["accepted_by_item_policy"],
        "rejected_by_item_policy": totals["rejected_by_item_policy"],
        "malformed_or_empty": totals["malformed_or_empty"],
        "rejection_reasons": dict(reasons),
    }


def write_manual_yield_report(
    root: Path,
    run_id: str,
    language: str,
    source: dict[str, Any],
    clean_paths: Iterable[Path],
) -> dict[str, Any]:
    """Measure a source before global deduplication and write a reviewable report."""
    paths = sorted(set(clean_paths))
    accepted_documents = accepted_tokens = accepted_characters = 0
    rejected_rules: Counter[str] = Counter()
    input_records = budget_skipped = 0
    for path in paths:
        for record in read_jsonl_gzip(path):
            text = str(record.get("cleaned_text", ""))
            accepted_documents += 1
            accepted_tokens += len(TOKEN_RE.findall(text))
            accepted_characters += len(text)
        rejection_path = root / "audit" / run_id / "rejections" / language / path.name
        if rejection_path.exists():
            for record in read_jsonl_gzip(rejection_path):
                rejected_rules[str(record.get("rejection_rule", "unknown"))] += 1
        sampling_path = root / "audit" / run_id / "sampling" / language / f"{path.stem}.json"
        if sampling_path.exists():
            try:
                sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                sampling = {}
            input_records += int(sampling.get("input_records", 0))
            budget_skipped += int(sampling.get("budget_skipped", 0))

    estimated_train_tokens = round(accepted_tokens * 0.98)
    target = int(source.get("target_clean_tokens", 0) or 0)
    report = {
        "schema_version": 1,
        "created_at": utcnow(),
        "measurement_stage": "clean_pre_global_deduplication",
        "run_id": run_id,
        "language": language,
        "source_id": source["id"],
        "collection_method": source.get("collection_method"),
        "rights_basis": source.get("rights_basis"),
        "license_expression": source.get("license_expression"),
        "attribution_text": source.get("attribution_text"),
        "policy_snapshot": source.get("policy_snapshot"),
        "input_records": input_records,
        "budget_skipped": budget_skipped,
        "accepted_documents": accepted_documents,
        "accepted_characters": accepted_characters,
        "accepted_whitespace_tokens": accepted_tokens,
        "estimated_98_percent_train_tokens_before_dedup": estimated_train_tokens,
        "quality_rejections": sum(rejected_rules.values()),
        "quality_rejection_counts": dict(rejected_rules),
        "item_policy": _policy_totals(root, run_id, language, source["id"]),
        "target_clean_tokens": target or None,
        "target_progress": accepted_tokens / target if target else None,
        "clean_shards": [str(path.relative_to(root)) for path in paths],
    }
    output = root / "audit" / run_id / "manual-yield" / language / f"{source['id']}.json"
    atomic_write_text(output, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report
