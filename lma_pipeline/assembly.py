"""Final split assignment, audit generation, and corpus assembly."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import (
    LSH_BANDS,
    LSH_ROWS,
    MINHASH_SIZE,
    MODEL_NAME,
    MODEL_VERSION,
    TOKEN_RE,
    atomic_gzip_jsonl,
    atomic_write_text,
    read_jsonl_gzip,
    sha256_bytes,
    sha256_file,
    utcnow,
)
from .source_policy import is_budget_exempt_source, is_course_manual_source
from .text import split_for


class AssemblyMixin:
    def _decontaminate(self, records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            record["split"] = split_for(record["document_id"], self.seed)
            by_split[record["split"]].append(record)
        # Global exact + verified-near deduplication was performed before split
        # assignment. Therefore no verified duplicate can cross a split. Doing a
        # fresh all-train-vs-heldout comparison here would be quadratic and would
        # not find anything the global duplicate graph did not already remove.
        return records, 0

    def _review_and_rejections(self, language: str, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        rejection_counts: Counter[str] = Counter()
        source_rejections: Counter[str] = Counter()
        samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

        def consider(rule: str, record: dict[str, Any]) -> None:
            choices = samples[rule]
            choices.append(record)
            # Keeping just the 20 deterministic lowest hashes is equivalent to
            # sorting every record, but remains bounded for multi-million-doc runs.
            if len(choices) > 20:
                worst = max(
                    range(len(choices)),
                    key=lambda index: sha256_bytes(f"{self.seed}:{rule}:{choices[index]['document_id']}".encode()),
                )
                choices.pop(worst)

        for record in records:
            consider("accepted", record)
        directory = self.root / "audit" / self.run_id / "rejections" / language
        if directory.exists():
            for path in directory.glob("*.jsonl.gz"):
                for record in read_jsonl_gzip(path):
                    rule = str(record.get("rejection_rule", "unknown"))
                    rejection_counts[rule] += 1
                    source_rejections[f"{record.get('source_id')}:{rule}"] += 1
                    consider(rule, record)
        review: dict[str, list[dict[str, Any]]] = {}
        for rule, choices in samples.items():
            chosen = sorted(choices, key=lambda item: sha256_bytes(f"{self.seed}:{rule}:{item['document_id']}".encode()))[:20]
            review[rule] = [{key: item.get(key) for key in ("document_id", "source_id", "rejection_rule", "language_prediction", "language_score", "cleaned_text")} for item in chosen]
        review_path = self.root / "audit" / self.run_id / "review_samples" / f"{language}.json"
        atomic_write_text(review_path, json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True))
        return {"rejection_counts": dict(rejection_counts), "rejection_counts_by_source_rule": dict(source_rejections), "review_samples": str(review_path.relative_to(self.root))}

    def assemble(self, language: str, finalize: bool = False, allow_low_manual_fraction: bool = False) -> dict[str, Any] | None:
        paths = self.clean_paths(language)
        if not paths:
            return None
        clean_fingerprint = self._fingerprint(paths)
        fingerprint = sha256_bytes(
            (
                f"{clean_fingerprint}:assembly-v2:finalize={int(finalize)}:"
                f"manual-waiver={int(allow_low_manual_fraction)}:"
                f"minimum-train-tokens={self.minimum_train_tokens}:"
                # Final manifests name and classify sources from this registry.
                # It must therefore invalidate an otherwise-complete derived
                # manifest when the registry changes, even if cleaned inputs
                # have not changed.
                f"registry={sha256_file(self.config_path)}"
            ).encode("utf-8")
        )
        manifest_path = Path("audit") / self.run_id / "manifests" / f"{language}.json"
        if self.state.stage_complete(self.run_id, language, "assemble", fingerprint, self.root):
            try:
                completed_manifest = json.loads((self.root / manifest_path).read_text(encoding="utf-8"))
                output_paths = [
                    self.root / output
                    for split_outputs in completed_manifest.get("outputs", {}).values()
                    for output in split_outputs
                ]
                if not output_paths or not all(path.exists() and path.stat().st_size > 0 for path in output_paths):
                    raise ValueError("one or more final output shards are missing")
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self.progress.emit(f"assemble: {language}: completed checkpoint is incomplete ({exc}); rebuilding derived outputs")
            else:
                self.progress.emit(f"assemble: {language}: skipping completed manifest")
                self._cleanup_local_work(language)
                return completed_manifest
        self.progress.emit(f"assemble: {language}: starting {len(paths):,} cleaned shard(s)")
        # Finalization policy changes only derived output/manifest state.  Exact
        # and near-dedup checkpoints depend solely on the cleaned inputs.
        db_path, counts = self._assemble_db(language, paths, clean_fingerprint)
        db_path, counts["near_removed"] = self._near_deduplicate(language, db_path)
        db = sqlite3.connect(db_path)
        counts["split_decontamination_removed"] = 0
        # Keep at most one final-output shard per split in RAM.  The old version
        # accumulated the whole language corpus here, which could exceed free
        # Colab RAM long before writing the final JSONL shards.
        buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        outputs: dict[str, list[str]] = {}
        for split in ("train", "validation", "test"):
            outputs[split] = []
        statistics: dict[str, dict[str, int]] = {
            split: {"documents": 0, "characters": 0, "whitespace_tokens": 0}
            for split in ("train", "validation", "test")
        }
        source_tokens: Counter[str] = Counter()
        manual_train_tokens = 0
        review_records: list[dict[str, Any]] = []
        assembled_documents = 0

        def flush(split: str) -> None:
            shard = buffers[split]
            if not shard:
                return
            rel = Path("processed") / self.run_id / language / split / f"part-{len(outputs[split]):05d}.jsonl.gz"
            atomic_gzip_jsonl(self.root / rel, shard)
            outputs[split].append(str(rel))
            buffers[split] = []

        for (record_json,) in db.execute("SELECT record_json FROM docs WHERE dropped=0 ORDER BY document_id"):
            assembled_documents += 1
            record = json.loads(record_json)
            split = split_for(record["document_id"], self.seed)
            record["split"] = split
            tokens = len(TOKEN_RE.findall(record["cleaned_text"]))
            statistics[split]["documents"] += 1
            statistics[split]["characters"] += len(record["cleaned_text"])
            statistics[split]["whitespace_tokens"] += tokens
            if split == "train":
                source_tokens[record["source_id"]] += tokens
                if record.get("course_collection_category") == "manual":
                    manual_train_tokens += tokens
            review_records.append(record)
            if len(review_records) > 20:
                # `_review_and_rejections` receives a bounded deterministic sample
                # below; use a cheap deterministic key while streaming records.
                worst = max(range(len(review_records)), key=lambda index: sha256_bytes(f"{self.seed}:accepted:{review_records[index]['document_id']}".encode()))
                review_records.pop(worst)
            buffers[split].append(record)
            if len(buffers[split]) >= self.shard_records:
                flush(split)
            self.progress.tick(
                "assemble-output",
                language,
                assembled_documents,
                f" ({sum(len(values) for values in outputs.values()):,} shards written)",
            )
        db.close()
        for split in ("train", "validation", "test"):
            flush(split)
            # A successful reassembly may produce fewer parts than a previous one.
            # Remove only those stale derived parts after all current parts exist.
            directory = self.root / "processed" / self.run_id / language / split
            keep = {Path(value).name for value in outputs[split]}
            for stale in directory.glob("part-*.jsonl.gz") if directory.exists() else ():
                if stale.name not in keep:
                    try:
                        stale.unlink()
                    except OSError as exc:
                        self.progress.emit(f"assemble: {language}: could not remove stale derived shard {stale.name}: {exc}")
        self.progress.emit(
            f"assemble-output: {language}: {assembled_documents:,} documents written to "
            f"{sum(len(values) for values in outputs.values()):,} final shards"
        )
        review = self._review_and_rejections(language, review_records)
        train_tokens = statistics["train"]["whitespace_tokens"]
        model = {"name": MODEL_NAME, "version": MODEL_VERSION, "path": str(self.model_path), "sha256": sha256_file(self.model_path) if self.model_path.exists() else None}
        manual_sources: list[dict[str, Any]] = []
        for source_id in sorted(source_tokens):
            source = self.sources.get(source_id)
            if not source or not is_course_manual_source(source, self.config_root):
                continue
            snapshot_path = (self.config_root / source["policy_snapshot"]).resolve()
            manual_sources.append({
                "id": source_id,
                "source_type": source.get("source_type"),
                "collection_method": source.get("collection_method"),
                "course_collection_category": source.get("course_collection_category", "manual"),
                "course_classification_basis": source.get("course_classification_basis"),
                "registry_manual_collection": bool(source.get("manual_collection")),
                "rights_basis": source.get("rights_basis"),
                "license_expression": source.get("license_expression"),
                "attribution_text": source.get("attribution_text"),
                "policy_snapshot": source.get("policy_snapshot"),
                "policy_snapshot_sha256": sha256_file(snapshot_path),
                "url": source.get("url"),
            })
        manifest = {
            "schema_version": 5,
            "run_id": self.run_id,
            "language": language,
            "created_at": utcnow(),
            "seed": self.seed,
            "registry_sha256": sha256_file(self.config_path),
            "language_id_model": model,
            "sampling": {
                "sample_rate": self._selection_rate(language),
                "downloaded_sample_rate": self._selection_rate(language),
                "manual_sample_rate": 1.0,
                "source_sample_rates": {
                    source_id: float(self.sources[source_id].get("sample_rate", 1.0 if is_budget_exempt_source(self.sources[source_id], self.config_root) else self._selection_rate(language)))
                    for source_id in sorted(source_tokens)
                    if source_id in self.sources
                },
                "target_downloaded_tokens": self.target_downloaded_tokens,
            },
            "deduplication": {
                **counts,
                "minhash_size": MINHASH_SIZE,
                "lsh_bands": LSH_BANDS,
                "lsh_rows": LSH_ROWS,
                "max_shingles_for_candidate_generation": self.max_shingles,
                "verified_jaccard_threshold": 0.80,
                "split_decontamination": "global verified-near-dedup before deterministic split",
            },
            "inputs": [str(path.relative_to(self.root)) for path in paths],
            "outputs": outputs,
            "statistics": statistics,
            "train_source_token_fractions": {
                source: value / max(train_tokens, 1)
                for source, value in sorted(source_tokens.items())
            },
            "manual_sources": manual_sources,
            "manual_train_tokens": manual_train_tokens,
            "manual_train_token_fraction": manual_train_tokens / max(train_tokens, 1),
            "minimum_train_tokens": self.minimum_train_tokens,
            "minimum_train_tokens_met": self.minimum_train_tokens is None or train_tokens >= self.minimum_train_tokens,
            **review,
        }
        manual_ok = manifest["manual_train_token_fraction"] >= 0.20
        manual_satisfied = manual_ok or allow_low_manual_fraction
        token_target_ok = bool(manifest["minimum_train_tokens_met"])
        manifest["manual_fraction_requirement_met"] = manual_ok
        manifest["manual_fraction_requirement_waived"] = bool(allow_low_manual_fraction and not manual_ok)
        manifest["training_status"] = "final" if finalize and manual_satisfied and token_target_ok else "draft"
        manifest["finalization_requirement"] = (
            "At least 20% course-classified manual training tokens per language. "
            "Wiki sources and only the allow-listed HPLT exception count as manual; "
            "physical collection_method remains independent provenance."
        )
        if manifest["manual_fraction_requirement_waived"]:
            manifest["finalization_warning"] = (
                "The 20% manual-source requirement was explicitly waived. "
                "Source acquisition methods and course collection categories remain separately recorded in provenance."
            )
        errors = []
        if finalize and not manual_satisfied:
            errors.append(
                f"manual training token fraction is {manifest['manual_train_token_fraction']:.2%}; requires at least 20.00%"
            )
        if finalize and not token_target_ok:
            errors.append(
                f"training tokens are {train_tokens:,}; requires at least {self.minimum_train_tokens:,}"
            )
        if errors:
            manifest["finalization_error"] = "; ".join(errors)
        target = self.root / manifest_path
        atomic_write_text(target, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        self.state.complete_stage(self.run_id, language, "assemble", fingerprint, str(manifest_path), manifest)
        # The exact/near-dedup database on Drive is the durable checkpoint.  The
        # local copy is retained on failure for fast resume, but has no value once
        # the manifest and all final shards are durably complete.
        self._cleanup_local_work(language)
        self.progress.emit(f"assemble: {language}: complete; manifest {manifest_path}")
        return manifest
