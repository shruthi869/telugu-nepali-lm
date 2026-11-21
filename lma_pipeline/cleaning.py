"""Calibration, deterministic budget sampling, and quality cleaning."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .common import (
    LANGUAGE_CODES,
    TOKEN_RE,
    atomic_concatenate,
    atomic_clean_pairs,
    atomic_gzip_jsonl,
    atomic_write_text,
    read_jsonl_gzip,
    sha256_bytes,
    sha256_file,
)
from .manual_audit import write_manual_yield_report
from .source_policy import is_budget_exempt_source, is_course_manual_source
from .text import normalize_text, quality_rejection


# 100k records keeps interruption replay bounded while halving Drive checkpoint
# commits.  fastText has substantial Python-call overhead; 1,024 texts is still
# a small bounded batch but greatly reduces that overhead on Colab.
CLEAN_CHECKPOINT_RECORDS = 100_000
CLEAN_LID_BATCH_SIZE = 1_024
CLEAN_CHECKPOINT_SCHEMA = 1


class CleaningMixin:
    def _clean_checkpoint_directory(self, language: str, output: Path) -> Path:
        return self.root / "audit" / self.run_id / "clean-checkpoints" / language / output.stem

    @staticmethod
    def _remove_checkpoint_directory(directory: Path) -> None:
        if not directory.exists():
            return
        for path in directory.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass

    def _load_clean_checkpoint(self, directory: Path, input_sha: str) -> dict[str, Any]:
        progress_path = directory / "progress.json"
        empty = {
            "schema_version": CLEAN_CHECKPOINT_SCHEMA,
            "input_sha256": input_sha,
            "chunks": 0,
            "input_records": 0,
            "accepted": 0,
            "rejected": 0,
            "budget_skipped": 0,
        }
        if not progress_path.exists():
            return empty
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("schema_version") != CLEAN_CHECKPOINT_SCHEMA or progress.get("input_sha256") != input_sha:
                raise ValueError("checkpoint configuration changed")
            chunks = int(progress["chunks"])
            for index in range(chunks):
                for prefix in ("accepted", "rejected"):
                    part = directory / f"{prefix}-{index:06d}.jsonl.gz"
                    if not part.exists() or part.stat().st_size == 0:
                        raise ValueError(f"missing durable checkpoint part {part.name}")
            return progress
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.progress.emit(f"clean: discarding unusable checkpoint {directory.name}: {exc}")
            self._remove_checkpoint_directory(directory)
            return empty

    def _remove_replaced_clean_audit(self, language: str, previous_output: str | None, current_output: str) -> None:
        if not previous_output or previous_output == current_output:
            return
        previous = Path(previous_output)
        for stale in (
            self.root / "audit" / self.run_id / "rejections" / language / previous.name,
            self.root / "audit" / self.run_id / "sampling" / language / f"{previous.stem}.json",
        ):
            stale.unlink(missing_ok=True)

    def candidate_paths(self, language: str) -> list[Path]:
        prefix = f"processed/{self.run_id}/{language}/candidates/%"
        rows = self.state.conn.execute(
            "SELECT output_path FROM tasks WHERE run_id=? AND stage='extract' AND status='complete' "
            "AND output_path LIKE ? ORDER BY output_path",
            (self.run_id, prefix),
        )
        ledger_paths = [self.root / row[0] for row in rows if (self.root / row[0]).exists()]
        if ledger_paths:
            return ledger_paths
        # Fixture/backward-compatibility fallback for callers that intentionally
        # prepare candidate shards without scheduling extraction tasks.
        directory = self.root / "processed" / self.run_id / language / "candidates"
        return sorted(directory.glob("*.jsonl.gz")) if directory.exists() else []

    def _candidate_source(self, path: Path, sources: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        first = next(read_jsonl_gzip(path), None)
        if not first or first.get("source_id") not in sources:
            return None
        return sources[str(first["source_id"])], first

    def _selection_rate(self, language: str) -> float:
        if self.sample_rate is not None:
            return self.sample_rate
        calibration = self.root / "audit" / self.run_id / "calibration" / f"{language}.json"
        if calibration.exists():
            return float(json.loads(calibration.read_text(encoding="utf-8"))["recommended_sample_rate"])
        return 1.0

    def _selected_for_budget(self, item: dict[str, Any], language: str, rate: float) -> bool:
        if rate >= 1.0:
            return True
        key = f"{self.seed}:selection-v1:{language}:{item.get('source_id')}:{item.get('raw_file')}:{item.get('document_id')}"
        return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64 < rate

    def _candidate_identity(self, path: Path, source_id: str, raw_ref: str) -> str:
        """Return a constant-time identity for an immutable extracted candidate.

        Candidate paths are atomically written and are tied in the extraction
        ledger to the SHA-256 of their immutable raw input (and, where relevant,
        the extraction configuration).  Rehashing a multi-gigabyte Drive
        candidate before cleaning both duplicates the subsequent read and can
        leave Colab apparently idle for hours.  Use that durable identity when
        available; retain a content-hash fallback for fixture/legacy files that
        have no extraction-ledger entry.
        """
        task = self.state.existing_task(self.run_id, "extract", source_id, raw_ref)
        relative_path = str(path.relative_to(self.root))
        if (
            task
            and task["status"] == "complete"
            and task["output_path"] == relative_path
        ):
            return f"extract-ledger-v1:{task['input_sha256']}:{relative_path}"
        return f"content-sha256-v1:{sha256_file(path)}"

    def calibrate(self, sources: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Estimate clean-token yield from a bounded per-shard candidate sample."""
        by_id = {source["id"]: source for source in sources}
        results: dict[str, dict[str, Any]] = {}
        extract_counts = {row["output_path"]: int(row["records"]) for row in self.state.conn.execute("SELECT output_path,records FROM tasks WHERE run_id=? AND stage='extract' AND status='complete'", (self.run_id,))}
        for language in LANGUAGE_CODES:
            paths = []
            for path in self.candidate_paths(language):
                found = self._candidate_source(path, by_id)
                if (
                    found
                    and not is_budget_exempt_source(found[0], self.config_root)
                    and not found[1].get("budget_preselected")
                ):
                    paths.append(path)
            if not paths:
                continue
            fingerprint = self._fingerprint(paths) + f":calibration-v1:{self.calibration_records_per_shard}:{self.target_downloaded_tokens}"
            output = Path("audit") / self.run_id / "calibration" / f"{language}.json"
            if self.state.stage_complete(self.run_id, language, "calibrate", fingerprint, self.root):
                results[language] = json.loads((self.root / output).read_text(encoding="utf-8"))
                continue
            sample_records = clean_records = clean_tokens = capped = 0
            estimated_all_clean_tokens = 0.0
            for path in paths:
                found = self._candidate_source(path, by_id)
                if not found:
                    continue
                source, _ = found
                sampled = list(itertools.islice(read_jsonl_gzip(path), self.calibration_records_per_shard))
                clean_items: list[str] = []
                for item in sampled:
                    text = normalize_text(str(item.get("text", "")))
                    if quality_rejection(text, language) is None:
                        clean_items.append(text)
                    if max(0, len(TOKEN_RE.findall(text)) - 4) > self.max_shingles:
                        capped += 1
                predictions = self.predict_many(clean_items)
                kept = [text for text, (label, score) in zip(clean_items, predictions) if label == LANGUAGE_CODES[language] and score >= 0.70]
                sample_records += len(sampled)
                clean_records += len(kept)
                clean_tokens += sum(len(TOKEN_RE.findall(text)) for text in kept)
                total_records = extract_counts.get(str(path.relative_to(self.root)), len(sampled))
                if sampled:
                    estimated_all_clean_tokens += sum(len(TOKEN_RE.findall(text)) for text in kept) * total_records / len(sampled)
            # Oversample a little before exact/near dedup so the expected result
            # is near the requested downloaded-token budget rather than below it.
            recommended_rate = min(1.0, self.target_downloaded_tokens * 1.15 / max(estimated_all_clean_tokens, 1.0))
            result = {
                "language": language,
                "sample_records": sample_records,
                "clean_records": clean_records,
                "clean_whitespace_tokens": clean_tokens,
                "estimated_all_clean_tokens": round(estimated_all_clean_tokens),
                "recommended_sample_rate": recommended_rate,
                "target_downloaded_tokens": self.target_downloaded_tokens,
                "max_shingles": self.max_shingles,
                "sample_documents_over_shingle_cap": capped,
            }
            target = self.root / output
            atomic_write_text(target, json.dumps(result, indent=2, sort_keys=True))
            self.state.complete_stage(self.run_id, language, "calibrate", fingerprint, str(output), result)
            self.progress.emit(f"calibrate: {language}: estimated {result['estimated_all_clean_tokens']:,} clean tokens; recommended sample rate {recommended_rate:.6f}")
            results[language] = result
        return results

    def clean(self, sources: list[dict[str, Any]]) -> int:
        done = 0
        by_id = {source["id"]: source for source in sources}
        for language in LANGUAGE_CODES:
            rate = self._selection_rate(language)
            for path in self.candidate_paths(language):
                found = self._candidate_source(path, by_id)
                if not found:
                    continue
                source, first = found
                source_id = source["id"]
                raw_ref = str(first.get("raw_file", path))
                candidate_identity = self._candidate_identity(path, source_id, raw_ref)
                course_manual = is_course_manual_source(source, self.config_root)
                budget_exempt = is_budget_exempt_source(source, self.config_root)
                preselected = bool(first.get("budget_preselected"))
                effective_rate = 1.0 if budget_exempt or preselected else float(source.get("sample_rate", rate))
                if budget_exempt:
                    policy_path = (self.config_root / str(source["policy_snapshot"])).resolve()
                    policy_sha = sha256_file(policy_path)
                    classification_basis = source.get("course_classification_basis") or source.get("rights_basis") or "eligible_manual_source"
                    fingerprint_value = f"clean-budget-exempt-v4:{candidate_identity}:{policy_sha}:{classification_basis}"
                elif preselected:
                    fingerprint_value = f"clean-preselected-v5:{candidate_identity}:{first.get('budget_selection_rate')}"
                else:
                    fingerprint_value = f"clean-budget-sampled-v5:{candidate_identity}:{effective_rate:.12f}"
                clean_input_sha = sha256_bytes(fingerprint_value.encode("utf-8"))
                output = self.task_output("clean", language, source_id, clean_input_sha)
                existing = self.state.existing_task(self.run_id, "clean", source_id, raw_ref)
                task = self.state.task(self.run_id, "clean", source_id, raw_ref, clean_input_sha, str(output))
                if task["status"] == "complete" and (self.root / output).exists():
                    self._remove_checkpoint_directory(self._clean_checkpoint_directory(language, output))
                    self.progress.emit(f"clean: skipping completed {raw_ref}")
                    continue
                checkpoint_directory = self._clean_checkpoint_directory(language, output)
                checkpoint_directory.mkdir(parents=True, exist_ok=True)
                checkpoint = self._load_clean_checkpoint(checkpoint_directory, clean_input_sha)
                input_count = int(checkpoint["input_records"])
                accepted_count = int(checkpoint["accepted"])
                rejected_count = int(checkpoint["rejected"])
                sampled_out = int(checkpoint["budget_skipped"])
                completed_chunks = int(checkpoint["chunks"])
                def make_record(item: dict[str, Any], text: str, prediction: str, confidence: float) -> dict[str, Any]:
                    original_id = str(item.get("document_id"))
                    raw_identity = sha256_bytes(str(item.get("raw_file", "")).encode())[:16]
                    provenance = {
                        key: item[key]
                        for key in (
                            "extraction_method",
                            "page_start",
                            "page_end",
                            "epub_member",
                            "attribution_url",
                            "attribution_text",
                            "license_expression",
                            "policy_decision",
                            "policy_reason",
                            "mediawiki_page_id",
                            "mediawiki_title",
                            "mediawiki_revision_id",
                            "mediawiki_revision_timestamp",
                            "license_templates",
                            "matched_required_license_templates",
                            "license_categories",
                            "matched_required_license_categories",
                            "hplt_url",
                            "hplt_crawl_collection",
                            "hplt_warc_reference",
                            "hplt_quality_metadata",
                            "publication_date",
                            "ministry",
                            "pib_release_id",
                            "budget_selection_rate",
                        )
                        if key in item
                    }
                    return {"document_id": f"{source_id}:{raw_identity}:{original_id}", "source_document_id": original_id, "cleaned_text": text, "source_id": source_id, "source_type": item.get("source_type", source.get("source_type")), "document_type": item.get("document_type", "unknown"), "manual_collection": course_manual, "course_manual_source": course_manual, "budget_exempt_source": budget_exempt, "budget_preselected": preselected, "collection_label": "manual" if course_manual else "downloaded", "course_collection_category": "manual" if course_manual else "downloaded", "course_classification_basis": source.get("course_classification_basis"), "collection_method": source.get("collection_method"), "registry_manual_collection": bool(source.get("manual_collection")), "rights_basis": source.get("rights_basis") if course_manual else None, "policy_snapshot": source.get("policy_snapshot"), "raw_file": item.get("raw_file"), "language_prediction": prediction, "language_score": confidence, "content_hash": sha256_bytes(text.encode("utf-8")), **provenance}
                def flush_batch(batch: list[tuple[dict[str, Any], str]]) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
                    nonlocal accepted_count, rejected_count
                    for (item, text), (prediction, confidence) in zip(batch, self.predict_many([text for _, text in batch])):
                        record = make_record(item, text, prediction, confidence)
                        if prediction != LANGUAGE_CODES[language] or confidence < 0.70:
                            rejected_count += 1
                            yield None, {**record, "rejection_rule": "language_id"}
                        else:
                            accepted_count += 1
                            yield record, None
                def pairs(items: Iterable[dict[str, Any]]) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
                    nonlocal input_count, rejected_count, sampled_out
                    batch: list[tuple[dict[str, Any], str]] = []
                    for item in items:
                        input_count += 1
                        if not preselected and not self._selected_for_budget(item, language, effective_rate):
                            sampled_out += 1
                            continue
                        text = normalize_text(str(item.get("text", "")))
                        reason = quality_rejection(text, language)
                        if reason:
                            rejected_count += 1
                            yield None, {**make_record(item, text, "", 0.0), "rejection_rule": reason}
                        else:
                            batch.append((item, text))
                            if len(batch) >= CLEAN_LID_BATCH_SIZE:
                                yield from flush_batch(batch)
                                batch = []
                        self.progress.tick("clean", raw_ref, input_count, f" ({accepted_count:,} accepted, {rejected_count:,} rejected, {sampled_out:,} budget-skipped)")
                    if batch:
                        yield from flush_batch(batch)
                try:
                    label = "budget-exempt source; no budget sampling" if budget_exempt else (f"preselected before candidates at {first.get('budget_selection_rate'):.6f}" if preselected else f"deterministic sample rate {effective_rate:.6f}")
                    if input_count:
                        self.progress.emit(
                            f"clean: resuming {raw_ref} from durable checkpoint at {input_count:,} records; {label}"
                        )
                    else:
                        self.progress.emit(f"clean: starting {raw_ref}; {label}")
                    reject_path = self.root / "audit" / self.run_id / "rejections" / language / output.name
                    candidate_iterator = read_jsonl_gzip(path)
                    skipped = 0
                    while skipped < input_count:
                        try:
                            next(candidate_iterator)
                        except StopIteration as exc:
                            raise RuntimeError("clean checkpoint extends beyond the candidate shard") from exc
                        skipped += 1
                        self.progress.tick("clean-resume-scan", raw_ref, skipped, f" / {input_count:,} checkpointed")

                    chunk_index = completed_chunks
                    while True:
                        try:
                            first_item = next(candidate_iterator)
                        except StopIteration:
                            break
                        items = itertools.chain(
                            (first_item,),
                            itertools.islice(candidate_iterator, CLEAN_CHECKPOINT_RECORDS - 1),
                        )
                        accepted_part = checkpoint_directory / f"accepted-{chunk_index:06d}.jsonl.gz"
                        rejected_part = checkpoint_directory / f"rejected-{chunk_index:06d}.jsonl.gz"
                        atomic_clean_pairs(accepted_part, rejected_part, pairs(items))
                        chunk_index += 1
                        checkpoint = {
                            "schema_version": CLEAN_CHECKPOINT_SCHEMA,
                            "input_sha256": clean_input_sha,
                            "chunks": chunk_index,
                            "input_records": input_count,
                            "accepted": accepted_count,
                            "rejected": rejected_count,
                            "budget_skipped": sampled_out,
                        }
                        atomic_write_text(
                            checkpoint_directory / "progress.json",
                            json.dumps(checkpoint, sort_keys=True),
                        )
                        self.progress.emit(
                            f"clean-checkpoint: {raw_ref}: {input_count:,} records durable "
                            f"({accepted_count:,} accepted, {rejected_count:,} rejected, {sampled_out:,} budget-skipped)"
                        )

                    accepted_parts = [
                        checkpoint_directory / f"accepted-{index:06d}.jsonl.gz"
                        for index in range(chunk_index)
                    ]
                    rejected_parts = [
                        checkpoint_directory / f"rejected-{index:06d}.jsonl.gz"
                        for index in range(chunk_index)
                    ]
                    # Publish rejection/sampling audit state before the accepted
                    # shard. If interrupted after the accepted rename, State's
                    # normal reconciliation can safely mark the task complete.
                    if rejected_parts:
                        atomic_concatenate(reject_path, rejected_parts)
                    else:
                        atomic_gzip_jsonl(reject_path, [])
                    sampling_path = self.root / "audit" / self.run_id / "sampling" / language / f"{output.stem}.json"
                    atomic_write_text(sampling_path, json.dumps({"input_records": input_count, "budget_skipped": sampled_out, "accepted": accepted_count, "rejected": rejected_count, "sample_rate": effective_rate, "course_manual_source": course_manual, "budget_exempt_source": budget_exempt, "budget_preselected": preselected, "source_selection_rate": first.get("budget_selection_rate")}, sort_keys=True))
                    if accepted_parts:
                        atomic_concatenate(self.root / output, accepted_parts)
                    else:
                        atomic_gzip_jsonl(self.root / output, [])
                    count = accepted_count
                    self.state.complete_task(task, count)
                    self.remove_replaced_output(existing["output_path"] if existing else None, str(output))
                    self._remove_replaced_clean_audit(language, existing["output_path"] if existing else None, str(output))
                    self._remove_checkpoint_directory(checkpoint_directory)
                    self.progress.emit(f"clean: complete {raw_ref}: {accepted_count:,} accepted, {rejected_count:,} rejected, {sampled_out:,} budget-skipped")
                    done += 1
                except Exception as exc:
                    self.state.fail_task(task, exc)
                    self.progress.emit(f"clean: failed {raw_ref}: {type(exc).__name__}: {exc}")
                    raise
            for source in sources:
                if source.get("language") != language or not is_course_manual_source(source, self.config_root):
                    continue
                prefix = f"processed/{self.run_id}/{language}/clean/{source['id']}-%"
                rows = self.state.conn.execute(
                    "SELECT output_path FROM tasks WHERE run_id=? AND stage='clean' AND source_id=? "
                    "AND status='complete' AND output_path LIKE ? ORDER BY output_path",
                    (self.run_id, source["id"], prefix),
                )
                paths = [self.root / row[0] for row in rows if (self.root / row[0]).exists()]
                if not paths:
                    continue
                report = write_manual_yield_report(self.root, self.run_id, language, source, paths)
                self.progress.emit(
                    f"manual-yield: {source['id']}: {report['accepted_documents']:,} clean documents, "
                    f"{report['accepted_whitespace_tokens']:,} clean tokens before global dedup"
                )
        return done

    def clean_paths(self, language: str) -> list[Path]:
        # The task ledger is authoritative.  It prevents stale clean outputs from
        # a previous sampling configuration being silently mixed into a new run.
        prefix = f"processed/{self.run_id}/{language}/clean/%"
        rows = self.state.conn.execute(
            "SELECT output_path FROM tasks WHERE run_id=? AND stage='clean' AND status='complete' AND output_path LIKE ? ORDER BY output_path",
            (self.run_id, prefix),
        )
        return [self.root / row[0] for row in rows if (self.root / row[0]).exists()]
