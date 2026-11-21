"""Fixture coverage for the resumable preprocessing pipeline."""

from __future__ import annotations

import bz2
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import preprocess
import lma_pipeline.assembly as assembly
import lma_pipeline.cleaning as cleaning
import lma_pipeline.near_dedup as deduplication
from lma_pipeline.source_policy import is_budget_exempt_source, is_course_manual_source, is_manual_source


def indic_text(language: str, suffix: str = "") -> str:
    start = 0x0C15 if language == "tel" else 0x0915
    # One non-repeated long sentence: adequate script coverage without tripping the
    # repeated sentence filter used for boilerplate.
    return " ".join("".join(chr(start + ((index + offset) % 20)) for offset in range(5)) for index in range(70)) + suffix


class PreprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "data"
        self.root.mkdir()
        self.config_dir = Path(self.temp.name) / "configs"
        policies = self.config_dir / "policy_snapshots"
        policies.mkdir(parents=True)
        (policies / "policy.md").write_text("Dated policy snapshot allowing this exact crawl scope and training use.", encoding="utf-8")
        self.source = {
            "id": "fixture_tel", "approved": True, "language": "tel", "source_type": "fixture",
            "collection_method": "sitemap_crawl", "policy_snapshot": "configs/policy_snapshots/policy.md",
            "manual_collection": False,
        }
        self.config = self.config_dir / "sources.json"
        self.config.write_text(json.dumps({"sources": [self.source]}), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def processor(self, workers: int = 1) -> preprocess.Preprocessor:
        def predict(text: str) -> tuple[str, float]:
            return ("te", 0.99) if preprocess.target_script_count(text, "tel") else ("ne", 0.99)
        return preprocess.Preprocessor(self.root, self.config, "fixture", language_predictor=predict, shard_records=1, workers=workers)

    def candidate(self, records: list[dict]) -> Path:
        path = self.root / "processed" / "fixture" / "tel" / "candidates" / "fixture_tel-input.jsonl.gz"
        preprocess.atomic_gzip_jsonl(path, records)
        return path

    def test_normalization_redacts_pii_preserves_joiner_and_filters_quality(self) -> None:
        text = "a\u0301\u200d email@example.com +91 98765 43210 1234 5678 9012"
        normalized = preprocess.normalize_text(text)
        self.assertIn("\u200d", normalized)
        self.assertIn("[EMAIL_REDACTED]", normalized)
        self.assertIn("REDACTED", normalized)
        self.assertEqual(preprocess.quality_rejection("short", "tel"), "too_short")
        self.assertEqual(preprocess.quality_rejection("a" * 220, "tel"), "too_few_target_script_characters")

    def test_project_rule_counts_wiki_as_manual_but_direct_dataset_as_downloaded(self) -> None:
        wiki = {
            "source_type": "wikimedia_dump",
            "collection_method": "direct_assets",
            "course_collection_category": "manual",
            "manual_collection": False,
        }
        dataset = {
            "source_type": "open_dataset",
            "collection_method": "huggingface_parquet",
            "course_collection_category": "downloaded",
            "manual_collection": False,
        }
        self.assertTrue(is_manual_source(wiki, self.config_dir.parent))
        self.assertFalse(is_manual_source(dataset, self.config_dir.parent))

    def test_hplt_override_is_narrow_and_remains_budget_sampled(self) -> None:
        hplt = {
            "id": "hplt2_tel_cleaned", "approved": True, "language": "tel",
            "source_type": "open_dataset", "collection_method": "huggingface_parquet",
            "dataset": "HPLT/HPLT2.0_cleaned", "dataset_config": "tel_Telu",
            "course_collection_category": "manual",
            "course_classification_basis": "ta_whatsapp_hplt_manual_exception_2026_08",
            "manual_collection": False, "sample_rate": 0.01,
            "policy_snapshot": "configs/policy_snapshots/policy.md",
        }
        self.assertTrue(is_course_manual_source(hplt, self.config_dir.parent))
        self.assertFalse(is_budget_exempt_source(hplt, self.config_dir.parent))
        arbitrary = {**hplt, "id": "not_hplt", "dataset": "other/dataset"}
        self.assertFalse(is_course_manual_source(arbitrary, self.config_dir.parent))
        self.config.write_text(json.dumps({"sources": [hplt]}), encoding="utf-8")
        processor = self.processor()
        path = self.root / "processed" / "fixture" / "tel" / "candidates" / "hplt-input.jsonl.gz"
        preprocess.atomic_gzip_jsonl(path, [
            {"document_id": str(index), "text": indic_text("tel", f" {index}"), "source_id": hplt["id"], "source_type": "open_dataset", "raw_file": "raw/hplt"}
            for index in range(100)
        ])
        processor.clean([hplt])
        sampling = next((self.root / "audit" / "fixture" / "sampling" / "tel").glob("*.json"))
        result = json.loads(sampling.read_text(encoding="utf-8"))
        self.assertTrue(result["course_manual_source"])
        self.assertFalse(result["budget_exempt_source"])
        self.assertGreater(result["budget_skipped"], 0)
        processor.close()

    def test_hplt_parquet_provenance_and_cross_source_preference(self) -> None:
        processor = self.processor()
        parquet_path = self.root / "raw" / "assets" / "hplt2_tel_cleaned" / "a.parquet"
        parquet_path.parent.mkdir(parents=True)
        parquet_path.write_bytes(b"fixture")
        source = {"dataset": "HPLT/HPLT2.0_cleaned"}
        class Batch:
            def to_pylist(self) -> list[dict]:
                return [{"text": "text", "id": "h1", "url": "https://example.test/a", "crawl": "CC-MAIN", "warc": "x.warc", "quality": 0.9}]
        class ParquetFile:
            schema = types.SimpleNamespace(names=["text", "id", "url", "crawl", "warc", "quality"])
            def __init__(self, path: Path): pass
            def iter_batches(self, **kwargs: object): return [Batch()]
        pyarrow = types.ModuleType("pyarrow")
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.ParquetFile = ParquetFile  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"pyarrow": pyarrow, "pyarrow.parquet": parquet}):
            record = next(processor._parquet_documents(parquet_path, source))
        self.assertEqual(record["hplt_url"], "https://example.test/a")
        self.assertEqual(record["hplt_warc_reference"], "x.warc")
        content = indic_text("tel")
        common = {"cleaned_text": content, "content_hash": preprocess.sha256_bytes(content.encode()), "language_score": 0.99}
        hplt_path = self.root / "processed" / "fixture" / "tel" / "clean" / "hplt.jsonl.gz"
        sangraha_path = self.root / "processed" / "fixture" / "tel" / "clean" / "sangraha.jsonl.gz"
        preprocess.atomic_gzip_jsonl(sangraha_path, [{**common, "document_id": "s", "source_id": "sangraha_tel_verified", "manual_collection": False, "course_collection_category": "downloaded"}])
        preprocess.atomic_gzip_jsonl(hplt_path, [{**common, "document_id": "h", "source_id": "hplt2_tel_cleaned", "manual_collection": True, "course_collection_category": "manual"}])
        db_path, counts = processor._assemble_db("tel", [sangraha_path, hplt_path])
        db = sqlite3.connect(db_path)
        winner = json.loads(db.execute("SELECT record_json FROM docs").fetchone()[0])
        db.close()
        self.assertEqual(counts["exact_removed"], 1)
        self.assertEqual(winner["source_id"], "hplt2_tel_cleaned")
        processor.close()

    def test_hplt_is_selected_before_candidate_shards_are_written(self) -> None:
        hplt = {
            "id": "hplt2_tel_cleaned", "approved": True, "language": "tel",
            "source_type": "open_dataset", "collection_method": "huggingface_parquet",
            "dataset": "HPLT/HPLT2.0_cleaned", "dataset_config": "tel_Telu",
            "course_collection_category": "manual",
            "course_classification_basis": "ta_whatsapp_hplt_manual_exception_2026_08",
            "manual_collection": False, "sample_rate": 0.01,
            "policy_snapshot": "configs/policy_snapshots/policy.md",
        }
        self.config.write_text(json.dumps({"sources": [hplt]}), encoding="utf-8")
        raw = self.root / "raw" / "assets" / hplt["id"] / "fixture.parquet"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"fixture")
        processor = self.processor()
        items = [
            {"document_id": str(index), "text": indic_text("tel", f" {index}"), "document_type": "dataset_document"}
            for index in range(100)
        ]
        with patch.object(processor, "extracted_documents", return_value=iter(items)):
            processor.extract([hplt])
        candidate = next((self.root / "processed" / "fixture" / "tel" / "candidates").glob("*.jsonl.gz"))
        records = list(preprocess.read_jsonl_gzip(candidate))
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), len(items))
        self.assertTrue(all(record["budget_preselected"] for record in records))
        audit = next((self.root / "audit" / "fixture" / "extraction-sampling" / "tel").glob("*.json"))
        result = json.loads(audit.read_text(encoding="utf-8"))
        self.assertEqual(result["input_records"], 100)
        self.assertEqual(result["candidate_records"], len(records))
        processor.close()

    def test_clean_uses_extraction_ledger_identity_without_rehashing_candidate(self) -> None:
        """Drive candidates must not be read once just to construct a clean key."""
        processor = self.processor()
        candidate = self.candidate([
            {"document_id": "a", "text": indic_text("tel"), "source_id": "fixture_tel", "raw_file": "raw/a"}
        ])
        task = processor.state.task(
            "fixture", "extract", "fixture_tel", "raw/a", "raw-input-sha",
            str(candidate.relative_to(self.root)),
        )
        processor.state.complete_task(task, 1)
        with patch.object(cleaning, "sha256_file", side_effect=AssertionError("candidate was rehashed")):
            identity = processor._candidate_identity(candidate, "fixture_tel", "raw/a")
        self.assertEqual(
            identity,
            f"extract-ledger-v1:raw-input-sha:{candidate.relative_to(self.root)}",
        )
        processor.close()

    def test_streamed_minhash_is_signature_compatible_with_set_implementation(self) -> None:
        def legacy_signature(text: str, size: int = preprocess.MINHASH_SIZE, max_shingles: int = preprocess.DEFAULT_MAX_SHINGLES) -> tuple[int, ...]:
            shingles = preprocess.word_shingles(text, max_shingles=max_shingles)
            empty = (1 << 64) - 1
            values = [empty] * size
            base_seed = preprocess._fast_hash64(f"{preprocess.DEFAULT_SEED}:doph-v1")
            for shingle in shingles:
                hashed = preprocess._fast_hash64(shingle, base_seed)
                index = hashed % size
                rank = hashed // size
                if rank < values[index]:
                    values[index] = rank
            if all(value == empty for value in values):
                return tuple(preprocess._fast_hash64(f"{preprocess.DEFAULT_SEED}:empty:{index}") for index in range(size))
            for index, value in enumerate(values):
                if value != empty:
                    continue
                step = 1
                while values[(index + step) % size] == empty:
                    step += 1
                values[index] = preprocess._fast_hash64(values[(index + step) % size].to_bytes(8, "big"), index + 1)
            return tuple(values)

        texts = ["", "one two", " ".join(f"token-{index % 300}" for index in range(7_000)), indic_text("tel")]
        for text in texts:
            old_signature = legacy_signature(text)
            new_signature = preprocess.minhash_signature(text)
            self.assertEqual(new_signature, old_signature)
            old_buckets = [(band, int(bucket, 16)) for band, bucket in (
                (band, f"{preprocess._fast_hash64(b''.join(value.to_bytes(8, 'big') for value in old_signature[band * preprocess.LSH_ROWS:(band + 1) * preprocess.LSH_ROWS]), band):016x}")
                for band in range(preprocess.LSH_BANDS)
            )]
            self.assertEqual(list(preprocess.signature_buckets(new_signature)), old_buckets)

    def test_atomic_writers_remove_temporary_files_after_failure(self) -> None:
        output = self.root / "processed" / "broken.jsonl.gz"

        def broken_records():
            yield {"document_id": "written-before-failure"}
            raise OSError("simulated full disk")

        with self.assertRaises(OSError):
            preprocess.atomic_gzip_jsonl(output, broken_records())
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.glob("*.tmp")), [])

        text_output = self.root / "audit" / "broken.json"
        with patch("os.fsync", side_effect=OSError("simulated flush failure")):
            with self.assertRaises(OSError):
                preprocess.atomic_write_text(text_output, "payload")
        self.assertFalse(text_output.exists())
        self.assertEqual(list(text_output.parent.glob("*.tmp")), [])

    def test_mocked_parquet_and_wikimedia_extraction(self) -> None:
        processor = self.processor(workers=2)
        parquet_path = self.root / "raw" / "assets" / "fixture_tel" / "a.parquet"
        parquet_path.parent.mkdir(parents=True)
        parquet_path.write_bytes(b"fixture")

        class Batch:
            def to_pylist(self) -> list[dict]:
                return [{"text": "hello", "doc_id": "d1", "type": "news"}]
        class ParquetFile:
            schema = types.SimpleNamespace(names=["text", "doc_id", "type"])
            def __init__(self, path: Path): pass
            def iter_batches(self, **kwargs: object): return [Batch()]
        pyarrow = types.ModuleType("pyarrow")
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.ParquetFile = ParquetFile  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"pyarrow": pyarrow, "pyarrow.parquet": parquet}):
            records = list(processor._parquet_documents(parquet_path, self.source))
        self.assertEqual(records, [{"document_id": "d1", "text": "hello", "document_type": "news"}])

        class UnidentifiedParquetFile:
            schema = types.SimpleNamespace(names=["text"])
            def __init__(self, path: Path): pass
            def iter_batches(self, **kwargs: object):
                return [types.SimpleNamespace(to_pylist=lambda: [{"text": "first"}]),
                        types.SimpleNamespace(to_pylist=lambda: [{"text": "second"}])]
        parquet.ParquetFile = UnidentifiedParquetFile  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"pyarrow": pyarrow, "pyarrow.parquet": parquet}):
            fallback_records = list(processor._parquet_documents(parquet_path, self.source))
        self.assertEqual([record["document_id"] for record in fallback_records], ["0", "1"])

        xml = b"""<mediawiki><page><title>Article</title><ns>0</ns><id>1</id><revision><text>kept markup</text></revision></page><page><title>Empty</title><ns>0</ns><id>4</id><revision><text /></revision></page><page><title>Redirect</title><ns>0</ns><redirect title='Article'/><id>2</id><revision><text>ignored</text></revision></page><page><title>Talk</title><ns>1</ns><id>3</id><revision><text>ignored</text></revision></page></mediawiki>"""
        wiki_path = self.root / "raw" / "assets" / "fixture_tel" / "a.xml.bz2"
        wiki_path.write_bytes(bz2.compress(xml))
        mw = types.ModuleType("mwparserfromhell")
        mw.parse = lambda text: types.SimpleNamespace(strip_code=lambda normalize, collapse: text.replace(" markup", ""))  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"mwparserfromhell": mw}):
            wiki_records = list(processor._wiki_documents(wiki_path, self.source))
        self.assertEqual(wiki_records, [{"document_id": "1", "text": "kept", "document_type": "wikimedia_article"}])
        processor.close()

    def test_exact_dedup_resolves_reused_document_ids_without_losing_content(self) -> None:
        processor = self.processor()
        path = self.root / "processed" / "fixture" / "tel" / "clean" / "reused-ids.jsonl.gz"
        first_text = indic_text("tel", " first")
        second_text = indic_text("tel", " second")
        first_hash = preprocess.sha256_bytes(first_text.encode("utf-8"))
        second_hash = preprocess.sha256_bytes(second_text.encode("utf-8"))
        base = {
            "document_id": "reused-id",
            "source_document_id": "reused-id",
            "source_id": "fixture_tel",
            "source_type": "fixture",
            "manual_collection": False,
            "language_score": 0.99,
        }
        preprocess.atomic_gzip_jsonl(path, [
            {**base, "cleaned_text": first_text, "content_hash": first_hash},
            {**base, "cleaned_text": second_text, "content_hash": second_hash},
            {**base, "cleaned_text": first_text, "content_hash": first_hash},
        ])
        db_path, counts = processor._assemble_db("tel", [path])
        db = sqlite3.connect(db_path)
        rows = db.execute("SELECT document_id,record_json FROM docs ORDER BY document_id").fetchall()
        db.close()
        self.assertEqual(len(rows), 2)
        self.assertEqual(counts["exact_removed"], 1)
        self.assertEqual(counts["document_id_collisions_resolved"], 1)
        self.assertEqual(
            {row[0] for row in rows},
            {"reused-id", f"reused-id:content-{second_hash}"},
        )
        resolved = json.loads(next(row[1] for row in rows if row[0] != "reused-id"))
        self.assertEqual(resolved["document_id_collision_original"], "reused-id")
        processor.close()

    def test_exact_near_dedup_and_split_decontamination(self) -> None:
        processor = self.processor(workers=2)
        base = indic_text("tel")
        copy = base
        near = base.replace("కఖగఘఙ", "కఖగఘచ", 1)
        self.candidate([
            {"document_id": "a", "text": base, "source_id": "fixture_tel", "source_type": "fixture", "raw_file": "raw/a"},
            {"document_id": "b", "text": copy, "source_id": "fixture_tel", "source_type": "fixture", "raw_file": "raw/b"},
            {"document_id": "c", "text": near, "source_id": "fixture_tel", "source_type": "fixture", "raw_file": "raw/c"},
        ])
        processor.clean([self.source])
        paths = processor.clean_paths("tel")
        db_path, counts = processor._assemble_db("tel", paths)
        self.assertEqual(counts["exact_removed"], 1)
        # A reset while signatures are being written must discard an incomplete
        # marker/partition set rather than treating it as resumable work.
        partition_root = db_path.with_name(db_path.stem + ".lsh-partitions")
        partition_root.mkdir()
        (partition_root / "signatures-complete.json").write_text("{broken", encoding="utf-8")
        (partition_root / "part-000.bin").write_bytes(b"partial")
        interrupted_db = sqlite3.connect(db_path)
        interrupted_db.execute("UPDATE docs SET dropped=1 WHERE document_id='c'")
        interrupted_db.commit()
        interrupted_db.close()
        # Force more than one parallel CPU batch so ordered future handling is
        # covered without needing a large fixture.
        with patch.object(deduplication, "SIGNATURE_BATCH_SIZE", 1):
            near_db, near_removed = processor._near_deduplicate("tel", db_path)
        self.assertEqual(near_db, db_path)
        self.assertGreaterEqual(near_removed, 1)
        self.assertFalse(partition_root.exists())
        records = [{"document_id": "train", "content_hash": "a", "cleaned_text": base}, {"document_id": "test", "content_hash": "b", "cleaned_text": near}]
        # Duplicates are removed by the global verified-near graph before split
        # assignment, so this function performs only the deterministic split.
        with patch.object(assembly, "split_for", side_effect=lambda document_id, seed: document_id):
            survivors, removed = processor._decontaminate(records)
        self.assertEqual(removed, 0)
        self.assertEqual([record["document_id"] for record in survivors], ["train", "test"])
        processor.close()

    def test_signature_writes_resume_from_last_durable_frontier(self) -> None:
        processor = self.processor(workers=1)
        self.candidate([
            {
                "document_id": f"resume-{index}",
                "text": indic_text("tel", suffix=f" {chr(0x0C15 + index)}"),
                "source_id": "fixture_tel",
                "source_type": "fixture",
                "raw_file": f"raw/{index}",
            }
            for index in range(5)
        ])
        processor.clean([self.source])
        db_path, _ = processor._assemble_db("tel", processor.clean_paths("tel"))
        original = deduplication.minhash_bucket_batch
        successful_documents = 0

        def interrupted(records, seed, max_shingles):
            nonlocal successful_documents
            if successful_documents >= 2:
                raise RuntimeError("simulated runtime disconnect")
            successful_documents += len(records)
            return original(records, seed, max_shingles)

        with patch.object(deduplication, "SIGNATURE_BATCH_SIZE", 1), patch.object(deduplication, "SIGNATURE_CHECKPOINT_DOCUMENTS", 2):
            with patch.object(deduplication, "minhash_bucket_batch", side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    processor._near_deduplicate("tel", db_path)

            progress_path = db_path.with_name(db_path.stem + ".lsh-partitions") / "signatures-progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(progress["documents_written"], 2)
            drive_progress = self.root / "audit" / "fixture" / "lsh-checkpoints" / "tel" / "signatures-progress.json"
            self.assertTrue(drive_progress.exists())
            # Simulate a Colab runtime reset: local SSD partitions disappear, but
            # the small incremental Drive chunks remain available for restoration.
            shutil.rmtree(progress_path.parent)

            resumed_rowids: list[int] = []

            def counted(records, seed, max_shingles):
                resumed_rowids.extend(rowid for rowid, _ in records)
                return original(records, seed, max_shingles)

            with patch.object(deduplication, "minhash_bucket_batch", side_effect=counted):
                processor._near_deduplicate("tel", db_path)
            self.assertEqual(len(resumed_rowids), 3)
            self.assertTrue(all(rowid > progress["last_rowid"] for rowid in resumed_rowids))
            self.assertFalse(drive_progress.parent.exists())
        processor.close()

    def test_resume_reconciles_shard_and_finalization_requires_manual_fraction(self) -> None:
        processor = self.processor()
        output = self.root / "processed" / "fixture" / "tel" / "candidates" / "recovered.jsonl.gz"
        preprocess.atomic_gzip_jsonl(output, [{"document_id": "x", "text": "x"}])
        task = processor.state.task("fixture", "extract", "fixture_tel", "raw/input", "hash", str(output.relative_to(self.root)))
        self.assertEqual(task["status"], "pending")
        processor.close()
        restarted = self.processor()
        self.assertEqual(restarted.state.reconcile(self.root, "fixture"), 1)
        row = restarted.state.conn.execute("SELECT status FROM tasks WHERE raw_ref='raw/input'").fetchone()
        self.assertEqual(row[0], "complete")

        candidate = self.candidate([{"document_id": "a", "text": indic_text("tel"), "source_id": "fixture_tel", "source_type": "fixture", "raw_file": "raw/a"}])
        candidate_task = restarted.state.task(
            "fixture", "extract", "fixture_tel", "raw/a", "candidate-hash", str(candidate.relative_to(self.root))
        )
        restarted.state.complete_task(candidate_task, 1)
        restarted.clean([self.source])
        manifest = restarted.assemble("tel")
        self.assertEqual(manifest["training_status"], "draft")
        self.assertEqual(manifest["manual_train_token_fraction"], 0.0)
        with self.assertRaises(preprocess.PreprocessError):
            restarted.run(stages={"assemble"}, finalize=True)

        # A token-count-only project may explicitly waive the manual fraction,
        # but it still cannot finalize below its declared training-token floor.
        restarted.minimum_train_tokens = 1_000_000_000
        short = restarted.assemble("tel", finalize=True, allow_low_manual_fraction=True)
        self.assertEqual(short["training_status"], "draft")
        self.assertTrue(short["manual_fraction_requirement_waived"])
        self.assertFalse(short["minimum_train_tokens_met"])
        self.assertIn("requires at least 1,000,000,000", short["finalization_error"])

        restarted.minimum_train_tokens = 1
        waived = restarted.assemble("tel", finalize=True, allow_low_manual_fraction=True)
        self.assertEqual(waived["training_status"], "final")
        self.assertTrue(waived["minimum_train_tokens_met"])
        self.assertFalse(waived["manual_fraction_requirement_met"])
        self.assertTrue(waived["manual_fraction_requirement_waived"])
        self.assertIn("acquisition methods", waived["finalization_warning"])
        restarted.close()

    def test_clean_resumes_from_last_durable_in_file_checkpoint(self) -> None:
        records = [
            {
                "document_id": f"doc-{index}",
                "text": indic_text("tel", f" {chr(0x0C15 + index)}"),
                "source_id": "fixture_tel",
                "source_type": "fixture",
                "raw_file": "raw/large.parquet",
            }
            for index in range(5)
        ]
        candidate = self.candidate(records)

        predictions = 0

        def interrupted_predictor(text: str) -> tuple[str, float]:
            nonlocal predictions
            predictions += 1
            if predictions == 3:
                raise RuntimeError("simulated runtime interruption")
            return "te", 0.99

        processor = preprocess.Preprocessor(
            self.root,
            self.config,
            "fixture",
            language_predictor=interrupted_predictor,
        )
        extract_task = processor.state.task(
            "fixture",
            "extract",
            "fixture_tel",
            "raw/large.parquet",
            "candidate-hash",
            str(candidate.relative_to(self.root)),
        )
        processor.state.complete_task(extract_task, len(records))

        with patch.object(cleaning, "CLEAN_CHECKPOINT_RECORDS", 2):
            with self.assertRaises(RuntimeError):
                processor.clean([self.source])

        checkpoint_paths = list(
            (self.root / "audit" / "fixture" / "clean-checkpoints" / "tel").glob("*/progress.json")
        )
        self.assertEqual(len(checkpoint_paths), 1)
        checkpoint = json.loads(checkpoint_paths[0].read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["input_records"], 2)
        self.assertEqual(checkpoint["accepted"], 2)
        processor.close()

        restarted = self.processor()
        with patch.object(cleaning, "CLEAN_CHECKPOINT_RECORDS", 2):
            restarted.clean([self.source])
        row = restarted.state.conn.execute(
            "SELECT status,records,output_path FROM tasks WHERE run_id='fixture' AND stage='clean'"
        ).fetchone()
        self.assertEqual((row["status"], row["records"]), ("complete", 5))
        self.assertEqual(len(list(preprocess.read_jsonl_gzip(self.root / row["output_path"]))), 5)
        self.assertEqual(
            list((self.root / "audit" / "fixture" / "clean-checkpoints" / "tel").glob("*/progress.json")),
            [],
        )
        restarted.close()

    def test_completed_assembly_cleans_only_local_regenerable_scratch(self) -> None:
        processor = self.processor()
        local_root = Path(self.temp.name) / "local-ssd"
        with patch.dict(os.environ, {"LMA_LOCAL_WORK_ROOT": str(local_root)}):
            db_path = processor._local_work_db_path("tel")
            self.assertIsNotNone(db_path)
            assert db_path is not None
            db_path.parent.mkdir(parents=True)
            db_path.write_bytes(b"scratch")
            db_path.with_name(db_path.name + "-wal").write_bytes(b"scratch")
            partitions = db_path.with_name(db_path.stem + ".lsh-partitions")
            partitions.mkdir()
            (partitions / "part-000.bin").write_bytes(b"scratch")
            unrelated = db_path.parent / "keep-me.txt"
            unrelated.write_text("user file", encoding="utf-8")
            processor._cleanup_local_work("tel")
            self.assertFalse(db_path.exists())
            self.assertFalse(partitions.exists())
            self.assertTrue(unrelated.exists())
        processor.close()

    def test_drive_completed_checkpoint_replaces_older_local_copy(self) -> None:
        processor = self.processor()
        drive_db = self.root / "audit" / "fixture" / "work" / "tel.sqlite"
        drive_db.parent.mkdir(parents=True)
        local_root = Path(self.temp.name) / "local-ssd"

        def create_db(path: Path, near_complete: int | None) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT INTO metadata VALUES('input_fingerprint','same-input')")
            connection.execute("CREATE TABLE dedup_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            configuration = json.dumps({"algorithm": preprocess.PARTITIONED_LSH_ALGORITHM})
            connection.execute("INSERT INTO dedup_metadata VALUES('configuration',?)", (configuration,))
            if near_complete is not None:
                connection.execute("INSERT INTO dedup_metadata VALUES('near_complete',?)", (str(near_complete),))
            connection.commit()
            connection.close()

        create_db(drive_db, 17)
        with patch.dict(os.environ, {"LMA_LOCAL_WORK_ROOT": str(local_root)}):
            local_db = processor._local_work_db_path("tel")
            assert local_db is not None
            create_db(local_db, None)
            selected = processor._near_work_db("tel", drive_db)
            self.assertEqual(selected, local_db)
            self.assertEqual(processor._db_near_complete(local_db), 17)
        processor.close()

    def test_manual_source_requires_crawl_flag_and_real_policy(self) -> None:
        self.assertFalse(preprocess.is_manual_source(self.source, self.config.parent.parent))
        source = {
            **self.source,
            "manual_collection": True,
            "rights_basis": "open_license",
            "license_expression": "CC-BY-4.0",
            "attribution_text": "Credit the fixture source.",
            "allowed_domains": ["example.org"],
            "allowed_paths": ["/articles/"],
            "pilot_max_documents": 10,
        }
        self.assertTrue(preprocess.is_manual_source(source, self.config.parent.parent))
        source["policy_snapshot"] = "configs/policy_snapshots/ADD_POLICY_SNAPSHOT_BEFORE_APPROVAL.md"
        self.assertFalse(preprocess.is_manual_source(source, self.config.parent.parent))

    def test_tiny_bilingual_run_writes_separate_draft_manifests_and_audit(self) -> None:
        nep = {**self.source, "id": "fixture_nep", "language": "nep"}
        self.config.write_text(json.dumps({"sources": [self.source, nep]}), encoding="utf-8")
        processor = self.processor()
        tel_candidate = self.root / "processed" / "fixture" / "tel" / "candidates" / "tel.jsonl.gz"
        nep_candidate = self.root / "processed" / "fixture" / "nep" / "candidates" / "nep.jsonl.gz"
        preprocess.atomic_gzip_jsonl(tel_candidate, [{"document_id": "tel", "text": indic_text("tel"), "source_id": "fixture_tel", "source_type": "fixture", "raw_file": "raw/tel"}])
        preprocess.atomic_gzip_jsonl(nep_candidate, [{"document_id": "nep", "text": indic_text("nep"), "source_id": "fixture_nep", "source_type": "fixture", "raw_file": "raw/nep"}])
        # This predictor is deliberately a filter only; it is never passed to any
        # training component.
        processor._language_predictor = lambda text: ("te", 0.99) if preprocess.target_script_count(text, "tel") else ("ne", 0.99)
        manifests = processor.run(stages={"clean", "assemble"})
        self.assertEqual(set(manifests), {"tel", "nep"})
        for language in manifests:
            self.assertEqual(manifests[language]["training_status"], "draft")
            self.assertTrue((self.root / manifests[language]["review_samples"]).exists())
            self.assertTrue((self.root / "audit" / "fixture" / "manifests" / f"{language}.json").exists())
        processor.close()


if __name__ == "__main__":
    unittest.main()
