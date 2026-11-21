"""Tests for tokenizer gating, sampling, and vocabulary selection."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from lma_pipeline.common import atomic_gzip_jsonl
from lma_tokenizer.common import sha256_file
from lma_tokenizer.corpus import build_training_sample
from lma_tokenizer.metrics import choose_candidate, evaluate_tokenizer
from lma_tokenizer.preflight import PreprocessingNotComplete, validate_preprocessing
from lma_tokenizer.trainer import TokenizerTrainer


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.data_root = self.base / "data"
        self.data_root.mkdir()
        self.registry = self.base / "sources.json"
        self.registry.write_text(json.dumps({"sources": []}), encoding="utf-8")
        self.output_paths: dict[str, list[str]] = {}
        for split in ("train", "validation", "test"):
            relative = Path("processed") / "run" / "tel" / split / "part-00000.jsonl.gz"
            atomic_gzip_jsonl(
                self.data_root / relative,
                [{"document_id": f"{split}-1", "cleaned_text": "కఖగఘఙ చఛజఝఞ " * 100}],
            )
            self.output_paths[split] = [str(relative)]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest(self, manual_met: bool = True) -> dict:
        return {
            "schema_version": 5,
            "run_id": "run",
            "language": "tel",
            "training_status": "final",
            "minimum_train_tokens_met": True,
            "manual_fraction_requirement_met": manual_met,
            "manual_fraction_requirement_waived": not manual_met,
            "registry_sha256": sha256_file(self.registry),
            "statistics": {
                "train": {"whitespace_tokens": 500_000_000},
                "validation": {"whitespace_tokens": 100},
                "test": {"whitespace_tokens": 100},
            },
            "outputs": self.output_paths,
        }

    def write_manifest(self, value: dict) -> Path:
        path = self.data_root / "audit" / "run" / "manifests" / "tel.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_preflight_refuses_manual_waiver_and_accepts_completed_current_run(self) -> None:
        self.write_manifest(self.manifest(manual_met=False))
        with self.assertRaisesRegex(PreprocessingNotComplete, "20%"):
            validate_preprocessing(self.data_root, "run", "tel", self.registry, 500_000_000)
        self.write_manifest(self.manifest(manual_met=True))
        accepted = validate_preprocessing(self.data_root, "run", "tel", self.registry, 500_000_000)
        self.assertEqual(accepted["statistics"]["train"]["whitespace_tokens"], 500_000_000)
        self.registry.write_text(json.dumps({"sources": [{"id": "changed"}]}), encoding="utf-8")
        with self.assertRaisesRegex(PreprocessingNotComplete, "different source registry"):
            validate_preprocessing(self.data_root, "run", "tel", self.registry, 500_000_000)

    def test_explicit_manual_fraction_override_keeps_other_gates_strict(self) -> None:
        manifest = self.manifest(manual_met=False)
        manifest["training_status"] = "draft"
        manifest["manual_train_token_fraction"] = 0.1793
        self.write_manifest(manifest)
        accepted = validate_preprocessing(
            self.data_root,
            "run",
            "tel",
            self.registry,
            500_000_000,
            allow_low_manual_fraction=True,
        )
        self.assertTrue(accepted["_tokenizer_manual_fraction_override"])
        self.assertEqual(accepted["manual_train_token_fraction"], 0.1793)

        manifest["minimum_train_tokens_met"] = False
        manifest["statistics"]["train"]["whitespace_tokens"] = 499_999_999
        self.write_manifest(manifest)
        with self.assertRaisesRegex(PreprocessingNotComplete, "minimum-token"):
            validate_preprocessing(
                self.data_root,
                "run",
                "tel",
                self.registry,
                500_000_000,
                allow_low_manual_fraction=True,
            )

    def test_training_sample_is_deterministic_and_reusable(self) -> None:
        manifest_path = self.write_manifest(self.manifest())
        manifest = self.manifest()
        manifest["_manifest_sha256"] = sha256_file(manifest_path)
        manifest["statistics"]["train"]["whitespace_tokens"] = 200
        config = {
            "training_sample_whitespace_tokens": 200,
            "max_training_line_characters": 80,
        }
        messages: list[str] = []
        first_path, first_metadata = build_training_sample(
            self.data_root, "run", "tel", manifest, config, messages.append
        )
        second_path, second_metadata = build_training_sample(
            self.data_root, "run", "tel", manifest, config, messages.append
        )
        self.assertEqual(first_path, second_path)
        self.assertEqual(first_metadata["sha256"], second_metadata["sha256"])
        self.assertTrue(any("reusing" in message for message in messages))
        with gzip.open(first_path, "rt", encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle]
        self.assertTrue(lines)
        self.assertTrue(all("\n" not in line and len(line) <= 80 for line in lines))

    def test_sample_probability_uses_measured_whitespace_population(self) -> None:
        train_path = self.data_root / self.output_paths["train"][0]
        atomic_gzip_jsonl(
            train_path,
            [
                {
                    "document_id": f"train-{index}",
                    "cleaned_text": " ".join(f"word{word}" for word in range(10)),
                }
                for index in range(1_000)
            ],
        )
        manifest = self.manifest()
        # Reproduce the real mismatch: the manifest's historical Unicode-word
        # count is much larger than the actual whitespace-word population.
        manifest["statistics"]["train"]["whitespace_tokens"] = 1_000_000
        manifest_path = self.write_manifest(manifest)
        manifest["_manifest_sha256"] = sha256_file(manifest_path)
        messages: list[str] = []
        _, metadata = build_training_sample(
            self.data_root,
            "run",
            "tel",
            manifest,
            {
                "training_sample_whitespace_tokens": 1_000,
                "max_training_line_characters": 1_000,
            },
            messages.append,
        )
        self.assertEqual(metadata["measured_train_whitespace_tokens"], 10_000)
        self.assertGreaterEqual(metadata["whitespace_tokens"], 800)
        self.assertLess(metadata["probability"], 0.2)
        self.assertTrue(any("sample-calibrate" in message for message in messages))

    def test_selection_uses_smallest_vocab_near_best_fertility(self) -> None:
        def candidate(size: int, fertility: float, unknown: float) -> dict:
            return {
                "vocabulary_size": size,
                "metrics": {
                    "fertility_subwords_per_whitespace_token": fertility,
                    "unknown_token_rate": unknown,
                },
            }

        selected = choose_candidate(
            [
                candidate(16_000, 1.60, 0.0001),
                candidate(24_000, 1.50, 0.0001),
                candidate(32_000, 1.48, 0.0001),
                candidate(48_000, 1.47, 0.0001),
            ],
            maximum_unknown_rate=0.001,
            fertility_tolerance=0.02,
        )
        self.assertEqual(selected["vocabulary_size"], 32_000)

    def test_selection_refuses_to_fallback_when_every_candidate_has_unknowns(self) -> None:
        candidates = [
            {
                "vocabulary_size": size,
                "metrics": {
                    "fertility_subwords_per_whitespace_token": fertility,
                    "unknown_token_rate": unknown,
                },
            }
            for size, fertility, unknown in (
                (16_000, 1.98, 0.0023),
                (24_000, 1.86, 0.0025),
                (32_000, 1.79, 0.0026),
                (48_000, 1.70, 0.0027),
            )
        ]
        with self.assertRaisesRegex(ValueError, "no tokenizer candidate"):
            choose_candidate(candidates, maximum_unknown_rate=0.001, fertility_tolerance=0.02)

    def test_evaluation_reports_unknown_and_byte_fallback_separately(self) -> None:
        class Processor:
            pieces = ["<pad>", "<unk>", "known", "<0xE0>", "<0xA4>"]

            def unk_id(self) -> int:
                return 1

            def vocab_size(self) -> int:
                return len(self.pieces)

            def id_to_piece(self, token_id: int) -> str:
                return self.pieces[token_id]

            def encode(self, values, out_type=int, num_threads=1):
                if isinstance(values, list):
                    return [[2, 3, 4] for _ in values]
                return [2, 3, 4]

        metrics = evaluate_tokenizer(
            Processor(),
            [{"document_id": "one", "cleaned_text": "known rare"}],
        )
        self.assertEqual(metrics["unknown_tokens"], 0)
        self.assertEqual(metrics["byte_fallback_tokens"], 2)
        self.assertEqual(metrics["documents_with_byte_fallback"], 1)
        self.assertAlmostEqual(metrics["byte_fallback_token_rate"], 2 / 3)

    def test_tracked_tokenizer_configs_enable_byte_fallback(self) -> None:
        repository = Path(__file__).resolve().parent.parent
        for language in ("tel", "nep"):
            config = json.loads(
                (repository / language / "tokenizer" / "config.json").read_text(encoding="utf-8")
            )
            self.assertTrue(config["byte_fallback"])

    def test_status_marks_a_model_from_an_old_config_as_stale(self) -> None:
        config_path = self.base / "tokenizer-config.json"
        config_path.write_text(json.dumps({"language": "tel", "byte_fallback": True}), encoding="utf-8")
        trainer = TokenizerTrainer(
            data_root=self.data_root,
            run_id="run",
            language="tel",
            config_path=config_path,
            registry_path=self.registry,
        )
        trainer.output_root.mkdir(parents=True)
        model = trainer.output_root / "tokenizer.model"
        vocab = trainer.output_root / "tokenizer.vocab"
        model.write_bytes(b"model")
        vocab.write_bytes(b"vocab")
        (trainer.output_root / "tokenizer_manifest.json").write_text(
            json.dumps(
                {
                    "config": {"language": "tel", "byte_fallback": False},
                    "selected_vocabulary_size": 16_000,
                    "model_path": str(model.relative_to(trainer.data_root)),
                    "vocab_path": str(vocab.relative_to(trainer.data_root)),
                    "model_sha256": sha256_file(model),
                    "vocab_sha256": sha256_file(vocab),
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(trainer.status()["status"], "stale")


if __name__ == "__main__":
    unittest.main()
