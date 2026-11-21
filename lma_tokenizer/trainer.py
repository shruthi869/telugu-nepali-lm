"""Resume-capable, independent SentencePiece training for one language."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .common import atomic_write_bytes, atomic_write_json, sha256_file, sha256_json
from .corpus import build_training_sample, iter_split_records, materialize_sample
from .metrics import choose_candidate, evaluate_tokenizer
from .preflight import validate_preprocessing


Progress = Callable[[str], None]


class TokenizerTrainer:
    """Train and select a from-scratch tokenizer after strict corpus validation."""

    def __init__(
        self,
        data_root: Path,
        run_id: str,
        language: str,
        config_path: Path,
        registry_path: Path,
        local_work_root: Path | None = None,
        allow_low_manual_fraction: bool = False,
        progress: Progress = print,
    ) -> None:
        self.data_root = data_root.resolve()
        self.run_id = run_id
        self.language = language
        self.config_path = config_path.resolve()
        self.registry_path = registry_path.resolve()
        self.allow_low_manual_fraction = allow_low_manual_fraction
        self.progress = progress
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        if self.config.get("language") != language:
            raise ValueError(f"tokenizer config language does not match {language}: {self.config_path}")
        default_local = Path("/content/lma_tokenizer_work") if Path("/content").is_dir() else Path(tempfile.gettempdir()) / "lma_tokenizer_work"
        self.local_work_root = (local_work_root or default_local) / run_id / language
        self.output_root = self.data_root / "tokenizers" / run_id / language

    @staticmethod
    def _sentencepiece() -> Any:
        try:
            import sentencepiece as sentencepiece  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sentencepiece is required; install requirements-colab.txt") from exc
        return sentencepiece

    def preflight(self) -> dict[str, Any]:
        return validate_preprocessing(
            self.data_root,
            self.run_id,
            self.language,
            self.registry_path,
            int(self.config["minimum_train_tokens"]),
            allow_low_manual_fraction=self.allow_low_manual_fraction,
        )

    def _candidate(
        self,
        sentencepiece: Any,
        manifest: dict[str, Any],
        local_sample: Path,
        sample_metadata: dict[str, Any],
        vocabulary_size: int,
    ) -> dict[str, Any]:
        candidate_directory = self.output_root / "candidates" / f"vocab-{vocabulary_size}"
        model_path = candidate_directory / "tokenizer.model"
        vocab_path = candidate_directory / "tokenizer.vocab"
        report_path = candidate_directory / "evaluation.json"
        fingerprint_payload = {
            "schema_version": 2,
            "language": self.language,
            "manifest_sha256": manifest["_manifest_sha256"],
            "sample_sha256": sample_metadata["sha256"],
            "config": self.config,
            "sentencepiece_version": sentencepiece.__version__,
            "vocabulary_size": vocabulary_size,
        }
        fingerprint = sha256_json(fingerprint_payload)
        if model_path.exists() and vocab_path.exists() and report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if (
                    report.get("fingerprint") == fingerprint
                    and report.get("model_sha256") == sha256_file(model_path)
                    and report.get("vocab_sha256") == sha256_file(vocab_path)
                ):
                    self.progress(f"tokenizer-train: {self.language}: reusing completed vocabulary {vocabulary_size:,}")
                    return report
            except (OSError, json.JSONDecodeError):
                pass
        candidate_directory.mkdir(parents=True, exist_ok=True)
        local_candidate = self.local_work_root / f"candidate-{vocabulary_size}"
        if local_candidate.exists():
            shutil.rmtree(local_candidate)
        local_candidate.mkdir(parents=True)
        prefix = local_candidate / "tokenizer"
        self.progress(
            f"tokenizer-train: {self.language}: training from scratch; BPE vocabulary {vocabulary_size:,}"
        )
        train_function = getattr(sentencepiece.SentencePieceTrainer, "train", None) or sentencepiece.SentencePieceTrainer.Train
        train_function(
            input=str(local_sample),
            model_prefix=str(prefix),
            model_type=self.config["algorithm"],
            vocab_size=vocabulary_size,
            character_coverage=float(self.config["character_coverage"]),
            normalization_rule_name=self.config["normalization_rule_name"],
            num_threads=int(self.config["num_threads"]),
            input_sentence_size=0,
            shuffle_input_sentence=False,
            max_sentence_length=int(self.config["max_training_line_characters"]) + 64,
            hard_vocab_limit=False,
            byte_fallback=bool(self.config.get("byte_fallback", False)),
            split_digits=True,
            split_by_unicode_script=True,
            split_by_number=True,
            remove_extra_whitespaces=True,
            bos_id=2,
            eos_id=3,
            pad_id=0,
            unk_id=1,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            user_defined_symbols=list(self.config.get("user_defined_symbols", [])),
        )
        local_model = prefix.with_suffix(".model")
        local_vocab = prefix.with_suffix(".vocab")
        if not local_model.exists() or not local_vocab.exists():
            raise RuntimeError(f"SentencePiece did not create candidate {vocabulary_size} artifacts")
        atomic_write_bytes(model_path, local_model.read_bytes())
        atomic_write_bytes(vocab_path, local_vocab.read_bytes())
        processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
        self.progress(f"tokenizer-eval: {self.language}: evaluating vocabulary {vocabulary_size:,} on validation")
        metrics = evaluate_tokenizer(
            processor,
            iter_split_records(self.data_root, manifest, "validation"),
            maximum_whitespace_tokens=int(self.config.get("evaluation_max_whitespace_tokens", 0)),
            num_threads=int(self.config["num_threads"]),
        )
        report = {
            **fingerprint_payload,
            "fingerprint": fingerprint,
            "actual_vocabulary_size": int(processor.vocab_size()),
            "model_path": str(model_path.relative_to(self.data_root)),
            "vocab_path": str(vocab_path.relative_to(self.data_root)),
            "model_sha256": sha256_file(model_path),
            "vocab_sha256": sha256_file(vocab_path),
            "metrics": metrics,
        }
        atomic_write_json(report_path, report)
        self.progress(
            f"tokenizer-eval: {self.language}: vocab={vocabulary_size:,}; "
            f"fertility={metrics['fertility_subwords_per_whitespace_token']:.4f}; "
            f"unknown={metrics['unknown_token_rate']:.6%}; "
            f"bytes={metrics['byte_fallback_token_rate']:.6%}; "
            f"chars/token={metrics['average_characters_per_token']:.4f}"
        )
        shutil.rmtree(local_candidate, ignore_errors=True)
        return report

    def _markdown_report(self, manifest: dict[str, Any], selected: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
        manual_fraction = float(manifest.get("manual_train_token_fraction", 0.0))
        override = bool(manifest.get("_tokenizer_manual_fraction_override"))
        lines = [
            f"# {self.language.upper()} tokenizer report",
            "",
            "A separate SentencePiece BPE tokenizer was trained from scratch for this language.",
            "No pretrained or cross-language tokenizer artifacts were used.",
            "",
            f"- Preprocessing manifest SHA-256: `{manifest['_manifest_sha256']}`",
            f"- Preprocessing status: `{manifest.get('training_status', 'unknown')}`",
            f"- Wiki/manual training-token fraction: {manual_fraction:.2%}",
            f"- Manual-fraction override used: `{'yes' if override else 'no'}`",
            f"- Selected vocabulary: {selected['actual_vocabulary_size']:,}",
            f"- Normalization: `{self.config['normalization_rule_name']}` (the corpus was already NFC-normalized)",
            f"- Special IDs: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`",
            f"- Byte fallback: `{'enabled' if self.config.get('byte_fallback') else 'disabled'}`",
            "",
            "## Held-out validation comparison",
            "",
            "| Vocabulary | Fertility | Unknown rate | Byte-fallback rate | Characters/token | Vocabulary utilization |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for candidate in sorted(candidates, key=lambda value: value["vocabulary_size"]):
            metrics = candidate["metrics"]
            marker = " **(selected)**" if candidate["fingerprint"] == selected["fingerprint"] else ""
            lines.append(
                f"| {candidate['actual_vocabulary_size']:,}{marker} | "
                f"{metrics['fertility_subwords_per_whitespace_token']:.4f} | "
                f"{metrics['unknown_token_rate']:.6%} | "
                f"{metrics['byte_fallback_token_rate']:.6%} | "
                f"{metrics['average_characters_per_token']:.4f} | "
                f"{metrics['vocabulary_utilization']:.2%} |"
            )
        lines.extend(["", "## Tokenization examples", ""])
        for index, example in enumerate(selected["metrics"]["tokenization_examples"][:5], 1):
            pieces = " ".join(example["pieces"])
            lines.extend([f"### Example {index}", "", example["text"], "", f"`{pieces}`", ""])
        return "\n".join(lines) + "\n"

    def train(self) -> dict[str, Any]:
        """Run preflight, train missing candidates, select one, and publish artifacts."""
        manifest = self.preflight()
        sentencepiece = self._sentencepiece()
        self.output_root.mkdir(parents=True, exist_ok=True)
        sample_path, sample_metadata = build_training_sample(
            self.data_root,
            self.run_id,
            self.language,
            manifest,
            self.config,
            self.progress,
        )
        local_sample = materialize_sample(sample_path, self.local_work_root, sample_metadata["sha256"])
        candidates = [
            self._candidate(sentencepiece, manifest, local_sample, sample_metadata, int(vocabulary_size))
            for vocabulary_size in self.config["vocabulary_candidates"]
        ]
        selected = choose_candidate(
            candidates,
            float(self.config["maximum_unknown_token_rate"]),
            float(self.config["fertility_tolerance_from_best"]),
        )
        selected_model = self.data_root / selected["model_path"]
        selected_vocab = self.data_root / selected["vocab_path"]
        final_model = self.output_root / "tokenizer.model"
        final_vocab = self.output_root / "tokenizer.vocab"
        atomic_write_bytes(final_model, selected_model.read_bytes())
        atomic_write_bytes(final_vocab, selected_vocab.read_bytes())
        token_frequencies = selected["metrics"]["top_tokens"]
        examples = selected["metrics"]["tokenization_examples"]
        atomic_write_json(self.output_root / "token_frequencies.json", token_frequencies)
        atomic_write_json(self.output_root / "tokenization_examples.json", examples)
        result = {
            "schema_version": 2,
            "run_id": self.run_id,
            "language": self.language,
            "trained_from_scratch": True,
            "shared_vocabulary": False,
            "algorithm": self.config["algorithm"],
            "config": self.config,
            "config_path": str(self.config_path),
            "preprocessing_manifest": manifest["_manifest_path"],
            "preprocessing_manifest_sha256": manifest["_manifest_sha256"],
            "preprocessing_gate": {
                "training_status": manifest.get("training_status"),
                "minimum_train_tokens_met": bool(manifest.get("minimum_train_tokens_met")),
                "manual_fraction_requirement_met": bool(manifest.get("manual_fraction_requirement_met")),
                "manual_fraction_override_used": bool(manifest.get("_tokenizer_manual_fraction_override")),
                "manual_train_token_fraction": float(manifest.get("manual_train_token_fraction", 0.0)),
            },
            "training_sample": sample_metadata,
            "selection_rule": (
                "smallest vocabulary within the configured fertility tolerance of the best candidate, "
                "subject to the maximum held-out unknown-token rate"
            ),
            "selected_vocabulary_size": selected["actual_vocabulary_size"],
            "selected_metrics": selected["metrics"],
            "candidates": [
                {
                    "vocabulary_size": candidate["vocabulary_size"],
                    "actual_vocabulary_size": candidate["actual_vocabulary_size"],
                    "fingerprint": candidate["fingerprint"],
                    "model_sha256": candidate["model_sha256"],
                    "vocab_sha256": candidate["vocab_sha256"],
                    "metrics": candidate["metrics"],
                }
                for candidate in candidates
            ],
            "model_path": str(final_model.relative_to(self.data_root)),
            "vocab_path": str(final_vocab.relative_to(self.data_root)),
            "model_sha256": sha256_file(final_model),
            "vocab_sha256": sha256_file(final_vocab),
        }
        atomic_write_json(self.output_root / "tokenizer_manifest.json", result)
        atomic_write_bytes(
            self.output_root / "report.md",
            self._markdown_report(manifest, selected, candidates).encode("utf-8"),
        )
        self.progress(
            f"tokenizer: {self.language}: selected {selected['actual_vocabulary_size']:,} pieces; "
            f"model {final_model.relative_to(self.data_root)}"
        )
        return result

    def status(self) -> dict[str, Any]:
        manifest_path = self.output_root / "tokenizer_manifest.json"
        if not manifest_path.exists():
            return {"language": self.language, "status": "not-trained"}
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            model = self.data_root / value["model_path"]
            vocab = self.data_root / value["vocab_path"]
            valid = (
                model.exists()
                and vocab.exists()
                and value.get("model_sha256") == sha256_file(model)
                and value.get("vocab_sha256") == sha256_file(vocab)
            )
            current_config = value.get("config") == self.config
            return {
                "language": self.language,
                "status": "complete" if valid and current_config else "stale" if valid else "incomplete",
                "vocabulary_size": value.get("selected_vocabulary_size"),
                "model_path": value.get("model_path"),
                "reason": None if current_config else "tokenizer configuration changed; retraining required",
            }
        except (OSError, json.JSONDecodeError, KeyError):
            return {"language": self.language, "status": "incomplete"}
