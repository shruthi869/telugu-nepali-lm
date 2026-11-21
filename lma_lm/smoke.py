"""Tiny end-to-end pipeline smoke: pack → train → eval on a toy corpus."""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path
from typing import Any

from .common import LMError, atomic_write_json, emit, sha256_file
from .data import pack_language
from .evaluate import evaluate_language
from .train import select_device, train_from_packed


TINY_TEXTS = [
    "సరస్వతీ స్తుతి తెలుగు భాష పుస్తకం రచన.",
    "माघ दाङ नेपालमा समाचार लेखिएको छ।",
    "the cat sat on the mat and then ran home.",
    "0123456789 abcdefghijklmnopqrstuvwxyz.",
] + [f"extra line number {index} with more words here" for index in range(20)]


def _write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index, text in enumerate(texts):
            handle.write(
                json.dumps({"document_id": f"d{index}", "cleaned_text": text}, ensure_ascii=False) + "\n"
            )


def _train_tiny_sentencepiece(texts: list[str], model_path: Path) -> int:
    try:
        import sentencepiece as sentencepiece
    except ImportError as exc:
        raise LMError("smoke needs sentencepiece (Colab: pip install -r requirements.txt)") from exc
    model_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = model_path.parent / "smoke_corpus.txt"
    corpus.write_text("\n".join(texts * 8) + "\n", encoding="utf-8")
    prefix = model_path.with_suffix("")
    train = getattr(sentencepiece.SentencePieceTrainer, "train", None) or sentencepiece.SentencePieceTrainer.Train
    required = len(set("".join(texts))) + 8  # characters plus pad/unk/bos/eos and a few extra pieces
    vocab_size = max(256, required)
    train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=0.9995,
        bos_id=2,
        eos_id=3,
        pad_id=0,
        unk_id=1,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    produced = prefix.with_suffix(".model")
    if produced != model_path:
        model_path.write_bytes(produced.read_bytes())
    processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    return int(processor.vocab_size())


def _tiny_model_json(language: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "language": language,
        "model_label": "H" if language == "tel" else "L",
        "architecture": "decoder-only-prenorm-gpt",
        "positional_embeddings": "learned_absolute",
        "normalization": "pre_norm",
        "activation": "gelu",
        "tie_word_embeddings": True,
        "d_model": 32,
        "n_layer": 2,
        "n_head": 4,
        "d_ff": 64,
        "seq_len": 32,
        "dropout": 0.0,
        "vocab_size": None,
        "training": {
            "learning_rate": 0.003,
            "min_learning_rate": 0.0003,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "warmup_fraction": 0.1,
            "batch_size": 2,
            "grad_accum": 1,
            "eval_every": 2,
            "ckpt_every": 1,
            "eval_batches": 1,
            "keep_last": 2,
            "n_epochs": 1,
            "seed": 0,
        },
        "generation": {
            "prefix_tokens": 4,
            "max_new_tokens": 4,
            "temperatures": [0.0],
            "num_examples": 2,
        },
        "attention_examples": [TINY_TEXTS[0] if language == "tel" else TINY_TEXTS[1]],
    }


def _build_toy_run(root: Path, language: str = "tel") -> tuple[Path, Path, str]:
    """Create a miniature data_root + config tree that the real pipeline can consume."""
    run_id = "smoke"
    data_root = root / "lma_data"
    config_root = root / "repo"
    texts = TINY_TEXTS * 12
    splits = {
        "train": texts,
        "validation": texts[:8],
        "test": texts[:8],
    }
    outputs: dict[str, list[str]] = {}
    for split, rows in splits.items():
        relative = Path("processed") / run_id / language / split / "part-00000.jsonl.gz"
        _write_jsonl(data_root / relative, rows)
        outputs[split] = [str(relative)]
    tokenizer_model = data_root / "tokenizers" / run_id / language / "tokenizer.model"
    vocab_size = _train_tiny_sentencepiece(texts, tokenizer_model)
    atomic_write_json(
        tokenizer_model.with_name("tokenizer_manifest.json"),
        {
            "selected_vocabulary_size": vocab_size,
            "model_path": str(tokenizer_model.relative_to(data_root)),
            "model_sha256": sha256_file(tokenizer_model),
        },
    )
    manifest_path = data_root / "audit" / run_id / "manifests" / f"{language}.json"
    atomic_write_json(
        manifest_path,
        {
            "run_id": run_id,
            "language": language,
            "outputs": outputs,
        },
    )
    atomic_write_json(config_root / language / "configs" / "model.json", _tiny_model_json(language))
    return data_root, config_root, run_id


def run_smoke(device=None, progress=emit, **_ignored) -> dict[str, Any]:
    """Run pack, a couple of train steps, and eval on toy data. Fails only on code bugs."""
    device = device or select_device()
    progress(f"smoke: device={device}; tiny pack → train → eval")
    with tempfile.TemporaryDirectory() as tmp:
        data_root, config_root, run_id = _build_toy_run(Path(tmp))
        language = "tel"
        progress("smoke: packing")
        packed = pack_language(data_root, run_id, language, progress=progress)
        train_count = int(packed["splits"]["train"]["token_count"])
        if train_count < 40:
            raise LMError(f"smoke: packed only {train_count} train tokens; tokenizer/pack path is broken")
        progress("smoke: training 2 steps")
        trained = train_from_packed(
            data_root,
            run_id,
            language,
            device=device,
            resume=False,
            max_steps=2,
            batch_size=2,
            grad_accum=1,
            config_root=config_root,
            progress=progress,
        )
        if int(trained.get("steps") or 0) != 2:
            raise LMError(f"smoke: trainer stopped at step {trained.get('steps')}, expected 2")
        if not trained.get("latest"):
            raise LMError("smoke: trainer did not write a checkpoint")
        progress("smoke: evaluating")
        evaluated = evaluate_language(
            data_root,
            run_id,
            language,
            device=device,
            output_dir=Path(tmp) / "report",
            max_windows=2,
            config_root=config_root,
            progress=progress,
        )
        if "perplexity" not in evaluated.get("intrinsic", {}):
            raise LMError("smoke: eval did not produce perplexity")
    progress("smoke: PASS (pack, train, eval all ran on toy data)")
    return {
        "passed": True,
        "packed_train_tokens": train_count,
        "train_steps": trained["steps"],
        "eval_perplexity": evaluated["intrinsic"]["perplexity"],
        "device": str(device),
    }
