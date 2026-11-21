"""CPU tests for the Phase 2 decoder-only LM, packing, checkpoints, and metrics."""

from __future__ import annotations

import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lma_lm.attention import analyze_attention, causal_future_leak, save_causal_mask_figure
from lma_lm.checkpoint import apply_checkpoint, load_checkpoint, save_checkpoint
from lma_lm.config import ModelConfig, load_language_config
from lma_lm.data import PackedTokens, encode_document, pack_split
from lma_lm.evaluate import bits_per_byte, distinct_n, repetition_rate, rouge_l
from lma_lm.model import DecoderLM, count_parameters, parameter_breakdown
from lma_lm.smoke import run_smoke
from lma_lm.train import build_scheduler, configure_optimizer


class FakeProcessor:
    def __init__(self, vocab_size: int = 64) -> None:
        self._vocab = vocab_size

    def vocab_size(self) -> int:
        return self._vocab

    def eos_id(self) -> int:
        return 3

    def pad_id(self) -> int:
        return 0

    def bos_id(self) -> int:
        return 2

    def unk_id(self) -> int:
        return 1

    def encode(self, text: str, out_type: type = int) -> list[int]:
        ids = [4 + (ord(char) % (self._vocab - 5)) for char in text if not char.isspace()]
        return ids[:24] or [4]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(65 + (int(token) % 26)) for token in ids if int(token) > 3)

    def id_to_piece(self, token_id: int) -> str:
        if token_id <= 3:
            return f"<{token_id}>"
        return chr(65 + (int(token_id) % 26))


def tiny_config(**overrides) -> ModelConfig:
    payload = dict(
        language="tel",
        vocab_size=64,
        d_model=32,
        n_layer=2,
        n_head=4,
        d_ff=64,
        seq_len=16,
        dropout=0.0,
    )
    payload.update(overrides)
    return ModelConfig(**payload)


def write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index, text in enumerate(texts):
            handle.write(json.dumps({"document_id": f"d{index}", "cleaned_text": text}, ensure_ascii=False) + "\n")


class ModelTests(unittest.TestCase):
    def test_forward_shapes_and_loss(self) -> None:
        config = tiny_config()
        model = DecoderLM(config)
        idx = torch.randint(0, config.vocab_size, (3, 11))
        targets = torch.randint(0, config.vocab_size, (3, 11))
        logits, loss = model(idx, targets)
        self.assertEqual(tuple(logits.shape), (3, 11, config.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_causal_mask_blocks_the_future(self) -> None:
        torch.manual_seed(0)
        config = tiny_config()
        model = DecoderLM(config)
        model.eval()
        x = torch.randint(0, config.vocab_size, (1, 12))
        with torch.no_grad():
            logits_a, _ = model(x)
            mutated = x.clone()
            mutated[0, 7] = (mutated[0, 7] + 1) % config.vocab_size
            logits_b, _ = model(mutated)
            _logits, _loss, attentions = model(x, return_attentions=True)
        self.assertTrue(torch.allclose(logits_a[0, :7], logits_b[0, :7], atol=1e-5, rtol=1e-5))
        self.assertFalse(torch.allclose(logits_a[0, 7:], logits_b[0, 7:], atol=1e-5, rtol=1e-5))
        for layer in attentions:
            self.assertLess(causal_future_leak(layer), 1e-5)

    def test_weight_tying_and_parameter_count(self) -> None:
        config = ModelConfig(
            language="tel",
            vocab_size=48000,
            d_model=384,
            n_layer=4,
            n_head=6,
            d_ff=1536,
            seq_len=128,
            dropout=0.1,
        )
        model = DecoderLM(config)
        self.assertIs(model.lm_head.weight, model.wte.weight)
        unique = count_parameters(model, unique=True)
        doubled = count_parameters(model, unique=False)
        self.assertGreater(doubled, unique)
        self.assertEqual(unique, 25_579_776)
        breakdown = parameter_breakdown(model)
        self.assertEqual(breakdown["total_unique"], unique)
        self.assertTrue(breakdown["output_head_tied"])
        self.assertEqual(breakdown["output_head"], 0)
        self.assertEqual(breakdown["n_layer"], 4)
        self.assertEqual(breakdown["seq_len"], 128)

    def test_language_configs_share_architecture(self) -> None:
        tel = load_language_config("tel", vocab_size=48000)
        nep = load_language_config("nep", vocab_size=48000)
        self.assertEqual(tel.d_model, nep.d_model)
        self.assertEqual(tel.n_layer, nep.n_layer)
        self.assertEqual(tel.seq_len, 128)
        self.assertEqual(tel.n_layer, 4)
        self.assertEqual(tel.training.batch_size, 128)
        self.assertEqual(tel.model_label, "H")
        self.assertEqual(nep.model_label, "L")
        self.assertNotEqual(tel.attention_examples, nep.attention_examples)

    def test_no_position_ablation_drops_wpe_and_stays_causal(self) -> None:
        torch.manual_seed(0)
        config = tiny_config(positional_embeddings="none")
        model = DecoderLM(config)
        self.assertIsNone(model.wpe)
        self.assertFalse(any("wpe" in name for name, _ in model.named_parameters()))
        idx = torch.randint(0, config.vocab_size, (2, 9))
        logits, loss = model(idx, idx)
        self.assertEqual(tuple(logits.shape), (2, 9, config.vocab_size))
        self.assertTrue(torch.isfinite(loss))
        model.eval()
        x = torch.randint(0, config.vocab_size, (1, 12))
        with torch.no_grad():
            logits_a, _ = model(x)
            mutated = x.clone()
            mutated[0, 7] = (mutated[0, 7] + 1) % config.vocab_size
            logits_b, _ = model(mutated)
            _logits, _loss, attentions = model(x, return_attentions=True)
        self.assertTrue(torch.allclose(logits_a[0, :7], logits_b[0, :7], atol=1e-5, rtol=1e-5))
        self.assertFalse(torch.allclose(logits_a[0, 7:], logits_b[0, 7:], atol=1e-5, rtol=1e-5))
        for layer in attentions:
            self.assertLess(causal_future_leak(layer), 1e-5)

    def test_no_position_full_size_parameter_count(self) -> None:
        learned = DecoderLM(
            ModelConfig(
                language="nep",
                vocab_size=48000,
                d_model=384,
                n_layer=4,
                n_head=6,
                d_ff=1536,
                seq_len=128,
            )
        )
        nopos = DecoderLM(
            ModelConfig(
                language="nep",
                vocab_size=48000,
                d_model=384,
                n_layer=4,
                n_head=6,
                d_ff=1536,
                seq_len=128,
                positional_embeddings="none",
            )
        )
        dropped = 128 * 384
        self.assertEqual(count_parameters(learned) - count_parameters(nopos), dropped)
        self.assertEqual(count_parameters(nopos), 25_530_624)
        self.assertEqual(parameter_breakdown(nopos)["position_embeddings"], 0)
        self.assertEqual(parameter_breakdown(nopos)["positional_embeddings"], "none")

    def test_nepali_nopos_config_loads(self) -> None:
        config = load_language_config("nep", vocab_size=48000, no_positions=True)
        self.assertEqual(config.positional_embeddings, "none")
        self.assertFalse(config.uses_positional_embeddings)
        self.assertEqual(config.language, "nep")
        self.assertEqual(config.d_model, 384)


class CheckpointTests(unittest.TestCase):
    def test_round_trip_restores_logits_and_step(self) -> None:
        config = tiny_config()
        model = DecoderLM(config)
        optimizer = configure_optimizer(model, config)
        scheduler = build_scheduler(optimizer, 100, 5, 3e-4, 3e-5)
        idx = torch.randint(0, config.vocab_size, (2, 8))
        model.eval()
        with torch.no_grad():
            before, _ = model(idx)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_checkpoint(path, model, optimizer, scheduler, None, step=17, tokens_seen=4096, best_val_loss=3.2)
            payload = load_checkpoint(path)
            restored = DecoderLM(config)
            restored_opt = configure_optimizer(restored, config)
            restored_sched = build_scheduler(restored_opt, 100, 5, 3e-4, 3e-5)
            apply_checkpoint(payload, restored, restored_opt, restored_sched, restore_random=False)
            restored.eval()
            with torch.no_grad():
                after, _ = restored(idx)
            self.assertTrue(torch.allclose(before, after, atol=1e-6))
            self.assertEqual(payload["step"], 17)
            self.assertEqual(payload["tokens_seen"], 4096)
            self.assertIsNotNone(payload["scheduler"])
            self.assertEqual(payload["config"]["language"], "tel")


class PackingTests(unittest.TestCase):
    def test_encode_appends_eos_once(self) -> None:
        processor = FakeProcessor()
        ids = encode_document(processor, "abc", eos_id=3)
        self.assertEqual(ids[-1], 3)
        self.assertNotEqual(ids[-2], 3)

    def test_pack_is_deterministic_and_resumable(self) -> None:
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = Path("processed/run/tel/train/part-00000.jsonl.gz")
            shard_b = Path("processed/run/tel/train/part-00001.jsonl.gz")
            write_jsonl(root / shard_a, ["aaaa bbbb", "cccc"])
            write_jsonl(root / shard_b, ["dddd eeee"])
            manifest = {
                "_manifest_sha256": "abc",
                "outputs": {"train": [str(shard_a), str(shard_b)]},
            }
            first = pack_split(root, "run", "tel", "train", processor, manifest, "tokhash")
            second = pack_split(root, "run", "tel", "train", processor, manifest, "tokhash")
            self.assertEqual(first["token_count"], second["token_count"])
            self.assertEqual(first["status"], "complete")
            packed = PackedTokens(root / "tokenized/run/tel/train.bin", seq_len=4)
            self.assertGreaterEqual(len(packed), first["token_count"])
            self.assertEqual(packed.data.dtype, np.uint16)

    def test_resume_skips_completed_shard(self) -> None:
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_a = Path("processed/run/tel/train/part-00000.jsonl.gz")
            shard_b = Path("processed/run/tel/train/part-00001.jsonl.gz")
            write_jsonl(root / shard_a, ["aaaa"])
            write_jsonl(root / shard_b, ["bbbb cccc"])
            first_manifest = {"_manifest_sha256": "abc", "outputs": {"train": [str(shard_a)]}}
            first = pack_split(root, "run", "tel", "train", processor, first_manifest, "tokhash")
            # Same identity (manifest hash kept) but a second shard appears in the list.
            resumed_manifest = {"_manifest_sha256": "abc", "outputs": {"train": [str(shard_a), str(shard_b)]}}
            second = pack_split(root, "run", "tel", "train", processor, resumed_manifest, "tokhash")
            self.assertGreater(second["token_count"], first["token_count"])
            self.assertEqual(len(second["completed_shards"]), 2)


class MetricTests(unittest.TestCase):
    def test_rouge_and_diversity(self) -> None:
        self.assertGreater(rouge_l("the cat sat", "the cat sat"), 0.99)
        self.assertGreater(rouge_l("the cat sat on the mat", "the cat sat"), 0.5)
        self.assertEqual(distinct_n(["a b c", "a b d"], 1), 4 / 6)
        self.assertGreater(repetition_rate(["yes yes yes yes"], 2), 0.0)

    def test_bits_per_byte(self) -> None:
        nll = math.log(2.0) * 8
        self.assertAlmostEqual(bits_per_byte(nll, 8), 1.0)


class SmokeAndAttentionTests(unittest.TestCase):
    def test_smoke_pipeline_on_toy_data(self) -> None:
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            self.skipTest("sentencepiece is required for the pipeline smoke test")
        result = run_smoke(device=torch.device("cpu"))
        self.assertTrue(result["passed"])
        self.assertEqual(result["train_steps"], 2)
        self.assertGreater(result["packed_train_tokens"], 40)

    def test_attention_analysis_writes_figures(self) -> None:
        config = tiny_config()
        model = DecoderLM(config)
        processor = FakeProcessor()
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze_attention(
                model,
                processor,
                config,
                torch.device("cpu"),
                Path(tmp),
                texts=["hello world"],
            )
            self.assertLess(result["summary"]["causal_future_mass"], 1e-5)
            self.assertTrue(any(Path(path).is_file() for path in result["figures"]))
            mask = Path(tmp) / "causal_mask.png"
            save_causal_mask_figure(16, mask)
            self.assertTrue(mask.is_file())


if __name__ == "__main__":
    unittest.main()
