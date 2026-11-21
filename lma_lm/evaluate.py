"""Held-out language-modeling, generation, and n-gram evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import apply_checkpoint, best_checkpoint, latest_checkpoint, load_checkpoint
from .common import LMError, atomic_write_json, emit, positional_ablation, repository_root
from .data import PackedTokens, config_with_tokenizer, load_packed_meta, materialize_split
from .model import DecoderLM, count_parameters
from .train import _autocast, batch_to_device, select_device


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, 1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 with the standard recall bias (beta = 1.2) on whitespace tokens."""
    hyp = hypothesis.split()
    ref = reference.split()
    if not hyp or not ref:
        return 0.0
    overlap = _lcs_length(hyp, ref)
    precision = overlap / len(hyp)
    recall = overlap / len(ref)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = 1.2 ** 2
    return (1.0 + beta2) * precision * recall / (recall + beta2 * precision)


def distinct_n(texts: Iterable[str], order: int) -> float:
    ngrams: set[tuple[str, ...]] = set()
    total = 0
    for text in texts:
        tokens = text.split()
        if len(tokens) < order:
            continue
        for index in range(len(tokens) - order + 1):
            ngrams.add(tuple(tokens[index : index + order]))
            total += 1
    if total == 0:
        return 0.0
    return len(ngrams) / total


def repetition_rate(texts: Iterable[str], order: int = 3) -> float:
    repeated = 0
    total = 0
    for text in texts:
        tokens = text.split()
        seen: set[tuple[str, ...]] = set()
        for index in range(max(len(tokens) - order + 1, 0)):
            gram = tuple(tokens[index : index + order])
            total += 1
            if gram in seen:
                repeated += 1
            seen.add(gram)
    if total == 0:
        return 0.0
    return repeated / total


def corpus_bleu_chrf(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    try:
        import sacrebleu
    except ImportError as exc:
        raise LMError("sacrebleu is required for BLEU/chrF; pip install sacrebleu") from exc
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="intl")
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return {
        "bleu4": float(bleu.score),
        "chrf": float(chrf.score),
        "bleu_signature": str(bleu),
        "chrf_signature": str(chrf),
    }


def bits_per_byte(nll_nats: float, utf8_bytes: int) -> float:
    if utf8_bytes <= 0:
        return float("inf")
    return (nll_nats / math.log(2.0)) / utf8_bytes


@torch.no_grad()
def packed_nll(
    model: DecoderLM,
    tokens: PackedTokens,
    batch_size: int,
    device: torch.device,
    max_windows: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    starts = list(range(0, tokens.n_windows, tokens.seq_len))
    if max_windows is not None:
        starts = starts[:max_windows]
    for index in range(0, len(starts), batch_size):
        chunk = np.asarray(starts[index : index + batch_size], dtype=np.int64)
        inputs, targets = tokens.batch(chunk)
        idx, tgt = batch_to_device(inputs, targets, device)
        with _autocast(device, enabled=device.type == "cuda"):
            logits, _loss = model(idx)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="sum",
            )
        total_nll += float(nll.item())
        total_tokens += int(tgt.numel())
    mean = total_nll / max(total_tokens, 1)
    return {
        "nll_nats": total_nll,
        "token_count": total_tokens,
        "cross_entropy": mean,
        "perplexity": float(math.exp(min(mean, 20.0))),
    }


@torch.no_grad()
def generate_from_prefix(
    model: DecoderLM,
    prefix: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    eos_id: int,
) -> list[int]:
    output = model.generate(prefix, max_new_tokens=max_new_tokens, temperature=temperature, eos_id=eos_id)
    return [int(token) for token in output[0].tolist()]


def decode_ids(processor, token_ids: list[int], eos_id: int) -> str:
    trimmed = []
    for token in token_ids:
        if token == eos_id and trimmed:
            break
        if token == eos_id:
            continue
        trimmed.append(token)
    if not trimmed:
        return ""
    return processor.decode(trimmed)


def collect_generation_examples(
    model: DecoderLM,
    processor,
    val_tokens: PackedTokens,
    config,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, Any]:
    gen = config.generation
    eos_id = config.eos_id
    examples = []
    hypotheses: dict[str, list[str]] = {}
    references: list[str] = []
    n_windows = val_tokens.n_windows
    starts = rng.integers(0, max(n_windows, 1), size=min(gen.num_examples, max(n_windows, 1)))
    temperatures = list(gen.temperatures)
    for start in starts:
        inputs, targets = val_tokens.window(int(start))
        prefix_len = min(gen.prefix_tokens, config.seq_len - 1)
        prefix_ids = [int(token) for token in inputs[:prefix_len]]
        reference_ids = [int(token) for token in targets[prefix_len - 1 : prefix_len - 1 + gen.max_new_tokens]]
        prefix_text = decode_ids(processor, prefix_ids, eos_id)
        reference_text = decode_ids(processor, reference_ids, eos_id)
        references.append(reference_text)
        record: dict[str, Any] = {"prefix": prefix_text, "reference": reference_text, "continuations": {}}
        prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        for temperature in temperatures:
            label = "greedy" if temperature <= 0 else f"t{temperature:.1f}"
            generated = generate_from_prefix(model, prefix_tensor, gen.max_new_tokens, temperature, eos_id)
            continuation_ids = generated[len(prefix_ids) :]
            text = decode_ids(processor, continuation_ids, eos_id)
            record["continuations"][label] = text
            hypotheses.setdefault(label, []).append(text)
        examples.append(record)
    metrics = {}
    for label, hyps in hypotheses.items():
        scores = corpus_bleu_chrf(hyps, references)
        scores["rouge_l"] = float(np.mean([rouge_l(hyp, ref) for hyp, ref in zip(hyps, references)]))
        scores["distinct_1"] = distinct_n(hyps, 1)
        scores["distinct_2"] = distinct_n(hyps, 2)
        scores["repetition_rate"] = repetition_rate(hyps, 3)
        metrics[label] = scores
    return {"examples": examples, "metrics": metrics}


def evaluate_language(
    data_root: Path,
    run_id: str,
    language: str,
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
    output_dir: Path | None = None,
    max_windows: int | None = 2_000,
    config_root: Path | None = None,
    no_positions: bool = False,
    progress=emit,
) -> dict[str, Any]:
    device = device or select_device()
    config, processor, tokenizer_manifest = config_with_tokenizer(
        language, data_root, run_id, root=config_root, no_positions=no_positions
    )
    ablation = positional_ablation(config.positional_embeddings)
    meta = load_packed_meta(data_root, run_id, language)
    chosen = (
        checkpoint_path
        or best_checkpoint(data_root, run_id, language, ablation=ablation)
        or latest_checkpoint(data_root, run_id, language, ablation=ablation)
    )
    if chosen is None:
        raise LMError(f"no checkpoint for {language}; train_lm.py must finish at least one save")
    progress(f"eval: {language}: loading {chosen}")
    payload = load_checkpoint(chosen, map_location=device)
    ckpt_pos = (payload.get("config") or {}).get("positional_embeddings")
    if ckpt_pos:
        config.positional_embeddings = ckpt_pos
        ablation = positional_ablation(config.positional_embeddings)
    model = DecoderLM(config).to(device)
    apply_checkpoint(payload, model, restore_random=False)
    model.eval()
    val_tokens = materialize_split(data_root, run_id, language, "validation", config.seq_len, progress=progress)
    nll = packed_nll(model, val_tokens, batch_size=config.training.batch_size, device=device, max_windows=max_windows)
    utf8_bytes = int(meta["splits"]["validation"].get("utf8_bytes") or 0)
    # Packed windows include EOS tokens; scale NLL to the recorded UTF-8 byte count.
    token_fraction = nll["token_count"] / max(int(meta["splits"]["validation"]["token_count"]), 1)
    estimated_bytes = max(int(utf8_bytes * token_fraction), 1)
    bpb = bits_per_byte(nll["nll_nats"], estimated_bytes)
    rng = np.random.default_rng(config.training.seed + 7)
    generation = collect_generation_examples(model, processor, val_tokens, config, device, rng)
    from .attention import analyze_attention

    destination = output_dir or repository_root() / "report"
    if ablation:
        if destination.name == "figures":
            attention_dir = destination / language / ablation
        else:
            attention_dir = destination / "figures" / language / ablation
        eval_name = f"phase2_{language}_{ablation}_eval.json"
    else:
        attention_dir = (output_dir or repository_root() / "report" / "figures") / language
        eval_name = f"phase2_{language}_eval.json"

    attention = analyze_attention(
        model,
        processor,
        config,
        device,
        output_dir=attention_dir,
        progress=progress,
    )
    result = {
        "language": language,
        "run_id": run_id,
        "ablation": ablation,
        "positional_embeddings": config.positional_embeddings,
        "checkpoint": str(chosen),
        "parameter_count": count_parameters(model),
        "tokenizer_sha256": meta.get("tokenizer_sha256"),
        "tokenizer_vocab_size": tokenizer_manifest.get("selected_vocabulary_size", config.vocab_size),
        "intrinsic": {
            "cross_entropy": nll["cross_entropy"],
            "perplexity": nll["perplexity"],
            "bits_per_byte": bpb,
            "eval_tokens": nll["token_count"],
            "eval_utf8_bytes_estimate": estimated_bytes,
        },
        "generation": generation["metrics"],
        "samples": generation["examples"][:8],
        "attention": attention["summary"],
        "attention_figures": attention["figures"],
    }
    atomic_write_json(destination / eval_name, result)
    progress(
        f"eval: {language}: ppl={nll['perplexity']:.2f} bpb={bpb:.4f} "
        f"params={count_parameters(model):,}"
        + (f" ablation={ablation}" if ablation else "")
    )
    return result
