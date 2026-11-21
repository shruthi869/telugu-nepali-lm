"""Attention heatmaps, entropy, and mean attention-distance summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import emit
from .config import ModelConfig
from .data import Encoder
from .model import DecoderLM


def _ensure_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Per-head mean entropy over query positions. ``weights`` is (B, H, T, T)."""
    clamped = weights.clamp_min(1e-12)
    entropy = -(clamped * clamped.log()).sum(dim=-1)
    return entropy.mean(dim=(0, 2))


def mean_attention_distance(weights: torch.Tensor) -> torch.Tensor:
    """Average |query - key| distance weighted by attention mass, per head."""
    _batch, _heads, queries, keys = weights.shape
    query_idx = torch.arange(queries, device=weights.device).view(1, 1, queries, 1)
    key_idx = torch.arange(keys, device=weights.device).view(1, 1, 1, keys)
    distance = (query_idx - key_idx).abs().to(weights.dtype)
    mass = weights.sum(dim=-1).clamp_min(1e-12)
    return (weights * distance).sum(dim=-1).div(mass).mean(dim=(0, 2))


def causal_future_leak(weights: torch.Tensor) -> float:
    """Total attention mass above the diagonal; should be ~0 with a causal mask."""
    seq = weights.size(-1)
    future = torch.triu(torch.ones(seq, seq, device=weights.device, dtype=torch.bool), diagonal=1)
    return float(weights[:, :, future].sum().item())


def encode_example(processor: Encoder, text: str, seq_len: int) -> list[int]:
    ids = [int(token) for token in processor.encode(text, out_type=int)]
    if not ids:
        ids = [int(processor.eos_id())]
    return ids[:seq_len]


def save_heatmap(
    weights: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
) -> None:
    plt = _ensure_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    image = axis.imshow(weights, cmap="viridis", aspect="auto", vmin=0.0, vmax=max(float(weights.max()), 1e-6))
    axis.set_title(title)
    axis.set_xlabel("Key position")
    axis.set_ylabel("Query position")
    ticks = range(len(labels))
    axis.set_xticks(list(ticks))
    axis.set_yticks(list(ticks))
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Attention weight")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_head_summary(
    entropy: np.ndarray,
    distance: np.ndarray,
    path: Path,
    title: str,
) -> None:
    plt = _ensure_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    layers, heads = entropy.shape
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, values, label in (
        (axes[0], entropy, "Mean attention entropy (nats)"),
        (axes[1], distance, "Mean attention distance (tokens)"),
    ):
        image = axis.imshow(values, cmap="magma", aspect="auto")
        axis.set_title(label)
        axis.set_xlabel("Head")
        axis.set_ylabel("Layer")
        axis.set_xticks(range(heads))
        axis.set_yticks(range(layers))
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_causal_mask_figure(seq_len: int, path: Path) -> None:
    plt = _ensure_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.tril(np.ones((seq_len, seq_len)))
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    image = axis.imshow(mask, cmap="Greys", vmin=0, vmax=1, aspect="equal")
    axis.set_title("Causal mask (1 = allowed, 0 = blocked future)")
    axis.set_xlabel("Key position")
    axis.set_ylabel("Query position")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


@torch.no_grad()
def analyze_attention(
    model: DecoderLM,
    processor: Encoder,
    config: ModelConfig,
    device: torch.device,
    output_dir: Path,
    texts: list[str] | None = None,
    progress=emit,
) -> dict[str, Any]:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = texts or config.attention_examples or ["placeholder"]
    text = examples[0]
    token_ids = encode_example(processor, text, config.seq_len)
    pieces = []
    for token_id in token_ids:
        if hasattr(processor, "id_to_piece"):
            piece = str(processor.id_to_piece(token_id))
        else:
            piece = processor.decode([token_id])
        pieces.append(piece.replace("▁", "␣") or "·")
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    _logits, _loss, attentions = model(idx, return_attentions=True)
    n_layer = len(attentions)
    n_head = attentions[0].size(1)
    entropy = torch.stack([attention_entropy(layer) for layer in attentions], dim=0)
    distance = torch.stack([mean_attention_distance(layer) for layer in attentions], dim=0)
    leak = max(causal_future_leak(layer) for layer in attentions)
    entropy_np = entropy.detach().cpu().numpy()
    distance_np = distance.detach().cpu().numpy()
    save_head_summary(
        entropy_np,
        distance_np,
        output_dir / "attention_entropy_distance.png",
        f"{config.language} attention entropy and distance",
    )
    early = 0
    late = n_layer - 1
    figures = [str(output_dir / "attention_entropy_distance.png")]
    selected_heads = [0, min(1, n_head - 1), n_head - 1]
    for layer_index in (early, late):
        layer_weights = attentions[layer_index][0].detach().cpu().numpy()
        for head in selected_heads:
            path = output_dir / f"heatmap_layer{layer_index:02d}_head{head:02d}.png"
            save_heatmap(
                layer_weights[head],
                pieces,
                path,
                f"{config.language} layer {layer_index} head {head}",
            )
            figures.append(str(path))
    mask_path = output_dir / "causal_mask.png"
    save_causal_mask_figure(min(len(token_ids), 32), mask_path)
    figures.append(str(mask_path))
    summary = {
        "example_text": text,
        "n_tokens": len(token_ids),
        "n_layer": n_layer,
        "n_head": n_head,
        "entropy_per_layer_head": entropy_np.tolist(),
        "distance_per_layer_head": distance_np.tolist(),
        "mean_entropy_early": float(entropy_np[early].mean()),
        "mean_entropy_late": float(entropy_np[late].mean()),
        "mean_distance_early": float(distance_np[early].mean()),
        "mean_distance_late": float(distance_np[late].mean()),
        "causal_future_mass": leak,
        "local_heads": [
            {"layer": int(layer), "head": int(head)}
            for layer in range(n_layer)
            for head in range(n_head)
            if distance_np[layer, head] <= 2.0
        ],
        "long_range_heads": [
            {"layer": int(layer), "head": int(head)}
            for layer in range(n_layer)
            for head in range(n_head)
            if distance_np[layer, head] >= max(len(token_ids) * 0.25, 3.0)
        ],
    }
    progress(
        f"attention: {config.language}: leak={leak:.2e} "
        f"early_entropy={summary['mean_entropy_early']:.3f} "
        f"late_entropy={summary['mean_entropy_late']:.3f}"
    )
    return {"summary": summary, "figures": figures}
