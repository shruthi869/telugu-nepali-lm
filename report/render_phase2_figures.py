#!/usr/bin/env python3
"""Write architecture figures that do not require a trained checkpoint."""

from __future__ import annotations

from pathlib import Path

from lma_lm.attention import save_causal_mask_figure
from lma_lm.config import load_language_config
from lma_lm.model import DecoderLM, parameter_breakdown


def parameter_bar(output: Path) -> dict[str, int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_language_config("tel", vocab_size=48000)
    model = DecoderLM(config)
    breakdown = parameter_breakdown(model)
    labels = [
        "Token embeddings",
        "Position embeddings",
        "Transformer blocks",
        "Final LayerNorm",
        "Tied LM head",
    ]
    values = [
        breakdown["token_embeddings"],
        breakdown["position_embeddings"],
        breakdown["transformer_blocks"],
        breakdown["final_layer_norm"],
        breakdown["output_head"],
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.4, 4.4))
    axis.bar(labels, [value / 1e6 for value in values], color="#087E8B")
    axis.set_ylabel("Parameters (millions)")
    axis.set_title(f"Telugu/Nepali unique parameters: {breakdown['total_unique']:,}")
    axis.set_ylim(0, max(values) / 1e6 * 1.2)
    plt.xticks(rotation=20, ha="right")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    (output.parent / "parameter_count.json").write_text(
        __import__("json").dumps(breakdown, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return breakdown


def main() -> int:
    root = Path(__file__).resolve().parent / "figures"
    save_causal_mask_figure(32, root / "causal_mask.png")
    breakdown = parameter_bar(root / "parameter_breakdown.png")
    print(f"unique parameters: {breakdown['total_unique']:,}", flush=True)
    print(f"wrote figures under {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
