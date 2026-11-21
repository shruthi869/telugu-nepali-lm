#!/usr/bin/env python3
"""Build Phase 2 result figures from eval JSON and the Nepali train log.

Pixel heatmaps were written on Kaggle under /kaggle/working/lma/report/{tel,nep}/
and did not land in the download zip. Entropy/distance grids here are rebuilt
from the numbers stored in phase2_{lang}_eval.json.

Uses matplotlib when it is installed; otherwise a stdlib PNG writer so the
report figures can be rebuilt without a local torch/matplotlib stack.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"

TEAL = (8, 126, 139)
NAVY = (23, 50, 77)
WHITE = (255, 255, 255)
INK = (28, 32, 38)
GRID = (220, 226, 230)
LIGHT = (243, 246, 248)


def _try_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        return plt, np
    except ImportError:
        return None, None


# 5x7 uppercase-ish glyphs for axis labels. Unknown chars are skipped.
_FONT = {
    " ": [0, 0, 0, 0, 0],
    "-": [0, 0, 31, 0, 0],
    ",": [0, 1, 2, 0, 0],
    ".": [0, 0, 16, 0, 0],
    "/": [16, 8, 4, 2, 1],
    ">": [17, 10, 4, 0, 0],
    "(": [14, 17, 0, 0, 0],
    ")": [17, 14, 0, 0, 0],
    "0": [14, 17, 17, 17, 14],
    "1": [0, 9, 31, 1, 0],
    "2": [9, 19, 21, 9, 0],
    "3": [17, 21, 21, 14, 0],
    "4": [6, 10, 31, 2, 0],
    "5": [29, 21, 21, 18, 0],
    "6": [14, 21, 21, 18, 0],
    "7": [16, 16, 19, 28, 0],
    "8": [10, 21, 21, 10, 0],
    "9": [8, 21, 21, 14, 0],
    "=": [10, 10, 10, 10, 0],
    "A": [15, 20, 20, 15, 0],
    "B": [31, 21, 21, 10, 0],
    "C": [14, 17, 17, 10, 0],
    "D": [31, 17, 17, 14, 0],
    "E": [31, 21, 21, 17, 0],
    "F": [31, 20, 20, 16, 0],
    "G": [14, 17, 21, 7, 0],
    "H": [31, 4, 4, 31, 0],
    "I": [17, 31, 17, 0, 0],
    "K": [31, 4, 10, 17, 0],
    "L": [31, 1, 1, 1, 0],
    "M": [31, 8, 4, 8, 31],
    "N": [31, 8, 4, 2, 31],
    "O": [14, 17, 17, 14, 0],
    "P": [31, 20, 20, 8, 0],
    "Q": [14, 17, 19, 15, 0],
    "R": [31, 20, 22, 9, 0],
    "S": [9, 21, 21, 18, 0],
    "T": [16, 16, 31, 16, 16],
    "U": [30, 1, 1, 30, 0],
    "V": [28, 2, 1, 2, 28],
    "W": [31, 2, 4, 2, 31],
    "X": [17, 10, 4, 10, 17],
    "Y": [16, 8, 7, 8, 16],
    "Z": [17, 19, 21, 25, 17],
}


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0])
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for red, green, blue in row:
            raw.extend((red, green, blue))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _png_chunk(b"IEND", b"")
    )


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def magma(t: float) -> tuple[int, int, int]:
    stops = [(0.0, (0, 0, 4)), (0.25, (80, 18, 123)), (0.5, (188, 55, 84)), (0.75, (249, 142, 8)), (1.0, (252, 253, 191))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            return _lerp(c0, c1, 0.0 if t1 == t0 else (t - t0) / (t1 - t0))
    return stops[-1][1]


class Canvas:
    def __init__(self, width: int, height: int, fill: tuple[int, int, int] = WHITE) -> None:
        self.w = width
        self.h = height
        self.p = [[fill for _ in range(width)] for _ in range(height)]

    def fill_rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        for yy in range(max(0, y), min(self.h, y + h)):
            row = self.p[yy]
            for xx in range(max(0, x), min(self.w, x + w)):
                row[xx] = color

    def line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 1) -> None:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            self.fill_rect(x0 - width // 2, y0 - width // 2, width, width, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def text(self, x: int, y: int, message: str, color: tuple[int, int, int] = INK, scale: int = 2) -> None:
        cursor = x
        for char in message.upper():
            glyph = _FONT.get(char)
            if glyph is None:
                cursor += 6 * scale
                continue
            for col, bits in enumerate(glyph):
                for row in range(7):
                    if bits & (1 << (6 - row)):
                        self.fill_rect(cursor + col * scale, y + row * scale, scale, scale, color)
            cursor += 6 * scale

    def save(self, path: Path) -> None:
        write_png(path, self.p)


def save_causal_mask_stdlib(seq_len: int, path: Path) -> None:
    cell = 12
    pad = 56
    canvas = Canvas(pad + seq_len * cell + 24, pad + seq_len * cell + 40, WHITE)
    canvas.text(pad, 12, "Causal mask 1=allowed 0=blocked", NAVY, 2)
    for i in range(seq_len):
        for j in range(seq_len):
            color = (40, 40, 40) if j <= i else (235, 235, 235)
            canvas.fill_rect(pad + j * cell, pad + i * cell, cell - 1, cell - 1, color)
    canvas.text(pad, pad + seq_len * cell + 8, "Key position", INK, 2)
    canvas.save(path)


def save_parameter_bar_stdlib(path: Path) -> None:
    labels = ["Tok emb", "Pos emb", "Blocks", "Final LN", "Tied head"]
    values = [18.432, 0.049, 7.098, 0.001, 0.0]
    canvas = Canvas(900, 420, WHITE)
    canvas.fill_rect(0, 0, 900, 420, WHITE)
    canvas.text(24, 18, "Unique parameters 25,579,776", NAVY, 2)
    origin_x, origin_y, plot_w, plot_h = 80, 70, 780, 280
    canvas.fill_rect(origin_x, origin_y, plot_w, plot_h, LIGHT)
    bar_w = plot_w // len(labels) - 24
    peak = max(values) * 1.15
    for i, (label, value) in enumerate(zip(labels, values)):
        h = int(value / peak * (plot_h - 8))
        x = origin_x + 20 + i * (bar_w + 24)
        y = origin_y + plot_h - h
        canvas.fill_rect(x, y, bar_w, h, TEAL)
        canvas.text(x, origin_y + plot_h + 12, label, INK, 2)
        canvas.text(x, y - 18, f"{value:.2f}M", INK, 2)
    canvas.text(24, 390, "Parameters in millions", INK, 2)
    canvas.save(path)
    (path.parent / "parameter_count.json").write_text(
        json.dumps(
            {
                "token_embeddings": 18432000,
                "position_embeddings": 49152,
                "transformer_blocks": 7097856,
                "final_layer_norm": 768,
                "output_head": 0,
                "total_unique": 25579776,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def save_head_summary_stdlib(entropy: Sequence[Sequence[float]], distance: Sequence[Sequence[float]], path: Path, title: str) -> None:
    layers = len(entropy)
    heads = len(entropy[0])
    cell = 54
    left = 90
    top = 70
    gap = 40
    width = left + heads * cell + gap + heads * cell + 80
    canvas = Canvas(width, top + layers * cell + 80, WHITE)
    canvas.text(24, 16, title, NAVY, 2)
    panels = (("Entropy nats", entropy), ("Distance tokens", distance))
    for p, (label, grid) in enumerate(panels):
        ox = left + p * (heads * cell + gap)
        canvas.text(ox, 42, label, INK, 2)
        lo = min(v for row in grid for v in row)
        hi = max(v for row in grid for v in row) or 1.0
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                t = 0.0 if hi == lo else (value - lo) / (hi - lo)
                canvas.fill_rect(ox + j * cell, top + i * cell, cell - 3, cell - 3, magma(t))
                canvas.text(ox + j * cell + 6, top + i * cell + 18, f"{value:.2f}", WHITE if t > 0.45 else INK, 2)
            canvas.text(18, top + i * cell + 18, f"L{i}", INK, 2)
        for j in range(heads):
            canvas.text(ox + j * cell + 12, top + layers * cell + 10, f"H{j}", INK, 2)
    canvas.save(path)


def save_intrinsic_bars_stdlib(tel: dict, nep: dict, path: Path) -> None:
    canvas = Canvas(920, 420, WHITE)
    canvas.text(24, 16, "H vs L after one epoch", NAVY, 2)
    panels = (
        (40, "Validation PPL", [tel["intrinsic"]["perplexity"], nep["intrinsic"]["perplexity"]]),
        (500, "Bits per byte", [tel["intrinsic"]["bits_per_byte"], nep["intrinsic"]["bits_per_byte"]]),
    )
    colors = [NAVY, TEAL]
    names = ["Telugu H", "Nepali L"]
    for ox, title, values in panels:
        canvas.text(ox, 50, title, INK, 2)
        peak = max(values) * 1.2
        for i, value in enumerate(values):
            h = int(value / peak * 260)
            x = ox + 40 + i * 160
            y = 360 - h
            canvas.fill_rect(x, y, 110, h, colors[i])
            canvas.text(x, 372, names[i], INK, 2)
            canvas.text(x, y - 20, f"{value:.2f}" if value > 2 else f"{value:.3f}", INK, 2)
    canvas.save(path)


def save_nep_loss_stdlib(log_path: Path, path: Path) -> None:
    steps: list[int] = []
    losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "loss" in row and "val_loss" not in row:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
        if "val_loss" in row:
            val_steps.append(int(row["step"]))
            val_losses.append(float(row["val_loss"]))
    canvas = Canvas(980, 440, WHITE)
    canvas.text(24, 16, "Nepali L one epoch training", NAVY, 2)
    ox, oy, pw, ph = 70, 60, 860, 320
    canvas.fill_rect(ox, oy, pw, ph, LIGHT)
    xs = steps or [0, 1]
    ys = losses or [0.0, 1.0]
    xmin, xmax = xs[0], xs[-1]
    ymin, ymax = min(ys + val_losses), max(ys + val_losses)
    pad_y = (ymax - ymin) * 0.08 or 0.2
    ymin -= pad_y
    ymax += pad_y

    def map_pt(step: int, loss: float) -> tuple[int, int]:
        x = ox + int((step - xmin) / max(1, xmax - xmin) * (pw - 2))
        y = oy + ph - int((loss - ymin) / (ymax - ymin) * (ph - 2))
        return x, y

    stride = max(1, len(steps) // 400)
    pts = [map_pt(s, v) for s, v in list(zip(steps, losses))[::stride]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        canvas.line(x0, y0, x1, y1, NAVY, 2)
    if val_steps:
        vpts = [map_pt(s, v) for s, v in zip(val_steps, val_losses)]
        for (x0, y0), (x1, y1) in zip(vpts, vpts[1:]):
            canvas.line(x0, y0, x1, y1, TEAL, 3)
        for x, y in vpts:
            canvas.fill_rect(x - 3, y - 3, 7, 7, TEAL)
    canvas.text(ox, oy + ph + 12, "Step", INK, 2)
    canvas.text(24, oy, "CE", INK, 2)
    canvas.text(ox + 500, oy + ph + 12, "Navy=train  Teal=val", INK, 2)
    canvas.save(path)


def save_causal_mask(seq_len: int, path: Path, plt, np) -> None:
    if plt is None:
        save_causal_mask_stdlib(seq_len, path)
        return
    mask = np.tril(np.ones((seq_len, seq_len)))
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    image = axis.imshow(mask, cmap="Greys", vmin=0, vmax=1, aspect="equal")
    axis.set_title("Causal mask (1 = allowed, 0 = blocked future)")
    axis.set_xlabel("Key position")
    axis.set_ylabel("Query position")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_parameter_bar(path: Path, plt) -> None:
    if plt is None:
        save_parameter_bar_stdlib(path)
        return
    labels = [
        "Token embeddings",
        "Position embeddings",
        "Transformer blocks",
        "Final LayerNorm",
        "Tied LM head",
    ]
    values = [18_432_000, 49_152, 7_097_856, 768, 0]
    total = 25_579_776
    figure, axis = plt.subplots(figsize=(8.4, 4.4))
    axis.bar(labels, [value / 1e6 for value in values], color="#087E8B")
    axis.set_ylabel("Parameters (millions)")
    axis.set_title(f"Telugu/Nepali unique parameters: {total:,}")
    axis.set_ylim(0, max(values) / 1e6 * 1.2)
    plt.xticks(rotation=20, ha="right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    (path.parent / "parameter_count.json").write_text(
        json.dumps(
            {
                "token_embeddings": 18432000,
                "position_embeddings": 49152,
                "transformer_blocks": 7097856,
                "final_layer_norm": 768,
                "output_head": 0,
                "total_unique": total,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def save_head_summary(entropy, distance, path: Path, title: str, plt, np) -> None:
    if plt is None:
        save_head_summary_stdlib(entropy, distance, path, title)
        return
    entropy_a = np.asarray(entropy, dtype=np.float64)
    distance_a = np.asarray(distance, dtype=np.float64)
    layers, heads = entropy_a.shape
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, values, label in (
        (axes[0], entropy_a, "Mean attention entropy (nats)"),
        (axes[1], distance_a, "Mean attention distance (tokens)"),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_intrinsic_bars(tel: dict, nep: dict, path: Path, plt) -> None:
    if plt is None:
        save_intrinsic_bars_stdlib(tel, nep, path)
        return
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    languages = ["Telugu (H)", "Nepali (L)"]
    ppl = [tel["intrinsic"]["perplexity"], nep["intrinsic"]["perplexity"]]
    bpb = [tel["intrinsic"]["bits_per_byte"], nep["intrinsic"]["bits_per_byte"]]
    axes[0].bar(languages, ppl, color=["#17324D", "#087E8B"])
    axes[0].set_ylabel("Validation PPL")
    axes[0].set_title("Packed-window perplexity")
    axes[1].bar(languages, bpb, color=["#17324D", "#087E8B"])
    axes[1].set_ylabel("Bits per byte")
    axes[1].set_title("BPB (tokenizer-fair)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_nep_loss(log_path: Path, path: Path, plt) -> None:
    if plt is None:
        save_nep_loss_stdlib(log_path, path)
        return
    steps, losses, val_steps, val_losses = [], [], [], []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "loss" in row and "val_loss" not in row:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
        if "val_loss" in row:
            val_steps.append(int(row["step"]))
            val_losses.append(float(row["val_loss"]))
    figure, axis = plt.subplots(figsize=(8.8, 4.2))
    axis.plot(steps, losses, color="#17324D", linewidth=0.8, alpha=0.55, label="train CE")
    if val_steps:
        axis.plot(val_steps, val_losses, color="#087E8B", marker="o", linewidth=1.6, label="val CE")
    axis.set_xlabel("Step")
    axis.set_ylabel("Cross-entropy")
    axis.set_title("Nepali (L) one-epoch training")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> int:
    tel = json.loads((ROOT / "phase2_tel_eval.json").read_text(encoding="utf-8"))
    nep = json.loads((ROOT / "phase2_nep_eval.json").read_text(encoding="utf-8"))
    plt, np = _try_matplotlib()
    FIGURES.mkdir(parents=True, exist_ok=True)
    save_causal_mask(32, FIGURES / "causal_mask.png", plt, np)
    save_parameter_bar(FIGURES / "parameter_breakdown.png", plt)
    save_intrinsic_bars(tel, nep, FIGURES / "intrinsic_h_vs_l.png", plt)
    for language, payload in (("tel", tel), ("nep", nep)):
        out = FIGURES / language
        save_head_summary(
            payload["attention"]["entropy_per_layer_head"],
            payload["attention"]["distance_per_layer_head"],
            out / "attention_entropy_distance.png",
            f"{language} attention entropy and distance",
            plt,
            np,
        )
        save_causal_mask(32, out / "causal_mask.png", plt, np)
    nep_log = ROOT / "logs" / "nep_train.jsonl"
    if not nep_log.is_file():
        nep_log = (
            ROOT.parent
            / "artifacts"
            / "extracted"
            / "models"
            / "draft-v1"
            / "nep"
            / "logs"
            / "train.jsonl"
        )
    if nep_log.is_file():
        save_nep_loss(nep_log, FIGURES / "nep" / "train_loss.png", plt)
    backend = "matplotlib" if plt is not None else "stdlib-png"
    print(f"wrote figures under {FIGURES} ({backend})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
