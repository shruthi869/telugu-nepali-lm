"""Decoder-only Transformer language model implemented from first principles.

The model deliberately avoids ``nn.Transformer*`` modules and HuggingFace
model classes. Every tensor operation in the forward pass is therefore explicit:
token/position embeddings, multi-head causal self-attention, a GELU
feed-forward network, residual connections, pre-norm LayerNorm, and a
weight-tied next-token head.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an explicit additive mask.

    For an input ``X`` of shape ``(B, T, d)`` this module:

    1. Projects to concatenated Q, K, V with one linear layer ``c_attn``.
    2. Reshapes ``(B, T, 3d)`` into three tensors of shape ``(B, h, T, d_k)``
       where ``d_k = d / h``.
    3. Computes scaled dot-product attention with a causal mask ``M``:
       ``softmax((Q K^T) / sqrt(d_k) + M) V``.
    4. Concatenates heads back to ``(B, T, d)`` and applies ``c_proj``.

    The ``1 / sqrt(d_k)`` scale keeps the dot-product variance near 1 so
    softmax does not saturate as ``d_k`` grows. The mask ``M`` is 0 on
    allowed (past and present) positions and ``-inf`` on future positions,
    so token ``t`` cannot attend to tokens ``> t``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self._sdpa = hasattr(F, "scaled_dot_product_attention")
        causal = torch.tril(torch.ones(config.seq_len, config.seq_len)).view(1, 1, config.seq_len, config.seq_len)
        self.register_buffer("causal_mask", causal, persistent=False)

    def _explicit_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (query @ key.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        return weights @ value, weights

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, width = x.shape
        if width != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {width}")
        qkv = self.c_attn(x)
        query, key, value = qkv.split(self.d_model, dim=2)
        # (B, T, d) -> (B, T, h, d_k) -> (B, h, T, d_k)
        query = query.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        weights = None
        if return_weights or not self._sdpa:
            attended, weights = self._explicit_attention(query, key, value, seq_len)
        else:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d)
        concatenated = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.resid_dropout(self.c_proj(concatenated))
        if return_weights:
            return output, weights
        return output


class FeedForward(nn.Module):
    """Position-wise two-layer MLP with GELU and an expanded inner width."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.d_ff)
        self.proj = nn.Linear(config.d_ff, config.d_model)
        try:
            self.activation = nn.GELU(approximate="tanh")
        except TypeError:
            self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.activation(self.fc(x))))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN -> attention -> residual, LN -> FFN -> residual.

    Pre-norm is generally more stable for stacked blocks than post-norm because
    residual branches carry a clean identity path and gradients do not have to
    pass through a LayerNorm on every skip connection.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_weights:
            attended, weights = self.attn(self.ln_1(x), return_weights=True)
            x = x + attended
            x = x + self.mlp(self.ln_2(x))
            return x, weights
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderLM(nn.Module):
    """GPT-style decoder-only language model.

    Learned absolute positional embeddings are the default: they inject order
    because self-attention is permutation-invariant, and they cap usable
    context at ``seq_len``. The optional no-position ablation sets
    ``positional_embeddings="none"`` and skips that table entirely. Causal
    masking still remains, so the model is not fully permutation-invariant.
    The output head is tied to the token embedding matrix, which removes a
    second ``V x d`` table (~18M parameters at V=48k, d=384).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.seq_len, config.d_model) if config.uses_positional_embeddings else None
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight") or name.endswith("proj.weight"):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def tie_weights(self) -> None:
        if self.config.tie_weights:
            self.lm_head.weight = self.wte.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        _batch, seq_len = idx.shape
        if seq_len > self.config.seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds model context {self.config.seq_len}")
        x = self.wte(idx)
        if self.wpe is not None:
            positions = torch.arange(seq_len, device=idx.device, dtype=torch.long)
            x = x + self.wpe(positions)
        x = self.drop(x)
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            if return_attentions:
                x, weights = block(x, return_weights=True)
                attentions.append(weights)
            else:
                x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if return_attentions:
            return logits, loss, attentions
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive decoding. ``temperature <= 0`` selects greedy argmax."""
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.seq_len :]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and int(next_id.item()) == eos_id:
                break
        return idx


def count_parameters(model: nn.Module, unique: bool = True) -> int:
    """Count trainable parameters. ``unique=True`` does not double-count tied weights."""
    if unique:
        seen: set[int] = set()
        total = 0
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            total += parameter.numel()
        return total
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_breakdown(model: DecoderLM) -> dict[str, Any]:
    """Named parameter groups for the report. Tied lm_head is not counted twice."""
    config = model.config
    token_embeddings = model.wte.weight.numel()
    position_embeddings = 0 if model.wpe is None else model.wpe.weight.numel()
    per_block = 0
    for parameter in model.blocks[0].parameters():
        per_block += parameter.numel()
    blocks = per_block * config.n_layer
    final_norm = sum(parameter.numel() for parameter in model.ln_f.parameters())
    output_head = 0 if config.tie_weights else model.lm_head.weight.numel()
    unique = count_parameters(model, unique=True)
    return {
        "token_embeddings": token_embeddings,
        "position_embeddings": position_embeddings,
        "transformer_blocks": blocks,
        "parameters_per_block": per_block,
        "final_layer_norm": final_norm,
        "output_head": output_head,
        "output_head_tied": bool(config.tie_weights),
        "total_unique": unique,
        "total_with_tied_double_count": count_parameters(model, unique=False),
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "d_ff": config.d_ff,
        "seq_len": config.seq_len,
        "positional_embeddings": config.positional_embeddings,
    }
