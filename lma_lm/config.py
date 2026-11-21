"""Language-specific model and training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .common import (
    LMError,
    POSITIONAL_LEARNED,
    POSITIONAL_NONE,
    language_config_path,
    read_json,
    require_language,
)


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_fraction: float = 0.02
    batch_size: int = 128
    grad_accum: int = 1
    max_tokens: int | None = None
    eval_every: int = 2500
    ckpt_every: int = 2500
    eval_batches: int = 2
    keep_last: int = 1
    n_epochs: int = 1
    seed: int = 2026


@dataclass
class GenerationConfig:
    prefix_tokens: int = 16
    max_new_tokens: int = 32
    temperatures: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5])
    num_examples: int = 16


@dataclass
class ModelConfig:
    language: str
    vocab_size: int
    d_model: int = 384
    n_layer: int = 4
    n_head: int = 6
    d_ff: int = 1536
    seq_len: int = 128
    dropout: float = 0.1
    tie_weights: bool = True
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    positional_embeddings: str = POSITIONAL_LEARNED
    normalization: str = "pre_norm"
    activation: str = "gelu"
    model_label: str = "H"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    attention_examples: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_language(self.language)
        if self.d_model % self.n_head != 0:
            raise LMError(f"d_model={self.d_model} must be divisible by n_head={self.n_head}")
        if self.seq_len < 2:
            raise LMError("seq_len must be at least 2")
        if self.n_head < 1 or self.n_layer < 1:
            raise LMError("n_head and n_layer must be positive")
        if self.positional_embeddings not in {POSITIONAL_LEARNED, POSITIONAL_NONE}:
            raise LMError("positional_embeddings must be learned_absolute or none")
        if self.normalization != "pre_norm":
            raise LMError("this implementation only ships pre-norm Transformer blocks")
        if self.activation != "gelu":
            raise LMError("this implementation only ships GELU feed-forward networks")
        self.head_dim = self.d_model // self.n_head

    @property
    def uses_positional_embeddings(self) -> bool:
        return self.positional_embeddings == POSITIONAL_LEARNED

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["head_dim"] = self.head_dim
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        training_raw = dict(payload.get("training") or {})
        if "betas" in training_raw:
            betas = training_raw.pop("betas")
            training_raw["beta1"] = float(betas[0])
            training_raw["beta2"] = float(betas[1])
        generation_raw = dict(payload.get("generation") or {})
        training = TrainingConfig(
            **{key: value for key, value in training_raw.items() if key in TrainingConfig.__dataclass_fields__}
        )
        generation = GenerationConfig(
            **{key: value for key, value in generation_raw.items() if key in GenerationConfig.__dataclass_fields__}
        )
        ignored = {
            "schema_version",
            "architecture",
            "training",
            "generation",
            "head_dim",
            "vocab_size_source",
        }
        fields = {
            key: value
            for key, value in payload.items()
            if key in cls.__dataclass_fields__ and key not in {"training", "generation"} and key not in ignored
        }
        if payload.get("tie_word_embeddings") is not None and "tie_weights" not in fields:
            fields["tie_weights"] = bool(payload["tie_word_embeddings"])
        return cls(training=training, generation=generation, **fields)


def load_language_config(
    language: str,
    vocab_size: int | None = None,
    root: Path | None = None,
    *,
    no_positions: bool = False,
) -> ModelConfig:
    filename = "model_nopos.json" if no_positions else "model.json"
    path = language_config_path(language, root, filename=filename)
    if no_positions and not path.is_file():
        path = language_config_path(language, root)
        if not path.is_file():
            raise LMError(f"missing language config: {path}")
        payload = read_json(path)
        payload["positional_embeddings"] = POSITIONAL_NONE
    else:
        if not path.is_file():
            raise LMError(f"missing language config: {path}")
        payload = read_json(path)
    if payload.get("language") != language:
        raise LMError(f"config language {payload.get('language')!r} does not match {language}")
    if vocab_size is not None:
        payload["vocab_size"] = int(vocab_size)
    if payload.get("vocab_size") in (None, 0):
        raise LMError("vocab_size is unset; load it from the SentencePiece model before constructing ModelConfig")
    return ModelConfig.from_dict(payload)
