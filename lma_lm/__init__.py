"""From-scratch decoder-only Transformer pretraining for Telugu and Nepali."""

from .config import ModelConfig, TrainingConfig, load_language_config

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "load_language_config",
]
