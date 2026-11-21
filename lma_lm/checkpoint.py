"""Resume-capable checkpoints: weights, optimizer, scaler, scheduler, step, config."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import atomic_write_json, checkpoint_dir, model_dir, read_json
from .config import ModelConfig
from .model import DecoderLM, count_parameters


CHECKPOINT_KEYS = (
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "step",
    "tokens_seen",
    "best_val_loss",
    "config",
    "rng",
)


def save_checkpoint(
    path: Path,
    model: DecoderLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | torch.cuda.amp.GradScaler | None,
    step: int,
    tokens_seen: int,
    best_val_loss: float,
    extra: dict[str, Any] | None = None,
    durable: bool = False,
    weights_only: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": None if weights_only else optimizer.state_dict(),
        "scheduler": None if weights_only or scheduler is None else scheduler.state_dict(),
        "scaler": None if weights_only or scaler is None else scaler.state_dict(),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "best_val_loss": float(best_val_loss),
        "config": model.config.to_dict(),
        "parameter_count": count_parameters(model),
        "weights_only": weights_only,
        "rng": None
        if weights_only
        else {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().detach().cpu().clone(),
            "cuda": (
                [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    if extra:
        payload.update(extra)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        if durable:
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _cpu_byte_tensor(value: Any) -> torch.Tensor:
    """torch.set_rng_state requires a CPU uint8 tensor, even if the ckpt was mapped to CUDA."""
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(value, dtype=np.uint8))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    else:
        tensor = torch.as_tensor(value, dtype=torch.uint8, device="cpu")
    return tensor.cpu().contiguous().byte()


def restore_rng(payload: dict[str, Any]) -> None:
    rng = payload.get("rng") or {}
    if not rng:
        return
    try:
        python_state = rng.get("python")
        if python_state is not None:
            random.setstate(python_state)
        numpy_state = rng.get("numpy")
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        torch_state = rng.get("torch")
        if torch_state is not None:
            torch.set_rng_state(_cpu_byte_tensor(torch_state))
        cuda_state = rng.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            tensors = [_cpu_byte_tensor(item) for item in cuda_state]
            n_devices = torch.cuda.device_count()
            if len(tensors) == n_devices:
                torch.cuda.set_rng_state_all(tensors)
            elif tensors:
                torch.cuda.set_rng_state(tensors[0])
    except Exception as exc:
        print(f"checkpoint: skipped RNG restore ({type(exc).__name__}: {exc})", flush=True)


def apply_checkpoint(
    payload: dict[str, Any],
    model: DecoderLM,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | torch.cuda.amp.GradScaler | None = None,
    restore_random: bool = True,
) -> dict[str, Any]:
    model.load_state_dict(payload["model"])
    model.tie_weights()
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_random:
        restore_rng(payload)
    return payload


def latest_checkpoint(
    data_root: Path, run_id: str, language: str, *, ablation: str | None = None
) -> Path | None:
    directory = checkpoint_dir(data_root, run_id, language, ablation=ablation)
    latest = directory / "latest.pt"
    if latest.is_file():
        return latest
    numbered = sorted(directory.glob("step-*.pt"))
    return numbered[-1] if numbered else None


def best_checkpoint(
    data_root: Path, run_id: str, language: str, *, ablation: str | None = None
) -> Path | None:
    path = checkpoint_dir(data_root, run_id, language, ablation=ablation) / "best.pt"
    return path if path.is_file() else None


def prune_checkpoints(directory: Path, keep_last: int = 0) -> None:
    """Drop numbered snapshots. Drive only keeps overwritten latest.pt / best.pt."""
    del keep_last
    for stale in directory.glob("step-*.pt"):
        stale.unlink(missing_ok=True)
    for stale in directory.glob("*.pt.tmp"):
        stale.unlink(missing_ok=True)


def write_run_config(
    data_root: Path,
    run_id: str,
    language: str,
    config: ModelConfig,
    extra: dict[str, Any] | None = None,
    *,
    ablation: str | None = None,
) -> Path:
    path = model_dir(data_root, run_id, language, ablation=ablation) / "config.json"
    payload = config.to_dict()
    if extra:
        payload["run"] = extra
    atomic_write_json(path, payload)
    return path


def training_status(
    data_root: Path, run_id: str, language: str, *, ablation: str | None = None
) -> dict[str, Any]:
    directory = checkpoint_dir(data_root, run_id, language, ablation=ablation)
    latest = latest_checkpoint(data_root, run_id, language, ablation=ablation)
    best = best_checkpoint(data_root, run_id, language, ablation=ablation)
    payload: dict[str, Any] = {
        "language": language,
        "run_id": run_id,
        "ablation": ablation,
        "checkpoint_dir": str(directory),
        "latest": str(latest) if latest else None,
        "best": str(best) if best else None,
        "status": "not-started",
    }
    if latest:
        try:
            checkpoint = load_checkpoint(latest, map_location="cpu")
            payload.update(
                {
                    "status": "complete" if checkpoint.get("complete") else "in-progress",
                    "step": checkpoint.get("step"),
                    "total_steps": checkpoint.get("total_steps"),
                    "epoch": checkpoint.get("epoch"),
                    "n_epochs": checkpoint.get("n_epochs"),
                    "tokens_seen": checkpoint.get("tokens_seen"),
                    "best_val_loss": checkpoint.get("best_val_loss"),
                    "parameter_count": checkpoint.get("parameter_count"),
                }
            )
        except Exception as exc:
            payload["status"] = "unreadable"
            payload["error"] = str(exc)
    config_path = model_dir(data_root, run_id, language, ablation=ablation) / "config.json"
    if config_path.is_file():
        payload["config"] = read_json(config_path)
    return payload
