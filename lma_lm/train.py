"""Independent AdamW pretraining with cosine schedule, AMP, and Drive checkpoints."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import (
    apply_checkpoint,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    write_run_config,
)
from .common import (
    LMError,
    checkpoint_dir,
    emit,
    log_path,
    positional_ablation,
)
from .config import ModelConfig
from .data import PackedTokens, config_with_tokenizer, load_packed_meta, materialize_split
from .model import DecoderLM, count_parameters, parameter_breakdown


def select_device(explicit: str | None = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("cpu")  # AMP and some ops are more predictable on CPU for tests
    return torch.device("cpu")


def configure_fast_kernels(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except (TypeError, AttributeError):
        pass


def batch_to_device(inputs: np.ndarray, targets: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.int64))
    tgt = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.int64))
    return idx.to(device), tgt.to(device)


def configure_optimizer(model: DecoderLM, config: ModelConfig) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim >= 2 and "norm" not in name.lower() and "ln_" not in name:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    kwargs = {
        "lr": config.training.learning_rate,
        "betas": (config.training.beta1, config.training.beta2),
    }
    groups = [
        {"params": decay, "weight_decay": config.training.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    try:
        if torch.cuda.is_available():
            return torch.optim.AdamW(groups, fused=True, **kwargs)
    except (TypeError, ValueError, RuntimeError):
        pass
    return torch.optim.AdamW(groups, **kwargs)


def build_scheduler(
    optimizer: torch.optim.AdamW,
    total_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(int(warmup_steps), 1)
    total_steps = max(int(total_steps), warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return (min_lr / max_lr) + cosine * (1.0 - (min_lr / max_lr))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _make_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def _autocast(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(device.type, enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


@torch.no_grad()
def evaluate_batches(
    model: DecoderLM,
    tokens: PackedTokens,
    batch_size: int,
    n_batches: int,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    max_start = tokens.n_windows
    if max_start <= 0:
        raise LMError("validation packed split is shorter than seq_len")
    for _ in range(n_batches):
        starts = rng.integers(0, max_start, size=min(batch_size, max_start))
        inputs, targets = tokens.batch(starts)
        idx, tgt = batch_to_device(inputs, targets, device)
        with _autocast(device, enabled=device.type == "cuda"):
            _logits, loss = model(idx, tgt)
        losses.append(float(loss.item()))
    model.train()
    mean = float(np.mean(losses))
    return {"val_loss": mean, "val_ppl": float(math.exp(min(mean, 20.0)))}


def train_from_packed(
    data_root: Path,
    run_id: str,
    language: str,
    device: torch.device | None = None,
    resume: bool = True,
    max_steps: int | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    config_root: Path | None = None,
    no_positions: bool = False,
    progress=emit,
) -> dict[str, Any]:
    device = device or select_device()
    configure_fast_kernels(device)
    config, _processor, tokenizer_manifest = config_with_tokenizer(
        language, data_root, run_id, root=config_root, no_positions=no_positions
    )
    ablation = positional_ablation(config.positional_embeddings)
    meta = load_packed_meta(data_root, run_id, language)
    train_tokens = materialize_split(data_root, run_id, language, "train", config.seq_len, progress=progress)
    val_tokens = materialize_split(data_root, run_id, language, "validation", config.seq_len, progress=progress)
    train_cfg = config.training
    micro_batch = int(batch_size or train_cfg.batch_size)
    accum = int(grad_accum or train_cfg.grad_accum)
    tokens_per_step = micro_batch * config.seq_len * accum
    available = int(meta["splits"]["train"]["token_count"])
    n_epochs = max(int(getattr(train_cfg, "n_epochs", 1)), 1)
    one_epoch_tokens = int(train_cfg.max_tokens or available)
    target_tokens = one_epoch_tokens * n_epochs
    total_steps = max(target_tokens // tokens_per_step, 1)
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))
    warmup_steps = max(int(train_cfg.warmup_fraction * total_steps), 1)

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    model = DecoderLM(config).to(device)
    optimizer = configure_optimizer(model, config)
    scheduler = build_scheduler(
        optimizer, total_steps, warmup_steps, train_cfg.learning_rate, train_cfg.min_learning_rate
    )
    use_amp = device.type == "cuda"
    scaler = _make_scaler(device, enabled=use_amp)
    rng = np.random.default_rng(train_cfg.seed)
    step = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    ckpt_dir = checkpoint_dir(data_root, run_id, language, ablation=ablation)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prune_checkpoints(ckpt_dir)

    def persist(kind: str, complete: bool = False) -> Path:
        epoch = min(tokens_seen // max(one_epoch_tokens, 1) + 1, n_epochs)
        extra = {
            "epoch": epoch,
            "n_epochs": n_epochs,
            "total_steps": total_steps,
            "complete": complete or step >= total_steps,
            "interrupted": kind == "interrupt",
        }
        durable = kind in {"interrupt", "final"}
        finished = kind == "final" or extra["complete"]
        path = ckpt_dir / "latest.pt"
        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler,
            scaler,
            step,
            tokens_seen,
            best_val_loss,
            extra=extra,
            durable=durable,
            weights_only=finished and kind != "interrupt",
        )
        if kind == "best":
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                tokens_seen,
                best_val_loss,
                extra=extra,
                durable=durable,
                weights_only=True,
            )
        elif finished and not (ckpt_dir / "best.pt").is_file() and best_val_loss < float("inf"):
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                step,
                tokens_seen,
                best_val_loss,
                extra=extra,
                durable=durable,
                weights_only=True,
            )
        prune_checkpoints(ckpt_dir)
        return path

    write_run_config(
        data_root,
        run_id,
        language,
        config,
        extra={
            "tokenizer_sha256": meta.get("tokenizer_sha256"),
            "packed_train_tokens": available,
            "target_tokens": target_tokens,
            "n_epochs": n_epochs,
            "total_steps": total_steps,
            "tokens_per_step": tokens_per_step,
            "device": str(device),
            "parameter_count": count_parameters(model),
            "parameter_breakdown": parameter_breakdown(model),
            "tokenizer_vocab_size": tokenizer_manifest.get("selected_vocabulary_size", config.vocab_size),
            "ablation": ablation,
        },
        ablation=ablation,
    )

    if resume:
        latest = latest_checkpoint(data_root, run_id, language, ablation=ablation)
        if latest is not None:
            payload = load_checkpoint(latest, map_location=device)
            apply_checkpoint(payload, model, optimizer, scheduler, scaler)
            step = int(payload.get("step", 0))
            tokens_seen = int(payload.get("tokens_seen", 0))
            best_val_loss = float(payload.get("best_val_loss", best_val_loss))
            epoch = int(payload.get("epoch") or min(tokens_seen // max(one_epoch_tokens, 1) + 1, n_epochs))
            progress(
                f"train: {language}: resumed {latest.name} at epoch {epoch}/{n_epochs} "
                f"step {step:,}/{total_steps:,} ({tokens_seen:,} tokens)"
            )
            if payload.get("complete") or step >= total_steps:
                progress(f"train: {language}: already complete; nothing to do")
                return {
                    "language": language,
                    "steps": step,
                    "tokens_seen": tokens_seen,
                    "best_val_loss": best_val_loss,
                    "parameter_count": count_parameters(model),
                    "latest": str(latest),
                    "best": str(ckpt_dir / "best.pt") if (ckpt_dir / "best.pt").is_file() else None,
                    "complete": True,
                }

    progress(
        f"train: {language}: {count_parameters(model):,} unique params; "
        f"{n_epochs} epoch(s); context {config.seq_len}; batch {micro_batch} x accum {accum}; "
        f"{total_steps:,} steps; device {device}"
        + (f"; ablation={ablation}" if ablation else "")
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    step_times: list[float] = []
    log_file = log_path(data_root, run_id, language, ablation=ablation)

    try:
        while step < total_steps:
            used_batch = micro_batch
            loss_tensor = None
            for _micro in range(accum):
                while True:
                    try:
                        starts = rng.integers(0, train_tokens.n_windows, size=used_batch)
                        inputs, targets = train_tokens.batch(starts)
                        idx, tgt = batch_to_device(inputs, targets, device)
                        with _autocast(device, enabled=use_amp):
                            _logits, loss = model(idx, tgt)
                            loss = loss / accum
                        scaler.scale(loss).backward()
                        loss_tensor = loss.detach() if loss_tensor is None else loss_tensor + loss.detach()
                        break
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower() or used_batch <= 1:
                            raise
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        used_batch = max(used_batch // 2, 1)
                        micro_batch = used_batch
                        progress(f"train: {language}: OOM; reduced microbatch to {used_batch}")
                        optimizer.zero_grad(set_to_none=True)
                tokens_seen += used_batch * config.seq_len
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            now = time.time()
            step_times.append(now - t0)
            t0 = now
            epoch = min(tokens_seen // max(one_epoch_tokens, 1) + 1, n_epochs)
            if step == 1 or step % 20 == 0 or step == total_steps:
                loss_accum = float(loss_tensor.item()) if loss_tensor is not None else 0.0
                toks = used_batch * config.seq_len * accum
                recent = max(sum(step_times[-10:]) / max(len(step_times[-10:]), 1), 1e-6)
                tok_s = toks / recent
                remaining = (total_steps - step) * recent
                progress(
                    f"train: {language}: epoch {epoch}/{n_epochs} step {step}/{total_steps} "
                    f"loss={loss_accum:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"{tok_s:,.0f} tok/s ETA {remaining / 3600:.1f}h"
                )
                append_jsonl(
                    log_file,
                    {
                        "time": time.time(),
                        "language": language,
                        "epoch": epoch,
                        "n_epochs": n_epochs,
                        "step": step,
                        "total_steps": total_steps,
                        "tokens_seen": tokens_seen,
                        "loss": round(loss_accum, 6),
                        "lr": scheduler.get_last_lr()[0],
                        "micro_batch": used_batch,
                        "tokens_per_second": tok_s,
                        "eta_hours": remaining / 3600,
                    },
                )

            do_eval = step % train_cfg.eval_every == 0 or step == total_steps
            if do_eval:
                metrics = evaluate_batches(
                    model, val_tokens, used_batch, train_cfg.eval_batches, device, rng
                )
                progress(
                    f"train: {language}: val step {step} loss={metrics['val_loss']:.4f} "
                    f"ppl={metrics['val_ppl']:.2f}"
                )
                append_jsonl(log_file, {"step": step, "tokens_seen": tokens_seen, **metrics})
                improved = metrics["val_loss"] < best_val_loss
                if improved:
                    best_val_loss = metrics["val_loss"]
                    persist("best")
                    progress(f"train: {language}: new best val loss {best_val_loss:.4f} -> best.pt")
                elif step < total_steps:
                    persist("eval")
                    progress(f"train: {language}: resume checkpoint {step}/{total_steps} -> latest.pt")
    except KeyboardInterrupt:
        persist("interrupt")
        progress(
            f"train: {language}: interrupt saved at epoch "
            f"{min(tokens_seen // max(one_epoch_tokens, 1) + 1, n_epochs)}/{n_epochs} "
            f"step {step}/{total_steps}. Rerun the same command to continue."
        )
        raise

    persist("final", complete=True)
    return {
        "language": language,
        "ablation": ablation,
        "positional_embeddings": config.positional_embeddings,
        "steps": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "parameter_count": count_parameters(model),
        "latest": str(ckpt_dir / "latest.pt"),
        "best": str(ckpt_dir / "best.pt") if (ckpt_dir / "best.pt").is_file() else None,
        "complete": True,
    }


def smoke_train(
    steps: int = 2,
    device: torch.device | None = None,
    progress=emit,
) -> dict[str, Any]:
    """Tiny pack → train → eval. Only checks that the pipeline runs."""
    from .smoke import run_smoke

    return run_smoke(device=device, progress=progress)
