#!/usr/bin/env python3
"""Command-line entry points for packing, pretraining, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import DEFAULT_RUN_ID, LANGUAGES, LMError, emit, packed_meta_path, read_json, repository_root
from .data import pack_language


def _add_common(parser: argparse.ArgumentParser, require_data_root: bool = True) -> None:
    parser.add_argument("--data-root", type=Path, required=require_data_root)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--language", choices=("all", *LANGUAGES), default="all")
    parser.add_argument("--status", action="store_true")


def _languages(argument: str) -> tuple[str, ...]:
    return LANGUAGES if argument == "all" else (argument,)


def pack_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack Phase 1 JSONL shards into uint16 token bins.")
    _add_common(parser)
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    args = parser.parse_args(argv)
    try:
        from .common import SPLITS

        splits = SPLITS if args.split == "all" else (args.split,)
        for language in _languages(args.language):
            if args.status:
                path = packed_meta_path(args.data_root, args.run_id, language)
                if path.is_file():
                    print(json.dumps(read_json(path), indent=2, sort_keys=True))
                else:
                    print(json.dumps({"language": language, "status": "not-packed"}, indent=2))
                continue
            emit(f"pack: starting {language} splits={list(splits)}")
            pack_language(args.data_root, args.run_id, language, splits=splits)
        return 0
    except KeyboardInterrupt:
        print("\npack_tokens.py: interrupted; completed shards will resume.", flush=True)
        return 130
    except Exception as exc:
        print(f"pack_tokens.py: {type(exc).__name__}: {exc}", flush=True)
        return 2


def train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretrain an independent decoder-only LM.")
    _add_common(parser, require_data_root=False)
    parser.add_argument("--device")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run pack → train → eval on a tiny toy corpus to catch code bugs.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument(
        "--no-positions",
        action="store_true",
        help="Ablation: train without positional embeddings. "
        "Reuses packed tokens; writes models/<run>/<lang>/nopos/ so pretrained checkpoints stay intact.",
    )
    args = parser.parse_args(argv)
    try:
        from .checkpoint import training_status
        from .train import select_device, train_from_packed

        device = select_device(args.device)
        if args.smoke:
            from .smoke import run_smoke

            result = run_smoke(device=device)
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
            return 0
        if args.data_root is None:
            raise LMError("--data-root is required unless --smoke is set")
        for language in _languages(args.language):
            if args.status:
                print(json.dumps(
                    training_status(
                        args.data_root,
                        args.run_id,
                        language,
                        ablation="nopos" if args.no_positions else None,
                    ),
                    indent=2,
                    sort_keys=True,
                    default=str,
                ))
                continue
            emit(f"train: starting {language} on {device}")
            result = train_from_packed(
                args.data_root,
                args.run_id,
                language,
                device=device,
                resume=not args.no_resume,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                no_positions=args.no_positions,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        print("\ntrain_lm.py: interrupted; latest Drive checkpoint will resume.", flush=True)
        return 130
    except Exception as exc:
        print(f"train_lm.py: {type(exc).__name__}: {exc}", flush=True)
        return 2


def eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a pretrained decoder-only LM.")
    _add_common(parser)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, default=repository_root() / "report")
    parser.add_argument("--max-windows", type=int, default=2000)
    parser.add_argument(
        "--no-positions",
        action="store_true",
        help="Evaluate the no-position ablation under models/<run>/<lang>/nopos/.",
    )
    args = parser.parse_args(argv)
    try:
        from .checkpoint import training_status
        from .evaluate import evaluate_language
        from .train import select_device

        device = select_device(args.device)
        for language in _languages(args.language):
            if args.status:
                print(json.dumps(
                    training_status(
                        args.data_root,
                        args.run_id,
                        language,
                        ablation="nopos" if args.no_positions else None,
                    ),
                    indent=2,
                    sort_keys=True,
                    default=str,
                ))
                continue
            result = evaluate_language(
                args.data_root,
                args.run_id,
                language,
                checkpoint_path=args.checkpoint,
                device=device,
                output_dir=args.output_dir,
                max_windows=args.max_windows,
                no_positions=args.no_positions,
            )
            print(json.dumps({key: result[key] for key in result if key != "samples"}, indent=2, sort_keys=True, default=str))
        return 0
    except KeyboardInterrupt:
        print("\neval_lm.py: interrupted.", flush=True)
        return 130
    except Exception as exc:
        print(f"eval_lm.py: {type(exc).__name__}: {exc}", flush=True)
        return 2
