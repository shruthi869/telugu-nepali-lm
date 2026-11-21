"""Command-line interface for independent Telugu/Nepali tokenizer training."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .preflight import PreprocessingNotComplete
from .trainer import TokenizerTrainer


LANGUAGES = ("tel", "nep")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-id", default="draft-v1")
    parser.add_argument("--language", choices=("all", *LANGUAGES), default="all")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--registry", type=Path, default=repository / "configs" / "sources.json")
    parser.add_argument("--local-work-root", type=Path)
    parser.add_argument(
        "--allow-low-manual-fraction",
        action="store_true",
        help=(
            "Explicitly train on a completed 500M+ corpus below the 20%% manual-token target; "
            "the override and actual fraction are recorded in tokenizer artifacts"
        ),
    )
    return parser.parse_args(argv)


def emit(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repository = Path(__file__).resolve().parent.parent
    languages = LANGUAGES if args.language == "all" else (args.language,)
    try:
        for language in languages:
            trainer = TokenizerTrainer(
                data_root=args.data_root,
                run_id=args.run_id,
                language=language,
                config_path=repository / language / "tokenizer" / "config.json",
                registry_path=args.registry,
                local_work_root=args.local_work_root,
                allow_low_manual_fraction=args.allow_low_manual_fraction,
                progress=emit,
            )
            if args.status:
                print(json.dumps(trainer.status(), indent=2, sort_keys=True))
            else:
                emit(f"tokenizer-preflight: {language}: validating completed preprocessing")
                if args.allow_low_manual_fraction:
                    emit(
                        f"tokenizer-preflight: {language}: manual-fraction override requested; "
                        "500M minimum, registry, and split-output checks remain enforced"
                    )
                trainer.train()
        return 0
    except PreprocessingNotComplete as exc:
        print(f"train_tokenizer.py: preprocessing is not complete: {exc}", file=sys.stderr, flush=True)
        return 3
    except KeyboardInterrupt:
        print("\ntrain_tokenizer.py: interrupted safely; completed candidates and corpus samples will resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"train_tokenizer.py: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
