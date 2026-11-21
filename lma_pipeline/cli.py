"""Command-line interface for Drive-backed preprocessing."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from .common import (
    DEFAULT_MAX_SHINGLES,
    DEFAULT_SEED,
    MODEL_NAME,
    MODEL_URL,
    PreprocessError,
    default_data_root,
    sha256_file,
)
from .processor import Preprocessor

def download_model(path: Path) -> None:
    if path.exists():
        print(f"Using existing {MODEL_NAME}: {path} ({sha256_file(path)})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise PreprocessError(f"could not download {MODEL_NAME}: {exc}") from exc
    print(f"Downloaded {MODEL_NAME}: {path} ({sha256_file(path)})")




def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Drive-backed directory containing raw/")
    parser.add_argument("--source", help="Process one approved source id")
    parser.add_argument("--status", action="store_true", help="Show preprocessing ledger status")
    parser.add_argument("--run-id", default="draft-v1", help="Stable output/ledger namespace to resume (default: draft-v1)")
    parser.add_argument("--stage", action="append", choices=("extract", "calibrate", "clean", "assemble", "all"), help="Run only a stage; may be repeated")
    parser.add_argument("--finalize", action="store_true", help="Require 20%% manual training tokens per language before marking final")
    parser.add_argument("--allow-low-manual-fraction", action="store_true", help="With --finalize, explicitly waive (but still report) the 20%% manual-token requirement")
    parser.add_argument("--download-model", action="store_true", help="Download the frozen lid.176.ftz model once into data root")
    parser.add_argument("--model-path", type=Path, help="Override the frozen fastText model path")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--shard-records", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1, help="CPU worker processes for independent extraction/cleaning sources (use 2 on free Colab)")
    parser.add_argument("--sample-rate", type=float, help="Deterministic candidate selection rate; omitted uses the calibration recommendation")
    parser.add_argument("--target-downloaded-tokens", type=int, default=400_000_000, help="Per-language downloaded token budget used by calibration")
    parser.add_argument("--minimum-train-tokens", type=int, help="Minimum final training tokens required per language")
    parser.add_argument("--calibration-records-per-shard", type=int, default=5_000, help="Bounded candidate records read from each shard during --stage calibrate")
    parser.add_argument("--max-shingles", type=int, default=DEFAULT_MAX_SHINGLES, help="Maximum evenly spaced shingles used only for MinHash candidate generation")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "configs" / "sources.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.allow_low_manual_fraction and not args.finalize:
            raise PreprocessError("--allow-low-manual-fraction requires --finalize")
        root = (args.data_root.expanduser() if args.data_root else default_data_root()).resolve()
        root.mkdir(parents=True, exist_ok=True)
        model_path = (args.model_path.expanduser() if args.model_path else root / "lid.176.ftz").resolve()
        if args.download_model:
            download_model(model_path)
            return 0
        processor = Preprocessor(root, args.config.resolve(), args.run_id, args.seed, args.shard_records, model_path, workers=args.workers, sample_rate=args.sample_rate, target_downloaded_tokens=args.target_downloaded_tokens, calibration_records_per_shard=args.calibration_records_per_shard, max_shingles=args.max_shingles, minimum_train_tokens=args.minimum_train_tokens)
        try:
            if args.status:
                processor.print_status()
            else:
                requested = set(args.stage or ["all"])
                stages = {"extract", "calibrate", "clean", "assemble"} if "all" in requested else requested
                manifests = processor.run(args.source, stages, args.finalize, args.allow_low_manual_fraction)
                for language, manifest in manifests.items():
                    print(f"{language}: {manifest['training_status']} ({manifest['manual_train_token_fraction']:.2%} manual training tokens)")
        finally:
            processor.close()
        return 0
    except PreprocessError as exc:
        print(f"preprocess.py: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\npreprocess.py: interrupted safely; rerun with the same --run-id.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
