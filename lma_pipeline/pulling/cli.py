"""Command-line interface for resumable corpus collection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .collector import Collector
from .common import CollectorError, default_data_root, ensure_space

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Drive-backed directory for raw assets and pull_state.sqlite")
    parser.add_argument("--source", help="Collect or show only one source id")
    parser.add_argument("--status", action="store_true", help="Show ledger counts without collecting")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs" / "sources.json")
    parser.add_argument("--timeout", type=float, default=45.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--pilot-max-documents",
        type=int,
        help="Stop a selected MediaWiki source after this many fetched pages",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        data_root = (args.data_root.expanduser() if args.data_root else default_data_root()).resolve()
        if not args.status:
            ensure_space(data_root)
        elif not data_root.exists():
            print("No ledger exists yet.")
            return 0
        collector = Collector(
            data_root,
            args.config.resolve(),
            args.timeout,
            args.pilot_max_documents,
        )
        try:
            if args.status:
                collector.print_status(args.source)
            else:
                processed = collector.run(args.source)
                print(f"Processed {processed} job(s).")
                collector.print_status(args.source)
        finally:
            collector.close()
        return 0
    except CollectorError as exc:
        print(f"pull.py: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\npull.py: interrupted safely; rerun the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
