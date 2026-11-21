#!/usr/bin/env python3
"""Copy eval JSON into the versioned Phase 2 metrics export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-id", default="draft-v1")
    parser.add_argument("--output", type=Path, default=Path("phase2_metrics.json"))
    return parser.parse_args()


def read_eval(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    previous: dict[str, Any] = {}
    if args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "run_id": args.run_id,
        "architecture": previous.get("architecture"),
        "languages": {},
        "notes": previous.get("notes"),
        "training": previous.get("training"),
    }
    for language in ("tel", "nep"):
        candidates = [
            Path("report") / f"phase2_{language}_eval.json",
            Path(__file__).resolve().parent / f"phase2_{language}_eval.json",
            args.data_root / "report" / f"phase2_{language}_eval.json",
            args.data_root / "models" / args.run_id / language / "eval.json",
        ]
        found = next((path for path in candidates if path.is_file()), None)
        payload["languages"][language] = None if found is None else json.loads(found.read_text(encoding="utf-8"))
        if found is None:
            print(f"missing {language} eval JSON; ran eval_lm.py yet?", flush=True)
        else:
            print(f"{language}: {found}", flush=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
