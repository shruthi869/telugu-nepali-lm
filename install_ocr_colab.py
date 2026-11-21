#!/usr/bin/env python3
"""Install and verify Telugu/Nepali Tesseract language packs in Colab."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


REQUIRED_LANGUAGES = {"eng", "nep", "tel"}


def run(command: list[str]) -> None:
    print("\n▶", " ".join(command), "\n", flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    apt_get = shutil.which("apt-get")
    if not apt_get:
        print("apt-get is unavailable. Install Tesseract and the nep/tel traineddata files manually.", file=sys.stderr)
        return 2
    prefix: list[str] = []
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        sudo = shutil.which("sudo")
        if not sudo:
            print("root or sudo is required to install Tesseract language packs.", file=sys.stderr)
            return 2
        prefix = [sudo]
    run([*prefix, apt_get, "update", "-qq"])
    run([
        *prefix,
        apt_get,
        "install",
        "-y",
        "tesseract-ocr",
        "tesseract-ocr-eng",
        "tesseract-ocr-nep",
        "tesseract-ocr-tel",
    ])
    completed = subprocess.run(
        ["tesseract", "--list-langs"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    available = {line.strip() for line in completed.stdout.splitlines()}
    missing = sorted(REQUIRED_LANGUAGES - available)
    print(completed.stdout, flush=True)
    if missing:
        print(f"Missing Tesseract language data after installation: {missing}", file=sys.stderr)
        return 2
    print("Telugu and Nepali OCR dependencies are ready.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
