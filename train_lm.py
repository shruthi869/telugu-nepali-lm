#!/usr/bin/env python3
"""Compatibility launcher for independent decoder-only pretraining."""

from lma_lm.cli import train_main


if __name__ == "__main__":
    raise SystemExit(train_main())
