#!/usr/bin/env python3
"""Compatibility entry point for the organized preprocessing package."""

from lma_pipeline import *
from lma_pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
