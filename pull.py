#!/usr/bin/env python3
"""Compatibility entry point for the organized collector package."""

from lma_pipeline.pulling import *
from lma_pipeline.pulling.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
