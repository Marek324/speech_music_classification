"""Convenience wrapper — equivalent to: uv run python build.py --tier mini"""
import sys
from build import main

sys.argv = [sys.argv[0], "--tier", "mini"] + sys.argv[1:]
main()
