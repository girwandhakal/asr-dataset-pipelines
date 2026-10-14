"""Run the Child ASR manifest workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing_pipeline.build_manifest import build_manifest


def run_preprocessing_pipeline(*, refresh_data=False, cap_center="mean", cap_sd=3, seed=42):
    """Run the preprocessing pipeline."""
    build_manifest(refresh=refresh_data, cap_center=cap_center, cap_sd=cap_sd, seed=seed)


def _parse_args() -> argparse.Namespace:
    """Parse preprocessing pipeline command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download fresh Redivis tables instead of using cached CSV files.",
    )
    parser.add_argument("--cap-center", choices=["mean", "median"], default="mean")
    parser.add_argument("--cap-sd", type=float, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_preprocessing_pipeline(refresh_data=args.refresh_data, cap_center=args.cap_center,
                               cap_sd=args.cap_sd, seed=args.seed)
