"""Run the Child ASR manifest workflow."""

from __future__ import annotations

import argparse

try:
    from .build_manifest import build_manifest
except ImportError:  # Supports: python src/preprocessing_pipeline/main.py
    from build_manifest import build_manifest


def run_preprocessing_pipeline(*, refresh_data: bool = False) -> None:
    """Run the preprocessing pipeline."""
    build_manifest(refresh=refresh_data)


def _parse_args() -> argparse.Namespace:
    """Parse preprocessing pipeline command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download fresh Redivis tables instead of using cached CSV files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_preprocessing_pipeline(refresh_data=_parse_args().refresh_data)
