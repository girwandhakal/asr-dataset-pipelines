"""Run the complete dataset pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .media_pipeline.main import run_media_pipeline
    from .preprocessing_pipeline.main import run_preprocessing_pipeline
except ImportError:  # Supports: python src/main.py
    from media_pipeline.main import run_media_pipeline
    from preprocessing_pipeline.main import run_preprocessing_pipeline


def run_dataset_pipeline(
    *,
    refresh_data: bool = False,
    cookies_file: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Run preprocessing first, then transcript and media processing."""
    run_preprocessing_pipeline(refresh_data=refresh_data)
    return run_media_pipeline(cookies_file=cookies_file, dry_run=dry_run)


def _parse_args() -> argparse.Namespace:
    """Parse complete pipeline command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download fresh Redivis tables instead of using cached CSV files.",
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        help="JSON or Netscape cookies for authenticated TalkBank access.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan media downloads without creating clips.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_dataset_pipeline(
        refresh_data=args.refresh_data,
        cookies_file=args.cookies_file,
        dry_run=args.dry_run,
    )
