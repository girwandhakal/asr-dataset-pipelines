"""Run the transcript and TalkBank media pipeline from start to finish."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .create_manifest import create_manifest
    from .download_media import download_media
    from .extract_transcripts import extract_archives
    from .validation import validate_export
except ImportError:  # Supports: python src/media_pipeline/main.py
    from create_manifest import create_manifest
    from download_media import download_media
    from extract_transcripts import extract_archives
    from validation import validate_export


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
WORKBOOK = DATA / "processed" / "mlu_balancing_results.xlsx"
TRANSCRIPTS = DATA / "transcripts"
MANIFEST = DATA / "processed" / "utterance_manifest.csv"
LOG = DATA / "out" / "manifest.log"
SUMMARY = DATA / "out" / "manifest_summary.txt"
MEDIA = DATA / "media"
MASTER_FILE = DATA / "childes_talkbank_master_file.xlsx"


def run_media_pipeline(
    *,
    cookies_file: Path | None = None,
    dry_run: bool = False,
    reuse_only: bool = False,
) -> int:
    """Run the three media-pipeline stages in order."""
    extract_archives(TRANSCRIPTS)
    create_manifest(
        WORKBOOK,
        TRANSCRIPTS,
        MANIFEST,
        LOG,
        SUMMARY,
    )
    status = download_media(
        MANIFEST,
        WORKBOOK,
        TRANSCRIPTS,
        MEDIA,
        master_file=MASTER_FILE,
        cookies_file=cookies_file,
        dry_run=dry_run,
        reuse_only=reuse_only,
    )
    if dry_run:
        return status
    complete = validate_export(MANIFEST, MEDIA)
    return status if status else 0 if complete else 2


def _parse_args() -> argparse.Namespace:
    """Parse media pipeline command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cookies-file",
        type=Path,
        help="JSON or Netscape cookies for authenticated TalkBank access.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the manifest and plan media downloads without creating clips.",
    )
    parser.add_argument("--reuse-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(run_media_pipeline(cookies_file=args.cookies_file, dry_run=args.dry_run, reuse_only=args.reuse_only))
