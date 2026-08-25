"""Extract transcript ZIP files once into ``data/transcripts``."""

from __future__ import annotations

import zipfile
from pathlib import Path


def extract_archives(
    transcripts_dir: Path = Path(__file__).resolve().parents[2] / "data" / "transcripts",
    *,
    dry_run: bool = False,
) -> int:
    """Extract transcript ZIPs unless their destination folders already exist.

    Args:
        transcripts_dir: Folder containing ZIP archives and extracted corpus
            folders. Defaults to ``data/transcripts``.
        dry_run: If true, validate and list archives without writing files.

    Returns:
        Number of archives extracted. Existing destination folders are not
        counted. In dry-run mode, this is the number that would be extracted.

    Raises:
        NotADirectoryError: If ``transcripts_dir`` does not exist.
        ValueError: If an archive contains a path that would escape its
            destination folder.
    """
    if not transcripts_dir.is_dir():
        raise NotADirectoryError(f"Transcripts directory not found: {transcripts_dir}")

    archives = sorted(
        path for path in transcripts_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".zip"
    )

    processed = 0
    skipped = 0
    for archive in archives:
        destination = transcripts_dir / archive.stem
        if destination.is_dir():
            print(f"Skip: {archive.name}; {destination.name}/ already exists")
            skipped += 1
            continue

        print(f"Extract: {archive.name} -> {destination.name}/")

        with zipfile.ZipFile(archive) as zip_file:
            # Do not allow a ZIP entry to write outside its destination folder.
            destination_root = destination.resolve()
            for member in zip_file.infolist():
                member_path = (destination / member.filename).resolve()
                if destination_root not in member_path.parents and member_path != destination_root:
                    raise ValueError(f"Unsafe path in archive: {member.filename!r}")

            if not dry_run:
                destination.mkdir(parents=True, exist_ok=True)
                zip_file.extractall(destination)
        processed += 1

    print(f"Archives found: {len(archives)}; processed: {processed}; skipped: {skipped}")
    return processed
