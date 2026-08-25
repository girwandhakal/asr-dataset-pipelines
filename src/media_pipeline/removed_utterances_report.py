"""Summarize the target-child utterances skipped for missing timestamps.

create_manifest.py logs one "SKIP utterance (no timestamp): <path>:<line>"
entry per target-child (CHI) line whose CHAT transcript has no `%snd`/`%mov`
bullet. This script reads data/out/manifest.log and reports how many there are, from
how many datasets/transcripts/children, broken down by dataset and by group
(LT/TD, with SLI broken out of LT), plus how many transcripts lost *every*
target-child utterance versus only some.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

try:
    from .create_manifest import _files_by_name, _resolve_file, _selected_transcripts, _table
except ImportError:  # Supports: python src/media_pipeline/removed_utterances_report.py
    from create_manifest import _files_by_name, _resolve_file, _selected_transcripts, _table


SKIP_LINE = re.compile(r"^SKIP utterance \(no timestamp\): (.+):(\d+)$")


def _resolved_lookup(
    workbook: Path, transcripts_dir: Path
) -> dict[Path, dict[str, str]]:
    """Map each resolved local transcript file to its group/child metadata.

    Reuses the exact same workbook selection and file-resolution logic
    create_manifest.py used to build the manifest, so the dataset/group
    breakdown here matches it precisely.
    """
    selected, _ = _selected_transcripts(workbook)
    files = _files_by_name(transcripts_dir)
    lookup: dict[Path, dict[str, str]] = {}
    for item in selected:
        try:
            transcript = _resolve_file(transcripts_dir, item["corpus"], item["filename"], files)
        except ValueError:
            continue
        if transcript is not None:
            lookup[transcript] = item
    return lookup


def _parse_skips(log_path: Path, data_dir: Path) -> list[tuple[Path, str, int]]:
    """Parse skip log entries into (resolved path, relative path, line)."""
    skips = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = SKIP_LINE.match(line)
        if match is None:
            continue
        relative, line_number = match.group(1), int(match.group(2))
        skips.append((data_dir / relative, relative, line_number))
    return skips


def summarize(
    log_path: Path,
    workbook: Path,
    transcripts_dir: Path,
    manifest_path: Path,
) -> str:
    """Build the human-readable removed-utterances summary text."""
    data_dir = transcripts_dir.parent
    skips = _parse_skips(log_path, data_dir)
    lookup = _resolved_lookup(workbook, transcripts_dir)

    # Transcripts that still have at least one timestamped row in the final
    # manifest -- used to tell "lost every utterance" apart from "lost some".
    kept_transcripts: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            kept_transcripts = {row["filepath"] for row in csv.DictReader(handle)}

    by_dataset = defaultdict(
        lambda: {"utterances": 0, "transcripts": set(), "children": set()}
    )
    by_group = defaultdict(lambda: {"utterances": 0, "transcripts": set()})
    by_raw_group = defaultdict(lambda: {"utterances": 0, "transcripts": set()})
    transcripts_affected: dict[str, dict[str, str]] = {}
    unmatched = 0

    for resolved, relative, _line_number in skips:
        metadata = lookup.get(resolved)
        corpus = metadata["corpus"] if metadata else (relative.split("/")[1] if "/" in relative else "UNKNOWN")
        bucket = by_dataset[corpus]
        bucket["utterances"] += 1
        bucket["transcripts"].add(relative)
        if metadata:
            bucket["children"].add((corpus, metadata["child_id"]))
            by_group[metadata["group"]]["utterances"] += 1
            by_group[metadata["group"]]["transcripts"].add(relative)
            by_raw_group[metadata["raw_group"]]["utterances"] += 1
            by_raw_group[metadata["raw_group"]]["transcripts"].add(relative)
            transcripts_affected[relative] = metadata
        else:
            unmatched += 1

    fully_removed = sorted(t for t in transcripts_affected if t not in kept_transcripts)
    partially_removed = sorted(t for t in transcripts_affected if t in kept_transcripts)

    dataset_rows = [
        [
            dataset,
            f"{values['utterances']:,}",
            len(values["transcripts"]),
            len(values["children"]),
        ]
        for dataset, values in sorted(by_dataset.items())
    ]

    group_rows = []
    for group, values in sorted(by_group.items()):
        group_rows.append([group, f"{values['utterances']:,}", len(values["transcripts"])])
        if group == "LT" and "SLI" in by_raw_group:
            sli = by_raw_group["SLI"]
            group_rows.append(
                ["  of which SLI", f"{sli['utterances']:,}", len(sli["transcripts"])]
            )

    total_utterances = len(skips)
    total_transcripts = len({relative for _, relative, _ in skips})
    total_datasets = len(by_dataset)
    total_children = len({child for values in by_dataset.values() for child in values["children"]})

    lines = [
        "Removed / skipped utterance summary",
        "====================================",
        "",
        f"Source log: {log_path}",
        "",
        "These are target-child (CHI) utterances that create_manifest.py",
        "skipped because their CHAT line has no %snd/%mov timestamp bullet.",
        "",
        "Overall",
        "-------",
        f"Removed utterances: {total_utterances:,}",
        f"Transcripts affected: {total_transcripts:,}",
        f"  - lost every target-child utterance: {len(fully_removed):,}",
        f"  - lost only some target-child utterances: {len(partially_removed):,}",
        f"Datasets affected: {total_datasets:,}",
        f"Children affected: {total_children:,}",
    ]
    if unmatched:
        lines.append(
            f"Utterances not matched to workbook metadata: {unmatched:,} "
            "(counted in the overall/by-dataset totals above, but excluded "
            "from the group breakdown and child counts below)"
        )
    lines += [
        "",
        "By dataset",
        "----------",
        _table(["Dataset", "Removed utterances", "Transcripts", "Children"], dataset_rows),
        "",
        "By group",
        "--------",
        _table(["Group", "Removed utterances", "Transcripts"], group_rows),
        "",
        "Transcripts that lost every target-child utterance",
        "----------------------------------------------------",
    ]
    lines += [f"  {relative}" for relative in fully_removed] if fully_removed else ["  (none)"]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the removed-utterances report."""
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[2]
    data_dir = repository / "data"
    parser.add_argument("--log", type=Path, default=data_dir / "out" / "manifest.log")
    parser.add_argument(
        "--workbook", type=Path, default=data_dir / "processed" / "mlu_balancing_results.xlsx"
    )
    parser.add_argument("--transcripts-dir", type=Path, default=data_dir / "transcripts")
    parser.add_argument(
        "--manifest", type=Path, default=data_dir / "processed" / "utterance_manifest.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=data_dir / "out" / "removed_utterances_summary.txt"
    )
    return parser


def main() -> None:
    """Generate and write the removed-utterances summary report."""
    args = build_parser().parse_args()
    report = summarize(
        args.log.resolve(),
        args.workbook.resolve(),
        args.transcripts_dir.resolve(),
        args.manifest.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()
