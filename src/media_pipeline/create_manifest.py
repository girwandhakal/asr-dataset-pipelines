"""Export the exact selected utterances, with participant IDs and timestamps."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


FIELDS = ["corpus", "filepath", "start_seconds", "end_seconds", "raw_utterance", "utterance",
          "child_id", "transcript_id", "utterance_id", "group", "source_group", "design", "speaker_code", "clip_ordinal"]

def _group(row: dict[str, str]) -> str:
    """Normalize workbook group labels, mapping SLI into LT."""
    group = (row.get("group") or row.get("source_group") or "").strip().upper()
    return "LT" if group in {"LT", "SLI"} else group or "UNKNOWN"


def _raw_group(row: dict[str, str]) -> str:
    """Return the workbook's original group label, keeping SLI distinct.

    ``source_group`` carries the pre-balancing LT/SLI/TD label; ``group``
    already has SLI folded into LT for balancing purposes. This is used only
    to report an SLI breakdown alongside the LT/TD summary, never to change
    which utterances are selected.
    """
    raw = (row.get("source_group") or row.get("group") or "").strip().upper()
    return raw or "UNKNOWN"


def _child_id(row: dict[str, str], filename: str) -> str:
    """Choose the best available child ID for a selected transcript row."""
    return (
        row.get("child_id")
        or row.get("new_id")
        or row.get("manifest_new_id")
        or row.get("api_target_child_id")
        or Path(filename).stem
    ).strip()


def _selected_transcripts(workbook: Path) -> tuple[list[dict[str, str]], int]:
    """Read unique selected transcripts from the final CSV beside the workbook."""
    source = workbook.parent / "all_utterances_clean_mlu_balanced.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"corpus", "filename"}.issubset(rows[0]):
        raise ValueError("Balanced Utterances must contain corpus and filename")

    selected = []
    seen = set()
    for row in rows:
        corpus, filename = row.get("corpus", ""), row.get("filename", "")
        key = (corpus.casefold(), filename.casefold())
        if corpus and filename and key not in seen:
            selected.append(
                {
                    "corpus": corpus,
                    "filename": filename,
                    "group": _group(row),
                    "raw_group": _raw_group(row),
                    "child_id": _child_id(row, filename),
                    "age_months": _age_months(row),
                }
            )
            seen.add(key)
    return selected, len(rows)


def _files_by_name(transcripts_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """Index local CHAT files by corpus and case-insensitive filename stem."""
    files = defaultdict(list)
    for corpus_dir in transcripts_dir.iterdir():
        if corpus_dir.is_dir():
            for path in corpus_dir.rglob("*.cha"):
                files[(corpus_dir.name.casefold(), path.stem.casefold())].append(path)
    return files


def _resolve_file(
    transcripts_dir: Path,
    corpus: str,
    filename: str,
    files: dict[tuple[str, str], list[Path]],
) -> Path | None:
    """Resolve one workbook transcript reference to a local CHAT file."""
    workbook_parts = filename.replace("\\", "/").strip("/").split("/")
    key = (corpus.casefold(), Path(workbook_parts[-1]).stem.casefold())
    candidates = files.get(key, [])
    if len(candidates) <= 1:
        return candidates[0] if candidates else None

    # Reused filenames are disambiguated by matching directory names.
    wanted = {part.casefold() for part in workbook_parts[:-1]}
    corpus_dir = transcripts_dir / corpus
    scored = [
        (sum(part.casefold() in wanted for part in path.relative_to(corpus_dir).parts), path)
        for path in candidates
    ]
    best_score = max(score for score, _ in scored)
    best = [path for score, path in scored if score == best_score]
    if len(best) != 1:
        raise ValueError(f"Ambiguous transcript: {filename}")
    return best[0]


def _age_months(row: dict[str, str]) -> float | None:
    """Read the CHAT-derived target age, already expressed in months."""
    raw = (row.get("age_months") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value


def _format_age(months: float) -> str:
    """Format decimal months as a CHAT-style ``years;months`` age string."""
    years, remainder = divmod(round(months), 12)
    return f"{years};{remainder:02d}"


def _format_age_range(ages: list[float]) -> str:
    """Format an age range, or a single age when there's only one value."""
    if not ages:
        return "n/a"
    low, high = _format_age(min(ages)), _format_age(max(ages))
    return low if low == high else f"{low}–{high}"


def _aggregate(items: list[dict], key_name: str) -> dict[str, dict]:
    """Aggregate transcript statistics by group or corpus."""
    result = defaultdict(
        lambda: {
            "children": set(),
            "datasets": set(),
            "transcripts": 0,
            "utterances": 0,
            "duration": Decimal("0"),
            "ages": [],
        }
    )
    for item in items:
        bucket = result[item[key_name]]
        bucket["children"].add((item["corpus"], item["child_id"]))
        bucket["datasets"].add(item["corpus"])
        bucket["transcripts"] += 1
        bucket["utterances"] += item["utterances"]
        bucket["duration"] += item["duration"]
        if item.get("age_months") is not None:
            bucket["ages"].append(item["age_months"])
    return result


def _table(headers: list[str], rows: list[list[object]]) -> str:
    """Format rows as a plain-text aligned table for the summary file."""
    values = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in values:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    lines = [
        "  ".join(header.ljust(width) for header, width in zip(headers, widths)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(width) for value, width in zip(row, widths))
        for row in values
    )
    return "\n".join(lines)


def _summary(
    workbook: Path,
    output: Path,
    selected_count: int,
    workbook_utterances: int,
    found_count: int,
    transcript_stats: list[dict],
    counts: dict[str, int],
) -> str:
    """Build the human-readable manifest summary text."""
    total_utterances = sum(item["utterances"] for item in transcript_stats)
    total_duration = sum(
        (item["duration"] for item in transcript_stats), Decimal("0")
    )
    all_children = {
        (item["corpus"], item["child_id"]) for item in transcript_stats
    }
    all_ages = [
        item["age_months"] for item in transcript_stats if item.get("age_months") is not None
    ]
    group_stats = _aggregate(transcript_stats, "group")
    raw_group_stats = _aggregate(transcript_stats, "raw_group")
    dataset_stats = _aggregate(transcript_stats, "corpus")
    dataset_group_stats = defaultdict(list)
    dataset_raw_group_stats = defaultdict(list)
    for item in transcript_stats:
        dataset_group_stats[(item["corpus"], item["group"])].append(item)
        dataset_raw_group_stats[(item["corpus"], item["raw_group"])].append(item)

    group_rows = []
    for group, values in sorted(group_stats.items()):
        group_rows.append(
            [
                group,
                _format_age_range(values["ages"]),
                len(values["children"]),
                values["transcripts"],
                values["utterances"],
                len(values["datasets"]),
                f"{values['duration']:,.3f}",
            ]
        )
        # SLI is folded into LT for balancing; break it back out here so the
        # summary still shows how much of LT is SLI specifically.
        if group == "LT" and "SLI" in raw_group_stats:
            sli = raw_group_stats["SLI"]
            group_rows.append(
                [
                    "  of which SLI",
                    _format_age_range(sli["ages"]),
                    len(sli["children"]),
                    sli["transcripts"],
                    sli["utterances"],
                    len(sli["datasets"]),
                    f"{sli['duration']:,.3f}",
                ]
            )

    def _group_cell(dataset: str, group: str, items: list[dict]) -> str:
        """Format one group's utterance count, breaking out SLI under LT."""
        utterances = sum(item["utterances"] for item in items)
        cell = f"{group}: {utterances:,}"
        if group == "LT":
            sli_items = dataset_raw_group_stats.get((dataset, "SLI"))
            if sli_items:
                sli_utterances = sum(item["utterances"] for item in sli_items)
                cell += f" (SLI: {sli_utterances:,})"
        return cell

    dataset_rows = []
    for dataset, values in sorted(dataset_stats.items()):
        dataset_rows.append(
            [
                dataset,
                _format_age_range(values["ages"]),
                ", ".join(
                    _group_cell(dataset, group, items)
                    for (name, group), items in sorted(dataset_group_stats.items())
                    if name == dataset
                ),
                len(values["children"]),
                values["transcripts"],
                values["utterances"],
                f"{values['duration']:,.3f}",
            ]
        )

    return f"""Manifest summary
================

Source workbook: {workbook}
Manifest:        {output}

The counts below describe rows written to the final manifest. SLI is grouped
with LT, as requested; the "of which SLI" row and "(SLI: n)" figures below
break that subset back out for reference. Ages come from each transcript's
target-child CHAT header and are displayed as years;months.

Overall
-------
Balanced workbook utterance rows: {workbook_utterances:,}
Unique selected transcript references: {selected_count:,}
Transcript files found: {found_count:,}
Transcripts represented in manifest: {len(transcript_stats):,}
Datasets represented: {len(dataset_stats):,}
Children represented: {len(all_children):,}
Utterances written: {total_utterances:,}
Total timestamped audio duration (seconds): {total_duration:,.3f}
Target-child age range (years;months): {_format_age_range(all_ages)}

By group
--------
{_table(
    ["Group", "Age", "Children", "Transcripts", "Utterances", "Datasets", "Duration (s)"],
    group_rows,
)}

By dataset
----------
{_table(
    ["Dataset", "Age", "Utterances by group", "Children", "Transcripts", "Utterances", "Duration (s)"],
    dataset_rows,
)}

Skipped items
-------------
Missing transcript files: {counts['missing_transcripts']:,}
Ambiguous transcript paths: {counts['ambiguous_transcripts']:,}
Transcripts without a usable target speaker: {counts['no_target_speaker']:,}
Target-child utterances without timestamps: {counts['untimestamped_utterances']:,}
"""


def create_selected_manifest(workbook, output, log_path, summary_path):
    """Export the exact selected rows instead of reintroducing dropped CHAT turns."""
    source = workbook.parent / "all_utterances_clean_mlu_balanced.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Rerun preprocessing to produce child-aware timestamped utterances")
        rows = list(reader)
    if not rows:
        raise ValueError("The final utterance selection is empty")
    if any(not row["child_id"].startswith(row["corpus"] + "::") or not row["transcript_id"] for row in rows):
        raise ValueError("Selected utterances need canonical child IDs and source transcript IDs")
    if len({row["utterance_id"] for row in rows}) != len(rows):
        raise ValueError("Selected utterance IDs are not unique")
    by_file = defaultdict(list)
    for row in rows:
        by_file[row["filepath"]].append(row)
    stats = []
    for values in by_file.values():
        row = values[0]
        stats.append(dict(corpus=row["corpus"], group=row["group"], raw_group=row["source_group"],
                          child_id=row["child_id"], age_months=float(row["age_months"]) if row["age_months"] else None,
                          utterances=len(values), duration=sum(
                              (Decimal(r["end_seconds"]) - Decimal(r["start_seconds"]) for r in values), Decimal(0))))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions = workbook.parent / "utterance_exclusions.csv"
    log = []
    counts = defaultdict(int)
    if exclusions.exists():
        with exclusions.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["filepath"] in by_file:
                    reason = row["reason"].replace("_", " ")
                    log.append(f"SKIP utterance ({reason}): {row['filepath']}:{row['line']}")
                    counts["untimestamped_utterances"] += row["reason"] == "no_timestamp"
    log_path.write_text("\n".join(log) + "\nSee participant_transcript_mapping.csv for recording exclusions.\n", encoding="utf-8")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_summary(workbook, output, len(by_file), len(rows), len(by_file), stats, counts), encoding="utf-8")
    return len(by_file), len(rows), len(log)


def create_manifest(workbook, transcripts_dir, output, log_path, summary_path):
    """Export the exact child-aware selection produced by preprocessing."""
    return create_selected_manifest(workbook, output, log_path, summary_path)


def main() -> None:
    """Parse command-line paths and create the manifest files."""
    repository = Path(__file__).resolve().parents[2]
    data_dir = repository / "data"
    parser = argparse.ArgumentParser(description="Create a CHI utterance manifest.")
    parser.add_argument(
        "--workbook",
        type=Path,
        default=data_dir / "processed" / "mlu_balancing_results.xlsx",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=data_dir / "transcripts",
    )
    parser.add_argument(
        "--output", type=Path, default=data_dir / "processed" / "utterance_manifest.csv"
    )
    parser.add_argument("--log", type=Path, default=data_dir / "out" / "manifest.log")
    parser.add_argument(
        "--summary", type=Path, default=data_dir / "out" / "manifest_summary.txt"
    )
    args = parser.parse_args()

    found, utterances, skipped = create_manifest(
        args.workbook.resolve(),
        args.transcripts_dir.resolve(),
        args.output.resolve(),
        args.log.resolve(),
        args.summary.resolve(),
    )
    print(f"Transcripts used: {found}")
    print(f"Utterances written: {utterances}")
    print(f"Items skipped; see log: {skipped}")
    print(f"Manifest: {args.output.resolve()}")
    print(f"Log: {args.log.resolve()}")
    print(f"Summary: {args.summary.resolve()}")


if __name__ == "__main__":
    main()
