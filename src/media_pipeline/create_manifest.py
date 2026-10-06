"""Create a timestamped CHI utterance manifest from CHAT transcripts."""

from __future__ import annotations

import argparse
import csv
import posixpath
import re
import zipfile
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
TIMESTAMP = re.compile(r"\x15(\d+)_(\d+)\x15")
SPEAKER = re.compile(r"^\*([^:]+):\s*(.*)$")
FIELDS = ["corpus", "filepath", "start_seconds", "end_seconds", "raw_utterance", "utterance"]

try:
    from .ground_truth import clean_ground_truth
except ImportError:
    from ground_truth import clean_ground_truth

# The workbook's "age_months" column is not consistently in months: some
# corpora's upstream metadata source reports age in days instead, and both
# land in this one column undistinguished. True target-child ages in this
# dataset never exceed ~110 months (roughly 9 years), so a raw value below
# this threshold is already in months; at or above it, it is in days and
# needs converting. This is a data-quality workaround, not a unit choice.
AGE_DAY_THRESHOLD = 200
DAYS_PER_MONTH = 30.4368


def _column_number(reference: str) -> int:
    """Convert an Excel column reference such as ``C`` to a zero-based index."""
    letters = re.match(r"[A-Z]+", reference).group()
    number = 0
    for letter in letters:
        number = number * 26 + ord(letter) - ord("A") + 1
    return number - 1


def _xlsx_rows(path: Path, sheet_name: str) -> list[dict[str, str]]:
    """Read the simple string data needed from one XLSX worksheet."""
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(item.itertext()) for item in root.findall("m:si", NS)]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            item.attrib["Id"]: item.attrib["Target"]
            for item in relationships.findall(f"{{{REL_NS}}}Relationship")
        }
        for sheet in workbook.findall("m:sheets/m:sheet", NS):
            if sheet.attrib["name"] != sheet_name:
                continue
            relation = sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
            sheet_path = posixpath.normpath(posixpath.join("xl", targets[relation]))
            break
        else:
            raise KeyError(f"Worksheet not found: {sheet_name}")

        root = ET.fromstring(archive.read(sheet_path))
        rows = []
        for xml_row in root.findall(".//m:sheetData/m:row", NS):
            values = {}
            for cell in xml_row.findall("m:c", NS):
                value = cell.find("m:v", NS)
                text = "" if value is None else value.text or ""
                if cell.attrib.get("t") == "s" and text:
                    text = shared[int(text)]
                values[_column_number(cell.attrib["r"])] = text
            rows.append([values.get(i, "") for i in range(max(values, default=-1) + 1)])

    headers = [value.strip() for value in rows[0]]
    return [
        {header: value.strip() for header, value in zip(headers, row) if header}
        for row in rows[1:]
    ] if rows else []


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
        row.get("new_id")
        or row.get("manifest_new_id")
        or row.get("api_target_child_id")
        or Path(filename).stem
    ).strip()


def _selected_transcripts(workbook: Path) -> tuple[list[dict[str, str]], int]:
    """Read unique selected transcripts from the workbook's balance sheet."""
    rows = _xlsx_rows(workbook, "Balanced Utterances")
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


def _age_in_years(value: str) -> float | None:
    """Parse a CHILDES age string such as ``5;06`` into decimal years."""
    match = re.match(r"^\s*(\d+)(?:;(\d+))?", value)
    if not match:
        return None
    years = int(match.group(1))
    months = int(match.group(2) or 0)
    return years + months / 12


def _age_months(row: dict[str, str]) -> float | None:
    """Read the workbook's target-child age, normalized to decimal months.

    See ``AGE_DAY_THRESHOLD`` for why this isn't a straight float parse.
    """
    raw = (row.get("age_months") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value / DAYS_PER_MONTH if value >= AGE_DAY_THRESHOLD else value


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


def _target_speaker(text: str) -> str | None:
    """Find the target child speaker code from CHAT participant metadata."""
    participants = re.search(r"^@Participants:\s*(.*)$", text, re.MULTILINE)
    if participants and re.search(r"(?<!\w)CHI(?:\s|,|$)", participants.group(1)):
        return "CHI"

    # Some corpora use the participant's initials instead of CHI.
    candidates = []
    for line in re.findall(r"^@ID:\s*(.*)$", text, re.MULTILINE):
        fields = line.split("|")
        if len(fields) < 4 or not fields[2]:
            continue
        age = _age_in_years(fields[3])
        if age is not None and 5 < age < 9:
            role = fields[7].casefold() if len(fields) > 7 else ""
            candidates.append((role != "child", fields[2]))
    candidates.sort()
    return candidates[0][1] if candidates else None


def _seconds(milliseconds: str) -> str:
    """Convert a CHAT millisecond timestamp into a clean seconds string."""
    value = format(Decimal(milliseconds) / 1000, "f")
    # Strip zeroes only from the fractional part. Calling rstrip("0") on the
    # whole value would turn an integer such as 160.000 into 16.
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value or "0"


def _clean_utterance(text: str) -> str:
    """Lowercase a CHAT utterance and strip its annotation codes."""
    return clean_ground_truth(text)


def _read_utterances(
    transcript: Path,
    corpus: str,
    transcripts_dir: Path,
    log: list[str],
    counts: dict[str, int],
) -> list[dict[str, str]]:
    """Extract timestamped utterances for the target speaker in one file."""
    relative = transcript.relative_to(transcripts_dir.parent).as_posix()
    text = transcript.read_text(encoding="utf-8-sig", errors="replace")
    speaker = _target_speaker(text)
    if speaker is None:
        counts["no_target_speaker"] += 1
        log.append(f"SKIP transcript (no target speaker): {relative}")
        return []

    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = SPEAKER.match(line)
        if match is None or match.group(1).strip() != speaker:
            continue
        timestamp = TIMESTAMP.search(match.group(2))
        if timestamp is None:
            counts["untimestamped_utterances"] += 1
            log.append(f"SKIP utterance (no timestamp): {relative}:{line_number}")
            continue
        utterance = TIMESTAMP.sub("", match.group(2)).strip()
        rows.append(
            {
                "corpus": corpus,
                "filepath": relative,
                "start_seconds": _seconds(timestamp.group(1)),
                "end_seconds": _seconds(timestamp.group(2)),
                "raw_utterance": utterance,
                "utterance": _clean_utterance(utterance),
            }
        )
    return rows


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
break that subset back out for reference. Age ranges come from the
workbook's target-child age at each transcript and are normalized to
years;months (see AGE_DAY_THRESHOLD in create_manifest.py: some corpora's
source metadata records this in days rather than months).

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


def create_manifest(
    workbook: Path,
    transcripts_dir: Path,
    output: Path,
    log_path: Path,
    summary_path: Path,
) -> tuple[int, int, int]:
    """Create the timestamp manifest, skip log, and summary report.

    The workbook's selected transcript references are matched to local CHAT
    files. Target-speaker utterances with valid CHAT timestamps are written to
    the manifest; missing files, ambiguous matches, and unusable utterances
    are recorded in the log.

    Args:
        workbook: Workbook containing the ``Balanced Utterances`` sheet.
        transcripts_dir: Root directory containing extracted corpus folders.
        output: Destination CSV manifest path.
        log_path: Destination text log for skipped items.
        summary_path: Destination text summary path.

    Returns:
        ``(transcripts_found, utterances_written, skipped_items)``.

    Raises:
        FileNotFoundError: If the workbook or transcript directory is missing.
        KeyError: If the workbook does not contain the required sheet.
    """
    selected, workbook_utterances = _selected_transcripts(workbook)
    files = _files_by_name(transcripts_dir)
    log = []
    manifest = []
    found = 0
    counts = defaultdict(int)
    transcript_stats = []

    for item in selected:
        corpus, filename = item["corpus"], item["filename"]
        try:
            transcript = _resolve_file(transcripts_dir, corpus, filename, files)
        except ValueError as error:
            counts["ambiguous_transcripts"] += 1
            log.append(f"SKIP transcript ({error})")
            continue
        if transcript is None:
            counts["missing_transcripts"] += 1
            log.append(f"SKIP transcript (file not found): {corpus}: {filename}")
            continue
        found += 1
        rows = _read_utterances(transcript, corpus, transcripts_dir, log, counts)
        if rows:
            duration = sum(
                (
                    Decimal(row["end_seconds"]) - Decimal(row["start_seconds"])
                    for row in rows
                ),
                Decimal("0"),
            )
            transcript_stats.append(
                {
                    "corpus": corpus,
                    "group": item["group"],
                    "raw_group": item["raw_group"],
                    "child_id": item["child_id"],
                    "age_months": item["age_months"],
                    "utterances": len(rows),
                    "duration": duration,
                }
            )
        manifest.extend(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(manifest)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log) + ("\n" if log else ""), encoding="utf-8")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        _summary(
            workbook,
            output,
            len(selected),
            workbook_utterances,
            found,
            transcript_stats,
            counts,
        ),
        encoding="utf-8",
    )
    return found, len(manifest), len(log)


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
