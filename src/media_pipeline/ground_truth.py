"""Finalize the media report as the audio/ground-truth handoff to GenSEC.

Can update an existing report without downloading, decoding, or rewriting audio:
    python src/media_pipeline/ground_truth.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path, PureWindowsPath

REFERENCE_VERSION = "chat_ground_truth_v1"
REPORT_COLUMNS = ["utterance_id", "utterance", "raw_utterance", "audio_path",
                  "corpus", "group", "child_id", "filepath",
                  "start_seconds", "end_seconds", "manifest_row"]
BRACKET = re.compile(r"\[[^\[\]]*\]")
# Match complete annotation tokens, never the + inside snow+man.
EVENT = re.compile(r"(?<!\S)[&+]\S*")
SPECIAL = re.compile(r"@[^\s<>\[\]]+")
UNSPOKEN = re.compile(r"(?<!\S)0\S*")
UNINTELLIGIBLE = re.compile(r"\b(?:xxx|yyy|www)\b")
PUNCTUATION = re.compile(r"[^\w\s']")
HUM_FORMS = {"mm", "mmm", "mhm", "mmhm", "mmhmm", "hm", "hmm", "hmhm", "mhmm"}
MOJIBAKE = {
    "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"',
    "â€“": "-", "â€”": "-", "â tms": "'s", "â tm": "'", "â€": '"',
}


def clean_ground_truth(text: str) -> str:
    """Apply the documented lexical-reference policy once, upstream.

    Filled pauses (&-), fragments (&+), and events (&=) are excluded by the
    existing policy. Preserve words within scope brackets and before @ tags.
    Canonicalize hum spellings here so downstream WER never rewrites references.
    """
    # Repair encoding before NFKC can alter characters in mojibake sequences.
    for broken, fixed in MOJIBAKE.items():
        text = text.replace(broken, fixed)
    text = unicodedata.normalize("NFKC", text).replace("’", "'")
    while True:
        stripped = BRACKET.sub(" ", text)
        if stripped == text:
            break
        text = stripped
    # Scope boundaries can sit immediately before annotation tokens.
    text = text.replace("<", " ").replace(">", " ")
    text = EVENT.sub(" ", text)
    text = SPECIAL.sub("", text)
    text = UNSPOKEN.sub(" ", text)
    text = text.replace("(", "").replace(")", "").replace("_", " ")
    # CHAT uses + as a compound boundary: snow+man -> snowman.
    text = re.sub(r"(?<=\w)\+(?=\w)", "", text)
    text = UNINTELLIGIBLE.sub(" ", text.lower())
    text = " ".join(PUNCTUATION.sub(" ", text).split())
    return " ".join("mm" if word in HUM_FORMS else word for word in text.split())


def finalize_rows(rows: list[dict], media_dir: Path) -> list[dict]:
    """Clean references and resolve paths against this media copy, not old hosts."""
    media_dir = media_dir.resolve()
    seen: set[str] = set()
    directories = {}
    finalized = []
    for source in rows:
        row = dict(source)
        version = row.get("reference_cleaning_version", "")
        if version and "raw_utterance" not in row:
            raise ValueError("Versioned report is missing raw_utterance; cannot reclean safely")
        raw = row.get("raw_utterance", row.get("utterance", "")) or ""
        row["raw_utterance"] = raw
        row["utterance"] = clean_ground_truth(raw)
        row.pop("clean_utterance", None)  # one authoritative cleaned-text column
        row["reference_cleaning_version"] = REFERENCE_VERSION
        output = str(row.get("output") or row.get("audio_path") or "").strip()
        row["utterance_id"] = PureWindowsPath(output).stem if output else ""
        row["audio_path"] = ""
        if output:
            # Reports historically contain absolute Windows paths; use their
            # suffix under media to keep the handoff portable to Linux/HPC.
            parts = PureWindowsPath(output).parts
            indices = [i for i, part in enumerate(parts) if part.casefold() == "media"]
            if indices:
                relative = Path(*parts[indices[-1] + 1:])
            elif not PureWindowsPath(output).is_absolute() and not Path(output).is_absolute():
                relative = Path(*parts)
            else:
                try:
                    relative = Path(output).resolve().relative_to(media_dir)
                except ValueError as error:
                    raise ValueError(f"Audio path is outside the media directory: {output}") from error
            if ".." in relative.parts or relative.is_absolute():
                raise ValueError(f"Audio path escapes media directory: {output}")
            if relative.parent not in directories:
                parent = (media_dir / relative.parent).resolve()
                if not parent.is_relative_to(media_dir):
                    raise ValueError(f"Audio path escapes media directory: {output}")
                files = {}
                if parent.is_dir():
                    with os.scandir(parent) as entries:
                        files = {entry.name: entry.stat().st_size for entry in entries
                                 if entry.is_file(follow_symlinks=False)}
                directories[relative.parent] = files
            audio_exists = directories[relative.parent].get(relative.name, 0) > 0
            row["audio_path"] = relative.as_posix()
            # The filename encodes transcript identity and the exact extraction
            # interval. Catch mismatched text/audio rows before writing the CSV.
            if all(key in row for key in ("corpus", "filepath", "start_seconds", "end_seconds")):
                source_hash = hashlib.blake2s(
                    f"{row['corpus']}|{row['filepath']}".encode("utf-8"), digest_size=4
                ).hexdigest()
                start, end = float(row["start_seconds"]), float(row["end_seconds"])
                if not math.isfinite(start) or not math.isfinite(end):
                    raise ValueError("Non-finite report timestamps")
                expected = f"{source_hash}_{round(start * 1000)}_{round(end * 1000)}"
                actual = "_".join(row["utterance_id"].split("_")[:-1])
                if actual != expected:
                    raise ValueError(f"Transcript/timestamp mismatch for {row['utterance_id']}")

            if row["utterance_id"] in seen:
                raise ValueError(f"Duplicate clip ID: {row['utterance_id']}")
            seen.add(row["utterance_id"])
        if not row["utterance"]:
            reason = "empty_cleaned_reference"
        elif not output:
            reason = "no_audio_output"
        elif "status" in row and row.get("status") not in {"created", "already_exists"}:
            reason = "audio_not_successful"
        elif not audio_exists:
            reason = "missing_or_empty_audio"
        else:
            reason = ""
        if reason in {"audio_not_successful", "missing_or_empty_audio"}:
            row["audio_path"] = ""
        row["reference_status"] = "excluded" if reason else "ready"
        row["reference_exclusion_reason"] = reason
        finalized.append(row)
    return finalized


def finalize_report(report_path: Path, media_dir: Path | None = None) -> Counter:
    """Atomically update the CSV and preserve a one-time pre-migration backup."""
    csv.field_size_limit(10_000_000)
    with report_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"utterance", "raw_utterance"}.issubset(fields) and "output" not in fields:
            raise ValueError("Report requires final/raw utterances or a legacy output column")
        if not ({"audio_path", "output"} & fields):
            raise ValueError("Report requires an audio_path column")
        rows = finalize_rows(list(reader), media_dir or report_path.parent)
    backup = report_path.with_name(report_path.stem + ".before_ground_truth_v1.csv")
    if not backup.exists():
        shutil.copy2(report_path, backup)
    fields = REPORT_COLUMNS
    temporary = report_path.with_name(report_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in REPORT_COLUMNS} for row in rows)
    temporary.replace(report_path)
    counts = Counter(row["reference_exclusion_reason"] or "ready" for row in rows)
    print(f"Finalized ground truth: {dict(counts)}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[2] / "data/media/media_download_report.csv")
    args = parser.parse_args()
    finalize_report(args.report)
