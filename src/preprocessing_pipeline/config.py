"""Paths, mappings, and column configuration for the Child ASR workflow."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MASTER_MANIFEST = ROOT / "data" / "childes_talkbank_master_file.xlsx"
OUTPUT_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

GROUP_MAP = {"TD": "TD", "LT": "LT", "SLI": "LT"}
EXCLUDED_CORPORA = set()

SOURCE_COLUMNS = [
    "transcript_path", "manifest_new_id", "language", "media_type",
    "recording_date_raw", "pid", "design", "activity", "source_group",
]

TRANSCRIPT_VARIABLES = [
    "id", "corpus_name", "language", "date", "filename", "target_child_name",
    "target_child_age", "target_child_sex", "pid",
]

UTTERANCE_VARIABLES = [
    "id", "transcript_id", "gloss", "speaker_code", "speaker_role",
    "num_morphemes", "num_tokens", "target_child_name", "target_child_age",
    "target_child_sex", "corpus_name",
]
