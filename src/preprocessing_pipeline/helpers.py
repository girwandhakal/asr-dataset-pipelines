"""Reusable helper functions for the Child ASR workflow."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

try:
    from .config import (
        EXCLUDED_CORPORA,
        GROUP_MAP,
        RAW_DIR,
        SOURCE_COLUMNS,
        TRANSCRIPT_VARIABLES,
        UTTERANCE_VARIABLES,
    )
except ImportError:  # Supports direct execution from the script directory.
    from config import (
        EXCLUDED_CORPORA,
        GROUP_MAP,
        RAW_DIR,
        SOURCE_COLUMNS,
        TRANSCRIPT_VARIABLES,
        UTTERANCE_VARIABLES,
    )


def clean_id(value):
    """Normalize an identifier and convert missing values to ``pd.NA``.

    Numeric-looking values such as ``12.0`` become ``"12"``. Other values
    are returned as stripped strings.

    Args:
        value: Identifier value from a workbook or Redivis table.

    Returns:
        A normalized string identifier or ``pd.NA`` when the value is empty.
    """
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return pd.NA
    try:
        number = float(text)
        return str(int(number)) if number.is_integer() else text
    except ValueError:
        return text


def path_key(path, corpus):
    """Create a case-insensitive transcript path used for matching records.

    The key starts at the corpus directory, uses forward slashes, and removes
    ``.cha`` or ``.xml`` extensions.

    Args:
        path: Transcript path from the workbook or API.
        corpus: Corpus name used to find the start of the path.

    Returns:
        A normalized relative path string.
    """
    parts = [part for part in str(path).replace("\\", "/").split("/") if part]
    parts = [re.sub(r"\.(cha|xml)$", "", part, flags=re.I) for part in parts]
    corpus = str(corpus).lower()
    start = next((i for i, part in enumerate(parts) if part.lower() == corpus), 0)
    return "/".join(parts[start:]).lower()


def first_column(frame, names):
    """Return the first requested column that exists in a DataFrame.

    Args:
        frame: DataFrame whose columns should be checked.
        names: Candidate column names in priority order.

    Returns:
        The matching column name, or ``None`` when none are present.
    """
    return next((name for name in names if name in frame.columns), None)


def first_value(*values):
    """Return the first value that is not missing or a null-like string.

    Args:
        *values: Candidate values in priority order.

    Returns:
        The first usable value, or ``pd.NA`` if all candidates are empty.
    """
    for value in values:
        if pd.notna(value) and str(value).strip().lower() not in {"", "nan", "none"}:
            return value
    return pd.NA


def child_id_from_filename(filename, corpus):
    """Derive a fallback child ID from a transcript filename.

    Rescorla filenames ending in ``36``, ``48``, or ``60`` have that age suffix
    removed before the stem is returned.

    Args:
        filename: Transcript filename or path.
        corpus: Corpus name controlling corpus-specific cleanup.

    Returns:
        The filename stem used as a fallback child ID.
    """
    stem = Path(str(filename)).stem
    return re.sub(r"(36|48|60)$", "", stem) if corpus == "Rescorla" else stem


def fallback_age(filename, corpus):
    """Extract MacWhinney age from a ``YYMMDD`` filename prefix.

    Args:
        filename: Transcript filename or path.
        corpus: Corpus name; parsing is only applied to MacWhinney.

    Returns:
        Age in months as a float, or ``numpy.nan`` when it cannot be derived.
    """
    if corpus != "MacWhinney":
        return np.nan
    match = re.match(r"^(\d{2})(\d{2})(\d{2})", Path(str(filename)).stem)
    if not match:
        return np.nan
    years, months, days = map(int, match.groups())
    return years * 12 + months + days / 30.44


def load_manifest(path):
    """Load and normalize the transcript-selection Excel workbook.

    Both the original positional workbook layout and a workbook with named
    columns are supported. Rows without a recognized group or child ID are
    removed, and duplicate normalized transcript paths are collapsed.

    Args:
        path: Path to the Excel workbook.

    Returns:
        A DataFrame containing the selected, normalized transcript records.

    Raises:
        ValueError: If required manifest columns are missing.
    """
    raw = pd.read_excel(path, header=None, dtype=object)
    first_row = {str(value).strip().lower() for value in raw.iloc[0].dropna()}

    if "transcript_path" in first_row:
        frame = pd.read_excel(path, dtype=object).rename(columns={
            "new_id": "manifest_new_id",
            "child_id": "manifest_new_id",
            "group": "source_group",
        })
    else:
        frame = raw.iloc[:, :len(SOURCE_COLUMNS)].copy()
        frame.columns = SOURCE_COLUMNS

    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    frame = frame[SOURCE_COLUMNS].copy()
    frame["transcript_path"] = frame["transcript_path"].astype("string").str.strip().str.replace("\\", "/", regex=False)
    frame["manifest_new_id"] = frame["manifest_new_id"].map(clean_id).astype("string")
    frame["source_group"] = frame["source_group"].astype("string").str.upper().str.strip()
    frame["corpus"] = frame["transcript_path"].str.split("/").str[2]
    frame["group"] = frame["source_group"].map(GROUP_MAP)
    frame["label"] = frame["group"].map({"TD": 0, "LT": 1}).astype("Int64")
    frame["file_stem"] = (
        frame["transcript_path"].str.rsplit("/", n=1).str[-1]
        .str.replace(r"\.(cha|xml)$", "", regex=True, case=False).str.lower()
    )
    frame["manifest_match_key"] = frame.apply(
        lambda row: path_key(row["transcript_path"], row["corpus"]), axis=1
    )
    return frame[
        frame["group"].notna()
        & ~frame["corpus"].isin(EXCLUDED_CORPORA)
        & frame["manifest_new_id"].notna()
    ].drop_duplicates("manifest_match_key").reset_index(drop=True)


def load_tables(refresh=False):
    """Load cached CHILDES tables or download them from Redivis.

    The selected transcript and utterance columns are cached as CSV files in
    the configured raw-data directory. Cached files are reused unless
    ``refresh`` is true.

    Args:
        refresh: If true, fetch fresh Redivis data instead of using cached CSVs.

    Returns:
        A ``(transcripts, utterances)`` tuple of normalized DataFrames.

    Raises:
        RuntimeError: If fresh data are needed but the Redivis package is not
            installed.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    transcript_csv = RAW_DIR / "redivis_transcripts.csv"
    utterance_csv = RAW_DIR / "redivis_utterances.csv"

    if not refresh and transcript_csv.exists() and utterance_csv.exists():
        transcripts = pd.read_csv(transcript_csv)
        utterances = pd.read_csv(utterance_csv)
    else:
        try:
            import redivis
        except ImportError as error:
            raise RuntimeError(
                "Redivis data are not cached. Install redivis and log in when "
                "the Redivis client prompts for authentication."
            ) from error

        childes = redivis.organization("datapages").dataset("childes-db")
        transcripts = childes.table("transcript").to_pandas_dataframe(variables=TRANSCRIPT_VARIABLES)
        utterances = childes.table("utterance").to_pandas_dataframe(variables=UTTERANCE_VARIABLES)
        transcripts.to_csv(transcript_csv, index=False)
        utterances.to_csv(utterance_csv, index=False)

    transcripts = transcripts.rename(columns={"id": "transcript_id"})
    utterances = utterances.rename(columns={"id": "utterance_id"})
    transcripts["transcript_id"] = transcripts["transcript_id"].map(clean_id).astype("string")
    utterances["transcript_id"] = utterances["transcript_id"].map(clean_id).astype("string")
    return transcripts, utterances


def transcript_metadata(api, selected, corpus):
    """Match selected workbook transcripts to Redivis transcript metadata.

    Matching is attempted by normalized path, persistent ID, and filename
    stem, in that order. The function also fills missing child IDs, ages, and
    sexes using filename- or ID-based fallbacks.

    Args:
        api: Redivis transcript metadata for one corpus.
        selected: Normalized workbook rows for the same corpus.
        corpus: Corpus name being processed.

    Returns:
        A DataFrame containing one metadata record per matched transcript.

    Raises:
        ValueError: If the API data lack a transcript ID or filename column.
    """
    filename_col = first_column(api, ["filename", "file_path", "path"])
    if filename_col is None or "transcript_id" not in api.columns:
        raise ValueError(f"{corpus}: transcript metadata lacks filename or transcript_id")

    child_id_col = first_column(api, ["target_child_id", "child_id", "participant_id"])
    name_col = first_column(api, ["target_child_name", "child_name", "participant_name", "name"])
    age_col = first_column(api, ["target_child_age", "age_months", "target_child_age_months", "age"])
    sex_col = first_column(api, ["target_child_sex", "sex", "child_sex"])
    pid_col = first_column(api, ["pid", "transcript_pid", "persistent_id"])

    meta = pd.DataFrame(index=api.index)
    meta["transcript_id"] = api["transcript_id"].map(clean_id).astype("string")
    meta["api_filename"] = api[filename_col].astype("string").str.replace("\\", "/", regex=False)
    meta["match_key"] = meta["api_filename"].map(lambda value: path_key(value, corpus))
    meta["file_stem"] = meta["api_filename"].str.rsplit("/", n=1).str[-1].str.replace(
        r"\.(cha|xml)$", "", regex=True, case=False
    ).str.lower()
    meta["api_pid"] = api[pid_col].astype("string").str.strip() if pid_col else pd.Series(pd.NA, index=api.index, dtype="string")
    meta["api_child_id"] = api[child_id_col].map(clean_id).astype("string") if child_id_col else pd.Series(pd.NA, index=api.index, dtype="string")
    meta["target_child_name"] = api[name_col].astype("string").str.strip() if name_col else pd.Series("unknown", index=api.index)
    meta["age_months"] = pd.to_numeric(api[age_col], errors="coerce") if age_col else np.nan
    meta["target_child_sex"] = api[sex_col].astype("string").str.lower().str.strip() if sex_col else pd.Series(pd.NA, index=api.index, dtype="string")

    def unique_lookup(key):
        """Build a lookup only for keys that identify one transcript."""
        valid = meta[meta[key].notna()]
        valid = valid[~valid[key].duplicated(keep=False)]
        return valid.set_index(key).to_dict("index")

    lookups = {key: unique_lookup(key) for key in ["match_key", "api_pid", "file_stem"]}
    rows = []
    unmatched = 0

    for _, selected_row in selected.iterrows():
        record = None
        for lookup, key in [
            (lookups["match_key"], selected_row["manifest_match_key"]),
            (lookups["api_pid"], str(selected_row["pid"]).strip()),
            (lookups["file_stem"], selected_row["file_stem"]),
        ]:
            if pd.notna(key) and str(key) in lookup:
                record = lookup[str(key)]
                break
        if record is None:
            unmatched += 1
            continue

        new_id = first_value(
            selected_row["manifest_new_id"], record["api_child_id"],
            record["target_child_name"], child_id_from_filename(record["api_filename"], corpus),
        )
        digits = "".join(character for character in str(new_id) if character.isdigit())
        fallback_sex = {"1": "female", "2": "male"}.get(digits[1], "unknown") if len(digits) > 1 else "unknown"

        rows.append({
            "corpus": corpus, "transcript_id": record["transcript_id"], "filename": record["api_filename"],
            "manifest_transcript_path": selected_row["transcript_path"], "manifest_new_id": selected_row["manifest_new_id"],
            "api_target_child_id": record["api_child_id"], "new_id": new_id,
            "target_child_name": first_value(record["target_child_name"], "unknown"),
            "source_group": selected_row["source_group"], "group": selected_row["group"], "label": selected_row["label"],
            "age_months": first_value(record["age_months"], fallback_age(record["api_filename"], corpus)),
            "target_child_sex": first_value(record["target_child_sex"], fallback_sex),
            "media_type": selected_row["media_type"], "design": selected_row["design"], "activity": selected_row["activity"],
        })

    print(f"{corpus}: matched {len(rows):,}/{len(selected):,}; unmatched {unmatched:,}")
    return pd.DataFrame(rows)


def normalize_utterances(raw, metadata, corpus):
    """Filter, normalize, and enrich child utterances.

    The function keeps child-speaker rows when speaker information is
    available, joins transcript metadata, and chooses morpheme counts or
    token counts as the MLU measure when sufficiently populated. Otherwise it
    estimates MLU units by counting cleaned words.

    Args:
        raw: Redivis utterance rows for the selected transcripts.
        metadata: Output of :func:`transcript_metadata`.
        corpus: Corpus name used in validation errors.

    Returns:
        A DataFrame of cleaned utterances with metadata and ``mlu_units``.

    Raises:
        ValueError: If transcript ID or utterance-text columns are unavailable.
    """
    transcript_col = first_column(raw, ["transcript_id", "transcript_file"])
    text_col = first_column(raw, ["gloss", "utterance", "text", "transcript", "cleaned_utterance"])
    speaker_col = first_column(raw, ["speaker_code", "speaker", "role", "speaker_role"])
    if transcript_col is None or text_col is None:
        raise ValueError(f"{corpus}: utterance data lacks transcript ID or text")

    frame = raw.copy()
    if speaker_col:
        speakers = frame[speaker_col].astype(str).str.lower()
        child_mask = speakers.eq("chi")
        if not child_mask.any():
            child_mask = speakers.str.contains("target|child", na=False)
        if child_mask.any():
            frame = frame[child_mask].copy()

    id_col = first_column(frame, ["utterance_id", "id"])
    out = pd.DataFrame({
        "transcript_id": frame[transcript_col].map(clean_id).astype("string"),
        "utterance_id": frame[id_col] if id_col else np.arange(len(frame)) + 1,
        "utterance_text": frame[text_col].astype(str).str.strip(),
        "speaker_code": frame[speaker_col].astype(str) if speaker_col else "unknown",
    })
    for column in ["num_morphemes", "num_tokens"]:
        out[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else np.nan

    metadata_columns = ["corpus", "filename", "manifest_transcript_path", "manifest_new_id", "api_target_child_id", "new_id", "target_child_name", "source_group", "group", "label", "age_months", "target_child_sex", "media_type", "design", "activity"]
    out = out.merge(metadata[["transcript_id"] + metadata_columns], on="transcript_id", how="inner", validate="many_to_one")
    out = out[out["utterance_text"].ne("") & out["utterance_text"].ne("nan")].copy()

    for column in ["num_morphemes", "num_tokens"]:
        values = pd.to_numeric(out[column], errors="coerce")
        if values.notna().mean() >= 0.5 and (values > 0).mean() >= 0.5:
            out["mlu_units"] = values
            break
    else:
        cleaned = out["utterance_text"].str.lower().str.replace(r"\[.*?\]|<.*?>|&[=\w:.-]+", " ", regex=True)
        cleaned = cleaned.str.replace(r"[^\w\s\'-]", " ", regex=True).str.split().str.len()
        out["mlu_units"] = cleaned

    return out[out["mlu_units"].gt(0)].copy()
