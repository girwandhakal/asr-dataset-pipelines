"""Functions for building and balancing the Child ASR dataset manifest.

The command-line entry point lives in :mod:`main`. This module contains the
data-processing functions used by that runner.
"""

import pandas as pd

try:
    from .config import MASTER_MANIFEST, OUTPUT_DIR
    from .helpers import (
        load_manifest,
        load_tables,
        normalize_utterances,
        transcript_metadata,
    )
except ImportError:  # Supports: python src/preprocessing_pipeline/build_manifest.py
    from config import MASTER_MANIFEST, OUTPUT_DIR
    from helpers import (
        load_manifest,
        load_tables,
        normalize_utterances,
        transcript_metadata,
    )


def candidate_table(utterances):
    """Summarize cleaned utterances into one row per child.

    The summary includes each child's group, age, average MLU, transcript
    count, and utterance count. These rows are the input to the balancing
    step.

    Args:
        utterances: Cleaned utterance DataFrame from ``normalize_utterances``.

    Returns:
        A DataFrame containing one candidate row per child.
    """
    keys = ["corpus", "new_id", "source_group", "group", "label"]
    per_transcript = utterances.groupby(keys + ["transcript_id"]).size().reset_index(name="count")
    average = per_transcript.groupby(keys)["count"].mean().reset_index(name="avg_utterances_per_transcript")
    children = utterances.groupby(keys).agg(
        target_child_name=("target_child_name", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
        target_child_sex=("target_child_sex", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
        mean_age_months=("age_months", "mean"), mlu=("mlu_units", "mean"),
        number_of_transcripts=("transcript_id", "nunique"), total_child_utterances=("utterance_id", "count"),
    ).reset_index()
    result = children.merge(average, on=keys)
    result["mlu"] = result["mlu"].round(3)
    result["mean_age_months"] = result["mean_age_months"].round(2)
    result["avg_utterances_per_transcript"] = result["avg_utterances_per_transcript"].round(2)
    return result.sort_values(["corpus", "group", "new_id"]).reset_index(drop=True)


def balance_candidates(candidates, alpha=0.05, min_group_size=20, max_removals=1000):
    """Reduce the LT/TD MLU difference by removing TD candidates.

    At each iteration, the candidate with the most extreme TD MLU is tested
    for removal. Removal stops when the groups are no longer significantly
    different, the next removal would not improve the MLU difference, the TD
    group reaches its minimum size, or the removal limit is reached.

    Args:
        candidates: Candidate DataFrame with ``group`` and ``mlu`` columns.
        alpha: P-value threshold used as the stopping condition.
        min_group_size: Smallest allowed number of TD candidates.
        max_removals: Maximum number of candidates that may be removed.

    Returns:
        A tuple containing the balanced candidates, removal log, and summary.
    """
    from scipy.stats import ttest_ind

    def stats(frame):
        """Calculate group sizes, MLU statistics, and Welch's t-test."""
        values = {group: frame.loc[frame["group"].eq(group), "mlu"].dropna().to_numpy() for group in ["LT", "TD"]}
        t_stat, p_value = ttest_ind(values["LT"], values["TD"], equal_var=False)
        return {"lt_n": len(values["LT"]), "td_n": len(values["TD"]), "lt_mean_mlu": values["LT"].mean(), "td_mean_mlu": values["TD"].mean(), "lt_sd_mlu": values["LT"].std(ddof=1), "td_sd_mlu": values["TD"].std(ddof=1), "abs_mean_diff": abs(values["LT"].mean() - values["TD"].mean()), "t_stat": t_stat, "p_value": p_value}

    current = candidates.reset_index(drop=True).copy()
    removals = []
    for iteration in range(1, max_removals + 1):
        before = stats(current)
        if before["p_value"] >= alpha:
            stop_reason = f"p-value >= {alpha}"; break
        if before["td_n"] <= min_group_size:
            stop_reason = f"TD would fall below {min_group_size} candidates"; break
        td = current[current["group"].eq("TD")]
        direction = "highest" if before["td_mean_mlu"] > before["lt_mean_mlu"] else "lowest"
        index = td["mlu"].idxmax() if direction == "highest" else td["mlu"].idxmin()
        trial = current.drop(index)
        after = stats(trial)
        if after["abs_mean_diff"] >= before["abs_mean_diff"]:
            stop_reason = "next TD removal would not reduce the MLU difference"; break
        removed = current.loc[index]
        removals.append({"iteration": iteration, "removed_corpus": removed["corpus"], "removed_new_id": removed["new_id"], "removed_group": removed["group"], "removed_mlu": removed["mlu"], "removal_logic": f"remove {direction}-MLU TD candidate", **{f"{key}_before": value for key, value in before.items()}, **{f"{key}_after": value for key, value in after.items()}})
        current = trial
    else:
        stop_reason = f"MAX_REMOVALS={max_removals} reached"

    final = stats(current)
    summary = pd.DataFrame([
        {"stage": "before_balancing", **stats(candidates), "removed_total": 0, "stop_reason": ""},
        {"stage": "after_balancing", **final, "removed_total": len(removals), "stop_reason": stop_reason},
    ])
    return current, pd.DataFrame(removals), summary


def build_manifest(refresh: bool = False) -> None:
    """Run the complete manifest-building workflow.

    Args:
        refresh: If true, download fresh Redivis tables instead of using the
            cached CSV files.

    Raises:
        FileNotFoundError: If the selection workbook is missing.
        RuntimeError: If no corpus produces usable utterances.
    """
    print("=" * 72)
    print("Child ASR manifest build started")
    print(f"Selection workbook: {MASTER_MANIFEST}")
    print(f"Output directory:   {OUTPUT_DIR}")
    print(f"Refresh Redivis data: {refresh}")
    print("=" * 72)

    # Load the selected transcripts and the source data.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MASTER_MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MASTER_MANIFEST}")

    manifest = load_manifest(MASTER_MANIFEST)
    print("\n[1/5] Selection workbook loaded")
    print(f"  Selected transcripts: {len(manifest):,}")
    print(f"  Corpora: {manifest['corpus'].nunique():,}")
    print(f"  Groups: {manifest['group'].value_counts().to_dict()}")

    transcripts, utterances = load_tables(refresh)
    print("\n[2/5] CHILDES tables loaded")
    print(f"  Transcript rows: {len(transcripts):,}")
    print(f"  Utterance rows: {len(utterances):,}")

    # Process each corpus independently so one problematic corpus is reported
    # and skipped without stopping the other corpora.
    print("\n[3/5] Processing corpora")
    all_utterances = []
    all_candidates = []
    for corpus in sorted(manifest["corpus"].dropna().unique()):
        selected = manifest[manifest["corpus"].eq(corpus)]
        print(f"  {corpus}: {len(selected):,} selected transcript(s)")
        try:
            api = transcripts[transcripts["corpus_name"].eq(corpus)]
            metadata = transcript_metadata(api, selected, corpus)
            if metadata.empty:
                continue
            raw = utterances[utterances["transcript_id"].isin(metadata["transcript_id"])]
            normalized = normalize_utterances(raw, metadata, corpus)
            all_utterances.append(normalized)
            all_candidates.append(candidate_table(normalized))
            print(
                f"  {corpus}: {len(metadata):,} matched transcript(s), "
                f"{len(normalized):,} utterance(s), {len(all_candidates[-1]):,} candidate(s)"
            )
        except Exception as error:
            print(f"  {corpus}: skipped because {type(error).__name__}: {error}")

    if not all_utterances:
        raise RuntimeError("No corpus produced usable utterances.")

    all_utterances = pd.concat(all_utterances, ignore_index=True)
    all_candidates = pd.concat(all_candidates, ignore_index=True)
    print(f"  Combined utterances: {len(all_utterances):,}")
    print(f"  Combined candidates: {len(all_candidates):,}")
    # MacWhinney filenames include ages; keep only candidates aged 5–9 years.
    mac = all_candidates["corpus"].eq("MacWhinney")
    age_years = all_candidates["mean_age_months"] / 12
    keep = (~mac | age_years.between(5, 9, inclusive="both")).fillna(False)
    all_candidates = all_candidates[keep].reset_index(drop=True)
    print("\n[4/5] Applying age filter and writing intermediate tables")
    print(f"  MacWhinney candidates removed: {(~keep & mac).sum():,}")
    print(f"  Candidates retained: {len(all_candidates):,}")
    allowed = all_candidates[["corpus", "new_id", "group"]].drop_duplicates()
    all_utterances = all_utterances.merge(allowed, on=["corpus", "new_id", "group"], how="inner")

    # Save the unbalanced intermediate tables.
    all_utterances.to_csv(OUTPUT_DIR / "all_utterances_clean.csv", index=False)
    all_candidates.to_csv(OUTPUT_DIR / "all_candidate_table.csv", index=False)

    # Balance the candidate groups, then keep only utterances belonging to the
    # retained candidates.
    print("\n[5/5] Balancing LT and TD candidates")
    balanced_candidates, removal_log, balance_summary = balance_candidates(all_candidates)
    balanced_keys = balanced_candidates[["corpus", "new_id", "group"]].drop_duplicates()
    balanced_utterances = all_utterances.merge(balanced_keys, on=["corpus", "new_id", "group"], how="inner")
    balanced_candidates.to_csv(OUTPUT_DIR / "all_candidate_table_mlu_balanced.csv", index=False)
    balanced_utterances.to_csv(OUTPUT_DIR / "all_utterances_clean_mlu_balanced.csv", index=False)
    removal_log.to_csv(OUTPUT_DIR / "mlu_balance_removal_log.csv", index=False)
    balance_summary.to_csv(OUTPUT_DIR / "mlu_balance_summary.csv", index=False)

    # Put the final tables and balancing diagnostics in one workbook.
    with pd.ExcelWriter(OUTPUT_DIR / "mlu_balancing_results.xlsx") as writer:
        for name, frame in {
            "Balanced Candidates": balanced_candidates,
            "Balanced Utterances": balanced_utterances,
            "Removal Log": removal_log,
            "Balance Summary": balance_summary,
        }.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    print(f"  Balanced candidates: {len(balanced_candidates):,}")
    print(f"  Balanced utterances: {len(balanced_utterances):,}")
    print(f"  Removed TD candidates: {len(removal_log):,}")
    print(f"  Workbook: {OUTPUT_DIR / 'mlu_balancing_results.xlsx'}")
    print("\nBuild complete.")


if __name__ == "__main__":
    # Keep the old direct command working while using the shared runner.
    try:
        from .main import _parse_args, run_preprocessing_pipeline
    except ImportError:  # Supports: python src/preprocessing_pipeline/build_manifest.py
        from main import _parse_args, run_preprocessing_pipeline

    run_preprocessing_pipeline(refresh_data=_parse_args().refresh_data)
