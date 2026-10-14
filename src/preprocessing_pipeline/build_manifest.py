"""Build a child-level, capped and MLU-balanced dataset from selected CHAT files."""

import json
import hashlib
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import ttest_ind

from .config import (MASTER_MANIFEST, OUTPUT_DIR, TRANSCRIPTS_DIR, RAW_DIR, CAP_CENTER,
                     CAP_STANDARD_DEVIATIONS, SAMPLING_SEED)
from .helpers import load_manifest, load_transcripts
from .selection import (collect_utterances, cap_longitudinal, write_diagnostics,
                        validate_selection)
from ..media_pipeline.extract_transcripts import extract_archives


def candidate_table(utterances):
    """Pool eligible utterances across recordings to obtain one MLU-w per child."""
    keys = ["corpus", "child_id", "group", "label", "design"]
    children = utterances.groupby(keys, sort=True).agg(
        source_group=("source_group", lambda s: "/".join(sorted(set(s)))),
        target_child_name=("target_child_name", "first"),
        target_child_sex=("target_child_sex", "first"),
        mean_age_months=("age_months", "mean"), mlu=("mlu_units", "mean"),
        number_of_transcripts=("transcript_id", "nunique"),
        total_child_utterances=("utterance_id", "size"),
    ).reset_index()
    children["avg_utterances_per_transcript"] = children["total_child_utterances"] / children["number_of_transcripts"]
    children["mlu_basis"] = "words"
    # Keep new_id as a compatibility alias; source IDs remain on utterance/mapping rows.
    children["new_id"] = children["child_id"]
    return children


def balance_candidates(candidates, alpha=0.05, min_group_size=20, max_removals=1000):
    """Reduce the group MLU-w difference by removing whole TD children."""
    def stats(frame):
        values = {g: frame.loc[frame["group"].eq(g), "mlu"].dropna() for g in ["LT", "TD"]}
        if any(len(v) < 2 for v in values.values()):
            raise ValueError("MLU balancing requires at least two resolved children in each group")
        result = ttest_ind(values["LT"], values["TD"], equal_var=False)
        return dict(lt_n=len(values["LT"]), td_n=len(values["TD"]),
                    lt_mean_mlu=float(values["LT"].mean()), td_mean_mlu=float(values["TD"].mean()),
                    lt_sd_mlu=float(values["LT"].std()), td_sd_mlu=float(values["TD"].std()),
                    abs_mean_diff=float(abs(values["LT"].mean() - values["TD"].mean())),
                    t_stat=float(result.statistic), p_value=float(result.pvalue))

    current = candidates.copy()
    removals = []
    stop_reason = "removal_limit"
    for iteration in range(1, max_removals + 1):
        before = stats(current)
        if before["p_value"] >= alpha:
            stop_reason = f"p-value >= {alpha}"
            break
        if before["td_n"] <= min_group_size:
            stop_reason = f"minimum TD children: {min_group_size}"
            break
        td = current[current["group"].eq("TD")]
        higher = before["td_mean_mlu"] > before["lt_mean_mlu"]
        index = td["mlu"].idxmax() if higher else td["mlu"].idxmin()
        trial = current.drop(index)
        after = stats(trial)
        if after["abs_mean_diff"] >= before["abs_mean_diff"]:
            stop_reason = "next removal would not improve the mean difference"
            break
        removed = current.loc[index]
        removals.append(dict(iteration=iteration, child_id=removed["child_id"],
                             removed_corpus=removed["corpus"], removed_new_id=removed["child_id"],
                             removed_group="TD", removed_mlu=removed["mlu"],
                             **{f"{k}_before": v for k, v in before.items()},
                             **{f"{k}_after": v for k, v in after.items()}))
        current = trial
    summary = pd.DataFrame([
        dict(stage="before_balancing", **stats(candidates), removed_total=0, stop_reason=""),
        dict(stage="after_balancing", **stats(current), removed_total=len(removals), stop_reason=stop_reason),
    ])
    columns = list(removals[0]) if removals else ["iteration", "child_id", "removed_corpus", "removed_new_id", "removed_group", "removed_mlu"]
    return current.reset_index(drop=True), pd.DataFrame(removals, columns=columns), summary


def build_manifest(refresh=False, cap_center=CAP_CENTER, cap_sd=CAP_STANDARD_DEVIATIONS, seed=SAMPLING_SEED):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extract_archives(TRANSCRIPTS_DIR)
    master = load_manifest(MASTER_MANIFEST, include_excluded=True)
    api = load_transcripts(refresh)
    inputs = [MASTER_MANIFEST, RAW_DIR / "redivis_transcripts.csv", *sorted(TRANSCRIPTS_DIR.glob("*.zip"))]
    hashes = {}
    for path in inputs:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path.relative_to(MASTER_MANIFEST.parent))] = digest.hexdigest()
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(dict(
        created_utc=datetime.now(timezone.utc).isoformat(), refresh_data=refresh,
        cap_center=cap_center, cap_sd=cap_sd, seed=seed, mlu_basis="words", sha256=hashes,
    ), indent=2), encoding="utf-8")
    utterances, mapping, skipped = collect_utterances(master, api, TRANSCRIPTS_DIR)
    print(f"Resolved {utterances['child_id'].nunique()} children from {len(master)} master records")
    before = candidate_table(utterances)
    utterances.to_csv(OUTPUT_DIR / "all_utterances_clean.csv", index=False)
    before.to_csv(OUTPUT_DIR / "all_candidate_table.csv", index=False)
    skipped.to_csv(OUTPUT_DIR / "utterance_exclusions.csv", index=False)

    capped, decisions, cap_stats = cap_longitudinal(utterances, cap_center, cap_sd, seed)
    capped_candidates = candidate_table(capped)
    print(f"Longitudinal cap: {cap_stats['cap']} utterances; retained {len(capped):,} utterances")
    balanced, removal_log, balance_summary = balance_candidates(capped_candidates)
    final = capped[capped["child_id"].isin(balanced["child_id"])].copy()
    final = final.sort_values(["corpus", "filepath", "clip_ordinal"]).reset_index(drop=True)
    balanced.to_csv(OUTPUT_DIR / "all_candidate_table_mlu_balanced.csv", index=False)
    final.to_csv(OUTPUT_DIR / "all_utterances_clean_mlu_balanced.csv", index=False)
    removal_log.to_csv(OUTPUT_DIR / "mlu_balance_removal_log.csv", index=False)
    balance_summary.to_csv(OUTPUT_DIR / "mlu_balance_summary.csv", index=False)

    decisions["exported"] = decisions["filepath"].isin(final["filepath"])
    decisions["final_reason"] = decisions.apply(
        lambda r: r["reason"] if not r["retained"] else "retained" if r["exported"] else "mlu_balance", axis=1)
    mapping["retained"] = mapping["transcript_path"].isin(final["transcript_path"])
    initial = master.groupby(["corpus", "group"], dropna=False).size().rename("transcripts").reset_index()
    initial.to_csv(OUTPUT_DIR / "input_transcript_counts.csv", index=False)
    stages = dict(after_identity_and_exclusions=utterances, after_longitudinal_cap=capped,
                  after_mlu_balance=final)
    write_diagnostics(OUTPUT_DIR, stages, mapping, decisions, cap_stats)
    checks = validate_selection(final, balanced, cap_stats)
    (OUTPUT_DIR / "dataset_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    # Keep the workbook compact; full source metadata remains in the CSV exports.
    excel_columns = ["utterance_id", "child_id", "corpus", "group", "source_group", "design",
                     "transcript_id", "filename", "filepath", "speaker_code", "age_months",
                     "start_seconds", "end_seconds", "raw_utterance", "utterance"]
    with pd.ExcelWriter(OUTPUT_DIR / "mlu_balancing_results.xlsx", engine="xlsxwriter") as writer:
        for name, frame in {"Balanced Candidates": balanced, "Balanced Utterances": final[excel_columns],
                            "Removal Log": removal_log, "Balance Summary": balance_summary,
                            "Transcript Selection": decisions}.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    print(balance_summary.to_string(index=False))
    print(f"Final selection: {len(balanced)} children, {final['filepath'].nunique()} transcripts, {len(final):,} utterances")
