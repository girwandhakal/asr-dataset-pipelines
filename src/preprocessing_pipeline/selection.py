"""Resolve participants, cap longitudinal contributions and audit selections."""

import hashlib
import json
import math
from pathlib import PurePosixPath
import random
import re

import pandas as pd

from .config import EXCLUDED_CORPORA
from .helpers import path_key, transcript_metadata
from ..media_pipeline.chat import read_header, read_utterances

LONGITUDINAL = {"Braunwald", "EHS", "EllisWeismer", "Forrester", "HSLLD", "Rescorla", "Weist", "MacWhinney"}
FILENAME_CORPORA = {"ENNI", "OCSC", "POLER", "Nippold", "Gleason"}


def child_identity(corpus, relative):
    parts = PurePosixPath(relative.replace("\\", "/")).parts
    stem = PurePosixPath(relative).stem.lower()
    if corpus in {"MacWhinney", "Sprott"}:
        raise ValueError("Unresolved child speakers across recordings")
    if corpus in {"Braunwald", "Forrester"}:
        key = {"Braunwald": "laura", "Forrester": "ella"}[corpus]
    elif corpus == "Weist":
        if len(parts) < 2 or parts[0] not in {"Ben", "Emily", "Emma", "Jillian", "Matt", "Roman"}:
            raise ValueError("Unknown Weist child directory")
        key = parts[0].lower()
    elif corpus == "HSLLD":
        match = re.fullmatch(r"([a-z0-9]{3})[a-z]{2}[1-7][a-z]?", stem)
        if not match:
            raise ValueError("Irregular HSLLD filename needs an explicit participant mapping")
        key = match[1]
    elif corpus == "Rescorla":
        age = next((p for p in reversed(parts[:-1]) if p in {"36", "48", "60", "108", "156"}), "")
        if not age or not stem.endswith(age):
            raise ValueError("Rescorla filename has no unambiguous age suffix")
        key = stem[:-len(age)]
        if not re.fullmatch(r"[a-z]{2,3}[12]?", key):
            raise ValueError("Unrecognized Rescorla participant prefix")
    elif corpus in {"EHS", "EllisWeismer"}:
        width = 4 if corpus == "EHS" else 5
        if not re.fullmatch(rf"\d{{{width}}}", stem):
            raise ValueError(f"Expected a {width}-digit participant code")
        key = stem
    elif corpus == "Gelman":
        if len(parts) < 2 or parts[0] not in {"1998-Books", "2004-Gender", "2016-Robodog"}:
            raise ValueError("Gelman substudy needs a validated participant rule")
        key = f"{parts[0].lower()}::{stem}"
    elif corpus in FILENAME_CORPORA:
        if corpus == "ENNI" and stem == "523":
            raise ValueError("ENNI 523 is reused across TD and SLI; identity unresolved")
        key = stem
    else:
        raise ValueError("No validated participant rule for this corpus")
    design = "longitudinal" if corpus in LONGITUDINAL else "cross-sectional"
    return f"{corpus}::{key}", design


def collect_utterances(master, api, transcripts_dir):
    """Return eligible CHAT utterances plus a decision for every master row."""
    files = {path_key(str(p.relative_to(transcripts_dir)), p.relative_to(transcripts_dir).parts[0]): p
             for p in transcripts_dir.rglob("*.cha")}
    matches = []
    for corpus, selected in master.groupby("corpus", sort=True):
        if corpus not in EXCLUDED_CORPORA:
            matches.append(transcript_metadata(api[api["corpus_name"].eq(corpus)], selected, corpus))
    metadata = pd.concat(matches, ignore_index=True).set_index("manifest_transcript_path").to_dict("index")
    decisions, utterances, skipped = [], [], []
    for row in master.to_dict("records"):
        corpus = row["corpus"]
        decision = dict(row, child_id="", identity_status="excluded", exclusion_reason="",
                        transcript_id=str(row["pid"]), eligible_utterances=0, sex="", birthdate="")
        decisions.append(decision)
        if corpus in EXCLUDED_CORPORA:
            decision["exclusion_reason"] = "MacWhinney excluded: longitudinal participant identities unresolved"
            continue
        if pd.isna(row["group"]):
            decision["exclusion_reason"] = "Unsupported source group"
            continue
        path = files.get(row["manifest_match_key"])
        try:
            if path is None:
                raise ValueError("CHAT transcript not found")
            relative = path.relative_to(transcripts_dir / corpus).as_posix()
            child, design = child_identity(corpus, relative)
            content = path.read_bytes()
            text = content.decode("utf-8-sig", errors="replace")
            decision["chat_sha256"] = hashlib.sha256(content).hexdigest()
            header = read_header(text)
            source = metadata.get(row["transcript_path"], {})
            filepath = path.relative_to(transcripts_dir.parent).as_posix()
            decision.update(child_id=child, design=design, source_design=row["design"],
                            filename=path.relative_to(transcripts_dir).as_posix(), filepath=filepath,
                            transcript_id=str(source.get("transcript_id", row["pid"])),
                            age_months=header["age_months"], sex=header["sex"],
                            birthdate=header["birthdate"], speaker_code=header["code"],
                            recording_date=header["recording_date"])
            rows, rejected = read_utterances(text, corpus, filepath, header["code"])
            skipped.extend(rejected)
            decision["eligible_utterances"] = len(rows)
            if not rows:
                raise ValueError("No usable timestamped target-child utterances")
            decision["identity_status"] = "resolved"
            for utterance in rows:
                utterances.append({**source, **decision, **utterance,
                                   "new_id": child, "target_child_sex": header["sex"] or "unknown",
                                   "target_child_name": source.get("target_child_name", "unknown")})
        except ValueError as error:
            decision["exclusion_reason"] = str(error)
    mapping = pd.DataFrame(decisions)
    # Matching birthdates never merge different codes; twins may share a birthdate.
    for child, group in mapping[mapping["identity_status"].eq("resolved")].groupby("child_id"):
        conflicts = [column for column in ["sex", "birthdate", "group", "design"]
                     if group[column].replace("", pd.NA).dropna().nunique() > 1]
        if conflicts:
            mask = mapping["child_id"].eq(child)
            mapping.loc[mask, "identity_status"] = "excluded"
            mapping.loc[mask, "exclusion_reason"] = "Conflicting " + ", ".join(conflicts)
    data = pd.DataFrame(utterances)
    allowed = set(mapping.loc[mapping["identity_status"].eq("resolved"), "child_id"])
    data = data[data["child_id"].isin(allowed)].copy()
    if data.empty:
        raise ValueError("No resolved participants have usable CHAT utterances")
    return data, mapping, pd.DataFrame(skipped, columns=["corpus", "filepath", "line", "reason"])


def cap_longitudinal(utterances, center="mean", deviations=3, seed=42):
    counts = utterances.groupby(["child_id", "design"]).size().rename("utterances").reset_index()
    reference = counts.loc[counts["design"].eq("cross-sectional"), "utterances"]
    if len(reference) < 2 or center not in {"mean", "median"} or not math.isfinite(deviations) or deviations < 0:
        raise ValueError("The cap needs at least two cross-sectional children and valid settings")
    stats = dict(children=len(reference), mean=float(reference.mean()), median=float(reference.median()),
                 standard_deviation=float(reference.std(ddof=1)), minimum=int(reference.min()),
                 maximum=int(reference.max()), center=center, standard_deviations=deviations,
                 seed=seed, sd_ddof=1, mlu_basis="words", eligible_pool="timestamped CHAT with nonempty reference")
    cap = math.floor(stats[center] + deviations * stats["standard_deviation"])
    stats["cap"] = cap
    stats["formula"] = f"floor({center} + {deviations} * sample_standard_deviation)"
    transcripts = utterances.groupby(["child_id", "corpus", "design", "transcript_id", "filepath"], sort=True).size().rename("utterances").reset_index()
    transcripts["retained"] = True
    transcripts["reason"] = "within_cap_or_cross_sectional"
    for child, rows in transcripts[transcripts["design"].eq("longitudinal")].groupby("child_id", sort=True):
        if rows["utterances"].sum() <= cap:
            continue
        indices = sorted(rows.index, key=lambda i: transcripts.at[i, "filepath"])
        # A per-child seed makes selection independent of input row order and other children.
        child_seed = int.from_bytes(hashlib.sha256(f"{seed}:{child}".encode()).digest()[:8], "big")
        random.Random(child_seed).shuffle(indices)
        used = 0
        for index in indices:
            size = int(transcripts.at[index, "utterances"])
            keep = used + size <= cap
            transcripts.at[index, "retained"] = keep
            transcripts.at[index, "reason"] = "sampled_within_cap" if keep else "transcript_exceeds_cap" if size > cap else "child_cap"
            if keep:
                used += size
    keep = set(transcripts.loc[transcripts["retained"], "filepath"])
    return utterances[utterances["filepath"].isin(keep)].copy(), transcripts, stats


def summarize(utterances, stage):
    return utterances.groupby(["corpus", "group", "design"], sort=True).agg(
        children=("child_id", "nunique"), transcripts=("transcript_id", "nunique"),
        utterances=("utterance_id", "size")
    ).reset_index().assign(stage=stage)


def write_diagnostics(output, stages, mapping, decisions, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = [summarize(data, name) for name, data in stages.items()]
    pd.concat(summaries, ignore_index=True).to_csv(output / "dataset_stage_summary.csv", index=False)
    mapping.to_csv(output / "participant_transcript_mapping.csv", index=False)
    decisions.to_csv(output / "longitudinal_selection.csv", index=False)
    (output / "utterance_cap.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    distributions = []
    fig, axes = plt.subplots(1, len(stages), figsize=(6 * len(stages), 4.5), sharey=True, squeeze=False)
    maximum = max(data.groupby("child_id").size().max() for data in stages.values())
    bins = [i * maximum / 30 for i in range(31)]
    for ax, (stage, data) in zip(axes.flat, stages.items()):
        counts = data.groupby(["child_id", "corpus", "group", "design"]).size().rename("utterances").reset_index()
        distributions.append(counts.assign(stage=stage))
        for (design, group), subset in counts.groupby(["design", "group"]):
            ax.hist(subset["utterances"], bins=bins, alpha=0.45, label=f"{design} {group}")
        ax.axvline(stats["cap"], color="black", linestyle="--", label="longitudinal cap")
        ax.set(title=stage.replace("_", " "), xlabel="Utterances per child", ylabel="Children")
        ax.legend(fontsize=7)
    for ax in axes.flat[len(stages):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "child_utterance_distribution.png", dpi=160)
    plt.close(fig)
    pd.concat(distributions, ignore_index=True).to_csv(output / "child_utterance_counts.csv", index=False)
    final = list(stages.values())[-1]
    transcript_counts = final.groupby(["corpus", "group", "design", "child_id", "transcript_id"]).size().rename("utterances").reset_index()
    transcript_counts.to_csv(output / "transcript_utterance_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    for (design, group), subset in transcript_counts.groupby(["design", "group"]):
        ax.hist(subset["utterances"], bins=40, alpha=0.45, label=f"{design} {group}")
    ax.set(xlabel="Utterances per transcript", ylabel="Transcripts", title="Final retained transcripts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "transcript_utterance_distribution.png", dpi=160)
    plt.close(fig)


def validate_selection(utterances, candidates, stats):
    checks = {
        "nonempty_child_ids": utterances["child_id"].fillna("").ne("").all(),
        "only_resolved_identities": utterances["identity_status"].eq("resolved").all(),
        "unique_utterance_ids": not utterances["utterance_id"].duplicated().any(),
        "one_child_per_transcript": utterances.groupby("filepath")["child_id"].nunique().eq(1).all(),
        "corpus_namespaced_ids": all(r.child_id.startswith(r.corpus + "::") for r in candidates.itertuples()),
        "one_candidate_per_child": not candidates["child_id"].duplicated().any(),
        "one_group_per_child": utterances.groupby("child_id")["group"].nunique().eq(1).all(),
        "macwhinney_excluded": not utterances["corpus"].isin(EXCLUDED_CORPORA).any(),
        "valid_intervals": (utterances["start_seconds"].ge(0) & utterances["end_seconds"].gt(utterances["start_seconds"])).all(),
        "nonempty_references": utterances["utterance"].str.strip().ne("").all(),
        "candidate_counts_reconcile": int(candidates["total_child_utterances"].sum()) == len(utterances),
        "longitudinal_cap_respected": utterances[utterances["design"].eq("longitudinal")].groupby("child_id").size().le(stats["cap"]).all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"Dataset validation failed: {checks}")
    return checks
