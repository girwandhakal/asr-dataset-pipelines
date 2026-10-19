"""Check that the actual audio handoff matches the final selected dataset."""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json


def validate_export(manifest_path, media_dir):
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    selected = {row["utterance_id"]: row for row in manifest_rows}
    report_path = media_dir / "media_download_report.csv"
    with report_path.open(encoding="utf-8-sig", newline="") as handle:
        report = list(csv.DictReader(handle))
    ready = [row for row in report if row["reference_status"] == "ready"]
    same = ["child_id", "transcript_id", "corpus", "group", "design", "filepath", "raw_utterance", "utterance"]
    identity_matches = all(row["utterance_id"] in selected and
                           all(row[key] == selected[row["utterance_id"]][key] for key in same) and
                           all(float(row[key]) == float(selected[row["utterance_id"]][key])
                               for key in ["start_seconds", "end_seconds"]) for row in report)
    cap_path = manifest_path.parent / "utterance_cap.json"
    cap_settings = json.loads(cap_path.read_text())
    cap = cap_settings["cap"]
    counts = Counter(row["child_id"] for row in ready if row["design"] == "longitudinal")
    # Audio existence and filename/interval agreement are checked by the report finalizer.
    checks = dict(report_matches_manifest=identity_matches,
                  unique_manifest_ids=len(selected) == len(manifest_rows),
                  unique_utterance_ids=len({r["utterance_id"] for r in report}) == len(report),
                  all_selected_audio_ready=len(ready) == len(selected) == len(report),
                  all_ready_audio_paths=all(row["audio_path"] for row in ready),
                  longitudinal_cap_respected=all(value <= cap for value in counts.values()),
                  no_macwhinney=all(row["corpus"] != "MacWhinney" for row in report))
    result = dict(checks=checks, selected_utterances=len(selected), ready_utterances=len(ready),
                  ready_children=len({row["child_id"] for row in ready}),
                  ready_transcripts=len({row["transcript_id"] for row in ready}),
                  missing_utterances=len(selected) - len(ready), passed=all(checks.values()))
    result["checked_utc"] = datetime.now(timezone.utc).isoformat()
    result["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    destination = manifest_path.parent / "media_validation.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Regenerate the status note so an earlier missing-audio warning cannot survive a successful rerun.
    summary = ["# Dataset run status", "", f"Checked (UTC): {result['checked_utc']}", "",
               f"Audio validation: {'passed' if result['passed'] else 'incomplete'}.",
               f"Ready: {len(ready):,} of {len(selected):,} selected utterances; "
               f"{result['ready_children']:,} children; {result['ready_transcripts']:,} transcripts.",
               f"Missing utterances: {result['missing_utterances']:,}.", "",
               f"Longitudinal cap: {cap} utterances per child.",
               f"Formula: {cap_settings.get('formula', 'see utterance_cap.json')}.", "",
               "See dataset_stage_summary.csv for stage counts, mlu_balance_removal_log.csv for removed children,",
               "participant_transcript_mapping.csv for exclusions, and README.md for methods.", "",
               "Handoff: data/media/media_download_report.csv and its referenced audio clips.",
               "Child-based training splits are the responsibility of gensec-asr."]
    (manifest_path.parent / "run_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Audio handoff: {len(ready):,}/{len(selected):,} ready; validation {'passed' if result['passed'] else 'incomplete'}")
    return result["passed"]
