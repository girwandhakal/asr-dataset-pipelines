"""Check that the actual audio handoff matches the final selected dataset."""

from collections import Counter
import csv
import json


def validate_export(manifest_path, media_dir):
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        selected = {row["utterance_id"]: row for row in csv.DictReader(handle)}
    with (media_dir / "media_download_report.csv").open(encoding="utf-8-sig", newline="") as handle:
        report = list(csv.DictReader(handle))
    ready = [row for row in report if row["reference_status"] == "ready"]
    same = ["child_id", "transcript_id", "corpus", "group", "design", "filepath", "raw_utterance", "utterance"]
    identity_matches = all(row["utterance_id"] in selected and
                           all(row[key] == selected[row["utterance_id"]][key] for key in same) and
                           all(float(row[key]) == float(selected[row["utterance_id"]][key])
                               for key in ["start_seconds", "end_seconds"]) for row in report)
    cap_path = manifest_path.parent / "utterance_cap.json"
    cap = json.loads(cap_path.read_text())["cap"]
    counts = Counter(row["child_id"] for row in ready if row["design"] == "longitudinal")
    # Audio existence and filename/interval agreement are checked by the report finalizer.
    checks = dict(report_matches_manifest=identity_matches,
                  unique_utterance_ids=len({r["utterance_id"] for r in report}) == len(report),
                  all_selected_audio_ready=len(ready) == len(selected) == len(report),
                  all_ready_audio_paths=all(row["audio_path"] for row in ready),
                  longitudinal_cap_respected=all(value <= cap for value in counts.values()),
                  no_macwhinney=all(row["corpus"] != "MacWhinney" for row in report))
    result = dict(checks=checks, selected_utterances=len(selected), ready_utterances=len(ready),
                  ready_children=len({row["child_id"] for row in ready}),
                  ready_transcripts=len({row["transcript_id"] for row in ready}),
                  missing_utterances=len(selected) - len(ready), passed=all(checks.values()))
    destination = manifest_path.parent / "media_validation.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Audio handoff: {len(ready):,}/{len(selected):,} ready; validation {'passed' if result['passed'] else 'incomplete'}")
    return result["passed"]
