"""Regression checks for participant identity, whole-recording sampling and export."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.preprocessing_pipeline.selection import child_identity, cap_longitudinal, collect_utterances, validate_selection
from src.preprocessing_pipeline.helpers import path_key
from src.preprocessing_pipeline.build_manifest import candidate_table, balance_candidates
from src.media_pipeline.chat import read_header, read_utterances
from src.media_pipeline.create_manifest import create_manifest, FIELDS
from src.media_pipeline.download_media import download_media, clip_name
from src.media_pipeline.validation import validate_export


def fixture():
    rows = []
    for child, design, sizes in [("X::a", "cross-sectional", [4]), ("X::b", "cross-sectional", [4]),
                                 ("X::c", "longitudinal", [3, 2, 2]), ("X::d", "longitudinal", [8])]:
        for transcript, size in enumerate(sizes):
            for turn in range(size):
                rows.append(dict(child_id=child, corpus="X", design=design, group="TD", label=0,
                                 source_group="TD", transcript_id=f"{child}-{transcript}",
                                 filepath=f"{child}/{transcript}.cha", utterance_id=f"{child}-{transcript}-{turn}",
                                 mlu_units=2, age_months=60, target_child_name="unknown", target_child_sex="unknown"))
    return pd.DataFrame(rows)


class SelectionTests(unittest.TestCase):
    def test_filename_scope_and_longitudinal_identity(self):
        self.assertEqual(child_identity("EHS", "24-mot/0020.cha")[0], "EHS::0020")
        self.assertEqual(child_identity("HSLLD", "HV1/BR/rasbr1.cha")[0], child_identity("HSLLD", "HV3/MT/rasmt3.cha")[0])
        self.assertEqual(child_identity("Rescorla", "LT/60/ly160.cha")[0], "Rescorla::ly1")
        self.assertEqual(child_identity("Rescorla", "LT/108/ly1108.cha")[0], "Rescorla::ly1")
        self.assertNotEqual(child_identity("Rescorla", "TD/60/gr160.cha")[0], child_identity("Rescorla", "TD/108/gr2108.cha")[0])
        self.assertNotEqual(child_identity("Gelman", "1998-Books/12.cha")[0], child_identity("Gelman", "2016-Robodog/12.cha")[0])
        self.assertEqual(child_identity("Forrester", "realfood.cha")[1], "longitudinal")

    def test_ambiguous_cases_fail_closed(self):
        for corpus, filename in [("Sprott", "13FireEngine.cha"), ("ENNI", "TD/B/523.cha"),
                                 ("HSLLD", "HV1/bob2br1.cha"), ("NewCorpus", "001.cha")]:
            with self.subTest(corpus=corpus), self.assertRaises(ValueError):
                child_identity(corpus, filename)

    def test_only_explicit_target_child_is_used(self):
        header = "@ID:\teng|X|AM|5;01.|female|||Child|||\n"
        with self.assertRaises(ValueError):
            read_header(header)
        target = read_header("@ID:\teng|X|CHI|5;06.15|female|||Target_Child|||\n")
        self.assertAlmostEqual(target["age_months"], 66 + 15 / 30.4368)

    def test_conflicting_header_identity_excludes_all_affected_recordings(self):
        temp_root = Path(__file__).resolve().parents[1] / "data"
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            self.assertTrue(root.resolve().is_relative_to(temp_root.resolve()))
            rows = []
            for corpus, relative, sex in [("EHS", "pre-K-mot/0020.cha", "male"),
                                          ("EHS", "pre-K-fat/0020.cha", "female"),
                                          ("ENNI", "TD/A/529.cha", "male")]:
                path = root / corpus / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"@ID:\teng|{corpus}|CHI|5;00.|{sex}|||Target_Child|||\n*CHI:\thello . \x151000_2000\x15\n", encoding="utf-8")
                name = path.relative_to(root).as_posix()
                rows.append(dict(corpus=corpus, transcript_path=name, manifest_match_key=path_key(name, corpus),
                                 pid=name, design="long", group="TD", source_group="TD", label=0))
            master = pd.DataFrame(rows)
            def metadata(api, selected, corpus):
                return pd.DataFrame([dict(manifest_transcript_path=r.transcript_path, transcript_id=r.pid,
                                          target_child_name="unknown") for r in selected.itertuples()])
            with patch("src.preprocessing_pipeline.selection.transcript_metadata", side_effect=metadata):
                data, mapping, _ = collect_utterances(master, pd.DataFrame(columns=["corpus_name"]), root)
            self.assertEqual(set(data.child_id), {"ENNI::529"})
            excluded = mapping[mapping.corpus.eq("EHS")]
            self.assertTrue(excluded.identity_status.eq("excluded").all())
            self.assertTrue(excluded.exclusion_reason.eq("Conflicting sex").all())

    def test_same_utterances_feed_counts_and_audio(self):
        text = "*CHI:\thello [/] there . \x151000_2000\x15\n*INV:\tno . \x152000_3000\x15\n*CHI:\txxx . \x153000_4000\x15\n*CHI:\tno time .\n*CHI:\tbad . \x155000_4000\x15\n*CHI:\tbye . \x156000_7000\x15\n"
        rows, skipped = read_utterances(text, "X", "transcripts/X/a.cha", "CHI")
        self.assertEqual([r["utterance"] for r in rows], ["hello there", "bye"])
        self.assertEqual([r["clip_ordinal"] for r in rows], [1, 4])
        self.assertEqual(len(skipped), 3)

    def test_continuation_lines_preserve_audit_locations(self):
        text = "*CHI:\twe\n\tgo . \x151000_2000\x15\n*CHI:\tuntimed .\n"
        rows, skipped = read_utterances(text, "X", "transcripts/X/a.cha", "CHI")
        self.assertEqual(rows[0]["utterance"], "we go")
        self.assertEqual(skipped[0]["line"], 3)

    def test_cap_is_per_child_and_deterministic(self):
        data = fixture()
        kept, decisions, stats = cap_longitudinal(data, deviations=0)
        shuffled, _, shuffled_stats = cap_longitudinal(data.sample(frac=1, random_state=9), deviations=0)
        self.assertEqual(stats["cap"], 4)
        self.assertEqual(stats["children"], 2)
        self.assertEqual(stats, shuffled_stats)
        self.assertEqual(set(kept["utterance_id"]), set(shuffled["utterance_id"]))
        self.assertLessEqual(len(kept[kept["child_id"].eq("X::c")]), 4)
        self.assertEqual(len(kept[kept["design"].eq("cross-sectional")]), 8)
        self.assertFalse(kept["child_id"].eq("X::d").any())
        self.assertIn("transcript_exceeds_cap", set(decisions["reason"]))
        for filepath, group in data.groupby("filepath"):
            actual = kept[kept["filepath"].eq(filepath)]
            self.assertIn(len(actual), [0, len(group)])

    def test_candidate_mlu_pools_utterances(self):
        data = fixture()
        mask = data["child_id"].eq("X::c")
        data.loc[mask, "mlu_units"] = [2, 2, 2, 4, 4, 6, 6]
        candidates = candidate_table(data)
        child = candidates[candidates["child_id"].eq("X::c")].iloc[0]
        self.assertEqual(child["number_of_transcripts"], 3)
        self.assertAlmostEqual(child["mlu"], 26 / 7)

    def test_longitudinal_children_at_or_below_cap_keep_all_recordings(self):
        data = fixture()
        extra = data[data["child_id"].eq("X::c")].copy()
        extra["child_id"] = "X::below"
        extra["filepath"] = "below/" + extra["filepath"]
        extra["transcript_id"] = "below/" + extra["transcript_id"]
        extra["utterance_id"] = "below/" + extra["utterance_id"]
        for size in [3, 4]:
            with self.subTest(utterances=size):
                child = extra.iloc[:size]
                kept, _, stats = cap_longitudinal(pd.concat([data, child], ignore_index=True), deviations=0)
                self.assertEqual(stats["cap"], 4)
                self.assertEqual(set(kept.loc[kept.child_id.eq("X::below"), "utterance_id"]), set(child.utterance_id))

    def test_validation_rejects_incorrect_child_counts_even_when_total_matches(self):
        data = fixture().assign(identity_status="resolved", start_seconds=0, end_seconds=1, utterance="hello")
        candidates = candidate_table(data)
        self.assertTrue(all(validate_selection(data, candidates, {"cap": 8}).values()))
        candidates.loc[candidates.child_id.eq("X::a"), "total_child_utterances"] = 7
        candidates.loc[candidates.child_id.eq("X::c"), "total_child_utterances"] = 4
        with self.assertRaisesRegex(ValueError, "per_child_counts_reconcile"):
            validate_selection(data, candidates, {"cap": 8})

    def test_manifest_exports_exact_selected_rows(self):
        temp_root = Path(__file__).resolve().parents[1] / "data"
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            self.assertTrue(root.resolve().is_relative_to(temp_root.resolve()))
            source = root / "all_utterances_clean_mlu_balanced.csv"
            row = dict.fromkeys(FIELDS, "x")
            row.update(start_seconds="1", end_seconds="2", age_months="60", group="TD", source_group="TD", child_id="x::001")
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            count = create_manifest(root / "unused.xlsx", root / "transcripts", root / "manifest.csv", root / "log.txt", root / "summary.txt")
            self.assertEqual(count[:2], (1, 1))
            with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
                result = list(csv.DictReader(handle))
            self.assertEqual(result[0]["child_id"], "x::001")
            self.assertEqual(result[0]["utterance_id"], "x")

    def test_balancing_removes_children_and_preserves_lt(self):
        rows = [dict(child_id=f"X::{i}", corpus="X", group="TD", mlu=float(i % 5)) for i in range(40)]
        rows += [dict(child_id=f"X::lt{i}", corpus="X", group="LT", mlu=5 + i / 10) for i in range(10)]
        original = pd.DataFrame(rows)
        kept, log, summary = balance_candidates(original)
        self.assertEqual(set(kept[kept.group.eq("LT")].child_id), set(original[original.group.eq("LT")].child_id))
        self.assertTrue(set(log.child_id).isdisjoint(set(kept.child_id)))
        self.assertEqual(len(original) - len(kept), len(log))
        self.assertLessEqual(summary.iloc[-1].abs_mean_diff, summary.iloc[0].abs_mean_diff)

    def test_media_reuses_old_folder_and_preserves_canonical_id(self):
        temp_root = Path(__file__).resolve().parents[1] / "data"
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            self.assertTrue(root.resolve().is_relative_to(temp_root.resolve()))
            media = root / "media"
            old = media / "X" / "TD" / "old_id"
            old.mkdir(parents=True)
            filepath = "transcripts/X/001.cha"
            name = clip_name("X", filepath, 1, 2, 1)
            (old / name).write_bytes(b"existing-encoded-audio")
            row = dict.fromkeys(FIELDS, "")
            row.update(corpus="X", filepath=filepath, start_seconds=1, end_seconds=2,
                       child_id="X::001", transcript_id="source-123", group="TD", design="cross-sectional",
                       utterance_id=Path(name).stem, clip_ordinal=1, raw_utterance="hello .", utterance="hello")
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(row)
            with patch("src.media_pipeline.download_media.load_collection_map", return_value={"X": "Eng-NA"}), patch("src.media_pipeline.download_media.browser_login") as login:
                result = download_media(manifest, root / "unused.xlsx", root / "transcripts", media, reuse_only=True)
                login.assert_not_called()
            self.assertEqual(result, 0)
            with (media / "media_download_report.csv").open(encoding="utf-8", newline="") as handle:
                report = list(csv.DictReader(handle))
            self.assertEqual(report[0]["child_id"], "X::001")
            self.assertEqual(report[0]["transcript_id"], "source-123")
            self.assertEqual(report[0]["reference_status"], "ready")
            self.assertTrue((old / name).exists())
            (root / "utterance_cap.json").write_text('{"cap": 4}', encoding="utf-8")
            self.assertTrue(validate_export(manifest, media))
            self.assertIn("Missing utterances: 0", (root / "run_summary.md").read_text())
            with manifest.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writerow(row)
            self.assertFalse(validate_export(manifest, media))
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(row)
            report[0]["child_id"] = "X::wrong-child"
            with (media / "media_download_report.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(report[0]))
                writer.writeheader()
                writer.writerows(report)
            self.assertFalse(validate_export(manifest, media))


if __name__ == "__main__":
    unittest.main()
