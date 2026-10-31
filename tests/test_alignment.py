"""Alignment screening at CHAT extraction and manifest boundaries."""

import unittest
import csv
from pathlib import Path
import tempfile

from src.media_pipeline.chat import alignment_reason, read_utterances
from src.media_pipeline.create_manifest import create_manifest, FIELDS


class AlignmentTests(unittest.TestCase):
    def test_plausibility_reasons(self):
        self.assertEqual(alignment_reason(0, 0.1, "hello"), "point_bullet_timestamp")
        self.assertEqual(alignment_reason(0, 0.4, "hello"), "clip_too_short")
        self.assertEqual(alignment_reason(0, 1, "one two three four five"),
                         "impossible_speech_rate")
        self.assertEqual(alignment_reason(0, 6, "yes"), "long_clip_short_reference")
        self.assertEqual(alignment_reason(0, 1, "hello there"), "")

    def test_manifest_logs_alignment_drops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "all_utterances_clean_mlu_balanced.csv"
            base = dict.fromkeys(FIELDS + ["age_months"], "")
            base.update(corpus="X", filepath="child.cha", child_id="X::child",
                        transcript_id="t1", group="TD", source_group="TD",
                        speaker_code="CHI", utterance="hello", age_months="48")
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS + ["age_months"])
                writer.writeheader()
                writer.writerow(dict(base, utterance_id="abc_0_100_1",
                                     start_seconds="0", end_seconds="0.1"))
                writer.writerow(dict(base, utterance_id="abc_100_1100_2",
                                     start_seconds="0.1", end_seconds="1.1"))
            output, log, summary = (root / name for name in
                                    ("manifest.csv", "manifest.log", "summary.txt"))
            _, count, skipped = create_manifest(root / "workbook.xlsx", root,
                                                output, log, summary)
            self.assertEqual((count, skipped), (1, 1))
            with output.open(newline="", encoding="utf-8") as handle:
                self.assertEqual([row["utterance_id"] for row in csv.DictReader(handle)],
                                 ["abc_100_1100_2"])
            self.assertIn("point bullet timestamp", log.read_text(encoding="utf-8"))
            self.assertIn("point_bullet_timestamp", summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
