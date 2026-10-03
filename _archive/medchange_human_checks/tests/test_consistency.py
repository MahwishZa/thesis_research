"""Tests of the retired human consistency check (kept so the archived code stays verifiable)."""

import unittest

from _archive.medchange_human_checks.consistency import sample_rows, score_sheet


class ConsistencyTests(unittest.TestCase):

    def test_sample_is_seeded_and_capped(self):
        rows = [{"item_id": f"i{k}", "arm": "B1"} for k in range(100)]
        a, b = sample_rows(rows, 50, 7), sample_rows(rows, 50, 7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 50)
        self.assertNotEqual(a, sample_rows(rows, 50, 8))

    def test_score_requires_every_row_marked_and_applies_thresholds(self):
        sheet = [{"consistent": "Y"}] * 9 + [{"consistent": "N"}]
        self.assertEqual(score_sheet(sheet, 0.97)["G1"], "PASS")
        self.assertEqual(score_sheet(sheet, 0.90)["G1"], "FAIL")           # parse rate too low
        self.assertEqual(score_sheet([{"consistent": "Y"}] * 8 + [{"consistent": "N"}] * 2, 1.0)["G1"], "FAIL")
        with self.assertRaises(ValueError):
            score_sheet([{"consistent": "Y"}, {"consistent": ""}], 1.0)
