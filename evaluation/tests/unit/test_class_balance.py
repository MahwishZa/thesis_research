"""Per-verdict behaviour and macro-F1 (no model): the code, the committed file and the documents that quote it."""

import json
import unittest
from pathlib import Path

import numpy as np

from experiments.medchange import class_balance as C

ROOT = Path(__file__).resolve().parents[3]
REP = json.loads((ROOT / "experiments" / "medchange" / "results" / "class_balance.json").read_text(encoding="utf-8"))


class MetricTests(unittest.TestCase):
    def test_macro_f1_is_the_mean_of_three_f1_and_zero_for_a_verdict_never_found(self):
        gold = np.array([0, 0, 1, 1, 2, 2])
        self.assertEqual(C.macro_f1(gold, gold), 1.0)
        always_first = np.zeros(6, dtype=int)
        self.assertAlmostEqual(C.macro_f1(gold, always_first), (2 * 2 / (2 * 2 + 4 + 0)) / 3)     # F1 of class 0 = 0.5; others 0

    def test_recall_of_a_verdict_and_the_paired_interval_contains_the_point(self):
        gold = np.array([1, 1, 1, 1, 0, 0, 2, 2])
        a, b = np.array([1, 1, 1, 0, 0, 0, 2, 2]), np.array([0, 0, 1, 0, 0, 0, 2, 2])
        self.assertEqual(C.recall(gold, a, 1), 0.75)
        r = C.paired_interval(gold, a, b, lambda g, p: C.recall(g, p, 1), seed="t", iterations=500)
        self.assertEqual(r["difference"], 0.5)
        self.assertLessEqual(r["ci95"][0], 0.5 <= r["ci95"][1])


class CommittedFileTests(unittest.TestCase):
    def test_sizes_and_the_pairs_are_those_of_the_two_splits(self):
        self.assertEqual((REP["splits"]["confirm"]["n"], REP["splits"]["ad"]["n"]), (528, 208))
        for sp in REP["splits"].values():
            self.assertEqual(set(sp["paired"]), {"R2V vs R2", "R2C vs R2", "R2V vs R2C"})

    def test_the_gains_quoted_in_the_protocol_are_those_of_the_file(self):
        text = (ROOT / "docs" / "protocol.md").read_text(encoding="utf-8")
        for sp in REP["splits"].values():
            d = sp["paired"]["R2V vs R2"]["macro_f1"]
            self.assertIn(f"+{d['difference']:.3f}", text)

    def test_every_verdict_share_sums_to_one(self):
        for sp in REP["splits"].values():
            for v in sp["arms"].values():
                self.assertAlmostEqual(sum(v["predicted_share"].values()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
