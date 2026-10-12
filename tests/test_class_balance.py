"""Per-verdict behaviour and macro-F1 (no model): the code, the committed file and the documents that quote it."""

import json
import unittest
from pathlib import Path

import numpy as np

from experiments.medchange import class_balance as C

ROOT = Path(__file__).resolve().parents[1]
REP = json.loads((ROOT / "experiments" / "medchange" / "results" / "class_balance.json").read_text(encoding="utf-8"))


class MetricTests(unittest.TestCase):
    def test_macro_f1_is_the_mean_of_three_f1_and_zero_for_a_verdict_never_found(self):
        gold = np.array([0, 0, 1, 1, 2, 2])
        self.assertEqual(C.macro_f1(gold, gold), 1.0)
        always_first = np.zeros(6, dtype=int)
        self.assertAlmostEqual(C.macro_f1(gold, always_first), (2 * 2 / (2 * 2 + 4 + 0)) / 3)     # F1 of class 0 = 0.5; others 0

    def test_macro_f1_equals_the_hand_computed_value_when_there_are_false_positives_and_false_negatives(self):
        # gold S S R R N N, predicted S R R R N S (codes 0 0 1 1 2 2 and 0 1 1 1 2 0):
        # S: tp 1, predicted 2, gold 2 -> F1 0.5; R: tp 2, predicted 3, gold 2 -> precision 2/3, recall 1 -> F1 0.8; N: tp 1, predicted 1, gold 2 -> F1 2/3
        gold, pred = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 2, 0])
        self.assertAlmostEqual(C.macro_f1(gold, pred), (0.5 + 0.8 + 2 / 3) / 3)
        self.assertEqual(C.macro_f1(gold, np.array([3, 3, 3, 3, 3, 3])), 0.0)                 # only unparsed answers: nothing is ever right

    def test_recall_of_a_verdict_and_the_paired_interval_contains_the_point(self):
        gold = np.array([1, 1, 1, 1, 0, 0, 2, 2])
        a, b = np.array([1, 1, 1, 0, 0, 0, 2, 2]), np.array([0, 0, 1, 0, 0, 0, 2, 2])
        self.assertEqual(C.recall(gold, a, 1), 0.75)
        r = C.paired_interval(gold, a, b, lambda g, p: C.recall(g, p, 1), seed="t", iterations=500)
        self.assertEqual(r["difference"], 0.5)
        self.assertTrue(r["ci95"][0] <= 0.5 <= r["ci95"][1])


class UnparsedAnswerTests(unittest.TestCase):
    def test_an_unparsed_answer_is_a_fourth_code_that_is_never_right(self):
        items = {f"Q{k}": {"newest": {"label": lab}} for k, lab in enumerate(("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"))}
        answers = {("Q0", "R2"): "SUPPORTED", ("Q1", "R2"): None, ("Q2", "R2"): "NOT ENOUGH INFORMATION"}
        gold, vec = C.vectors(items, answers, ["R2"])
        self.assertEqual(list(vec["R2"]), [0, 3, 2])
        self.assertEqual(C.recall(gold, vec["R2"], 1), 0.0)
        self.assertAlmostEqual(C.macro_f1(gold, vec["R2"]), 2 / 3)           # the unparsed REFUTED answer counts as a miss, not as a crash


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
