"""Accuracy intervals, per-class recall, paired comparisons and Holm families (no model, no network)."""

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import scoring as SC

S, R, N = "SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"


def world(n_changed=40, n_unchanged=20, a_better=0, b_better=0):
    """Arms A and B answer every item; A alone is right on ``a_better`` items, B alone on ``b_better`` items."""
    items, answers = {}, {}
    for k in range(n_changed + n_unchanged):
        iid = f"MC-{k:03d}"
        kind = "changed" if k < n_changed else "unchanged"
        items[iid] = {"item_id": iid, "kind": kind, "newest": {"label": S}, "previous": {"label": R}}
        a_right, b_right = (k < a_better), (a_better <= k < a_better + b_better)
        both = k >= a_better + b_better and k % 2 == 0
        answers[(iid, "A")] = {"verdict": S if (a_right or both) else R}
        answers[(iid, "B")] = {"verdict": S if (b_right or both) else R}
    return items, answers


class IntervalTests(unittest.TestCase):

    def test_wilson_interval(self):
        lo, hi = SC.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertTrue(math.isnan(SC.wilson(0, 0)[0]))


class SummaryTests(unittest.TestCase):

    def test_class_statistics(self):
        items = {"a": {"kind": "changed", "newest": {"label": S}}, "b": {"kind": "changed", "newest": {"label": S}}}
        answers = {("a", "X"): {"verdict": S}, ("b", "X"): {"verdict": R}}
        stats = SC.class_stats(items, answers, "X", ("changed",))
        self.assertEqual(stats["recall"][S], 0.5)
        self.assertIsNone(stats["recall"][R])
        self.assertEqual(stats["predicted_share"][R], 0.5)

    def test_summary_counts_accuracy_unparsed_answers_and_the_outdated_rate(self):
        items, answers = world(4, 2)
        answers[("MC-000", "A")]["verdict"] = None
        row = SC.summarize(items, answers, ["A"])["A"]
        self.assertEqual(row["all"]["n"], 6)
        self.assertEqual(row["unparsed"], 1)
        self.assertEqual(row["changed"]["n"], 4)
        self.assertEqual(row["outdated_verdict_rate_changed"], 0.5)      # R is the previous label; 2 of 4 changed items

    def test_load_answers_reads_the_answers_file_only(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "answers_dev.jsonl").write_text(json.dumps({"item_id": "x", "arm": "B1", "verdict": S}) + "\n", encoding="utf-8")
            (d / "synthesis_dev.jsonl").write_text(json.dumps({"item_id": "x", "arm": "H0", "verdict": S}) + "\n", encoding="utf-8")
            self.assertEqual(set(SC.load_answers(d, "dev")), {("x", "B1")})


class PairedTests(unittest.TestCase):

    def test_a_clear_gain_is_confirmed_in_a_family_and_identical_arms_are_not(self):
        items, answers = world(a_better=30)
        gain = SC.paired(items, answers, "A", "B", ("changed", "unchanged"))
        self.assertEqual((gain["a_only_correct"], gain["b_only_correct"]), (30, 0))
        same = SC.paired(items, answers, "B", "B", ("changed", "unchanged"))
        self.assertEqual((same["a_only_correct"], same["b_only_correct"]), (0, 0))
        family = SC.holm_family({"gain": gain, "same": same, "missing": None})
        self.assertTrue(family["gain"]["confirmed"])
        self.assertFalse(family["same"]["confirmed"])
        self.assertNotIn("missing", family)
        self.assertLess(family["gain"]["holm_p"], 0.001)

    def test_a_positive_but_insignificant_difference_is_not_confirmed(self):
        items, answers = world(a_better=6, b_better=4)
        result = SC.paired(items, answers, "A", "B", ("changed", "unchanged"))
        self.assertGreater(result["diff_a_minus_b"], 0)
        self.assertFalse(SC.holm_family({"x": result})["x"]["confirmed"])

    def test_paired_needs_a_minimum_number_of_pairs(self):
        items, answers = world(2, 1)
        self.assertIsNone(SC.paired(items, answers, "A", "B", ("changed", "unchanged")))

    def test_stable_difference_is_computed_on_the_stable_items_only(self):
        items = {"a": {"kind": "changed", "newest": {"label": S}}, "b": {"kind": "changed", "newest": {"label": S}}}
        ans = {("a", "X"): {"verdict": S}, ("a", "Y"): {"verdict": R},
               ("b", "X"): {"verdict": R}, ("b", "Y"): {"verdict": S}}
        self.assertEqual(SC.stable_difference(items, ans, "X", "Y", ["a"]), {"n": 1, "diff_a_minus_b": 1.0})
        self.assertIsNone(SC.stable_difference(items, ans, "X", "Y", []))


if __name__ == "__main__":
    unittest.main()
