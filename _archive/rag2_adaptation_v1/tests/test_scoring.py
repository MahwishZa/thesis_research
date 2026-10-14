"""Accuracy intervals, per-class recall and macro-F1, paired comparisons, bootstrap intervals and Holm families (no model, no network)."""

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

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

    def test_macro_f1_is_the_mean_of_the_three_f1_values(self):
        # gold S S R R N N, answered S R R R N S: F1 of S 0.5, of R 0.8 (precision 2/3, recall 1), of N 2/3
        gold = [S, S, R, R, N, N]
        answered = [S, R, R, R, N, S]
        items = {f"q{k}": {"kind": "changed", "newest": {"label": g}} for k, g in enumerate(gold)}
        answers = {(f"q{k}", "X"): {"verdict": v} for k, v in enumerate(answered)}
        stats = SC.class_stats(items, answers, "X", ("changed",))
        self.assertAlmostEqual(stats["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3, places=5)

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

    def test_a_result_is_confirmed_only_when_p_interval_and_sign_all_agree(self):
        base = {"n": 100, "a_only_correct": 20, "b_only_correct": 5}
        significant_but_interval_reaches_zero = dict(base, mcnemar_p=0.001, diff_a_minus_b=0.05, ci95=[0.0, 0.1])
        significant_but_negative = dict(base, mcnemar_p=0.001, diff_a_minus_b=-0.05, ci95=[-0.1, -0.01])
        clean = dict(base, mcnemar_p=0.001, diff_a_minus_b=0.05, ci95=[0.01, 0.1])
        family = SC.holm_family({"zero": significant_but_interval_reaches_zero, "neg": significant_but_negative, "clean": clean})
        self.assertEqual({k: v["confirmed"] for k, v in family.items()}, {"zero": False, "neg": False, "clean": True})

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


#: The first raw 64-bit output of the generators seeded as the committed intervals were (numpy keeps the stream of its bit generator).
PREFIX_RAW = 5729317375224220984
HASHED_RAW = 18042913583985041560


class VectorMetricTests(unittest.TestCase):
    def test_macro_f1_is_the_mean_of_three_f1_and_zero_for_a_verdict_never_found(self):
        gold = np.array([0, 0, 1, 1, 2, 2])
        self.assertEqual(SC.macro_f1(gold, gold), 1.0)
        always_first = np.zeros(6, dtype=int)
        self.assertAlmostEqual(SC.macro_f1(gold, always_first), (2 * 2 / (2 * 2 + 4 + 0)) / 3)     # F1 of class 0 = 0.5; others 0

    def test_macro_f1_equals_the_hand_computed_value_when_there_are_false_positives_and_false_negatives(self):
        # gold S S R R N N, predicted S R R R N S (codes 0 0 1 1 2 2 and 0 1 1 1 2 0):
        # S: tp 1, predicted 2, gold 2 -> F1 0.5; R: tp 2, predicted 3, gold 2 -> precision 2/3, recall 1 -> F1 0.8; N: tp 1, predicted 1, gold 2 -> F1 2/3
        gold, pred = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 2, 0])
        self.assertAlmostEqual(SC.macro_f1(gold, pred), (0.5 + 0.8 + 2 / 3) / 3)
        self.assertEqual(SC.macro_f1(gold, np.array([3, 3, 3, 3, 3, 3])), 0.0)                 # only unparsed answers: nothing is ever right

    def test_an_unparsed_answer_is_a_fourth_code_that_is_never_right(self):
        items = {f"Q{k}": {"newest": {"label": lab}} for k, lab in enumerate((S, R, N))}
        answers = {("Q0", "R2"): S, ("Q1", "R2"): None, ("Q2", "R2"): N}
        gold, vec = SC.vectors(items, answers, ["R2"])
        self.assertEqual(list(vec["R2"]), [0, 3, 2])
        self.assertEqual(SC.recall(gold, vec["R2"], 1), 0.0)
        self.assertAlmostEqual(SC.macro_f1(gold, vec["R2"]), 2 / 3)           # the unparsed REFUTED answer counts as a miss, not as a crash

    def test_an_arm_that_misses_an_answer_gets_no_vector(self):
        items = {"Q0": {"newest": {"label": S}}, "Q1": {"newest": {"label": R}}}
        gold, vec = SC.vectors(items, {("Q0", "A"): S, ("Q1", "A"): R, ("Q0", "B"): S}, ["A", "B"])
        self.assertEqual(list(gold), [0, 1])
        self.assertEqual(set(vec), {"A"})

    def test_the_vector_metric_and_the_dictionary_metric_compute_the_same_numbers(self):
        rng = np.random.default_rng(3)
        verdicts = [S, R, N, None]
        items = {f"q{k:02d}": {"item_id": f"q{k:02d}", "kind": "unchanged", "newest": {"label": (S, R, N)[k % 3]}, "previous": {"label": S}}
                 for k in range(90)}
        answers = {(i, "A"): {"verdict": verdicts[int(rng.integers(0, 4))]} for i in items}
        gold, vec = SC.vectors(items, {key: rec["verdict"] for key, rec in answers.items()}, ["A"])
        stats = SC.class_stats(items, answers, "A", ("unchanged",))
        self.assertAlmostEqual(stats["macro_f1"], SC.macro_f1(gold, vec["A"]), places=5)
        for k, label in enumerate(SC.LABELS):
            self.assertAlmostEqual(stats["recall"][label], SC.recall(gold, vec["A"], k), places=5)


class BootstrapTests(unittest.TestCase):
    def test_recall_of_a_verdict_and_the_paired_interval_contains_the_point(self):
        gold = np.array([1, 1, 1, 1, 0, 0, 2, 2])
        a, b = np.array([1, 1, 1, 0, 0, 0, 2, 2]), np.array([0, 0, 1, 0, 0, 0, 2, 2])
        self.assertEqual(SC.recall(gold, a, 1), 0.75)
        r = SC.paired_interval(gold, a, b, lambda g, p: SC.recall(g, p, 1), seed="t", iterations=500)
        self.assertEqual(r["difference"], 0.5)
        self.assertTrue(r["ci95"][0] <= 0.5 <= r["ci95"][1])

    def test_the_interval_is_the_2_5_to_97_5_percentile_range_of_the_resampled_differences(self):
        calls = iter(range(10 ** 6))

        def metric(g, p):                                  # the first two calls are the point estimate; then resample k returns k - 1 and 0
            index = next(calls)
            return float(index // 2 - 1) if index >= 2 and index % 2 == 0 else 0.0
        gold = np.array([0, 1, 2])
        r = SC.bootstrap_difference(gold, gold, gold, metric, SC.hashed_rng("t"), 1000)
        self.assertEqual(r["difference"], 0.0)
        self.assertAlmostEqual(r["ci95"][0], 0.025 * 999, places=6)           # resampled differences are 0, 1, ..., 999
        self.assertAlmostEqual(r["ci95"][1], 0.975 * 999, places=6)

    def test_both_systems_are_resampled_with_the_same_questions(self):
        accuracy = lambda g, p: float((g == p).mean())
        gold, same = np.array([0, 1, 2, 0, 1, 2, 0, 1]), np.array([0, 1, 1, 0, 2, 2, 1, 1])
        r = SC.bootstrap_difference(gold, same, same.copy(), accuracy, SC.prefix_rng("t"), 200)
        self.assertEqual((r["difference"], r["ci95"]), (0.0, [0.0, 0.0]))     # identical systems never differ, in any resample

    def test_stored_values_keep_six_decimals(self):
        self.assertEqual(SC.DIGITS, 6)
        gold = np.array([0, 1, 2])
        r = SC.bootstrap_difference(gold, np.array([0, 1, 0]), np.array([0, 0, 0]), SC.macro_f1, SC.prefix_rng("t"), 20)
        self.assertEqual(r["difference"], round(r["difference"], 6))
        self.assertNotEqual(r["difference"], round(r["difference"], 4))               # 1/3-type values would be cut by four decimals

    def test_the_two_seeding_rules_are_those_of_the_committed_intervals(self):
        """``class_balance`` and ``ad_subgroup`` seed from the first 8 bytes of the seed text, the pre-registered ``analyze_fresh`` from its SHA-256.
        The committed intervals were made with these streams, so the first raw numbers of each are pinned."""
        self.assertEqual(SC.prefix_rng("class-balance-v1|confirm|R2V|R2|f1").bit_generator.random_raw(), PREFIX_RAW)
        self.assertEqual(SC.prefix_rng("class-balance-v1|ad|R2C|R2|rec").bit_generator.random_raw(), PREFIX_RAW)       # the same first 8 bytes: "class-ba"
        self.assertNotEqual(SC.prefix_rng("ad-subgroup-macro-f1").bit_generator.random_raw(), PREFIX_RAW)
        self.assertEqual(SC.hashed_rng("fresh-macro-f1").bit_generator.random_raw(), HASHED_RAW)
        self.assertNotEqual(SC.hashed_rng("fresh-macro-f1-R2V-R2C").bit_generator.random_raw(), HASHED_RAW)
        self.assertEqual(SC.prefix_rng("short").bit_generator.random_raw(), SC.prefix_rng("short").bit_generator.random_raw())        # a short seed is padded with zeros


if __name__ == "__main__":
    unittest.main()
