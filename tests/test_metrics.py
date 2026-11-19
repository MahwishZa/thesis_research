"""The metrics (src/metrics.py), checked against values worked out by hand."""

import json
import unittest

from src import metrics

METRIC_LABELS = ["a", "b", "c", "d"]  # d never occurs as a true class below
METRIC_TRUE = list("aaaabbbccc")
METRIC_PRED = ["a", "a", "b", None, "b", "b", "c", "c", "c", "d"]  # one invalid answer, and one answer naming the absent class d
# Confusion matrix (rows: true a, b, c, d; columns: answered a, b, c, d, invalid)
#   a: 2 1 0 0 1    b: 0 2 1 0 0    c: 0 0 2 1 0    d: 0 0 0 0 0
# Worked by hand: precision a 2/2, b 2/3, c 2/3; recall a 2/4, b 2/3, c 2/3; F1 a 4/6, b 4/6, c 4/6.
# Balanced accuracy (a, b, c) = (1/2 + 2/3 + 2/3) / 3 = 11/18; macro-F1 = 2/3. Both agree with scikit-learn on this example.

# An answer of the current form: (true class, diagnosis, differential, evidence sufficient)
JSON_ANSWERS = [("a", "a", ["a", "b", "c"], "yes"),   # right, not flagged
                ("a", "b", ["b", "a", "c"], "no"),    # wrong, flagged; the true class is second in the differential
                ("a", None, None, None),              # invalid
                ("b", "b", ["b", "c", "a"], "no"),    # right, flagged
                ("b", "a", ["a", "c", "d"], "yes"),   # wrong, not flagged; the true class is not in the differential
                ("b", "c", ["c", "a", "b"], "no")]    # wrong, flagged; the true class is third


def json_summary(groups=None):
    true, pred, differentials, flags = (list(x) for x in zip(*JSON_ANSWERS))
    return metrics.summarize(true, pred, ["a", "b", "c", "d"], groups, differentials=differentials, flags=flags)


class MetricTests(unittest.TestCase):
    def test_wilson_known_value(self):
        low, high = metrics.wilson(8, 10)
        self.assertAlmostEqual(low, 0.4902, places=3)
        self.assertAlmostEqual(high, 0.9433, places=3)
        self.assertEqual(metrics.wilson(0, 0), (0.0, 0.0))

    def test_the_confusion_matrix_has_a_column_for_invalid_answers(self):
        m = metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        self.assertEqual(m["confusion_matrix"]["columns"], METRIC_LABELS + [metrics.INVALID])
        self.assertEqual(m["confusion_matrix"]["counts"], [[2, 1, 0, 0, 1], [0, 2, 1, 0, 0], [0, 0, 2, 1, 0], [0, 0, 0, 0, 0]])
        self.assertEqual((m["n"], m["correct"], m["invalid"], m["invalid_rate"]), (10, 6, 1, 0.1))

    def test_balanced_accuracy_macro_f1_and_per_class_by_hand(self):
        m = metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        self.assertAlmostEqual(m["balanced_accuracy"], 11 / 18)
        self.assertAlmostEqual(m["macro_f1"], 2 / 3)
        self.assertAlmostEqual(m["macro_precision"], (1 + 2 / 3 + 2 / 3) / 3)
        a, d = m["per_class"]["a"], m["per_class"]["d"]
        self.assertEqual((a["support"], a["answered"], a["tp"], a["precision"], a["recall"]), (4, 2, 2, 1.0, 0.5))
        self.assertAlmostEqual(a["f1"], 4 / 6)
        self.assertEqual(a["recall_ci95"], metrics.wilson(2, 4))  # every answer its own group: Wilson
        self.assertEqual((d["support"], d["answered"], d["recall"], d["recall_ci95"]), (0, 1, None, None))
        self.assertEqual(m["classes_absent"], ["d"])  # left out of every mean
        self.assertEqual(metrics.balanced_accuracy(METRIC_TRUE, METRIC_PRED, METRIC_LABELS), m["balanced_accuracy"])

    def test_a_class_that_was_never_given_has_no_precision_but_counts_as_zero_in_the_macro_scores(self):
        m = metrics.summarize(list("aabb"), list("aaaa"), ["a", "b"])
        self.assertIsNone(m["per_class"]["b"]["precision"])
        self.assertEqual((m["per_class"]["b"]["recall"], m["per_class"]["b"]["f1"]), (0.0, 0.0))
        self.assertAlmostEqual(m["macro_precision"], 0.25)
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + 0) / 2)  # a: F1 = 4 / (4 + 2 + 0)

    def test_an_invalid_answer_lowers_recall_and_is_nobodys_false_positive(self):
        gaps = metrics.summarize(list("aabb"), ["a", None, "b", "b"], ["a", "b"])
        self.assertEqual((gaps["per_class"]["a"]["recall"], gaps["per_class"]["a"]["precision"]), (0.5, 1.0))
        self.assertEqual(gaps["invalid"], 1)

    def test_top3_detection_and_selective_scores_by_hand(self):
        m = json_summary()
        # top-3: class a: the true class in 2 of its 3 differentials (the invalid answer is a miss); class b: in 2 of 3
        self.assertAlmostEqual(m["top3_balanced_accuracy"], (2 / 3 + 2 / 3) / 2)
        d = m["detection"]
        # the 5 valid answers: right 2 (a yes, b no), wrong 3 (no, yes, no)
        self.assertEqual((d["valid"], d["wrong"], d["flagged_wrong"], d["right"], d["unflagged_right"]), (5, 3, 2, 2, 1))
        self.assertAlmostEqual(d["sensitivity"], 2 / 3)
        self.assertAlmostEqual(d["specificity"], 1 / 2)
        # coverage: answers with "yes" (2) of all 6 questions; selective risk: of those 2, 1 is wrong
        self.assertEqual((d["covered"], d["covered_wrong"]), (2, 1))
        self.assertAlmostEqual(d["coverage"], 2 / 6)
        self.assertAlmostEqual(d["selective_risk"], 1 / 2)
        self.assertEqual((d["sensitivity_ci95"], d["coverage_ci95"]), (metrics.wilson(2, 3), metrics.wilson(2, 6)))

    def test_an_invalid_answer_is_never_counted_as_detected_or_covered_even_with_a_flag(self):
        m = metrics.summarize(["a", "a", "b"], [None, None, "b"], ["a", "b"], differentials=[None, None, ["b", "a"]],
                              flags=["no", "yes", "yes"])
        d = m["detection"]
        self.assertEqual((d["valid"], d["wrong"], d["flagged_wrong"], d["covered"], d["covered_wrong"]), (1, 0, 0, 1, 0))
        self.assertAlmostEqual(d["coverage"], 1 / 3)

    def test_answers_without_a_differential_or_a_flag_have_no_top3_or_detection_scores(self):
        m = metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        self.assertEqual((m["top3_balanced_accuracy"], m["top3_balanced_accuracy_ci95"], m["detection"]), (None, None, None))

    def test_invalid_input_is_refused_instead_of_scored(self):
        for true, pred, labels in ((["a"], ["a", "a"], ["a"]), (["z"], ["a"], ["a", "b"]), (["a"], ["z"], ["a", "b"]),
                                   (["a"], ["a"], ["a", "a"])):
            with self.subTest(true=true, pred=pred, labels=labels), self.assertRaises(ValueError):
                metrics.summarize(true, pred, labels)
        with self.assertRaises(ValueError):
            metrics.summarize(["a"], ["a"], ["a"], groups=["g1", "g2"])

    def test_no_answers_at_all_gives_no_score(self):
        m = metrics.summarize([], [], ["a", "b"], differentials=[], flags=[])
        self.assertEqual((m["n"], m["invalid"], m["invalid_rate"], m["classes_absent"]), (0, 0, None, ["a", "b"]))
        for value in (m["balanced_accuracy"], m["balanced_accuracy_ci95"], m["macro_f1"], m["top3_balanced_accuracy"],
                      m["detection"]["sensitivity"], m["detection"]["coverage"]):
            self.assertIsNone(value)

    def test_the_bootstrap_is_reproducible_and_collapses_when_all_is_right(self):
        first = metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        self.assertEqual(first["macro_f1_ci95"], metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)["macro_f1_ci95"])
        self.assertAlmostEqual(first["macro_f1_ci95"][0], 0.3, places=6)  # pinned: a change here means the resampling changed
        self.assertAlmostEqual(first["macro_f1_ci95"][1], 14 / 15, places=6)
        perfect = metrics.summarize(METRIC_TRUE, METRIC_TRUE, METRIC_LABELS)
        self.assertEqual((perfect["macro_f1_ci95"], perfect["balanced_accuracy_ci95"]), ((1.0, 1.0), (1.0, 1.0)))

    def test_photos_of_one_lesion_are_resampled_together_so_the_intervals_widen(self):
        true = ["a"] * 12 + ["b"] * 12
        pred = ["a"] * 8 + ["b"] * 4 + ["b"] * 6 + ["a"] * 6
        lesions = [f"L{i // 4}" for i in range(24)]  # four photos per lesion
        alone = metrics.summarize(true, pred, ["a", "b"])
        grouped = metrics.summarize(true, pred, ["a", "b"], groups=lesions)
        self.assertEqual(grouped["balanced_accuracy"], alone["balanced_accuracy"])  # the scores themselves do not change ...

        def width(pair):
            return pair[1] - pair[0]
        for key in ("balanced_accuracy_ci95", "macro_f1_ci95"):  # ... only how certain they are
            self.assertGreater(width(grouped[key]), width(alone[key]), key)
        self.assertGreater(width(grouped["per_class"]["a"]["recall_ci95"]), width(alone["per_class"]["a"]["recall_ci95"]))
        self.assertTrue(grouped["interval_method"].startswith("bootstrap over"))

    def test_grouped_answers_get_bootstrap_intervals_for_the_detection_scores(self):
        d = json_summary(groups=["L1", "L1", "L2", "L3", "L3", "L4"])["detection"]
        for name in ("sensitivity", "specificity", "coverage", "selective_risk"):
            pair = d[f"{name}_ci95"]
            self.assertTrue(pair is None or 0 <= pair[0] <= pair[1] <= 1, name)

    def test_mcnemar_exact_values_by_hand(self):
        self.assertEqual(metrics.mcnemar(0, 0), 1.0)
        self.assertEqual(metrics.mcnemar(0, 5), 2 / 32)
        self.assertEqual(metrics.mcnemar(9, 1), 2 * 11 / 1024)
        self.assertEqual(metrics.mcnemar(3, 3), 1.0)

    def test_the_holm_correction_by_hand(self):
        # four tests: 0.01 x 4 = 0.04; 0.03 x 3 = 0.09; 0.04 x 2 = 0.08, raised to the 0.09 before it; 0.5 x 1
        self.assertEqual(metrics.holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])
        self.assertEqual(metrics.holm([0.6, 0.9]), [1.0, 1.0])
        self.assertEqual((metrics.holm([]), metrics.holm([0.2])), ([], [0.2]))

    def test_the_summary_can_be_saved_as_json(self):
        saved = json.loads(json.dumps(json_summary()))
        self.assertEqual(saved["confusion_matrix"]["columns"][-1], metrics.INVALID)
        self.assertIsNone(saved["per_class"]["d"]["recall"])


if __name__ == "__main__":
    unittest.main()
