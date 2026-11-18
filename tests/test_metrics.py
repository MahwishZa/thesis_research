"""The metrics (src/metrics.py), checked against values worked out by hand."""

import json
import unittest
from pathlib import Path

from src import metrics


ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================================================================
# The metrics
# ==============================================================================================================================

METRIC_LABELS = ["a", "b", "c", "d"]  # d never occurs as a true class below
METRIC_TRUE = list("aaaabbbccc")
METRIC_PRED = ["a", "a", "b", None, "b", "b", "c", "c", "c", "d"]  # one unreadable answer, and one answer naming the absent class d
# Confusion matrix (rows: true a, b, c, d; columns: answered a, b, c, d, unreadable)
#   a: 2 1 0 0 1    b: 0 2 1 0 0    c: 0 0 2 1 0    d: 0 0 0 0 0
# Worked by hand: precision a 2/2, b 2/3, c 2/3; recall a 2/4, b 2/3, c 2/3; F1 a 4/6, b 4/6, c 4/6.
# MCC = (6*10 - (2*4 + 3*3 + 3*3)) / sqrt((100 - (4+9+9+1+1)) * (100 - (16+9+9))) = 34 / sqrt(76*66);
# kappa = (0.6 - 26/100) / (1 - 26/100). Both agree with scikit-learn on this example.


class MetricTests(unittest.TestCase):
    def test_wilson_known_value(self):
        low, high = metrics.wilson(8, 10)
        self.assertAlmostEqual(low, 0.4902, places=3)
        self.assertAlmostEqual(high, 0.9433, places=3)
        self.assertEqual(metrics.wilson(0, 0), (0.0, 0.0))

    def test_accuracy_counts_unreadable_as_wrong_and_keeps_its_original_fields(self):
        m = metrics.summarize(["a", "a", "b", "b"], ["a", "b", "b", None], ["a", "b"])
        self.assertEqual((m["n"], m["correct"], m["unreadable"]), (4, 2, 1))
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertEqual(m["accuracy_ci95"], metrics.wilson(2, 4))

    def test_the_confusion_matrix_has_a_column_for_unreadable_answers(self):
        m = metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)["confusion_matrix"]
        self.assertEqual(m["rows"], METRIC_LABELS)
        self.assertEqual(m["columns"], METRIC_LABELS + [metrics.UNREADABLE])
        self.assertEqual(m["counts"], [[2, 1, 0, 0, 1], [0, 2, 1, 0, 0], [0, 0, 2, 1, 0], [0, 0, 0, 0, 0]])

    def test_per_class_precision_recall_and_f1_by_hand(self):
        per_class = metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)["per_class"]
        a, b, c, d = (per_class[x] for x in METRIC_LABELS)
        self.assertEqual((a["support"], a["answered"], a["tp"], a["fp"], a["fn"]), (4, 2, 2, 0, 2))
        self.assertAlmostEqual(a["precision"], 1.0)
        self.assertAlmostEqual(a["recall"], 0.5)
        self.assertAlmostEqual(a["f1"], 4 / 6)
        for x in (b, c):
            self.assertAlmostEqual(x["precision"], 2 / 3)
            self.assertAlmostEqual(x["recall"], 2 / 3)
            self.assertAlmostEqual(x["f1"], 2 / 3)
        self.assertEqual(a["recall_ci95"], metrics.wilson(2, 4))

    def test_a_class_without_true_images_has_no_recall_and_stays_out_of_the_averages(self):
        m = metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        d = m["per_class"]["d"]
        self.assertEqual((d["support"], d["answered"], d["fp"]), (0, 1, 1))  # a wrong answer naming it is still a false positive
        self.assertIsNone(d["recall"])
        self.assertIsNone(d["recall_ci95"])
        self.assertEqual(m["classes_absent"], ["d"])
        self.assertAlmostEqual(m["macro"]["precision"], (1 + 2 / 3 + 2 / 3) / 3)  # a, b, c only
        self.assertAlmostEqual(m["macro"]["recall"], (0.5 + 2 / 3 + 2 / 3) / 3)
        self.assertAlmostEqual(m["macro"]["f1"], 2 / 3)

    def test_weighted_averages_use_the_number_of_true_images(self):
        w = metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)["weighted"]
        self.assertAlmostEqual(w["precision"], (4 * 1 + 3 * 2 / 3 + 3 * 2 / 3) / 10)
        self.assertAlmostEqual(w["recall"], (4 * 0.5 + 3 * 2 / 3 + 3 * 2 / 3) / 10)
        self.assertAlmostEqual(w["f1"], 2 / 3)

    def test_balanced_accuracy_mcc_and_kappa(self):
        m = metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        self.assertAlmostEqual(m["balanced_accuracy"], m["macro"]["recall"])
        self.assertAlmostEqual(m["mcc"], 34 / (76 * 66) ** 0.5)
        self.assertAlmostEqual(m["kappa"], (0.6 - 0.26) / (1 - 0.26))

    def test_a_class_that_was_never_answered_has_no_precision_but_counts_as_zero_in_the_macro_precision(self):
        m = metrics.summarize(list("aabb"), list("aaaa"), ["a", "b"])
        self.assertIsNone(m["per_class"]["b"]["precision"])
        self.assertAlmostEqual(m["per_class"]["b"]["recall"], 0.0)
        self.assertAlmostEqual(m["per_class"]["b"]["f1"], 0.0)
        self.assertAlmostEqual(m["macro"]["precision"], (0.5 + 0.0) / 2)  # a: 2 of 4 answers right; b: undefined, counted as 0

    def test_unreadable_answers_lower_recall_and_are_never_dropped(self):
        full = metrics.summarize(list("aabb"), list("aabb"), ["a", "b"])
        gaps = metrics.summarize(list("aabb"), ["a", None, "b", "b"], ["a", "b"])
        self.assertAlmostEqual(full["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(gaps["per_class"]["a"]["recall"], 0.5)
        self.assertAlmostEqual(gaps["per_class"]["a"]["precision"], 1.0)  # the unreadable answer is nobody's false positive
        self.assertEqual((gaps["unreadable"], gaps["confusion_matrix"]["counts"][0]), (1, [1, 0, 1]))

    def test_invalid_input_is_refused_instead_of_scored(self):
        for true, pred, labels in ((["a"], ["a", "a"], ["a"]),            # different lengths
                                   (["z"], ["a"], ["a", "b"]),            # a true label that is not a class
                                   (["a"], ["z"], ["a", "b"]),            # an answer that is neither a class nor None
                                   (["a"], ["a"], ["a", "a"])):           # a class listed twice
            with self.subTest(true=true, pred=pred, labels=labels), self.assertRaises(ValueError):
                metrics.summarize(true, pred, labels)

    def test_no_answers_at_all_gives_accuracy_zero_and_no_other_score(self):
        m = metrics.summarize([], [], ["a", "b"])
        self.assertEqual((m["n"], m["accuracy"], m["unreadable"]), (0, 0.0, 0))
        self.assertEqual(m["classes_absent"], ["a", "b"])
        for value in (m["macro"]["f1"], m["macro"]["f1_ci95"], m["weighted"]["f1"], m["balanced_accuracy"], m["mcc"], m["kappa"]):
            self.assertIsNone(value)

    def test_an_answer_pattern_that_cannot_be_scored_gives_none_not_a_crash(self):
        m = metrics.summarize(list("aaaa"), list("aaaa"), ["a", "b"])  # one class only: MCC and kappa are 0/0
        self.assertIsNone(m["mcc"])
        self.assertIsNone(m["kappa"])
        self.assertAlmostEqual(m["accuracy"], 1.0)

    def test_the_bootstrap_interval_is_reproducible_stays_with_the_class_sizes_and_collapses_when_all_is_right(self):
        first = metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        again = metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)
        self.assertEqual(first["macro"]["f1_ci95"], again["macro"]["f1_ci95"])  # seeded: the same answers give the same interval
        self.assertAlmostEqual(first["macro"]["f1_ci95"][0], 0.3, places=6)  # pinned: a change here means the resampling changed
        self.assertAlmostEqual(first["macro"]["f1_ci95"][1], 14 / 15, places=6)
        perfect = metrics.summarize(METRIC_TRUE, METRIC_TRUE, METRIC_LABELS)
        self.assertEqual(perfect["macro"]["f1_ci95"], (1.0, 1.0))
        self.assertEqual(perfect["balanced_accuracy_ci95"], (1.0, 1.0))

    def test_photos_of_one_lesion_are_resampled_together_so_the_intervals_widen(self):
        true = ["a"] * 12 + ["b"] * 12
        pred = ["a"] * 8 + ["b"] * 4 + ["b"] * 6 + ["a"] * 6
        lesions = [f"L{i // 4}" for i in range(24)]  # four photos per lesion
        alone = metrics.summarize(true, pred, ["a", "b"])
        grouped = metrics.summarize(true, pred, ["a", "b"], groups=lesions)
        self.assertEqual(grouped["balanced_accuracy"], alone["balanced_accuracy"])  # the scores themselves do not change ...
        def width(pair):
            return pair[1] - pair[0]
        for key in ("balanced_accuracy_ci95", "accuracy_ci95", "macro"):  # ... only how certain they are
            a, g = (alone[key]["f1_ci95"], grouped[key]["f1_ci95"]) if key == "macro" else (alone[key], grouped[key])
            self.assertGreater(width(g), width(a), key)
        self.assertGreater(width(grouped["per_class"]["a"]["recall_ci95"]), width(alone["per_class"]["a"]["recall_ci95"]))
        self.assertEqual((grouped["groups"], grouped["interval_method"], alone["groups"]), (6, "bootstrap over lesions", 24))
        self.assertEqual(metrics.summarize(true, pred, ["a", "b"], groups=[str(i) for i in range(24)]), alone)  # one image each
        with self.assertRaisesRegex(ValueError, "one group per answer"):
            metrics.summarize(true, pred, ["a", "b"], groups=["L1"])

    def test_cases_of_one_article_are_resampled_together_and_the_summary_says_so(self):
        m = metrics.summarize(["a", "a", "a", "b"], ["a", "b", "a", "b"], ["a", "b"], groups=["p1", "p1", "p2", "p3"], group_name="article")
        self.assertEqual((m["groups"], m["group_name"], m["interval_method"]), (3, "article", "bootstrap over articles"))

    def test_missed_malignancy_and_over_calls_are_counted_by_hand(self):
        labels = ["actinic keratosis", "basal cell carcinoma", "melanoma", "nevus"]
        true = ["melanoma", "melanoma", "melanoma", "basal cell carcinoma", "nevus", "nevus", "actinic keratosis"]
        pred = ["melanoma", "nevus", None, "actinic keratosis", "nevus", "melanoma", "nevus"]
        risk = metrics.summarize(true, pred, labels)["risk"]
        # cancers: 4, of which 1 answered benign (melanoma -> nevus); unreadable and actinic keratosis count as neither
        self.assertEqual((risk["missed"], risk["cancers"], risk["missed_malignancy"]), (1, 4, 0.25))
        self.assertEqual((risk["overcalled"], risk["benign"], risk["overcall"]), (1, 2, 0.5))  # nevus -> melanoma
        self.assertEqual(risk["missed_malignancy_ci95"], metrics.wilson(1, 4))  # every answer its own lesion: Wilson
        grouped = metrics.summarize(true, pred, labels, groups=["L1", "L1", "L2", "L3", "L4", "L5", "L6"])["risk"]
        low, high = grouped["missed_malignancy_ci95"]
        self.assertTrue(0 <= low <= 0.25 <= high <= 1)  # bootstrap over lesions
        none = metrics.summarize(["a", "b"], ["a", "a"], ["a", "b"])["risk"]  # classes outside both groups: undefined, not 0
        self.assertEqual((none["missed_malignancy"], none["missed_malignancy_ci95"], none["overcall"]), (None, None, None))

    def test_a_hallucination_is_a_wrong_diagnosis_and_a_refusal_is_not(self):
        true = ["a"] * 4 + ["b"] * 2 + ["c"]
        pred = ["a", "b", None, "c", "b", "a", None]
        # class a: 1 right, 2 wrong diagnoses (b, c), 1 no valid answer; class b: 1 right, 1 wrong; class c: 1 no valid answer
        h = metrics.summarize(true, pred, ["a", "b", "c"])["hallucination"]
        self.assertEqual((h["wrong"], h["n"], h["rate"], h["no_valid_answer"], h["no_valid_answer_rate"]), (3, 7, 3 / 7, 2, 2 / 7))
        self.assertAlmostEqual(h["balanced_rate"], (2 / 4 + 1 / 2 + 0 / 1) / 3)
        self.assertEqual(h["rate_ci95"], metrics.wilson(3, 7))  # every answer its own lesion
        self.assertEqual(2 + h["wrong"] + h["no_valid_answer"], 7)  # right + hallucination + no valid answer = every question
        equal = metrics.summarize(["a", "a", "b", "b"], ["a", "b", "b", None], ["a", "b"])["hallucination"]
        self.assertEqual((equal["rate"], equal["balanced_rate"]), (0.25, 0.25))  # the same number of questions in every class
        empty = metrics.summarize([], [], ["a"])["hallucination"]
        self.assertEqual((empty["rate"], empty["balanced_rate"]), (None, None))  # nothing answered: undefined, not 0
        grouped = metrics.summarize(true, pred, ["a", "b", "c"], groups=["L1", "L1", "L2", "L3", "L4", "L5", "L6"])["hallucination"]
        low, high = grouped["balanced_rate_ci95"]
        self.assertTrue(0 <= low <= grouped["balanced_rate"] <= high <= 1)

    def test_mcnemar_exact_values_by_hand(self):
        self.assertEqual(metrics.mcnemar(0, 0), 1.0)  # no question decides between the runs
        self.assertEqual(metrics.mcnemar(0, 5), 2 / 32)  # 2 x (1/2)^5
        self.assertEqual(metrics.mcnemar(9, 1), 2 * 11 / 1024)  # 2 x (C(10,0) + C(10,1)) / 2^10
        self.assertEqual(metrics.mcnemar(3, 3), 1.0)

    def test_the_summary_can_be_saved_as_json(self):
        saved = json.loads(json.dumps(metrics.summarize(METRIC_TRUE, METRIC_PRED, METRIC_LABELS)))
        self.assertEqual(saved["confusion_matrix"]["columns"][-1], metrics.UNREADABLE)
        self.assertIsNone(saved["per_class"]["d"]["recall"])

    def test_specificity_and_malignancy_sensitivity_by_hand(self):
        """Five files: melanoma answered melanoma and actinic keratosis; nevus answered melanoma and unreadably; actinic keratosis
        answered nevus."""
        labels = ["actinic keratosis", "melanoma", "nevus"]
        truth = ["melanoma", "melanoma", "nevus", "nevus", "actinic keratosis"]
        answers = ["melanoma", "actinic keratosis", "melanoma", None, "nevus"]
        m = metrics.summarize(truth, answers, labels)
        per = m["per_class"]
        # of the 3 files that are not melanoma, one was answered melanoma; likewise nevus; of the 4 not actinic keratosis, one
        self.assertEqual([(per[c]["tn"], per[c]["specificity"]) for c in labels], [(3, 3 / 4), (2, 2 / 3), (2, 2 / 3)])
        self.assertAlmostEqual(m["macro"]["specificity"], (3 / 4 + 2 / 3 + 2 / 3) / 3)
        risk = m["risk"]
        # 2 cancers: one answered as a cancer, the other as actinic keratosis, which is neither detected nor missed
        self.assertEqual((risk["detected"], risk["malignancy_sensitivity"], risk["missed"], risk["missed_malignancy"]), (1, 0.5, 0, 0.0))
        # 2 benign lesions: one called a cancer, the other unreadable, which is neither cleared nor over-called
        self.assertEqual((risk["cleared"], risk["benign_specificity"], risk["overcalled"], risk["overcall"]), (0, 0.0, 1, 0.5))
        for interval in (risk["malignancy_sensitivity_ci95"], risk["benign_specificity_ci95"], m["macro"]["precision_ci95"],
                         m["macro"]["specificity_ci95"], per["melanoma"]["specificity_ci95"]):
            self.assertTrue(0.0 <= interval[0] <= interval[1] <= 1.0)
        self.assertEqual(risk["malignancy_sensitivity_ci95"], metrics.wilson(1, 2))  # every file its own group: Wilson
        self.assertEqual(per["melanoma"]["specificity_ci95"], metrics.wilson(2, 3))
        self.assertTrue(risk["malignancy_sensitivity_ci95"][0] <= 0.5 <= risk["malignancy_sensitivity_ci95"][1])
        grouped = metrics.summarize(truth, answers, labels, groups=["g1", "g1", "g2", "g3", "g4"])  # two files of one lesion: bootstrap
        low, high = grouped["risk"]["malignancy_sensitivity_ci95"]
        self.assertEqual((low, high), (0.5, 0.5))  # the one melanoma lesion is drawn whole in every resample
        self.assertEqual(grouped["per_class"]["melanoma"]["specificity"], 2 / 3)

    def test_a_perfect_run_has_full_specificity_and_sensitivity_and_a_class_with_no_negatives_has_none(self):
        labels = ["melanoma", "nevus"]
        m = metrics.summarize(["melanoma", "nevus", "nevus"], ["melanoma", "nevus", "nevus"], labels)
        self.assertEqual((m["macro"]["specificity"], m["risk"]["malignancy_sensitivity"], m["risk"]["benign_specificity"]), (1.0, 1.0, 1.0))
        self.assertEqual((m["risk"]["malignancy_sensitivity_ci95"], m["risk"]["benign_specificity_ci95"]),
                         (metrics.wilson(1, 1), metrics.wilson(2, 2)))  # 1 cancer and 2 benign lesions: each share over its own group
        only = metrics.summarize(["melanoma", "melanoma"], ["melanoma", "nevus"], labels)  # no file that is not melanoma
        self.assertIsNone(only["per_class"]["melanoma"]["specificity"])
        self.assertIsNone(only["per_class"]["melanoma"]["specificity_ci95"])
        self.assertIsNone(only["macro"]["specificity"])
        self.assertEqual(only["per_class"]["nevus"]["specificity"], 0.5)  # of the 2 files that are not nevus, one was answered nevus


if __name__ == "__main__":
    unittest.main()
