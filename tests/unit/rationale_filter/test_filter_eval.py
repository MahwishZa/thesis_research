"""rationale_filter.filter_eval: accuracy as in the authors' evaluation, the majority rate, the AUC."""

import itertools
import random
import unittest

from rationale_filter import filter_eval
from tests.spec.targets import TOKEN_HELPFUL as H
from tests.spec.targets import TOKEN_NOT_HELPFUL as N


class AccuracyTests(unittest.TestCase):
    def test_overall_and_per_class_accuracy_and_the_confusion_counts(self):
        gold = {"a": H, "b": H, "c": H, "d": N, "e": N, "f": N, "g": N, "h": N}
        predictions = {"a": H, "b": H, "c": N, "d": N, "e": N, "f": N, "g": H, "h": N}
        report = filter_eval.accuracy_report(predictions, gold)
        self.assertEqual(report["n"], 8)
        self.assertEqual(report["accuracy"], 0.75)
        self.assertEqual(report["accuracy_per_class"], {H: 0.6667, N: 0.8})
        self.assertEqual(report["majority_rate"], 0.625)
        self.assertEqual(report["confusion"], {H: {H: 2, N: 1}, N: {H: 1, N: 4}})

    def test_a_filter_that_always_says_not_helpful_scores_the_majority_rate(self):
        gold = {str(i): (H if i < 3 else N) for i in range(10)}
        report = filter_eval.accuracy_report({key: N for key in gold}, gold)
        self.assertEqual(report["accuracy"], report["majority_rate"])
        self.assertEqual(report["accuracy_per_class"][H], 0.0)

    def test_a_class_without_gold_pairs_has_no_accuracy(self):
        report = filter_eval.accuracy_report({"a": H}, {"a": H})
        self.assertIsNone(report["accuracy_per_class"][N])

    def test_refused_input(self):
        for predictions, gold in (
            ({"a": H}, {"a": H, "b": N}),
            ({"a": H, "z": H}, {"a": H}),
            ({}, {}),
            ({"a": "helpful"}, {"a": H}),
            ({"a": H}, {"a": "x"}),
        ):
            with self.subTest(predictions=predictions, gold=gold), self.assertRaises(ValueError):
                filter_eval.accuracy_report(predictions, gold)


class AucTests(unittest.TestCase):
    def test_a_known_example(self):
        scores = {"a": 0.8, "b": 0.6, "c": 0.7, "d": 0.3, "e": 0.2}
        gold = {"a": H, "b": H, "c": N, "d": N, "e": N}
        self.assertAlmostEqual(filter_eval.auc(scores, gold), 5 / 6)

    def test_perfect_reversed_and_tied_scores(self):
        gold = {"a": H, "b": N}
        self.assertEqual(filter_eval.auc({"a": 2.0, "b": 1.0}, gold), 1.0)
        self.assertEqual(filter_eval.auc({"a": 1.0, "b": 2.0}, gold), 0.0)
        self.assertEqual(filter_eval.auc({"a": 1.0, "b": 1.0}, gold), 0.5)

    def test_it_equals_a_brute_force_count_on_random_data_with_ties(self):
        rng = random.Random(3)
        scores = {str(i): rng.choice([0.1, 0.2, 0.3, 0.4, 0.5]) for i in range(60)}
        gold = {key: (H if rng.random() < 0.4 else N) for key in scores}
        helpful = [scores[k] for k in gold if gold[k] == H]
        other = [scores[k] for k in gold if gold[k] == N]
        wins = sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(helpful, other))
        self.assertAlmostEqual(filter_eval.auc(scores, gold), wins / (len(helpful) * len(other)))

    def test_one_class_only_is_refused(self):
        with self.assertRaises(ValueError):
            filter_eval.auc({"a": 1.0, "b": 2.0}, {"a": H, "b": H})

    def test_different_ids_or_unknown_labels_are_refused(self):
        with self.assertRaises(ValueError):
            filter_eval.auc({"a": 1.0}, {"b": H})
        with self.assertRaises(ValueError):
            filter_eval.auc({"a": 1.0, "b": 2.0}, {"a": H, "b": "x"})


class LogitTests(unittest.TestCase):
    def test_predictions_follow_the_larger_logit_and_a_tie_is_helpful(self):
        logits = {"a": (2.0, 1.0), "b": (1.0, 2.0), "c": (0.5, 0.5)}
        self.assertEqual(filter_eval.predictions_from_logits(logits), {"a": H, "b": N, "c": H})

    def test_the_scores_are_the_differences_of_the_logits(self):
        self.assertEqual(filter_eval.helpful_scores({"a": (2.0, 0.5), "b": (-1.0, 1.0)}), {"a": 1.5, "b": -2.0})

    def test_a_non_finite_logit_is_an_error(self):
        with self.assertRaises(ValueError):
            filter_eval.predictions_from_logits({"a": (float("nan"), 1.0)})


if __name__ == "__main__":
    unittest.main()
