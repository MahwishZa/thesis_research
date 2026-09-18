"""Tests for experiments/evaluation/rag_metrics.py, the standard-metrics
ablation track (current objective 2)."""

import unittest

from experiments.evaluation import rag_metrics as rm


class NormalizeTextTests(unittest.TestCase):

    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(rm.normalize_text("The Answer, is 42."), "answer is 42")

    def test_drops_articles(self):
        self.assertEqual(rm.normalize_text("a an the cat"), "cat")

    def test_none_and_empty(self):
        self.assertEqual(rm.normalize_text(None), "")
        self.assertEqual(rm.normalize_text(""), "")


class ExactMatchTests(unittest.TestCase):

    def test_exact_match_after_normalization(self):
        self.assertEqual(rm.exact_match("The Cat.", "a cat"), 1.0)

    def test_no_match(self):
        self.assertEqual(rm.exact_match("dog", "cat"), 0.0)


class TokenF1Tests(unittest.TestCase):

    def test_identical_strings_score_one(self):
        self.assertEqual(rm.token_f1("value is 42", "value is 42"), 1.0)

    def test_partial_overlap(self):
        # pred: {value, is, 5}; ref: {value, is, 42} -> overlap 2/3 each side
        score = rm.token_f1("value is 5", "value is 42")
        self.assertAlmostEqual(score, 2 / 3)

    def test_no_overlap_scores_zero(self):
        self.assertEqual(rm.token_f1("apple", "orange"), 0.0)

    def test_both_empty_scores_one(self):
        self.assertEqual(rm.token_f1("", ""), 1.0)

    def test_one_empty_scores_zero(self):
        self.assertEqual(rm.token_f1("something", ""), 0.0)
        self.assertEqual(rm.token_f1("", "something"), 0.0)


class RougeLTests(unittest.TestCase):

    def test_identical_strings_score_one(self):
        self.assertEqual(rm.rouge_l_f1("a b c d", "a b c d"), 1.0)

    def test_reordering_lowers_the_score(self):
        # LCS("a b c", "c b a") = 1 token
        score = rm.rouge_l_f1("a b c", "c b a")
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_disjoint_scores_zero(self):
        self.assertEqual(rm.rouge_l_f1("apple pear", "orange grape"), 0.0)


class ContextScoresTests(unittest.TestCase):

    def test_perfect_overlap(self):
        ctx = rm.context_scores(["a", "b"], ["a", "b"])
        self.assertEqual((ctx.precision, ctx.recall, ctx.f1), (1.0, 1.0, 1.0))

    def test_partial_overlap(self):
        ctx = rm.context_scores(["a", "b", "c"], ["a", "z"])
        self.assertAlmostEqual(ctx.precision, 1 / 3)
        self.assertAlmostEqual(ctx.recall, 1 / 2)

    def test_no_gold_and_nothing_admitted_is_vacuously_perfect(self):
        ctx = rm.context_scores([], [])
        self.assertEqual((ctx.precision, ctx.recall, ctx.f1), (1.0, 1.0, 1.0))

    def test_nothing_admitted_but_gold_exists_scores_zero(self):
        ctx = rm.context_scores([], ["a"])
        self.assertEqual((ctx.precision, ctx.recall, ctx.f1), (0.0, 0.0, 0.0))

    def test_admitted_but_no_gold_scores_zero(self):
        ctx = rm.context_scores(["a"], [])
        self.assertEqual((ctx.precision, ctx.recall, ctx.f1), (0.0, 0.0, 0.0))


class GroundednessTests(unittest.TestCase):

    def test_fully_grounded(self):
        self.assertEqual(
            rm.groundedness("the value is 42", "context: the value is 42"), 1.0
        )

    def test_partially_grounded(self):
        score = rm.groundedness("value is 42 today", "value is 42")
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_empty_prediction_is_vacuously_grounded(self):
        self.assertEqual(rm.groundedness("", "some context"), 1.0)

    def test_nonempty_prediction_with_no_context_scores_zero(self):
        self.assertEqual(rm.groundedness("value is 42", ""), 0.0)

    def test_stopwords_do_not_count_against_groundedness(self):
        # "is", "the" are stopwords/articles; only "value" and "42" count.
        self.assertEqual(rm.groundedness("the value is 42", "42 value"), 1.0)


class ScoreRecordTests(unittest.TestCase):

    def test_scores_a_runner_style_record(self):
        record = {
            "question_id": "Q1",
            "system": "proposed",
            "generated_answer": "value is 42",
            "reference_answer": "value is 42",
            "admitted_evidence_ids": ["E1"],
            "admitted_evidence_text": ["value is 42"],
        }
        row = rm.score_record(record, gold_evidence_ids={"E1"})
        self.assertEqual(row.question_id, "Q1")
        self.assertEqual(row.system, "proposed")
        self.assertEqual(row.exact_match, 1.0)
        self.assertEqual(row.token_f1, 1.0)
        self.assertEqual(row.context_precision, 1.0)
        self.assertEqual(row.context_recall, 1.0)
        self.assertEqual(row.groundedness, 1.0)

    def test_missing_fields_default_sensibly(self):
        record = {
            "question_id": "Q2",
            "system": "baseline",
            "generated_answer": None,
            "reference_answer": None,
            "admitted_evidence_ids": None,
            "admitted_evidence_text": None,
        }
        row = rm.score_record(record, gold_evidence_ids=set())
        self.assertEqual(row.exact_match, 1.0)  # both empty


class AggregateTests(unittest.TestCase):

    def test_aggregate_averages_across_rows(self):
        rows = [
            rm.MetricRow("Q1", "s", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            rm.MetricRow("Q2", "s", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ]
        agg = rm.aggregate(rows)
        self.assertEqual(agg["n"], 2)
        self.assertEqual(agg["exact_match"], 0.5)
        self.assertEqual(agg["groundedness"], 0.5)

    def test_aggregate_of_empty_rows(self):
        agg = rm.aggregate([])
        self.assertEqual(agg["n"], 0)
        self.assertEqual(agg["exact_match"], 0.0)

    def test_aggregate_by_system_groups_correctly(self):
        rows = [
            rm.MetricRow("Q1", "a", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            rm.MetricRow("Q1", "b", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            rm.MetricRow("Q2", "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ]
        by_system = rm.aggregate_by_system(rows)
        self.assertEqual(set(by_system), {"a", "b"})
        self.assertEqual(by_system["a"]["n"], 2)
        self.assertEqual(by_system["a"]["exact_match"], 0.5)
        self.assertEqual(by_system["b"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
