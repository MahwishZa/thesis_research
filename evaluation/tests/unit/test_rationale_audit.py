import unittest

from experiments.baseline.filter_training.rationale_audit import (
    classify_generation, summarize,
)


class ClassifyTests(unittest.TestCase):

    def test_clean_explicit_answer(self):
        g = classify_generation("reasoning...\nAnswer: C", "stop", "C")
        self.assertFalse(g["truncated"])
        self.assertTrue(g["has_explicit_answer"])
        self.assertFalse(g["fallback_or_mismatch"])

    def test_truncated_with_stray_article_letter_is_flagged(self):
        g = classify_generation("A 45-year-old man presents with", "length", "A")
        self.assertTrue(g["truncated"])
        self.assertFalse(g["has_explicit_answer"])
        self.assertTrue(g["fallback_or_mismatch"])

    def test_explicit_disagreeing_with_scored_letter_is_flagged(self):
        g = classify_generation("Answer: B\n\nNote that option D is odd", "stop", "D")
        self.assertEqual(g["explicit_letter"], "B")
        self.assertTrue(g["fallback_or_mismatch"])

    def test_case_and_parenthesis_variants(self):
        self.assertEqual(
            classify_generation("answer: (b)", "stop", "B")["explicit_letter"], "B")


class SummaryTests(unittest.TestCase):

    def test_rates(self):
        ok = classify_generation("Answer: A", "stop", "A")
        bad = classify_generation("A man", "length", None)
        s = summarize([{"without": ok, "with_evidence": bad},
                       {"without": ok, "with_evidence": ok}])
        self.assertEqual(s["with_evidence"]["truncated_rate"], 0.5)
        self.assertEqual(s["without"]["explicit_answer_rate"], 1.0)
        self.assertEqual(s["with_evidence"]["no_letter_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
