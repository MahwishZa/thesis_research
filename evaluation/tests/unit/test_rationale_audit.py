import unittest

from experiments.baseline.filter_training.rationale_audit import (
    classify_generation, strict_flip_summary, strict_triple_summary, summarize,
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


def _g(text, finish, scored, correct):
    g = classify_generation(text, finish, scored)
    g["correct"] = correct
    return g


class StrictFlipTests(unittest.TestCase):

    def rows(self):
        return [
            # clean flip to wrong (explicit answers both sides)
            dict(gold_letter="B", without=_g("Answer: B", "stop", "B", True),
                 with_evidence=_g("Answer: C", "stop", "C", False)),
            # lenient flip caused by a guessed (truncated) generation
            dict(gold_letter="B", without=_g("Answer: B", "stop", "B", True),
                 with_evidence=_g("A man with", "length", "A", False)),
            # no flip
            dict(gold_letter="B", without=_g("Answer: B", "stop", "B", True),
                 with_evidence=_g("Answer: B", "stop", "B", True)),
        ]

    def test_counts_flips_and_flags_guess_driven_ones(self):
        s = strict_flip_summary(self.rows())
        self.assertEqual(s["lenient_flips"], 2)
        self.assertEqual(s["lenient_flips_involving_a_guessed_generation"], 1)
        self.assertEqual(s["strict_scorable_pairs"], 2)
        self.assertEqual((s["strict_flip_to_correct"], s["strict_flip_to_wrong"]), (0, 1))
        self.assertEqual(s["strict_flip_rate"], 0.5)


class TripleTests(unittest.TestCase):

    def test_matched_pairs_only_and_none_without_control(self):
        a = lambda L: _g(f"Answer: {L}", "stop", L, False)
        rows = [
            dict(gold_letter="B", without=a("B"), with_evidence=a("B"), control=a("C")),
            dict(gold_letter="B", without=a("C"), with_evidence=a("B"), control=a("C")),
            dict(gold_letter="B", without=a("B"), with_evidence=a("B"),
                 control=_g("A man", "length", "A", False)),   # excluded
        ]
        s = strict_triple_summary(rows)
        self.assertEqual(s["matched_pairs"], 2)
        self.assertEqual(s["accuracy_without"], 0.5)
        self.assertEqual(s["accuracy_with_evidence"], 1.0)
        self.assertEqual(s["accuracy_control"], 0.0)
        self.assertEqual(s["control_flip_to_wrong"], 1)
        self.assertEqual(s["with_evidence_flip_to_correct"], 1)
        self.assertIsNone(strict_triple_summary([rows[0]["without"] and
                          dict(gold_letter="B", without=a("B"), with_evidence=a("B"))]))


if __name__ == "__main__":
    unittest.main()
