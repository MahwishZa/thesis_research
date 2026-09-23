"""Tests for the pure-Python logic in the new filter-training modules
(medqa_data.py, train.py's data split). Nothing here loads a model or
dataset - the parts that do (load_medqa, load_textbook_passages,
Llama3RationaleScorer, train.py's Trainer loop) need torch/transformers/
datasets and a GPU, and are exercised on the actual Colab run, not here -
matching this test suite's existing rule (test_retrieval.py's own docstring)
that model-loading code stays out of the unit-test path.
"""

import unittest

from experiments.baseline.filter_training.config import ConfigError
from experiments.baseline.filter_training.medqa_data import (
    MedQADataError, MedQAItem, extract_answer_letter,
)
from experiments.baseline.filter_training.train import split_train_val


def item(**kw):
    base = dict(
        item_id="medqa_train_0",
        question="Which vessel supplies the anterior cerebral circulation?",
        options={"A": "a", "B": "b", "C": "c", "D": "d"},
        answer_letter="B",
    )
    base.update(kw)
    return MedQAItem(**base)


class MedQAItemTests(unittest.TestCase):

    def test_rendered_question_appends_options_in_order(self):
        rendered = item().rendered_question()
        self.assertTrue(rendered.startswith(
            "Which vessel supplies the anterior cerebral circulation? "
        ))
        self.assertIn("A) a", rendered)
        self.assertIn("B) b", rendered)
        self.assertIn("C) c", rendered)
        self.assertIn("D) d", rendered)
        # Order matters: A before B before C before D.
        self.assertLess(rendered.index("A) a"), rendered.index("B) b"))
        self.assertLess(rendered.index("B) b"), rendered.index("C) c"))
        self.assertLess(rendered.index("C) c"), rendered.index("D) d"))

    def test_refuses_wrong_option_keys(self):
        with self.assertRaises(MedQADataError):
            item(options={"A": "a", "B": "b", "C": "c", "E": "e"})

    def test_refuses_missing_option(self):
        with self.assertRaises(MedQADataError):
            item(options={"A": "a", "B": "b", "C": "c"})

    def test_refuses_answer_letter_outside_options(self):
        with self.assertRaises(MedQADataError):
            item(answer_letter="E")


class ExtractAnswerLetterTests(unittest.TestCase):

    def test_finds_a_clearly_labelled_answer(self):
        self.assertEqual(
            extract_answer_letter("Some reasoning...\nAnswer: C"), "C"
        )

    def test_uses_the_last_letter_when_several_appear(self):
        """A CoT rationale often mentions "option A" while reasoning before
        settling on the real answer at the end - the last one wins."""
        text = "Option A seems plausible, but actually B is correct. Answer: B"
        self.assertEqual(extract_answer_letter(text), "B")

    def test_returns_none_rather_than_guessing(self):
        self.assertIsNone(extract_answer_letter("I am not sure about this one."))

    def test_ignores_surrounding_punctuation(self):
        self.assertEqual(extract_answer_letter("Answer: (D)."), "D")


class SplitTrainValTests(unittest.TestCase):

    def setUp(self):
        self.records = [
            {"question": f"q{i}", "answer": "[HELPFUL]"} for i in range(20)
        ]

    def test_split_sizes_match_the_requested_fraction(self):
        train, val = split_train_val(self.records, val_fraction=0.1, seed=1)
        self.assertEqual(len(val), 2)
        self.assertEqual(len(train), 18)

    def test_train_and_val_are_disjoint_and_complete(self):
        train, val = split_train_val(self.records, val_fraction=0.25, seed=1)
        train_qs = {r["question"] for r in train}
        val_qs = {r["question"] for r in val}
        self.assertEqual(train_qs & val_qs, set())
        self.assertEqual(train_qs | val_qs, {r["question"] for r in self.records})

    def test_deterministic_given_the_same_seed(self):
        a = split_train_val(self.records, seed=7)
        b = split_train_val(self.records, seed=7)
        self.assertEqual(a, b)

    def test_different_seeds_can_differ(self):
        a = split_train_val(self.records, seed=1)
        b = split_train_val(self.records, seed=2)
        self.assertNotEqual(a, b)

    def test_rejects_a_fraction_outside_zero_one(self):
        with self.assertRaises(ConfigError):
            split_train_val(self.records, val_fraction=0.0)
        with self.assertRaises(ConfigError):
            split_train_val(self.records, val_fraction=1.0)

    def test_rejects_a_fraction_that_would_empty_the_training_set(self):
        with self.assertRaises(ConfigError):
            split_train_val([{"question": "q", "answer": "x"}], val_fraction=0.99)


if __name__ == "__main__":
    unittest.main()
