"""Exact-text tests of src/rationale_query/prompt.py and query.py.

The acceptance tests (test_spec_query.py) compare the prompt with white space squeezed, because the paper's PDF does not
show where its line break falls. These tests fix the layout this code actually sends, so a change to it is deliberate.
"""

import unittest

from rationale_query.prompt import RATIONALE_PROMPT, build_rationale_prompt
from rationale_query.query import dense_query, rerank_query
from tests.spec_targets import RATIONALE_PROMPT as PAPERS_PROMPT

# The words of the paper (kept once, in tests/spec_targets.py, with the layout squeezed out); the layout is pinned below.
PAPER_TEXT = PAPERS_PROMPT.removesuffix(" Here is the question:")


class PromptTextTests(unittest.TestCase):
    def test_the_prompt_is_the_papers_text_one_line_break_and_the_question(self):
        self.assertEqual(
            build_rationale_prompt("Which drug? A) x B) y"),
            PAPER_TEXT + "\nHere is the question: Which drug? A) x B) y",
        )

    def test_the_text_before_the_question_is_the_constant(self):
        self.assertEqual(build_rationale_prompt("Q?"), RATIONALE_PROMPT + "Q?")
        self.assertTrue(RATIONALE_PROMPT.endswith("Here is the question: "))

    def test_the_question_loses_surrounding_white_space_and_keeps_its_own_line_breaks(self):
        self.assertEqual(
            build_rationale_prompt("  \nFirst line?\nA) x\nB) y \n\t"), RATIONALE_PROMPT + "First line?\nA) x\nB) y"
        )

    def test_the_words_of_the_paper_are_not_doubled_or_missing_a_space(self):
        self.assertNotIn("  ", RATIONALE_PROMPT)
        self.assertEqual(len(RATIONALE_PROMPT.split("\n")), 2)

    def test_a_question_that_is_not_text_is_a_type_error(self):
        for bad in (None, 5, b"bytes", ["a question"]):
            with self.subTest(question=bad), self.assertRaises(TypeError):
                build_rationale_prompt(bad)


class QueryTextTests(unittest.TestCase):
    def test_a_rationale_that_is_not_text_is_a_type_error(self):
        for bad in (None, 5, b"bytes"):
            with self.subTest(rationale=bad), self.assertRaises(TypeError):
                dense_query(bad)

    def test_the_rerank_text_that_is_not_text_is_a_type_error(self):
        with self.assertRaises(TypeError):
            rerank_query(question=None, rationale="r")
        with self.assertRaises(TypeError):
            rerank_query(question="q", rationale=None, against="rationale")

    def test_the_text_that_is_not_used_may_be_anything(self):
        self.assertEqual(rerank_query(question="q", rationale=None), "q")
        self.assertEqual(rerank_query(question=None, rationale="r", against="rationale"), "r")

    def test_the_unknown_variant_is_named_in_the_message(self):
        with self.assertRaises(ValueError) as caught:
            rerank_query(question="q", rationale="r", against="both")
        self.assertIn("'both'", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
