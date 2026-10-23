"""rationale_filter.inference beyond the acceptance tests: types, empty parts, and the constant."""

import unittest

from rationale_filter import inference
from tests.spec.targets import FILTER_INSTRUCTION


class BuildInputTests(unittest.TestCase):
    def test_the_instruction_equals_the_one_the_acceptance_tests_check(self):
        self.assertEqual(inference.FILTER_INSTRUCTION, FILTER_INSTRUCTION)

    def test_an_empty_or_blank_part_is_an_error_that_names_it(self):
        for blank in ("", "   ", "\n\t"):
            with self.subTest(blank=blank):
                with self.assertRaises(ValueError) as caught:
                    inference.build_filter_input(evidence=blank, question="Which?")
                self.assertIn("evidence", str(caught.exception))
                with self.assertRaises(ValueError) as caught:
                    inference.build_filter_input(evidence="Some text.", question=blank)
                self.assertIn("question", str(caught.exception))

    def test_a_part_that_is_not_text_is_a_type_error(self):
        for bad in (None, 5, b"x", ["a"]):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                inference.build_filter_input(evidence=bad, question="Which?")

    def test_the_released_sample_has_this_shape(self):
        # The first and last lines of a released example: the instruction, then Evidence:, then Question:.
        text = inference.build_filter_input(evidence="hours after delivery.", question="Which? A) one B) two")
        lines = text.split("\n")
        self.assertEqual(lines[0], FILTER_INSTRUCTION)
        self.assertEqual(lines[1], "")
        self.assertTrue(lines[2].startswith("Evidence: "))
        self.assertEqual(lines[3], "")
        self.assertTrue(lines[4].startswith("Question: "))


class DecideTests(unittest.TestCase):
    def test_whole_number_logits_work(self):
        self.assertIs(inference.decide_helpful(logit_helpful=2, logit_not_helpful=1), True)
        self.assertIs(inference.decide_helpful(logit_helpful=1, logit_not_helpful=2), False)

    def test_a_bool_or_a_string_is_not_a_logit(self):
        for bad in (True, "1.0", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                inference.decide_helpful(logit_helpful=bad, logit_not_helpful=0.0)

    def test_the_result_is_a_bool_not_an_integer(self):
        self.assertIs(type(inference.decide_helpful(logit_helpful=3.0, logit_not_helpful=1.0)), bool)


class JudgeTests(unittest.TestCase):
    def test_an_empty_snippet_is_an_error_not_a_default(self):
        with self.assertRaises(ValueError):
            inference.judge_snippets(question="Which?", snippets=["fine", "  "], judge=lambda text: True)

    def test_a_judge_is_not_called_after_an_error(self):
        calls = []
        with self.assertRaises(ValueError):
            inference.judge_snippets(question="Which?", snippets=["fine", ""], judge=lambda text: calls.append(text) or True)
        self.assertEqual(len(calls), 1)  # the first snippet was judged; the empty one stopped the loop


class KeepTests(unittest.TestCase):
    def test_a_decision_that_is_not_a_bool_is_an_error(self):
        for bad in (1, 0, None, "yes"):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                inference.keep_helpful(["a"], [bad])

    def test_the_length_error_says_both_lengths(self):
        with self.assertRaises(ValueError) as caught:
            inference.keep_helpful(["a", "b"], [True])
        self.assertIn("2 items", str(caught.exception))
        self.assertIn("1 decisions", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
