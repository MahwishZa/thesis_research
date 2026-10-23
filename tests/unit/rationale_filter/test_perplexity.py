"""rationale_filter.perplexity: the formula, what it refuses, and the pairing of the two prompts."""

import math
import unittest

from rationale_filter import perplexity


class FormulaTests(unittest.TestCase):
    def test_it_is_exp_of_the_negative_mean_log_probability(self):
        self.assertAlmostEqual(perplexity.perplexity([math.log(0.5), math.log(0.25)]), math.exp(-(math.log(0.5) + math.log(0.25)) / 2))
        self.assertAlmostEqual(perplexity.perplexity([math.log(0.5), math.log(0.25)]), 2 ** 1.5)

    def test_a_certain_text_has_perplexity_one(self):
        self.assertEqual(perplexity.perplexity([0.0, 0.0, 0.0]), 1.0)

    def test_one_token_over_a_vocabulary_of_n_is_n(self):
        self.assertAlmostEqual(perplexity.perplexity([math.log(1 / 128256)]), 128256.0, places=3)

    def test_the_length_is_the_mean_not_the_sum(self):
        one = perplexity.perplexity([math.log(0.1)])
        many = perplexity.perplexity([math.log(0.1)] * 50)
        self.assertAlmostEqual(one, many)

    def test_refused_input(self):
        for bad in ([], [0.5], [float("nan")], [float("inf")], [-float("inf")], [-1.0, True], [-1.0, "x"], [None]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                perplexity.perplexity(bad)

    def test_the_message_names_the_token(self):
        with self.assertRaises(ValueError) as caught:
            perplexity.perplexity([-1.0, -2.0, 0.3])
        self.assertIn("token 2", str(caught.exception))

    def test_a_rounding_error_just_above_zero_is_accepted(self):
        self.assertAlmostEqual(perplexity.perplexity([1e-12, -1.0]), math.exp(0.5), places=6)


class FakeScorer:
    """Log-probabilities set by the test: a table of (prompt, text) to a list."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def token_logprobs(self, user_message, completion):
        self.calls.append((user_message, completion))
        return self.table[user_message]


class PairingTests(unittest.TestCase):
    def test_the_same_text_is_scored_under_both_prompts_in_order(self):
        scorer = FakeScorer({"plain": [-1.0, -1.0], "with snippet": [-0.5, -0.5]})
        without, with_snippet = perplexity.rationale_perplexities(
            scorer, prompt_without="plain", prompt_with="with snippet", rationale="the rationale"
        )
        self.assertEqual(scorer.calls, [("plain", "the rationale"), ("with snippet", "the rationale")])
        self.assertAlmostEqual(without, math.e)
        self.assertAlmostEqual(with_snippet, math.exp(0.5))
        self.assertGreater(without - with_snippet, 0)  # a snippet that helps lowers the perplexity

    def test_scorings_of_different_lengths_are_refused(self):
        scorer = FakeScorer({"a": [-1.0, -1.0], "b": [-1.0]})
        with self.assertRaises(ValueError) as caught:
            perplexity.rationale_perplexities(scorer, prompt_without="a", prompt_with="b", rationale="text")
        self.assertIn("2 tokens", str(caught.exception))

    def test_an_empty_rationale_is_refused_before_any_scoring(self):
        scorer = FakeScorer({})
        for bad in ("", "  \n", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                perplexity.rationale_perplexities(scorer, prompt_without="a", prompt_with="b", rationale=bad)
        self.assertEqual(scorer.calls, [])


if __name__ == "__main__":
    unittest.main()
