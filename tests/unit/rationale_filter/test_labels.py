"""rationale_filter.labels beyond the acceptance tests: what counts as a number, the rounding of the quarter, the constants."""

import unittest

from rationale_filter import labels
from tests.spec.targets import DISCARD, HELPFUL, NOT_HELPFUL, TOKEN_HELPFUL, TOKEN_NOT_HELPFUL


def label(correct_without, correct_with, ppl_without, ppl_with, tau):
    return labels.label_pair(
        correct_without=correct_without, correct_with=correct_with, ppl_without=ppl_without, ppl_with=ppl_with, tau=tau
    )


class ConstantTests(unittest.TestCase):
    def test_the_constants_equal_the_ones_the_acceptance_tests_use(self):
        self.assertEqual(
            (labels.HELPFUL, labels.NOT_HELPFUL, labels.DISCARD, labels.TOKEN_HELPFUL, labels.TOKEN_NOT_HELPFUL),
            (HELPFUL, NOT_HELPFUL, DISCARD, TOKEN_HELPFUL, TOKEN_NOT_HELPFUL),
        )

    def test_the_paper_takes_the_top_quarter(self):
        self.assertEqual(labels.TOP_FRACTION, 0.25)


class LabelPairTests(unittest.TestCase):
    def test_whole_numbers_are_numbers(self):
        self.assertEqual(label(True, True, 10, 7, 2), HELPFUL)

    def test_a_correctness_that_is_not_a_bool_is_a_type_error_that_names_the_argument(self):
        for bad in (1, 0, None, "True", 1.0):
            with self.subTest(bad=bad), self.assertRaises(TypeError) as caught:
                label(bad, True, 10.0, 7.0, 2.0)
            self.assertIn("correct_without", str(caught.exception))
            with self.subTest(bad=bad, side="with"), self.assertRaises(TypeError) as caught:
                label(True, bad, 10.0, 7.0, 2.0)
            self.assertIn("correct_with", str(caught.exception))

    def test_a_bool_is_not_a_perplexity_or_a_threshold(self):
        with self.assertRaises(ValueError):
            label(True, True, True, 5.0, 1.0)
        with self.assertRaises(ValueError):
            label(True, True, 5.0, 5.0, True)

    def test_a_bad_perplexity_is_an_error_even_when_the_answer_flipped(self):
        for correct_without, correct_with in ((True, False), (False, True)):
            with self.subTest(flip=(correct_without, correct_with)), self.assertRaises(ValueError):
                label(correct_without, correct_with, float("nan"), 5.0, 1.0)

    def test_the_message_names_the_perplexity_that_is_wrong(self):
        with self.assertRaises(ValueError) as caught:
            label(True, True, 5.0, -1.0, 1.0)
        self.assertIn("ppl_with", str(caught.exception))


class TrainingTokenTests(unittest.TestCase):
    def test_the_message_for_a_discard_says_which_labels_have_a_token(self):
        with self.assertRaises(ValueError) as caught:
            labels.to_training_token(DISCARD)
        self.assertIn("helpful", str(caught.exception))


class ThresholdTests(unittest.TestCase):
    def test_the_quarter_of_30_tenths_is_not_rounded_up_by_float_error(self):
        # 30 * 0.1 is 3.0000000000000004 in floating point; the top tenth of 30 values is 3 values, not 4.
        values = [float(i) for i in range(30)]
        tau = labels.perplexity_threshold(values, top_fraction=0.1)
        self.assertEqual(sum(1 for v in values if v >= tau), 3)

    def test_one_value_is_its_own_threshold(self):
        self.assertEqual(labels.perplexity_threshold([2.5]), 2.5)

    def test_a_negative_pool_gives_a_negative_threshold(self):
        self.assertEqual(labels.perplexity_threshold([-4.0, -3.0, -2.0, -1.0]), -1.0)

    def test_integers_and_a_generator_work(self):
        self.assertEqual(labels.perplexity_threshold(i for i in range(1, 9)), 7.0)

    def test_a_bool_in_the_pool_is_an_error(self):
        with self.assertRaises(ValueError):
            labels.perplexity_threshold([1.0, True, 3.0, 4.0])

    def test_the_threshold_of_a_pool_equals_the_kth_largest(self):
        values = [9.0, 1.0, 7.0, 3.0, 5.0, 8.0, 2.0, 6.0]  # n = 8, top quarter = 2 values: 9 and 8
        self.assertEqual(labels.perplexity_threshold(values), 8.0)

    def test_the_rule_and_the_threshold_agree_at_the_boundary_value(self):
        values = [1.0, 2.0, 3.0, 4.0]
        tau = labels.perplexity_threshold(values)
        self.assertEqual(tau, 4.0)
        self.assertEqual(label(True, True, 14.0, 10.0, tau), HELPFUL)  # a difference of exactly tau counts
        self.assertEqual(label(True, True, 13.0, 10.0, tau), DISCARD)


if __name__ == "__main__":
    unittest.main()
