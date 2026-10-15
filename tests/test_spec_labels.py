"""Acceptance tests for docs/methodology.md section 5 (labels). They skip until src/rationale_filter exists."""

import itertools
import random

from tests.spec_targets import (
    DISCARD,
    HELPFUL,
    NOT_HELPFUL,
    TOKEN_HELPFUL,
    TOKEN_NOT_HELPFUL,
    SpecTestCase,
    resolve,
)

NAN = float("nan")
INF = float("inf")


def label(correct_without, correct_with, ppl_without, ppl_with, tau):
    return resolve("label_pair")(
        correct_without=correct_without,
        correct_with=correct_with,
        ppl_without=ppl_without,
        ppl_with=ppl_with,
        tau=tau,
    )


class LabelRuleTests(SpecTestCase):
    """Figure 2 of the paper: one test per leaf, then the boundary and the order of the rules."""

    requires = ("label_pair",)

    TAU = 2.0
    DROP = (10.0, 7.0)  # perplexity falls by 3, at least tau
    SMALL = (10.0, 9.5)  # falls by 0.5, below tau
    RISE = (10.0, 12.0)  # rises

    def test_both_correct_with_lower_perplexity_is_helpful(self):
        self.assertEqual(label(True, True, *self.DROP, self.TAU), HELPFUL)

    def test_both_correct_without_lower_perplexity_is_discarded(self):
        self.assertEqual(label(True, True, *self.SMALL, self.TAU), DISCARD)
        self.assertEqual(label(True, True, *self.RISE, self.TAU), DISCARD)

    def test_correct_then_wrong_is_not_helpful(self):
        self.assertEqual(label(True, False, *self.DROP, self.TAU), NOT_HELPFUL)
        self.assertEqual(label(True, False, *self.RISE, self.TAU), NOT_HELPFUL)

    def test_wrong_then_correct_is_helpful(self):
        self.assertEqual(label(False, True, *self.DROP, self.TAU), HELPFUL)
        self.assertEqual(label(False, True, *self.RISE, self.TAU), HELPFUL)

    def test_both_wrong_with_lower_perplexity_is_not_helpful(self):
        self.assertEqual(label(False, False, *self.DROP, self.TAU), NOT_HELPFUL)

    def test_both_wrong_without_lower_perplexity_is_discarded(self):
        self.assertEqual(label(False, False, *self.SMALL, self.TAU), DISCARD)
        self.assertEqual(label(False, False, *self.RISE, self.TAU), DISCARD)

    def test_full_truth_table(self):
        perplexities = {True: self.DROP, False: self.SMALL}
        expected = {
            (True, True, True): HELPFUL,
            (True, True, False): DISCARD,
            (True, False, True): NOT_HELPFUL,
            (True, False, False): NOT_HELPFUL,
            (False, True, True): HELPFUL,
            (False, True, False): HELPFUL,
            (False, False, True): NOT_HELPFUL,
            (False, False, False): DISCARD,
        }
        for (correct_without, correct_with, lower), want in expected.items():
            with self.subTest(correct_without=correct_without, correct_with=correct_with, lower=lower):
                self.assertEqual(label(correct_without, correct_with, *perplexities[lower], self.TAU), want)

    def test_a_drop_equal_to_tau_counts_as_lower(self):
        # Equation 3 reads "greater than or equal to tau".
        self.assertEqual(label(True, True, 10.0, 8.0, 2.0), HELPFUL)
        self.assertEqual(label(False, False, 10.0, 8.0, 2.0), NOT_HELPFUL)

    def test_a_drop_just_below_tau_does_not_count(self):
        self.assertEqual(label(True, True, 10.0, 8.001, 2.0), DISCARD)
        self.assertEqual(label(False, False, 10.0, 8.001, 2.0), DISCARD)

    def test_the_difference_is_perplexity_without_minus_perplexity_with(self):
        # A snippet that raises perplexity is never a drop, even when the numbers are swapped by mistake.
        self.assertEqual(label(True, True, 7.0, 10.0, 2.0), DISCARD)
        self.assertEqual(label(False, False, 7.0, 10.0, 2.0), DISCARD)

    def test_answer_flips_outrank_perplexity_and_tau(self):
        for ppl_without, ppl_with, tau in itertools.product((1.5, 10.0, 300.0), (1.2, 10.0, 450.0), (-5.0, 0.0, 2.0, 1e6)):
            with self.subTest(ppl_without=ppl_without, ppl_with=ppl_with, tau=tau):
                self.assertEqual(label(True, False, ppl_without, ppl_with, tau), NOT_HELPFUL)
                self.assertEqual(label(False, True, ppl_without, ppl_with, tau), HELPFUL)

    def test_tau_may_be_negative(self):
        # tau is a quantile of the differentials, which can lie below zero.
        self.assertEqual(label(True, True, 10.0, 10.5, -1.0), HELPFUL)
        self.assertEqual(label(True, True, 10.0, 12.0, -1.0), DISCARD)

    def test_the_result_is_a_plain_string(self):
        for result in (label(True, True, 10.0, 7.0, 2.0), label(True, False, 10.0, 7.0, 2.0), label(False, False, 10.0, 9.5, 2.0)):
            self.assertIsInstance(result, str)


class LabelInputTests(SpecTestCase):
    """A bad input is an error, never a label."""

    requires = ("label_pair",)

    def test_non_finite_or_non_positive_perplexity_is_an_error(self):
        for bad in (NAN, INF, -INF, 0.0, -3.0):
            for correct_without, correct_with in itertools.product((True, False), repeat=2):
                with self.subTest(bad=bad, correct_without=correct_without, correct_with=correct_with):
                    with self.assertRaises(ValueError):
                        label(correct_without, correct_with, bad, 5.0, 1.0)
                    with self.assertRaises(ValueError):
                        label(correct_without, correct_with, 5.0, bad, 1.0)

    def test_non_finite_tau_is_an_error(self):
        for bad in (NAN, INF, -INF):
            with self.subTest(tau=bad), self.assertRaises(ValueError):
                label(True, True, 10.0, 7.0, bad)

    def test_unknown_correctness_is_an_error(self):
        # How an unparseable answer counts is settled upstream (docs/methodology.md, open item); never guessed here.
        for bad in (None, "yes", "no"):
            with self.subTest(value=bad):
                with self.assertRaises((TypeError, ValueError)):
                    label(bad, True, 10.0, 7.0, 2.0)
                with self.assertRaises((TypeError, ValueError)):
                    label(True, bad, 10.0, 7.0, 2.0)


class TrainingTokenTests(SpecTestCase):
    requires = ("to_training_token",)

    def test_labels_map_to_the_two_new_tokens(self):
        to_token = resolve("to_training_token")
        self.assertEqual(to_token(HELPFUL), TOKEN_HELPFUL)
        self.assertEqual(to_token(NOT_HELPFUL), TOKEN_NOT_HELPFUL)

    def test_a_discarded_pair_has_no_training_token(self):
        to_token = resolve("to_training_token")
        for bad in (DISCARD, None, "", "maybe", "HELPFUL", "[HELPFUL]"):
            with self.subTest(label=bad), self.assertRaises((TypeError, ValueError)):
                to_token(bad)


class ThresholdTests(SpecTestCase):
    """tau is the top quarter of the perplexity differentials (Paper p.4)."""

    requires = ("perplexity_threshold", "label_pair")

    @staticmethod
    def at_or_above(values, tau):
        return sum(1 for value in values if value >= tau)

    def test_exactly_a_quarter_when_the_pool_divides_by_four(self):
        threshold = resolve("perplexity_threshold")
        for n in (4, 8, 12, 40, 100, 400):
            values = [1.5 * i - 20.0 for i in range(n)]
            with self.subTest(n=n):
                self.assertEqual(self.at_or_above(values, threshold(values)), n // 4)

    def test_about_a_quarter_for_every_pool_size(self):
        threshold = resolve("perplexity_threshold")
        for n in range(1, 61):
            values = [0.5 * i for i in range(n)]
            count = self.at_or_above(values, threshold(values))
            with self.subTest(n=n):
                self.assertGreaterEqual(count, max(1, n // 4))
                self.assertLessEqual(count, -(-n // 4))

    def test_the_default_fraction_is_one_quarter(self):
        threshold = resolve("perplexity_threshold")
        values = [float(i) for i in range(1, 101)]
        self.assertEqual(threshold(values), threshold(values, top_fraction=0.25))

    def test_other_fractions(self):
        threshold = resolve("perplexity_threshold")
        values = [float(i) for i in range(1, 11)]
        self.assertEqual(self.at_or_above(values, threshold(values, top_fraction=0.5)), 5)
        self.assertEqual(threshold(values, top_fraction=1.0), 1.0)

    def test_the_order_of_the_pool_does_not_matter(self):
        threshold = resolve("perplexity_threshold")
        values = [0.37 * i - 4.0 for i in range(57)]
        shuffled = list(values)
        random.Random(11).shuffle(shuffled)
        self.assertEqual(threshold(values), threshold(shuffled))

    def test_the_pool_is_not_changed(self):
        threshold = resolve("perplexity_threshold")
        values = [5.0, 1.0, 4.0, 2.0, 3.0, 9.0, 7.0, 8.0]
        before = list(values)
        threshold(values)
        self.assertEqual(values, before)

    def test_a_tuple_works(self):
        threshold = resolve("perplexity_threshold")
        self.assertEqual(self.at_or_above(range(8), threshold(tuple(range(8)))), 2)

    def test_ties_at_the_boundary_are_all_kept(self):
        threshold = resolve("perplexity_threshold")
        self.assertEqual(threshold([0.0] * 60 + [1.0] * 40), 1.0)
        self.assertEqual(threshold([3.5] * 7), 3.5)

    def test_bad_input_is_an_error(self):
        threshold = resolve("perplexity_threshold")
        with self.assertRaises(ValueError):
            threshold([])
        for bad in (0.0, -0.25, 1.5, NAN):
            with self.subTest(top_fraction=bad), self.assertRaises(ValueError):
                threshold([1.0, 2.0, 3.0, 4.0], top_fraction=bad)
        for bad in (NAN, INF, -INF):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                threshold([1.0, 2.0, bad, 4.0])

    def test_the_threshold_and_the_rule_make_a_quarter_helpful(self):
        deltas = [float(d) for d in range(1, 101)]
        tau = resolve("perplexity_threshold")(deltas)
        labels = [label(True, True, 50.0 + d, 50.0, tau) for d in deltas]
        self.assertEqual(labels.count(HELPFUL), 25)
        self.assertEqual(labels.count(DISCARD), 75)

