"""answer_evaluation.paired_tests: the exact McNemar test, the paired bootstrap, Holm; the archived tests plus exact values."""

import math
import random
import unittest

from answer_evaluation import paired_tests as st


class McNemarTests(unittest.TestCase):
    def test_only_the_questions_where_the_systems_differ_count(self):
        baseline = {"a": 1, "b": 1, "c": 0, "d": 1}
        proposed = {"a": 0, "b": 1, "c": 0, "d": 0}
        result = st.mcnemar(baseline, proposed)
        self.assertEqual((result.baseline_only, result.proposed_only), (2, 0))
        self.assertEqual((result.both, result.neither, result.discordant, result.n_pairs), (1, 1, 2, 4))

    def test_no_difference_gives_p_one(self):
        same = {"a": 1, "b": 0}
        self.assertEqual(st.mcnemar(same, dict(same)).p_value, 1.0)

    def test_mismatched_question_sets_are_refused(self):
        with self.assertRaises(st.StatsError):
            st.mcnemar({"a": 1}, {"b": 1})
        with self.assertRaises(st.StatsError):
            st.mcnemar({"a": 1, "b": 0}, {"a": 1})
        with self.assertRaises(st.StatsError):
            st.mcnemar({}, {})

    def test_an_outcome_that_is_not_zero_or_one_is_refused(self):
        for bad in (2, -1, True, 0.5, None):
            with self.subTest(bad=bad), self.assertRaises(st.StatsError):
                st.mcnemar({"a": bad}, {"a": 1})

    def test_a_large_consistent_difference_is_detected(self):
        baseline = {f"q{i}": 1 if i < 30 else 0 for i in range(100)}
        proposed = {f"q{i}": 1 if i < 15 else 0 for i in range(100)}
        self.assertLess(st.mcnemar(baseline, proposed).p_value, 0.01)

    def test_exact_values(self):
        # 10 questions only the proposed system gets right: 2 * (1/2)**10.
        baseline = {f"q{i}": 0 for i in range(10)}
        proposed = {f"q{i}": 1 for i in range(10)}
        self.assertAlmostEqual(st.mcnemar(baseline, proposed).p_value, 2 / 1024)
        # 3 against 17: twice the lower tail, (1 + 20 + 190 + 1140) / 2**20 each.
        baseline = {f"q{i}": 1 if i < 17 else 0 for i in range(20)}
        proposed = {f"q{i}": 0 if i < 17 else 1 for i in range(20)}
        self.assertAlmostEqual(st.mcnemar(baseline, proposed).p_value, 2 * 1351 / 2**20)

    def test_the_result_is_symmetric_in_the_two_systems(self):
        baseline = {f"q{i}": int(i % 3 == 0) for i in range(60)}
        proposed = {f"q{i}": int(i % 5 == 0) for i in range(60)}
        self.assertEqual(st.mcnemar(baseline, proposed).p_value, st.mcnemar(proposed, baseline).p_value)

    def test_the_dictionary_form_rounds_the_p_value(self):
        data = st.mcnemar({"a": 1, "b": 0, "c": 1}, {"a": 0, "b": 1, "c": 1}).to_dict()
        self.assertEqual(data["discordant"], 2)
        self.assertEqual(data["p_value"], 1.0)


class BinomialTests(unittest.TestCase):
    def test_the_two_tails_and_the_middle(self):
        self.assertAlmostEqual(st.binomial_two_sided_p(0, 10), 2 / 1024)
        self.assertAlmostEqual(st.binomial_two_sided_p(10, 10), 2 / 1024)
        self.assertEqual(st.binomial_two_sided_p(5, 10), 1.0)
        self.assertEqual(st.binomial_two_sided_p(0, 0), 1.0)

    def test_the_log_space_sum_agrees_with_the_direct_sum(self):
        for n, k in ((20, 3), (200, 80), (1000, 400), (1000, 500), (1000, 0)):
            with self.subTest(n=n, k=k):
                direct, logged = st._two_sided_direct(k, n, 0.5), st._two_sided_log_space(k, n, 0.5)
                self.assertTrue(math.isclose(direct, logged, rel_tol=1e-6), (direct, logged))

    def test_a_count_beyond_the_direct_limit_does_not_overflow(self):
        # math.comb(2600, 1300) * 0.5**2600 would be an overflow or an underflow in floating point.
        self.assertGreater(st.binomial_two_sided_p(1300, 2600), 0.999)
        self.assertLess(st.binomial_two_sided_p(1000, 2600), 1e-20)
        self.assertEqual(st.binomial_two_sided_p(1300, 2600), st.binomial_two_sided_p(1300, 2600))

    def test_bad_arguments(self):
        for successes, n in ((11, 10), (-1, 10), (True, 10), (1, -1), (1.5, 10), (1, True)):
            with self.subTest(successes=successes, n=n), self.assertRaises(st.StatsError):
                st.binomial_two_sided_p(successes, n)
        for p in (0, 1, -0.1, 1.5):
            with self.subTest(p=p), self.assertRaises(st.StatsError):
                st.binomial_two_sided_p(1, 10, p)


class BootstrapTests(unittest.TestCase):
    def test_the_interval_brackets_the_point_estimate(self):
        baseline = {f"q{i}": 1.0 if i < 40 else 0.0 for i in range(100)}
        proposed = {f"q{i}": 1.0 if i < 20 else 0.0 for i in range(100)}
        ci = st.paired_bootstrap_ci(baseline, proposed, iterations=2000, seed="t")
        self.assertAlmostEqual(ci["difference"], -0.20, places=6)
        self.assertLessEqual(ci["ci_low"], ci["difference"])
        self.assertGreaterEqual(ci["ci_high"], ci["difference"])
        self.assertEqual((ci["n_pairs"], ci["iterations"], ci["confidence"]), (100, 2000, 0.95))

    def test_the_interval_takes_the_2_5_and_97_5_percentiles_of_the_resampled_differences(self):
        baseline = {f"q{i}": float(i % 2) for i in range(40)}
        proposed = {f"q{i}": float((i * 7) % 3 == 0) for i in range(40)}
        ci = st.paired_bootstrap_ci(baseline, proposed, iterations=1000, seed="percentiles")
        differences = [proposed[k] - baseline[k] for k in sorted(baseline)]
        rng = random.Random("percentiles")
        means = sorted(math.fsum(differences[rng.randrange(40)] for _ in range(40)) / 40 for _ in range(1000))
        self.assertEqual((ci["ci_low"], ci["ci_high"]), (round(means[25], 6), round(means[975], 6)))

    def test_the_same_seed_gives_the_same_interval_and_another_seed_another(self):
        baseline = {f"q{i}": float(i % 2) for i in range(50)}
        proposed = {f"q{i}": float(i % 3 == 0) for i in range(50)}
        one = st.paired_bootstrap_ci(baseline, proposed, iterations=500, seed="a")
        self.assertEqual(one, st.paired_bootstrap_ci(baseline, proposed, iterations=500, seed="a"))
        self.assertNotEqual(one, st.paired_bootstrap_ci(baseline, proposed, iterations=500, seed="b"))

    def test_the_pairing_is_kept(self):
        # The systems differ on every question, always in the same direction: the paired interval has no width.
        baseline = {f"q{i}": 0.0 for i in range(30)}
        proposed = {f"q{i}": 1.0 for i in range(30)}
        ci = st.paired_bootstrap_ci(baseline, proposed, iterations=300, seed="p")
        self.assertEqual((ci["ci_low"], ci["ci_high"]), (1.0, 1.0))

    def test_bad_arguments(self):
        data = {"a": 1.0, "b": 0.0}
        for options in ({"confidence": 0}, {"confidence": 1}, {"iterations": 0}, {"iterations": 1.5}):
            with self.subTest(options=options), self.assertRaises(st.StatsError):
                st.paired_bootstrap_ci(data, data, **options)
        with self.assertRaises(st.StatsError):
            st.paired_bootstrap_ci({"a": 1.0}, {"b": 1.0})


class HolmTests(unittest.TestCase):
    def test_two_p_values_are_exact(self):
        self.assertEqual(st.holm({"first": 0.01, "second": 0.04}), {"first": 0.02, "second": 0.04})
        self.assertEqual(st.holm({"only": 0.3}), {"only": 0.3})
        self.assertEqual(st.holm({"a": 0.9, "b": 0.8}), {"a": 1.0, "b": 1.0})

    def test_the_adjusted_values_never_decrease_with_the_p_value(self):
        adjusted = st.holm({"a": 0.03, "b": 0.01, "c": 0.20})
        self.assertAlmostEqual(adjusted["b"], 0.03)
        self.assertGreaterEqual(adjusted["a"], adjusted["b"])
        self.assertGreaterEqual(adjusted["c"], adjusted["a"])
        self.assertLessEqual(max(adjusted.values()), 1.0)

    def test_a_textbook_example(self):
        # p = 0.01, 0.04, 0.03, 0.005 with m = 4: sorted 0.005, 0.01, 0.03, 0.04 -> 0.02, 0.03, 0.06, 0.06 (running maximum).
        self.assertEqual(
            st.holm({"p1": 0.01, "p2": 0.04, "p3": 0.03, "p4": 0.005}), {"p1": 0.03, "p2": 0.06, "p3": 0.06, "p4": 0.02}
        )

    def test_a_p_value_outside_zero_to_one_is_refused(self):
        for bad in (-0.1, 1.1, True, "0.1", None):
            with self.subTest(bad=bad), self.assertRaises(st.StatsError):
                st.holm({"a": bad})

    def test_no_comparisons_give_nothing(self):
        self.assertEqual(st.holm({}), {})


class CompareTests(unittest.TestCase):
    def test_one_comparison_gives_accuracies_the_test_and_the_interval(self):
        baseline = {f"q{i}": int(i < 50) for i in range(100)}
        proposed = {f"q{i}": int(i < 60) for i in range(100)}
        result = st.compare(baseline, proposed, iterations=200, seed="c")
        self.assertEqual((result["baseline_accuracy"], result["proposed_accuracy"]), (0.5, 0.6))
        self.assertEqual(result["mcnemar"]["proposed_only"], 10)
        self.assertEqual(result["difference"]["difference"], 0.1)


if __name__ == "__main__":
    unittest.main()
