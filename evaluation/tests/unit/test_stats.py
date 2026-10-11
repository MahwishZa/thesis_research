"""Paired statistics (``evaluation/stats.py``): the exact McNemar test, the paired bootstrap interval, Holm."""

import unittest

from evaluation import stats as st


class StatsTests(unittest.TestCase):

    def test_mcnemar_uses_only_discordant_pairs(self):
        b = {"a": 1, "b": 1, "c": 0, "d": 1}
        p = {"a": 0, "b": 1, "c": 0, "d": 0}
        r = st.mcnemar(b, p)
        self.assertEqual((r.baseline_only, r.proposed_only), (2, 0))
        self.assertEqual((r.both, r.neither), (1, 1))
        self.assertEqual(r.discordant, 2)

    def test_no_discordant_pairs_gives_p_one(self):
        same = {"a": 1, "b": 0}
        self.assertEqual(st.mcnemar(same, dict(same)).p_value, 1.0)

    def test_mismatched_question_sets_refused(self):
        with self.assertRaises(st.StatsError):
            st.mcnemar({"a": 1}, {"b": 1})

    def test_large_consistent_difference_is_detected(self):
        b = {f"q{i}": 1 if i < 30 else 0 for i in range(100)}
        p = {f"q{i}": 1 if i < 15 else 0 for i in range(100)}
        self.assertLess(st.mcnemar(b, p).p_value, 0.01)

    def test_bootstrap_ci_brackets_the_point_estimate(self):
        b = {f"q{i}": 1.0 if i < 40 else 0.0 for i in range(100)}
        p = {f"q{i}": 1.0 if i < 20 else 0.0 for i in range(100)}
        ci = st.paired_bootstrap_ci(b, p, iterations=2000, seed="t")
        self.assertAlmostEqual(ci["difference"], -0.20, places=6)
        self.assertLessEqual(ci["ci_low"], ci["difference"])
        self.assertGreaterEqual(ci["ci_high"], ci["difference"])

    def test_holm_of_two_p_values_is_exact(self):
        self.assertEqual(st.holm({"first": 0.01, "second": 0.04}), {"first": 0.02, "second": 0.04})     # 2 x 0.01, then max(0.02, 1 x 0.04)
        self.assertEqual(st.holm({"only": 0.3}), {"only": 0.3})
        self.assertEqual(st.holm({"a": 0.9, "b": 0.8}), {"a": 1.0, "b": 1.0})                         # adjusted values are capped at 1

    def test_holm_orders_by_p_value_and_never_decreases(self):
        adj = st.holm({"a": 0.03, "b": 0.01, "c": 0.20})
        self.assertAlmostEqual(adj["b"], 0.03)
        self.assertGreaterEqual(adj["a"], adj["b"])
        self.assertGreaterEqual(adj["c"], adj["a"])
        self.assertLessEqual(max(adj.values()), 1.0)

    def test_exact_binomial_p_is_two_sided_and_symmetric(self):
        self.assertAlmostEqual(st.binomial_two_sided_p(0, 10), 2 / 1024)
        self.assertAlmostEqual(st.binomial_two_sided_p(10, 10), 2 / 1024)
        self.assertEqual(st.binomial_two_sided_p(5, 10), 1.0)
        self.assertEqual(st.binomial_two_sided_p(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
