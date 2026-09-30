"""Rank statistics used by diagnose_labels (numpy only)."""

import unittest

from experiments.baseline.filter_training.diagnose_labels import auc, spearman


class RankStatTests(unittest.TestCase):

    def test_perfect_separation_is_auc_one(self):
        self.assertEqual(auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]), 1.0)

    def test_inverted_is_zero_and_constant_scores_are_half(self):
        self.assertEqual(auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]), 0.0)
        self.assertEqual(auc([1, 1, 1, 1], [False, True, False, True]), 0.5)

    def test_single_class_is_nan(self):
        self.assertNotEqual(auc([1, 2], [True, True]), auc([1, 2], [True, True]))

    def test_spearman_monotone_and_anti(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 400]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


if __name__ == "__main__":
    unittest.main()
