import unittest

from src import metrics


class MetricTests(unittest.TestCase):
    def test_wilson_known_value(self):
        low, high = metrics.wilson(8, 10)
        self.assertAlmostEqual(low, 0.4902, places=3)
        self.assertAlmostEqual(high, 0.9433, places=3)
        self.assertEqual(metrics.wilson(0, 0), (0.0, 0.0))

    def test_summary_counts_unreadable_as_wrong(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", None]
        m = metrics.summarize(y_true, y_pred, ["a", "b"], n_boot=200)
        self.assertEqual((m["n"], m["correct"], m["unreadable"]), (4, 2, 1))
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(m["per_class"]["b"]["precision"], 0.5)
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + 0.5) / 2)
        self.assertEqual(m["confusion"]["b"][metrics.UNREADABLE], 1)
        self.assertEqual(m["baselines"]["chance_accuracy"], 0.5)
        self.assertEqual(metrics.summarize(["b", "a", "a", "b"], ["a"] * 4, ["a", "b"])["baselines"]["majority_classes"],
                         ["a", "b"])  # a tie names every largest class, in sorted order

    def test_cohen_kappa_known_values(self):
        # observed 4/5, chance (3*2 + 2*3)/25 = 12/25: kappa = (0.8 - 0.48) / 0.52 = 8/13 (Cohen 1960; same as scikit-learn)
        self.assertAlmostEqual(metrics.cohen_kappa(["a", "a", "a", "b", "b"], ["a", "a", "b", "b", "b"]), 8 / 13)
        # an unreadable answer is its own category: observed 2/4, chance (2*1 + 2*2)/16 = 6/16, kappa = 0.125/0.625
        self.assertAlmostEqual(metrics.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "b", None]), 0.2)
        self.assertEqual(metrics.cohen_kappa(["a", "b"], ["a", "b"]), 1.0)
        self.assertLess(metrics.cohen_kappa(["a", "b"], ["b", "a"]), 0)  # worse than chance
        self.assertIsNone(metrics.cohen_kappa(["a", "a"], ["a", "a"]))  # one label, one answer: not defined
        self.assertIsNone(metrics.cohen_kappa([], []))
        m = metrics.summarize(["a", "a", "b", "b"], ["a", "b", "b", None], ["a", "b"], n_boot=200)
        self.assertAlmostEqual(m["cohen_kappa"], 0.2)
        low, high = m["cohen_kappa_ci95"]
        self.assertLessEqual(low, high)

    def test_bootstrap_is_seeded(self):
        args = (["a", "b", "b", "a", "a"], ["a", "b", "a", None, "a"], ["a", "b"])
        self.assertEqual(metrics.summarize(*args, n_boot=300, seed=1), metrics.summarize(*args, n_boot=300, seed=1))


if __name__ == "__main__":
    unittest.main()
