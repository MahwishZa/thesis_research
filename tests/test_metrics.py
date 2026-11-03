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

    def test_bootstrap_is_seeded(self):
        args = (["a", "b", "b", "a", "a"], ["a", "b", "a", None, "a"], ["a", "b"])
        self.assertEqual(metrics.summarize(*args, n_boot=300, seed=1), metrics.summarize(*args, n_boot=300, seed=1))


if __name__ == "__main__":
    unittest.main()
