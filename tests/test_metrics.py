import unittest

from src import metrics


class MetricTests(unittest.TestCase):
    def test_wilson_known_value(self):
        low, high = metrics.wilson(8, 10)
        self.assertAlmostEqual(low, 0.4902, places=3)
        self.assertAlmostEqual(high, 0.9433, places=3)
        self.assertEqual(metrics.wilson(0, 0), (0.0, 0.0))

    def test_accuracy_counts_unreadable_as_wrong(self):
        m = metrics.summarize(["a", "a", "b", "b"], ["a", "b", "b", None])
        self.assertEqual((m["n"], m["correct"], m["unreadable"]), (4, 2, 1))
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertEqual(m["accuracy_ci95"], metrics.wilson(2, 4))
        self.assertEqual(set(m), {"n", "correct", "unreadable", "accuracy", "accuracy_ci95"})  # accuracy only
        self.assertEqual(metrics.summarize([], [])["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
