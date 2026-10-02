import unittest

from _archive.rag2_filter_reproduction.filter_training.label_noise_control import (
    derangement, summarize,
)


class DerangementTests(unittest.TestCase):

    def test_no_fixed_points_and_is_a_permutation(self):
        for n in (2, 3, 10, 150):
            p = derangement(n, seed=42)
            self.assertEqual(sorted(p), list(range(n)))
            self.assertTrue(all(i != v for i, v in enumerate(p)))

    def test_deterministic_for_resume(self):
        self.assertEqual(derangement(50, 7), derangement(50, 7))

    def test_needs_two(self):
        with self.assertRaises(ValueError):
            derangement(1, 0)


class SummaryTests(unittest.TestCase):

    def test_counts_and_rates(self):
        rows = [
            dict(without_correct=False, retrieved_correct=True, control_correct=False),
            dict(without_correct=True, retrieved_correct=False, control_correct=False),
            dict(without_correct=True, retrieved_correct=True, control_correct=True),
            dict(without_correct=False, retrieved_correct=False, control_correct=True),
        ]
        s = summarize(rows)
        self.assertEqual((s["retrieved_flip_to_correct"], s["retrieved_flip_to_wrong"]), (1, 1))
        self.assertEqual((s["control_flip_to_correct"], s["control_flip_to_wrong"]), (1, 1))
        self.assertEqual(s["retrieved_flip_rate"], 0.5)
        self.assertEqual(s["accuracy_without"], 0.5)

    def test_empty_refused(self):
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()
