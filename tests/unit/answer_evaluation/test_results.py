"""answer_evaluation.results: accuracy per system, the paired comparison, and the choice of k on validation data."""

import unittest

from answer_evaluation import results
from answer_evaluation.paired_tests import StatsError
from answer_evaluation.records import AnswerRecord, answer_id


def answer(question, arm, k, correct, *, letter="A", status="ok"):
    return AnswerRecord(
        id=answer_id(arm, k, question), question_id=question, arm=arm, k=k, question_sha256="x", status=status,
        letter=letter if status == "ok" else None, correct=correct if status == "ok" else None, error=None if status == "ok" else "boom",
    )


def system(arm, k, flags):
    return [answer(f"q{i}", arm, k, bool(flag)) for i, flag in enumerate(flags)]


class OutcomeTests(unittest.TestCase):
    def test_outcomes_are_one_and_zero_per_question_for_one_system(self):
        found = results.outcomes(system("rag2", 8, [1, 0, 1]) + system("rag2", 4, [0, 0, 0]), "rag2", 8)
        self.assertEqual(found, {"q0": 1, "q1": 0, "q2": 1})

    def test_error_records_are_left_out(self):
        records = system("rag2", 8, [1, 1]) + [answer("q2", "rag2", 8, None, status="error")]
        self.assertEqual(set(results.outcomes(records, "rag2", 8)), {"q0", "q1"})


class TableTests(unittest.TestCase):
    def test_one_row_per_system_sorted_with_the_unreadable_answers_counted_inside_the_wrong_ones(self):
        records = system("rag2", 8, [1, 1, 0, 0]) + system("no_retrieval", 0, [1, 0, 0, 0])
        records.append(answer("q4", "rag2", 8, False, letter=None))
        table = results.accuracy_table(records)
        self.assertEqual([(row["arm"], row["k"]) for row in table], [("no_retrieval", 0), ("rag2", 8)])
        self.assertEqual(table[1], {"arm": "rag2", "k": 8, "n": 5, "correct": 2, "unreadable": 1, "accuracy": 0.4})
        self.assertEqual(table[0]["accuracy"], 0.25)

    def test_no_records_give_an_empty_table(self):
        self.assertEqual(results.accuracy_table([]), [])


class CompareTests(unittest.TestCase):
    def test_two_systems_on_the_same_questions_are_compared(self):
        records = system("rag2", 8, [1] * 60 + [0] * 40) + system("no_retrieval", 0, [1] * 50 + [0] * 50)
        found = results.compare_systems(records, ("no_retrieval", 0), ("rag2", 8), iterations=200)
        self.assertEqual((found["baseline_accuracy"], found["proposed_accuracy"]), (0.5, 0.6))
        self.assertEqual(found["mcnemar"]["proposed_only"], 10)

    def test_a_system_with_a_missing_question_refuses_the_comparison(self):
        records = system("rag2", 8, [1, 1, 1]) + system("no_retrieval", 0, [1, 0, 1])[:2]
        with self.assertRaises(StatsError):
            results.compare_systems(records, ("no_retrieval", 0), ("rag2", 8), iterations=10)

    def test_a_system_without_answers_is_named(self):
        with self.assertRaises(StatsError) as caught:
            results.compare_systems(system("rag2", 8, [1]), ("no_retrieval", 0), ("rag2", 8))
        self.assertIn("no_retrieval", str(caught.exception))


class BestKTests(unittest.TestCase):
    def records(self):
        return system("rag2", 1, [1, 0, 0, 0]) + system("rag2", 4, [1, 1, 0, 0]) + system("rag2", 8, [1, 1, 1, 0]) + system("rag2", 16, [1, 1, 1, 0])

    def test_the_highest_accuracy_wins_and_a_tie_goes_to_the_smaller_k(self):
        found = results.best_k(self.records(), "rag2", [1, 4, 8, 16])
        self.assertEqual((found["k"], found["accuracy"], found["n"]), (8, 0.75, 4))
        self.assertEqual(found["all"], {1: 0.25, 4: 0.5, 8: 0.75, 16: 0.75})

    def test_only_the_candidates_given_are_considered(self):
        self.assertEqual(results.best_k(self.records(), "rag2", [1, 4])["k"], 4)

    def test_values_of_k_that_answered_different_questions_are_refused(self):
        records = system("rag2", 4, [1, 1, 0, 0]) + system("rag2", 8, [1, 1, 1])
        with self.assertRaises(StatsError):
            results.best_k(records, "rag2", [4, 8])

    def test_a_candidate_without_answers_is_named(self):
        with self.assertRaises(StatsError) as caught:
            results.best_k(system("rag2", 4, [1]), "rag2", [4, 32])
        self.assertIn("32", str(caught.exception))

    def test_no_candidate_is_an_error(self):
        with self.assertRaises(StatsError):
            results.best_k(system("rag2", 4, [1]), "rag2", [])


if __name__ == "__main__":
    unittest.main()
