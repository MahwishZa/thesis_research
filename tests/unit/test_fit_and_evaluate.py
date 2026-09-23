"""Unit test for the torch-free part of fit_and_evaluate.py: the grid
sweep itself. Retrieval, the RAG2 checkpoint and the final test-split run
need torch/the real corpus (exercised by the student's real run, not
here) - what's tested here is that the sweep logic correctly recovers a
strong configuration when one exists, using run_end_to_end's own designed
fixture (a stale-vs-recent contrast with a known best answer).
"""

import unittest
from datetime import date

from experiments.runners.fit_and_evaluate import fit_on_validation
from experiments.runners.run_end_to_end import QUESTION_DATE, make_fixture_items
from experiments.runners.run_real_evaluation import _extractive_answer
from systems.interfaces.generator import CallableGenerator


class FitOnValidationTests(unittest.TestCase):

    def test_recovers_a_strong_config_on_the_designed_fixture(self):
        items = make_fixture_items(10)
        generator = CallableGenerator(_extractive_answer)
        result = fit_on_validation(items, generator, budget=1,
                                   question_date=QUESTION_DATE)

        self.assertGreater(result["grid_size"], 0)
        # The fixture is built so a sufficiently temporal-weighted policy
        # recovers the current passage every time - the sweep must find
        # that, not settle for a mediocre cell.
        self.assertGreater(result["fitted_validation_token_f1"], 0.9)
        self.assertGreater(result["fitted_lambda"], 0)
        # And it must find it by actually admitting something different
        # from plain relevance ranking, not by coincidence.
        self.assertTrue(result["any_config_diverges_from_relevance_only"])

    def test_grid_contains_every_declared_combination(self):
        items = make_fixture_items(3)
        generator = CallableGenerator(_extractive_answer)
        result = fit_on_validation(items, generator, budget=1,
                                   question_date=QUESTION_DATE)

        seen = {(g["theta"], g["half_life_days"], g["lambda"])
                for g in result["grid"]}
        self.assertEqual(len(seen), result["grid_size"])

    def test_a_different_question_date_is_actually_used(self):
        """Regression guard for the bug this module fixed: earlier real-data
        runs silently reused run_end_to_end.py's fixture QUESTION_DATE.
        Moving the reference date far enough away must change T(s) for
        every dated candidate and therefore can change what gets admitted -
        this asserts the parameter is wired through, not decorative."""
        items = make_fixture_items(5)
        generator = CallableGenerator(_extractive_answer)

        near = fit_on_validation(items, generator, budget=1,
                                 question_date=date(2025, 11, 1))
        far = fit_on_validation(items, generator, budget=1,
                                question_date=date(2035, 11, 1))

        # Same grid of (theta, half_life, lambda) cells either way; only the
        # reference date differs, so at least one cell's token_f1 must move.
        near_scores = {(g["theta"], g["half_life_days"], g["lambda"]): g["token_f1"]
                       for g in near["grid"]}
        far_scores = {(g["theta"], g["half_life_days"], g["lambda"]): g["token_f1"]
                      for g in far["grid"]}
        self.assertNotEqual(near_scores, far_scores)


if __name__ == "__main__":
    unittest.main()
