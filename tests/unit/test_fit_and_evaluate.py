"""Unit test for the torch-free part of fit_and_evaluate.py: the grid
sweep itself. Retrieval, the RAG2 checkpoint and the final test-split run
need torch/the real corpus (exercised by the student's real run, not
here) - what's tested here is that the sweep logic correctly recovers a
strong configuration when one exists, using run_end_to_end's own designed
fixture (a stale-vs-recent contrast with a known best answer).
"""

import unittest

from experiments.runners.fit_and_evaluate import fit_on_validation
from experiments.runners.run_end_to_end import make_fixture_items
from experiments.runners.run_real_evaluation import _extractive_answer
from systems.interfaces.generator import CallableGenerator


class FitOnValidationTests(unittest.TestCase):

    def test_recovers_a_strong_config_on_the_designed_fixture(self):
        items = make_fixture_items(10)
        generator = CallableGenerator(_extractive_answer)
        result = fit_on_validation(items, generator, budget=1)

        self.assertEqual(result["grid_size"], 120)
        # The fixture is built so a sufficiently temporal-weighted policy
        # recovers the current passage every time - the sweep must find
        # that, not settle for a mediocre cell.
        self.assertGreater(result["fitted_validation_token_f1"], 0.9)
        self.assertGreater(result["fitted_lambda"], 0)

    def test_grid_contains_every_declared_combination(self):
        items = make_fixture_items(3)
        generator = CallableGenerator(_extractive_answer)
        result = fit_on_validation(items, generator, budget=1)

        seen = {(g["theta"], g["half_life_days"], g["lambda"])
                for g in result["grid"]}
        self.assertEqual(len(seen), result["grid_size"])


if __name__ == "__main__":
    unittest.main()
