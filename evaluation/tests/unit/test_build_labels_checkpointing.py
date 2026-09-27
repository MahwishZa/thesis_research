"""Checkpoint/resume/calibration logic in build_labels.py's run_labeling()
and load_checkpoint(). All fakes, no real model, no real retrieval -
isolates the checkpointing behaviour from everything else the CLI does.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.baseline.filter_training.build_labels import (
    BuildLabelsError, _fingerprint, _progress_paths, load_checkpoint,
    run_labeling,
)
from experiments.baseline.filter_training.labeling import RationaleOutcome


class FakeItem:
    def __init__(self, i):
        self.item_id = f"Q-{i:03d}"
        self.question = f"question {i}"
        self.options = {"A": "wrong-a", "B": "right", "C": "wrong-c", "D": "wrong-d"}
        self.answer_letter = "B"

    def rendered_question(self):
        return f"{self.question} A) wrong-a B) right C) wrong-c D) wrong-d"


class FakePassage:
    def __init__(self, i):
        self.text = f"textbook passage {i}"


class FakeIndex:
    def search(self, query_vector, *, top_k):
        return [(0, 1.0)]


class FakeQueryEncoder:
    def encode(self, texts):
        return [[0.0]]


class FakeScorer:
    name = "fake-scorer"

    def __init__(self):
        self.calls = []

    def score(self, question, choices, answer, evidence):
        self.calls.append((question, evidence))
        return RationaleOutcome(correct=(evidence is not None), perplexity=2.0)


def _run(tmp, questions, scorer, **kwargs):
    output = Path(tmp) / "labels.json"
    outcomes = run_labeling(
        questions, FakeIndex(), [FakePassage(0)], FakeQueryEncoder(), scorer,
        output=output,
        fingerprint=_fingerprint(scorer_name=scorer.name, n_questions=len(questions),
                                 seed=42, n_textbook_passages=100),
        **kwargs,
    )
    return output, outcomes


class RunLabelingTests(unittest.TestCase):

    def test_scores_every_question_twice(self):
        with TemporaryDirectory() as tmp:
            questions = [FakeItem(i) for i in range(5)]
            scorer = FakeScorer()
            _output, outcomes = _run(tmp, questions, scorer)
        self.assertEqual(len(outcomes), 5)
        self.assertEqual(len(scorer.calls), 10)  # without + with_evidence per pair

    def test_checkpoint_is_written_incrementally(self):
        with TemporaryDirectory() as tmp:
            questions = [FakeItem(i) for i in range(3)]
            output, _outcomes = _run(tmp, questions, FakeScorer(), checkpoint_every=1)
            outcomes_path, state_path = _progress_paths(output)
            self.assertTrue(outcomes_path.exists())
            self.assertTrue(state_path.exists())
            lines = outcomes_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["n_questions"], 3)

    def test_second_run_without_resume_is_refused(self):
        with TemporaryDirectory() as tmp:
            questions = [FakeItem(i) for i in range(3)]
            output, _ = _run(tmp, questions, FakeScorer())
            # Second call, same output, resume defaults to False.
            with self.assertRaises(BuildLabelsError) as ctx:
                run_labeling(
                    questions, FakeIndex(), [FakePassage(0)], FakeQueryEncoder(),
                    FakeScorer(), output=output,
                    fingerprint=_fingerprint(scorer_name="fake-scorer", n_questions=3,
                                             seed=42, n_textbook_passages=100),
                )
        self.assertIn("--resume", str(ctx.exception))

    def test_resume_skips_already_scored_pairs_and_completes_the_rest(self):
        with TemporaryDirectory() as tmp:
            questions = [FakeItem(i) for i in range(5)]
            output = Path(tmp) / "labels.json"
            fingerprint = _fingerprint(scorer_name="crash-then-resume",
                                       n_questions=5, seed=42,
                                       n_textbook_passages=100)

            class CrashAfterTwoPairs(FakeScorer):
                name = "crash-then-resume"

                def score(self, question, choices, answer, evidence):
                    # 2 calls/pair (without, with_evidence); crash partway
                    # through the 2nd pair's 2nd call - after exactly 2 of
                    # 5 pairs are fully checkpointed.
                    if len(self.calls) >= 3:
                        raise RuntimeError("simulated interruption")
                    return super().score(question, choices, answer, evidence)

            crashing = CrashAfterTwoPairs()
            with self.assertRaises(RuntimeError):
                run_labeling(
                    questions, FakeIndex(), [FakePassage(0)], FakeQueryEncoder(),
                    crashing, output=output, fingerprint=fingerprint,
                    checkpoint_every=1,
                )
            outcomes_path, _state_path = _progress_paths(output)
            self.assertEqual(
                len(outcomes_path.read_text(encoding="utf-8").splitlines()), 1,
                "exactly 1 pair should have completed and been checkpointed "
                "before the simulated crash",
            )

            second_scorer = FakeScorer()
            outcomes = run_labeling(
                questions, FakeIndex(), [FakePassage(0)], FakeQueryEncoder(),
                second_scorer, output=output, fingerprint=fingerprint,
                resume=True, checkpoint_every=1,
            )
        self.assertEqual(len(outcomes), 5)
        self.assertEqual({o.pair_id for o in outcomes},
                         {f"Q-{i:03d}" for i in range(5)})
        # The 1 already-checkpointed pair must not be re-scored.
        self.assertEqual(len(second_scorer.calls), 8)  # 4 remaining x 2 generations

    def test_resume_refuses_on_fingerprint_mismatch(self):
        with TemporaryDirectory() as tmp:
            questions = [FakeItem(i) for i in range(3)]
            output, _ = _run(tmp, questions, FakeScorer())
            with self.assertRaises(BuildLabelsError) as ctx:
                run_labeling(
                    questions, FakeIndex(), [FakePassage(0)], FakeQueryEncoder(),
                    FakeScorer(), output=output,
                    fingerprint=_fingerprint(scorer_name="fake-scorer", n_questions=3,
                                             seed=999, n_textbook_passages=100),  # different seed
                    resume=True,
                )
        self.assertIn("seed", str(ctx.exception))

    def test_resume_refuses_on_scorer_mismatch(self):
        with TemporaryDirectory() as tmp:
            questions = [FakeItem(i) for i in range(3)]
            output, _ = _run(tmp, questions, FakeScorer())
            with self.assertRaises(BuildLabelsError) as ctx:
                run_labeling(
                    questions, FakeIndex(), [FakePassage(0)], FakeQueryEncoder(),
                    FakeScorer(), output=output,
                    fingerprint=_fingerprint(scorer_name="a-different-scorer",
                                             n_questions=3, seed=42,
                                             n_textbook_passages=100),
                    resume=True,
                )
        self.assertIn("scorer_name", str(ctx.exception))


class CalibrationTests(unittest.TestCase):

    def test_calibrate_scores_only_the_first_n(self):
        with TemporaryDirectory() as tmp:
            questions = [FakeItem(i) for i in range(10)]
            scorer = FakeScorer()
            _output, outcomes = _run(tmp, questions, scorer, calibrate_n=3)
        self.assertEqual(len(outcomes), 3)
        self.assertEqual(len(scorer.calls), 6)

    def test_calibrate_writes_no_checkpoint(self):
        with TemporaryDirectory() as tmp:
            questions = [FakeItem(i) for i in range(10)]
            output, _outcomes = _run(tmp, questions, FakeScorer(), calibrate_n=3)
            outcomes_path, state_path = _progress_paths(output)
            self.assertFalse(outcomes_path.exists())
            self.assertFalse(state_path.exists())
            self.assertFalse(output.exists())

    def test_calibrate_is_freely_repeatable(self):
        with TemporaryDirectory() as tmp:
            questions = [FakeItem(i) for i in range(10)]
            output = Path(tmp) / "labels.json"
            for _ in range(3):
                outcomes = run_labeling(
                    questions, FakeIndex(), [FakePassage(0)], FakeQueryEncoder(),
                    FakeScorer(), output=output,
                    fingerprint=_fingerprint(scorer_name="fake-scorer",
                                             n_questions=10, seed=42,
                                             n_textbook_passages=100),
                    calibrate_n=4,
                )
                self.assertEqual(len(outcomes), 4)  # no error, no leftover state


class LoadCheckpointTests(unittest.TestCase):

    def test_no_checkpoint_returns_empty(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "labels.json"
            outcomes, done_ids = load_checkpoint(
                output, expected=_fingerprint(scorer_name="x", n_questions=1,
                                              seed=1, n_textbook_passages=1))
        self.assertEqual(outcomes, [])
        self.assertEqual(done_ids, set())


if __name__ == "__main__":
    unittest.main()
