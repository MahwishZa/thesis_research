"""Unit tests for the torch-free parts of run_real_evaluation.py.

The rest of that script (retrieval, the RAG2 checkpoint, the run itself)
needs torch, transformers, the real corpus and a built index - none of
which exist in this sandbox (see docs/current_objectives.md) - so it is
exercised by the student running it for real, not by this suite. What is
tested here is the glue this script adds on top of already-tested modules:
joining the real question files correctly, and the new admitted-recency
metric's arithmetic.
"""

import unittest
from pathlib import Path

from experiments.evaluation.freezing import FrozenCandidate, FrozenItem
from experiments.runners.run_real_evaluation import (
    admitted_recency_by_system, load_usable_questions,
)

QUESTIONS_DIR = Path("experiments/questions")


class LoadUsableQuestionsTests(unittest.TestCase):

    def test_all_split_matches_splits_json_total_usable(self):
        questions = load_usable_questions(QUESTIONS_DIR, "all")
        self.assertEqual(len(questions), 113)

    def test_validation_and_test_partition_all(self):
        validation = load_usable_questions(QUESTIONS_DIR, "validation")
        test = load_usable_questions(QUESTIONS_DIR, "test")
        all_q = load_usable_questions(QUESTIONS_DIR, "all")
        self.assertEqual(len(validation) + len(test), len(all_q))
        self.assertEqual(
            {q["question_id"] for q in validation} | {q["question_id"] for q in test},
            {q["question_id"] for q in all_q},
        )

    def test_every_question_carries_the_fields_freezing_needs(self):
        for q in load_usable_questions(QUESTIONS_DIR, "all"):
            for field in ("question_id", "question", "reference_answer",
                          "reference_source", "reference_date"):
                self.assertIn(field, q)
                self.assertTrue(q[field], f"{field} empty for {q['question_id']}")
            self.assertIn(q["review_decision"], ("ACCEPT", "REVISE"))


class AdmittedRecencyBySystemTests(unittest.TestCase):

    def _item(self, temporal_candidate=True):
        return FrozenItem(
            question_id="Q1", question="q", reference_answer="a",
            reference_source="s", reference_date="2020",
            corpus_snapshot="test@1", temporal_candidate=temporal_candidate,
            candidates=(
                FrozenCandidate(evidence_id="old", text="old", retrieval_rank=1,
                                publication_date="2015-01"),
                FrozenCandidate(evidence_id="new", text="new", retrieval_rank=2,
                                publication_date="2025-11"),
            ),
        )

    def test_distinguishes_stale_from_recent_admission(self):
        item = self._item()
        records = [
            {"question_id": "Q1", "system": "baseline", "status": "ok",
             "admitted_evidence_ids": ["old"]},
            {"question_id": "Q1", "system": "proposed_lambda_1", "status": "ok",
             "admitted_evidence_ids": ["new"]},
        ]
        result = admitted_recency_by_system(
            records, {"Q1": item}, {"Q1": True})
        self.assertEqual(
            result["all_questions"]["baseline"]["admitted_most_recent_rate"], 0.0)
        self.assertEqual(
            result["all_questions"]["proposed_lambda_1"]["admitted_most_recent_rate"],
            1.0)

    def test_skips_error_records(self):
        item = self._item()
        records = [
            {"question_id": "Q1", "system": "baseline", "status": "error",
             "admitted_evidence_ids": []},
        ]
        result = admitted_recency_by_system(records, {"Q1": item}, {"Q1": True})
        self.assertEqual(result["all_questions"], {})

    def test_non_temporal_question_excluded_from_temporal_subgroup(self):
        item = self._item(temporal_candidate=False)
        records = [
            {"question_id": "Q1", "system": "baseline", "status": "ok",
             "admitted_evidence_ids": ["new"]},
        ]
        result = admitted_recency_by_system(records, {"Q1": item}, {"Q1": False})
        self.assertIn("baseline", result["all_questions"])
        self.assertNotIn("baseline", result["temporal_candidate_questions"])


if __name__ == "__main__":
    unittest.main()
