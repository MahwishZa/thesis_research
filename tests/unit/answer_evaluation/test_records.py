"""answer_evaluation.records: the record, its refusals, and the store."""

import json
import unittest

from answer_evaluation import records
from answer_evaluation.records import AnswerRecord, AnswerStore
from tests.support import TempDirTestCase


def record(question="q1", arm="rag2", k=8, **changes):
    values = {
        "id": records.answer_id(arm, k, question), "question_id": question, "arm": arm, "k": k, "question_sha256": "a" * 64,
        "status": "ok", "evidence_count": 3, "text": "the answer is (A)", "letter": "A", "reading": "ok", "correct": True,
    }
    values.update(changes)
    return AnswerRecord(**values)


class RecordTests(unittest.TestCase):
    def test_the_id_holds_arm_k_and_question(self):
        self.assertEqual(records.answer_id("rag2", 16, "medqa:test:5"), "rag2|k16|medqa:test:5")

    def test_a_round_trip(self):
        original = record(finish_reason="stop", prompt_tokens=900, completion_tokens=300, seconds=61.5)
        self.assertEqual(AnswerRecord.from_dict(json.loads(original.to_json())), original)

    def test_refused_dictionaries(self):
        base = json.loads(record().to_json())
        for name, change in {
            "unknown status": {"status": "unknown"},
            "correct is a number": {"correct": 1},
            "correct is missing": {"correct": None},
            "k is text": {"k": "8"},
            "k is a bool": {"k": True},
            "id does not match": {"id": "rag2|k4|q1"},
        }.items():
            with self.subTest(name), self.assertRaises(ValueError):
                AnswerRecord.from_dict({**base, **change})
        with self.assertRaises(TypeError):
            AnswerRecord.from_dict([])

    def test_an_error_record_needs_no_answer(self):
        error = AnswerRecord(id="rag2|k8|q1", question_id="q1", arm="rag2", k=8, question_sha256="a", status="error", error="Boom")
        self.assertEqual(AnswerRecord.from_dict(json.loads(error.to_json())).error, "Boom")


class StoreTests(TempDirTestCase):
    def test_appending_reading_and_the_latest_record_per_id(self):
        store = AnswerStore(self.dir / "run")
        store.bind({"dataset": "medqa"}, records.RELEVANT)
        store.load()
        store.append(AnswerRecord(id="rag2|k8|q1", question_id="q1", arm="rag2", k=8, question_sha256="a", status="error", error="x"))
        store.append(record())
        loaded = AnswerStore(self.dir / "run").load()
        self.assertEqual(list(loaded), ["rag2|k8|q1"])
        self.assertEqual(loaded["rag2|k8|q1"].status, "ok")

    def test_the_relevant_settings(self):
        for name in ("dataset", "split", "generator", "answer_prompt_sha256", "reader_pattern"):
            self.assertIn(name, records.RELEVANT)


if __name__ == "__main__":
    unittest.main()
