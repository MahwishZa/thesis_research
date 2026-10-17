"""Tests of src/balanced_retrieval/records.py: the resumable record of each question's evidence."""

import json
import unittest

from balanced_retrieval.records import RECORDS_FILE, RetrievalRecord, RetrievalStore
from rationale_query.batch import plan
from rationale_query.store import ConfigMismatch, Question, question_hash
from tests.support import TempDirTestCase

EVIDENCE = [{"corpus": "pmc", "doc_key": "PMC1", "chunk_index": 0, "row": 3, "dense_score": 1.5, "rerank_score": 2.5, "text": "plaque β"}]


def record(identifier="q1", **changes):
    fields = {"id": identifier, "question_sha256": "aa", "rationale_sha256": "bb", "pool_size": 8, "evidence": EVIDENCE}
    fields.update(changes)
    return RetrievalRecord(**fields)


class RecordTests(unittest.TestCase):
    def test_a_record_survives_json(self):
        original = record()
        self.assertEqual(RetrievalRecord.from_dict(json.loads(original.to_json())), original)

    def test_the_line_is_compact_and_keeps_unicode(self):
        line = record().to_json()
        self.assertNotIn("\n", line)
        self.assertIn("plaque β", line)

    def test_a_record_always_has_the_status_ok(self):
        self.assertEqual(record().status, "ok")

    def test_what_is_not_a_record_is_refused(self):
        for bad in ({}, {"id": 5, "status": "ok", "evidence": []}, {"id": "q", "status": "error", "evidence": []}, {"id": "q", "status": "ok"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RetrievalRecord.from_dict(bad)
        with self.assertRaises(TypeError):
            RetrievalRecord.from_dict([1, 2])


class RetrievalStoreTests(TempDirTestCase):
    def store(self):
        return RetrievalStore(self.dir / "out")

    def test_records_are_appended_and_read_back_by_id(self):
        store = self.store()
        store.bind({"top_k": 4}, ["top_k"])
        store.append(record("q1"))
        store.append(record("q2", pool_size=5))
        loaded = self.store().load()
        self.assertEqual(sorted(loaded), ["q1", "q2"])
        self.assertEqual(loaded["q2"].pool_size, 5)
        self.assertEqual((self.dir / "out" / RECORDS_FILE).read_text(encoding="utf-8").count("\n"), 2)

    def test_other_settings_are_refused_for_the_same_folder(self):
        self.store().bind({"top_k": 4, "rerank_against": "question"}, ["top_k", "rerank_against"])
        with self.assertRaises(ConfigMismatch):
            self.store().bind({"top_k": 4, "rerank_against": "rationale"}, ["top_k", "rerank_against"])

    def test_a_cut_off_last_line_is_ignored_and_then_repaired(self):
        store = self.store()
        store.bind({"top_k": 4}, ["top_k"])
        store.append(record("q1"))
        path = self.dir / "out" / RECORDS_FILE
        with open(path, "ab") as handle:
            handle.write(b'{"id":"q2","evi')
        fresh = self.store()
        self.assertEqual(list(fresh.load()), ["q1"])
        self.assertTrue(fresh.recovered_partial_line)
        repaired = self.store()
        self.assertEqual(list(repaired.load(repair=True)), ["q1"])
        repaired.append(record("q2"))
        self.assertEqual(sorted(self.store().load()), ["q1", "q2"])

    def test_the_planner_of_part_1_works_on_these_records(self):
        store = self.store()
        store.bind({"top_k": 4}, ["top_k"])
        done = Question("q1", "question one")
        store.append(record("q1", question_sha256=question_hash(done.text)))
        todo = plan([done, Question("q2", "question two")], store.load())
        self.assertEqual((todo.already_done, [q.id for q in todo.pending]), (1, ["q2"]))


if __name__ == "__main__":
    unittest.main()
