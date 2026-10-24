"""rationale_query.query: the rationale alone is the query, and the list goes to the retriever in the order of the questions."""

import json
import unittest

from rationale_query.query import export_queries, queries_for, rationale_to_query
from rationale_query.store import Question, RationaleRecord, StoreError, question_hash
from tests.support import TempDirTestCase
from tests.unit.rationale_query.examples import FIGURE_4_QUESTION

RATIONALE = "The patient has COPD.\nIntubation may be necessary.\nTherefore, the answer is (C) Intubation."


def record(question, rationale=RATIONALE, **kwargs):
    return RationaleRecord(id=question.id, question_sha256=question_hash(question.text), status="ok", rationale=rationale, finish_reason="stop", **kwargs)


class QueryTests(unittest.TestCase):
    def test_the_query_is_the_rationale_alone_and_does_not_contain_the_question(self):
        query = rationale_to_query(RATIONALE)
        self.assertEqual(query, RATIONALE)
        self.assertNotIn("BiPAP", query)
        self.assertNotIn(FIGURE_4_QUESTION, query)

    def test_only_the_whitespace_at_the_two_ends_is_removed(self):
        self.assertEqual(rationale_to_query("\n  a\n\nb  \n"), "a\n\nb")

    def test_a_long_rationale_is_not_shortened_here(self):
        long = "word " * 5000
        self.assertEqual(rationale_to_query(long), long.strip())

    def test_an_empty_or_non_text_rationale_is_refused_never_replaced_by_the_question(self):
        for bad in ("", "  ", "\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                rationale_to_query(bad)
        for bad in (None, 3):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                rationale_to_query(bad)


class QueriesForTests(unittest.TestCase):
    QUESTIONS = [Question("a", "First? (A) x (B) y"), Question("b", FIGURE_4_QUESTION)]

    def test_one_query_per_question_in_the_order_of_the_questions(self):
        records = {q.id: record(q, rationale=f"Rationale for {q.id}. Therefore, the answer is (A).") for q in self.QUESTIONS}
        self.assertEqual(queries_for(self.QUESTIONS, records), ["Rationale for a. Therefore, the answer is (A).", "Rationale for b. Therefore, the answer is (A)."])
        self.assertEqual(queries_for(list(reversed(self.QUESTIONS)), records)[0], "Rationale for b. Therefore, the answer is (A).")

    def test_a_missing_failed_or_mismatched_rationale_stops_the_export_and_names_the_question(self):
        good = record(self.QUESTIONS[0])
        failed = RationaleRecord(id="b", question_sha256="h", status="error", error="out of memory")
        wrong_text = record(Question("b", "another text"))
        for name, records in {"missing": {"a": good}, "failed": {"a": good, "b": failed}, "other text": {"a": good, "b": wrong_text}}.items():
            with self.subTest(name), self.assertRaises(StoreError) as caught:
                queries_for(self.QUESTIONS, records)
            self.assertIn("b", str(caught.exception))


class ExportTests(TempDirTestCase):
    def test_the_file_is_a_json_list_of_strings_that_the_retriever_can_read(self):
        path = self.dir / "out" / "queries.json"
        export_queries(["first rationale", "second °F rationale\nline two"], path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), ["first rationale", "second °F rationale\nline two"])

    def test_an_existing_file_is_never_overwritten(self):
        path = self.write("queries.json", "[]")
        with self.assertRaises(StoreError):
            export_queries(["x"], path)
        self.assertEqual(path.read_text(encoding="utf-8"), "[]")


if __name__ == "__main__":
    unittest.main()
