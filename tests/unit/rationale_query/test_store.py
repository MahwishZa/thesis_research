"""rationale_query.store: reading questions, saving rationales, binding a folder to its settings, and surviving a stop."""

import json
import unittest

from rationale_query.store import (
    ConfigMismatch,
    CorruptStore,
    Question,
    RationaleRecord,
    RationaleStore,
    StoreError,
    question_hash,
    read_questions,
)
from tests.support import TempDirTestCase


def ok(identifier="q1", text="Q (A) a (B) b", rationale="Because... Therefore, the answer is (A) a.", **extra):
    return RationaleRecord(id=identifier, question_sha256=question_hash(text), status="ok", rationale=rationale, finish_reason="stop", **extra)


class ReadQuestionsTests(TempDirTestCase):
    def test_questions_are_read_in_order_with_their_options_inline(self):
        path = self.write("q.jsonl", '{"id": "a", "question": "First? (A) x (B) y"}\n\n{"id": "b", "question": "Second? (A) z (B) w"}\n')
        self.assertEqual(read_questions(path), [Question("a", "First? (A) x (B) y"), Question("b", "Second? (A) z (B) w")])

    def test_a_byte_order_mark_is_accepted(self):
        path = self.dir / "q.jsonl"
        path.write_bytes(b'\xef\xbb\xbf{"id": "a", "question": "Q"}\n')
        self.assertEqual(read_questions(path), [Question("a", "Q")])

    def test_refused_files(self):
        cases = {
            "repeated id": '{"id": "a", "question": "Q"}\n{"id": "a", "question": "R"}\n',
            "no id": '{"question": "Q"}\n',
            "empty id": '{"id": "", "question": "Q"}\n',
            "blank question": '{"id": "a", "question": "  "}\n',
            "question not text": '{"id": "a", "question": 5}\n',
            "broken line": "{broken\n",
            "no questions": "\n",
        }
        for name, text in cases.items():
            with self.subTest(name), self.assertRaises(StoreError):
                read_questions(self.write("bad.jsonl", text))

    def test_a_missing_file_is_refused(self):
        with self.assertRaises(StoreError):
            read_questions(self.dir / "absent.jsonl")


class RecordTests(unittest.TestCase):
    def test_a_record_survives_a_round_trip_with_unicode_and_line_breaks_in_the_rationale(self):
        record = ok(rationale="Line one\nLine two °F – done.\nTherefore, the answer is (A) a.", prompt_tokens=10, completion_tokens=20, seconds=1.5)
        self.assertEqual(RationaleRecord.from_dict(json.loads(record.to_json())), record)

    def test_an_ok_record_needs_a_rationale(self):
        for bad in (None, "", "  ", 5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RationaleRecord.from_dict({"id": "q", "question_sha256": "x", "status": "ok", "rationale": bad})

    def test_other_bad_records_are_refused(self):
        for bad in ({}, {"id": 1, "status": "ok"}, {"id": "q", "question_sha256": "x", "status": "maybe"}, [], "x"):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                RationaleRecord.from_dict(bad)


class StoreTests(TempDirTestCase):
    def store(self):
        return RationaleStore(self.dir / "run")

    def test_records_are_saved_and_the_latest_one_per_question_wins(self):
        store = self.store()
        store.bind({"n_ctx": 2048}, ("n_ctx",))
        store.append(RationaleRecord(id="q1", question_sha256="h", status="error", error="boom"))
        store.append(ok("q1"))
        store.append(ok("q2"))
        loaded = store.load()
        self.assertEqual(sorted(loaded), ["q1", "q2"])
        self.assertEqual(loaded["q1"].status, "ok")

    def test_a_folder_refuses_other_settings_and_accepts_the_same_ones(self):
        store = self.store()
        store.bind({"n_ctx": 2048, "threads": 4}, ("n_ctx",))
        store.bind({"n_ctx": 2048, "threads": 8}, ("n_ctx",))  # a setting that does not change the text is free
        with self.assertRaises(ConfigMismatch):
            store.bind({"n_ctx": 4096}, ("n_ctx",))

    def test_a_last_line_cut_short_by_a_stop_is_ignored_and_can_be_repaired(self):
        store = self.store()
        store.bind({}, ())
        store.append(ok("q1"))
        with open(store.records_path, "a", encoding="utf-8") as handle:
            handle.write('{"id": "q2", "question_sha')
        self.assertEqual(sorted(store.load()), ["q1"])
        self.assertEqual(sorted(store.load(repair=True)), ["q1"])
        store.append(ok("q2"))
        self.assertEqual(sorted(store.load()), ["q1", "q2"])

    def test_a_broken_line_in_the_middle_is_an_error_not_something_to_skip(self):
        store = self.store()
        store.bind({}, ())
        store.append(ok("q1"))
        with open(store.records_path, "a", encoding="utf-8") as handle:
            handle.write("not json\n")
        store.append(ok("q2"))
        with self.assertRaises(CorruptStore):
            store.load()

    def test_an_empty_folder_has_no_records(self):
        self.assertEqual(self.store().load(), {})


if __name__ == "__main__":
    unittest.main()
