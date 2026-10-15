"""Tests of src/rationale_query/store.py: question files, records, the output folder and its bound configuration."""

import json
import unittest

from rationale_query.generator import text_sha256
from rationale_query.store import (
    ConfigMismatch,
    CorruptStore,
    Question,
    RationaleRecord,
    RationaleStore,
    StoreError,
    load_rationales,
    question_hash,
    read_questions,
    write_atomic,
)
from tests.rationale_support import QUESTIONS, TempDirTestCase, question_lines


def ok(question_id, text="a rationale", question="a question", **fields):
    return RationaleRecord(id=question_id, question_sha256=question_hash(question), status="ok", rationale=text, **fields)


def failed(question_id, message="Boom: it failed", question="a question"):
    return RationaleRecord(id=question_id, question_sha256=question_hash(question), status="error", error=message)


class ReadQuestionsTests(TempDirTestCase):
    def read(self, text, name="q.jsonl"):
        return read_questions(self.write(name, text))

    def test_json_lines_are_read_in_order_with_trimmed_text(self):
        questions = self.read('{"id": "a", "question": "  First?\\n"}\n{"id": "b", "question": "Second?"}\n')
        self.assertEqual(questions, [Question("a", "First?"), Question("b", "Second?")])

    def test_a_json_list_is_read_too(self):
        questions = self.read('[{"id": "a", "question": "First?"}, {"id": "b", "question": "Second?"}]', "q.json")
        self.assertEqual([q.id for q in questions], ["a", "b"])

    def test_blank_lines_and_windows_line_ends_are_accepted(self):
        path = self.dir / "q.jsonl"
        path.write_bytes(b'{"id": "a", "question": "First?"}\r\n\r\n{"id": "b", "question": "Second?"}\r\n')
        self.assertEqual([q.id for q in read_questions(path)], ["a", "b"])

    def test_a_byte_order_mark_is_accepted(self):
        path = self.dir / "q.jsonl"
        path.write_bytes(b'\xef\xbb\xbf{"id": "a", "question": "First?"}\n')
        self.assertEqual(read_questions(path), [Question("a", "First?")])

    def test_other_line_break_characters_stay_inside_the_question(self):
        text = "line one line two\x85end"
        questions = self.read(json.dumps({"id": "a", "question": text}, ensure_ascii=False) + "\n")
        self.assertEqual(questions[0].text, text)

    def test_ids_may_be_whole_numbers_and_are_trimmed(self):
        questions = self.read('{"id": 7, "question": "A?"}\n{"id": " x ", "question": "B?"}\n')
        self.assertEqual([q.id for q in questions], ["7", "x"])

    def test_unusable_ids_are_refused(self):
        for bad in ("null", "true", "1.5", '["a"]', '""', '"  "'):
            with self.subTest(id=bad), self.assertRaises(StoreError):
                self.read(f'{{"id": {bad}, "question": "A?"}}\n')

    def test_an_id_that_appears_twice_is_refused_and_named(self):
        with self.assertRaises(StoreError) as caught:
            self.read('{"id": "a", "question": "A?"}\n{"id": "a", "question": "B?"}\n')
        self.assertIn("a appears twice", str(caught.exception))
        self.assertIn("line 2", str(caught.exception))

    def test_a_question_without_text_is_refused(self):
        for bad in ('""', '"  \\n"', "5", "null"):
            with self.subTest(question=bad), self.assertRaises(StoreError):
                self.read(f'{{"id": "a", "question": {bad}}}\n')

    def test_a_line_without_id_or_question_is_refused(self):
        for bad in ('{"question": "A?"}', '{"id": "a"}', '"just text"', "[1]"):
            with self.subTest(line=bad), self.assertRaises(StoreError):
                self.read(bad + "\n" + '{"id": "z", "question": "Z?"}\n')

    def test_a_file_without_questions_is_refused(self):
        for text in ("", "\n\n  \n", "[]"):
            with self.subTest(text=text), self.assertRaises(StoreError):
                self.read(text)

    def test_broken_json_is_refused_with_the_line_number(self):
        with self.assertRaises(StoreError) as caught:
            self.read('{"id": "a", "question": "A?"}\n{"id": "b", "question": \n')
        self.assertIn("line 2", str(caught.exception))
        with self.assertRaises(StoreError):
            self.read('[{"id": "a"', "broken.json")

    def test_a_file_that_is_not_utf8_is_refused(self):
        path = self.dir / "q.jsonl"
        path.write_bytes(b'{"id": "a", "question": "caf\xe9?"}\n')
        with self.assertRaises(StoreError) as caught:
            read_questions(path)
        self.assertIn("UTF-8", str(caught.exception))

    def test_a_missing_file_or_a_folder_is_refused(self):
        with self.assertRaises(StoreError):
            read_questions(self.dir / "absent.jsonl")
        with self.assertRaises(StoreError):
            read_questions(self.dir)


class RecordTests(unittest.TestCase):
    def test_a_record_survives_the_round_trip_through_json(self):
        record = ok("q1", "Nicht müde β-amyloid\nline two   \"quoted\"", finish_reason="stop", prompt_tokens=3, completion_tokens=4, seconds=1.5)
        self.assertEqual(RationaleRecord.from_dict(json.loads(record.to_json())), record)

    def test_a_record_is_one_line_with_sorted_keys_and_readable_characters(self):
        line = ok("q1", "β\nsecond   line").to_json()
        self.assertNotIn("\n", line)
        self.assertIn("β", line)
        keys = list(json.loads(line))
        self.assertEqual(keys, sorted(keys))
        self.assertNotIn(", ", line)

    def test_unknown_fields_are_ignored_and_missing_ones_are_none(self):
        record = RationaleRecord.from_dict({"id": "a", "question_sha256": "h", "status": "ok", "future_field": 1})
        self.assertIsNone(record.rationale)
        self.assertIsNone(record.error)

    def test_a_record_needs_a_text_id_and_a_known_status(self):
        for data in ({"id": 1, "status": "ok"}, {"id": "a", "status": "maybe"}, {"id": "a"}, {}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                RationaleRecord.from_dict(data)
        with self.assertRaises(TypeError):
            RationaleRecord.from_dict(["a"])


class QuestionHashTests(unittest.TestCase):
    def test_surrounding_whitespace_does_not_change_the_hash(self):
        self.assertEqual(question_hash("  A question?\n"), question_hash("A question?"))
        self.assertEqual(question_hash("A question?"), text_sha256("A question?"))

    def test_other_text_gives_another_hash(self):
        self.assertNotEqual(question_hash("A question?"), question_hash("A question!"))
        self.assertNotEqual(question_hash("A  question?"), question_hash("A question?"))


class WriteAtomicTests(TempDirTestCase):
    def test_text_is_written_as_utf8_with_line_feeds(self):
        path = self.dir / "a.txt"
        write_atomic(path, "ä\nβ\n")
        self.assertEqual(path.read_bytes(), "ä\nβ\n".encode())

    def test_bytes_are_written_as_they_are(self):
        path = self.dir / "a.bin"
        write_atomic(path, b"\x00\x01\r\n")
        self.assertEqual(path.read_bytes(), b"\x00\x01\r\n")

    def test_an_existing_file_is_replaced_and_no_temporary_file_stays(self):
        path = self.write("a.txt", "old")
        write_atomic(path, "new")
        self.assertEqual(path.read_text(encoding="utf-8"), "new")
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["a.txt"])

    def test_missing_folders_are_created(self):
        path = self.dir / "deep" / "er" / "a.txt"
        write_atomic(path, "x")
        self.assertEqual(path.read_text(encoding="utf-8"), "x")


class BindTests(TempDirTestCase):
    RELEVANT = ("model_sha256", "max_new_tokens")

    def config(self, **changes):
        return {"model_sha256": "abc", "max_new_tokens": 512, "model_name": "label", "n_threads": 4, **changes}

    def test_the_first_run_records_the_config_as_sorted_json(self):
        store = RationaleStore(self.dir / "out folder")
        store.bind(self.config(), self.RELEVANT)
        path = self.dir / "out folder" / "config.json"
        text = path.read_text(encoding="utf-8")
        self.assertEqual(json.loads(text), self.config())
        self.assertTrue(text.endswith("}\n"))
        self.assertEqual(list(json.loads(text)), sorted(json.loads(text)))
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["config.json"])

    def test_the_same_settings_are_accepted_again(self):
        RationaleStore(self.dir).bind(self.config(), self.RELEVANT)
        RationaleStore(self.dir).bind(self.config(), self.RELEVANT)

    def test_a_changed_setting_that_matters_is_refused_and_named(self):
        RationaleStore(self.dir).bind(self.config(), self.RELEVANT)
        with self.assertRaises(ConfigMismatch) as caught:
            RationaleStore(self.dir).bind(self.config(max_new_tokens=256), self.RELEVANT)
        message = str(caught.exception)
        self.assertIn("max_new_tokens: recorded 512, now 256", message)
        self.assertNotIn("model_sha256", message)
        self.assertIn("new output folder", message)

    def test_a_changed_setting_that_does_not_matter_is_accepted_and_the_record_is_kept(self):
        RationaleStore(self.dir).bind(self.config(), self.RELEVANT)
        RationaleStore(self.dir).bind(self.config(model_name="other label", n_threads=8), self.RELEVANT)
        self.assertEqual(RationaleStore(self.dir).read_config()["model_name"], "label")

    def test_a_setting_that_the_old_record_lacks_counts_as_changed(self):
        RationaleStore(self.dir).bind({"model_sha256": "abc"}, self.RELEVANT)
        with self.assertRaises(ConfigMismatch):
            RationaleStore(self.dir).bind(self.config(), self.RELEVANT)

    def test_records_without_a_config_are_refused_because_nobody_knows_how_they_were_made(self):
        RationaleStore(self.dir).append(ok("q1"))
        with self.assertRaises(StoreError) as caught:
            RationaleStore(self.dir).bind(self.config(), self.RELEVANT)
        self.assertIn("config.json is missing", str(caught.exception))
        self.assertFalse((self.dir / "config.json").exists())

    def test_an_empty_records_file_does_not_block_the_first_run(self):
        self.write("rationales.jsonl", "")
        RationaleStore(self.dir).bind(self.config(), self.RELEVANT)
        self.assertTrue((self.dir / "config.json").is_file())

    def test_an_unreadable_config_is_reported(self):
        self.write("config.json", "{not json")
        with self.assertRaises(StoreError):
            RationaleStore(self.dir).bind(self.config(), self.RELEVANT)


class LoadAndAppendTests(TempDirTestCase):
    def store(self):
        return RationaleStore(self.dir / "out")

    def test_an_empty_or_new_folder_has_no_records(self):
        self.assertEqual(self.store().load(), {})

    def test_records_are_appended_one_line_each_and_the_latest_wins(self):
        store = self.store()
        store.append(failed("q1"))
        store.append(ok("q2"))
        store.append(ok("q1", "second try"))
        lines = (self.dir / "out" / "rationales.jsonl").read_text(encoding="utf-8").split("\n")
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[-1], "")
        records = self.store().load()
        self.assertEqual(records["q1"].rationale, "second try")
        self.assertEqual(records["q1"].status, "ok")
        self.assertEqual(set(records), {"q1", "q2"})

    def test_a_cut_off_last_line_is_ignored_and_a_reader_leaves_the_file_alone(self):
        store = self.store()
        store.append(ok("q1"))
        path = self.dir / "out" / "rationales.jsonl"
        with open(path, "ab") as handle:
            handle.write(b'{"id":"q2","question_sha256":"h","sta')
        before = path.read_bytes()
        reader = self.store()
        self.assertEqual(set(reader.load()), {"q1"})
        self.assertTrue(reader.recovered_partial_line)
        self.assertEqual(path.read_bytes(), before)

    def test_the_writer_removes_the_cut_off_line_so_the_next_record_starts_cleanly(self):
        store = self.store()
        store.append(ok("q1"))
        path = self.dir / "out" / "rationales.jsonl"
        with open(path, "ab") as handle:
            handle.write(b'{"id":"q2","question_sha256":"h","sta')
        writer = self.store()
        self.assertEqual(set(writer.load(repair=True)), {"q1"})
        self.assertTrue(writer.recovered_partial_line)
        writer.append(ok("q2"))
        self.assertEqual(set(self.store().load()), {"q1", "q2"})
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_a_file_that_is_one_cut_off_line_becomes_empty(self):
        path = self.dir / "out" / "rationales.jsonl"
        path.parent.mkdir()
        path.write_bytes(b'{"id":"q1"')
        writer = self.store()
        self.assertEqual(writer.load(repair=True), {})
        self.assertEqual(path.read_bytes(), b"")

    def test_a_complete_file_is_not_reported_as_cut_off(self):
        store = self.store()
        store.append(ok("q1"))
        reader = self.store()
        reader.load()
        self.assertFalse(reader.recovered_partial_line)

    def test_writing_after_a_read_that_left_a_cut_off_line_is_refused(self):
        path = self.dir / "out" / "rationales.jsonl"
        path.parent.mkdir()
        path.write_bytes(b'{"id":"q1","question_sha256":"h","status":"ok"}\n{"id":"q2"')
        store = self.store()
        store.load()
        with self.assertRaises(StoreError):
            store.append(ok("q3"))
        self.assertEqual(path.read_bytes().count(b"\n"), 1)

    def test_damage_in_the_middle_is_an_error_naming_the_line(self):
        path = self.dir / "out" / "rationales.jsonl"
        path.parent.mkdir()
        good = ok("q1").to_json()
        for bad in ("not json at all", '{"id":"q2"}', "[1,2]", '"text"'):
            with self.subTest(bad=bad):
                path.write_text(f"{good}\n{bad}\n{good}\n", encoding="utf-8", newline="\n")
                with self.assertRaises(CorruptStore) as caught:
                    self.store().load()
                self.assertIn("line 2", str(caught.exception))

    def test_a_file_that_is_not_utf8_is_an_error(self):
        path = self.dir / "out" / "rationales.jsonl"
        path.parent.mkdir()
        path.write_bytes(b'{"id":"q1","question_sha256":"h","status":"ok","rationale":"caf\xe9"}\n')  # a valid record but Latin-1
        with self.assertRaises(CorruptStore) as caught:
            self.store().load()
        self.assertIn("UTF-8", str(caught.exception))

    def test_windows_line_ends_are_read(self):
        path = self.dir / "out" / "rationales.jsonl"
        path.parent.mkdir()
        path.write_bytes((ok("q1").to_json() + "\r\n" + ok("q2").to_json() + "\r\n").encode())
        self.assertEqual(set(self.store().load()), {"q1", "q2"})

    def test_text_with_unusual_characters_comes_back_unchanged(self):
        text = "β-amyloid   line\nbreak \"quote\" \\ backslash \U0001F642"
        store = self.store()
        store.append(ok("q1", text))
        self.assertEqual(self.store().load()["q1"].rationale, text)

    def test_a_second_writer_to_the_same_folder_is_detected(self):
        first, second = self.store(), self.store()
        first.load(repair=True)
        second.load(repair=True)
        first.append(ok("q1"))
        with self.assertRaises(StoreError) as caught:
            second.append(ok("q2"))
        self.assertIn("another run", str(caught.exception))
        self.assertEqual(set(self.store().load()), {"q1"})

    def test_a_writer_may_continue_after_its_own_writes(self):
        store = self.store()
        store.load(repair=True)
        for number in range(3):
            store.append(ok(f"q{number}"))
        self.assertEqual(len(self.store().load()), 3)

    def test_a_store_that_never_read_the_file_can_still_append(self):
        store = self.store()
        store.append(ok("q1"))
        store.append(ok("q2"))
        self.assertEqual(len(self.store().load()), 2)


class LoadRationalesTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.store = RationaleStore(self.dir)
        self.questions = [Question(f"q{n}", f"Question {n}?") for n in range(1, 4)]

    def record(self, number, text=None, **fields):
        question = self.questions[number - 1]
        return RationaleRecord(
            id=question.id, question_sha256=question_hash(question.text), status="ok", rationale=text or f"rationale {number}", **fields
        )

    def test_every_question_gets_its_rationale(self):
        for number in (1, 2, 3):
            self.store.append(self.record(number))
        self.assertEqual(
            load_rationales(self.dir, self.questions), {"q1": "rationale 1", "q2": "rationale 2", "q3": "rationale 3"}
        )

    def test_a_missing_record_is_an_error_and_is_never_skipped(self):
        self.store.append(self.record(1))
        self.store.append(self.record(3))
        with self.assertRaises(StoreError) as caught:
            load_rationales(self.dir, self.questions)
        self.assertIn("q2: no record", str(caught.exception))

    def test_a_failed_question_is_an_error_naming_the_failure(self):
        self.store.append(self.record(1))
        self.store.append(self.record(3))
        self.store.append(RationaleRecord(id="q2", question_sha256="x", status="error", error="Boom: failed"))
        with self.assertRaises(StoreError) as caught:
            load_rationales(self.dir, self.questions)
        self.assertIn("q2: Boom: failed", str(caught.exception))

    def test_a_rationale_written_for_another_question_text_is_refused(self):
        for number in (1, 2, 3):
            self.store.append(self.record(number))
        changed = [*self.questions[:2], Question("q3", "A different question?")]
        with self.assertRaises(StoreError) as caught:
            load_rationales(self.dir, changed)
        self.assertIn("q3", str(caught.exception))
        self.assertIn("differs", str(caught.exception))

    def test_only_five_problems_are_listed_and_the_total_is_given(self):
        questions = [Question(f"q{n}", f"Question {n}?") for n in range(1, 9)]
        with self.assertRaises(StoreError) as caught:
            load_rationales(self.dir, questions)
        message = str(caught.exception)
        self.assertIn("q5: no record", message)
        self.assertNotIn("q6: no record", message)
        self.assertIn("8 problems in all", message)

    def test_the_total_is_given_from_the_sixth_problem_on(self):
        for count, total_shown in ((5, False), (6, True)):
            questions = [Question(f"q{n}", f"Question {n}?") for n in range(1, count + 1)]
            with self.subTest(problems=count), self.assertRaises(StoreError) as caught:
                load_rationales(self.dir, questions)
            self.assertEqual(f"{count} problems in all" in str(caught.exception), total_shown)

    def test_a_reader_never_changes_the_file(self):
        self.store.append(self.record(1))
        path = self.dir / "rationales.jsonl"
        with open(path, "ab") as handle:
            handle.write(b'{"id":"q2"')
        before = path.read_bytes()
        with self.assertRaises(StoreError):
            load_rationales(self.dir, self.questions[:2])
        self.assertEqual(path.read_bytes(), before)


class FixtureTests(TempDirTestCase):
    def test_the_shared_questions_make_a_valid_file(self):
        self.assertEqual(read_questions(self.write("q.jsonl", question_lines())), QUESTIONS)


if __name__ == "__main__":
    unittest.main()
