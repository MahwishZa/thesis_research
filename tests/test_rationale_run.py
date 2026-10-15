"""Tests of src/rationale_query/run.py: the loop that writes one record per question and can be started again."""

import itertools
import unittest

from rationale_query.generator import RESULT_RELEVANT, ContextOverflowError
from rationale_query.prompt import build_rationale_prompt
from rationale_query.run import (
    MAX_CONSECUTIVE_ERRORS,
    RunAborted,
    generate_rationales,
    plan,
)
from rationale_query.store import (
    ConfigMismatch,
    Question,
    QuestionChanged,
    RationaleRecord,
    RationaleStore,
    question_hash,
)
from tests.rationale_support import (
    QUESTIONS,
    ScriptedGenerator,
    TempDirTestCase,
    default_config,
    ok,
)


def counter(step=2.0):
    """A clock that moves by ``step`` at every reading."""
    ticks = itertools.count(0, step)
    return lambda: float(next(ticks))


def record_for(question, status="ok", **fields):
    return RationaleRecord(id=question.id, question_sha256=question_hash(question.text), status=status, **fields)


class PlanTests(unittest.TestCase):
    def test_every_question_is_pending_when_nothing_is_written(self):
        todo = plan(QUESTIONS, {})
        self.assertEqual(todo.pending, QUESTIONS)
        self.assertEqual(todo.already_done, 0)

    def test_a_question_with_a_successful_record_is_done(self):
        existing = {"q2": record_for(QUESTIONS[1], rationale="r")}
        todo = plan(QUESTIONS, existing)
        self.assertEqual([q.id for q in todo.pending], ["q1", "q3"])
        self.assertEqual(todo.already_done, 1)

    def test_a_question_whose_record_is_an_error_is_pending_again(self):
        existing = {"q2": record_for(QUESTIONS[1], status="error", error="Boom: x")}
        self.assertEqual([q.id for q in plan(QUESTIONS, existing).pending], ["q1", "q2", "q3"])

    def test_records_of_other_questions_are_ignored(self):
        existing = {"zzz": RationaleRecord(id="zzz", question_sha256="h", status="ok", rationale="r")}
        self.assertEqual(plan(QUESTIONS, existing).already_done, 0)

    def test_a_changed_question_text_is_refused_and_named_whatever_the_status(self):
        for status in ("ok", "error"):
            existing = {"q1": RationaleRecord(id="q1", question_sha256="other", status=status, rationale="r", error="e")}
            with self.subTest(status=status), self.assertRaises(QuestionChanged) as caught:
                plan(QUESTIONS, existing, "some folder")
            self.assertIn("q1", str(caught.exception))
            self.assertIn("some folder", str(caught.exception))


class RunTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.store = RationaleStore(self.dir / "out folder")

    def run_with(self, generator, questions=QUESTIONS, **kwargs):
        return generate_rationales(questions, generator, self.store, **kwargs)

    def records(self):
        return RationaleStore(self.dir / "out folder").load()

    def test_every_question_gets_a_record_in_order(self):
        generator = ScriptedGenerator([ok("first"), ok("second"), ok("third")])
        summary = self.run_with(generator)
        self.assertEqual((summary.total, summary.already_done, summary.written, summary.errors), (3, 0, 3, []))
        lines = (self.dir / "out folder" / "rationales.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        records = self.records()
        self.assertEqual([records[q.id].rationale for q in QUESTIONS], ["first", "second", "third"])
        self.assertTrue(all(r.status == "ok" for r in records.values()))
        self.assertEqual(records["q1"].question_sha256, question_hash(QUESTIONS[0].text))

    def test_the_prompt_sent_for_each_question_is_the_papers_prompt(self):
        generator = ScriptedGenerator()
        self.run_with(generator)
        self.assertEqual(generator.prompts, [build_rationale_prompt(q.text) for q in QUESTIONS])

    def test_the_record_carries_the_counts_of_the_generation(self):
        generator = ScriptedGenerator([ok("text", finish="length", prompt_tokens=123, completion_tokens=45)])
        self.run_with(generator, QUESTIONS[:1])
        record = self.records()["q1"]
        self.assertEqual((record.finish_reason, record.prompt_tokens, record.completion_tokens), ("length", 123, 45))

    def test_the_time_of_each_question_and_of_the_run_is_measured_with_the_clock(self):
        summary = self.run_with(ScriptedGenerator(), clock=counter(2.0))
        self.assertEqual([r.seconds for r in self.records().values()], [2.0, 2.0, 2.0])
        self.assertEqual(summary.seconds, 14.0)

    def test_the_time_of_a_failed_or_empty_question_is_measured_too(self):
        generate_rationales(QUESTIONS[:2], ScriptedGenerator([RuntimeError("x"), ok("   ")]), self.store, clock=counter(2.0))
        records = self.records()
        self.assertEqual((records["q1"].seconds, records["q2"].seconds), (2.0, 2.0))

    def test_the_config_of_the_generator_is_bound_to_the_folder(self):
        self.run_with(ScriptedGenerator(config=default_config(seed=11)))
        self.assertEqual(self.store.read_config()["seed"], 11)

    def test_a_second_run_does_not_call_the_model(self):
        self.run_with(ScriptedGenerator())
        generator = ScriptedGenerator()
        summary = self.run_with(generator)
        self.assertEqual(generator.prompts, [])
        self.assertEqual((summary.already_done, summary.written), (3, 0))

    def test_an_interrupted_run_continues_with_the_question_that_was_not_finished(self):
        interrupting = ScriptedGenerator([ok("first"), KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.run_with(interrupting)
        self.assertEqual(set(self.records()), {"q1"})
        generator = ScriptedGenerator()
        summary = self.run_with(generator)
        self.assertEqual(generator.prompts, [build_rationale_prompt(q.text) for q in QUESTIONS[1:]])
        self.assertEqual((summary.already_done, summary.written), (1, 2))
        self.assertEqual(self.records()["q1"].rationale, "first")

    def test_a_failure_is_recorded_with_its_kind_and_the_run_goes_on(self):
        generator = ScriptedGenerator([ok("first"), RuntimeError("it broke"), ok("third")])
        summary = self.run_with(generator)
        self.assertEqual(summary.errors, [("q2", "RuntimeError: it broke")])
        records = self.records()
        self.assertEqual(records["q2"].status, "error")
        self.assertIsNone(records["q2"].rationale)
        self.assertEqual(records["q2"].error, "RuntimeError: it broke")
        self.assertEqual((records["q1"].status, records["q3"].status), ("ok", "ok"))

    def test_every_kind_of_failure_is_recorded_under_its_name(self):
        failures = [
            ValueError("bad value"),
            OSError("disk"),
            KeyError("k"),
            UnicodeEncodeError("utf-8", "x", 0, 1, "surrogates not allowed"),
            ContextOverflowError("too long"),
        ]
        for number, failure in enumerate(failures):
            with self.subTest(failure=type(failure).__name__):
                store = RationaleStore(self.dir / f"failures {number}")
                summary = generate_rationales(QUESTIONS[:1], ScriptedGenerator([failure]), store)
                record = store.load()["q1"]
                self.assertEqual(record.status, "error")
                self.assertTrue(record.error.startswith(type(failure).__name__ + ": "), record.error)
                self.assertEqual(len(summary.errors), 1)

    def test_a_failed_question_is_tried_again_on_the_next_run_and_the_new_record_wins(self):
        self.run_with(ScriptedGenerator([ok("first"), RuntimeError("it broke"), ok("third")]))
        generator = ScriptedGenerator([ok("second, finally")])
        summary = self.run_with(generator)
        self.assertEqual(len(generator.prompts), 1)
        self.assertEqual((summary.already_done, summary.written, summary.errors), (2, 1, []))
        self.assertEqual(self.records()["q2"].rationale, "second, finally")

    def test_an_empty_answer_is_an_error_and_never_replaced_by_the_question(self):
        for text in ("", "   \n\t"):
            with self.subTest(text=text):
                store = RationaleStore(self.dir / f"out {len(text)}")
                generate_rationales(QUESTIONS[:1], ScriptedGenerator([ok(text)]), store)
                record = store.load()["q1"]
                self.assertEqual(record.status, "error")
                self.assertTrue(record.error.startswith("EmptyRationale"))
                self.assertIsNone(record.rationale)
                self.assertEqual(record.completion_tokens, 7)

    def test_the_run_stops_after_five_questions_in_a_row_have_failed(self):
        questions = [Question(f"q{n}", f"Question {n}?") for n in range(1, 9)]
        generator = ScriptedGenerator([RuntimeError("model gone")])
        with self.assertRaises(RunAborted) as caught:
            self.run_with(generator, questions)
        self.assertEqual(MAX_CONSECUTIVE_ERRORS, 5)
        self.assertEqual(len(generator.prompts), 5)
        self.assertIn("model gone", str(caught.exception))
        self.assertEqual(len(self.records()), 5)

    def test_a_success_in_between_starts_the_count_again(self):
        questions = [Question(f"q{n}", f"Question {n}?") for n in range(1, 11)]
        bad, good = RuntimeError("bad"), ok("good")
        generator = ScriptedGenerator([bad] * 4 + [good] + [bad] * 4 + [good])
        summary = self.run_with(generator, questions)
        self.assertEqual(len(summary.errors), 8)

    def test_a_question_that_changed_is_refused_before_anything_is_written(self):
        self.run_with(ScriptedGenerator())
        changed = [*QUESTIONS[:2], Question("q3", "A different third question?")]
        generator = ScriptedGenerator()
        with self.assertRaises(QuestionChanged):
            self.run_with(generator, changed)
        self.assertEqual(generator.prompts, [])

    def test_a_change_late_in_the_file_is_found_before_earlier_questions_are_run(self):
        self.run_with(ScriptedGenerator(), QUESTIONS[2:])
        changed = [*QUESTIONS[:2], Question("q3", "A different third question?")]
        generator = ScriptedGenerator()
        with self.assertRaises(QuestionChanged):
            self.run_with(generator, changed)
        self.assertEqual(generator.prompts, [])
        self.assertEqual(set(self.records()), {"q3"})

    def test_other_settings_than_the_folder_was_made_with_are_refused_before_the_model_runs(self):
        self.run_with(ScriptedGenerator(config=default_config(n_ctx=100)))
        generator = ScriptedGenerator(config=default_config(n_ctx=200))
        with self.assertRaises(ConfigMismatch):
            self.run_with(generator)
        self.assertEqual(generator.prompts, [])

    def test_a_label_that_is_not_bound_to_the_folder_may_change(self):
        self.run_with(ScriptedGenerator(config=default_config(model_name="first label")), QUESTIONS[:1])
        generator = ScriptedGenerator(config=default_config(model_name="second label"))
        self.assertEqual(self.run_with(generator).written, 2)

    def test_every_field_that_is_bound_to_the_folder_is_checked(self):
        self.run_with(ScriptedGenerator(), QUESTIONS[:1])
        for name in RESULT_RELEVANT:
            with self.subTest(field=name), self.assertRaises(ConfigMismatch):
                self.run_with(ScriptedGenerator(config=default_config(**{name: "different"})))

    def test_progress_reports_the_position_among_all_questions(self):
        self.run_with(ScriptedGenerator(), QUESTIONS[:1])
        seen = []
        self.run_with(ScriptedGenerator(), progress=lambda position, total, record: seen.append((position, total, record.id)))
        self.assertEqual(seen, [(2, 3, "q2"), (3, 3, "q3")])

    def test_a_cut_off_last_line_of_an_interrupted_run_is_removed_and_reported(self):
        self.run_with(ScriptedGenerator(), QUESTIONS[:2])
        path = self.dir / "out folder" / "rationales.jsonl"
        with open(path, "ab") as handle:
            handle.write(b'{"id":"q3","quest')
        summary = self.run_with(ScriptedGenerator())
        self.assertTrue(summary.removed_cut_off_line)
        self.assertEqual((summary.already_done, summary.written), (2, 1))
        self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 3)

    def test_a_clean_file_is_not_reported_as_cut_off(self):
        self.assertFalse(self.run_with(ScriptedGenerator()).removed_cut_off_line)

    def test_a_second_run_in_the_same_folder_at_the_same_time_is_detected(self):
        other = RationaleStore(self.dir / "out folder")

        class Intruder(ScriptedGenerator):
            def generate(self, user_message):
                result = super().generate(user_message)
                if len(self.prompts) == 1:  # another run writes while this one is generating its first question
                    other.append(RationaleRecord(id="q9", question_sha256="h", status="ok", rationale="x"))
                return result

        with self.assertRaises(Exception) as caught:
            self.run_with(Intruder())
        self.assertIn("another run", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
