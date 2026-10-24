"""rationale_query.run: every question gets a rationale, saved at once; restarts, failures and changed questions."""

import unittest

from rationale_query import run
from rationale_query.generator import GenerationResult
from rationale_query.prompt import build_rationale_prompt
from rationale_query.run import MAX_CONSECUTIVE_ERRORS, RunAborted, generate_rationales
from rationale_query.store import ConfigMismatch, Question, QuestionChanged, RationaleStore
from tests.support import TempDirTestCase
from tests.unit.rationale_query.examples import FIGURE_4_QUESTION

RATIONALE = "The patient has COPD with respiratory failure. Therefore, the answer is (C) Intubation."


class Generator:
    config = {"model_sha256": "m" * 64}

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies[len(self.prompts) - 1] if len(self.prompts) <= len(self.replies) else RATIONALE
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, GenerationResult):
            return reply
        return GenerationResult(text=reply, finish_reason="stop", prompt_tokens=300, completion_tokens=40)


QUESTIONS = [Question("q1", FIGURE_4_QUESTION), Question("q2", "Second? (A) x (B) y"), Question("q3", "Third? (A) x (B) y")]


class RunTests(TempDirTestCase):
    def go(self, questions=QUESTIONS, generator=None, config=None, **kwargs):
        store = RationaleStore(self.dir / "run")
        generator = generator or Generator()
        summary = generate_rationales(questions, generator, store, config or {"n_ctx": 2048}, ("n_ctx",), **kwargs)
        return summary, store, generator

    def test_every_question_is_asked_with_the_papers_prompt_and_its_rationale_is_kept_complete(self):
        summary, store, generator = self.go()
        self.assertEqual((summary.total, summary.written, summary.already_done), (3, 3, 0))
        self.assertEqual(generator.prompts[0], build_rationale_prompt(FIGURE_4_QUESTION))
        record = store.load()["q1"]
        self.assertEqual((record.status, record.rationale, record.finish_reason), ("ok", RATIONALE, "stop"))
        self.assertEqual((record.prompt_tokens, record.completion_tokens), (300, 40))

    def test_the_record_has_a_rationale_but_no_answer_and_no_evidence_field(self):
        self.go()
        fields = RationaleStore(self.dir / "run").load()["q1"].__dataclass_fields__
        self.assertIn("rationale", fields)
        for forbidden in ("answer", "evidence", "gold", "letter"):
            self.assertNotIn(forbidden, fields)

    def test_a_second_run_does_nothing(self):
        self.go()
        summary, _, generator = self.go()
        self.assertEqual((summary.already_done, summary.written), (3, 0))
        self.assertEqual(generator.prompts, [])

    def test_a_stopped_run_continues_where_it_ended(self):
        self.go(questions=QUESTIONS[:2])
        summary, store, generator = self.go()
        self.assertEqual((summary.already_done, summary.written), (2, 1))
        self.assertEqual(generator.prompts, [build_rationale_prompt(QUESTIONS[2].text)])
        self.assertEqual(sorted(store.load()), ["q1", "q2", "q3"])

    def test_a_failure_is_saved_as_a_failure_and_tried_again_next_time(self):
        summary, store, _ = self.go(generator=Generator([RuntimeError("out of memory")]))
        self.assertEqual((summary.written, len(summary.errors)), (2, 1))
        self.assertEqual(store.load()["q1"].status, "error")
        self.assertIn("out of memory", store.load()["q1"].error)
        self.assertIsNone(store.load()["q1"].rationale, "the question text must never stand in for a rationale")
        again, store, _ = self.go()
        self.assertEqual((again.already_done, again.written), (2, 1))
        self.assertEqual(store.load()["q1"].status, "ok")

    def test_an_empty_rationale_is_a_failure(self):
        summary, store, _ = self.go(generator=Generator(["   "]))
        self.assertEqual(store.load()["q1"].status, "error")
        self.assertEqual(len(summary.errors), 1)

    def test_a_rationale_that_filled_the_window_is_kept_and_counted(self):
        cut = GenerationResult(text="and so on and so on", finish_reason="length", prompt_tokens=300, completion_tokens=1748)
        summary, store, _ = self.go(generator=Generator([cut]))
        self.assertEqual(summary.cut_off, 1)
        self.assertEqual((store.load()["q1"].status, store.load()["q1"].rationale), ("ok", "and so on and so on"))

    def test_several_failures_in_a_row_stop_the_run(self):
        many = [Question(f"q{i}", f"Question {i}? (A) x (B) y") for i in range(MAX_CONSECUTIVE_ERRORS + 3)]
        generator = Generator([RuntimeError("broken")] * len(many))
        with self.assertRaises(RunAborted):
            self.go(questions=many, generator=generator)
        self.assertEqual(len(generator.prompts), MAX_CONSECUTIVE_ERRORS)

    def test_a_success_between_failures_resets_the_count(self):
        many = [Question(f"q{i}", f"Question {i}? (A) x (B) y") for i in range(10)]
        errors = [RuntimeError("x")] * (MAX_CONSECUTIVE_ERRORS - 1)
        replies = errors + [RATIONALE] + errors + [RATIONALE] * 5
        summary, _, _ = self.go(questions=many, generator=Generator(replies))
        self.assertEqual(len(summary.errors), 2 * (MAX_CONSECUTIVE_ERRORS - 1))

    def test_a_question_whose_text_changed_stops_the_run(self):
        self.go()
        changed = [Question("q1", "A different question? (A) x (B) y")] + QUESTIONS[1:]
        with self.assertRaises(QuestionChanged):
            self.go(questions=changed)

    def test_other_settings_are_refused(self):
        self.go()
        with self.assertRaises(ConfigMismatch):
            self.go(config={"n_ctx": 4096})

    def test_progress_is_reported_for_each_new_rationale(self):
        seen = []
        self.go(progress=lambda position, total, record: seen.append((position, total, record.id)))
        self.assertEqual(seen, [(1, 3, "q1"), (2, 3, "q2"), (3, 3, "q3")])

    def test_the_module_does_not_import_a_heavy_library(self):
        self.assertFalse(hasattr(run, "torch") or hasattr(run, "llama_cpp"))


if __name__ == "__main__":
    unittest.main()
