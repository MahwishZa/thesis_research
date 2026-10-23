"""answer_evaluation.runner: answering with every system, resuming, the refusals, and the ledger on a test split."""

import unittest

from answer_evaluation import records, runner
from answer_evaluation.ledger import AlreadyScored, Ledger
from answer_evaluation.records import AnswerStore
from answer_evaluation.runner import NO_RETRIEVAL, Arm, EvalQuestion
from rationale_query.batch import MAX_CONSECUTIVE_ERRORS, RunAborted
from rationale_query.generator import GenerationResult
from rationale_query.store import ConfigMismatch, QuestionChanged
from tests.support import TempDirTestCase

STEM = "Which drug is a cholinesterase inhibitor? A) Donepezil B) Aspirin C) Insulin D) Warfarin"


def question(identifier="q1", gold="A", rationale="Thinking... Therefore, the answer is (B) Aspirin."):
    return EvalQuestion(identifier, STEM, gold, rationale)


class Generator:
    config = {"model_sha256": "m" * 64}

    def __init__(self, answers=("Therefore, the answer is (A) Donepezil.",)):
        self.answers = list(answers)
        self.prompts = []

    def generate(self, user_message):
        self.prompts.append(user_message)
        answer = self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]
        if isinstance(answer, BaseException):
            raise answer
        return GenerationResult(text=answer, finish_reason="stop", prompt_tokens=80, completion_tokens=12)


def snippets(q, arm):
    return [f"snippet {n} for {q.id}" for n in range(1, arm.k + 1)]


class ArmTests(unittest.TestCase):
    def test_k_is_zero_only_for_no_retrieval(self):
        self.assertEqual(Arm(NO_RETRIEVAL, 0).k, 0)
        self.assertEqual(Arm("rag2", 8).k, 8)
        for name, k in ((NO_RETRIEVAL, 3), ("rag2", 0), ("rag2", -1), ("rag2", True), ("rag2", 1.5), ("", 2), ("a|b", 2)):
            with self.subTest(name=name, k=k), self.assertRaises(ValueError):
                Arm(name, k)


class AnswerOneTests(unittest.TestCase):
    def test_no_retrieval_reads_the_rationale_and_writes_nothing(self):
        generator = Generator()
        result = runner.answer_one(question(gold="B"), Arm(NO_RETRIEVAL, 0), [], generator)
        self.assertEqual((result.status, result.letter, result.correct, result.reading), ("ok", "B", True, "ok"))
        self.assertEqual(generator.prompts, [])
        self.assertIsNone(result.finish_reason)

    def test_an_arm_with_snippets_answers_from_the_answer_prompt(self):
        generator = Generator()
        result = runner.answer_one(question(), Arm("rag2", 2), ["one", "two"], generator)
        self.assertTrue(result.correct)
        self.assertEqual((result.evidence_count, result.finish_reason, result.prompt_tokens), (2, "stop", 80))
        self.assertIn("Evidence:\none\n\ntwo\nHere is the question: " + STEM, generator.prompts[0])

    def test_no_snippets_gives_the_plain_prompt(self):
        generator = Generator()
        runner.answer_one(question(), Arm("rag2", 2), [], generator)
        self.assertNotIn("Evidence:", generator.prompts[0])

    def test_an_unreadable_answer_is_wrong_and_says_why(self):
        result = runner.answer_one(question(), Arm("rag2", 2), ["one"], Generator(["I think donepezil."]))
        self.assertEqual((result.status, result.correct, result.letter, result.reading), ("ok", False, None, "none"))

    def test_more_snippets_than_k_is_an_error_record(self):
        result = runner.answer_one(question(), Arm("rag2", 1), ["one", "two"], Generator())
        self.assertEqual(result.status, "error")
        self.assertIn("2 snippets for k = 1", result.error)

    def test_a_failing_generator_or_an_empty_answer_is_an_error_not_a_wrong_answer(self):
        failing = runner.answer_one(question(), Arm("rag2", 1), ["one"], Generator([RuntimeError("memory")]))
        empty = runner.answer_one(question(), Arm("rag2", 1), ["one"], Generator(["  "]))
        for result in (failing, empty):
            self.assertEqual((result.status, result.correct), ("error", None))

    def test_the_evidence_hash_follows_the_snippets(self):
        one = runner.answer_one(question(), Arm("rag2", 2), ["a", "b"], Generator())
        same = runner.answer_one(question(), Arm("rag2", 2), [" a ", "b\n"], Generator())
        other = runner.answer_one(question(), Arm("rag2", 2), ["a", "c"], Generator())
        self.assertEqual(one.evidence_sha256, same.evidence_sha256)
        self.assertNotEqual(one.evidence_sha256, other.evidence_sha256)


class RunTests(TempDirTestCase):
    CONFIG = {"dataset": "medqa", "split": "validation", "generator": {"model_sha256": "m" * 64}}

    def run_arms(self, questions, arms, generator=None, name="run", config=None, **options):
        store = AnswerStore(self.dir / name)
        generator = generator or Generator()
        summary = runner.run_arms(questions, arms, snippets, generator, store, config or self.CONFIG, records.RELEVANT, **options)
        return summary, generator, store

    def test_every_question_is_answered_by_every_arm_and_a_second_run_does_nothing(self):
        questions = [question("q1"), question("q2")]
        arms = [Arm(NO_RETRIEVAL, 0), Arm("rag2", 2)]
        summary, generator, store = self.run_arms(questions, arms)
        self.assertEqual((summary.total, summary.written, summary.already_done), (4, 4, 0))
        self.assertEqual(len(generator.prompts), 2)  # no_retrieval writes nothing
        self.assertEqual(sorted(store.load()), ["no_retrieval|k0|q1", "no_retrieval|k0|q2", "rag2|k2|q1", "rag2|k2|q2"])
        again, generator2, _ = self.run_arms(questions, arms)
        self.assertEqual((again.written, again.already_done, generator2.prompts), (0, 4, []))

    def test_a_stopped_run_continues_and_an_error_is_tried_again(self):
        questions = [question("q1"), question("q2"), question("q3")]
        arms = [Arm("rag2", 2)]
        self.run_arms(questions[:1], arms)
        summary, generator, store = self.run_arms(questions, arms, Generator([RuntimeError("boom"), "the answer is (A)"]))
        self.assertEqual((summary.written, summary.already_done, len(summary.errors)), (2, 1, 1))
        again, generator2, store = self.run_arms(questions, arms)
        self.assertEqual((again.written, again.already_done), (1, 2))
        self.assertTrue(all(r.status == "ok" for r in store.load().values()))

    def test_the_evidence_function_is_not_called_for_no_retrieval_and_a_failure_in_it_is_an_error_record(self):
        calls = []

        def evidence(q, arm):
            calls.append(arm.name)
            raise RuntimeError("retrieval failed")

        store = AnswerStore(self.dir / "run")
        summary = runner.run_arms([question()], [Arm(NO_RETRIEVAL, 0), Arm("rag2", 2)], evidence, Generator(), store, self.CONFIG, records.RELEVANT)
        self.assertEqual(calls, ["rag2"])
        self.assertEqual([e[0] for e in summary.errors], ["rag2|k2|q1"])

    def test_five_failures_in_a_row_stop_the_run(self):
        questions = [question(f"q{i}") for i in range(8)]
        with self.assertRaises(RunAborted):
            self.run_arms(questions, [Arm("rag2", 2)], Generator([RuntimeError("boom")]))
        self.assertEqual(len(AnswerStore(self.dir / "run").load()), MAX_CONSECUTIVE_ERRORS)

    def test_a_changed_question_or_answer_key_is_refused_before_any_work(self):
        self.run_arms([question("q1", gold="A")], [Arm("rag2", 2)])
        with self.assertRaises(QuestionChanged):
            self.run_arms([question("q1", gold="B")], [Arm("rag2", 2)], Generator())

    def test_another_setting_is_refused(self):
        self.run_arms([question()], [Arm("rag2", 2)])
        with self.assertRaises(ConfigMismatch):
            self.run_arms([question()], [Arm("rag2", 2)], config={**self.CONFIG, "split": "train"})

    def test_an_arm_listed_twice_is_refused(self):
        with self.assertRaises(ValueError):
            self.run_arms([question()], [Arm("rag2", 2), Arm("rag2", 2)])

    def test_progress_is_reported_in_question_then_arm_order(self):
        seen = []
        self.run_arms([question("q1"), question("q2")], [Arm(NO_RETRIEVAL, 0), Arm("rag2", 2)], progress=lambda n, t, r: seen.append((n, t, r.id)))
        self.assertEqual([item[0] for item in seen], [1, 2, 3, 4])
        self.assertEqual(seen[1][2], "rag2|k2|q1")

    def test_the_hash_of_the_settings_depends_on_the_relevant_fields_only(self):
        base = runner.settings_sha256({**self.CONFIG, "note": "a"}, records.RELEVANT)
        self.assertEqual(base, runner.settings_sha256({**self.CONFIG, "note": "b"}, records.RELEVANT))
        self.assertNotEqual(base, runner.settings_sha256({**self.CONFIG, "split": "test"}, records.RELEVANT))


class TestSplitTests(TempDirTestCase):
    CONFIG = {"dataset": "medqa", "split": "test", "generator": {"model_sha256": "m" * 64}}

    def run_test_split(self, book=None, system="rag2", generator=None, name="run", questions=None, **options):
        store = AnswerStore(self.dir / name)
        return runner.run_arms(
            questions or [question("q1"), question("q2")], [Arm("rag2", 2)], snippets, generator or Generator(), store, self.CONFIG,
            records.RELEVANT, ledger=book, system=system, **options,
        )

    def test_a_test_run_without_a_ledger_or_a_system_name_is_refused_before_anything_is_written(self):
        with self.assertRaises(ValueError):
            self.run_test_split(book=None)
        with self.assertRaises(ValueError):
            self.run_test_split(book=Ledger(self.dir / "scored.jsonl"), system="")
        self.assertFalse((self.dir / "run").exists())

    def test_the_run_is_claimed_before_the_first_answer_and_finished_after_the_last(self):
        book = Ledger(self.dir / "scored.jsonl")
        generator = Generator()
        self.run_test_split(book, generator=generator)
        self.assertEqual([e["event"] for e in book.events()], ["claim", "finish"])

    def test_a_second_scoring_of_the_same_system_is_refused_before_any_answer_is_written(self):
        book = Ledger(self.dir / "scored.jsonl")
        self.run_test_split(book)
        generator = Generator()
        with self.assertRaises(AlreadyScored):
            self.run_test_split(book, generator=generator, name="again")
        self.assertEqual(generator.prompts, [])
        self.assertFalse((self.dir / "again").exists())

    def test_a_stopped_test_run_continues_without_a_second_claim(self):
        book = Ledger(self.dir / "scored.jsonl")
        questions = [question(f"q{i}") for i in range(8)]
        with self.assertRaises(RunAborted):
            self.run_test_split(book, generator=Generator([RuntimeError("boom")]), questions=questions)
        self.assertEqual([e["event"] for e in book.events()], ["claim"])
        summary = self.run_test_split(book, questions=questions)  # the same settings: the same run continues
        self.assertEqual((summary.written, summary.already_done), (8, 0))
        self.assertEqual([e["event"] for e in book.events()], ["claim", "finish"])  # one claim, then finished

    def test_a_run_with_errors_is_not_marked_finished(self):
        book = Ledger(self.dir / "scored.jsonl")
        summary = self.run_test_split(book, generator=Generator([RuntimeError("once"), "the answer is (A)"]))
        self.assertEqual(len(summary.errors), 1)
        self.assertEqual([e["event"] for e in book.events()], ["claim"])
        self.run_test_split(book)
        self.assertEqual([e["event"] for e in book.events()], ["claim", "finish"])

    def test_the_ledger_is_not_used_on_validation(self):
        book = Ledger(self.dir / "scored.jsonl")
        runner.run_arms(
            [question()], [Arm("rag2", 2)], snippets, Generator(), AnswerStore(self.dir / "val"),
            {**self.CONFIG, "split": "validation"}, records.RELEVANT, ledger=book, system="rag2",
        )
        self.assertEqual(book.events(), [])


if __name__ == "__main__":
    unittest.main()
