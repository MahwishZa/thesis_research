"""rationale_filter.labelling: one pair, the loop, resuming, and the refusals, with a scripted generator and scorer."""

import math
import unittest

from rationale_filter import label_records, labelling
from rationale_filter.label_records import PairStore
from rationale_filter.labelling import PairTask
from rationale_query.batch import MAX_CONSECUTIVE_ERRORS, RunAborted
from rationale_query.generator import GenerationResult
from rationale_query.store import QuestionChanged, StoreError
from tests.support import TempDirTestCase

STEM = QUESTION = "Which drug is a cholinesterase inhibitor? A) Donepezil B) Aspirin C) Insulin D) Warfarin"
RIGHT = "Therefore, the answer is (A) Donepezil."
WRONG = "Therefore, the answer is (B) Aspirin."


def task(rank=1, *, question_id="q1", rationale=WRONG, gold="A", snippet="Donepezil inhibits cholinesterase."):
    return PairTask(question_id, QUESTION, gold, rationale, rank, snippet)


class Generator:
    config = {"model_sha256": "m" * 64}

    def __init__(self, answers):
        self.answers = answers
        self.prompts = []

    def generate(self, user_message):
        self.prompts.append(user_message)
        answer = self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]
        if isinstance(answer, BaseException):
            raise answer
        return GenerationResult(text=answer, finish_reason="stop", prompt_tokens=10, completion_tokens=5)


class Scorer:
    config = {"model_sha256": "m" * 64}

    def __init__(self, without=-1.0, with_snippet=-0.5, tokens=4):
        self.calls = []
        self.without, self.with_snippet, self.tokens = without, with_snippet, tokens

    def token_logprobs(self, user_message, completion):
        self.calls.append(user_message)
        return [self.with_snippet if "Evidence:" in user_message else self.without] * self.tokens


class ScorePairTests(unittest.TestCase):
    def test_a_pair_with_two_readable_answers_gets_both_perplexities(self):
        record = labelling.score_pair(task(), Generator([RIGHT]), Scorer())
        self.assertEqual(record.status, "ok")
        self.assertEqual((record.correct_without, record.correct_with), (False, True))
        self.assertAlmostEqual(record.ppl_without, math.e)
        self.assertAlmostEqual(record.ppl_with, math.exp(0.5))
        self.assertEqual(record.generation_with, RIGHT)
        self.assertEqual(record.id, "q1#1")

    def test_the_answer_with_the_snippet_is_written_from_the_answer_prompt(self):
        generator = Generator([RIGHT])
        labelling.score_pair(task(snippet="  A snippet. "), generator, Scorer())
        self.assertEqual(len(generator.prompts), 1)
        self.assertIn("Evidence:\nA snippet.\nHere is the question: " + QUESTION, generator.prompts[0])

    def test_the_rationale_is_scored_under_the_prompt_without_and_the_prompt_with_the_snippet(self):
        scorer = Scorer()
        labelling.score_pair(task(), Generator([RIGHT]), scorer)
        self.assertEqual(len(scorer.calls), 2)
        self.assertNotIn("Evidence:", scorer.calls[0])
        self.assertIn("Evidence:", scorer.calls[1])

    def test_an_unreadable_rationale_skips_the_pair_before_anything_is_generated(self):
        generator, scorer = Generator([RIGHT]), Scorer()
        record = labelling.score_pair(task(rationale="no clear answer"), generator, scorer)
        self.assertEqual((record.status, record.reason), ("skipped", "without_unreadable"))
        self.assertEqual((generator.prompts, scorer.calls), ([], []))

    def test_an_unreadable_answer_with_the_snippet_skips_the_pair_before_scoring_and_keeps_the_text(self):
        scorer = Scorer()
        record = labelling.score_pair(task(), Generator(["I think it is probably donepezil."]), scorer)
        self.assertEqual((record.status, record.reason), ("skipped", "with_unreadable"))
        self.assertEqual(record.generation_with, "I think it is probably donepezil.")
        self.assertEqual(record.correct_without, False)
        self.assertEqual(scorer.calls, [])

    def test_a_failing_generator_gives_an_error_record_not_an_exception(self):
        record = labelling.score_pair(task(), Generator([RuntimeError("out of memory")]), Scorer())
        self.assertEqual(record.status, "error")
        self.assertIn("RuntimeError: out of memory", record.error)

    def test_an_empty_answer_is_an_error(self):
        record = labelling.score_pair(task(), Generator(["  "]), Scorer())
        self.assertEqual(record.status, "error")
        self.assertIn("no text", record.error)

    def test_different_token_counts_under_the_two_prompts_are_an_error(self):
        class Uneven(Scorer):
            def token_logprobs(self, user_message, completion):
                return [-1.0] * (3 if "Evidence:" in user_message else 4)

        record = labelling.score_pair(task(), Generator([RIGHT]), Uneven())
        self.assertEqual(record.status, "error")
        self.assertIn("different number of tokens", record.error)

    def test_the_scoring_without_a_snippet_is_done_once_per_question_when_a_cache_is_given(self):
        scorer, cache = Scorer(), {}
        for rank in (1, 2, 3):
            labelling.score_pair(task(rank), Generator([RIGHT]), scorer, without_cache=cache)
        without_calls = [call for call in scorer.calls if "Evidence:" not in call]
        self.assertEqual(len(without_calls), 1)
        self.assertEqual(len(scorer.calls), 4)

    def test_the_hashes_follow_the_question_the_answer_key_and_the_snippet(self):
        base = labelling.hashes(task())
        self.assertNotEqual(base, labelling.hashes(task(gold="B")))
        self.assertNotEqual(base, labelling.hashes(task(snippet="Another snippet.")))
        self.assertEqual(base, labelling.hashes(task(snippet="  Donepezil inhibits cholinesterase.\n")))


class LoopTests(TempDirTestCase):
    def run_loop(self, tasks, generator=None, scorer=None, config=None, name="run", **options):
        store = PairStore(self.dir / name)
        generator = generator or Generator([RIGHT])
        summary = labelling.label_pairs(
            tasks, generator, scorer or Scorer(), store, config or {"dataset": "medqa", "snippets_per_question": 2}, label_records.RELEVANT, **options
        )
        return summary, generator, store

    def test_every_pair_gets_a_record_and_a_second_run_does_nothing(self):
        tasks = [task(1), task(2), task(1, question_id="q2")]
        summary, generator, store = self.run_loop(tasks)
        self.assertEqual((summary.total, summary.written, summary.already_done), (3, 3, 0))
        self.assertEqual(sorted(store.load()), ["q1#1", "q1#2", "q2#1"])
        again, generator2, _ = self.run_loop(tasks)
        self.assertEqual((again.written, again.already_done), (0, 3))
        self.assertEqual(generator2.prompts, [])

    def test_a_stopped_run_continues_where_it_stopped(self):
        tasks = [task(1), task(2), task(3)]
        self.run_loop(tasks[:2])
        summary, generator, store = self.run_loop(tasks)
        self.assertEqual((summary.written, summary.already_done), (1, 2))
        self.assertEqual(len(generator.prompts), 1)

    def test_an_error_is_tried_again_by_the_next_run(self):
        tasks = [task(1), task(2)]
        summary, _, store = self.run_loop(tasks, Generator([RuntimeError("boom"), RIGHT]))
        self.assertEqual(summary.errors[0][0], "q1#1")
        again, generator, store = self.run_loop(tasks, Generator([RIGHT]))
        self.assertEqual((again.written, again.already_done), (1, 1))
        self.assertEqual(store.load()["q1#1"].status, "ok")

    def test_five_failures_in_a_row_stop_the_run(self):
        tasks = [task(rank) for rank in range(1, 9)]
        with self.assertRaises(RunAborted):
            self.run_loop(tasks, Generator([RuntimeError("boom")]))
        self.assertEqual(len(PairStore(self.dir / "run").load()), MAX_CONSECUTIVE_ERRORS)

    def test_a_changed_snippet_or_answer_key_is_refused_before_any_work(self):
        self.run_loop([task(1)])
        for changed in (task(1, snippet="A different snippet."), task(1, gold="B")):
            with self.subTest(changed=changed), self.assertRaises(QuestionChanged):
                self.run_loop([changed], Generator([RIGHT]))

    def test_another_setting_is_refused(self):
        from rationale_query.store import ConfigMismatch

        self.run_loop([task(1)], config={"dataset": "medqa", "snippets_per_question": 2})
        with self.assertRaises(ConfigMismatch):
            self.run_loop([task(1)], config={"dataset": "medqa", "snippets_per_question": 4})

    def test_progress_is_reported_for_every_new_record(self):
        seen = []
        self.run_loop([task(1), task(2)], progress=lambda number, total, record: seen.append((number, total, record.id)))
        self.assertEqual(seen, [(1, 2, "q1#1"), (2, 2, "q1#2")])

    def test_the_labels_can_be_made_from_what_the_loop_wrote(self):
        tasks = [task(1), task(2, rationale=RIGHT), task(3, rationale="unclear")]
        _, _, store = self.run_loop(tasks)
        records = store.load()
        tau, labelled = label_records.label_records(records)
        self.assertEqual(len(labelled), 2)
        self.assertEqual(records["q1#3"].status, "skipped")


def write_inputs(case):
    """Write what the earlier steps leave behind: rationales (Part 1), evidence lists (Part 2) and an answer key, into ``case.dir``."""
    from balanced_retrieval.records import RetrievalRecord, RetrievalStore
    from rationale_query.generator import text_sha256
    from rationale_query.search_text import dense_query
    from rationale_query.store import Question, RationaleRecord, RationaleStore, question_hash

    case.Question = Question
    case.questions = [Question("q1", STEM), Question("q2", STEM + " Extra.")]
    rationales = RationaleStore(case.dir / "rationales")
    rationales.bind({"model_sha256": "m"}, ())
    rationales.load()
    retrieved = RetrievalStore(case.dir / "retrieved")
    retrieved.bind({"top_k": 4}, ())
    retrieved.load()
    case.rationale = {question.id: f"Rationale for {question.id}. Therefore, the answer is (A) Donepezil." for question in case.questions}
    for question in case.questions:
        rationales.append(RationaleRecord(id=question.id, question_sha256=question_hash(question.text), status="ok", rationale=case.rationale[question.id]))
        retrieved.append(
            RetrievalRecord(
                id=question.id, question_sha256=question_hash(question.text),
                rationale_sha256=text_sha256(dense_query(case.rationale[question.id])), pool_size=8,
                evidence=[{"corpus": "textbooks", "text": f"snippet {n} of {question.id}"} for n in range(1, 4)],
            )
        )
    case.answers = case.write("answers.jsonl", '{"id": "q1", "answer": "A"}\n{"id": "q2", "answer": "C"}\n')


class FileGlueTests(TempDirTestCase):
    """read_answers and build_tasks: the files of the earlier steps become pair tasks, or the call stops and says why."""

    def setUp(self):
        super().setUp()
        write_inputs(self)

    def tasks(self, snippets=2, questions=None):
        return labelling.build_tasks(
            questions or self.questions, labelling.read_answers(self.answers), self.dir / "rationales", self.dir / "retrieved", snippets
        )

    def test_the_answers_are_read_as_a_dictionary(self):
        self.assertEqual(labelling.read_answers(self.answers), {"q1": "A", "q2": "C"})

    def test_refused_answer_files(self):
        for name, text in {
            "letter outside": '{"id": "q1", "answer": "E"}\n',
            "no id": '{"answer": "A"}\n',
            "repeated id": '{"id": "q1", "answer": "A"}\n{"id": "q1", "answer": "B"}\n',
            "broken line": "{broken\n",
            "empty": "\n",
        }.items():
            path = self.write(f"{name}.jsonl", text)
            with self.subTest(name), self.assertRaises(StoreError):
                labelling.read_answers(path)
        with self.assertRaises(StoreError):
            labelling.read_answers(self.dir / "missing.jsonl")

    def test_one_task_per_question_and_rank_in_the_order_of_the_list(self):
        tasks = self.tasks(snippets=2)
        self.assertEqual([task.id for task in tasks], ["q1#1", "q1#2", "q2#1", "q2#2"])
        self.assertEqual(tasks[1].snippet, "snippet 2 of q1")
        self.assertEqual((tasks[2].gold, tasks[2].question, tasks[2].rationale), ("C", STEM + " Extra.", self.rationale["q2"]))

    def test_a_question_with_fewer_snippets_than_asked_gets_as_many_as_it_has(self):
        self.assertEqual(len(self.tasks(snippets=10)), 6)

    def test_a_missing_answer_key_evidence_or_rationale_stops_everything(self):
        extra = self.questions + [self.Question("q3", "A third question? A) a B) b C) c D) d")]
        with self.assertRaises(StoreError):
            self.tasks(questions=extra)  # the rationales folder has no q3

    def test_evidence_found_for_another_rationale_is_refused(self):
        from rationale_query.store import RationaleRecord, RationaleStore, question_hash

        folder = RationaleStore(self.dir / "rationales2")
        folder.bind({"model_sha256": "m"}, ())
        folder.load()
        folder.append(RationaleRecord(id="q1", question_sha256=question_hash(STEM), status="ok", rationale="Another rationale. the answer is (B)"))
        with self.assertRaises(StoreError) as caught:
            labelling.build_tasks([self.questions[0]], {"q1": "A"}, self.dir / "rationales2", self.dir / "retrieved", 2)
        self.assertIn("another question or rationale", str(caught.exception))

    def test_a_missing_answer_key_for_a_question_is_named(self):
        with self.assertRaises(StoreError) as caught:
            labelling.build_tasks(self.questions, {"q1": "A"}, self.dir / "rationales", self.dir / "retrieved", 2)
        self.assertIn("q2 has no answer key", str(caught.exception))

    def test_bad_snippet_counts_are_refused(self):
        for bad in (0, -1, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.tasks(snippets=bad)


if __name__ == "__main__":
    unittest.main()
