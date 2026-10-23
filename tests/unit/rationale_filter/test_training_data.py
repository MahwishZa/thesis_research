"""rationale_filter.training_data: the records of the released format, the matching of labels to their pairs, the file."""

import json
import unittest

from rationale_filter import training_data
from rationale_filter.label_records import LabelledPair
from rationale_filter.labelling import PairTask, hashes
from rationale_filter.training_data import TrainingDataError
from tests.spec.targets import DISCARD, FILTER_INSTRUCTION, HELPFUL, NOT_HELPFUL, TOKEN_HELPFUL, TOKEN_NOT_HELPFUL
from tests.support import TempDirTestCase
from tests.unit.rationale_filter.test_label_records import ok

QUESTION = "Which drug? A) one B) two C) three D) four"


def task_for(question_id, rank, snippet="A snippet of text."):
    return PairTask(question_id, QUESTION, "A", "the answer is (B)", rank, snippet)


def labelled(task, label):
    question_hash, snippet_hash = hashes(task)
    return LabelledPair(ok(task.question_id, task.rank, question_sha256=question_hash, snippet_sha256=snippet_hash), label)


class RecordTests(unittest.TestCase):
    def test_a_record_has_the_four_fields_of_the_released_sample(self):
        task = task_for("q1", 2, "  The snippet.\n")
        (record,) = training_data.training_records([labelled(task, HELPFUL)], [task], "medqa")
        self.assertEqual(list(record), ["id", "answer", "dataset_name", "question"])
        self.assertEqual(record["id"], "medqa_q1#2")
        self.assertEqual(record["answer"], TOKEN_HELPFUL)
        self.assertEqual(record["dataset_name"], "medqa")
        self.assertEqual(record["question"], FILTER_INSTRUCTION + "\n\nEvidence: The snippet.\n\nQuestion: " + QUESTION)

    def test_discarded_pairs_have_no_record_and_the_others_keep_their_order(self):
        tasks = [task_for("a", 1), task_for("b", 1), task_for("c", 1)]
        items = [labelled(tasks[0], NOT_HELPFUL), labelled(tasks[1], DISCARD), labelled(tasks[2], HELPFUL)]
        records = training_data.training_records(items, tasks, "medqa")
        self.assertEqual([r["answer"] for r in records], [TOKEN_NOT_HELPFUL, TOKEN_HELPFUL])
        self.assertEqual([r["id"] for r in records], ["medqa_a#1", "medqa_c#1"])

    def test_a_label_without_its_task_is_refused(self):
        task = task_for("q1", 1)
        with self.assertRaises(TrainingDataError) as caught:
            training_data.training_records([labelled(task, HELPFUL)], [], "medqa")
        self.assertIn("q1#1", str(caught.exception))

    def test_a_label_made_from_another_snippet_or_question_is_refused(self):
        made_from = task_for("q1", 1, "The snippet the label was made from.")
        other_snippet = task_for("q1", 1, "A different snippet.")
        with self.assertRaises(TrainingDataError):
            training_data.training_records([labelled(made_from, HELPFUL)], [other_snippet], "medqa")
        other_gold = PairTask("q1", QUESTION, "B", "x", 1, made_from.snippet)
        with self.assertRaises(TrainingDataError):
            training_data.training_records([labelled(made_from, HELPFUL)], [other_gold], "medqa")

    def test_a_discarded_pair_does_not_need_its_task(self):
        task = task_for("q1", 1)
        self.assertEqual(training_data.training_records([labelled(task, DISCARD)], [], "medqa"), [])

    def test_the_dataset_name_must_be_text(self):
        for bad in ("", None, 5):
            with self.subTest(bad=bad), self.assertRaises(TrainingDataError):
                training_data.training_records([], [], bad)


class FileTests(TempDirTestCase):
    def records(self, n=3):
        return [
            {"id": f"medqa_q{i}#1", "answer": TOKEN_HELPFUL if i % 2 else TOKEN_NOT_HELPFUL, "dataset_name": "medqa", "question": f"text {i}"}
            for i in range(n)
        ]

    def test_the_file_is_a_json_list_that_reads_back_the_same(self):
        path = training_data.write_training_file(self.dir / "labels.json", self.records())
        self.assertEqual(training_data.read_training_file(path), self.records())
        self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), list)

    def test_an_existing_file_is_never_replaced(self):
        path = self.write("labels.json", "[]")
        with self.assertRaises(TrainingDataError) as caught:
            training_data.write_training_file(path, self.records())
        self.assertIn("refusing to overwrite", str(caught.exception))
        self.assertEqual(path.read_text(encoding="utf-8"), "[]")

    def test_refused_records_leave_no_file(self):
        target = self.dir / "labels.json"
        base = self.records(1)[0]
        for name, bad in {
            "extra field": [{**base, "rule": "x"}],
            "missing field": [{k: v for k, v in base.items() if k != "id"}],
            "fields in another order": [{"answer": base["answer"], "id": base["id"], "dataset_name": "medqa", "question": "q"}],
            "unknown token": [{**base, "answer": "helpful"}],
            "repeated id": [base, dict(base)],
        }.items():
            with self.subTest(name), self.assertRaises(TrainingDataError):
                training_data.write_training_file(target, bad)
            self.assertFalse(target.exists())

    def test_a_missing_or_broken_file_is_a_training_data_error(self):
        broken = self.write("broken.json", "{")
        not_a_list = self.write("object.json", "{}")
        for path in (self.dir / "missing.json", broken, not_a_list):
            with self.subTest(path=path.name), self.assertRaises(TrainingDataError):
                training_data.read_training_file(path)

    def test_the_balance_reports_the_larger_class(self):
        balance = training_data.label_balance(self.records(5))
        self.assertEqual((balance[TOKEN_HELPFUL], balance[TOKEN_NOT_HELPFUL], balance["majority_share"]), (2, 3, 0.6))
        self.assertIsNone(training_data.label_balance([])["majority_share"])

    def test_records_from_labelled_pairs_can_be_written(self):
        tasks = [task_for(f"q{i}", 1) for i in range(3)]
        pairs = [labelled(t, HELPFUL if i else NOT_HELPFUL) for i, t in enumerate(tasks)]
        path = training_data.write_training_file(self.dir / "out.json", training_data.training_records(pairs, tasks, "medqa"))
        self.assertEqual(len(training_data.read_training_file(path)), 3)


if __name__ == "__main__":
    unittest.main()
