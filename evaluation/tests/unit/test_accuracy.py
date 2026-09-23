"""Tests for the QA accuracy judgement schema (Step 9)."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation import accuracy as acc


def judgment(**kw):
    base = dict(
        question_id="ADQ-1",
        system="baseline",
        question="Does donepezil improve cognition in mild AD?",
        generated_answer="Yes, modestly.",
        reference_answer="Yes, a modest average benefit.",
        correct=1,
        judge="med-annotator-1",
    )
    base.update(kw)
    return acc.QAJudgment(**base)


class QAJudgmentSchemaTests(unittest.TestCase):

    def test_valid_judgment_constructs(self):
        j = judgment()
        self.assertEqual(j.correct, 1)

    def test_correct_must_be_binary(self):
        with self.assertRaises(acc.AccuracyError):
            judgment(correct=2)

    def test_judge_is_required(self):
        with self.assertRaises(acc.AccuracyError):
            judgment(judge="")
        with self.assertRaises(acc.AccuracyError):
            judgment(judge="   ")

    def test_judge_names_who_or_what_decided(self):
        """A rule-based judge is distinguishable from a human one by name."""
        rule = judgment(judge="exact_match")
        human = judgment(judge="med-annotator-1")
        self.assertNotEqual(rule.judge, human.judge)

    def test_to_dict_round_trips(self):
        j = judgment(note="borderline case")
        d = j.to_dict()
        self.assertEqual(d["correct"], 1)
        self.assertEqual(d["note"], "borderline case")


class QAJudgmentIOTests(unittest.TestCase):

    def test_write_then_read_preserves_every_field(self):
        original = [judgment(question_id="Q1"), judgment(question_id="Q2",
                                                          correct=0)]
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "j.jsonl")
            acc.write_judgments(original, path)
            back = acc.read_judgments(path)
        self.assertEqual(back, tuple(original))

    def test_read_validates_each_row(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text(json.dumps({
                "question_id": "Q1", "system": "baseline", "question": "q?",
                "generated_answer": "a", "reference_answer": "a",
                "correct": 1, "judge": "", "note": None,
            }) + "\n")
            with self.assertRaises(acc.AccuracyError):
                acc.read_judgments(str(path))

    def test_by_question_reshapes_for_comparison(self):
        judgments = [
            judgment(question_id="Q1", system="baseline"),
            judgment(question_id="Q1", system="proposed", correct=0),
            judgment(question_id="Q2", system="baseline", correct=0),
        ]
        grouped = acc.by_question(judgments)
        self.assertEqual(set(grouped), {"baseline", "proposed"})
        self.assertEqual(grouped["baseline"]["Q1"].correct, 1)
        self.assertEqual(grouped["baseline"]["Q2"].correct, 0)

    def test_by_question_refuses_a_duplicate(self):
        dup = [judgment(question_id="Q1"), judgment(question_id="Q1")]
        with self.assertRaises(acc.AccuracyError):
            acc.by_question(dup)


if __name__ == "__main__":
    unittest.main()
