"""answer_evaluation.cli: run and report end to end, with a scripted generator and filter in place of llama.cpp and Flan-T5."""

import io
import json
import unittest

from answer_evaluation import cli
from answer_evaluation.records import AnswerStore
from rationale_query.generator import GenerationResult
from tests.support import TempDirTestCase
from tests.unit.rationale_filter.test_labelling import write_inputs


class Generator:
    config = {"model_sha256": "m" * 64}

    def __init__(self):
        self.prompts = []

    def generate(self, user_message):
        self.prompts.append(user_message)
        text = "Therefore, the answer is (A) Donepezil." if "snippet 1" in user_message else "Therefore, the answer is (B) Aspirin."
        return GenerationResult(text=text, finish_reason="stop", prompt_tokens=50, completion_tokens=9)


class Filter:
    """Calls snippet 1 helpful and the others not helpful."""

    def judge_many(self, texts, batch_size=8):
        return ["Evidence: snippet 1" in text for text in texts]


class CliTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        write_inputs(self)
        self.questions_file = self.write("questions.jsonl", "".join(json.dumps({"id": q.id, "question": q.text}) + "\n" for q in self.questions))
        self.filter_dir = self.dir / "filter"
        self.filter_dir.mkdir()
        (self.filter_dir / "config.json").write_text("{}", encoding="utf-8")

    def run_run(self, *extra, split="validation"):
        out, err = io.StringIO(), io.StringIO()
        model = self.write("model.gguf", "x")
        arguments = [
            "run", "--questions", str(self.questions_file), "--answers", str(self.answers), "--rationales", str(self.dir / "rationales"),
            "--retrieved", f"4={self.dir / 'retrieved'}", "--out", str(self.dir / "answers"), "--dataset", "medqa", "--split", split,
            "--model", str(model), "--filter-model", str(self.filter_dir), "--quiet",
        ]
        status = cli.main(
            arguments + list(extra), generator_factory=lambda args, streams: Generator(),
            filter_factory=lambda args, streams: Filter(), streams=cli.Streams(out, err),
        )
        return status, out.getvalue(), err.getvalue()

    def run_report(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        status = cli.main(["report", "--out", str(self.dir / "answers"), *extra], streams=cli.Streams(out, err))
        return status, out.getvalue(), err.getvalue()

    def test_run_answers_every_question_with_every_system(self):
        status, out, err = self.run_run()
        self.assertEqual((status, err), (0, ""))
        self.assertIn("6 answers: 0 already done, 6 written, 0 failed", out)
        self.assertIn('"kept_share": 0.3333', out)
        arms = sorted({(r.arm, r.k) for r in AnswerStore(self.dir / "answers").load().values()})
        self.assertEqual(arms, [("no_retrieval", 0), ("rag2", 4), ("rag2_no_filter", 4)])

    def test_a_second_run_does_nothing(self):
        self.run_run()
        status, out, _ = self.run_run()
        self.assertEqual(status, 0)
        self.assertIn("6 already done, 0 written", out)

    def test_report_gives_accuracy_and_the_paired_comparison(self):
        self.run_run()
        status, out, err = self.run_report("--baseline", "rag2_no_filter", "4", "--proposed", "rag2", "4", "--iterations", "200")
        self.assertEqual((status, err), (0, ""))
        report = json.loads(out)
        self.assertEqual(len(report["accuracy"]), 3)
        self.assertIn("comparison", report)

    def test_a_test_split_needs_a_ledger_and_a_retrieval_folder_must_match_its_k(self):
        status, _, err = self.run_run(split="test")
        self.assertEqual(status, 1)
        self.assertIn("ledger", err)
        out, err = io.StringIO(), io.StringIO()
        status = cli.main(
            ["run", "--questions", str(self.questions_file), "--answers", str(self.answers), "--rationales", str(self.dir / "rationales"),
             "--retrieved", f"8={self.dir / 'retrieved'}", "--out", str(self.dir / "x"), "--dataset", "medqa", "--split", "validation",
             "--model", str(self.write("m2.gguf", "x"))],
            generator_factory=lambda a, s: Generator(), streams=cli.Streams(out, err),
        )
        self.assertEqual(status, 1)
        self.assertIn("top_k = 4", err.getvalue())

    def test_the_filter_arm_needs_a_filter_model(self):
        out, err = io.StringIO(), io.StringIO()
        status = cli.main(
            ["run", "--questions", str(self.questions_file), "--answers", str(self.answers), "--rationales", str(self.dir / "rationales"),
             "--retrieved", f"4={self.dir / 'retrieved'}", "--out", str(self.dir / "y"), "--dataset", "medqa", "--split", "validation",
             "--model", str(self.write("m3.gguf", "x"))],
            generator_factory=lambda a, s: Generator(), streams=cli.Streams(out, err),
        )
        self.assertEqual(status, 1)
        self.assertIn("--filter-model", err.getvalue())


if __name__ == "__main__":
    unittest.main()
