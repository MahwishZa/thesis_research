"""rationale_query.cli: the three commands end to end, with a scripted generator in place of llama.cpp."""

import io
import json
import unittest

from rationale_query import cli
from rationale_query.generator import GenerationResult
from rationale_query.prompt import build_rationale_prompt
from rationale_query.store import RationaleStore
from tests.support import TempDirTestCase
from tests.unit.rationale_query.examples import FIGURE_4_QUESTION

RATIONALE = "The patient has COPD. Therefore, the answer is (A) BiPAP."


class Generator:
    config = {"model_sha256": "m" * 64, "chat_format": "meerkat", "n_ctx": 2048, "sampling": {"temperature": 0.0}}

    def __init__(self, fail_on=()):
        self.fail_on = fail_on
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if any(text in prompt for text in self.fail_on):
            raise RuntimeError("out of memory")
        reason = "length" if "Third" in prompt else "stop"
        return GenerationResult(text=RATIONALE, finish_reason=reason, prompt_tokens=300, completion_tokens=20)


class CliTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        rows = [("q1", FIGURE_4_QUESTION), ("q2", "Second? (A) x (B) y"), ("q3", "Third? (A) x (B) y")]
        self.questions = self.write("q.jsonl", "".join(json.dumps({"id": i, "question": q}) + "\n" for i, q in rows))
        self.model = self.write("model.gguf", "x")

    def run_cli(self, *arguments, generator=None):
        out, err = io.StringIO(), io.StringIO()
        status = cli.main(list(arguments), generator_factory=lambda args, streams: generator or Generator(), streams=cli.Streams(out, err))
        return status, out.getvalue(), err.getvalue()

    def generate(self, *extra, generator=None):
        return self.run_cli(
            "generate", "--questions", str(self.questions), "--out", str(self.dir / "run"), "--model", str(self.model), "--quiet", *extra,
            generator=generator,
        )

    def test_generate_writes_a_rationale_for_every_question_and_reports_the_ones_cut_by_the_window(self):
        status, out, err = self.generate()
        self.assertEqual((status, err), (0, ""))
        self.assertIn("3 questions: 0 already done, 3 written, 0 failed, 1 stopped by the memory window", out)
        self.assertEqual(sorted(RationaleStore(self.dir / "run").load()), ["q1", "q2", "q3"])

    def test_a_second_run_does_nothing_and_a_limit_cuts_the_questions(self):
        self.generate("--limit", "1")
        self.assertEqual(sorted(RationaleStore(self.dir / "run").load()), ["q1"])
        status, out, _ = self.generate()
        self.assertEqual(status, 0)
        self.assertIn("1 already done, 2 written", out)

    def test_the_settings_are_bound_to_the_folder(self):
        self.generate()
        generator = Generator()
        generator.config = {**Generator.config, "n_ctx": 4096}
        status, _, err = self.generate(generator=generator)
        self.assertEqual(status, 1)
        self.assertIn("other settings", err)

    def test_failures_are_listed_and_the_status_says_to_run_again(self):
        status, out, _ = self.generate(generator=Generator(fail_on=("Second",)))
        self.assertEqual(status, 2)
        self.assertIn("failed q2: RuntimeError: out of memory", out)
        self.assertIn("run the same command again", out)

    def test_a_bad_limit_and_a_missing_questions_file_are_reported_as_errors(self):
        status, _, err = self.generate("--limit", "0")
        self.assertEqual(status, 1)
        self.assertIn("--limit", err)
        status, _, err = self.run_cli("generate", "--questions", str(self.dir / "none.jsonl"), "--out", str(self.dir / "r"), "--model", str(self.model))
        self.assertEqual(status, 1)
        self.assertIn("cannot read", err)

    def test_show_prompt_prints_the_papers_prompt_and_what_the_model_reads(self):
        status, out, _ = self.run_cli("show-prompt", "--question", FIGURE_4_QUESTION)
        self.assertEqual(status, 0)
        self.assertIn(build_rationale_prompt(FIGURE_4_QUESTION), out)
        self.assertIn("USER: " + build_rationale_prompt(FIGURE_4_QUESTION) + " ASSISTANT: ", out)
        self.assertIn("no system message", out)

    def test_export_queries_writes_the_list_in_the_order_of_the_questions(self):
        self.generate()
        target = self.dir / "queries.json"
        status, out, _ = self.run_cli("export-queries", "--questions", str(self.questions), "--rationales", str(self.dir / "run"), "--out", str(target))
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), [RATIONALE] * 3)
        self.assertIn("wrote 3 queries", out)
        self.assertIn("1 of them filled the memory window", out)

    def test_export_refuses_to_overwrite_and_refuses_an_unfinished_run(self):
        status, _, err = self.run_cli("export-queries", "--questions", str(self.questions), "--rationales", str(self.dir / "run"), "--out", str(self.dir / "q.json"))
        self.assertEqual(status, 1)
        self.assertIn("generate command first", err)
        self.generate("--limit", "1")
        status, _, err = self.run_cli("export-queries", "--questions", str(self.questions), "--rationales", str(self.dir / "run"), "--out", str(self.dir / "q.json"))
        self.assertEqual(status, 1)
        self.assertIn("q2", err)
        self.generate()
        self.write("taken.json", "[]")
        status, _, err = self.run_cli("export-queries", "--questions", str(self.questions), "--rationales", str(self.dir / "run"), "--out", str(self.dir / "taken.json"))
        self.assertEqual(status, 1)
        self.assertIn("does not overwrite", err)


if __name__ == "__main__":
    unittest.main()
