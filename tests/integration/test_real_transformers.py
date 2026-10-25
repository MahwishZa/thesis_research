"""rationale_query with the real Transformers library, on a tiny made-up Mistral model (needs torch, transformers and accelerate)."""

import io
import json
import unittest

from rationale_query import cli
from rationale_query.generator import ContextOverflowError, GenerationSettings, TransformersGenerator
from rationale_query.prompt import build_rationale_prompt
from rationale_query.store import RationaleStore
from tests.integration.tiny_models import split_words
from tests.support import TempDirTestCase
from tests.unit.rationale_query.examples import FIGURE_4_QUESTION

try:
    import accelerate  # noqa: F401
    import torch
    import transformers  # noqa: F401
except ImportError as exc:
    LIBRARIES = False
    REASON = f"needs torch, transformers and accelerate ({exc})"
else:
    LIBRARIES = True
    REASON = ""
    from tests.integration.tiny_models import write_tiny_mistral

QUESTION = "Which drug is best? (A) a (B) b"
TEXTS = [build_rationale_prompt(FIGURE_4_QUESTION), build_rationale_prompt(QUESTION), "USER: ASSISTANT:"]
WINDOW = 600


@unittest.skipUnless(LIBRARIES, REASON)
class RealTransformersTests(TempDirTestCase):
    def make(self, eos_push, **settings):
        folder = write_tiny_mistral(self.dir / f"tiny_{eos_push}", TEXTS, eos_push=eos_push)
        generator = TransformersGenerator(folder, GenerationSettings(n_ctx=settings.pop("n_ctx", WINDOW), n_threads=1, **settings))
        self.addCleanup(generator.close)
        return generator

    def test_the_model_runs_in_bf16(self):
        generator = self.make(-5.0)
        self.assertEqual(next(generator.model.parameters()).dtype, torch.bfloat16)

    def test_the_start_token_stands_once_and_the_tokens_spell_the_papers_prompt_and_nothing_else(self):
        generator = self.make(-5.0)
        prompt = build_rationale_prompt(FIGURE_4_QUESTION)
        ids = generator.prompt_tokens(prompt)
        self.assertEqual(ids[0], generator.tokenizer.bos_token_id)
        self.assertNotIn(generator.tokenizer.bos_token_id, ids[1:])
        self.assertEqual(generator.tokenizer.convert_ids_to_tokens(ids[1:]), split_words(prompt))

    def test_meerkats_template_is_added_only_when_asked_for(self):
        generator = self.make(-5.0, chat_format="meerkat")
        prompt = build_rationale_prompt(QUESTION)
        spelled = generator.tokenizer.convert_ids_to_tokens(generator.prompt_tokens(prompt)[1:])
        self.assertEqual(spelled, split_words("USER: " + prompt + " ASSISTANT: "))

    def test_a_model_that_never_stops_writes_until_the_window_is_full(self):
        generator = self.make(-5.0)
        result = generator.generate(build_rationale_prompt(QUESTION))
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.prompt_tokens + result.completion_tokens, WINDOW)

    def test_a_model_that_wants_to_stop_stops_and_reports_it(self):
        generator = self.make(+5.0)
        result = generator.generate(build_rationale_prompt(QUESTION))
        self.assertEqual(result.finish_reason, "stop")
        self.assertLess(result.prompt_tokens + result.completion_tokens, WINDOW)

    def test_greedy_decoding_gives_the_same_text_every_time_and_after_another_question(self):
        generator = self.make(-5.0)
        prompt = build_rationale_prompt(QUESTION)
        first = generator.generate(prompt)
        generator.generate(build_rationale_prompt(FIGURE_4_QUESTION))
        self.assertEqual(generator.generate(prompt), first)

    def test_progress_is_reported_every_25_tokens_and_changes_nothing(self):
        generator = self.make(-5.0)
        prompt = build_rationale_prompt(QUESTION)
        without = generator.generate(prompt)
        seen = []
        generator.progress = seen.append
        with_report = generator.generate(prompt)
        self.assertEqual(with_report, without)
        self.assertEqual(seen, list(range(25, with_report.completion_tokens + 1, 25)))

    def test_a_prompt_that_does_not_fit_the_window_is_refused(self):
        generator = self.make(-5.0, n_ctx=20)
        with self.assertRaises(ContextOverflowError):
            generator.generate(build_rationale_prompt(FIGURE_4_QUESTION))


@unittest.skipUnless(LIBRARIES, REASON)
class RealCommandLineTests(TempDirTestCase):
    """The commands with the real library: questions in, saved rationales, the query file for the retriever."""

    def run_cli(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        status = cli.main(list(arguments), streams=cli.Streams(out, err))
        return status, out.getvalue(), err.getvalue()

    def questions(self):
        rows = [("q1", FIGURE_4_QUESTION), ("q2", QUESTION)]
        return self.write("q.jsonl", "".join(json.dumps({"id": i, "question": q}) + "\n" for i, q in rows))

    def test_generate_then_export_with_a_model_that_never_stops(self):
        questions = self.questions()
        model = write_tiny_mistral(self.dir / "never", TEXTS, eos_push=-5.0)
        status, out, err = self.run_cli("generate", "--questions", str(questions), "--out", str(self.dir / "run"), "--model", str(model), "--n-ctx", str(WINDOW), "--quiet")
        self.assertEqual(status, 0, err)
        self.assertIn("2 questions: 0 already done, 2 written, 0 failed, 2 stopped by the memory window", out)
        self.assertIn("model: never, hash ", out)
        records = RationaleStore(self.dir / "run").load()
        self.assertEqual({r.finish_reason for r in records.values()}, {"length"})
        config = json.loads((self.dir / "run" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((config["dtype"], config["chat_format"], config["n_ctx"]), ("bfloat16", "none", WINDOW))
        target = self.dir / "queries.json"
        status, _, _ = self.run_cli("export-queries", "--questions", str(questions), "--rationales", str(self.dir / "run"), "--out", str(target))
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), [records["q1"].rationale.strip(), records["q2"].rationale.strip()])

    def test_a_model_that_writes_nothing_is_a_failure_and_the_question_is_not_used_instead(self):
        questions = self.write("q.jsonl", json.dumps({"id": "q1", "question": QUESTION}) + "\n")
        model = write_tiny_mistral(self.dir / "silent", TEXTS, eos_push=+5.0)
        status, out, _ = self.run_cli("generate", "--questions", str(questions), "--out", str(self.dir / "run"), "--model", str(model), "--n-ctx", str(WINDOW), "--quiet")
        self.assertEqual(status, 2)
        self.assertIn("empty rationale", out)
        record = RationaleStore(self.dir / "run").load()["q1"]
        self.assertEqual((record.status, record.rationale), ("error", None))

    def test_a_folder_made_with_one_model_refuses_another_and_a_wrong_pin_is_refused(self):
        questions = self.write("q.jsonl", json.dumps({"id": "q1", "question": QUESTION}) + "\n")
        first = write_tiny_mistral(self.dir / "first", TEXTS, eos_push=-5.0)
        second = write_tiny_mistral(self.dir / "second", TEXTS, eos_push=-4.0)
        base = ["generate", "--questions", str(questions), "--out", str(self.dir / "run"), "--n-ctx", str(WINDOW), "--quiet"]
        self.assertEqual(self.run_cli(*base, "--model", str(first))[0], 0)
        status, _, err = self.run_cli(*base, "--model", str(second))
        self.assertEqual(status, 1)
        self.assertIn("other settings", err)
        status, _, err = self.run_cli(*base, "--model", str(first), "--model-sha256", "0" * 64)
        self.assertEqual(status, 1)
        self.assertIn("pinned", err)


if __name__ == "__main__":
    unittest.main()
