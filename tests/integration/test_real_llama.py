"""rationale_query.generator with the real llama.cpp, on tiny made-up models (needs llama-cpp-python, gguf and numpy)."""

import io
import json
import unittest

from rationale_query import cli
from rationale_query.generator import GenerationSettings, LlamaGenerator, render_user_message
from rationale_query.prompt import build_rationale_prompt
from rationale_query.store import RationaleStore
from tests.support import TempDirTestCase
from tests.unit.rationale_query.examples import FIGURE_4_QUESTION

try:
    import gguf  # noqa: F401
    import llama_cpp  # noqa: F401
    import numpy  # noqa: F401
except ImportError as exc:
    LIBRARIES = False
    REASON = f"needs llama-cpp-python, gguf and numpy ({exc})"
else:
    LIBRARIES = True
    REASON = ""
    from tests.integration.tiny_models import write_tiny_llama


@unittest.skipUnless(LIBRARIES, REASON)
class RealLlamaTests(TempDirTestCase):
    WINDOW = 1024

    def make(self, eos_push):
        path = write_tiny_llama(self.dir / f"tiny_{eos_push}.gguf", eos_push=eos_push, context_length=self.WINDOW)
        generator = LlamaGenerator(path, GenerationSettings(n_ctx=self.WINDOW, n_threads=1))
        self.addCleanup(generator.close)
        return generator

    def test_the_start_token_stands_once_at_the_beginning(self):
        generator = self.make(-5.0)
        tokens = generator.prompt_tokens(build_rationale_prompt(FIGURE_4_QUESTION))
        self.assertEqual(tokens[0], generator.llm.token_bos())
        self.assertNotIn(generator.llm.token_bos(), tokens[1:])

    def test_the_tokens_spell_meerkats_format_and_the_paper_prompt(self):
        generator = self.make(-5.0)
        prompt = build_rationale_prompt("Which? (A) a (B) b")
        tokens = generator.prompt_tokens(prompt)
        spelled = generator.llm.detokenize(tokens[1:]).decode("utf-8")
        self.assertEqual(spelled.lstrip(" "), render_user_message(prompt))

    def test_a_model_that_wants_to_stop_stops_and_reports_it(self):
        generator = self.make(+5.0)
        result = generator.generate(build_rationale_prompt("Which? (A) a (B) b"))
        self.assertEqual(result.finish_reason, "stop")
        self.assertLess(result.prompt_tokens + result.completion_tokens, self.WINDOW)

    def test_a_model_that_never_stops_writes_until_the_window_is_full(self):
        generator = self.make(-5.0)
        result = generator.generate(build_rationale_prompt("Which? (A) a (B) b"))
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.prompt_tokens + result.completion_tokens, self.WINDOW)

    def test_greedy_decoding_gives_the_same_text_every_time_and_after_another_question(self):
        generator = self.make(-5.0)
        prompt = build_rationale_prompt("Which? (A) a (B) b")
        first = generator.generate(prompt)
        generator.generate(build_rationale_prompt("A different question? (A) c (B) d"))
        again = generator.generate(prompt)
        self.assertEqual(first, again)


@unittest.skipUnless(LIBRARIES, REASON)
class RealCommandLineTests(TempDirTestCase):
    """The commands with the real llama.cpp: questions in, saved rationales, the query file for the retriever."""

    def run_cli(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        status = cli.main(list(arguments), streams=cli.Streams(out, err))
        return status, out.getvalue(), err.getvalue()

    def test_a_model_that_never_stops_gives_rationales_that_fill_the_window_and_the_queries_can_be_exported(self):
        questions = self.write("q.jsonl", "".join(json.dumps({"id": i, "question": q}) + "\n" for i, q in [("q1", FIGURE_4_QUESTION), ("q2", "Which? (A) a (B) b")]))
        model = write_tiny_llama(self.dir / "never.gguf", eos_push=-5.0, context_length=1024)
        status, out, err = self.run_cli(
            "generate", "--questions", str(questions), "--out", str(self.dir / "run"), "--model", str(model), "--n-ctx", "1024", "--n-threads", "1", "--quiet"
        )
        self.assertEqual(status, 0, err)
        self.assertIn("2 questions: 0 already done, 2 written, 0 failed, 2 stopped by the memory window", out)
        records = RationaleStore(self.dir / "run").load()
        self.assertEqual({r.finish_reason for r in records.values()}, {"length"})
        self.assertEqual(json.loads((self.dir / "run" / "config.json").read_text(encoding="utf-8"))["n_ctx"], 1024)
        target = self.dir / "queries.json"
        status, out, _ = self.run_cli("export-queries", "--questions", str(questions), "--rationales", str(self.dir / "run"), "--out", str(target))
        self.assertEqual(status, 0)
        queries = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(queries, [records["q1"].rationale.strip(), records["q2"].rationale.strip()])

    def test_a_model_that_writes_nothing_is_a_failure_and_the_question_is_not_used_instead(self):
        questions = self.write("q.jsonl", json.dumps({"id": "q1", "question": "Which? (A) a (B) b"}) + "\n")
        model = write_tiny_llama(self.dir / "silent.gguf", eos_push=+5.0, context_length=1024)
        status, out, _ = self.run_cli(
            "generate", "--questions", str(questions), "--out", str(self.dir / "run"), "--model", str(model), "--n-ctx", "1024", "--n-threads", "1", "--quiet"
        )
        self.assertEqual(status, 2)
        self.assertIn("empty rationale", out)
        record = RationaleStore(self.dir / "run").load()["q1"]
        self.assertEqual((record.status, record.rationale), ("error", None))

    def test_a_folder_made_with_one_model_refuses_another(self):
        questions = self.write("q.jsonl", json.dumps({"id": "q1", "question": "Which? (A) a (B) b"}) + "\n")
        first = write_tiny_llama(self.dir / "first.gguf", eos_push=-5.0, context_length=1024)
        second = write_tiny_llama(self.dir / "second.gguf", eos_push=-4.0, context_length=1024)
        base = ["generate", "--questions", str(questions), "--out", str(self.dir / "run"), "--n-ctx", "1024", "--n-threads", "1", "--quiet"]
        self.assertEqual(self.run_cli(*base, "--model", str(first))[0], 0)
        status, _, err = self.run_cli(*base, "--model", str(second))
        self.assertEqual(status, 1)
        self.assertIn("other settings", err)


if __name__ == "__main__":
    unittest.main()
