"""Tests of src/rationale_query/cli.py: the five commands, run in this process with stand-ins for the model."""

import contextlib
import io
import itertools
import json
import os
import re
import subprocess
import sys
import unittest
from unittest import mock

from rationale_query.cli import duration
from rationale_query.generator import GenerationSettings, llama3_prompt_text
from rationale_query.prompt import RATIONALE_PROMPT, build_rationale_prompt
from rationale_query.store import (
    RationaleRecord,
    RationaleStore,
    question_hash,
    read_questions,
)
from tests.rationale_support import (
    QUESTIONS,
    FakeModel,
    FakeTokenizer,
    ScriptedGenerator,
    TempDirTestCase,
    default_config,
    np,
    ok,
    question_lines,
    run_cli,
)
from tests.spec_targets import SRC

try:
    from rationale_query.encoder import QueryEncoder
except ImportError:  # numpy is not installed
    QueryEncoder = None

needs_numpy = unittest.skipIf(QueryEncoder is None, "numpy is not installed")


class HelpTests(unittest.TestCase):
    def test_every_command_has_a_working_help_text(self):
        for command in ("show-prompt", "generate", "encode", "export-queries", "report"):
            with (
                self.subTest(command=command),
                contextlib.redirect_stdout(io.StringIO()) as shown,
                self.assertRaises(SystemExit) as caught,
            ):
                run_cli([command, "--help"])
            self.assertEqual(caught.exception.code, 0)
            self.assertIn("usage:", shown.getvalue())


class DurationTests(unittest.TestCase):
    def test_short_phrases(self):
        cases = {0: "0 s", 45: "45 s", 59.6: "1 min 00 s", 185: "3 min 05 s", 3599: "59 min 59 s", 3600: "1 h 00 min", 7620: "2 h 07 min", -5: "0 s"}
        for seconds, phrase in cases.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(duration(seconds), phrase)


class CliCase(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.questions = self.write("questions file.jsonl", question_lines())
        self.out = self.dir / "out folder"
        self.model = self.dir / "model.gguf"

    def generate(self, answers=None, *, extra=(), out=None, config=None, questions=None):
        generator = ScriptedGenerator(answers, config=config)
        built = []

        def factory(args, streams):
            built.append(args)
            return generator

        argv = [
            "generate",
            "--questions",
            str(questions or self.questions),
            "--out",
            str(out or self.out),
            "--model",
            str(self.model),
            *extra,
        ]
        status, text, err = run_cli(argv, generator_factory=factory)
        self.built, self.generator = built, generator
        return status, text, err

    def records(self, out=None):
        return RationaleStore(out or self.out).load()


class ShowPromptTests(CliCase):
    def test_one_question_on_the_command_line(self):
        status, text, err = run_cli(["show-prompt", "--question", "Which drug? A) x B) y"])
        self.assertEqual((status, err), (0, ""))
        prompt = RATIONALE_PROMPT + "Which drug? A) x B) y"
        self.assertIn("--- sent to the model as one user turn ---\n" + prompt + "\n", text)
        self.assertIn("--- the same turn in Llama-3 format ---\n" + llama3_prompt_text(prompt) + "\n", text)

    def test_a_question_of_a_file_by_id_and_the_first_by_default(self):
        status, text, _ = run_cli(["show-prompt", "--questions", str(self.questions), "--id", "q2"])
        self.assertEqual(status, 0)
        self.assertIn(QUESTIONS[1].text, text)
        self.assertNotIn(QUESTIONS[0].text, text)
        _, text, _ = run_cli(["show-prompt", "--questions", str(self.questions)])
        self.assertIn(QUESTIONS[0].text, text)

    def test_an_unknown_id_is_an_error_not_a_traceback(self):
        status, text, err = run_cli(["show-prompt", "--questions", str(self.questions), "--id", "nope"])
        self.assertEqual((status, text), (1, ""))
        self.assertIn("error: no question with id nope", err)

    def test_an_empty_question_is_an_error(self):
        status, _, err = run_cli(["show-prompt", "--question", "   "])
        self.assertEqual(status, 1)
        self.assertIn("error:", err)

    def test_exactly_one_source_is_needed(self):
        for argv in (["show-prompt"], ["show-prompt", "--question", "q", "--questions", "f"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                run_cli(argv)
            self.assertEqual(caught.exception.code, 2)


class GenerateTests(CliCase):
    def test_a_first_run_writes_every_rationale_and_says_so(self):
        status, text, _ = self.generate([ok("one"), ok("two"), ok("three")])
        self.assertEqual(status, 0)
        self.assertIn("3 questions: 0 already done, 3 written, 0 failed", text)
        self.assertEqual({r.id: r.rationale for r in self.records().values()}, {"q1": "one", "q2": "two", "q3": "three"})
        self.assertTrue((self.out / "config.json").is_file())
        self.assertEqual(self.generator.prompts[0], build_rationale_prompt(QUESTIONS[0].text))
        self.assertTrue(self.generator.closed)

    def test_progress_goes_to_the_error_stream_with_a_time_estimate(self):
        _, _, err = self.generate()
        lines = err.strip().split("\n")
        self.assertEqual(len(lines), 3)
        self.assertRegex(lines[0], r"^\[1/3\] q1 7 tokens, [0-9.e-]+ s; about .+ left$")
        self.assertTrue(lines[2].startswith("[3/3] q3"))
        self.assertTrue(lines[2].endswith("about 0 s left"))

    def test_the_time_left_is_the_average_so_far_times_the_questions_to_go(self):
        ticks = itertools.count(0, 100)  # the progress lines read the clock once each, 100 s apart
        with mock.patch("time.perf_counter", side_effect=lambda: float(next(ticks))):
            _, _, err = self.generate([ok("a"), ok("b"), ok("c")])
        left = [line.rsplit("about ", 1)[1] for line in err.strip().split("\n")]
        self.assertEqual(left, ["3 min 20 s left", "1 min 40 s left", "0 s left"])

    def test_quiet_switches_the_progress_off(self):
        _, _, err = self.generate(extra=["--quiet"])
        self.assertEqual(err, "")

    def test_a_second_run_has_nothing_to_do_and_does_not_load_the_model(self):
        self.generate()
        status, text, _ = self.generate()
        self.assertEqual(status, 0)
        self.assertIn("nothing to do: all 3 questions already have a rationale", text)
        self.assertEqual(self.built, [])

    def test_a_limit_runs_the_first_questions_and_a_later_run_continues(self):
        self.generate(extra=["--limit", "2"])
        self.assertEqual(set(self.records()), {"q1", "q2"})
        status, text, _ = self.generate()
        self.assertEqual(status, 0)
        self.assertIn("3 questions: 2 already done, 1 written, 0 failed", text)
        self.assertEqual(len(self.generator.prompts), 1)

    def test_a_limit_below_one_is_refused(self):
        status, _, err = self.generate(extra=["--limit", "0"])
        self.assertEqual(status, 1)
        self.assertIn("--limit must be at least 1", err)

    def test_a_failed_question_gives_status_2_and_names_the_failure(self):
        status, text, _ = self.generate([ok("one"), RuntimeError("boom"), ok("three")])
        self.assertEqual(status, 2)
        self.assertIn("1 failed", text)
        self.assertIn("failed q2: RuntimeError: boom", text)
        self.assertIn("run the same command again", text)
        status, text, _ = self.generate([ok("two, now")])
        self.assertEqual(status, 0)
        self.assertEqual(self.records()["q2"].rationale, "two, now")

    def test_five_failures_in_a_row_stop_the_run_with_status_2(self):
        many = self.write("many.jsonl", "".join(f'{{"id": "q{n}", "question": "Question {n}?"}}\n' for n in range(8)))
        status, _, err = self.generate([RuntimeError("model gone")], questions=many)
        self.assertEqual(status, 2)
        self.assertIn("stopped: 5 questions in a row failed", err)
        self.assertTrue(self.generator.closed)

    def test_an_interrupt_gives_status_130_and_the_finished_questions_are_kept(self):
        status, _, err = self.generate([ok("one"), KeyboardInterrupt()])
        self.assertEqual(status, 130)
        self.assertIn("run the same command again", err)
        self.assertEqual(set(self.records()), {"q1"})
        self.assertTrue(self.generator.closed)
        status, _, _ = self.generate()
        self.assertEqual((status, len(self.generator.prompts)), (0, 2))

    def test_a_changed_question_is_refused_before_the_model_is_loaded(self):
        self.generate()
        changed = self.write("changed.jsonl", question_lines([QUESTIONS[0], QUESTIONS[1]]) + '{"id": "q3", "question": "Something else?"}\n')
        status, text, err = self.generate(questions=changed)
        self.assertEqual((status, text), (1, ""))
        self.assertIn("q3 is in", err)
        self.assertEqual(self.built, [])

    def test_a_folder_made_with_other_settings_is_refused(self):
        self.generate(config=default_config(n_ctx=100), extra=["--limit", "1"])
        status, _, err = self.generate(config=default_config(n_ctx=200))
        self.assertEqual(status, 1)
        self.assertIn("other settings", err)
        self.assertEqual(self.generator.prompts, [])

    def test_a_missing_questions_file_is_an_error_not_a_traceback(self):
        status, _, err = self.generate(questions=self.dir / "absent.jsonl")
        self.assertEqual(status, 1)
        self.assertIn("error: cannot read", err)
        self.assertEqual(self.built, [])

    def test_an_output_path_that_is_a_file_is_an_error_not_a_traceback(self):
        blocker = self.write("blocker", "this is a file, not a folder")
        status, _, err = self.generate(out=blocker)
        self.assertEqual(status, 1)
        self.assertIn("error:", err)

    def test_a_missing_model_file_is_an_error_not_a_traceback(self):
        argv = ["generate", "--questions", str(self.questions), "--out", str(self.out), "--model", str(self.model)]
        status, text, err = run_cli(argv)
        self.assertEqual((status, text), (1, ""))
        self.assertIn("error: model file not found", err)

    def test_a_cut_off_line_of_an_interrupted_run_is_removed_with_a_note(self):
        self.generate(extra=["--limit", "2"])
        with open(self.out / "rationales.jsonl", "ab") as handle:
            handle.write(b'{"id":"q3","quest')
        status, text, _ = self.generate()
        self.assertEqual(status, 0)
        self.assertIn("note: the last line of the records file was cut short", text)

    def test_the_options_start_from_the_default_settings(self):
        self.generate()
        args, defaults = self.built[0], GenerationSettings()
        self.assertEqual(
            (args.chat_format, args.max_new_tokens, args.n_ctx, args.n_batch, args.n_ubatch, args.n_threads, args.n_gpu_layers, args.seed),
            (
                defaults.chat_format,
                defaults.max_new_tokens,
                defaults.n_ctx,
                defaults.n_batch,
                defaults.n_ubatch,
                defaults.n_threads,
                defaults.n_gpu_layers,
                defaults.seed,
            ),
        )
        self.assertFalse(args.allow_template_mismatch)
        self.assertFalse(args.verbose)

    def test_the_options_are_passed_on(self):
        self.generate(
            extra=["--max-new-tokens", "99", "--n-ctx", "1000", "--n-threads", "3", "--seed", "5", "--chat-format", "gguf", "--model-name", "label"]
        )
        args = self.built[0]
        self.assertEqual((args.max_new_tokens, args.n_ctx, args.n_threads, args.seed, args.chat_format, args.model_name), (99, 1000, 3, 5, "gguf", "label"))


@needs_numpy
class EncodeTests(CliCase):
    def encode(self, *, extra=(), encoder=None, out=None):
        made = encoder or QueryEncoder(FakeTokenizer(), FakeModel(), name="fake/encoder", revision="r1")
        argv = ["encode", "--rationales", str(out or self.out), "--questions", str(self.questions), *extra]
        return run_cli(argv, encoder_factory=lambda args: made)

    def test_the_default_encoder_says_what_is_missing_when_torch_cannot_be_imported(self):
        self.generate()
        argv = ["encode", "--rationales", str(self.out), "--questions", str(self.questions)]
        with mock.patch.dict(sys.modules, {"torch": None}):
            status, text, err = run_cli(argv)
        self.assertEqual((status, text), (1, ""))
        self.assertIn("encoding needs numpy, torch and transformers", err)
        self.assertFalse((self.out / "query_vectors.npy").exists())

    def test_vectors_and_their_description_are_written_next_to_the_rationales(self):
        self.generate([ok("a b"), ok("a b c d"), ok("b")])
        status, text, err = self.encode()
        self.assertEqual((status, err), (0, ""))
        vectors = np.load(self.out / "query_vectors.npy")
        self.assertEqual(vectors.shape, (3, 6))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_array_equal(vectors[:, 1], [4, 6, 3])
        meta = json.loads((self.out / "query_vectors.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["encoder"], meta["revision"], meta["shape"]), ("fake/encoder", "r1", [3, 6]))
        self.assertEqual(meta["ids"], ["q1", "q2", "q3"])
        self.assertEqual(meta["token_counts"], [4, 6, 3])
        self.assertEqual(meta["truncated"], [False, False, False])
        self.assertFalse(meta["normalised"])
        self.assertEqual(len(meta["rationales_sha256"]), 64)
        self.assertIn("3 query vectors of 6 numbers", text)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["config.json", "query_vectors.json", "query_vectors.npy", "rationales.jsonl"])

    def test_a_limit_encodes_only_the_first_questions(self):
        self.generate([ok("a b"), ok("a b c d"), ok("b")], extra=["--limit", "2"])
        status, text, _ = self.encode(extra=["--limit", "2"])
        self.assertEqual(status, 0)
        self.assertEqual(np.load(self.out / "query_vectors.npy").shape, (2, 6))
        meta = json.loads((self.out / "query_vectors.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["ids"], meta["shape"]), (["q1", "q2"], [2, 6]))
        self.assertIn("2 query vectors", text)
        status, _, err = self.encode(extra=["--overwrite"])  # without the limit the third question has no rationale
        self.assertEqual(status, 1)
        self.assertIn("q3: no record", err)

    def test_the_query_is_the_rationale_alone(self):
        self.generate([ok("  a b  \n"), ok("c"), ok("d")])
        encoded = []

        class Spy(QueryEncoder):
            def encode(self, texts):
                encoded.extend(texts)
                return super().encode(texts)

        self.encode(encoder=Spy(FakeTokenizer(), FakeModel()))
        self.assertEqual(encoded, ["a b", "c", "d"])

    def test_texts_longer_than_the_limit_are_counted_and_reported(self):
        self.generate([ok(" ".join(["w"] * 600)), ok("short"), ok("short")])
        status, text, _ = self.encode(encoder=QueryEncoder(FakeTokenizer(), FakeModel()))
        self.assertEqual(status, 0)
        meta = json.loads((self.out / "query_vectors.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["truncated"], [True, False, False])
        self.assertEqual(meta["token_counts"][0], 602)
        self.assertIn("1 longer than 512 tokens were cut by the encoder", text)

    def test_a_text_of_exactly_the_limit_is_not_reported_as_cut(self):
        self.generate([ok(" ".join(["w"] * 510)), ok(" ".join(["w"] * 511)), ok("short")])
        self.encode(encoder=QueryEncoder(FakeTokenizer(), FakeModel()))
        meta = json.loads((self.out / "query_vectors.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["token_counts"], [512, 513, 3])  # the two special tokens are counted
        self.assertEqual(meta["truncated"], [False, True, False])

    def test_existing_vectors_are_only_replaced_when_asked(self):
        self.generate()
        self.encode()
        before = (self.out / "query_vectors.npy").read_bytes()
        status, _, err = self.encode()
        self.assertEqual(status, 1)
        self.assertIn("already holds query vectors; add --overwrite", err)
        self.assertEqual((self.out / "query_vectors.npy").read_bytes(), before)
        status, _, _ = self.encode(extra=["--overwrite"])
        self.assertEqual(status, 0)

    def test_a_question_without_a_rationale_stops_everything_and_writes_nothing(self):
        self.generate([ok("a"), RuntimeError("boom"), ok("c")])
        status, _, err = self.encode()
        self.assertEqual(status, 1)
        self.assertIn("q2: RuntimeError: boom", err)
        self.assertFalse((self.out / "query_vectors.npy").exists())
        self.assertFalse((self.out / "query_vectors.json").exists())

    def test_a_vector_matrix_of_the_wrong_kind_is_refused(self):
        class Wrong(QueryEncoder):
            def encode(self, texts):
                return super().encode(texts).astype(np.float64)

        self.generate()
        status, _, err = self.encode(encoder=Wrong(FakeTokenizer(), FakeModel()))
        self.assertEqual(status, 1)
        self.assertIn("the encoder returned", err)
        self.assertFalse((self.out / "query_vectors.npy").exists())

    def test_the_options_reach_the_encoder_factory(self):
        self.generate()
        seen = []
        made = QueryEncoder(FakeTokenizer(), FakeModel())

        def factory(args):
            seen.append(args)
            return made

        argv = [
            "encode", "--rationales", str(self.out), "--questions", str(self.questions),
            "--device", "cpu", "--batch-size", "4", "--local-files-only", "--revision", "abc",
        ]  # fmt: skip
        self.assertEqual(run_cli(argv, encoder_factory=factory)[0], 0)
        args = seen[0]
        self.assertEqual((args.device, args.batch_size, args.local_files_only, args.revision, args.encoder), ("cpu", 4, True, "abc", "ncbi/MedCPT-Query-Encoder"))


class ExportTests(CliCase):
    def export(self, out=None):
        return run_cli(["export-queries", "--rationales", str(self.out), "--questions", str(self.questions), "--out", str(out or self.dir / "q.json")])

    def test_the_queries_are_a_json_list_in_the_order_of_the_questions(self):
        self.generate([ok("  first rationale\n"), ok("second"), ok("third")])
        status, text, _ = self.export()
        self.assertEqual(status, 0)
        self.assertIn("3 queries written", text)
        self.assertEqual(json.loads((self.dir / "q.json").read_text(encoding="utf-8")), ["first rationale", "second", "third"])

    def test_the_file_is_plain_ascii_and_reads_back_to_the_same_text(self):
        text = "β-amyloid ≥ 5\nsecond line \U0001F642"
        self.generate([ok(text), ok("b"), ok("c")])
        self.export()
        raw = (self.dir / "q.json").read_bytes()
        raw.decode("ascii")
        self.assertEqual(json.loads(raw)[0], text)

    def test_nothing_is_written_when_a_rationale_is_missing(self):
        self.generate([ok("a"), RuntimeError("boom"), ok("c")])
        status, _, err = self.export()
        self.assertEqual(status, 1)
        self.assertIn("q2", err)
        self.assertFalse((self.dir / "q.json").exists())

    def test_a_limit_cuts_the_list_to_the_first_questions(self):
        self.generate()
        status, text, _ = run_cli(
            ["export-queries", "--rationales", str(self.out), "--questions", str(self.questions), "--out", str(self.dir / "q.json"), "--limit", "2"]
        )
        self.assertEqual(status, 0)
        self.assertIn("2 queries written", text)
        self.assertEqual(len(json.loads((self.dir / "q.json").read_text(encoding="utf-8"))), 2)

    def test_the_output_folder_is_created(self):
        self.generate()
        self.assertEqual(self.export(self.dir / "new" / "deeper" / "q.json")[0], 0)
        self.assertTrue((self.dir / "new" / "deeper" / "q.json").is_file())


class ReportTests(CliCase):
    def test_one_folder(self):
        self.generate([ok("The answer is B."), ok("Option C."), ok("no letter", finish="length")])
        status, text, err = run_cli(["report", str(self.out)])
        self.assertEqual((status, err), (0, ""))
        lines = text.split("\n")
        self.assertIn("out folder", lines[0])
        rows = {line[:30].strip(): line[30:].split() for line in lines[1:11] if line.strip()}
        self.assertEqual(rows["records"], ["3"])
        self.assertEqual(rows["cut off at the limit"], ["1"])
        self.assertEqual(rows["final option readable"], ["2"])
        self.assertNotIn("same final option", text)

    def test_each_folder_says_how_its_rationales_were_made(self):
        config = default_config(model_file="writer.gguf", model_sha256="ab" * 32, chat_format="llama-3", max_new_tokens=7)
        self.generate(config=config)
        _, text, _ = run_cli(["report", str(self.out), "--json", str(self.dir / "numbers.json")])
        self.assertIn("out folder: writer.gguf (SHA-256 abababababab...), llama-3 format", text)
        self.assertIn("at most 7 new tokens", text)
        saved = json.loads((self.dir / "numbers.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["configs"]["out folder"]["model_sha256"], "ab" * 32)

    def test_a_folder_without_a_config_file_is_still_described(self):
        folder = self.dir / "bare"
        record = RationaleRecord(id="q1", question_sha256="h", status="ok", rationale="The answer is B.", finish_reason="stop")
        RationaleStore(folder).append(record)
        status, text, err = run_cli(["report", str(folder), "--json", str(self.dir / "numbers.json")])
        self.assertEqual((status, err), (0, ""))
        self.assertNotIn("SHA-256", text)
        self.assertEqual(json.loads((self.dir / "numbers.json").read_text(encoding="utf-8"))["configs"], {})

    def test_two_folders_are_compared_and_the_numbers_can_be_saved(self):
        other = self.dir / "second"
        self.generate([ok("The answer is B."), ok("The answer is C."), ok("The answer is D.")])
        self.generate([ok("The answer is B."), ok("The answer is A."), ok("no letter")], out=other)
        status, text, _ = run_cli(["report", str(self.out), str(other), "--json", str(self.dir / "numbers.json")])
        self.assertEqual(status, 0)
        self.assertIn("same final option, out folder vs second: 1 of 2", text)
        numbers = json.loads((self.dir / "numbers.json").read_text(encoding="utf-8"))
        self.assertEqual(set(numbers), {"summaries", "agreement", "configs"})
        self.assertEqual(numbers["agreement"]["out folder vs second"], {"compared": 2, "same_option": 1, "share": 0.5})
        self.assertEqual(numbers["summaries"]["second"]["final_option_found"], 2)

    def test_folders_with_the_same_name_get_different_labels(self):
        first, second = self.dir / "a" / "run", self.dir / "b" / "run"
        self.generate(out=first)
        self.generate(out=second)
        _, text, _ = run_cli(["report", str(first), str(second)])
        self.assertIn("run'", text.split("\n")[0])
        self.assertIn("same final option, run vs run':", text)

    def test_a_folder_without_records_is_an_error(self):
        (self.dir / "empty").mkdir()
        status, _, err = run_cli(["report", str(self.dir / "empty")])
        self.assertEqual(status, 1)
        self.assertIn("no rationales.jsonl", err)

    @needs_numpy
    def test_the_token_counts_of_the_vectors_feed_the_truncation_row(self):
        self.generate([ok(" ".join(["w"] * 600)), ok("short"), ok("short")])
        run_cli(["encode", "--rationales", str(self.out), "--questions", str(self.questions)], encoder_factory=lambda args: QueryEncoder(FakeTokenizer(), FakeModel()))
        _, text, _ = run_cli(["report", str(self.out)])
        self.assertTrue(text.split("\n")[9].rstrip().endswith("1"))
        self.assertIn("longer than the encoder limit", text.split("\n")[9])


class ModuleEntryPointTests(TempDirTestCase):
    def run_module(self, *args, **env):
        environment = {**os.environ, "PYTHONPATH": str(SRC), **env}
        return subprocess.run(
            [sys.executable, "-m", "rationale_query", *args], capture_output=True, cwd=self.dir, env=environment, check=False
        )

    def test_the_module_runs_from_any_folder(self):
        done = self.run_module("show-prompt", "--question", "Which drug? A) x B) y")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(done.stdout.decode("utf-8").startswith("--- sent to the model as one user turn ---\n"))

    def test_a_console_that_cannot_show_a_character_does_not_make_the_command_fail(self):
        done = self.run_module("show-prompt", "--question", "β-amyloid ≥ 5? A) x B) y", PYTHONIOENCODING="ascii")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("-amyloid", done.stdout.decode("ascii"))

    def test_a_missing_command_shows_the_usage(self):
        done = self.run_module()
        self.assertEqual(done.returncode, 2)
        self.assertIn(b"usage:", done.stderr)

    def test_the_question_file_of_the_command_line_may_sit_in_a_folder_with_spaces(self):
        path = self.write("a folder/questions file.jsonl", question_lines())
        done = self.run_module("show-prompt", "--questions", str(path), "--id", "q3")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(QUESTIONS[2].text, done.stdout.decode("utf-8"))


class SharedFixtureTests(TempDirTestCase):
    def test_the_hash_of_a_question_is_that_of_its_trimmed_text(self):
        path = self.write("q.jsonl", question_lines())
        self.assertEqual([question_hash(q.text) for q in read_questions(path)], [question_hash(q.text) for q in QUESTIONS])
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", question_hash("x")))


if __name__ == "__main__":
    unittest.main()
