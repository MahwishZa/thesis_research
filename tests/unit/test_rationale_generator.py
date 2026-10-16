"""Tests of src/rationale_query/generator.py with a stand-in for llama.cpp. No model file is needed."""

import hashlib
import importlib.metadata
import sys
import unittest
from unittest import mock

from rationale_query.generator import (
    ASSISTANT_HEAD,
    BOS,
    EOT,
    RESULT_RELEVANT,
    SAMPLING,
    TEMPLATE_MARGIN_TOKENS,
    USER_HEAD,
    ContextOverflowError,
    GenerationError,
    GenerationSettings,
    LlamaCppGenerator,
    ModelMismatchError,
    TemplateMismatchError,
    check_llama3_vocabulary,
    file_sha256,
    llama3_prompt_text,
    llama3_prompt_tokens,
    text_sha256,
)
from rationale_query.prompt import RATIONALE_PROMPT, build_rationale_prompt
from tests.rationale_support import (
    LLAMA3_JINJA,
    FakeLlama,
    TempDirTestCase,
    llama_factory,
    official_renderer,
)

MESSAGE = build_rationale_prompt("Which drug is a cholinesterase inhibitor? A) Donepezil B) Aspirin C) Insulin D) Warfarin")


def ids(text):
    return list(text.encode("utf-8"))


class SettingsTests(unittest.TestCase):
    def test_the_defaults_are_the_documented_ones(self):
        # A changed default changes the rationales without any visible sign, so each value is written down here.
        s = GenerationSettings()
        self.assertEqual(
            (s.chat_format, s.max_new_tokens, s.n_ctx, s.n_batch, s.n_ubatch, s.n_threads, s.n_gpu_layers, s.seed),
            ("llama-3", 512, 8192, 512, 512, None, 0, 42),
        )
        self.assertFalse(s.allow_template_mismatch)

    def test_bad_values_are_refused(self):
        bad = [
            {"chat_format": "chatml"},
            {"max_new_tokens": 0},
            {"n_ctx": 0},
            {"n_batch": 0},
            {"n_ubatch": -1},
            {"n_ctx": 512, "max_new_tokens": 512},
            {"n_threads": 0},
            {"n_gpu_layers": -2},
            {"seed": -1},
        ]
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                GenerationSettings(**changes)

    def test_the_edge_values_are_accepted(self):
        GenerationSettings(n_ctx=513, max_new_tokens=512, n_threads=1, n_gpu_layers=-1, seed=0)

    def test_decoding_is_greedy_and_every_sampling_value_is_neutral(self):
        self.assertEqual(SAMPLING["temperature"], 0.0)
        self.assertEqual(SAMPLING["top_k"], 1)
        self.assertEqual((SAMPLING["top_p"], SAMPLING["min_p"], SAMPLING["typical_p"]), (1.0, 0.0, 1.0))
        self.assertEqual((SAMPLING["repeat_penalty"], SAMPLING["frequency_penalty"], SAMPLING["presence_penalty"]), (1.0, 0.0, 0.0))

    def test_every_sampling_value_is_bound_to_the_output_folder(self):
        self.assertTrue(set(SAMPLING) <= set(RESULT_RELEVANT))

    def test_the_fields_bound_to_the_output_folder_are_exactly_those_that_can_change_a_rationale(self):
        # Written out in full: a field dropped from the list would otherwise drop out of every check that reads the list.
        self.assertEqual(
            set(RESULT_RELEVANT),
            {
                "model_sha256", "chat_format", "chat_template", "prompt_sha256", "max_new_tokens", "n_ctx",
                "n_batch", "n_ubatch", "n_gpu_layers", "llama_cpp_python", *SAMPLING,
            },
        )  # fmt: skip


class FileHashTests(TempDirTestCase):
    def test_the_hash_is_the_sha256_of_the_bytes(self):
        path = self.write("data.bin", "some model bytes")
        self.assertEqual(file_sha256(path), hashlib.sha256(b"some model bytes").hexdigest())

    def test_a_small_chunk_size_gives_the_same_hash(self):
        path = self.write("data.bin", "x" * 1000)
        self.assertEqual(file_sha256(path, chunk_size=7), file_sha256(path))

    def test_progress_is_reported_at_every_step_of_the_given_size(self):
        path = self.write("data.bin", "x" * 10)
        seen = []
        file_sha256(path, chunk_size=2, report_every=4, progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(4, 10), (8, 10)])

    def test_text_hash_is_utf8_sha256(self):
        self.assertEqual(text_sha256("β"), hashlib.sha256("β".encode()).hexdigest())


class Llama3FormatTests(unittest.TestCase):
    def test_the_text_is_one_user_turn_in_the_official_layout(self):
        self.assertEqual(
            llama3_prompt_text("Hello"),
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    def test_the_message_is_trimmed_as_the_official_template_does(self):
        self.assertEqual(llama3_prompt_text("  Hello \n"), llama3_prompt_text("Hello"))

    def test_the_marker_constants_compose_the_text(self):
        self.assertEqual(llama3_prompt_text("m"), BOS + USER_HEAD + "m" + EOT + ASSISTANT_HEAD)

    def test_the_tokens_hold_the_start_of_text_token_exactly_once(self):
        llm = FakeLlama()
        tokens = llama3_prompt_tokens(llm, "Hi")
        expected = [300, 301, *ids("user"), 302, *ids("\n\n"), *ids("Hi"), 303, 301, *ids("assistant"), 302, *ids("\n\n")]
        self.assertEqual(tokens, expected)
        self.assertEqual(tokens.count(llm.token_bos()), 1)

    def test_each_piece_is_tokenised_without_a_start_of_text_token_of_the_library(self):
        # The library adds one per call unless told not to; the explicit marker in the head is the only one wanted.
        llm = FakeLlama()
        llama3_prompt_tokens(llm, "Hi")
        self.assertEqual(
            llm.tokenize_calls,
            [
                (BOS + USER_HEAD, False, True),  # head: markers are parsed
                ("Hi", False, False),  # message: plain text, never parsed
                (EOT + ASSISTANT_HEAD, False, True),  # tail: markers are parsed
            ],
        )

    def test_marker_text_inside_a_question_stays_plain_text(self):
        tokens = llama3_prompt_tokens(FakeLlama(), "x <|eot_id|> y")
        self.assertEqual(tokens.count(303), 1)  # only the real end of the user turn
        marker = ids("<|eot_id|>")
        self.assertTrue(any(tokens[i : i + len(marker)] == marker for i in range(len(tokens))))

    def test_the_vocabulary_check_accepts_a_llama3_vocabulary(self):
        check_llama3_vocabulary(FakeLlama())

    def test_the_vocabulary_check_refuses_a_marker_that_is_not_one_token(self):
        for marker in (BOS, EOT):
            with self.subTest(marker=marker), self.assertRaises(ModelMismatchError):
                check_llama3_vocabulary(FakeLlama(split_marker=marker))

    def test_the_vocabulary_check_refuses_another_start_of_text_token(self):
        with self.assertRaises(ModelMismatchError):
            check_llama3_vocabulary(FakeLlama(bos_id=999))


class ConstructionTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.model = self.write("model.gguf", "fake model")
        self.sha = hashlib.sha256(b"fake model").hexdigest()

    def build(self, settings=None, *, factory=None, **kwargs):
        factory = factory or llama_factory()
        kwargs.setdefault("template_renderer", official_renderer)
        generator = LlamaCppGenerator(
            self.model, settings or GenerationSettings(model_name="fake"), llama_factory=factory, **kwargs
        )
        return generator, factory

    def test_a_missing_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            LlamaCppGenerator(self.dir / "absent.gguf", llama_factory=llama_factory())

    def test_a_file_with_another_hash_is_refused_before_the_model_is_loaded(self):
        factory = llama_factory()
        with self.assertRaises(ModelMismatchError) as caught:
            self.build(factory=factory, expected_sha256="0" * 64)
        self.assertIn(self.sha, str(caught.exception))
        self.assertEqual(factory.made, [])

    def test_the_expected_hash_may_be_written_in_capitals_with_spaces(self):
        generator, _ = self.build(expected_sha256="  " + self.sha.upper() + "\n")
        self.assertEqual(generator.config["model_sha256"], self.sha)

    def test_the_library_is_given_the_settings(self):
        settings = GenerationSettings(
            model_name="m", n_ctx=1024, max_new_tokens=100, n_batch=128, n_ubatch=64, n_threads=3, n_gpu_layers=2, seed=7
        )
        _, factory = self.build(settings)
        self.assertEqual(
            factory.made[0].options,
            {
                "model_path": str(self.model),
                "n_ctx": 1024,
                "n_batch": 128,
                "n_ubatch": 64,
                "n_threads": 3,
                "n_gpu_layers": 2,
                "seed": 7,
                "verbose": False,
            },
        )

    def test_the_library_stays_quiet_unless_it_is_asked_to_talk(self):
        _, factory = self.build()
        self.assertIs(factory.made[0].options["verbose"], False)
        _, factory = self.build(verbose=True)
        self.assertIs(factory.made[0].options["verbose"], True)

    def test_the_config_records_everything_that_is_bound_to_the_folder(self):
        generator, _ = self.build(library_version="9.9.9")
        config = generator.config
        self.assertTrue(set(RESULT_RELEVANT) <= set(config), set(RESULT_RELEVANT) - set(config))
        self.assertEqual(config["model_sha256"], self.sha)
        self.assertEqual(config["model_file"], "model.gguf")
        self.assertEqual(config["prompt_sha256"], hashlib.sha256(RATIONALE_PROMPT.encode("utf-8")).hexdigest())
        self.assertEqual(config["llama_cpp_python"], "9.9.9")
        self.assertEqual(config["decoding"], "greedy")
        self.assertIsNone(config["system_prompt"])
        self.assertTrue(config["fresh_cache_per_question"])
        for name, value in SAMPLING.items():
            self.assertEqual(config[name], value, name)

    def test_the_config_holds_what_the_library_really_uses(self):
        # The library lowers the batch sizes to the context size; the record must say what was used.
        settings = GenerationSettings(model_name="m", n_ctx=300, max_new_tokens=100, n_batch=512, n_ubatch=512, n_threads=3)
        generator, _ = self.build(settings)
        self.assertEqual((generator.config["n_batch"], generator.config["n_ubatch"]), (300, 300))
        self.assertEqual(generator.config["n_threads"], 3)
        self.assertEqual(generator.config["n_threads_batch"], 8)
        self.assertEqual((generator.config["flash_attn_type"], generator.config["type_k"], generator.config["type_v"]), (0, 1, 1))

    def test_the_label_of_the_model_is_not_bound_to_the_folder(self):
        self.assertNotIn("model_name", RESULT_RELEVANT)
        generator, _ = self.build(GenerationSettings(model_name="my label"))
        self.assertEqual(generator.config["model_name"], "my label")

    def test_the_thread_count_and_the_seed_are_recorded_but_not_bound_to_the_folder(self):
        # Greedy decoding ignores the seed and llama.cpp gives the same numbers for any number of threads, so a
        # run that is continued with another value must not be refused.
        for name in ("n_threads", "n_threads_batch", "seed"):
            self.assertNotIn(name, RESULT_RELEVANT)

    def test_the_llama3_format_does_not_ask_the_library_for_a_chat_format(self):
        _, factory = self.build()
        self.assertNotIn("chat_format", factory.made[0].options)
        self.assertEqual(_.config["chat_template"], "llama-3-instruct: one user turn, no system prompt")

    def test_a_vocabulary_that_is_not_llama3_is_refused(self):
        with self.assertRaises(ModelMismatchError):
            self.build(factory=llama_factory(split_marker=EOT))

    def test_the_files_own_template_is_compared_with_the_llama3_format(self):
        generator, _ = self.build(template_renderer=official_renderer)
        self.assertEqual(generator.config["template_check"], "matches the file's own chat template")

    def test_a_template_that_renders_something_else_is_refused(self):
        with self.assertRaises(TemplateMismatchError):
            self.build(template_renderer=lambda message: "something else")

    def test_a_template_mismatch_can_be_allowed_and_is_then_recorded(self):
        settings = GenerationSettings(model_name="m", allow_template_mismatch=True)
        generator, _ = self.build(settings, template_renderer=lambda message: "something else")
        self.assertEqual(generator.config["template_check"], "MISMATCH (allowed by setting)")

    def test_a_check_that_cannot_run_is_recorded_not_fatal(self):
        def broken(message):
            raise RuntimeError("renderer broke")

        generator, _ = self.build(template_renderer=broken)
        self.assertEqual(generator.config["template_check"], "not checkable: RuntimeError: renderer broke")

    def test_a_file_without_a_template_is_recorded_as_such(self):
        generator, _ = self.build(template_renderer=None)
        self.assertEqual(generator.config["template_check"], "no chat template in the file")

    def test_the_gguf_format_needs_the_template_of_the_file(self):
        settings = GenerationSettings(model_name="m", chat_format="gguf")
        with self.assertRaises(GenerationError) as caught:
            self.build(settings)
        self.assertIn("no chat template", str(caught.exception))

    def test_the_gguf_format_asks_the_library_for_the_template_of_the_file(self):
        settings = GenerationSettings(model_name="m", chat_format="gguf")
        generator, factory = self.build(settings, factory=llama_factory(template=LLAMA3_JINJA))
        self.assertEqual(factory.made[0].options["chat_format"], "chat_template.default")
        self.assertEqual(generator.config["chat_template"], "gguf-template-sha256:" + hashlib.sha256(LLAMA3_JINJA.encode()).hexdigest())

    def test_another_template_gives_another_template_id(self):
        settings = GenerationSettings(model_name="m", chat_format="gguf")
        first, _ = self.build(settings, factory=llama_factory(template="{{ a }}"))
        second, _ = self.build(settings, factory=llama_factory(template="{{ b }}"))
        self.assertNotEqual(first.config["chat_template"], second.config["chat_template"])


class GenerateTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.model = self.write("model.gguf", "fake model")

    def make(self, outputs=None, settings=None, **fake):
        factory = llama_factory(outputs=outputs, **fake)
        generator = LlamaCppGenerator(
            self.model, settings or GenerationSettings(model_name="fake"), llama_factory=factory, template_renderer=official_renderer
        )
        return generator, factory.made[0]

    def test_the_completion_gets_the_token_list_and_the_greedy_settings(self):
        generator, llm = self.make()
        generator.generate(MESSAGE)
        call = llm.completion_calls[0]
        self.assertEqual(call["prompt"], llama3_prompt_tokens(llm, MESSAGE))
        self.assertTrue(all(isinstance(token, int) for token in call["prompt"]))
        self.assertEqual((call["max_tokens"], call["seed"]), (512, 42))
        for name, value in SAMPLING.items():
            self.assertEqual(call[name], value, name)
        self.assertEqual(llm.chat_calls, [])

    def test_the_settings_reach_the_completion(self):
        generator, llm = self.make(settings=GenerationSettings(model_name="m", max_new_tokens=33, seed=5))
        generator.generate(MESSAGE)
        self.assertEqual((llm.completion_calls[0]["max_tokens"], llm.completion_calls[0]["seed"]), (33, 5))

    def test_the_cache_is_reset_before_every_completion(self):
        generator, llm = self.make()
        generator.generate(MESSAGE)
        generator.generate(MESSAGE)
        completions = [i for i, event in enumerate(llm.events) if event == "create_completion"]
        resets = [i for i, event in enumerate(llm.events) if event == "reset"]
        self.assertEqual(len(completions), 2)
        self.assertEqual(len(resets), 2)
        self.assertLess(resets[0], completions[0])
        self.assertLess(completions[0], resets[1])
        self.assertLess(resets[1], completions[1])

    def test_the_result_carries_text_reason_and_counts(self):
        generator, llm = self.make(outputs=[(" The answer is B.", "stop")])
        result = generator.generate(MESSAGE)
        self.assertEqual(result.text, " The answer is B.")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.prompt_tokens, len(llama3_prompt_tokens(llm, MESSAGE)))
        self.assertEqual(result.completion_tokens, 4)

    def test_a_cut_off_rationale_says_so(self):
        generator, _ = self.make(outputs=[("The patient", "length")])
        self.assertEqual(generator.generate(MESSAGE).finish_reason, "length")

    def test_an_unknown_finish_reason_is_an_error(self):
        generator, _ = self.make(outputs=[("text", "tool_calls")])
        with self.assertRaises(GenerationError):
            generator.generate(MESSAGE)

    def test_no_text_gives_an_empty_string(self):
        generator, _ = self.make(outputs=[("", "stop")])
        self.assertEqual(generator.generate(MESSAGE).text, "")

    def test_a_prompt_that_does_not_fit_is_refused_before_the_model_runs(self):
        settings = GenerationSettings(model_name="m", n_ctx=300, max_new_tokens=200)
        generator, llm = self.make(settings=settings)
        with self.assertRaises(ContextOverflowError):
            generator.generate(MESSAGE)
        self.assertEqual(llm.completion_calls, [])

    def test_a_prompt_that_just_fits_is_accepted_and_one_token_less_room_is_refused(self):
        length = len(llama3_prompt_tokens(FakeLlama(), MESSAGE))
        fits = GenerationSettings(model_name="m", n_ctx=length + 100, max_new_tokens=100)
        tight = GenerationSettings(model_name="m", n_ctx=length + 99, max_new_tokens=100)
        generator, _ = self.make(settings=fits)
        self.assertEqual(generator.generate(MESSAGE).finish_reason, "stop")
        generator, _ = self.make(settings=tight)
        with self.assertRaises(ContextOverflowError):
            generator.generate(MESSAGE)

    def test_a_library_that_adds_a_start_of_text_token_anyway_is_caught_when_the_model_is_loaded(self):
        with self.assertRaises(ModelMismatchError):
            self.make(add_bos_when_told_not_to=True)

    def test_a_prompt_with_two_start_of_text_tokens_is_refused_before_the_model_runs(self):
        generator, llm = self.make()
        for tokens in ([300, 65, 300], [65, 66]):
            with (
                self.subTest(tokens=tokens),
                mock.patch("rationale_query.generator.llama3_prompt_tokens", return_value=tokens),
                self.assertRaises(GenerationError),
            ):
                generator.generate(MESSAGE)
        self.assertEqual(llm.completion_calls, [])

    def test_close_closes_the_model(self):
        generator, llm = self.make()
        generator.close()
        self.assertTrue(llm.closed)


class GgufFormatTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.model = self.write("model.gguf", "fake model")

    def make(self, outputs=None, **settings):
        factory = llama_factory(outputs=outputs, template=LLAMA3_JINJA)
        generator = LlamaCppGenerator(
            self.model, GenerationSettings(model_name="m", chat_format="gguf", **settings), llama_factory=factory
        )
        return generator, factory.made[0]

    def test_the_message_goes_to_the_chat_completion_as_one_user_turn(self):
        generator, llm = self.make(outputs=[("The answer is B.", "stop")])
        result = generator.generate("  " + MESSAGE + "\n")
        call = llm.chat_calls[0]
        self.assertEqual(call["messages"], [{"role": "user", "content": MESSAGE}])
        self.assertEqual((call["max_tokens"], call["seed"]), (512, 42))
        for name, value in SAMPLING.items():
            self.assertEqual(call[name], value, name)
        self.assertEqual(llm.completion_calls, [])
        self.assertEqual((result.text, result.finish_reason), ("The answer is B.", "stop"))
        self.assertEqual(result.prompt_tokens, 50)

    def test_the_cache_is_reset_before_the_chat_completion(self):
        generator, llm = self.make()
        generator.generate(MESSAGE)
        self.assertLess(llm.events.index("reset"), llm.events.index("create_chat_completion"))

    def test_a_prompt_that_cannot_fit_is_refused_before_the_model_runs(self):
        generator, llm = self.make(n_ctx=200, max_new_tokens=150)
        with self.assertRaises(ContextOverflowError):
            generator.generate(MESSAGE)
        self.assertEqual(llm.chat_calls, [])

    def test_a_context_that_the_library_would_shorten_silently_is_an_error(self):
        # The estimate (message plus the margin) just fits, but the library reports one token more: it must have
        # lowered max_tokens to make the prompt fit, which is not the setting that was asked for.
        body = len(FakeLlama().tokenize(MESSAGE.strip().encode("utf-8"), add_bos=False))
        factory = llama_factory(template=LLAMA3_JINJA, prompt_tokens_in_usage=body + 64 + 1)
        settings = GenerationSettings(model_name="m", chat_format="gguf", n_ctx=body + 64 + 100, max_new_tokens=100)
        generator = LlamaCppGenerator(self.model, settings, llama_factory=factory)
        with self.assertRaises(ContextOverflowError):
            generator.generate(MESSAGE)

    def test_an_unknown_finish_reason_is_an_error(self):
        generator, _ = self.make(outputs=[("text", "weird")])
        with self.assertRaises(GenerationError):
            generator.generate(MESSAGE)

    def test_a_reply_without_content_gives_an_empty_string(self):
        generator, _ = self.make(outputs=[(None, "stop")])
        self.assertEqual(generator.generate(MESSAGE).text, "")

    def test_the_room_kept_for_the_template_is_64_tokens(self):
        # The estimate is the tokens of the message plus a fixed margin; the margin is pinned here as a number.
        body = len(FakeLlama().tokenize(MESSAGE.strip().encode("utf-8"), add_bos=False))
        self.assertEqual(TEMPLATE_MARGIN_TOKENS, 64)
        generator, llm = self.make(n_ctx=body + 64 + 100, max_new_tokens=100)
        generator.generate(MESSAGE)
        self.assertEqual(len(llm.chat_calls), 1)
        generator, llm = self.make(n_ctx=body + 64 + 99, max_new_tokens=100)
        with self.assertRaises(ContextOverflowError):
            generator.generate(MESSAGE)
        self.assertEqual(llm.chat_calls, [])


class LibraryTests(TempDirTestCase):
    def test_a_missing_llama_cpp_python_is_reported_as_such(self):
        model = self.write("model.gguf", "x")
        with mock.patch.dict(sys.modules, {"llama_cpp": None}), self.assertRaises(GenerationError) as caught:
            LlamaCppGenerator(model, GenerationSettings(model_name="m"))
        self.assertIn("llama-cpp-python is not installed", str(caught.exception))

    def test_the_library_version_is_recorded_and_unknown_when_it_cannot_be_found(self):
        model = self.write("model.gguf", "x")

        def build():
            return LlamaCppGenerator(
                model, GenerationSettings(model_name="m"), llama_factory=llama_factory(), template_renderer=official_renderer
            )

        with mock.patch("importlib.metadata.version", return_value="1.2.3"):
            self.assertEqual(build().config["llama_cpp_python"], "1.2.3")
        with mock.patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError("llama_cpp_python")):
            self.assertEqual(build().config["llama_cpp_python"], "unknown")


if __name__ == "__main__":
    unittest.main()
