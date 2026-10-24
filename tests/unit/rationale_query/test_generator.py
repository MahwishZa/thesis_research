"""rationale_query.generator: the paper's text alone, greedy decoding, no length limit of our own, and the refusals.

A scripted stand-in takes the place of llama.cpp, so that the test can see exactly what the model is given.
"""

import hashlib
import unittest

from rationale_query import generator
from rationale_query.generator import ContextOverflowError, GenerationError, GenerationSettings, LlamaGenerator, ModelMismatchError
from rationale_query.prompt import build_rationale_prompt
from tests.support import TempDirTestCase
from tests.unit.rationale_query.examples import FIGURE_4_QUESTION

BOS = 1


class FakeContext:
    def __init__(self, log):
        self.log = log

    def kv_cache_clear(self):
        self.log.append("kv_cache_clear")


class FakeLlama:
    """Tokenizes one token per word (start token first) and answers with a scripted text."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.log = []
        self.calls = []
        self.tokenized = []
        self.reply = {"choices": [{"text": "step by step. Therefore, the answer is (C) Intubation.", "finish_reason": "stop"}], "usage": {"completion_tokens": 9}}
        self.extra_bos = False
        self._ctx = FakeContext(self.log)

    def tokenize(self, text, add_bos=True, special=False):
        self.tokenized.append((text, add_bos, special))
        words = text.decode("utf-8").split()
        tokens = [BOS] if add_bos else []
        tokens += [100 + i for i, _ in enumerate(words)]
        if self.extra_bos:
            tokens.append(BOS)
        return tokens

    def token_bos(self):
        return BOS

    def reset(self):
        self.log.append("reset")

    def create_completion(self, **kwargs):
        self.log.append("create_completion")
        self.calls.append(kwargs)
        return self.reply


class GeneratorTestCase(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.model = self.dir / "meerkat.gguf"
        self.model.write_bytes(b"not a real model")
        self.made = []

    def factory(self, **kwargs):
        llama = FakeLlama(**kwargs)
        self.made.append(llama)
        return llama

    def make(self, **kwargs):
        settings = kwargs.pop("settings", None)
        return LlamaGenerator(self.model, settings, llama_factory=self.factory, **kwargs)


class FormatTests(unittest.TestCase):
    def test_by_default_the_model_reads_the_papers_text_alone(self):
        self.assertEqual(GenerationSettings().chat_format, "none")
        self.assertEqual(generator.render_model_input("PROMPT"), "PROMPT")
        self.assertEqual(generator.render_model_input("PROMPT", "none"), "PROMPT")

    def test_meerkats_own_template_is_a_named_option_for_a_labelled_check(self):
        self.assertEqual(generator.render_model_input("PROMPT", "meerkat"), "USER: PROMPT ASSISTANT: ")

    def test_a_blank_prompt_or_an_unknown_format_is_refused(self):
        for bad in ("", "  ", None, 3):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                generator.render_model_input(bad)
        with self.assertRaises(ValueError):
            generator.render_model_input("PROMPT", "vicuna")
        with self.assertRaises(ValueError):
            GenerationSettings(chat_format="vicuna")


class GenerateTests(GeneratorTestCase):
    def test_the_model_reads_the_papers_prompt_and_nothing_else(self):
        gen = self.make()
        prompt = build_rationale_prompt(FIGURE_4_QUESTION)
        gen.generate(prompt)
        text, add_bos, special = self.made[0].tokenized[-1]
        self.assertEqual(text.decode("utf-8"), prompt)
        self.assertTrue(add_bos)
        self.assertFalse(special, "text inside a question must never be read as a special token")

    def test_meerkats_template_is_used_only_when_asked_for(self):
        gen = self.make(settings=GenerationSettings(chat_format="meerkat"))
        gen.generate("a b c")
        self.assertEqual(self.made[0].tokenized[-1][0].decode("utf-8"), "USER: a b c ASSISTANT: ")
        self.assertEqual(gen.config["chat_format"], "meerkat")

    def test_the_prompt_is_given_as_a_token_list_with_the_start_token_once(self):
        gen = self.make()
        gen.generate("a b c")
        tokens = self.made[0].calls[0]["prompt"]
        self.assertIsInstance(tokens, list)
        self.assertEqual(tokens[0], BOS)
        self.assertNotIn(BOS, tokens[1:])

    def test_a_second_start_token_is_an_error(self):
        gen = self.make()
        self.made[0].extra_bos = True
        with self.assertRaises(GenerationError):
            gen.generate("a b c")

    def test_decoding_is_greedy_with_every_value_written_out(self):
        gen = self.make()
        gen.generate("a b c")
        call = self.made[0].calls[0]
        for name, value in generator.SAMPLING.items():
            self.assertEqual(call[name], value, name)
        self.assertEqual(call["temperature"], 0.0)
        self.assertEqual(call["top_k"], 1)
        self.assertIsNone(call["stop"])

    def test_there_is_no_length_limit_except_the_window(self):
        gen = self.make(settings=GenerationSettings(n_ctx=300))
        result = gen.generate("a b c d e")  # start token and five words
        call = self.made[0].calls[0]
        self.assertEqual(result.prompt_tokens, 6)
        self.assertEqual(call["max_tokens"], 300 - 6)

    def test_the_memory_of_the_previous_question_is_emptied_first(self):
        gen = self.make()
        gen.generate("a b c")
        gen.generate("d e f")
        log = self.made[0].log
        self.assertEqual(log, ["reset", "kv_cache_clear", "create_completion"] * 2)

    def test_the_result_carries_the_text_and_counts(self):
        gen = self.make()
        result = gen.generate("a b c")
        self.assertEqual(result.text, "step by step. Therefore, the answer is (C) Intubation.")
        self.assertEqual((result.finish_reason, result.prompt_tokens, result.completion_tokens), ("stop", 4, 9))  # start token, a b c

    def test_a_full_window_is_reported_as_length_and_the_text_is_kept(self):
        gen = self.make()
        self.made[0].reply = {"choices": [{"text": "cut off", "finish_reason": "length"}], "usage": {"completion_tokens": 40}}
        result = gen.generate("a b c")
        self.assertEqual((result.text, result.finish_reason), ("cut off", "length"))

    def test_an_unexpected_reply_is_an_error(self):
        gen = self.make()
        for reply in (
            {"choices": [{"text": "x", "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 1}},
            {"choices": [{"text": None, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}},
        ):
            self.made[0].reply = reply
            with self.subTest(reply=reply), self.assertRaises(GenerationError):
                gen.generate("a b c")

    def test_a_prompt_that_does_not_fit_is_refused_and_never_shortened(self):
        gen = self.make(settings=GenerationSettings(n_ctx=5))
        with self.assertRaises(ContextOverflowError):
            gen.generate("a b c d e")
        self.assertEqual(self.made[0].calls, [])


class ModelFileTests(GeneratorTestCase):
    def test_the_settings_reach_llama_cpp(self):
        self.make(settings=GenerationSettings(n_ctx=1024, n_threads=3, n_gpu_layers=5, seed=7))
        kwargs = self.made[0].kwargs
        self.assertEqual((kwargs["n_ctx"], kwargs["n_threads"], kwargs["n_gpu_layers"], kwargs["seed"]), (1024, 3, 5, 7))
        self.assertFalse(kwargs["logits_all"])

    def test_the_default_window_is_the_length_meerkat_was_trained_with(self):
        self.assertEqual(GenerationSettings().n_ctx, 2048)

    def test_the_configuration_records_the_file_by_its_hash(self):
        gen = self.make()
        self.assertEqual(gen.config["model_sha256"], hashlib.sha256(b"not a real model").hexdigest())
        self.assertEqual(gen.config["chat_format"], "none")
        self.assertEqual(gen.config["sampling"], generator.SAMPLING)
        self.assertEqual(gen.config["n_ctx"], 2048)

    def test_a_file_with_another_hash_than_the_pinned_one_is_refused(self):
        with self.assertRaises(ModelMismatchError):
            self.make(expected_sha256="0" * 64)
        self.make(expected_sha256=hashlib.sha256(b"not a real model").hexdigest().upper())

    def test_a_missing_file_is_refused(self):
        with self.assertRaises(GenerationError):
            LlamaGenerator(self.dir / "absent.gguf", llama_factory=self.factory)

    def test_bad_settings_are_refused(self):
        for bad in (0, -1, 1.5, True, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                GenerationSettings(n_ctx=bad)


if __name__ == "__main__":
    unittest.main()
