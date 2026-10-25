"""rationale_query.generator: the paper's text alone, greedy BF16 decoding, no length limit of our own, and the refusals.

Stand-ins take the place of the model and its tokenizer, so that the test can see exactly what the model is given.
"""

import hashlib
import unittest

from rationale_query import generator
from rationale_query.generator import (
    ContextOverflowError, GenerationError, GenerationSettings, ModelMismatchError, TransformersGenerator,
)
from rationale_query.prompt import build_rationale_prompt
from tests.support import TempDirTestCase
from tests.unit.rationale_query.examples import FIGURE_4_QUESTION

BOS, EOS = 1, 2


class FakeTokenizer:
    bos_token_id = BOS
    eos_token_id = EOS

    def __init__(self):
        self.seen = []
        self.extra_bos = False

    def __call__(self, text):
        self.seen.append(text)
        ids = [BOS] + [100 + i for i, _ in enumerate(text.split())]
        if self.extra_bos:
            ids.append(BOS)
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"w{i}" for i in ids if not (skip_special_tokens and i in (BOS, EOS)))


class FakeModel:
    pass


class ModelFolderTestCase(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.folder = self.dir / "meerkat"
        self.folder.mkdir()
        (self.folder / "config.json").write_text("{}", encoding="utf-8")
        (self.folder / "model-00001-of-00001.safetensors").write_bytes(b"weights")
        (self.folder / "tokenizer.model").write_bytes(b"tokenizer")
        (self.folder / "README.md").write_text("not part of the identity", encoding="utf-8")
        self.tokenizer = FakeTokenizer()

    def loader(self, folder, settings):
        return FakeModel(), self.tokenizer

    def make(self, settings=None, reply=None, **kwargs):
        gen = TransformersGenerator(self.folder, settings, loader=self.loader, **kwargs)
        gen.calls = []

        def complete(ids, max_new_tokens):
            gen.calls.append((list(ids), max_new_tokens))
            return list(reply) if reply is not None else [200, 201, EOS]

        gen._complete = complete
        return gen


class FormatTests(unittest.TestCase):
    def test_by_default_the_model_reads_the_papers_text_alone(self):
        self.assertEqual(GenerationSettings().chat_format, "none")
        self.assertEqual(generator.render_model_input("PROMPT"), "PROMPT")

    def test_meerkats_own_template_is_a_named_option_for_a_labelled_check(self):
        self.assertEqual(generator.render_model_input("PROMPT", "meerkat"), "USER: PROMPT ASSISTANT: ")

    def test_a_blank_prompt_or_an_unknown_format_is_refused(self):
        for bad in ("", "  ", None, 3):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                generator.render_model_input(bad)
        with self.assertRaises(ValueError):
            generator.render_model_input("PROMPT", "vicuna")

    def test_bad_settings_are_refused(self):
        for kwargs in ({"n_ctx": 0}, {"n_ctx": 1.5}, {"n_ctx": True}, {"n_threads": 0}, {"max_gpu_gib": 0}, {"max_cpu_gib": -1}, {"chat_format": "vicuna"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GenerationSettings(**kwargs)

    def test_the_default_window_is_the_length_meerkat_was_trained_with(self):
        self.assertEqual(GenerationSettings().n_ctx, 2048)

    def test_decoding_is_greedy_and_written_out(self):
        self.assertEqual(generator.DECODING, {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0})
        self.assertEqual(generator.DTYPE, "bfloat16")


class GenerateTests(ModelFolderTestCase):
    def test_the_model_reads_the_papers_prompt_and_nothing_else(self):
        gen = self.make()
        prompt = build_rationale_prompt(FIGURE_4_QUESTION)
        gen.generate(prompt)
        self.assertEqual(self.tokenizer.seen, [prompt])

    def test_meerkats_template_is_used_only_when_asked_for(self):
        gen = self.make(GenerationSettings(chat_format="meerkat"))
        gen.generate("a b c")
        self.assertEqual(self.tokenizer.seen, ["USER: a b c ASSISTANT: "])
        self.assertEqual(gen.config["chat_format"], "meerkat")

    def test_a_second_start_token_is_an_error(self):
        gen = self.make()
        self.tokenizer.extra_bos = True
        with self.assertRaises(GenerationError):
            gen.generate("a b c")

    def test_there_is_no_length_limit_except_the_window(self):
        gen = self.make(GenerationSettings(n_ctx=300))
        result = gen.generate("a b c d e")  # start token and five words
        ids, max_new = gen.calls[0]
        self.assertEqual(result.prompt_tokens, 6)
        self.assertEqual(max_new, 300 - 6)

    def test_a_model_that_ends_its_answer_is_reported_as_stop_and_the_end_token_is_not_in_the_text(self):
        gen = self.make(reply=[200, 201, EOS])
        result = gen.generate("a b c")
        self.assertEqual((result.finish_reason, result.completion_tokens, result.text), ("stop", 3, "w200 w201"))

    def test_a_full_window_is_reported_as_length_and_the_text_is_kept(self):
        gen = self.make(reply=[200, 201, 202])
        result = gen.generate("a b c")
        self.assertEqual((result.finish_reason, result.text), ("length", "w200 w201 w202"))

    def test_a_model_that_writes_nothing_gives_an_empty_text_for_the_caller_to_refuse(self):
        gen = self.make(reply=[EOS])
        result = gen.generate("a b c")
        self.assertEqual((result.text, result.finish_reason), ("", "stop"))

    def test_a_prompt_that_does_not_fit_is_refused_and_never_shortened(self):
        gen = self.make(GenerationSettings(n_ctx=5))
        with self.assertRaises(ContextOverflowError):
            gen.generate("a b c d e")
        self.assertEqual(gen.calls, [])


class ModelFolderTests(ModelFolderTestCase):
    def test_the_configuration_records_the_model_by_its_hash_and_the_precision(self):
        gen = self.make()
        self.assertEqual(gen.config["model_folder"], "meerkat")
        self.assertEqual(len(gen.config["model_sha256"]), 64)
        self.assertEqual((gen.config["dtype"], gen.config["chat_format"], gen.config["n_ctx"]), ("bfloat16", "none", 2048))
        self.assertEqual(gen.config["decoding"], generator.DECODING)

    def test_the_hash_depends_on_the_weights_and_the_tokenizer_but_not_on_other_files(self):
        before = generator.folder_sha256(self.folder)
        (self.folder / "README.md").write_text("changed", encoding="utf-8")
        self.assertEqual(generator.folder_sha256(self.folder), before)
        (self.folder / "model-00001-of-00001.safetensors").write_bytes(b"other weights")
        self.assertNotEqual(generator.folder_sha256(self.folder), before)

    def test_a_folder_with_another_hash_than_the_pinned_one_is_refused(self):
        pinned = generator.folder_sha256(self.folder)
        self.make(expected_sha256=pinned.upper())
        with self.assertRaises(ModelMismatchError):
            self.make(expected_sha256="0" * 64)

    def test_a_folder_that_is_not_a_model_is_refused(self):
        with self.assertRaises(GenerationError):
            TransformersGenerator(self.dir / "absent", loader=self.loader)
        (self.folder / "model-00001-of-00001.safetensors").unlink()
        with self.assertRaises(GenerationError):
            TransformersGenerator(self.folder, loader=self.loader)

    def test_the_settings_reach_the_loader(self):
        seen = {}

        def loader(folder, settings):
            seen["settings"] = settings
            return FakeModel(), self.tokenizer

        settings = GenerationSettings(n_ctx=1024, n_threads=3, max_gpu_gib=3.2, max_cpu_gib=10, offload_dir="off")
        TransformersGenerator(self.folder, settings, loader=loader)
        self.assertEqual(seen["settings"], settings)

    def test_file_hash_is_sha256(self):
        path = self.folder / "tokenizer.model"
        self.assertEqual(generator.file_sha256(path), hashlib.sha256(b"tokenizer").hexdigest())


if __name__ == "__main__":
    unittest.main()
