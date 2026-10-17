"""Checks against the real model files. They run only when asked, because they need a 5 GB file or a 440 MB download.

In PowerShell, from the repository folder (the paths are examples):

    $env:RATIONALE_LLAMA3_GGUF = "C:\\Users\\mahwi\\...\\models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
    $env:RATIONALE_MEDCPT = "1"
    python -m unittest tests.integration.test_real_models -v

``RATIONALE_MEDCPT=1`` loads ncbi/MedCPT-Query-Encoder (downloaded on first use, then read from the Hugging Face cache).
Any other value is taken as another encoder: a hub name, or a folder, which is then loaded without the network.

``RETRIEVAL_MEDCPT=1`` loads ncbi/MedCPT-Article-Encoder and ncbi/MedCPT-Cross-Encoder, the two models of Part 2 (each
about 440 MB on first use). Without the variables every test here is skipped and says why.
"""

import contextlib
import os
import unittest
from pathlib import Path

from rationale_query.generator import (
    GenerationSettings,
    LlamaCppGenerator,
    llama3_prompt_tokens,
)
from rationale_query.prompt import build_rationale_prompt
from tests.support import pinned_sha256

LLAMA3_FILE = os.environ.get("RATIONALE_LLAMA3_GGUF", "")
ENCODER = os.environ.get("RATIONALE_MEDCPT", "")
RETRIEVAL_MODELS = os.environ.get("RETRIEVAL_MEDCPT", "")

QUESTION_A = build_rationale_prompt(
    "A 68-year-old man is brought in by his wife because of progressive forgetfulness over two years. He now gets lost "
    "on familiar streets and cannot manage his medications. Neurological examination shows no focal deficit. Which of "
    "the following drugs is most appropriate to start? A) Donepezil B) Haloperidol C) Levodopa D) Phenytoin"
)
QUESTION_B = build_rationale_prompt(
    "A 30-year-old woman has a malar rash, joint pain and a positive anti-double-stranded DNA antibody test. Which of "
    "the following is the most likely diagnosis? A) Rheumatoid arthritis B) Systemic lupus erythematosus "
    "C) Polymyositis D) Gout"
)


@unittest.skipUnless(LLAMA3_FILE, "set RATIONALE_LLAMA3_GGUF to a Llama-3-8B-Instruct .gguf file to run these checks")
class RealLlama3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(LLAMA3_FILE)
        settings = GenerationSettings(model_name=path.stem, max_new_tokens=24, n_ctx=2048)
        cls.generator = LlamaCppGenerator(path, settings, expected_sha256=pinned_sha256(path.name))
        cls.addClassCleanup(cls.generator.close)

    def test_the_file_is_pinned_when_the_settings_list_it(self):
        # A hash mismatch would already have stopped setUpClass; this records which kind of check was done.
        expected = pinned_sha256(Path(LLAMA3_FILE).name)
        if expected is None:
            self.skipTest("this file name is not pinned in experiments/01_rationale_writers/settings.json")
        self.assertEqual(self.generator.config["model_sha256"], expected)

    def test_the_template_in_the_file_renders_the_llama3_prompt(self):
        self.assertTrue(
            self.generator.config["template_check"].startswith("matches"), self.generator.config["template_check"]
        )

    def test_a_rationale_is_written_and_repeats_exactly_after_another_question(self):
        first = self.generator.generate(QUESTION_A)
        self.generator.generate(QUESTION_B)
        again = self.generator.generate(QUESTION_A)
        self.assertEqual(again, first)
        self.assertTrue(first.text.strip())
        self.assertIn(first.finish_reason, ("stop", "length"))
        self.assertLessEqual(first.completion_tokens, 24)

    def test_the_reported_prompt_length_is_that_of_the_llama3_format(self):
        result = self.generator.generate(QUESTION_A)
        self.assertEqual(result.prompt_tokens, len(llama3_prompt_tokens(self.generator._llm, QUESTION_A)))

    def test_the_cache_holds_the_prompt_with_one_start_of_text_token(self):
        self.generator.generate(QUESTION_B)
        llm = self.generator._llm
        tokens = llama3_prompt_tokens(llm, QUESTION_B)
        self.assertEqual(list(llm.input_ids[: len(tokens)]), tokens)
        self.assertEqual(list(llm.input_ids).count(llm.token_bos()), 1)


@unittest.skipUnless(LLAMA3_FILE, "set RATIONALE_LLAMA3_GGUF to a Llama-3-8B-Instruct .gguf file to run these checks")
class RealThreadTests(unittest.TestCase):
    def test_one_thread_and_the_default_number_write_the_same_rationale(self):
        # The thread count is recorded but does not bind an output folder; this is the check that it may be left out.
        path = Path(LLAMA3_FILE)
        results = []
        for threads in (1, None):
            settings = GenerationSettings(model_name=path.stem, max_new_tokens=24, n_ctx=2048, n_threads=threads)
            with contextlib.closing(LlamaCppGenerator(path, settings)) as generator:
                results.append(generator.generate(QUESTION_A))
        self.assertEqual(results[0], results[1])


@unittest.skipUnless(ENCODER, "set RATIONALE_MEDCPT=1 to run these checks with ncbi/MedCPT-Query-Encoder")
class RealEncoderTests(unittest.TestCase):
    TEXTS = (
        (
            "The patient is an elderly man with progressive memory loss. Cholinesterase inhibitors such as donepezil "
            "are first-line treatment for mild Alzheimer disease. The answer is A) Donepezil."
        ),
        "Systemic lupus erythematosus presents with a malar rash and anti-dsDNA antibodies. The answer is B.",
    )

    @classmethod
    def setUpClass(cls):
        import numpy

        from rationale_query.encoder import ENCODER_NAME, QueryEncoder

        cls.np = numpy
        cls.is_medcpt = ENCODER == "1"
        name = ENCODER_NAME if cls.is_medcpt else ENCODER
        cls.encoder = QueryEncoder.from_pretrained(name, device="cpu", local_files_only=Path(name).is_dir())

    def test_the_model_is_in_evaluation_mode(self):
        self.assertFalse(self.encoder.model.training)

    def test_one_float32_vector_per_text_of_the_width_of_the_model(self):
        vectors = self.encoder.encode(list(self.TEXTS))
        self.assertEqual(vectors.shape, (2, self.encoder.model.config.hidden_size))
        self.assertEqual(vectors.dtype, self.np.float32)
        if self.is_medcpt:
            self.assertEqual(vectors.shape[1], 768)

    def test_the_vectors_are_not_normalised(self):
        if not self.is_medcpt:
            self.skipTest("the norm of a random test model says nothing about MedCPT")
        norms = self.np.linalg.norm(self.encoder.encode(list(self.TEXTS)), axis=1)
        self.assertTrue(self.np.all(self.np.abs(norms - 1.0) > 0.05), norms)

    def test_a_batch_gives_the_same_vectors_as_one_text_at_a_time(self):
        from rationale_query.encoder import QueryEncoder

        single = self.encoder.encode(list(self.TEXTS))
        batched = QueryEncoder(
            self.encoder.tokenizer, self.encoder.model, device="cpu", batch_size=2, no_grad=self.encoder._no_grad
        ).encode(list(self.TEXTS))
        self.np.testing.assert_allclose(batched, single, atol=1e-4)

    def test_the_same_text_gives_the_same_vector_every_time(self):
        self.assertTrue(
            self.np.array_equal(self.encoder.encode(list(self.TEXTS)), self.encoder.encode(list(self.TEXTS)))
        )

    def test_the_vectors_are_those_of_the_authors_code(self):
        """The lines of the authors' retriever/query_encode.py, with their graphics card replaced by the processor."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        name = self.encoder.name
        local = Path(name).is_dir()
        model = AutoModel.from_pretrained(name, local_files_only=local)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local)
        queries = []
        input_list = list(self.TEXTS)
        for i in range(0, len(input_list), 1):
            split_queries = input_list[i : i + 1]
            with torch.no_grad():
                encoded = tokenizer(split_queries, truncation=True, padding=True, return_tensors="pt", max_length=512)
                embeds = model(**encoded).last_hidden_state[:, 0, :]
                queries.extend(embeds.detach().cpu().numpy())
        self.assertTrue(self.np.array_equal(self.encoder.encode(input_list), self.np.vstack(queries)))

    def test_token_counts_are_the_real_lengths_and_long_texts_are_cut_at_512(self):
        tokenizer = self.encoder.tokenizer
        long_text = " ".join(["patient"] * 700)
        counts = self.encoder.token_counts([*self.TEXTS, long_text])
        self.assertEqual(counts[:2], [len(tokenizer(text)["input_ids"]) for text in self.TEXTS])
        self.assertGreater(counts[2], 512)
        self.assertEqual(len(tokenizer(long_text, truncation=True, max_length=512)["input_ids"]), 512)
        self.assertEqual(self.encoder.encode([long_text]).shape[0], 1)


@unittest.skipUnless(RETRIEVAL_MODELS, "set RETRIEVAL_MEDCPT=1 to run these checks with the MedCPT article encoder and cross-encoder")
class RealRetrievalModelTests(unittest.TestCase):
    RELEVANT = "Cholinesterase inhibitors such as donepezil are first-line treatment for mild Alzheimer disease."
    IRRELEVANT = "Systemic lupus erythematosus presents with a malar rash and anti-dsDNA antibodies."
    QUERY = "Which drug treats memory loss in Alzheimer disease?"

    @classmethod
    def setUpClass(cls):
        import numpy

        from balanced_retrieval.embedding import ArticleEncoder
        from balanced_retrieval.scorer import CrossEncoderScorer

        cls.np = numpy
        cls.articles = ArticleEncoder.from_pretrained(device="cpu")
        cls.scorer = CrossEncoderScorer.from_pretrained(device="cpu", batch_size=2)

    def test_the_article_encoder_gives_unnormalised_float32_vectors_of_width_768(self):
        vectors = self.articles.encode([self.RELEVANT, self.IRRELEVANT])
        self.assertEqual((vectors.shape, vectors.dtype), ((2, 768), self.np.float32))
        norms = self.np.linalg.norm(vectors, axis=1)
        self.assertTrue(self.np.all(self.np.abs(norms - 1.0) > 0.05), norms)
        self.assertFalse(self.articles.model.training)

    def test_a_passage_about_the_topic_is_closer_to_the_query_than_an_unrelated_one(self):
        from rationale_query.encoder import QueryEncoder

        query = QueryEncoder.from_pretrained(device="cpu").encode([self.QUERY])[0]
        near, far = self.articles.encode([self.RELEVANT, self.IRRELEVANT]) @ query
        self.assertGreater(near, far)

    def test_the_cross_encoder_scores_the_passage_about_the_topic_higher(self):
        candidates = [{"text": self.IRRELEVANT}, {"text": self.RELEVANT}]
        irrelevant, relevant = self.scorer(self.QUERY, candidates)
        self.assertGreater(relevant, irrelevant)
        self.assertFalse(self.scorer.model.training)

    def test_the_cross_encoder_scores_do_not_depend_on_the_batch_size(self):
        candidates = [{"text": self.IRRELEVANT}, {"text": self.RELEVANT}, {"text": "Aspirin lowers the risk of stroke."}]
        from balanced_retrieval.scorer import CrossEncoderScorer

        one_call = CrossEncoderScorer(self.scorer.tokenizer, self.scorer.model, batch_size=3, no_grad=self.scorer._no_grad)
        self.np.testing.assert_allclose(self.scorer(self.QUERY, candidates), one_call(self.QUERY, candidates), atol=1e-3)

    def test_a_pair_longer_than_512_tokens_is_cut_and_scored(self):
        long_text = " ".join(["patient"] * 700)
        self.assertEqual(len(self.scorer(self.QUERY, [{"text": long_text}])), 1)
        self.assertEqual(self.articles.encode([long_text]).shape[0], 1)


if __name__ == "__main__":
    unittest.main()
