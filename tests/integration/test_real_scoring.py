"""rationale_filter.scorer against the real llama.cpp, on the tiny synthetic model of tests/integration/tiny_models.py.

The tiny model predicts, after each token, the next token of a fixed chain, so a text that follows the chain must get
log-probabilities near zero and a text that leaves it must get very low ones. Needs llama-cpp-python, gguf and numpy; the class
skips, with the reason, when they are missing. Nothing is downloaded.
"""

import tempfile
import unittest
from pathlib import Path

from rationale_filter import perplexity, scorer
from rationale_query.generator import ContextOverflowError
from tests.integration.test_real_libraries import unusable
from tests.integration.tiny_models import ANSWER_TEXT, ANSWER_TOKENS, write_tiny_gguf
from tests.support import TempDirTestCase

QUESTION = "Which drug is a cholinesterase inhibitor? A) Donepezil B) Aspirin C) Insulin D) Warfarin"


class RealScoringTests(TempDirTestCase):
    @classmethod
    def setUpClass(cls):
        reason = unusable("llama_cpp", "gguf", "numpy")
        if reason:
            raise unittest.SkipTest(reason)
        holder = tempfile.TemporaryDirectory(prefix="rf models ", ignore_cleanup_errors=True)
        cls.addClassCleanup(holder.cleanup)
        cls.model = Path(holder.name) / "chain.gguf"
        write_tiny_gguf(cls.model)

    def open(self, **options):
        settings = {"n_ctx": 512, "n_batch": 64, "n_threads": 1}
        settings.update(options)
        opened = scorer.LlamaLogprobScorer(self.model, **settings)
        self.addCleanup(opened.close)  # a memory-mapped file cannot be deleted on Windows while it is open
        return opened

    def test_a_text_that_follows_the_chain_is_almost_certain(self):
        values = self.open().token_logprobs(QUESTION, ANSWER_TEXT)
        self.assertEqual(len(values), ANSWER_TOKENS)
        self.assertTrue(all(value > -0.01 for value in values), values)
        self.assertLess(perplexity.perplexity(values), 1.001)

    def test_a_text_that_leaves_the_chain_is_very_unlikely_where_it_leaves(self):
        values = self.open().token_logprobs(QUESTION, " The answer is C.")
        self.assertTrue(all(value > -0.01 for value in values[:3]), values)  # " The answer is" follows the chain
        self.assertLess(min(values[3:]), -10.0)
        self.assertGreater(perplexity.perplexity(values), 10.0)

    def test_the_same_text_gets_the_same_numbers_whatever_was_scored_before(self):
        opened = self.open()
        first = opened.token_logprobs(QUESTION, ANSWER_TEXT)
        opened.token_logprobs("A much longer question " * 5 + QUESTION, " The answer is C. and more words follow")
        self.assertEqual(opened.token_logprobs(QUESTION, ANSWER_TEXT), first)

    def test_the_prompt_changes_nothing_when_the_model_ignores_it(self):
        # The tiny model looks at the previous token only, so a snippet in the prompt cannot change the scores; the check
        # is that the evidence section is scored as part of the prompt and not as part of the text.
        opened = self.open()
        plain = opened.token_logprobs(QUESTION, ANSWER_TEXT)
        with_evidence = opened.token_logprobs("Evidence:\nDonepezil inhibits cholinesterase.\nHere is the question: " + QUESTION, ANSWER_TEXT)
        self.assertEqual(len(plain), len(with_evidence))
        for one, other in zip(plain, with_evidence, strict=True):
            self.assertAlmostEqual(one, other, places=4)

    def test_the_perplexity_pair_of_a_text_is_equal_for_a_model_that_ignores_the_prompt(self):
        without, with_snippet = perplexity.rationale_perplexities(
            self.open(), prompt_without=QUESTION, prompt_with="Evidence:\nSome snippet.\nHere is the question: " + QUESTION, rationale=ANSWER_TEXT
        )
        self.assertAlmostEqual(without, with_snippet, places=4)

    def test_a_prompt_that_does_not_fit_is_refused(self):
        with self.assertRaises(ContextOverflowError):
            self.open(n_ctx=24).token_logprobs(QUESTION, ANSWER_TEXT)

    def test_the_config_records_the_file_and_the_sizes(self):
        config = self.open().config
        self.assertEqual((config["model_file"], config["n_ctx"], config["n_batch"]), ("chain.gguf", 512, 64))
        self.assertEqual(len(config["model_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
