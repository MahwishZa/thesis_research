"""rationale_filter.scorer with a stand-in for llama.cpp whose outputs the test fixes, so the alignment of the scores is checked."""

import math
import unittest

from rationale_filter import scorer
from rationale_query.generator import ContextOverflowError, ModelMismatchError, llama3_prompt_tokens
from tests.support import FakeLlama, TempDirTestCase, np

VOCAB = 400
BOOST = 9.0


@unittest.skipIf(np is None, "numpy is not installed")
class ScorerTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.model = self.write("model.gguf", "not a real model").parent / "model.gguf"
        self.made = []

    def factory(self, **options):
        llm = ScoringLlama(**options)
        self.made.append(llm)
        return llm

    def make(self, **options):
        return scorer.LlamaLogprobScorer(self.model, llama_factory=self.factory, **options)

    def favoured_chain(self, llm, message, completion):
        """The tokens of the prompt and the text, and for every position the token the stand-in makes likely next."""
        prompt = llama3_prompt_tokens(llm, message)
        text = llm.tokenize(completion.encode("utf-8"), add_bos=False, special=False)
        tokens = prompt + text
        return prompt, text, tokens, {position: tokens[position + 1] for position in range(len(tokens) - 1)}

    def test_each_token_is_scored_by_the_output_of_the_position_before_it(self):
        s = self.make()
        llm = self.made[0]
        message, completion = "Which drug is a cholinesterase inhibitor?", "Donepezil, it is."
        prompt, text, tokens, favoured = self.favoured_chain(llm, message, completion)
        llm.favoured = favoured
        result = s.token_logprobs(message, completion)
        self.assertEqual(len(result), len(text))
        expected = -math.log1p((VOCAB - 1) * math.exp(-BOOST))  # the boosted token among VOCAB - 1 others at logit 0
        for value in result:
            self.assertAlmostEqual(value, expected, places=6)

    def test_an_off_by_one_alignment_would_be_far_from_zero(self):
        # The same stand-in, favouring the token after the next one: the scorer must then report very low log-probabilities.
        s = self.make()
        llm = self.made[0]
        message, completion = "Which drug?", "Donepezil"
        _, text, tokens, _ = self.favoured_chain(llm, message, completion)
        llm.favoured = {position: tokens[position + 2] for position in range(len(tokens) - 2)}
        self.assertTrue(all(value < -5 for value in s.token_logprobs(message, completion)[:-1]))

    def test_a_token_that_is_not_favoured_gets_a_low_log_probability(self):
        s = self.make()
        llm = self.made[0]
        message, completion = "Which drug?", "Dose"
        _, text, tokens, favoured = self.favoured_chain(llm, message, completion)
        other = {position: (token + 1) % 256 for position, token in favoured.items()}
        llm.favoured = other
        self.assertTrue(all(value < -5 for value in s.token_logprobs(message, completion)))

    def test_the_log_probabilities_of_all_tokens_sum_to_one_after_exponentiation(self):
        s = self.make()
        llm = self.made[0]
        llm.favoured = {}
        llm.flat = True
        value = s.token_logprobs("Q?", "ab")
        self.assertTrue(all(abs(v - (-math.log(VOCAB))) < 1e-9 for v in value))  # all logits equal: uniform

    def test_the_cache_is_reset_before_every_evaluation_and_the_same_tokens_are_given(self):
        s = self.make()
        llm = self.made[0]
        s.token_logprobs("Q?", "text")
        s.token_logprobs("Q?", "text")
        self.assertEqual(llm.events.count("reset"), 2)
        self.assertEqual(llm.evaluated[0], llm.evaluated[1])
        self.assertEqual(llm.events[-2:], ["reset", "eval"])

    def test_the_model_is_opened_with_logits_for_every_position(self):
        self.make(n_ctx=2048, n_batch=256, n_gpu_layers=3)
        options = self.made[0].options
        self.assertTrue(options["logits_all"])
        self.assertEqual((options["n_ctx"], options["n_batch"], options["n_gpu_layers"]), (2048, 256, 3))

    def test_a_prompt_and_text_that_do_not_fit_are_refused_before_evaluation(self):
        s = self.make(n_ctx=20)
        with self.assertRaises(ContextOverflowError):
            s.token_logprobs("a longer question than twenty tokens allow here", "and a text that is long too")
        self.assertNotIn("eval", self.made[0].events)

    def test_an_empty_text_is_refused(self):
        s = self.make()
        for bad in ("", "   ", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                s.token_logprobs("Q?", bad)

    def test_the_hash_of_the_file_is_checked_and_recorded(self):
        with self.assertRaises(ModelMismatchError):
            scorer.LlamaLogprobScorer(self.model, expected_sha256="0" * 64, llama_factory=self.factory)
        s = self.make()
        self.assertEqual(len(s.config["model_sha256"]), 64)
        self.assertEqual(s.config["n_ctx"], 4096)

    def test_a_missing_file_and_bad_sizes_are_refused(self):
        with self.assertRaises(FileNotFoundError):
            scorer.LlamaLogprobScorer(self.dir / "missing.gguf", llama_factory=self.factory)
        for bad in (0, -1, 2.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.make(n_ctx=bad)

    def test_close_closes_the_model(self):
        s = self.make()
        s.close()
        self.assertTrue(self.made[0].closed)


class ScoringLlama(FakeLlama):
    """FakeLlama with ``eval`` and ``scores``. The output at a position is flat (all logits 0) except for the token in ``favoured``."""

    def __init__(self, **options):
        super().__init__(**options)
        self.favoured = {}
        self.flat = False
        self.evaluated = []
        self.n_tokens = 0
        self.scores = np.zeros((0, VOCAB), dtype=np.float32)

    def reset(self):
        super().reset()
        self.n_tokens = 0

    def eval(self, tokens):
        self.events.append("eval")
        self.evaluated.append(list(tokens))
        rows = np.zeros((len(tokens), VOCAB), dtype=np.float32)
        if not self.flat:
            for position, token in self.favoured.items():
                if position < len(tokens):
                    rows[position, token] = BOOST
        self.scores = rows
        self.n_tokens = len(tokens)


if __name__ == "__main__":
    unittest.main()
