"""GGUFRationaleScorer (rationale_gguf.py): verifies it computes the exact
same quantities Llama3RationaleScorer does (correctness, perplexity from
mean negative log-likelihood), against a fake ``llama_cpp.Llama`` and a
fake HF tokenizer - no real model, no real package, matching this
project's standing rule that nothing in the test suite touches a real
model or the GPU/ML stack (neither ``llama_cpp`` nor ``transformers`` is
even installed in this environment - confirmed before writing this file).
"""

import math
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from _archive.rag2_filter_reproduction.filter_training.labeling import RationaleOutcome
from _archive.rag2_filter_reproduction.filter_training.rationale import RationaleGenerationError


class FakeTokenizer:
    """Stand-in for AutoTokenizer.from_pretrained's return value. Mirrors
    Llama-3's real chat template shape (a leading literal BOS token) so the
    duplicate-BOS stripping logic is actually exercised, not just assumed
    correct - this is exactly the class of bug a too-simple fake would
    have hidden (a real run caught it; the fake below is written to catch
    it next time without needing a real run)."""

    chat_template = None
    bos_token = "<|begin_of_text|>"

    def __init__(self, *, with_chat_template=False):
        if with_chat_template:
            self.chat_template = "fake-template"
        self.chat_calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.chat_calls.append(messages)
        return f"{self.bos_token}<chat>{messages[0]['content']}</chat>"


class FakeLlama:
    """Stand-in for llama_cpp.Llama. Records init kwargs and completion
    calls; a test sets .next_completion before calling score()."""

    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.completion_calls = []
        self.next_completion = None
        FakeLlama.instances.append(self)

    def create_completion(self, prompt, **kwargs):
        self.completion_calls.append((prompt, kwargs))
        return self.next_completion


def _completion(text, token_logprobs):
    return {"choices": [{"text": text, "logprobs": {"token_logprobs": token_logprobs}}]}


class GGUFRationaleScorerTests(unittest.TestCase):

    def setUp(self):
        self._saved_modules = {
            name: sys.modules.get(name) for name in ("llama_cpp", "transformers")
        }
        FakeLlama.instances = []
        self._tmp = TemporaryDirectory()
        self.model_path = Path(self._tmp.name) / "model.gguf"
        self.model_path.write_bytes(b"not a real gguf file, just needs to exist")

    def tearDown(self):
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        self._tmp.cleanup()

    def _install_fakes(self, *, tokenizer=None, llama_cls=FakeLlama):
        tokenizer = tokenizer or FakeTokenizer()
        llama_cpp_module = types.ModuleType("llama_cpp")
        llama_cpp_module.Llama = llama_cls
        sys.modules["llama_cpp"] = llama_cpp_module

        transformers_module = types.ModuleType("transformers")

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(model_id, revision=None):
                return tokenizer

        transformers_module.AutoTokenizer = _AutoTokenizer
        sys.modules["transformers"] = transformers_module
        return tokenizer

    def _make_scorer(self, **overrides):
        from _archive.rag2_filter_reproduction.filter_training.rationale_gguf import GGUFRationaleScorer
        kwargs = dict(
            model_path=self.model_path,
            tokenizer_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            tokenizer_revision="abc123pinnedsha",
        )
        kwargs.update(overrides)
        return GGUFRationaleScorer(**kwargs)

    # -- construction guards -------------------------------------------

    def test_missing_model_file_is_refused(self):
        self._install_fakes()
        from _archive.rag2_filter_reproduction.filter_training.rationale_gguf import GGUFRationaleScorer
        with self.assertRaises(RationaleGenerationError) as ctx:
            GGUFRationaleScorer(
                model_path=Path(self._tmp.name) / "does_not_exist.gguf",
                tokenizer_model_id="x", tokenizer_revision="abc123",
            )
        self.assertIn("not found", str(ctx.exception))

    def test_missing_llama_cpp_package_is_refused(self):
        sys.modules.pop("llama_cpp", None)
        with self.assertRaises(RationaleGenerationError) as ctx:
            self._make_scorer()
        self.assertIn("llama-cpp-python", str(ctx.exception))

    def test_missing_transformers_package_is_refused(self):
        llama_cpp_module = types.ModuleType("llama_cpp")
        llama_cpp_module.Llama = FakeLlama
        sys.modules["llama_cpp"] = llama_cpp_module
        sys.modules.pop("transformers", None)
        with self.assertRaises(RationaleGenerationError) as ctx:
            self._make_scorer()
        self.assertIn("transformers", str(ctx.exception))

    def test_unpinned_revision_is_refused(self):
        self._install_fakes()
        with self.assertRaises(RationaleGenerationError) as ctx:
            self._make_scorer(tokenizer_revision="main")
        self.assertIn("pinned commit sha", str(ctx.exception))

    # -- score() correctness --------------------------------------------

    def test_score_matches_the_mean_nll_perplexity_formula(self):
        self._install_fakes()
        scorer = self._make_scorer()
        llama = FakeLlama.instances[-1]
        logprobs = [-0.1, -0.2, -0.3, -0.4]
        llama.next_completion = _completion("Reasoning...\nAnswer: B", logprobs)

        outcome = scorer.score(
            "What treats it?", ["wrong", "right answer", "c", "d"],
            "right answer", evidence="some evidence",
        )

        expected_perplexity = math.exp(-sum(logprobs) / len(logprobs))
        self.assertAlmostEqual(outcome.perplexity, expected_perplexity, places=9)
        self.assertTrue(outcome.correct)  # letter B -> "right answer" matches answer

    def test_wrong_predicted_letter_is_scored_incorrect(self):
        self._install_fakes()
        scorer = self._make_scorer()
        llama = FakeLlama.instances[-1]
        llama.next_completion = _completion("Answer: A", [-0.5])

        outcome = scorer.score(
            "Q?", ["wrong", "right answer", "c", "d"], "right answer", evidence=None,
        )
        self.assertFalse(outcome.correct)

    def test_chat_template_is_applied_when_the_tokenizer_has_one(self):
        tokenizer = FakeTokenizer(with_chat_template=True)
        self._install_fakes(tokenizer=tokenizer)
        scorer = self._make_scorer()
        llama = FakeLlama.instances[-1]
        llama.next_completion = _completion("Answer: A", [-0.1])

        scorer.score("Q?", ["a", "b", "c", "d"], "a", evidence=None)

        self.assertEqual(len(tokenizer.chat_calls), 1)
        prompt_sent, _kwargs = llama.completion_calls[0]
        self.assertTrue(prompt_sent.startswith("<chat>"))

    def test_no_chat_template_sends_the_raw_prompt(self):
        self._install_fakes(tokenizer=FakeTokenizer(with_chat_template=False))
        scorer = self._make_scorer()
        llama = FakeLlama.instances[-1]
        llama.next_completion = _completion("Answer: A", [-0.1])

        scorer.score("Q?", ["a", "b", "c", "d"], "a", evidence=None)

        prompt_sent, _kwargs = llama.completion_calls[0]
        self.assertFalse(prompt_sent.startswith("<chat>"))
        self.assertIn("Q?", prompt_sent)

    def test_greedy_decoding_is_used(self):
        self._install_fakes()
        scorer = self._make_scorer()
        llama = FakeLlama.instances[-1]
        llama.next_completion = _completion("Answer: A", [-0.1])

        scorer.score("Q?", ["a", "b", "c", "d"], "a", evidence=None)

        _prompt, kwargs = llama.completion_calls[0]
        self.assertEqual(kwargs["temperature"], 0.0)

    def test_empty_rationale_is_refused(self):
        self._install_fakes()
        scorer = self._make_scorer()
        llama = FakeLlama.instances[-1]
        llama.next_completion = _completion("   ", [])
        with self.assertRaises(RationaleGenerationError) as ctx:
            scorer.score("Q?", ["a", "b", "c", "d"], "a", evidence=None)
        self.assertIn("empty rationale", str(ctx.exception))

    def test_none_logprobs_are_dropped_not_averaged_as_zero(self):
        self._install_fakes()
        scorer = self._make_scorer()
        llama = FakeLlama.instances[-1]
        # First token's logprob reported as None (a real llama.cpp quirk) -
        # must be excluded from the mean, not treated as 0.0 (which would
        # silently pull perplexity toward 1.0).
        llama.next_completion = _completion("Answer: A", [None, -0.2, -0.2])

        outcome = scorer.score("Q?", ["a", "b", "c", "d"], "a", evidence=None)

        expected = math.exp(-(-0.2 + -0.2) / 2)
        self.assertAlmostEqual(outcome.perplexity, expected, places=9)

    def test_all_none_logprobs_is_refused(self):
        self._install_fakes()
        scorer = self._make_scorer()
        llama = FakeLlama.instances[-1]
        llama.next_completion = _completion("Answer: A", [None, None])
        with self.assertRaises(RationaleGenerationError) as ctx:
            scorer.score("Q?", ["a", "b", "c", "d"], "a", evidence=None)
        self.assertIn("log-probabilities", str(ctx.exception))

    def test_wrong_number_of_choices_is_refused(self):
        self._install_fakes()
        scorer = self._make_scorer()
        with self.assertRaises(RationaleGenerationError):
            scorer.score("Q?", ["a", "b", "c"], "a", evidence=None)

    def test_cpu_only_is_the_default(self):
        """n_gpu_layers defaults to 0 - correctness must not depend on GPU
        offload being available."""
        self._install_fakes()
        self._make_scorer()
        llama = FakeLlama.instances[-1]
        self.assertEqual(llama.init_kwargs["n_gpu_layers"], 0)

    def test_logits_all_is_enabled(self):
        """Required for create_completion(logprobs=...) to work at all -
        a real run without this raised ValueError: 'logprobs is not
        supported for models created with logits_all=False' (2026-09-28).
        """
        self._install_fakes()
        self._make_scorer()
        llama = FakeLlama.instances[-1]
        self.assertTrue(llama.init_kwargs["logits_all"])

    def test_duplicate_leading_bos_token_is_stripped(self):
        """Llama-3's chat template already starts with a literal BOS
        token, and llama.cpp adds its own by default - sending both
        produced a real 'duplicate leading <|begin_of_text|>' warning on
        an actual run (2026-09-28). The prompt handed to the model must
        carry the BOS token at most once."""
        tokenizer = FakeTokenizer(with_chat_template=True)
        self._install_fakes(tokenizer=tokenizer)
        scorer = self._make_scorer()
        llama = FakeLlama.instances[-1]
        llama.next_completion = _completion("Answer: A", [-0.1])

        scorer.score("Q?", ["a", "b", "c", "d"], "a", evidence=None)

        prompt_sent, _kwargs = llama.completion_calls[0]
        self.assertEqual(prompt_sent.count(tokenizer.bos_token), 0)

    def test_name_is_recorded_for_provenance(self):
        self._install_fakes()
        from _archive.rag2_filter_reproduction.filter_training.rationale_gguf import GGUFRationaleScorer
        self.assertTrue(GGUFRationaleScorer.name)
        self.assertIn("gguf", GGUFRationaleScorer.name.lower())


if __name__ == "__main__":
    unittest.main()
