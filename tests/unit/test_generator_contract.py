"""The generator execution contract
(docs/research_experimental_specification.md §9).

No model is loaded here and none should ever be: these tests lock the
reproducibility guarantees around generation, which are configuration
properties, not model behaviour.

The guarantees: the weights are pinned hard enough to reproduce, decoding
cannot silently become stochastic, and the recorded config cannot describe
settings the run did not use.
"""

import unittest

from systems.interfaces.hf_generator import (
    FALLBACK_GENERATOR_ID,
    RAG2_GENERATOR_ID,
    GenerationConfig,
    GeneratorError,
    HuggingFaceGenerator,
    ModelSpec,
)

SHA = "0e9e39f249a16976918f6564b8830bc894c89659"


class ModelSpecTests(unittest.TestCase):

    def test_a_pinned_sha_is_accepted(self):
        spec = ModelSpec(model_id=RAG2_GENERATOR_ID, revision=SHA,
                         quantization="nf4")
        self.assertEqual(spec.revision, SHA)

    def test_a_branch_name_is_refused(self):
        """'main' resolves to different weights over time."""
        with self.assertRaises(GeneratorError) as ctx:
            ModelSpec(model_id=RAG2_GENERATOR_ID, revision="main")
        self.assertIn("would not be reproducible", str(ctx.exception))

    def test_an_empty_revision_is_refused(self):
        with self.assertRaises(GeneratorError):
            ModelSpec(model_id=RAG2_GENERATOR_ID, revision="")

    def test_an_unknown_quantization_is_refused(self):
        with self.assertRaises(GeneratorError):
            ModelSpec(model_id="m", revision=SHA, quantization="fp4-ish")

    def test_fingerprint_separates_precisions_of_the_same_weights(self):
        full = ModelSpec(model_id="m", revision=SHA)
        quantized = ModelSpec(model_id="m", revision=SHA, quantization="nf4")
        self.assertNotEqual(full.fingerprint(), quantized.fingerprint())

    def test_fingerprint_is_stable_across_equal_specs(self):
        a = ModelSpec(model_id="m", revision=SHA, quantization="nf4")
        b = ModelSpec(model_id="m", revision=SHA, quantization="nf4")
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_the_contract_names_rag2s_own_generator(self):
        """The model is kept, not swapped for something that fits locally."""
        self.assertEqual(RAG2_GENERATOR_ID,
                         "meta-llama/Meta-Llama-3-8B-Instruct")
        self.assertEqual(FALLBACK_GENERATOR_ID, "Qwen/Qwen2.5-7B-Instruct")

    def test_no_revision_is_pre_filled_in_code(self):
        """A placeholder sha would be fabricated provenance."""
        self.assertIsInstance(RAG2_GENERATOR_ID, str)
        with self.assertRaises(TypeError):
            ModelSpec(model_id=RAG2_GENERATOR_ID)  # revision has no default


class GenerationConfigTests(unittest.TestCase):

    def test_the_default_is_greedy(self):
        config = GenerationConfig()
        self.assertFalse(config.do_sample)
        self.assertIsNone(config.temperature)
        self.assertIsNone(config.top_p)

    def test_sampling_is_refused_with_the_reason_stated(self):
        with self.assertRaises(GeneratorError) as ctx:
            GenerationConfig(do_sample=True)
        self.assertIn("decoding variance", str(ctx.exception))

    def test_decoding_parameters_that_would_be_ignored_are_refused(self):
        """Recording a temperature a greedy run ignored misdescribes it."""
        with self.assertRaises(GeneratorError) as ctx:
            GenerationConfig(temperature=0.7)
        self.assertIn("Greedy decoding ignores them", str(ctx.exception))

    def test_a_non_positive_answer_cap_is_refused(self):
        with self.assertRaises(GeneratorError):
            GenerationConfig(max_new_tokens=0)

    def test_the_recorded_config_round_trips(self):
        config = GenerationConfig(max_new_tokens=128)
        self.assertEqual(config.to_dict(), {
            "max_new_tokens": 128, "do_sample": False,
            "temperature": None, "top_p": None,
        })


class GeneratorConstructionTests(unittest.TestCase):
    """Constructing must not load anything - loading happens in load()."""

    def test_construction_loads_no_model(self):
        generator = HuggingFaceGenerator(
            ModelSpec(model_id=RAG2_GENERATOR_ID, revision=SHA,
                      quantization="nf4"))
        self.assertIsNone(generator._model)
        self.assertEqual(generator.config.max_new_tokens, 256)

    def test_one_instance_is_what_both_arms_share(self):
        """Parity is object identity, so the arms must take the same object."""
        from experiments.evaluation.runner import assert_generator_parity
        from systems.baseline.admission import MockRAG2Filter
        from systems.baseline.rag2 import RAG2Config, RAG2System

        generator = HuggingFaceGenerator(
            ModelSpec(model_id=RAG2_GENERATOR_ID, revision=SHA))
        arms = {
            "baseline": RAG2System(answer_generator=generator,
                                   admission_filter=MockRAG2Filter({}),
                                   config=RAG2Config(max_admitted_passages=5)),
            "other": RAG2System(answer_generator=generator,
                                admission_filter=MockRAG2Filter({}),
                                config=RAG2Config(max_admitted_passages=5)),
        }
        assert_generator_parity(arms)  # does not raise


if __name__ == "__main__":
    unittest.main()
