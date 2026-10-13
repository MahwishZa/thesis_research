"""The encoders: the test doubles are deterministic, and the MedCPT classes load nothing until they are used (no torch needed)."""

import sys
import unittest
from unittest import mock

import numpy as np

from experiments.medchange import encoders as E
from tests.fixtures import HashingEncoder, LexicalOverlapReranker


class DoubleTests(unittest.TestCase):
    """The stand-ins of ``tests/fixtures.py`` (they are not part of the pipeline package)."""

    def test_the_hashing_encoder_is_deterministic_and_has_one_row_per_text(self):
        enc = HashingEncoder(dim=8)
        a, b = enc.encode(["one", "two"]), enc.encode(["one", "two"])
        self.assertEqual(a.shape, (2, 8))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a[0], a[1]))
        self.assertEqual(enc.encode([]).shape, (0, 8))

    def test_the_lexical_reranker_scores_the_share_of_query_words_a_passage_contains(self):
        scores = LexicalOverlapReranker().score("cranberry juice infection", ["cranberry juice", "nothing here", "cranberry juice infection"])
        np.testing.assert_allclose(scores, [2 / 3, 0.0, 1.0], rtol=1e-6)
        self.assertEqual(LexicalOverlapReranker().score("q", []).shape, (0,))


class MedCPTTests(unittest.TestCase):
    def test_the_package_holds_only_the_two_abstract_bases_and_the_medcpt_classes(self):
        """The stand-ins are in ``tests/fixtures.py`` (2026-10-13): a stand-in inside the package could be used by mistake in a run that produces numbers."""
        classes = sorted(n for n, o in vars(E).items() if isinstance(o, type) and o.__module__ == E.__name__)
        self.assertEqual(classes, ["CrossEncoderReranker", "Encoder", "MedCPTEncoder", "MedCPTReranker"])

    def test_the_published_checkpoints_and_input_limits(self):
        query, article = E.medcpt_query_encoder(), E.medcpt_article_encoder()
        self.assertEqual((query.name, query.max_tokens), ("ncbi/MedCPT-Query-Encoder", 64))
        self.assertEqual((article.name, article.max_tokens), ("ncbi/MedCPT-Article-Encoder", 512))
        self.assertEqual(E.MedCPTReranker().name, "ncbi/MedCPT-Cross-Encoder")

    def test_nothing_is_loaded_before_use_and_a_missing_torch_is_reported_by_name(self):
        encoder = E.medcpt_query_encoder(device="cpu")
        self.assertIsNone(encoder._model)                                          # construction loads no model
        with mock.patch.dict(sys.modules, {"torch": None}):                      # as if torch were not installed
            with self.assertRaises(RuntimeError) as caught:
                encoder._ensure()
        self.assertIn("ncbi/MedCPT-Query-Encoder needs torch and transformers installed", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
