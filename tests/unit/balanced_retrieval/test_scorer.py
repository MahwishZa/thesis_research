"""Tests of src/balanced_retrieval/scorer.py with a stand-in tokenizer and cross-encoder. They need numpy and skip without it."""

import unittest

from balanced_retrieval.scorer import CROSS_ENCODER_NAME, MAX_LENGTH, CrossEncoderScorer
from tests.support import FakeClassifier, FakePairTokenizer, RecordingNoGrad, np

CANDIDATES = [{"corpus": "pmc", "doc_key": str(i), "text": text} for i, text in enumerate(["a", "b c", "d e f", "g", "h i"])]


@unittest.skipIf(np is None, "numpy is not installed")
class CrossEncoderScorerTests(unittest.TestCase):
    def make(self, **options):
        self.tokenizer, self.model = FakePairTokenizer(), FakeClassifier()
        return CrossEncoderScorer(self.tokenizer, self.model, **options)

    def test_one_score_per_candidate_in_the_order_given(self):
        scores = self.make()("q w", CANDIDATES)
        self.assertEqual(len(scores), 5)
        self.assertTrue(all(isinstance(score, float) for score in scores))
        # the score of the stand-in grows with the number of passage words, so longer passages score higher
        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[1], scores[2])
        self.assertGreater(scores[2], scores[3])

    def test_the_tokenizer_gets_query_passage_pairs_as_in_the_authors_code(self):
        self.make(batch_size=100)("q w", CANDIDATES[:2])
        self.assertEqual(
            self.tokenizer.calls,
            [{"pairs": [["q w", "a"], ["q w", "b c"]], "truncation": True, "padding": True, "return_tensors": "pt", "max_length": 512}],
        )
        self.assertEqual(MAX_LENGTH, 512)

    def test_the_batch_size_does_not_change_the_scores(self):
        reference = self.make(batch_size=100)("q w", CANDIDATES)
        for batch_size in (1, 2, 3):
            with self.subTest(batch_size=batch_size):
                self.assertEqual(self.make(batch_size=batch_size)("q w", CANDIDATES), reference)

    def test_the_pool_is_cut_into_batches(self):
        self.make(batch_size=2)("q", CANDIDATES)
        self.assertEqual([len(call["pairs"]) for call in self.tokenizer.calls], [2, 2, 1])
        self.assertEqual(self.model.calls, 3)

    def test_no_candidates_give_no_scores_and_no_model_call(self):
        self.assertEqual(self.make()("q", []), [])
        self.assertEqual(self.model.calls, 0)

    def test_an_empty_query_is_an_error(self):
        for bad in ("", "  ", None):
            with self.subTest(query=bad), self.assertRaises(ValueError):
                self.make()(bad, CANDIDATES)

    def test_gradients_are_off_for_every_batch(self):
        guard = RecordingNoGrad()
        self.make(batch_size=2, no_grad=guard)("q", CANDIDATES)
        self.assertEqual((guard.entered, guard.left), (3, 3))

    def test_tensors_go_to_the_device(self):
        scorer = self.make(device="cuda")
        scorer("q", CANDIDATES[:1])
        self.assertEqual(scorer.info["device"], "cuda")

    def test_the_text_is_cut_at_the_maximum_length(self):
        long = [{"corpus": "pmc", "doc_key": "x", "text": "w " * 600}]
        self.make()("q", long)
        self.assertEqual(self.tokenizer.calls[0]["max_length"], 512)

    def test_info_names_the_model_and_the_settings(self):
        info = self.make(batch_size=4, revision="abc").info
        self.assertEqual(info, {"cross_encoder": CROSS_ENCODER_NAME, "revision": "abc", "max_length": 512, "device": "cpu", "batch_size": 4})
        self.assertEqual(CROSS_ENCODER_NAME, "ncbi/MedCPT-Cross-Encoder")

    def test_bad_settings_are_errors(self):
        for bad in (0, -1, 1.5):
            with self.subTest(batch_size=bad), self.assertRaises(ValueError):
                self.make(batch_size=bad)
        with self.assertRaises(ValueError):
            self.make(max_length=2)


if __name__ == "__main__":
    unittest.main()
