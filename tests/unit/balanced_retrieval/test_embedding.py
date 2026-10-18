"""Tests of src/balanced_retrieval/embedding.py with a stand-in tokenizer and model. They need numpy and skip without it."""

import gc
import json
import unittest

from tests.support import FakeModel, FakeTokenizer, TempDirTestCase, np

try:
    from balanced_retrieval.corpus import (
        EMBEDDINGS_FILE,
        EMBEDDINGS_STATE,
        Corpus,
        CorpusError,
        write_corpus,
    )
    from balanced_retrieval.embedding import (
        ARTICLE_ENCODER_NAME,
        ArticleEncoder,
        compare_precision,
        embed_corpus,
    )
except ImportError:  # numpy is not installed
    ArticleEncoder = None

TEXTS = [f"alpha {i} beta " + "gamma " * (i % 4) for i in range(11)]
DOCUMENTS = [(f"PMID:{i}", text) for i, text in enumerate(TEXTS)]


class CountingEncoder:
    """Wraps a real ArticleEncoder over the stand-ins, counts the texts it encodes, and can be told to fail."""

    def __init__(self, **options):
        self.inner = ArticleEncoder(FakeTokenizer(), FakeModel(), **options)
        self.info = self.inner.info
        self.encoded = 0
        self.fail_after = None

    def encode(self, texts):
        if self.fail_after is not None and self.encoded + len(texts) > self.fail_after:
            raise RuntimeError("out of memory")
        self.encoded += len(texts)
        return self.inner.encode(texts)


class NoisyEncoder:
    """Gives the vectors of another encoder plus a fixed small disturbance, as a lower precision would."""

    def __init__(self, inner, noise):
        self.inner, self.noise = inner, noise
        self.info = inner.info

    def encode(self, texts):
        vectors = self.inner.encode(texts)
        return vectors + self.noise * np.sin(np.arange(vectors.size, dtype=np.float32).reshape(vectors.shape))


@unittest.skipIf(ArticleEncoder is None, "numpy is not installed")
class ComparePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.full = CountingEncoder()

    def test_two_identical_encoders_agree_completely(self):
        result = compare_precision(TEXTS, self.full, CountingEncoder(), queries=5, top=3)
        self.assertAlmostEqual(result["cosine_mean"], 1.0)
        self.assertAlmostEqual(result["cosine_min"], 1.0)
        self.assertEqual((result["top3_overlap"], result["nearest_same"], result["texts"]), (1.0, 1.0, 11))

    def test_a_disturbed_encoder_lowers_the_cosine_and_may_change_the_neighbours(self):
        result = compare_precision(TEXTS, self.full, NoisyEncoder(CountingEncoder(), 2000.0), queries=11, top=3)
        self.assertLess(result["cosine_min"], 0.99)
        self.assertLessEqual(result["cosine_min"], result["cosine_mean"])
        self.assertLess(result["top3_overlap"], 1.0)

    def test_the_time_of_each_encoder_is_measured(self):
        ticks = iter([0.0, 2.0, 3.0])
        result = compare_precision(TEXTS, self.full, CountingEncoder(), queries=2, top=2, clock=lambda: next(ticks))
        self.assertEqual((result["seconds_full"], result["seconds_half"]), (2.0, 1.0))


@unittest.skipIf(ArticleEncoder is None, "numpy is not installed")
class ArticleEncoderTests(unittest.TestCase):
    def test_it_is_the_query_encoder_code_with_the_article_model_name(self):
        encoder = ArticleEncoder(FakeTokenizer(), FakeModel())
        self.assertEqual(encoder.name, ARTICLE_ENCODER_NAME)
        self.assertEqual(ARTICLE_ENCODER_NAME, "ncbi/MedCPT-Article-Encoder")
        self.assertEqual(encoder.info["encoder"], ARTICLE_ENCODER_NAME)
        self.assertEqual(encoder.info["max_length"], 512)
        self.assertFalse(encoder.info["normalised"])

    def test_the_vectors_are_the_first_token_vectors_as_for_queries(self):
        vectors = ArticleEncoder(FakeTokenizer(), FakeModel()).encode(["a b"])
        np.testing.assert_array_equal(vectors, [[218, 4, 108, 1, 107, 108]])

    def test_another_name_can_be_given(self):
        self.assertEqual(ArticleEncoder(FakeTokenizer(), FakeModel(), name="other").name, "other")


@unittest.skipIf(ArticleEncoder is None, "numpy is not installed")
class EmbedCorpusTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(gc.collect)
        write_corpus(self.dir, "textbooks", DOCUMENTS, words=100)  # one passage per text
        self.corpus = Corpus(self.dir, "textbooks")
        self.addCleanup(lambda: setattr(self, "corpus", None))
        self.expected = ArticleEncoder(FakeTokenizer(), FakeModel()).encode(TEXTS)

    def matrix(self):
        return np.array(Corpus(self.dir, "textbooks").embeddings())

    def test_row_i_is_the_vector_of_passage_i(self):
        state = embed_corpus(self.corpus, CountingEncoder(), batch_size=4, save_every=4)
        self.assertTrue(state["complete"])
        np.testing.assert_array_equal(self.matrix(), self.expected)

    def test_the_batch_size_does_not_change_the_vectors(self):
        embed_corpus(self.corpus, CountingEncoder(), batch_size=3, save_every=3)
        three = self.matrix()
        write_corpus(self.dir, "textbooks", DOCUMENTS, words=100, overwrite=True)
        embed_corpus(Corpus(self.dir, "textbooks"), CountingEncoder(), batch_size=11, save_every=11)
        np.testing.assert_array_equal(three, self.matrix())

    def test_the_state_records_the_settings_and_the_progress(self):
        embed_corpus(self.corpus, CountingEncoder(), batch_size=4, save_every=4)
        state = json.loads((self.dir / "textbooks" / EMBEDDINGS_STATE).read_text(encoding="utf-8"))
        self.assertEqual(state["rows"], 11)
        self.assertEqual(state["rows_done"], 11)
        self.assertTrue(state["complete"])
        self.assertEqual(state["settings"]["encoder"], ARTICLE_ENCODER_NAME)
        self.assertEqual(state["settings"]["dtype"], "float32")
        self.assertEqual(state["settings"]["max_length"], 512)

    def test_a_stopped_run_continues_where_it_stopped_and_gives_the_same_matrix(self):
        failing = CountingEncoder()
        failing.fail_after = 8
        with self.assertRaises(RuntimeError):
            embed_corpus(self.corpus, failing, batch_size=4, save_every=4)
        state = json.loads((self.dir / "textbooks" / EMBEDDINGS_STATE).read_text(encoding="utf-8"))
        self.assertEqual((state["rows_done"], state["complete"]), (8, False))
        again = CountingEncoder()
        embed_corpus(self.corpus, again, batch_size=4, save_every=4)
        self.assertEqual(again.encoded, 3)  # only the rows that were missing
        np.testing.assert_array_equal(self.matrix(), self.expected)

    def test_a_run_killed_before_the_first_save_can_be_resumed(self):
        failing = CountingEncoder()
        failing.fail_after = 4  # the first batch is written, the second fails, and save_every was not reached
        with self.assertRaises(RuntimeError):
            embed_corpus(self.corpus, failing, batch_size=4, save_every=100)
        again = CountingEncoder()
        embed_corpus(self.corpus, again, batch_size=4, save_every=100)
        np.testing.assert_array_equal(self.matrix(), self.expected)

    def test_a_finished_corpus_is_not_encoded_again(self):
        embed_corpus(self.corpus, CountingEncoder(), batch_size=4, save_every=4)
        again = CountingEncoder()
        state = embed_corpus(self.corpus, again, batch_size=4, save_every=4)
        self.assertEqual(again.encoded, 0)
        self.assertTrue(state["complete"])

    def test_other_encoder_settings_are_refused_instead_of_mixed(self):
        failing = CountingEncoder()
        failing.fail_after = 4
        with self.assertRaises(RuntimeError):
            embed_corpus(self.corpus, failing, batch_size=4, save_every=4)
        with self.assertRaisesRegex(CorpusError, "other settings"):
            embed_corpus(self.corpus, CountingEncoder(max_length=64), batch_size=4, save_every=4)
        with self.assertRaisesRegex(CorpusError, "other settings"):
            embed_corpus(self.corpus, CountingEncoder(), batch_size=4, save_every=4, dtype="float16")

    def test_half_precision_arithmetic_is_recorded_and_never_mixed_with_the_full_kind(self):
        failing = CountingEncoder(compute_dtype="float16")
        failing.fail_after = 4
        with self.assertRaises(RuntimeError):
            embed_corpus(self.corpus, failing, batch_size=4, save_every=4)
        self.assertEqual(self.corpus.embedding_state()["settings"]["compute_dtype"], "float16")
        with self.assertRaisesRegex(CorpusError, "compute_dtype"):
            embed_corpus(self.corpus, CountingEncoder(), batch_size=4, save_every=4)

    def test_the_usual_arithmetic_leaves_no_trace_in_the_state_so_earlier_embeddings_stay_valid(self):
        embed_corpus(self.corpus, CountingEncoder(), batch_size=4, save_every=4)
        self.assertNotIn("compute_dtype", self.corpus.embedding_state()["settings"])

    def test_half_precision_halves_the_file_and_keeps_the_values_to_half_precision(self):
        embed_corpus(self.corpus, CountingEncoder(), batch_size=4, save_every=4, dtype="float16")
        stored = Corpus(self.dir, "textbooks").embeddings()
        self.assertEqual(stored.dtype, np.float16)
        np.testing.assert_array_equal(np.array(stored), self.expected.astype(np.float16))
        del stored

    def test_a_matrix_without_its_state_is_not_overwritten(self):
        (self.dir / "textbooks" / EMBEDDINGS_FILE).write_bytes(b"not a matrix")
        with self.assertRaises(CorpusError):
            embed_corpus(self.corpus, CountingEncoder(), batch_size=4, save_every=4)
        self.assertEqual((self.dir / "textbooks" / EMBEDDINGS_FILE).read_bytes(), b"not a matrix")

    def test_a_state_for_another_number_of_rows_is_refused(self):
        failing = CountingEncoder()
        failing.fail_after = 4
        with self.assertRaises(RuntimeError):
            embed_corpus(self.corpus, failing, batch_size=4, save_every=4)
        path = self.dir / "textbooks" / EMBEDDINGS_STATE
        state = json.loads(path.read_text(encoding="utf-8"))
        state["rows"] = 99
        path.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaises(CorpusError):
            embed_corpus(self.corpus, CountingEncoder(), batch_size=4, save_every=4)

    def test_progress_is_reported_after_every_batch(self):
        seen = []
        embed_corpus(self.corpus, CountingEncoder(), batch_size=4, save_every=4, progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(4, 11), (8, 11), (11, 11)])

    def test_bad_arguments_are_errors(self):
        with self.assertRaises(ValueError):
            embed_corpus(self.corpus, CountingEncoder(), dtype="float64")
        with self.assertRaises(ValueError):
            embed_corpus(self.corpus, CountingEncoder(), batch_size=0)
        with self.assertRaises(ValueError):
            embed_corpus(self.corpus, CountingEncoder(), batch_size=8, save_every=4)


if __name__ == "__main__":
    unittest.main()
