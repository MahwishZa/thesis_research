"""Tests of src/balanced_retrieval/search.py: exact inner-product search in blocks. They need numpy and skip without it."""

import unittest

from tests.support import TempDirTestCase, np

try:
    from balanced_retrieval.search import ExactIndex
except ImportError:  # numpy is not installed
    ExactIndex = None


@unittest.skipIf(ExactIndex is None, "numpy is not installed")
class ExactSearchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # whole numbers, so that every inner product is exact in 32 bits whatever the order of the additions
        self.vectors = rng.integers(-5, 6, size=(1000, 8)).astype(np.float32)
        self.queries = rng.integers(-5, 6, size=(6, 8)).astype(np.float32)
        self.full = self.vectors @ self.queries.T  # (rows, queries)

    def check(self, indices, scores, k):
        self.assertEqual(indices.shape, (len(self.queries), k))
        for q in range(len(self.queries)):
            best = np.sort(self.full[:, q])[::-1][:k]
            np.testing.assert_array_equal(scores[q], best)  # the k best scores, best first
            np.testing.assert_array_equal(self.full[indices[q], q], scores[q])  # every row has the score it is given
            self.assertEqual(len(set(indices[q])), k)  # no row twice

    def test_the_k_best_inner_products_are_found_whatever_the_block_sizes(self):
        for block_rows in (1, 7, 64, 1000, 5000):
            for query_chunk in (1, 4, 512):
                with self.subTest(block_rows=block_rows, query_chunk=query_chunk):
                    indices, scores = ExactIndex(self.vectors, block_rows=block_rows, query_chunk=query_chunk).search(self.queries, 9)
                    self.check(indices, scores, 9)

    def test_a_larger_k_than_rows_returns_every_row_best_first(self):
        small = self.vectors[:5]
        indices, scores = ExactIndex(small, block_rows=2).search(self.queries, 50)
        self.assertEqual((indices.shape, scores.shape), ((6, 5), (6, 5)))
        for q in range(6):
            expected = np.lexsort((np.arange(5), -(small @ self.queries[q])))
            np.testing.assert_array_equal(indices[q], expected)

    def test_equal_scores_are_ordered_by_the_lower_row_number(self):
        vectors = np.zeros((9, 2), dtype=np.float32)
        vectors[[7, 2, 5]] = [1.0, 0.0]  # three equal best rows
        indices, scores = ExactIndex(vectors, block_rows=4).search(np.array([[1.0, 0.0]], dtype=np.float32), 5)
        self.assertEqual(list(indices[0][:3]), [2, 5, 7])
        self.assertEqual(list(scores[0]), [1.0, 1.0, 1.0, 0.0, 0.0])

    def test_the_same_input_gives_the_same_output(self):
        index = ExactIndex(self.vectors, block_rows=50)
        first = index.search(self.queries, 7)
        second = index.search(self.queries, 7)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_scores_are_32_bit_and_rows_are_64_bit_integers(self):
        indices, scores = ExactIndex(self.vectors).search(self.queries, 3)
        self.assertEqual((indices.dtype, scores.dtype), (np.int64, np.float32))

    def test_half_precision_vectors_are_read_as_32_bit(self):
        half = self.vectors.astype(np.float16)
        indices, scores = ExactIndex(half, block_rows=100).search(self.queries, 4)
        self.check(indices, scores, 4)  # whole numbers are exact in 16 bits too

    def test_an_empty_corpus_gives_empty_results(self):
        indices, scores = ExactIndex(np.zeros((0, 8), dtype=np.float32)).search(self.queries, 3)
        self.assertEqual((indices.shape, scores.shape), ((6, 0), (6, 0)))

    def test_queries_of_the_wrong_shape_or_with_bad_numbers_are_errors(self):
        index = ExactIndex(self.vectors)
        with self.assertRaises(ValueError):
            index.search(np.zeros((2, 3), dtype=np.float32), 2)
        with self.assertRaises(ValueError):
            index.search(np.zeros(8, dtype=np.float32), 2)
        bad = self.queries.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            index.search(bad, 2)
        bad[0, 0] = np.inf
        with self.assertRaises(ValueError):
            index.search(bad, 2)

    def test_a_bad_k_is_an_error(self):
        index = ExactIndex(self.vectors)
        for bad in (0, -1):
            with self.subTest(k=bad), self.assertRaises(ValueError):
                index.search(self.queries, bad)
        for bad in (2.0, True, "3"):
            with self.subTest(k=bad), self.assertRaises(TypeError):
                index.search(self.queries, bad)

    def test_the_index_refuses_vectors_that_are_not_a_matrix_and_bad_sizes(self):
        with self.assertRaises(ValueError):
            ExactIndex(np.zeros(5, dtype=np.float32))
        for bad in (0, -1, True):
            with self.subTest(block_rows=bad), self.assertRaises(ValueError):
                ExactIndex(self.vectors, block_rows=bad)
            with self.subTest(query_chunk=bad), self.assertRaises(ValueError):
                ExactIndex(self.vectors, query_chunk=bad)

    def test_rows_and_dimension_are_reported(self):
        index = ExactIndex(self.vectors)
        self.assertEqual((index.rows, index.dim), (1000, 8))


@unittest.skipIf(ExactIndex is None, "numpy is not installed")
class MemoryMapTests(TempDirTestCase):
    def test_a_memory_mapped_file_gives_the_same_result_as_the_array(self):
        rng = np.random.default_rng(1)
        vectors = rng.integers(-5, 6, size=(300, 5)).astype(np.float32)
        queries = rng.integers(-5, 6, size=(3, 5)).astype(np.float32)
        np.save(self.dir / "v.npy", vectors)
        mapped = np.load(self.dir / "v.npy", mmap_mode="r")
        from_file = ExactIndex(mapped, block_rows=64).search(queries, 6)
        from_memory = ExactIndex(vectors, block_rows=64).search(queries, 6)
        np.testing.assert_array_equal(from_file[0], from_memory[0])
        np.testing.assert_array_equal(from_file[1], from_memory[1])
        del mapped


if __name__ == "__main__":
    unittest.main()
