"""Tests of src/balanced_retrieval/pool.py beyond the acceptance tests of tests/spec/test_retrieval.py: argument handling."""

import unittest

from balanced_retrieval.pool import CORPUS_ORDER, check_top_k, pool_candidates, rerank
from tests.spec.targets import CORPUS_ORDER as SPECIFIED_ORDER


def item(corpus, key):
    return {"corpus": corpus, "doc_key": key, "text": key}


class CorpusOrderTests(unittest.TestCase):
    def test_the_order_is_the_one_the_specification_names(self):
        self.assertEqual(CORPUS_ORDER, SPECIFIED_ORDER)
        self.assertEqual(CORPUS_ORDER, ("pubmed", "pmc", "guidelines", "textbooks"))


class CheckTopKTests(unittest.TestCase):
    def test_a_whole_number_of_at_least_one_is_returned(self):
        self.assertEqual(check_top_k(1), 1)
        self.assertEqual(check_top_k(32), 32)

    def test_a_bool_a_float_and_text_are_type_errors(self):
        for bad in (True, 2.0, "3", None):
            with self.subTest(top_k=bad), self.assertRaises(TypeError):
                check_top_k(bad)

    def test_zero_and_negative_numbers_are_value_errors(self):
        for bad in (0, -5):
            with self.subTest(top_k=bad), self.assertRaises(ValueError):
                check_top_k(bad)


class PoolTests(unittest.TestCase):
    def test_tuples_and_other_sequences_are_accepted(self):
        source = {"pubmed": (item("pubmed", "a"), item("pubmed", "b")), "textbooks": [item("textbooks", "c")]}
        self.assertEqual([x["doc_key"] for x in pool_candidates(source, 1)], ["a", "c"])

    def test_a_bool_top_k_is_refused(self):
        with self.assertRaises(TypeError):
            pool_candidates({"pubmed": [item("pubmed", "a")]}, True)

    def test_an_unknown_name_is_refused_even_when_its_list_is_empty(self):
        with self.assertRaises(ValueError):
            pool_candidates({"pubmed": [item("pubmed", "a")], "web": []}, 1)


class RerankTests(unittest.TestCase):
    def test_infinite_scores_are_refused_like_nan(self):
        for bad in (float("inf"), float("-inf")):
            with self.subTest(score=bad), self.assertRaises(ValueError):
                rerank(["a", "b"], [1.0, bad], 1)

    def test_tuples_are_accepted_and_a_list_comes_back(self):
        self.assertEqual(rerank(("a", "b", "c"), (0.1, 0.3, 0.2), 2), ["b", "c"])

    def test_a_bool_top_k_is_refused(self):
        with self.assertRaises(TypeError):
            rerank(["a"], [1.0], True)

    def test_integer_scores_work(self):
        self.assertEqual(rerank(["a", "b", "c"], [1, 3, 2], 3), ["b", "c", "a"])


if __name__ == "__main__":
    unittest.main()
