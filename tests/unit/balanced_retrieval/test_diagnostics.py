"""Tests of src/balanced_retrieval/diagnostics.py beyond the acceptance tests: the summary over many questions."""

import unittest

from balanced_retrieval.diagnostics import corpus_share, duplicate_rate, summarise_lists


def item(corpus, key):
    return {"corpus": corpus, "doc_key": key, "text": "t"}


class SummaryTests(unittest.TestCase):
    def test_the_means_are_taken_over_the_questions_not_the_passages(self):
        first = [item("pubmed", "a"), item("pubmed", "b")]  # all pubmed
        second = [item("textbooks", "c"), item("pmc", "c")]  # half and half, and a duplicate
        summary = summarise_lists([first, second])
        self.assertEqual(summary["questions"], 2)
        self.assertEqual(summary["empty_lists"], 0)
        self.assertEqual(summary["mean_corpus_share"], {"pubmed": 0.5, "pmc": 0.25, "guidelines": 0.0, "textbooks": 0.25})
        self.assertEqual(summary["mean_duplicate_rate"], 0.5)
        self.assertEqual(summary["mean_list_length"], 2.0)

    def test_empty_lists_are_counted_and_left_out_of_the_means(self):
        summary = summarise_lists([[item("pubmed", "a")], []])
        self.assertEqual((summary["questions"], summary["empty_lists"]), (2, 1))
        self.assertEqual(summary["mean_corpus_share"]["pubmed"], 1.0)

    def test_only_empty_lists_give_only_the_counts(self):
        self.assertEqual(summarise_lists([[], []]), {"questions": 2, "empty_lists": 2})
        self.assertEqual(summarise_lists([]), {"questions": 0, "empty_lists": 0})

    def test_passages_without_a_document_key_are_reported(self):
        summary = summarise_lists([[item("pubmed", ""), item("pmc", None), item("guidelines", "k"), item("textbooks", "j")]])
        self.assertEqual(summary["share_without_document_key"], 0.5)


class ArgumentTests(unittest.TestCase):
    def test_a_candidate_without_a_corpus_is_an_error(self):
        with self.assertRaises(ValueError):
            corpus_share([{"doc_key": "a", "text": "t"}])
        with self.assertRaises(ValueError):
            duplicate_rate([{"doc_key": "a", "text": "t"}])

    def test_integer_document_keys_match_like_text(self):
        self.assertEqual(duplicate_rate([item("pubmed", 7), item("pmc", 7)]), 1.0)

    def test_a_tuple_is_accepted(self):
        self.assertEqual(corpus_share((item("pubmed", "a"),))["pubmed"], 1.0)


if __name__ == "__main__":
    unittest.main()
