"""Tests of src/balanced_retrieval/retrieve.py beyond the acceptance tests: call order, labels, failures."""

import unittest

from balanced_retrieval.pool import CORPUS_ORDER
from balanced_retrieval.retrieve import RERANK_AGAINST, balanced_retrieve

QUESTION = "Which drug is used? A) one B) two C) three D) four"
RATIONALE = "The patient needs a cholinesterase inhibitor, so the answer is A."


def hits(corpus, n):
    return [{"corpus": corpus, "doc_key": f"{corpus}:{i}", "text": f"{corpus} text {i}"} for i in range(n)]


class Recorder:
    def __init__(self):
        self.events = []

    def searcher(self, corpus, results):
        def search(query, k):
            self.events.append(("search", corpus, query, k))
            return results

        return search

    def scorer(self, query, candidates):
        self.events.append(("score", query, len(candidates)))
        return [float(i) for i in range(len(candidates))]


class OrderOfCallsTests(unittest.TestCase):
    def test_the_corpora_are_searched_in_the_fixed_order_whatever_the_mapping_order(self):
        recorder = Recorder()
        searchers = {name: recorder.searcher(name, hits(name, 3)) for name in reversed(CORPUS_ORDER)}
        balanced_retrieve(question=QUESTION, rationale=RATIONALE, searchers=searchers, scorer=recorder.scorer, top_k=2)
        self.assertEqual([event[1] for event in recorder.events if event[0] == "search"], list(CORPUS_ORDER))

    def test_every_search_comes_before_the_one_scoring(self):
        recorder = Recorder()
        searchers = {name: recorder.searcher(name, hits(name, 3)) for name in CORPUS_ORDER}
        balanced_retrieve(question=QUESTION, rationale=RATIONALE, searchers=searchers, scorer=recorder.scorer, top_k=2)
        self.assertEqual([event[0] for event in recorder.events], ["search"] * 4 + ["score"])
        self.assertEqual(recorder.events[-1], ("score", QUESTION, 8))

    def test_the_search_text_is_the_rationale_without_the_question(self):
        recorder = Recorder()
        searchers = {"pubmed": recorder.searcher("pubmed", hits("pubmed", 2))}
        balanced_retrieve(question=QUESTION, rationale="  " + RATIONALE + "\n", searchers=searchers, scorer=recorder.scorer, top_k=2)
        self.assertEqual(recorder.events[0], ("search", "pubmed", RATIONALE, 2))


class LabelTests(unittest.TestCase):
    def test_a_candidate_from_the_wrong_corpus_is_an_error_before_scoring(self):
        recorder = Recorder()
        searchers = {"pubmed": recorder.searcher("pubmed", hits("pmc", 2))}
        with self.assertRaises(ValueError):
            balanced_retrieve(question=QUESTION, rationale=RATIONALE, searchers=searchers, scorer=recorder.scorer, top_k=2)
        self.assertNotIn("score", [event[0] for event in recorder.events])

    def test_a_candidate_without_a_corpus_label_is_an_error(self):
        searchers = {"pubmed": lambda query, k: [{"doc_key": "x", "text": "t"}]}
        with self.assertRaises(ValueError):
            balanced_retrieve(question=QUESTION, rationale=RATIONALE, searchers=searchers, scorer=lambda q, c: [1.0], top_k=1)


class VariantTests(unittest.TestCase):
    def test_the_variants_are_the_two_the_specification_names(self):
        self.assertEqual(RERANK_AGAINST, ("question", "rationale"))

    def test_an_empty_question_is_an_error_before_any_search(self):
        recorder = Recorder()
        searchers = {"pubmed": recorder.searcher("pubmed", hits("pubmed", 2))}
        with self.assertRaises(ValueError):
            balanced_retrieve(question="  ", rationale=RATIONALE, searchers=searchers, scorer=recorder.scorer, top_k=2)
        self.assertEqual(recorder.events, [])

    def test_no_searchers_is_an_error(self):
        with self.assertRaises(ValueError):
            balanced_retrieve(question=QUESTION, rationale=RATIONALE, searchers={}, scorer=lambda q, c: [], top_k=2)

    def test_a_scorer_that_fails_is_not_swallowed(self):
        def broken(query, candidates):
            raise RuntimeError("model unavailable")

        searchers = {"pubmed": lambda query, k: hits("pubmed", 2)}
        with self.assertRaises(RuntimeError):
            balanced_retrieve(question=QUESTION, rationale=RATIONALE, searchers=searchers, scorer=broken, top_k=2)

    def test_a_scorer_may_return_any_sequence_of_numbers(self):
        searchers = {"pubmed": lambda query, k: hits("pubmed", 3)}
        best = balanced_retrieve(question=QUESTION, rationale=RATIONALE, searchers=searchers, scorer=lambda q, c: (1, 3, 2), top_k=3)
        self.assertEqual([x["doc_key"] for x in best], ["pubmed:1", "pubmed:2", "pubmed:0"])


if __name__ == "__main__":
    unittest.main()
