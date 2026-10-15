"""Acceptance tests for docs/methodology.md sections 4 and 7.4 (balanced retrieval). They skip until src/balanced_retrieval exists."""


from tests.spec_targets import CORPUS_ORDER, SpecTestCase, resolve

NAN = float("nan")
SCORES = {"textbooks:0": 9.0, "pmc:2": 8.0, "pubmed:1": 7.0}  # relevance scores in the wiring tests; every other candidate scores 0


def candidate(corpus, doc_key, text=""):
    return {"corpus": corpus, "doc_key": doc_key, "text": text or (str(doc_key) + " text")}


def ranked(corpus, n, key_prefix=None):
    """n candidates of one corpus, best first."""
    prefix = key_prefix or corpus
    return [candidate(corpus, f"{prefix}:{i}") for i in range(n)]


def all_corpora(n):
    return {name: ranked(name, n) for name in CORPUS_ORDER}


def count_of(candidates, corpus):
    return sum(1 for item in candidates if item["corpus"] == corpus)


class PoolTests(SpecTestCase):
    """Paper p.5 and the authors' rerank.py: the same number from each corpus, concatenated, nothing removed."""

    requires = ("pool_candidates",)

    @staticmethod
    def pool(per_corpus, top_k):
        return resolve("pool_candidates")(per_corpus, top_k)

    def test_the_same_number_comes_from_every_corpus(self):
        pooled = self.pool(all_corpora(10), 4)
        self.assertEqual(len(pooled), 16)
        for name in CORPUS_ORDER:
            self.assertEqual(count_of(pooled, name), 4)

    def test_the_best_of_each_corpus_are_taken_in_their_own_order(self):
        source = all_corpora(10)
        pooled = self.pool(source, 3)
        for name in CORPUS_ORDER:
            self.assertEqual([item for item in pooled if item["corpus"] == name], source[name][:3])

    def test_the_order_is_pubmed_pmc_guidelines_textbooks(self):
        pooled = self.pool(all_corpora(10), 2)
        self.assertEqual([item["corpus"] for item in pooled], ["pubmed"] * 2 + ["pmc"] * 2 + ["guidelines"] * 2 + ["textbooks"] * 2)

    def test_the_order_does_not_depend_on_how_the_mapping_was_built(self):
        source = all_corpora(6)
        backwards = {name: source[name] for name in reversed(CORPUS_ORDER)}
        self.assertEqual(self.pool(backwards, 3), self.pool(source, 3))

    def test_no_duplicate_is_removed_across_corpora(self):
        source = {name: ranked(name, 3, key_prefix="shared") for name in CORPUS_ORDER}
        pooled = self.pool(source, 3)
        self.assertEqual(len(pooled), 12)
        self.assertEqual(len({item["doc_key"] for item in pooled}), 3)

    def test_repeats_inside_one_corpus_are_kept(self):
        item = candidate("pubmed", "pmid:1")
        self.assertEqual(self.pool({"pubmed": [item, item, item]}, 3), [item, item, item])

    def test_a_short_corpus_is_not_topped_up_from_the_others(self):
        source = all_corpora(10)
        source["guidelines"] = ranked("guidelines", 2)
        pooled = self.pool(source, 5)
        self.assertEqual({name: count_of(pooled, name) for name in CORPUS_ORDER}, {"pubmed": 5, "pmc": 5, "guidelines": 2, "textbooks": 5})

    def test_an_empty_corpus_contributes_nothing(self):
        source = all_corpora(4)
        source["pmc"] = []
        pooled = self.pool(source, 2)
        self.assertEqual([item["corpus"] for item in pooled], ["pubmed"] * 2 + ["guidelines"] * 2 + ["textbooks"] * 2)

    def test_asking_for_more_than_exists_returns_everything(self):
        self.assertEqual(len(self.pool(all_corpora(3), 50)), 12)

    def test_a_subset_of_the_corpora_is_allowed_for_ablations(self):
        source = all_corpora(4)
        pooled = self.pool({"textbooks": source["textbooks"], "pubmed": source["pubmed"]}, 2)
        self.assertEqual(pooled, source["pubmed"][:2] + source["textbooks"][:2])

    def test_the_input_is_not_changed_and_the_same_objects_come_back(self):
        source = all_corpora(4)
        before = {name: list(items) for name, items in source.items()}
        pooled = self.pool(source, 2)
        self.assertEqual(source, before)
        self.assertIs(pooled[0], source["pubmed"][0])
        self.assertIsNot(pooled, source["pubmed"])

    def test_bad_arguments_are_errors(self):
        source = all_corpora(4)
        for bad in (0, -1):
            with self.subTest(top_k=bad), self.assertRaises(ValueError):
                self.pool(source, bad)
        with self.assertRaises((TypeError, ValueError)):
            self.pool(source, 2.5)
        with self.assertRaises(ValueError):
            self.pool({}, 2)
        with self.assertRaises(ValueError):
            self.pool({"web": ranked("web", 2)}, 2)
        with self.assertRaises(ValueError):
            self.pool(dict(source, web=ranked("web", 2)), 2)


class RerankTests(SpecTestCase):
    """The cross-encoder orders the pool by score, highest first; equal scores keep the pool order; the best k stay."""

    requires = ("rerank",)

    @staticmethod
    def rerank(candidates, scores, top_k):
        return resolve("rerank")(candidates, scores, top_k)

    def test_the_highest_scores_come_first(self):
        self.assertEqual(self.rerank(list("abcd"), [0.1, 0.9, 0.5, 0.7], 4), list("bdca"))

    def test_only_the_best_top_k_are_kept(self):
        self.assertEqual(self.rerank(list("abcd"), [0.1, 0.9, 0.5, 0.7], 2), list("bd"))

    def test_equal_scores_keep_the_pool_order(self):
        scores = [0.5, 0.9, 0.5, 0.9, 0.5]
        self.assertEqual(self.rerank(list("abcde"), scores, 5), list("bdace"))
        self.assertEqual(self.rerank(list("abcde"), scores, 3), list("bda"))
        self.assertEqual(self.rerank(list("abc"), [1.0, 1.0, 1.0], 3), list("abc"))

    def test_negative_scores_are_ordered_like_any_other(self):
        self.assertEqual(self.rerank(list("abc"), [-3.2, -1.1, -2.5], 3), list("bca"))

    def test_asking_for_more_than_there_is_returns_everything_in_order_of_score(self):
        self.assertEqual(self.rerank(list("ab"), [1.0, 2.0], 10), list("ba"))

    def test_nothing_in_gives_nothing_out(self):
        self.assertEqual(self.rerank([], [], 3), [])

    def test_no_quota_is_applied_after_ranking(self):
        pooled = [candidate(name, f"{name}:{i}") for name in CORPUS_ORDER for i in range(3)]
        scores = [0.0] * 9 + [1.0, 2.0, 3.0]
        best = self.rerank(pooled, scores, 3)
        self.assertEqual({item["corpus"] for item in best}, {"textbooks"})

    def test_the_input_is_not_changed_and_the_same_objects_come_back(self):
        items = [candidate("pubmed", "a"), candidate("pmc", "b"), candidate("pubmed", "c")]
        scores = [0.2, 0.8, 0.5]
        best = self.rerank(items, scores, 2)
        self.assertIs(best[0], items[1])
        self.assertIs(best[1], items[2])
        self.assertEqual([item["doc_key"] for item in items], ["a", "b", "c"])
        self.assertEqual(scores, [0.2, 0.8, 0.5])

    def test_bad_arguments_are_errors(self):
        with self.assertRaises(ValueError):
            self.rerank(list("ab"), [1.0], 1)
        with self.assertRaises(ValueError):
            self.rerank(list("ab"), [1.0, NAN], 1)
        for bad in (0, -1):
            with self.subTest(top_k=bad), self.assertRaises(ValueError):
                self.rerank(list("ab"), [1.0, 2.0], bad)


class Spy:
    """Stands in for the search of one corpus: records its calls and returns a fixed ranked list."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def __call__(self, query, k):
        self.calls.append((query, k))
        return list(self.results)


class BalancedRetrieveTests(SpecTestCase):
    """The wiring of Part 1 and Part 2: rationale alone for the search, one k for every corpus, one pooled re-ranking."""

    requires = ("balanced_retrieve",)

    QUESTION = "QUESTIONTEXT which drug is indicated? A) one B) two C) three D) four"
    RATIONALE = "  RATIONALETEXT step by step reasoning that ends with option B.  "

    def setUp(self):
        super().setUp()
        self.source = all_corpora(6)
        self.spies = {name: Spy(self.source[name]) for name in CORPUS_ORDER}
        self.scorer_calls = []

    def scorer(self, query, candidates):
        self.scorer_calls.append((query, list(candidates)))
        return [SCORES.get(item["doc_key"], 0.0) for item in candidates]

    def run_retrieve(self, **overrides):
        arguments = {"question": self.QUESTION, "rationale": self.RATIONALE, "searchers": self.spies, "scorer": self.scorer, "top_k": 3}
        arguments.update(overrides)
        return resolve("balanced_retrieve")(**arguments)

    def expected_pool(self, k):
        return [item for name in CORPUS_ORDER for item in self.source[name][:k]]

    def test_every_corpus_is_searched_once_with_the_rationale_alone_and_the_same_k(self):
        self.run_retrieve(top_k=5)
        for name, spy in self.spies.items():
            with self.subTest(corpus=name):
                self.assertEqual(spy.calls, [(self.RATIONALE.strip(), 5)])
                self.assertNotIn("QUESTIONTEXT", spy.calls[0][0])

    def test_the_pool_is_scored_once_against_the_question_by_default(self):
        self.run_retrieve(top_k=3)
        self.assertEqual(len(self.scorer_calls), 1)
        query, pooled = self.scorer_calls[0]
        self.assertEqual(query, self.QUESTION)
        self.assertEqual(pooled, self.expected_pool(3))

    def test_the_rationale_variant_is_scored_against_the_rationale(self):
        self.run_retrieve(top_k=3, rerank_against="rationale")
        self.assertEqual(self.scorer_calls[0][0], self.RATIONALE.strip())

    def test_hits_beyond_k_are_cut_before_pooling(self):
        self.run_retrieve(top_k=2)
        self.assertEqual(self.scorer_calls[0][1], self.expected_pool(2))

    def test_the_best_candidates_across_all_corpora_come_back_in_score_order(self):
        best = self.run_retrieve(top_k=3)
        self.assertEqual([item["doc_key"] for item in best], ["textbooks:0", "pmc:2", "pubmed:1"])

    def test_equal_scores_keep_the_pool_order(self):
        best = self.run_retrieve(top_k=5)
        self.assertEqual([item["doc_key"] for item in best], ["textbooks:0", "pmc:2", "pubmed:1", "pubmed:0", "pubmed:2"])

    def test_the_result_has_k_items_when_the_pool_is_larger(self):
        self.assertEqual(len(self.run_retrieve(top_k=4)), 4)

    def test_a_smaller_pool_returns_what_there_is(self):
        self.assertEqual(len(self.run_retrieve(top_k=50)), 24)

    def test_the_very_same_candidates_come_back(self):
        best = self.run_retrieve(top_k=3)
        self.assertIs(best[0], self.source["textbooks"][0])

    def test_a_subset_of_the_searchers_is_allowed_for_ablations(self):
        best = self.run_retrieve(top_k=3, searchers={"pubmed": self.spies["pubmed"], "pmc": self.spies["pmc"]})
        self.assertEqual({item["corpus"] for item in best}, {"pubmed", "pmc"})
        self.assertEqual(self.spies["guidelines"].calls, [])

    def test_an_empty_rationale_is_an_error_before_any_search(self):
        for bad in ("", "   "):
            with self.subTest(rationale=bad), self.assertRaises(ValueError):
                self.run_retrieve(rationale=bad)
        self.assertEqual([spy.calls for spy in self.spies.values()], [[], [], [], []])
        self.assertEqual(self.scorer_calls, [])

    def test_a_bad_k_is_an_error_before_any_search(self):
        for bad in (0, -3):
            with self.subTest(top_k=bad), self.assertRaises(ValueError):
                self.run_retrieve(top_k=bad)
        self.assertEqual([spy.calls for spy in self.spies.values()], [[], [], [], []])

    def test_an_empty_pool_gives_an_empty_list_without_scoring(self):
        empty = {name: Spy([]) for name in CORPUS_ORDER}
        self.assertEqual(self.run_retrieve(searchers=empty), [])
        self.assertEqual(self.scorer_calls, [])

    def test_a_failing_search_is_not_swallowed(self):
        def broken(query, k):
            raise RuntimeError("index unavailable")

        with self.assertRaises(RuntimeError):
            self.run_retrieve(searchers=dict(self.spies, pmc=broken))
        self.assertEqual(self.scorer_calls, [])

    def test_a_scorer_with_the_wrong_number_of_scores_is_an_error(self):
        with self.assertRaises(ValueError):
            self.run_retrieve(scorer=lambda query, candidates: [1.0])

    def test_an_unknown_corpus_name_is_an_error(self):
        with self.assertRaises(ValueError):
            self.run_retrieve(searchers=dict(self.spies, web=Spy(ranked("web", 3))))

    def test_an_unknown_rerank_variant_is_an_error(self):
        with self.assertRaises(ValueError):
            self.run_retrieve(rerank_against="both")


class DiagnosticsTests(SpecTestCase):
    """Docs section 7.4: report each corpus's share of the final list and the cross-corpus duplicate rate."""

    requires = ("corpus_share", "duplicate_rate")

    @staticmethod
    def share(final):
        return resolve("corpus_share")(final)

    @staticmethod
    def duplicates(final):
        return resolve("duplicate_rate")(final)

    def test_corpus_share_counts_each_corpus_over_the_list(self):
        final = ranked("pubmed", 2) + ranked("pmc", 1) + ranked("textbooks", 1)
        self.assertEqual(self.share(final), {"pubmed": 0.5, "pmc": 0.25, "guidelines": 0.0, "textbooks": 0.25})

    def test_corpus_share_of_a_single_corpus(self):
        self.assertEqual(self.share(ranked("guidelines", 3)), {"pubmed": 0.0, "pmc": 0.0, "guidelines": 1.0, "textbooks": 0.0})

    def test_corpus_shares_sum_to_one(self):
        final = ranked("pubmed", 3) + ranked("pmc", 2) + ranked("guidelines", 2)
        self.assertAlmostEqual(sum(self.share(final).values()), 1.0)

    def test_corpus_share_rejects_an_unknown_corpus_and_an_empty_list(self):
        with self.assertRaises(ValueError):
            self.share([candidate("pubmd", "x:1")])
        with self.assertRaises(ValueError):
            self.share([])

    def test_distinct_documents_give_zero(self):
        final = [candidate("pubmed", "pmid:1"), candidate("pmc", "pmid:2"), candidate("guidelines", "doi:3"), candidate("textbooks", "isbn:4")]
        self.assertEqual(self.duplicates(final), 0.0)

    def test_a_document_in_two_corpora_counts_both_snippets(self):
        final = [candidate("pubmed", "pmid:1"), candidate("pmc", "pmid:1"), candidate("guidelines", "doi:3"), candidate("textbooks", "isbn:4")]
        self.assertEqual(self.duplicates(final), 0.5)

    def test_a_document_in_three_corpora(self):
        final = [candidate("pubmed", "k"), candidate("pmc", "k"), candidate("guidelines", "k"), candidate("textbooks", "other")]
        self.assertEqual(self.duplicates(final), 0.75)

    def test_chunks_of_a_shared_document_each_count(self):
        final = [candidate("pubmed", "k", "chunk 1"), candidate("pubmed", "k", "chunk 2"), candidate("pmc", "k")]
        self.assertEqual(self.duplicates(final), 1.0)

    def test_the_same_document_twice_in_one_corpus_is_not_a_cross_corpus_duplicate(self):
        final = [candidate("pubmed", "k", "chunk 1"), candidate("pubmed", "k", "chunk 2"), candidate("pmc", "other")]
        self.assertEqual(self.duplicates(final), 0.0)

    def test_an_unknown_identity_never_matches(self):
        for unknown in (None, ""):
            with self.subTest(doc_key=unknown):
                final = [candidate("pubmed", unknown, "a"), candidate("pmc", unknown, "b")]
                self.assertEqual(self.duplicates(final), 0.0)

    def test_an_empty_list_is_an_error(self):
        with self.assertRaises(ValueError):
            self.duplicates([])
