"""Freezing one candidate pool per item: the abstracts of the as-of records, eligibility, ranking and the pool hash (fake encoders; no network)."""

import http.client
import unittest

from experiments.medchange.encoders import HashingEncoder, LexicalOverlapReranker
from experiments.medchange.eutils import EUtils
from experiments.medchange.freeze_candidates import fetch_abstracts, freeze_item, parse_efetch_xml, pool_hash


XML = """<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>11</PMID><Article>
<ArticleTitle>Cranberry <i>juice</i> trial</ArticleTitle>
<Abstract><AbstractText Label="BACKGROUND">Urinary infections are common.</AbstractText>
<AbstractText Label="RESULTS">Cranberry reduced <b>recurrence</b>.</AbstractText></Abstract>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>12</PMID><Article><ArticleTitle>No abstract</ArticleTitle></Article>
</MedlineCitation></PubmedArticle></PubmedArticleSet>"""


def _abs(pmid, title, n=60):
    return {"title": title, "abstract": ("cranberry juice urinary infection prevention trial " * n)[:400]}


class FreezeTests(unittest.TestCase):

    def test_efetch_parsing_keeps_labels_and_inline_markup(self):
        got = parse_efetch_xml(XML)
        self.assertEqual(got["11"]["title"], "Cranberry juice trial")
        self.assertEqual(got["11"]["abstract"],
                         "BACKGROUND: Urinary infections are common. RESULTS: Cranberry reduced recurrence.")
        self.assertEqual(got["12"]["abstract"], "")

    def _setup(self):
        item = {"item_id": "MC-1", "question": "Does cranberry juice prevent urinary infection?",
                "newest": {"date": "2012-10-17"}}
        def rec(pmid, upper, types=()):
            return {"pmid": pmid, "lower": upper, "upper": upper, "pubtypes": list(types), "journal": "J"}
        probe = {"records": [rec("1", "2010-01-01"), rec("2", "2011-02-02"),
                             rec("3", "2012-10-18"),                       # after cutoff
                             rec("4", "2009-05-05", ["Editorial"]),       # excluded type
                             rec("5", "2008-01-01"),                      # no abstract
                             rec("6", "2007-07-07", ["Systematic Review"])]}
        abstracts = {p: _abs(p, f"title {p}") for p in "1234 6".replace(" ", "")}
        abstracts["5"] = {"title": "t", "abstract": ""}
        return item, probe, abstracts

    def test_pool_excludes_future_ineligible_and_abstractless_records(self):
        item, probe, abstracts = self._setup()
        res = freeze_item(item, probe, abstracts, query_encoder=HashingEncoder(),
                          article_encoder=HashingEncoder(), reranker=LexicalOverlapReranker(),
                          dense_k=10, pool_size=5)
        pmids = [c["pmid"] for c in res["candidates"]]
        self.assertEqual(sorted(pmids), ["1", "2", "6"])
        self.assertTrue(all(c["upper"] <= "2012-10-17" for c in res["candidates"]))
        self.assertEqual([c["rank"] for c in res["candidates"]], [1, 2, 3])
        self.assertTrue(next(c for c in res["candidates"] if c["pmid"] == "6")["is_review"])

    def test_pool_size_caps_and_hash_is_deterministic_and_order_sensitive(self):
        item, probe, abstracts = self._setup()
        kw = dict(query_encoder=HashingEncoder(), article_encoder=HashingEncoder(),
                  reranker=LexicalOverlapReranker(), dense_k=10)
        a = freeze_item(item, probe, abstracts, pool_size=2, **kw)
        b = freeze_item(item, probe, abstracts, pool_size=2, **kw)
        self.assertEqual(len(a["candidates"]), 2)
        self.assertEqual(a["pool_hash"], b["pool_hash"])
        self.assertNotEqual(pool_hash(a["candidates"]), pool_hash(list(reversed(a["candidates"]))))

    def test_empty_pool_is_reported_not_crashed(self):
        item, probe, _ = self._setup()
        res = freeze_item(item, probe, {}, query_encoder=HashingEncoder(),
                          article_encoder=HashingEncoder(), reranker=LexicalOverlapReranker(),
                          dense_k=10, pool_size=5)
        self.assertEqual(res["candidates"], [])
        self.assertEqual(res["n_eligible"], 0)


class FetchAbstractsTests(unittest.TestCase):
    """A connection cut in the middle of a download (IncompleteRead) is retried, not fatal."""

    def test_efetch_retries_an_incomplete_read(self):
        calls = []

        def opener(url):
            calls.append(url)
            if len(calls) == 1:
                raise http.client.IncompleteRead(b"partial")
            return XML.encode("utf-8")

        got = fetch_abstracts(EUtils(opener=opener, sleep=lambda s: None), ["11", "12"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(sorted(got), ["11", "12"])
        self.assertIn("rettype=abstract", calls[0])

    def test_the_records_are_fetched_in_batches_of_one_hundred(self):
        calls = []

        def opener(url):
            calls.append(url)
            return b"<PubmedArticleSet></PubmedArticleSet>"
        self.assertEqual(fetch_abstracts(EUtils(opener=opener, sleep=lambda s: None), [str(i) for i in range(250)]), {})
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
