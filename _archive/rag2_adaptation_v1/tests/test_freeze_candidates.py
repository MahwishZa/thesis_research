"""Freezing one candidate pool per item: the abstracts of the as-of records, eligibility, ranking and the pool hash (fake encoders; no network)."""

import contextlib
import http.client
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.medchange import freeze_candidates as FC
from experiments.medchange.eutils import EUtils
from experiments.medchange.freeze_candidates import fetch_abstracts, freeze_item, parse_efetch_xml, pool_hash
from tests.fixtures import HashingEncoder, LexicalOverlapReranker


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



class MainTests(unittest.TestCase):
    """The command end to end: the abstract cache, the pools and their resumption, with a fake E-utilities server and fake encoders."""

    LONG = "cranberry juice reduces recurrence of urinary infection in women. " * 6

    def _xml(self):
        body = "".join(f"<PubmedArticle><MedlineCitation><PMID>{p}</PMID><Article><ArticleTitle>Trial {p}</ArticleTitle>"
                       f"<Abstract><AbstractText>{self.LONG}</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>" for p in ("11", "12"))
        return f"<PubmedArticleSet>{body}</PubmedArticleSet>".encode("utf-8")

    def _world(self, tmp):
        d = Path(tmp)
        item = {"item_id": "MC-1", "question": "Does cranberry juice prevent urinary infection?", "split": "dev", "likely_label_noise": False,
                "newest": {"date": "2012-10-17"}, "previous": {"date": "2008-01-23"}}
        (d / "benchmark.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
        (d / "pubmed_g0").mkdir()
        records = [{"pmid": p, "lower": "2010-01-01", "upper": "2010-01-31", "pubtypes": [], "journal": "J"} for p in ("11", "12")]
        (d / "pubmed_g0" / "MC-1.json").write_text(json.dumps({"records": records}), encoding="utf-8")
        return d

    def _args(self, d, *extra):
        return ["--benchmark", str(d / "benchmark.jsonl"), "--probe-dir", str(d / "pubmed_g0"), "--abstract-cache", str(d / "abstracts.jsonl"),
                "--out", str(d / "frozen_dev.jsonl"), "--split", "dev", *extra]

    def _run(self, argv, opener):
        eu = lambda key=None: EUtils(key, opener=opener, sleep=lambda s: None)
        with mock.patch.object(FC, "EUtils", eu), contextlib.redirect_stdout(io.StringIO()) as out:
            return FC.main(argv), out.getvalue()

    def test_abstracts_only_downloads_each_abstract_once_and_freezes_nothing(self):
        urls = []

        def opener(url):
            urls.append(url)
            return self._xml()
        with TemporaryDirectory() as tmp:
            d = self._world(tmp)
            code, printed = self._run(self._args(d, "--abstracts-only"), opener)
            self.assertEqual(code, 0)
            cached = [json.loads(line) for line in (d / "abstracts.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sorted(r["pmid"] for r in cached), ["11", "12"])
            self.assertFalse((d / "frozen_dev.jsonl").exists())
            self.assertIn("abstracts only", printed)
            self.assertEqual(len(urls), 1)
            self._run(self._args(d, "--abstracts-only"), opener)                            # the cache is complete: nothing is downloaded again
            self.assertEqual(len(urls), 1)

    def test_the_pools_are_frozen_once_and_a_rerun_finds_them_finished(self):
        encoders = [mock.patch("experiments.medchange.encoders.medcpt_query_encoder", lambda **k: HashingEncoder()),
                    mock.patch("experiments.medchange.encoders.medcpt_article_encoder", lambda **k: HashingEncoder()),
                    mock.patch("experiments.medchange.encoders.MedCPTReranker", lambda **k: LexicalOverlapReranker())]
        for patch in encoders:
            patch.start()
        try:
            with TemporaryDirectory() as tmp:
                d = self._world(tmp)
                code, printed = self._run(self._args(d), lambda url: self._xml())
                self.assertEqual(code, 0)
                frozen = [json.loads(line) for line in (d / "frozen_dev.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(frozen), 1)
                self.assertEqual(sorted(c["pmid"] for c in frozen[0]["candidates"]), ["11", "12"])
                self.assertEqual(frozen[0]["encoders"], {"query": "hashing-stub", "article": "hashing-stub", "reranker": "lexical-overlap-stub"})
                self.assertIn("frozen pools: 1; empty: 0", printed)
                before = (d / "frozen_dev.jsonl").read_bytes()
                code, printed = self._run(self._args(d), lambda url: self.fail("nothing is left to download"))
                self.assertEqual(code, 0)
                self.assertIn("1 already frozen, 0 to do", printed)
                self.assertEqual((d / "frozen_dev.jsonl").read_bytes(), before)
        finally:
            for patch in encoders:
                patch.stop()

    def test_a_missing_probe_is_reported_before_anything_is_downloaded(self):
        with TemporaryDirectory() as tmp:
            d = self._world(tmp)
            (d / "pubmed_g0" / "MC-1.json").unlink()
            with contextlib.redirect_stderr(io.StringIO()) as err:
                code, _ = self._run(self._args(d), lambda url: self.fail("no download without a probe"))
            self.assertEqual(code, 2)
            self.assertIn("missing PubMed probe for MC-1", err.getvalue())


if __name__ == "__main__":
    unittest.main()
