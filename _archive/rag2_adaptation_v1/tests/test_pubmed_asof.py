"""The as-of PubMed probe: the query, the cutoff, the update window and the records proven to precede it (a fake opener; no network)."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.medchange import pubmed_asof as PA
from experiments.medchange.eutils import EUtils
from experiments.medchange.pubmed_asof import (
    build_term, day_before, probe_item, query_terms, summarize, window_counts,
)


class FakeOpener:
    def __init__(self, responses, fail_first=0):
        self.responses = list(responses)
        self.urls = []
        self.fail_first = fail_first

    def __call__(self, url):
        self.urls.append(url)
        if self.fail_first:
            self.fail_first -= 1
            raise TimeoutError("simulated")
        return json.dumps(self.responses.pop(0)).encode()


def search(count, ids):
    return {"esearchresult": {"count": str(count), "idlist": ids}}


def summ(*recs):
    """recs: (pmid, pubdate, epubdate, pubtypes)"""
    return {"result": {p: {"pubdate": pd, "epubdate": ed, "sortpubdate": "", "pubtype": t,
                           "source": "J"} for p, pd, ed, t in recs}}


ITEM = {"item_id": "MC-00001", "question": "Does cranberry juice prevent urinary tract infections?",
        "newest": {"date": "2012-10-17"}, "previous": {"date": "2008-01-23"}}


class PubMedProbeTests(unittest.TestCase):

    def test_query_uses_content_words_and_excludes_cochrane(self):
        terms = query_terms(ITEM["question"])
        self.assertEqual(terms, ["cranberry", "juice", "urinary", "tract", "infections"])
        self.assertIn('NOT "Cochrane Database Syst Rev"[Journal]', build_term(terms, mode="and"))
        self.assertIn(" OR ", build_term(terms, mode="or"))

    def test_cutoff_is_strictly_before_newest_version(self):
        self.assertEqual(day_before("2012-10-17"), "2012/10/16")

    def test_window_counts_are_half_open(self):
        def r(d, t):
            return {"lower": d, "upper": d, "pubtypes": [t]}
        recs = [r("2008-01-23", "Randomized Controlled Trial"),   # on the previous date: not after it
                r("2010-05-01", "Randomized Controlled Trial"),
                r("2011-01-01", "Meta-Analysis"),
                r("2001-01-01", "Journal Article"),
                {"lower": "2012-10-01", "upper": "2012-10-31", "pubtypes": ["Meta-Analysis"]}]  # month straddles cutoff
        self.assertEqual(window_counts(recs, "2008-01-23", "2012-10-17", 50),
                         {"in_window": 2, "trials_in_window": 1, "reviews_in_window": 1})

    def test_falls_back_to_or_when_and_query_is_too_narrow(self):
        op = FakeOpener([search(3, ["1"]), search(500, ["1", "2"]),
                         summ(("1", "2009 Mar 1", "", ["Randomized Controlled Trial"]),
                              ("2", "1999", "", ["Review"]))])
        res = probe_item(ITEM, EUtils(opener=op, sleep=lambda s: None), retmax=200, min_hits=30)
        self.assertEqual(res["mode"], "or")
        self.assertEqual(res["top200"]["trials_in_window"], 1)
        self.assertTrue(all("maxdate=2012%2F10%2F16" in u for u in op.urls[:2]))

    def test_print_date_after_cutoff_but_epub_before_is_kept(self):
        """PubMed's [dp] filter matches print OR electronic date: this record
        passes the search, its print issue (2013) is after the cutoff, but it was
        e-published in 2012-09 - available before 2012-10-17, so it is kept."""
        op = FakeOpener([search(100, ["1"]), summ(("1", "2013 Jan", "2012 Sep 10", ["Randomized Controlled Trial"]))])
        res = probe_item(ITEM, EUtils(opener=op, sleep=lambda s: None), retmax=200, min_hits=30)
        self.assertEqual([r["pmid"] for r in res["records"]], ["1"])
        self.assertEqual(res["records"][0]["upper"], "2012-09-10")

    def test_record_not_provably_before_cutoff_is_dropped_not_admitted(self):
        op = FakeOpener([search(100, ["1", "2"]),
                         summ(("1", "2013 Jan", "", []),         # print 2013, no epub date
                              ("2", "2012 Oct", "", []))])        # month straddles the cutoff day
        res = probe_item(ITEM, EUtils(opener=op, sleep=lambda s: None), retmax=200, min_hits=30)
        self.assertEqual(res["records"], [])
        self.assertEqual(res["n_dropped"], 2)

    def test_systematic_drops_abort_the_run(self):
        ids = [str(i) for i in range(30)]
        op = FakeOpener([search(500, ids), summ(*[(i, "2020 Jan", "", []) for i in ids])])
        with self.assertRaises(RuntimeError):
            probe_item(ITEM, EUtils(opener=op, sleep=lambda s: None), retmax=200, min_hits=30)

    def test_retries_transient_failures(self):
        op = FakeOpener([search(100, [])], fail_first=2)
        res = probe_item(ITEM, EUtils(opener=op, sleep=lambda s: None), retmax=200, min_hits=30)
        self.assertEqual(res["records"], [])

    def test_summary(self):
        results = [{"item_id": "a", "count": 10, "top50": {"trials_in_window": 1, "reviews_in_window": 0, "in_window": 1},
                    "top200": {"trials_in_window": 1, "reviews_in_window": 0, "in_window": 3}},
                   {"item_id": "b", "count": 0, "top50": {"trials_in_window": 0, "reviews_in_window": 0, "in_window": 0},
                    "top200": {"trials_in_window": 0, "reviews_in_window": 0, "in_window": 0}}]
        s = summarize(results, {"a": {"kind": "changed"}, "b": {"kind": "changed"}})
        self.assertEqual(s["changed"]["share_with_trial_or_review_in_window_top200"], 0.5)
        self.assertEqual(s["changed"]["zero_hits"], 1)



class MainTests(unittest.TestCase):
    """The command: one cached file per item, reruns that skip it, the split filter and the pre-stated G0 line (a fake E-utilities server)."""

    def _benchmark(self, tmp):
        rows = [dict(ITEM, split="dev", likely_label_noise=False, kind="changed"),
                dict(ITEM, item_id="MC-2", split="confirm", likely_label_noise=False, kind="changed"),
                dict(ITEM, item_id="MC-3", split="dev", likely_label_noise=True, kind="changed")]
        path = Path(tmp) / "benchmark.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return path

    def _run(self, tmp, opener, *extra):
        eu = lambda key=None: EUtils(key, opener=opener, sleep=lambda s: None)
        argv = ["--benchmark", str(self._benchmark(tmp)), "--cache-dir", str(Path(tmp) / "cache"), "--min-hits", "1", *extra]
        with mock.patch.object(PA, "EUtils", eu), contextlib.redirect_stdout(io.StringIO()) as out:
            return PA.main(argv), out.getvalue()

    def test_a_dev_item_is_probed_once_cached_and_judged_by_the_pre_stated_gate(self):
        opener = FakeOpener([search(100, ["1"]), summ(("1", "2010 Mar 1", "", ["Randomized Controlled Trial"]))])
        with TemporaryDirectory() as tmp:
            code, printed = self._run(tmp, opener)
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "cache").iterdir()), ["MC-00001.json"])       # not the confirm item, not the noisy one
            cached = json.loads((Path(tmp) / "cache" / "MC-00001.json").read_text(encoding="utf-8"))
            self.assertEqual([r["pmid"] for r in cached["records"]], ["1"])
            self.assertIn("G0 (pre-stated): >= 50% of changed items have a trial or review in the update window among the as-of candidates -> PASS", printed)
            self.assertEqual(len(opener.urls), 2)
            code, _ = self._run(tmp, opener)                                       # cached: no request is made (the fake has no response left)
            self.assertEqual((code, len(opener.urls)), (0, 2))

    def test_the_split_option_selects_the_items_and_all_takes_every_usable_one(self):
        opener = FakeOpener([search(100, []), search(100, [])])
        with TemporaryDirectory() as tmp:
            code, _ = self._run(tmp, opener, "--split", "all")
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "cache").iterdir()), ["MC-00001.json", "MC-2.json"])


if __name__ == "__main__":
    unittest.main()
