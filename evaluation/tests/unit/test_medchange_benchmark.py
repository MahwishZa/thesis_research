"""MedChange as-of benchmark builder and G0 PubMed probe (no network)."""

import json
import unittest

from experiments.medchange.benchmark import (
    BenchmarkError, assign_splits, build_items, load_groups, parse_cochrane_date,
    rebuild_medchangeqa, verify_against_release,
)
from experiments.medchange.headroom import parse_label, score
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange.benchmark import file_sha256
from experiments.medchange.pubmed_asof import (
    EUtils, availability, build_term, date_bounds, day_before, probe_item,
    query_terms, summarize, window_counts,
)


def row(i, label, date, concl="conclusion text", q="Does X help Y?"):
    return {"": str(i), "Label": label, "Question": q, "objectives": "",
            "conclusions": concl, "PMID": f"/{1000 + i}/",
            "DOI_Date": f"Cochrane Database Syst Rev. {date};1:CD00{1000 + i:04d}. doi: x"}


def fixture():
    # group 1: changed (newest row 0 SUPPORTED, older row 1 NEI)
    # group 2: unchanged (rows 2,3 REFUTED); group 3: changed but near-identical text
    medrev = {
        0: row(0, "SUPPORTED", "2020 Mar 5", "New trials show a clear benefit of X."),
        1: row(1, "NOT ENOUGH INFORMATION", "2010 Jan 2", "Too few studies to judge."),
        2: row(2, "REFUTED", "2019 May 1"), 3: row(3, "REFUTED", "2012"),
        4: row(4, "SUPPORTED", "2018 Jun 1", "Identical wording of the conclusion here."),
        5: row(5, "NOT ENOUGH INFORMATION", "2014 Jun 1", "Identical wording of the conclusion here."),
    }
    groups = load_groups([
        {"Group_ID": "1", "Study_ID": "0"}, {"Group_ID": "", "Study_ID": "1"},
        {"Group_ID": "2", "Study_ID": "2"}, {"Group_ID": "", "Study_ID": "3"},
        {"Group_ID": "3", "Study_ID": "4"}, {"Group_ID": "", "Study_ID": "5"},
    ])
    return medrev, groups


class BenchmarkTests(unittest.TestCase):

    def test_dates_and_precision(self):
        self.assertEqual(parse_cochrane_date("Cochrane Database Syst Rev. 2024 Jan 18;1(1):CD011039."),
                         ("2024-01-18", "day"))
        self.assertEqual(parse_cochrane_date("Cochrane Database Syst Rev. 2001;(3):CD002120."),
                         ("2001-01-01", "year"))
        with self.assertRaises(BenchmarkError):
            parse_cochrane_date("no date here")

    def test_blank_group_id_continues_previous_group(self):
        _, groups = fixture()
        self.assertEqual(groups, {1: [0, 1], 2: [2, 3], 3: [4, 5]})

    def test_rebuild_uses_authors_rule_and_verification_catches_mismatch(self):
        medrev, groups = fixture()
        rebuilt = rebuild_medchangeqa(medrev, groups)
        self.assertEqual(rebuilt, [(1, 0, 1), (3, 4, 5)])
        good = [{"Newest Label": "SUPPORTED", "Outdated Label": "NOT ENOUGH INFORMATION"},
                {"Newest Label": "SUPPORTED", "Outdated Label": "NOT ENOUGH INFORMATION"}]
        verify_against_release(rebuilt, medrev, good)
        bad = [dict(good[0]), {"Newest Label": "REFUTED", "Outdated Label": "SUPPORTED"}]
        with self.assertRaises(BenchmarkError):
            verify_against_release(rebuilt, medrev, bad)

    def test_items_flag_noise_and_sample_only_unchanged_groups(self):
        medrev, groups = fixture()
        items = build_items(medrev, groups, rebuild_medchangeqa(medrev, groups),
                            n_unchanged=5, seed=1)
        by = {i.item_id: i for i in items}
        self.assertFalse(by["MC-00001"].likely_label_noise)
        self.assertTrue(by["MC-00003"].likely_label_noise)
        self.assertEqual(by["MC-00002"].kind, "unchanged")
        self.assertEqual(by["MC-00001"].previous.date, "2010-01-02")
        self.assertEqual(by["MC-00001"].change_type, "NOT ENOUGH INFORMATION -> SUPPORTED")

    def test_splits_are_deterministic(self):
        medrev, groups = fixture()
        def run():
            items = build_items(medrev, groups, rebuild_medchangeqa(medrev, groups),
                                n_unchanged=5, seed=1)
            assign_splits(items, dev_fraction=0.5, seed=7)
            return [(i.item_id, i.split) for i in items]
        self.assertEqual(run(), run())
        self.assertTrue(all(s in ("dev", "confirm") for _, s in run()))


class HeadroomTests(unittest.TestCase):

    def test_label_parsing_and_scoring(self):
        self.assertEqual(parse_label("LABEL: not enough information |||"), "NOT ENOUGH INFORMATION")
        self.assertIsNone(parse_label("no verdict"))
        items = [{"kind": "changed", "likely_label_noise": False,
                  "newest": {"row": 0, "label": "SUPPORTED"},
                  "previous": {"row": 1, "label": "REFUTED"}}]
        s = score(items, ["REFUTED", None])
        self.assertEqual(s["changed"]["newest_verdict_accuracy"], 0.0)
        self.assertEqual(s["changed"]["previous_verdict_match"], 1.0)


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


class HashPortabilityTests(unittest.TestCase):

    def test_crlf_checkout_hashes_like_lf(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            a.write_bytes(b"x,y\n1,2\n")
            b.write_bytes(b"x,y\r\n1,2\r\n")
            self.assertEqual(file_sha256(a), file_sha256(b))


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

    def test_date_bounds(self):
        self.assertEqual(date_bounds("2012 Oct 17"), ("2012-10-17", "2012-10-17"))
        self.assertEqual(date_bounds("2012 Oct"), ("2012-10-01", "2012-10-31"))
        self.assertEqual(date_bounds("2012 Dec"), ("2012-12-01", "2012-12-31"))
        self.assertEqual(date_bounds("2012 Oct-Dec"), ("2012-10-01", "2012-12-31"))
        self.assertEqual(date_bounds("2012"), ("2012-01-01", "2012-12-31"))
        self.assertIsNone(date_bounds("n/a"))
        self.assertIsNone(date_bounds(""))

    def test_availability_is_the_earliest_known_date(self):
        self.assertEqual(availability("2013 Jan", "2012 Sep 10", None),
                         {"lower": "2012-09-10", "upper": "2012-09-10"})
        self.assertEqual(availability("2013 Jan", "", None),
                         {"lower": "2013-01-01", "upper": "2013-01-31"})
        self.assertEqual(availability("", "", "2011-04-02"),
                         {"lower": "2011-04-02", "upper": "2011-04-02"})
        self.assertIsNone(availability("", "", None))

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


if __name__ == "__main__":
    unittest.main()
