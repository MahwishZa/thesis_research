"""MedChange as-of benchmark builder and G0 PubMed probe (no network)."""

import json
import unittest

from experiments.medchange.benchmark import (
    BenchmarkError, assign_splits, build_items, load_groups, parse_cochrane_date,
    rebuild_medchangeqa, verify_against_release,
)
from experiments.medchange.headroom import parse_label, score
from experiments.medchange.pubmed_asof import (
    EUtils, build_term, day_before, probe_item, query_terms, summarize, window_counts,
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
    return {"result": {p: {"sortpubdate": d, "pubtype": t, "source": "J"} for p, d, t in recs}}


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
        recs = [{"date": "2008-01-23", "pubtypes": ["Randomized Controlled Trial"]},
                {"date": "2010-05-01", "pubtypes": ["Randomized Controlled Trial"]},
                {"date": "2011-01-01", "pubtypes": ["Meta-Analysis"]},
                {"date": "2001-01-01", "pubtypes": ["Journal Article"]}]
        self.assertEqual(window_counts(recs, "2008-01-23", "2012-10-17", 50),
                         {"in_window": 2, "trials_in_window": 1, "reviews_in_window": 1})

    def test_falls_back_to_or_when_and_query_is_too_narrow(self):
        op = FakeOpener([search(3, ["1"]), search(500, ["1", "2"]),
                         summ(("1", "2009/03/01 00:00", ["Randomized Controlled Trial"]),
                              ("2", "1999/01/01 00:00", ["Review"]))])
        res = probe_item(ITEM, EUtils(opener=op, sleep=lambda s: None), retmax=200, min_hits=30)
        self.assertEqual(res["mode"], "or")
        self.assertEqual(res["top200"]["trials_in_window"], 1)
        self.assertTrue(all("maxdate=2012%2F10%2F16" in u for u in op.urls[:2]))

    def test_record_after_cutoff_is_refused_as_leakage(self):
        op = FakeOpener([search(100, ["1"]), summ(("1", "2013/01/01 00:00", []))])
        with self.assertRaises(RuntimeError):
            probe_item(ITEM, EUtils(opener=op, sleep=lambda s: None), retmax=200, min_hits=30)

    def test_retries_transient_failures(self):
        op = FakeOpener([search(100, []), ], fail_first=2)
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
