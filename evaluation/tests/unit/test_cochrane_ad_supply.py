"""Counting new Cochrane dementia reviews (fake E-utilities; no network, no model)."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from experiments.medchange import cochrane_ad_supply as C
from experiments.medchange.pubmed_asof import EUtils


def article(pmid, title, conclusions=True, pubtype="Review"):
    ab = '<AbstractText Label="Main results">x</AbstractText>'
    if conclusions:
        ab += "<AbstractText Label=\"Authors' conclusions\">y</AbstractText>"
    return (f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><ArticleTitle>{title}</ArticleTitle>"
            f"<Abstract>{ab}</Abstract><PublicationTypeList><PublicationType>{pubtype}</PublicationType></PublicationTypeList>"
            "</Article><MeshHeadingList><MeshHeading><DescriptorName MajorTopicYN=\"Y\">Dementia</DescriptorName></MeshHeading>"
            "</MeshHeadingList></MedlineCitation></PubmedArticle>")


def fake_eu(records, alz=()):
    """records: {pmid: (xml, sortpubdate)}"""
    def opener(url):
        if "esearch.fcgi" in url:
            ids = list(records) if "Alzheimer+Disease%5Bmajr%5D+OR" in url else list(alz)
            return json.dumps({"esearchresult": {"count": str(len(ids)), "idlist": ids}}).encode()
        if "esummary.fcgi" in url:
            res = {p: {"sortpubdate": records[p][1].replace("-", "/") + " 00:00", "pubdate": records[p][1][:4], "epubdate": ""} for p in records}
            return json.dumps({"result": res}).encode()
        return ("<PubmedArticleSet>" + "".join(x for x, _ in records.values()) + "</PubmedArticleSet>").encode()
    return EUtils(None, opener=opener, sleep=lambda s: None)


class ClassifyTests(unittest.TestCase):
    def rec(self, title, **kw):
        return dict({"pmid": "1", "title": title, "has_conclusions": True, "pubtypes": [], "majors": []}, **kw)

    def test_effect_test_and_exclusions(self):
        self.assertEqual(C.classify(self.rec("Donepezil for dementia"))["template"], "effect")
        self.assertEqual(C.classify(self.rec("Blood tests for diagnosis of dementia")), {"status": "usable", "template": "test", "names_alzheimer": False})
        self.assertEqual(C.classify(self.rec("Memantine for Alzheimer's disease"))["names_alzheimer"], True)
        self.assertEqual(C.classify(self.rec("Donepezil for dementia", has_conclusions=False))["status"], "no_authors_conclusions")
        self.assertEqual(C.classify(self.rec("Donepezil for dementia (Protocol)"))["status"], "protocol_or_withdrawn")
        self.assertEqual(C.classify(self.rec("Donepezil for dementia", pubtypes=["Withdrawn Publication"]))["status"], "protocol_or_withdrawn")
        self.assertEqual(C.classify(self.rec("Care in dementia: an overview"))["status"], "title_not_x_for_y")


class CollectTests(unittest.TestCase):
    def test_counts_and_decision(self):
        recs = {"1": (article(1, "Donepezil for dementia"), "2024-03-01"),
                "2": (article(2, "Tests for diagnosis of Alzheimer's disease"), "2024-05-01"),
                "3": (article(3, "Protocol for dementia"), "2024-06-01"),
                "4": (article(4, "Music for dementia", conclusions=False), "2024-07-01"),
                "5": (article(5, "Old review for dementia"), "2024-01-08")}
        rep = C.collect(fake_eu(recs, alz=["2"]), "2024-01-08", "2026-10-01")
        self.assertEqual(rep["records_dementia_or_alzheimer_major"], 5)
        self.assertEqual(rep["usable"], 2)
        self.assertEqual(rep["usable_by_template"], {"effect": 1, "test": 1})
        self.assertEqual(rep["status"]["not_after_snapshot"], 1)
        self.assertEqual(rep["usable_alzheimer_major"], 1)
        self.assertEqual(rep["decision"], "TOO FEW")

    def test_go_and_pilot_thresholds(self):
        for n, want in ((30, "GO"), (20, "PILOT"), (19, "TOO FEW")):
            recs = {str(i): (article(i, f"Drug{i} for dementia"), "2025-01-01") for i in range(1, n + 1)}
            self.assertEqual(C.collect(fake_eu(recs), "2024-01-08", "2026-10-01")["decision"], want)

    def test_main_writes_counts_without_titles(self):
        recs = {"1": (article(1, "Secret title for dementia"), "2025-01-01")}
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()) as out:
            code = C.main(["--since", "2024-01-08", "--until", "2026-10-01", "--out-dir", d], eu=fake_eu(recs))
            saved = (Path(d) / "cochrane_ad_supply.json").read_text(encoding="utf-8")
        self.assertEqual(code, 3)
        self.assertNotIn("Secret title", saved)
        self.assertIn("Secret title", out.getvalue())
        self.assertIn("TOO FEW", out.getvalue())

    def test_snapshot_end_falls_back_to_benchmark(self):
        with tempfile.TemporaryDirectory() as d:
            b = Path(d) / "b.jsonl"
            b.write_text(json.dumps({"newest": {"date": "2023-05-01"}}) + "\n" + json.dumps({"newest": {"date": "2024-01-08"}}) + "\n")
            self.assertEqual(C.snapshot_end(None, b), "2024-01-08")


if __name__ == "__main__":
    unittest.main()
