"""Checks of src/build_reference_texts.py without internet: a made-up wiki in the format of Dermoscopedia's API, with the page
layouts the site showed on 2026-11-22 (--inspect): chapter pages, criteria as sections, a dermoscopy heading, nail nevi."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_reference_texts as br  # noqa: E402
import evaluate as ev  # noqa: E402
from common import CLASSES  # noqa: E402

WORDS = " ".join(["Word"] + ["word"] * 29) + "."


def page(sections=(), intro="", links=(), texts=None):
    return {"sections": [{"index": str(k + 1), "line": h} for k, h in enumerate(sections)], "intro": intro, "links": list(links),
            "texts": texts or {}}


HUB = "<p>{0} – cite!</p><p>This chapter covers dermoscopy of {0}. It has the following subchapters: A B C</p>"
SEARCH = {
    "melanoma": ["Melanoma", "Acrolentiginous Melanoma", "04-Melanoma"],
    "classification of nevi / benign nevus pattern": ["Classification of nevi / benign nevus pattern"],
    "nevus": ["Nevus", "Congenital nevus"],
    "basal cell carcinoma": ["Basal cell carcinoma", "Superficial Basal cell carcinoma"],
    "actinic keratosis": ["Actinic keratosis"],
    "seborrheic keratosis": ["03-Seborrheic Keratosis / Lentigos", "Correlation of seborrheic keratosis"],
    "lichen planus-like keratosis": ["Lichen planus-like keratosis"],
    "dermatofibroma": ["03-Dermatofibroma", "Correlation of dermatofibroma"],
    "vascular lesions": ["Benign lesions - angioma and other vascular lesions", "Vascular lesions"],
    "squamous cell carcinoma": ["Squamous cell carcinoma"],
}
PAGES = {
    "Melanoma": page(intro=HUB.format("melanoma"), links=["Superficial Spreading Melanoma", "Nodular Melanoma"]),
    "Superficial Spreading Melanoma": page(["Clinical", "Dermoscopic criteria"], intro="<p>Intro.</p>"),
    "Classification of nevi / benign nevus pattern": page(["Globular pattern", "Reticular pattern"]),
    "Nevus": page(["Nevus of the nail matrix"]),
    "Basal cell carcinoma": page(["Large blue-gray ovoid nests:", "Arborizing telangiectasia:"], intro=HUB.format("bcc"),
                                 links=["Superficial Basal cell carcinoma"]),
    "Actinic keratosis": page(["Nonpigmented actinic keratosis", "Pigmented actinic keratosis"]),
    "03-Seborrheic Keratosis / Lentigos": page(["Introduction", "Dermoscopic features", "Milia-like cysts"]),
    "Lichen planus-like keratosis": page(["Dermatoscopic Criteria"]),
    "03-Dermatofibroma": page(["Introduction", "Dermoscopy features &#91;1&#93; &#91;2&#93;", "Classic dermoscopic features"]),
    "Vascular lesions": page(intro=HUB.format("vascular lesions"), links=["Angioma", "Angiokeratoma"]),
    "Angioma": page(intro=f"<p>Angioma introduction first sentence. {WORDS} {WORDS}</p>"),
    "Squamous cell carcinoma": page(["Invasive squamous cell carcinoma"]),
}


class FakeWiki:
    def __init__(self, *a, **k):
        self.calls = []

    def get(self, url, params=None, ncbi=False):
        self.calls.append(params)
        if params.get("meta") == "siteinfo":
            return "{}"
        if params.get("list") == "search":
            return json.dumps({"query": {"search": [{"title": t} for t in SEARCH.get(params["srsearch"], [])]}})
        p = PAGES[params["page"]]
        if params["prop"] == "sections":
            return json.dumps({"parse": {"sections": p["sections"]}})
        index = params["section"]
        if index == 0:
            html = p["intro"]
        else:
            heading = next(s["line"] for s in p["sections"] if s["index"] == index)
            html = f"<p>{params['page']} – cite!</p><p>{params['page']} / {br.clean_heading(heading)} text. {WORDS}</p>" \
                   "<table><tr><td><p>table text</p></td></tr></table>"
        out = {"text": html}
        if "links" in params["prop"]:
            out["links"] = [{"ns": 0, "title": t} for t in p["links"]]
        return json.dumps({"parse": out})


def build(max_words=200):
    with tempfile.TemporaryDirectory() as d, mock.patch.object(br, "Fetcher", FakeWiki), \
            contextlib.redirect_stdout(io.StringIO()):
        out = os.path.join(d, "reference_texts.json")
        br.main(["--out", out, "--cache", os.path.join(d, "cache"), "--max-words", str(max_words)])
        return json.loads(Path(out).read_text(encoding="utf-8"))


class RuleTests(unittest.TestCase):
    def test_the_first_term_with_results_decides_and_an_equal_title_wins_within_it(self):
        self.assertEqual(br.choose_page([("nevus", ["Nevi of the face", "Nevus"])]), ("Nevus", "title equals 'nevus'"))
        self.assertEqual(br.choose_page([("a", []), ("b", ["B page", "Other"])]), ("B page", "first result for 'b'"))
        self.assertEqual(br.choose_page([("a", ["A page"]), ("b", ["b"])]), ("A page", "first result for 'a'"))
        self.assertEqual(br.choose_page([("a", [])]), (None, "no page found"))

    def test_sections_from_the_dermoscopy_heading_else_all_else_the_introduction(self):
        sections = [{"index": "1", "line": "History"}, {"index": "2", "line": "<b>Dermatoscopic</b> features"},
                    {"index": "3", "line": "Milia-like cysts"}]
        self.assertEqual(br.choose_sections(sections), (["2", "3"], "from 'Dermatoscopic features'"))
        self.assertEqual(br.choose_sections([{"index": "1", "line": "Large nests:"}])[0], ["1"])
        self.assertEqual(br.choose_sections([]), ([0], "(introduction)"))

    def test_headings_and_template_lines_are_cleaned(self):
        self.assertEqual(br.clean_heading("Dermoscopy features &#91;1&#93; &#91;2&#93;"), "Dermoscopy features")
        self.assertEqual(br.text_paragraphs("<p>Nevus – cite!</p><p>Nevus (message)</p><p>A nevus is benign.</p>"),
                         ["A nevus is benign."])

    def test_whole_sentences_up_to_the_word_limit(self):
        paragraphs = ["One two three. Four five six seven. Eight nine."]
        self.assertEqual(br.first_words(paragraphs, 7), "One two three. Four five six seven.")
        self.assertEqual(br.first_words(paragraphs, 2), "One two")              # a too long first sentence is cut
        self.assertEqual(br.first_words([], 10), "")


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs = build()

    def test_one_reference_per_disease_with_its_source_and_licence(self):
        self.assertEqual(sorted(self.refs), sorted(CLASSES))
        for cls_, r in self.refs.items():
            with self.subTest(cls_):
                self.assertTrue(r["text"] and r["url"].startswith("https://dermoscopedia.org/"))
                self.assertEqual(r["licence"], "CC BY-NC-SA 4.0")
                self.assertNotIn("cite!", r["text"])
                self.assertNotIn("table text", r["text"])
                self.assertNotIn("subchapters", r["text"])

    def test_the_page_each_disease_gets_with_the_site_layout(self):
        source = {c: r["source"].removeprefix("Dermoscopedia: ") for c, r in self.refs.items()}
        self.assertEqual(source, {"MEL": "Superficial Spreading Melanoma",             # chapter page: first subchapter
                                  "NV": "Classification of nevi / benign nevus pattern",  # not the nail nevus page
                                  "BCC": "Basal cell carcinoma",                         # has sections: stays
                                  "AK": "Actinic keratosis", "BKL": "03-Seborrheic Keratosis / Lentigos",  # not LPLK
                                  "DF": "03-Dermatofibroma", "VASC": "Angioma", "SCC": "Squamous cell carcinoma"})
        self.assertIn("chapter page 'Melanoma'", self.refs["MEL"]["rule"])

    def test_the_section_each_disease_gets(self):
        self.assertTrue(self.refs["MEL"]["text"].startswith("Superficial Spreading Melanoma / Dermoscopic criteria text."))
        self.assertTrue(self.refs["BKL"]["text"].startswith("03-Seborrheic Keratosis / Lentigos / Dermoscopic features text."))
        self.assertTrue(self.refs["DF"]["text"].startswith("03-Dermatofibroma / Dermoscopy features text."))
        self.assertTrue(self.refs["BCC"]["text"].startswith("Basal cell carcinoma / Large blue-gray ovoid nests: text."))
        self.assertIn("no dermoscopy heading", self.refs["AK"]["section"])
        self.assertEqual(self.refs["VASC"]["section"], "(introduction)")
        self.assertTrue(self.refs["VASC"]["text"].startswith("Angioma introduction first sentence."))

    def test_the_word_limit_holds_across_sections(self):
        refs = build(max_words=40)
        self.assertTrue(all(r["n_words"] <= 40 for r in refs.values()))
        self.assertGreater(self.refs["BKL"]["n_words"], 40)          # with the default it reads on into the next section

    def test_the_file_is_what_the_semantic_similarity_reads(self):
        q = {"id": "ISIC_0000001", "gold_label": "MEL", "group_id": "g"}
        r = ev.semantic_similarity({"ISIC_0000001": q}, {"ISIC_0000001": {"rationale": self.refs["MEL"]["text"]}}, self.refs,
                                   lambda texts: [[len(t), t.count("word")] for t in texts])
        self.assertEqual(r["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
