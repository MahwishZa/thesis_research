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
    "Actinic keratosis": page(["Nonpigmented actinic keratosis", "Pigmented actinic keratosis", "Short note"],
                              texts={"Short note": "<p>Too short to be a reference.</p>"}),
    "03-Seborrheic Keratosis / Lentigos": page(["Introduction", "Dermoscopic features", "Milia-like cysts"],
                                               texts={"nested": "Dermoscopic features"}),
    "Lichen planus-like keratosis": page(["Dermatoscopic Criteria"]),
    "03-Dermatofibroma": page(["Introduction", "Dermoscopy features &#91;1&#93; &#91;2&#93;", "Classic dermoscopic features"]),
    "Vascular lesions": page(intro=HUB.format("vascular lesions"), links=["Angioma", "Angiokeratoma"]),
    "Angioma": page(intro=f"<p>Angioma introduction first sentence. {WORDS} {WORDS}</p>"),
    "Squamous cell carcinoma": page(["Invasive squamous cell carcinoma", "Differential diagnosis", "References"],
                                    texts={"Invasive squamous cell carcinoma":
                                           f"<p>Keratin mass (Figure ). Hairpin vessels. . {WORDS}</p>"
                                           "<p>An Atlas of Dermoscopy, Second Edition. Marghoob A. et al. CRC Press; 2012.</p>"
                                           "<p>Kittler, H., Riedl, E., 2008, Vessel morphology. Dermatol, 14, 4.</p>"}),
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
            body = p["texts"].get(heading, f"<p>{params['page']} / {br.clean_heading(heading)} text. {WORDS}</p>")
            html = f"<h2>{heading}</h2><p>{params['page']} – cite!</p>{body}<table><tr><td><p>table text</p></td></tr></table>"
            sub = p["sections"][p["sections"].index(next(s for s in p["sections"] if s["index"] == index)) + 1:]
            if p["texts"].get("nested") == heading and sub:      # the wiki returns a section with its subsections
                html += f"<h3>{sub[0]['line']}</h3><p>{params['page']} / {sub[0]['line']} text. {WORDS}</p>"
        out = {"text": html}
        if "links" in params["prop"]:
            out["links"] = [{"ns": 0, "title": t} for t in p["links"]]
        return json.dumps({"parse": out})


def build(max_words=200, min_words=20):
    with tempfile.TemporaryDirectory() as d, mock.patch.object(br, "Fetcher", FakeWiki), \
            contextlib.redirect_stdout(io.StringIO()):
        out = os.path.join(d, "reference_texts.json")
        br.main(["--out", out, "--cache", os.path.join(d, "cache"), "--max-words", str(max_words), "--min-words", str(min_words)])
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

    def test_texts_of_whole_sentences_up_to_the_word_limit_and_short_ones_left_out(self):
        paragraphs = ["One two three. Four five six seven. Eight nine."]
        self.assertEqual(br.split_texts(paragraphs, 7, 1), (["One two three. Four five six seven.", "Eight nine."], []))
        self.assertEqual(br.split_texts(paragraphs, 7, 3), (["One two three. Four five six seven."], [2]))
        self.assertEqual(br.split_texts(["One two three four five."], 2, 1)[0][0], "One two")   # a too long sentence is cut
        self.assertEqual(br.split_texts([], 10, 1), ([], []))

    def test_a_section_is_read_without_its_subsections(self):
        section = "<h2>Criteria</h2><p>own</p><h3>Sub</h3><p>sub</p>"
        self.assertEqual(br.own_html(section), "<h2>Criteria</h2><p>own</p>")
        self.assertEqual(br.own_html("<p>no heading</p>"), "<p>no heading</p>")

    def test_sections_not_about_the_disease_are_left_out(self):
        sections = [{"index": "1", "line": "Dermoscopy"}, {"index": "2", "line": "Differential diagnosis"},
                    {"index": "3", "line": "Vessels"}, {"index": "4", "line": "References"}, {"index": "5", "line": "See also"}]
        self.assertEqual(br.choose_sections(sections)[0], ["1", "3"])

    def test_a_reference_list_is_cut_from_its_first_entry_and_a_citation_in_a_sentence_is_kept(self):
        text = ["Pigmented SCC shows blue structures (Kittler et al., 2008). Vessels are not seen.",
                "Argenziano, G., Zalaudek, I., 2004, Vascular structures. Arch Dermatol, 140, 1485–9.",
                "Kreusch, J. & Koch, F., 1996, Gefassmuster. Hautarzt, 47, 264–72."]
        self.assertEqual(br.without_bibliography(text),
                         (["Pigmented SCC shows blue structures (Kittler et al., 2008). Vessels are not seen."], 21))
        self.assertEqual(br.without_bibliography(["An Atlas of Dermoscopy, 2nd edition. Text after it."]), ([], 9))
        self.assertEqual(br.without_bibliography(["Plain text. No list."]), (["Plain text. No list."], 0))
        self.assertTrue(br.BIBLIOGRAPHY.search("Kreusch, J.F., 2002, Vascular patterns."))
        self.assertTrue(br.BIBLIOGRAPHY.search("Kreusch, J. & Koch, F., 1996, Gefassmuster."))
        self.assertFalse(br.BIBLIOGRAPHY.search("Types A. & B. are rare."))           # an author needs the comma
        self.assertFalse(br.BIBLIOGRAPHY.search("Erythema: Structureless pale-red areas, as in AK."))
        self.assertFalse(br.BIBLIOGRAPHY.search("Blue nevi, also known as Jadassohn-Tieche nevi, are common."))

    def test_the_text_of_each_table_of_a_section_without_its_subsections(self):
        section = ("<h2>Criteria</h2><p>prose</p><table><tr><td>Atypical network</td><td>Blue-white veil</td></tr>"
                   "<tr><td><table><tr><td>nested</td></tr></table></td></tr></table><p>more</p>"
                   "<table><tr><th>Regression</th></tr></table><h3>Sub</h3><table><tr><td>sub table</td></tr></table>")
        self.assertEqual(br.table_texts(section), ["Atypical network Blue-white veil nested", "Regression"])
        self.assertEqual(br.table_texts("<p>no table</p>"), [])

    def test_the_text_left_out_other_than_tables(self):
        section = ('<h2><span class="mw-headline">Criteria</span><span class="mw-editsection">edit</span></h2>'
                   '<p>read <b>bold</b></p><ul><li>item</li></ul><table><tr><td>table</td></tr></table>'
                   '<div class="thumb"><img alt="Atypical network" src="x.jpg"><div class="thumbcaption">Blue-white veil'
                   '<br>and regression</div></div><dl><dd>Irregular dots</dd></dl>loose text'
                   '<h3>Sub</h3><div>sub div</div>')
        self.assertEqual(br.skipped_texts(section), [("img alt", "Atypical network"),
                                                     ("div.thumbcaption", "Blue-white veil and regression"),
                                                     ("dd", "Irregular dots"), ("(no element)", "loose text")])
        self.assertEqual(br.skipped_texts("<p>only a paragraph</p>"), [])
        self.assertEqual(br.skipped_texts('<div class="x">caption<span class="mw-editsection">[edit]</span></div>before'
                                          '<div>after</div>'), [("div.x", "caption"), ("(no element)", "before"), ("div", "after")])

    def test_show_skipped_prints_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(br, "Fetcher", FakeWiki), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(br.main(["--show-skipped", "--cache", os.path.join(d, "cache"), "--out", os.path.join(d, "r.json")]))
            self.assertFalse(os.path.exists(os.path.join(d, "r.json")))
        self.assertIn("===== MEL: Superficial Spreading Melanoma / from 'Dermoscopic criteria': 0 pieces of text left out",
                      out.getvalue())

    def test_show_tables_prints_the_tables_of_the_chosen_sections_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(br, "Fetcher", FakeWiki), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(br.main(["--show-tables", "--cache", os.path.join(d, "cache"), "--out", os.path.join(d, "r.json")]))
            self.assertFalse(os.path.exists(os.path.join(d, "r.json")))
        text = out.getvalue()
        self.assertIn("===== MEL: Superficial Spreading Melanoma / from 'Dermoscopic criteria': 1 tables", text)
        self.assertIn("[Dermoscopic criteria, 2 words] table text", text)
        self.assertIn("===== VASC: Angioma / (introduction): 0 tables", text)

    def test_empty_figure_references_are_removed(self):
        self.assertEqual(br.text_paragraphs("<p>Blue dots (Figure ). This is BCC. . Then (see fig. ) more.</p>"),
                         ["Blue dots. This is BCC. Then more."])


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs = build()

    def sections(self, c):
        return [t["section"] for t in self.refs[c]["texts"]]

    def test_every_disease_has_texts_with_their_source_and_licence(self):
        self.assertEqual(sorted(self.refs), sorted(CLASSES))
        for cls_, r in self.refs.items():
            with self.subTest(cls_):
                self.assertTrue(r["texts"] and r["url"].startswith("https://dermoscopedia.org/"))
                self.assertEqual(r["licence"], "CC BY-NC-SA 4.0")
                for t in r["texts"]:
                    self.assertEqual(t["n_words"], len(t["text"].split()))
                    for junk in ("cite!", "table text", "subchapters", "(Figure", ". ."):
                        self.assertNotIn(junk, t["text"])

    def test_the_page_each_disease_gets_with_the_site_layout(self):
        source = {c: r["source"].removeprefix("Dermoscopedia: ") for c, r in self.refs.items()}
        self.assertEqual(source, {"MEL": "Superficial Spreading Melanoma",             # chapter page: first subchapter
                                  "NV": "Classification of nevi / benign nevus pattern",  # not the nail nevus page
                                  "BCC": "Basal cell carcinoma",                         # has sections: stays
                                  "AK": "Actinic keratosis", "BKL": "03-Seborrheic Keratosis / Lentigos",  # not LPLK
                                  "DF": "03-Dermatofibroma", "VASC": "Angioma", "SCC": "Squamous cell carcinoma"})
        self.assertIn("chapter page 'Melanoma'", self.refs["MEL"]["rule"])

    def test_one_text_per_criterion_section_not_only_the_first(self):
        self.assertEqual(self.sections("BCC"), ["Large blue-gray ovoid nests:", "Arborizing telangiectasia:"])
        self.assertEqual(self.sections("NV"), ["Globular pattern", "Reticular pattern"])
        self.assertEqual(self.sections("MEL"), ["Dermoscopic criteria"])                # from the dermoscopy heading on
        self.assertEqual(self.sections("DF"), ["Dermoscopy features", "Classic dermoscopic features"])
        self.assertEqual(self.sections("VASC"), ["(introduction)"])
        self.assertTrue(self.refs["VASC"]["texts"][0]["text"].startswith("Angioma introduction first sentence."))
        self.assertTrue(self.refs["BCC"]["texts"][1]["text"].startswith("Basal cell carcinoma / Arborizing telangiectasia: text."))

    def test_a_subsection_is_read_once(self):
        self.assertEqual(self.sections("BKL"), ["Dermoscopic features", "Milia-like cysts"])
        milia = "03-Seborrheic Keratosis / Lentigos / Milia-like cysts text."
        self.assertEqual(sum(t["text"].count(milia) for t in self.refs["BKL"]["texts"]), 1)

    def test_short_texts_are_skipped_and_listed_and_other_topics_left_out(self):
        self.assertEqual(self.sections("AK"), ["Nonpigmented actinic keratosis", "Pigmented actinic keratosis"])
        self.assertEqual(self.refs["AK"]["skipped"], [{"section": "Short note", "n_words": 6}])
        self.assertEqual(self.sections("SCC"), ["Invasive squamous cell carcinoma"])
        self.assertTrue(self.refs["SCC"]["texts"][0]["text"].startswith("Keratin mass. Hairpin vessels. Word"))
        self.assertTrue(self.refs["SCC"]["texts"][0]["text"].endswith("word."))         # the reference list is cut
        self.assertEqual(self.refs["SCC"]["bibliography_removed"],
                         [{"section": "Invasive squamous cell carcinoma", "n_words": 23}])
        self.assertEqual(self.refs["AK"]["bibliography_removed"], [])

    def test_the_word_limit_holds_for_every_text(self):
        refs = build(max_words=20, min_words=1)
        self.assertTrue(all(t["n_words"] <= 20 for r in refs.values() for t in r["texts"]))
        self.assertGreater(len(refs["BCC"]["texts"]), len(self.refs["BCC"]["texts"]))   # nothing is lost: more, shorter texts
        vasc = [t["section"] for t in refs["VASC"]["texts"]]
        self.assertTrue(len(vasc) > 1 and set(vasc) == {"(introduction)"})          # a long introduction: several texts

    def test_the_file_is_what_the_semantic_similarity_reads(self):
        q = {"id": "ISIC_0000001", "gold_label": "BCC", "group_id": "g"}
        second = self.refs["BCC"]["texts"][1]["text"]
        r = ev.semantic_similarity({"ISIC_0000001": q}, {"ISIC_0000001": {"rationale": second}}, self.refs,
                                   lambda texts: [[len(t), t.count("Arborizing")] for t in texts])
        self.assertEqual(r["mean"], 1.0)                      # the best match is the second text
        self.assertEqual(r["n_reference_texts"], {"BCC": 2})

    def test_the_token_check_counts_with_the_model_tokenizer(self):
        class Model:
            max_seq_length = 12

            def __init__(self, name):
                self.tokenizer = lambda text: {"input_ids": text.split()}
        fake = type(sys)("sentence_transformers")
        fake.SentenceTransformer = Model
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake}), contextlib.redirect_stdout(io.StringIO()) as out:
            longest, limit = br.token_check({"MEL": {"texts": [{"text": "a b c"}, {"text": " ".join("x" * 13)}]}},
                                            br.get_logger("reference_texts"))
        self.assertEqual((longest, limit), (13, 12))
        self.assertIn("1 texts are longer", out.getvalue())
        self.assertIn("rebuild with a lower --max-words", out.getvalue())
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake}), contextlib.redirect_stdout(io.StringIO()) as out:
            br.token_check({"MEL": {"texts": [{"text": "a b c"}]}}, br.get_logger("reference_texts"))
        self.assertNotIn("rebuild", out.getvalue())


if __name__ == "__main__":
    unittest.main()
