"""Checks of src/build_knowledge_corpus.py without internet: the downloads are replaced by small made-up answers in the formats
of the real services (PubMed XML, PMC OA service XML, BioC JSON, the Dermoscopedia wiki API, the Meditron guideline rows)."""

import argparse
import contextlib
import csv
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
import build_knowledge_corpus as kc  # noqa: E402

LONG = ("Melanoma is a malignant tumour of melanocytes that occurs mostly in older adults on sun-exposed skin. "
        "Dermoscopy shows an atypical pigment network, irregular streaks and a blue-white veil. "
        "It must be distinguished from a melanocytic nevus, which usually shows a regular network. ")
PUBMED_XML = f"""<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>101</PMID><Article><ArticleTitle>Dermoscopy of melanoma</ArticleTitle>
<Abstract><AbstractText>{LONG}</AbstractText></Abstract><PublicationTypeList><PublicationType>Review</PublicationType>
</PublicationTypeList><Journal><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal></Article>
</MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>102</PMID><Article><ArticleTitle>Deep learning on the ISIC images</ArticleTitle>
<Abstract><AbstractText>{LONG}</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>103</PMID><Article><ArticleTitle>Melanoma diagnosis</ArticleTitle>
<Abstract><AbstractText>{LONG}</AbstractText></Abstract><PublicationTypeList>
<PublicationType>Retracted Publication</PublicationType></PublicationTypeList></Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>104</PMID><Article><ArticleTitle>Dermoscopy of seborrheic keratosis</ArticleTitle>
<Abstract><AbstractText>Seborrheic keratosis is a benign tumour of older adults. Dermoscopy shows milia-like cysts and
comedo-like openings, which help to distinguish it from melanoma.</AbstractText></Abstract></Article></MedlineCitation>
</PubmedArticle>
</PubmedArticleSet>"""
BIOC = [{"documents": [{"id": "PMC11", "passages": [
    {"infons": {"section_type": "TITLE", "type": "front", "year": "2021"}, "text": "Basal cell carcinoma: a review"},
    {"infons": {"section_type": "ABSTRACT", "type": "abstract"},
     "text": "Basal cell carcinoma is the most common skin cancer in older people. Dermoscopy shows arborizing vessels."},
    {"infons": {"section_type": "INTRO", "type": "paragraph"},
     "text": "Basal cell carcinoma mostly occurs on the face. It can mimic a seborrheic keratosis."},
    {"infons": {"section_type": "REF", "type": "ref"}, "text": "Smith J. A reference that must not be kept as a passage."},
    {"infons": {"section_type": "TABLE", "type": "table"}, "text": "a table that must not be kept as a passage"}]}]}]
BIOC_AK = {"documents": [{"id": "PMC13", "passages": [
    {"infons": {"section_type": "TITLE", "type": "front"}, "text": "Actinic keratosis in older adults"},
    {"infons": {"section_type": "DISCUSS", "type": "paragraph"},
     "text": "Actinic keratosis appears on sun-damaged skin of the face. Dermoscopy shows a strawberry pattern."}]}]}
WIKI_PAGE = ("<div><p>Dermatofibroma is a benign skin lesion, often on the legs of women. Dermoscopy shows a central white "
             "patch with a delicate peripheral pigment network.<sup>[1]</sup></p><table><tr><td><p>table text</p></td></tr>"
             "</table><span class='mw-editsection'>edit</span></div>")


class FakeFetcher:
    """Answers like the real services; every class gets the same made-up documents."""
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, ncbi=False):
        params = params or {}
        self.calls.append((url, dict(params), ncbi))
        if url.endswith("esearch.fcgi"):
            guideline = "guideline" in params["term"]
            ids = ["101", "102", "103", "104"] if params["db"] == "pubmed" else ["11", "12"] if guideline else ["13", "12"]
            return json.dumps({"esearchresult": {"idlist": ids}})
        if url.endswith("efetch.fcgi"):
            return PUBMED_XML
        if url == kc.PMC_OA:
            if params["id"] == "PMC12":
                return '<OA><error code="idIsNotOpenAccess">not open access</error></OA>'
            return f'<OA><records><record id="{params["id"]}" license="CC BY" retracted="no"/></records></OA>'
        if "BioC_json" in url:
            return json.dumps(BIOC_AK if "PMC13" in url else BIOC)
        if url == kc.DERMOSCOPEDIA_APIS[0]:
            if params.get("meta") == "siteinfo":
                return "{}"
            if params.get("list") == "search":
                return json.dumps({"query": {"search": [{"title": "Dermatofibroma"}]}})
            return json.dumps({"parse": {"text": WIKI_PAGE}})
        raise AssertionError(f"unexpected download {url}")


MEDITRON = [
    {"id": "g1", "source": "cco", "title": "Melanoma follow-up", "url": "https://example.org/g1",
     "clean_text": "Melanoma patients need regular skin checks. A changing lesion should be examined by dermoscopy.\n\n"
                   "Guideline text about melanoma staging and follow-up after excision of the primary tumour.\n\n" + LONG},
    {"id": "g2", "source": "wikidoc", "title": "Melanoma", "clean_text": "Crowdsourced text about melanoma."},
    {"id": "g3", "source": "nice", "title": "Melanoma", "clean_text": "NICE text about melanoma."},
    {"id": "g4", "source": "who", "title": "Malaria", "clean_text": "Text about malaria only."},
]


def args(**kw):
    base = dict(libraries=list(kc.LIBRARIES), per_class=50, max_words=200, min_words=5, pubmed_docs=10, pmc_docs=5,
                guideline_docs=5, textbook_pages=3)
    base.update(kw)
    return argparse.Namespace(**base)


class TextTests(unittest.TestCase):
    def test_passages_are_whole_sentences_within_the_word_limit_with_one_sentence_overlap(self):
        text = "One two three four. Five six seven eight. Nine ten eleven twelve. Thirteen fourteen."
        parts = kc.chunk(text, max_words=8)
        self.assertEqual(parts, ["One two three four. Five six seven eight.", "Five six seven eight. Nine ten eleven twelve.",
                                 "Nine ten eleven twelve. Thirteen fourteen."])
        self.assertTrue(all(len(p.split()) <= 8 for p in parts))

    def test_a_sentence_longer_than_the_limit_is_cut_and_nothing_is_lost(self):
        words = [f"w{i}" for i in range(25)]
        parts = kc.chunk(" ".join(words), max_words=10)
        self.assertEqual([len(p.split()) for p in parts], [10, 10, 5])
        self.assertEqual(" ".join(parts).split(), words)

    def test_a_short_text_is_one_passage_and_empty_text_is_none(self):
        self.assertEqual(kc.chunk("Dermatofibroma is benign.", 200), ["Dermatofibroma is benign."])
        self.assertEqual(kc.chunk("   ", 200), [])

    def test_tags_name_the_diagnoses_a_passage_talks_about(self):
        self.assertEqual(kc.tag("A seborrhoeic keratosis can mimic melanoma."), ["BKL", "MEL"])
        self.assertEqual(kc.tag("Bowen's disease is squamous cell carcinoma in situ."), ["SCC"])
        self.assertEqual(kc.tag("arborizing vessels", title="Basal cell carcinoma"), ["BCC"])
        self.assertEqual(kc.tag("nevirapine is a drug"), [])    # whole words only ("nevi" is inside "nevirapine")

    def test_ai_and_image_dataset_papers_are_recognised(self):
        self.assertTrue(kc.is_ai_paper("Deep learning for skin cancer", ""))
        self.assertTrue(kc.is_ai_paper("A study", "We trained on HAM10000 images."))
        self.assertFalse(kc.is_ai_paper("Dermoscopy of basal cell carcinoma", LONG))

    def test_only_creative_commons_licences_count_as_open(self):
        for licence in ("CC BY", "CC0", "CC BY-NC-ND", "cc by-sa"):
            self.assertTrue(kc.open_licence(licence), licence)
        for licence in ("NO-CC CODE", "", None, "CC BY-XYZ"):
            self.assertFalse(kc.open_licence(licence), licence)

    def test_topics_of_a_passage(self):
        self.assertEqual(kc.topics(LONG), ["appearance", "basics", "differential", "patients"])
        self.assertEqual(kc.topics("Nothing here."), [])

    def test_wiki_html_keeps_paragraphs_and_list_items_only(self):
        paras = kc.html_paragraphs(WIKI_PAGE + "<ul><li>first<br/>line</li></ul>")
        self.assertEqual(len(paras), 2)
        self.assertNotIn("[1]", paras[0])
        self.assertNotIn("table text", " ".join(paras))
        self.assertEqual(paras[1], "first line")


class ParserTests(unittest.TestCase):
    def test_pubmed_xml(self):
        docs = kc.parse_pubmed_xml(PUBMED_XML)
        self.assertEqual([d["doc_id"] for d in docs], ["PMID:101", "PMID:102", "PMID:103", "PMID:104"])
        self.assertEqual((docs[0]["year"], docs[0]["url"]), ("2020", "https://pubmed.ncbi.nlm.nih.gov/101/"))
        self.assertEqual([d["retracted"] for d in docs], [False, False, True, False])

    def test_pmc_oa_answer(self):
        self.assertEqual(kc.parse_oa_record('<OA><records><record id="PMC1" license="CC BY-NC" retracted="yes"/></records></OA>'),
                         ("CC BY-NC", True))
        self.assertEqual(kc.parse_oa_record('<OA><error code="idIsNotOpenAccess">x</error></OA>'), (None, None))

    def test_bioc_keeps_text_sections_and_drops_references_and_tables(self):
        for answer in (BIOC, BIOC[0]):                        # a list of collections, or one collection
            title, year, paras = kc.parse_bioc(answer)
            self.assertEqual((title, year, len(paras)), ("Basal cell carcinoma: a review", "2021", 2))
            self.assertFalse([p for p in paras if "reference" in p or "table" in p])

    def test_meditron_guidelines_without_wikidoc_nice_and_unrelated_topics(self):
        removed = kc.Counter()
        docs = list(kc.meditron_guideline_docs(removed, MEDITRON))
        self.assertEqual([d["doc_id"] for d in docs], ["meditron:g1"])
        self.assertEqual(len(docs[0]["paragraphs"]), 3)
        self.assertEqual(removed, {"guidelines: source wikidoc excluded": 1, "guidelines: source nice excluded": 1})


class BuildTests(unittest.TestCase):
    def build(self, **kw):
        fetch = FakeFetcher()
        with contextlib.redirect_stdout(io.StringIO()):
            passages, removed = kc.build(args(**kw), fetch, meditron_rows=MEDITRON)
        return passages, removed, fetch

    def test_every_library_contributes_and_every_passage_can_be_cited(self):
        passages, removed, _ = self.build()
        self.assertEqual({p["library"] for p in passages}, set(kc.LIBRARIES))
        for p in passages:
            self.assertTrue(p["url"] and p["licence"] and p["doc_id"] and p["text"], p["passage_id"])
        self.assertEqual(len({p["passage_id"] for p in passages}), len(passages))
        self.assertEqual(len({p["md5"] for p in passages}), len(passages))      # no duplicate text

    def test_the_rules_remove_ai_papers_retracted_papers_closed_papers_and_repeats(self):
        passages, removed, _ = self.build()
        ids = {p["doc_id"] for p in passages}
        self.assertNotIn("PMID:102", ids)                     # deep learning on ISIC
        self.assertNotIn("PMID:103", ids)                     # retracted
        self.assertNotIn("PMC12", ids)                        # not open access
        self.assertGreater(removed["pubmed: AI or image-dataset paper"], 0)
        self.assertGreater(removed["pubmed: retracted"], 0)
        self.assertGreater(removed["pmc: no open licence"], 0)
        self.assertEqual({p["library"] for p in passages if p["doc_id"] == "PMC11"}, {"guidelines"})  # not counted twice
        self.assertEqual(removed["pubmed: document already used"], 14)  # PMIDs 101 and 104 come back for all 8 diagnoses
        self.assertGreater(removed["pubmed: duplicate passage"], 0)      # the text of 101 is also in a guideline
        self.assertIn("PMID:104", ids)

    def test_no_library_gets_more_than_the_cap_per_diagnosis(self):
        passages, _, _ = self.build(per_class=1)
        counts = kc.Counter((p["library"], p["query_class"]) for p in passages)
        self.assertEqual(max(counts.values()), 1)

    def test_short_passages_are_dropped(self):
        passages, removed, _ = self.build(min_words=1000)
        self.assertEqual(passages, [])
        self.assertGreater(removed["pubmed: passage too short"], 0)

    def test_ncbi_requests_carry_no_email(self):
        _, _, fetch = self.build()
        ncbi = [c for c in fetch.calls if c[2]]
        self.assertTrue(ncbi)
        self.assertFalse([c for c in fetch.calls if "email" in c[1]])
        self.assertNotIn("--email", " ".join(kc.parse_args([]).__dict__))

    def test_the_card_counts_and_the_manifest_has_no_text(self):
        passages, removed, _ = self.build()
        card = kc.corpus_card(passages, removed, args(out="x", cache="y"), "2026-11-22")
        self.assertEqual(card["n_passages"], len(passages))
        self.assertEqual(sum(sum(v.values()) for v in card["passages_per_library_and_class"].values()), len(passages))
        self.assertIn("VASC/appearance", card["empty_topic_boxes"])   # no made-up text is about vascular lesions
        self.assertNotIn("out", card["settings"])
        with tempfile.TemporaryDirectory() as d:
            kc.write_outputs(passages, card, d)
            with open(os.path.join(d, "manifest.csv"), encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            with open(os.path.join(d, "passages.jsonl"), encoding="utf-8") as f:
                lines = [json.loads(x) for x in f]
        self.assertEqual(len(rows), len(passages))
        self.assertNotIn("text", rows[0])
        self.assertEqual([r["passage_id"] for r in rows], [p["passage_id"] for p in passages])
        texts = {p["text"] for p in passages}
        self.assertFalse([v for r in rows for v in r.values() if v in texts])
        self.assertEqual([x["passage_id"] for x in lines], [p["passage_id"] for p in passages])

    def test_a_missing_meditron_collection_does_not_stop_the_build(self):
        class Broken:
            def __iter__(self):
                raise RuntimeError("gated dataset")
        with contextlib.redirect_stdout(io.StringIO()):
            passages, removed = kc.build(args(libraries=["guidelines"]), FakeFetcher(), meditron_rows=Broken())
        self.assertEqual(removed["guidelines: Meditron collection not available"], 1)
        self.assertTrue(passages)                             # the PMC guideline papers are still used


class FetcherTests(unittest.TestCase):
    def test_the_api_key_is_sent_but_never_stored_and_answers_are_cached(self):
        seen = []

        class Answer:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"ok": 1}'

        def fake_open(req, timeout):
            seen.append(req.full_url)
            return Answer()

        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"NCBI_API_KEY": "SECRET123"}), \
                mock.patch("urllib.request.urlopen", fake_open), mock.patch("time.sleep"):
            f = kc.Fetcher(d)
            self.assertEqual(f.get(kc.NCBI + "esearch.fcgi", {"db": "pubmed"}, ncbi=True), '{"ok": 1}')
            self.assertEqual(f.get(kc.NCBI + "esearch.fcgi", {"db": "pubmed"}, ncbi=True), '{"ok": 1}')
            stored = [Path(d, n).read_text(encoding="utf-8") + n for n in os.listdir(d)]
        self.assertEqual(len(seen), 1)                        # the second answer came from the cache
        self.assertIn("api_key=SECRET123", seen[0])
        self.assertIn("tool=" + kc.TOOL, seen[0])
        self.assertNotIn("email", seen[0])
        self.assertFalse([s for s in stored if "SECRET123" in s])

    def test_the_cache_is_the_same_with_and_without_an_api_key(self):
        class Answer:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b"cached answer"

        def no_internet(req, timeout):
            raise AssertionError("this answer should come from the cache")

        with tempfile.TemporaryDirectory() as d, mock.patch("time.sleep"):
            with mock.patch.dict(os.environ, {"NCBI_API_KEY": "SECRET123"}), \
                    mock.patch("urllib.request.urlopen", lambda req, timeout: Answer()):
                kc.Fetcher(d).get(kc.NCBI + "efetch.fcgi", {"id": "1"}, ncbi=True)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch("urllib.request.urlopen", no_internet):
                self.assertEqual(kc.Fetcher(d).get(kc.NCBI + "efetch.fcgi", {"id": "1"}, ncbi=True), "cached answer")

if __name__ == "__main__":
    unittest.main()
