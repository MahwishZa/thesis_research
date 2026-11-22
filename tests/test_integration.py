"""Integration tests: several parts working together, on made-up folders and made-up answers of the outside services (no network,
no API call, no image needed). The corpus builder with fake downloads, the reference-text builder with a fake wiki, a whole run of
run.py with a fake model (answers saved, interrupted, resumed, then read by evaluate.py), the evaluate.py command, and check.py on
small folders. Run: python -m unittest discover -s tests"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build as bd  # noqa: E402
import check as cp  # noqa: E402
import evaluate as ev  # noqa: E402
import run as rl  # noqa: E402
from build import options_for, question  # noqa: E402
from common import ALL_CASES, CLASSES, PROCESSED_CASES, question_file, write_jsonl  # noqa: E402

DATA = ROOT / "data"


# ----------------------------------------------------------------------------------------------------------------------
# check.py on small made-up folders
# ----------------------------------------------------------------------------------------------------------------------


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data, self.results = Path(self.tmp.name, "data"), Path(self.tmp.name, "results")

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_empty_workspace_is_all_not_run(self):
        rows = {name: status for name, status, _ in cp.check(str(self.data), str(self.results))}
        self.assertEqual(set(rows.values()), {cp.NOT_RUN})

    def test_reference_texts_old_form_or_empty_are_a_problem(self):
        path = self.data / "knowledge_corpus" / "reference_texts.json"
        path.parent.mkdir(parents=True)
        good = {"texts": [{"section": "Dermoscopic criteria", "text": "x", "n_words": 120}]}
        path.write_text(json.dumps({**{c: good for c in CLASSES}, "BCC": {"texts": good["texts"] * 3}}), encoding="utf-8")
        status, note = cp.step_references(str(self.data))
        self.assertEqual(status, cp.DONE)
        self.assertEqual(note, "10 reference texts (MEL 1, NV 1, BCC 3, AK 1, BKL 1, DF 1, VASC 1, SCC 1)")
        path.write_text(json.dumps({**{c: good for c in CLASSES}, "NV": {"texts": []}}), encoding="utf-8")
        self.assertEqual(cp.step_references(str(self.data)), (cp.PROBLEM, "no text: ['NV']"))
        path.write_text(json.dumps({c: good for c in CLASSES if c != "DF"}), encoding="utf-8")
        self.assertEqual(cp.step_references(str(self.data)), (cp.PROBLEM, "missing ['DF']"))
        path.write_text(json.dumps({c: {"text": "x", "n_words": 120} for c in CLASSES}), encoding="utf-8")
        status, note = cp.step_references(str(self.data))
        self.assertEqual(status, cp.PROBLEM)
        self.assertIn("old form", note)

    def test_an_unbalanced_test_split_is_a_problem(self):
        cases = self.data / PROCESSED_CASES
        cases.mkdir(parents=True)
        for f in cp.PROCESSED_FILES:
            (cases / f).write_text("isic_id\n", encoding="utf-8")
        rows = "".join(f"T_{c}_{k},{c},test\n" for c in CLASSES for k in range(31 if c == "DF" else 30))
        (self.data / ALL_CASES).write_text("isic_id,label,split\n" + rows, encoding="utf-8")
        status, note = cp.step_clean(str(self.data))
        self.assertEqual(status, cp.PROBLEM)
        self.assertIn("not balanced to 30 per class", note)
        self.assertIn("DF 31", note)

    def test_missing_and_left_over_images_are_named(self):
        cases = self.data / PROCESSED_CASES
        cases.mkdir(parents=True)
        for f in cp.PROCESSED_FILES:
            (cases / f).write_text("isic_id,image_path\n", encoding="utf-8")
        test_rows = "".join(f"T_{c}_{k},{c},test,processed_images/T_{c}_{k}.jpg\n" for c in CLASSES for k in range(30))
        (self.data / ALL_CASES).write_text("isic_id,label,split,image_path\nISIC_1,MEL,train,processed_images/ISIC_1.jpg\n"
                                           "ISIC_2,NV,val,processed_images/ISIC_2.jpg\n" + test_rows, encoding="utf-8")
        images = self.data / "processed_images"
        images.mkdir()
        for name in ["ISIC_1.jpg", "ISIC_2.jpg"] + [f"T_{c}_{k}.jpg" for c in CLASSES for k in range(30)]:
            (images / name).write_bytes(b"")
        self.assertEqual(cp.step_clean(str(self.data)), (cp.DONE, "242 cases, 30 test images per class, every image present"))
        (images / "ISIC_9.jpg").write_bytes(b"")
        (images / "notes.txt").write_text("not an image", encoding="utf-8")
        status, note = cp.step_clean(str(self.data))
        self.assertEqual(status, cp.PROBLEM)
        self.assertIn("1 images not in the case table", note)
        self.assertIn("ISIC_9.jpg", note)
        self.assertNotIn("missing", note)
        (images / "ISIC_2.jpg").unlink()
        status, note = cp.step_clean(str(self.data))
        self.assertIn("1 images missing", note)
        self.assertIn("ISIC_2", note)
        self.assertIn("ISIC_9.jpg", note)

    @unittest.skipUnless((DATA / ALL_CASES).is_file(), "data/processed_cases/all_cases.csv is not there")
    def test_views_that_differ_from_a_fresh_build_are_a_problem(self):
        shutil.copytree(DATA / PROCESSED_CASES, self.data / PROCESSED_CASES)
        shutil.copytree(DATA / "views", self.data / "views")
        self.assertEqual(cp.step_views(str(self.data))[0], cp.DONE)
        with open(self.data / question_file("text_only", "test"), "a", encoding="utf-8") as f:
            f.write("\n")
        self.assertEqual(cp.step_views(str(self.data))[0], cp.PROBLEM)

    def run_folder(self, partial=False, sha="x", report=False):
        folder = self.results / "llm_only" / "gpt-4o" / ("text_only_test" + (".partial" if partial else ""))
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps({"settings": {"questions_sha256": sha}, "status": "stopped",
                                                     "answered": 3, "of": 315}), encoding="utf-8")
        if report:
            (folder / "report.json").write_text("{}", encoding="utf-8")
        return folder

    def runs(self):
        return {n: (s, note) for n, s, note in cp.step_runs(str(self.data), str(self.results), ["gpt-4o"])}

    def write_questions(self):
        q = self.data / question_file("text_only", "test")
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text('{"id": "a"}\n', encoding="utf-8")
        return cp.sha256(q)

    def test_an_unfinished_run_is_a_problem_with_its_progress(self):
        self.run_folder(partial=True)
        status, note = self.runs()["llm_only gpt-4o text_only test"]
        self.assertEqual(status, cp.PROBLEM)
        self.assertIn("3/315", note)

    def test_a_run_on_old_questions_or_without_a_report_is_a_problem(self):
        sha = self.write_questions()
        folder = self.run_folder(sha="old")
        self.assertIn("question file changed", self.runs()["llm_only gpt-4o text_only test"][1])
        shutil.rmtree(folder)
        folder = self.run_folder(sha=sha)
        self.assertIn("not evaluated", self.runs()["llm_only gpt-4o text_only test"][1])
        (folder / "report.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.runs()["llm_only gpt-4o text_only test"][0], cp.DONE)

    def test_the_exit_code_is_1_only_when_a_step_has_a_problem(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cp.main(["--data", str(self.data), "--results", str(self.results)]), 0)
            self.run_folder(partial=True)
            self.assertEqual(cp.main(["--data", str(self.data), "--results", str(self.results)]), 1)


# ----------------------------------------------------------------------------------------------------------------------
# the corpus builder with fake downloads (build.py corpus)
# ----------------------------------------------------------------------------------------------------------------------


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
        if url == bd.PMC_OA:
            if params["id"] == "PMC12":
                return '<OA><error code="idIsNotOpenAccess">not open access</error></OA>'
            return f'<OA><records><record id="{params["id"]}" license="CC BY" retracted="no"/></records></OA>'
        if "BioC_json" in url:
            return json.dumps(BIOC_AK if "PMC13" in url else BIOC)
        if url == bd.DERMOSCOPEDIA_APIS[0]:
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
    base = dict(libraries=list(bd.LIBRARIES), per_class=50, max_words=200, min_words=5, pubmed_docs=10, pmc_docs=5,
                guideline_docs=5, textbook_pages=3)
    base.update(kw)
    return argparse.Namespace(**base)


class TextTests(unittest.TestCase):
    def test_passages_are_whole_sentences_within_the_word_limit_with_one_sentence_overlap(self):
        text = "One two three four. Five six seven eight. Nine ten eleven twelve. Thirteen fourteen."
        parts = bd.chunk(text, max_words=8)
        self.assertEqual(parts, ["One two three four. Five six seven eight.", "Five six seven eight. Nine ten eleven twelve.",
                                 "Nine ten eleven twelve. Thirteen fourteen."])
        self.assertTrue(all(len(p.split()) <= 8 for p in parts))

    def test_a_sentence_longer_than_the_limit_is_cut_and_nothing_is_lost(self):
        words = [f"w{i}" for i in range(25)]
        parts = bd.chunk(" ".join(words), max_words=10)
        self.assertEqual([len(p.split()) for p in parts], [10, 10, 5])
        self.assertEqual(" ".join(parts).split(), words)

    def test_a_short_text_is_one_passage_and_empty_text_is_none(self):
        self.assertEqual(bd.chunk("Dermatofibroma is benign.", 200), ["Dermatofibroma is benign."])
        self.assertEqual(bd.chunk("   ", 200), [])

    def test_tags_name_the_diagnoses_a_passage_talks_about(self):
        self.assertEqual(bd.tag("A seborrhoeic keratosis can mimic melanoma."), ["BKL", "MEL"])
        self.assertEqual(bd.tag("Bowen's disease is squamous cell carcinoma in situ."), ["SCC"])
        self.assertEqual(bd.tag("arborizing vessels", title="Basal cell carcinoma"), ["BCC"])
        self.assertEqual(bd.tag("nevirapine is a drug"), [])    # whole words only ("nevi" is inside "nevirapine")

    def test_ai_and_image_dataset_papers_are_recognised(self):
        self.assertTrue(bd.is_ai_paper("Deep learning for skin cancer", ""))
        self.assertTrue(bd.is_ai_paper("A study", "We trained on HAM10000 images."))
        self.assertFalse(bd.is_ai_paper("Dermoscopy of basal cell carcinoma", LONG))

    def test_only_creative_commons_licences_count_as_open(self):
        for licence in ("CC BY", "CC0", "CC BY-NC-ND", "cc by-sa"):
            self.assertTrue(bd.open_licence(licence), licence)
        for licence in ("NO-CC CODE", "", None, "CC BY-XYZ"):
            self.assertFalse(bd.open_licence(licence), licence)

    def test_topics_of_a_passage(self):
        self.assertEqual(bd.topics(LONG), ["appearance", "basics", "differential", "patients"])
        self.assertEqual(bd.topics("Nothing here."), [])

    def test_wiki_html_keeps_paragraphs_and_list_items_only(self):
        paras = bd.html_paragraphs(WIKI_PAGE + "<ul><li>first<br/>line</li></ul>")
        self.assertEqual(len(paras), 2)
        self.assertNotIn("[1]", paras[0])
        self.assertNotIn("table text", " ".join(paras))
        self.assertEqual(paras[1], "first line")


class ParserTests(unittest.TestCase):
    def test_pubmed_xml(self):
        docs = bd.parse_pubmed_xml(PUBMED_XML)
        self.assertEqual([d["doc_id"] for d in docs], ["PMID:101", "PMID:102", "PMID:103", "PMID:104"])
        self.assertEqual((docs[0]["year"], docs[0]["url"]), ("2020", "https://pubmed.ncbi.nlm.nih.gov/101/"))
        self.assertEqual([d["retracted"] for d in docs], [False, False, True, False])

    def test_pmc_oa_answer(self):
        self.assertEqual(bd.parse_oa_record('<OA><records><record id="PMC1" license="CC BY-NC" retracted="yes"/></records></OA>'),
                         ("CC BY-NC", True))
        self.assertEqual(bd.parse_oa_record('<OA><error code="idIsNotOpenAccess">x</error></OA>'), (None, None))

    def test_bioc_keeps_text_sections_and_drops_references_and_tables(self):
        for answer in (BIOC, BIOC[0]):                        # a list of collections, or one collection
            title, year, paras = bd.parse_bioc(answer)
            self.assertEqual((title, year, len(paras)), ("Basal cell carcinoma: a review", "2021", 2))
            self.assertFalse([p for p in paras if "reference" in p or "table" in p])

    def test_meditron_guidelines_without_wikidoc_nice_and_unrelated_topics(self):
        removed = bd.Counter()
        docs = list(bd.meditron_guideline_docs(removed, MEDITRON))
        self.assertEqual([d["doc_id"] for d in docs], ["meditron:g1"])
        self.assertEqual(len(docs[0]["paragraphs"]), 3)
        self.assertEqual(removed, {"guidelines: source wikidoc excluded": 1, "guidelines: source nice excluded": 1})


class BuildTests(unittest.TestCase):
    def build(self, **kw):
        fetch = FakeFetcher()
        with contextlib.redirect_stdout(io.StringIO()):
            passages, removed = bd.build_corpus(args(**kw), fetch, meditron_rows=MEDITRON)
        return passages, removed, fetch

    def test_every_library_contributes_and_every_passage_can_be_cited(self):
        passages, removed, _ = self.build()
        self.assertEqual({p["library"] for p in passages}, set(bd.LIBRARIES))
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
        counts = bd.Counter((p["library"], p["query_class"]) for p in passages)
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
        self.assertNotIn("--email", " ".join(bd.corpus_args([]).__dict__))

    def test_the_card_counts_and_the_manifest_has_no_text(self):
        passages, removed, _ = self.build()
        card = bd.corpus_card(passages, removed, args(out="x", cache="y"), "2026-11-22")
        self.assertEqual(card["n_passages"], len(passages))
        self.assertEqual(sum(sum(v.values()) for v in card["passages_per_library_and_class"].values()), len(passages))
        self.assertIn("VASC/appearance", card["empty_topic_boxes"])   # no made-up text is about vascular lesions
        self.assertNotIn("out", card["settings"])
        with tempfile.TemporaryDirectory() as d:
            bd.write_outputs(passages, card, d)
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
            passages, removed = bd.build_corpus(args(libraries=["guidelines"]), FakeFetcher(), meditron_rows=Broken())
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
            f = bd.Fetcher(d)
            self.assertEqual(f.get(bd.NCBI + "esearch.fcgi", {"db": "pubmed"}, ncbi=True), '{"ok": 1}')
            self.assertEqual(f.get(bd.NCBI + "esearch.fcgi", {"db": "pubmed"}, ncbi=True), '{"ok": 1}')
            stored = [Path(d, n).read_text(encoding="utf-8") + n for n in os.listdir(d)]
        self.assertEqual(len(seen), 1)                        # the second answer came from the cache
        self.assertIn("api_key=SECRET123", seen[0])
        self.assertIn("tool=" + bd.TOOL, seen[0])
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
                bd.Fetcher(d).get(bd.NCBI + "efetch.fcgi", {"id": "1"}, ncbi=True)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch("urllib.request.urlopen", no_internet):
                self.assertEqual(bd.Fetcher(d).get(bd.NCBI + "efetch.fcgi", {"id": "1"}, ncbi=True), "cached answer")


# ----------------------------------------------------------------------------------------------------------------------
# the reference-text builder with a fake wiki (build.py references)
# ----------------------------------------------------------------------------------------------------------------------


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
            body = p["texts"].get(heading, f"<p>{params['page']} / {bd.clean_heading(heading)} text. {WORDS}</p>")
            html = f"<h2>{heading}</h2><p>{params['page']} – cite!</p>{body}<table><tr><td><p>table text</p></td></tr></table>"
            sub = p["sections"][p["sections"].index(next(s for s in p["sections"] if s["index"] == index)) + 1:]
            if p["texts"].get("nested") == heading and sub:      # the wiki returns a section with its subsections
                html += f"<h3>{sub[0]['line']}</h3><p>{params['page']} / {sub[0]['line']} text. {WORDS}</p>"
        out = {"text": html}
        if "links" in params["prop"]:
            out["links"] = [{"ns": 0, "title": t} for t in p["links"]]
        return json.dumps({"parse": out})


def build_references(max_words=200, min_words=20):
    with tempfile.TemporaryDirectory() as d, mock.patch.object(bd, "Fetcher", FakeWiki), \
            contextlib.redirect_stdout(io.StringIO()):
        out = os.path.join(d, "reference_texts.json")
        bd.references_main(["--out", out, "--cache", os.path.join(d, "cache"), "--max-words", str(max_words), "--min-words", str(min_words)])
        return json.loads(Path(out).read_text(encoding="utf-8"))


class ReferenceRuleTests(unittest.TestCase):
    def test_the_first_term_with_results_decides_and_an_equal_title_wins_within_it(self):
        self.assertEqual(bd.choose_page([("nevus", ["Nevi of the face", "Nevus"])]), ("Nevus", "title equals 'nevus'"))
        self.assertEqual(bd.choose_page([("a", []), ("b", ["B page", "Other"])]), ("B page", "first result for 'b'"))
        self.assertEqual(bd.choose_page([("a", ["A page"]), ("b", ["b"])]), ("A page", "first result for 'a'"))
        self.assertEqual(bd.choose_page([("a", [])]), (None, "no page found"))

    def test_sections_from_the_dermoscopy_heading_else_all_else_the_introduction(self):
        sections = [{"index": "1", "line": "History"}, {"index": "2", "line": "<b>Dermatoscopic</b> features"},
                    {"index": "3", "line": "Milia-like cysts"}]
        self.assertEqual(bd.choose_sections(sections), (["2", "3"], "from 'Dermatoscopic features'"))
        self.assertEqual(bd.choose_sections([{"index": "1", "line": "Large nests:"}])[0], ["1"])
        self.assertEqual(bd.choose_sections([]), ([0], "(introduction)"))

    def test_headings_and_template_lines_are_cleaned(self):
        self.assertEqual(bd.clean_heading("Dermoscopy features &#91;1&#93; &#91;2&#93;"), "Dermoscopy features")
        self.assertEqual(bd.text_paragraphs("<p>Nevus – cite!</p><p>Nevus (message)</p><p>A nevus is benign.</p>"),
                         ["A nevus is benign."])

    def test_texts_of_whole_sentences_up_to_the_word_limit_and_short_ones_left_out(self):
        paragraphs = ["One two three. Four five six seven. Eight nine."]
        self.assertEqual(bd.split_texts(paragraphs, 7, 1), (["One two three. Four five six seven.", "Eight nine."], []))
        self.assertEqual(bd.split_texts(paragraphs, 7, 3), (["One two three. Four five six seven."], [2]))
        self.assertEqual(bd.split_texts(["One two three four five."], 2, 1)[0][0], "One two")   # a too long sentence is cut
        self.assertEqual(bd.split_texts([], 10, 1), ([], []))

    def test_a_section_is_read_without_its_subsections(self):
        section = "<h2>Criteria</h2><p>own</p><h3>Sub</h3><p>sub</p>"
        self.assertEqual(bd.own_html(section), "<h2>Criteria</h2><p>own</p>")
        self.assertEqual(bd.own_html("<p>no heading</p>"), "<p>no heading</p>")

    def test_sections_not_about_the_disease_are_left_out(self):
        sections = [{"index": "1", "line": "Dermoscopy"}, {"index": "2", "line": "Differential diagnosis"},
                    {"index": "3", "line": "Vessels"}, {"index": "4", "line": "References"}, {"index": "5", "line": "See also"}]
        self.assertEqual(bd.choose_sections(sections)[0], ["1", "3"])

    def test_a_reference_list_is_cut_from_its_first_entry_and_a_citation_in_a_sentence_is_kept(self):
        text = ["Pigmented SCC shows blue structures (Kittler et al., 2008). Vessels are not seen.",
                "Argenziano, G., Zalaudek, I., 2004, Vascular structures. Arch Dermatol, 140, 1485–9.",
                "Kreusch, J. & Koch, F., 1996, Gefassmuster. Hautarzt, 47, 264–72."]
        self.assertEqual(bd.without_bibliography(text),
                         (["Pigmented SCC shows blue structures (Kittler et al., 2008). Vessels are not seen."], 21))
        self.assertEqual(bd.without_bibliography(["An Atlas of Dermoscopy, 2nd edition. Text after it."]), ([], 9))
        self.assertEqual(bd.without_bibliography(["Plain text. No list."]), (["Plain text. No list."], 0))
        self.assertTrue(bd.BIBLIOGRAPHY.search("Kreusch, J.F., 2002, Vascular patterns."))
        self.assertTrue(bd.BIBLIOGRAPHY.search("Kreusch, J. & Koch, F., 1996, Gefassmuster."))
        self.assertFalse(bd.BIBLIOGRAPHY.search("Types A. & B. are rare."))           # an author needs the comma
        self.assertFalse(bd.BIBLIOGRAPHY.search("Erythema: Structureless pale-red areas, as in AK."))
        self.assertFalse(bd.BIBLIOGRAPHY.search("Blue nevi, also known as Jadassohn-Tieche nevi, are common."))

    def test_the_text_of_each_table_of_a_section_without_its_subsections(self):
        section = ("<h2>Criteria</h2><p>prose</p><table><tr><td>Atypical network</td><td>Blue-white veil</td></tr>"
                   "<tr><td><table><tr><td>nested</td></tr></table></td></tr></table><p>more</p>"
                   "<table><tr><th>Regression</th></tr></table><h3>Sub</h3><table><tr><td>sub table</td></tr></table>")
        self.assertEqual(bd.table_texts(section), ["Atypical network Blue-white veil nested", "Regression"])
        self.assertEqual(bd.table_texts("<p>no table</p>"), [])

    def test_the_text_left_out_other_than_tables(self):
        section = ('<h2><span class="mw-headline">Criteria</span><span class="mw-editsection">edit</span></h2>'
                   '<p>read <b>bold</b></p><ul><li>item</li></ul><table><tr><td>table</td></tr></table>'
                   '<div class="thumb"><img alt="Atypical network" src="x.jpg"><div class="thumbcaption">Blue-white veil'
                   '<br>and regression</div></div><dl><dd>Irregular dots</dd></dl>loose text'
                   '<h3>Sub</h3><div>sub div</div>')
        self.assertEqual(bd.skipped_texts(section), [("img alt", "Atypical network"),
                                                     ("div.thumbcaption", "Blue-white veil and regression"),
                                                     ("dd", "Irregular dots"), ("(no element)", "loose text")])
        self.assertEqual(bd.skipped_texts("<p>only a paragraph</p>"), [])
        self.assertEqual(bd.skipped_texts('<div class="x">caption<span class="mw-editsection">[edit]</span></div>before'
                                          '<div>after</div>'), [("div.x", "caption"), ("(no element)", "before"), ("div", "after")])

    def test_show_skipped_prints_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(bd, "Fetcher", FakeWiki), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(bd.references_main(["--show-skipped", "--cache", os.path.join(d, "cache"), "--out", os.path.join(d, "r.json")]))
            self.assertFalse(os.path.exists(os.path.join(d, "r.json")))
        self.assertIn("===== MEL: Superficial Spreading Melanoma / from 'Dermoscopic criteria': 0 pieces of text left out",
                      out.getvalue())

    def test_show_tables_prints_the_tables_of_the_chosen_sections_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(bd, "Fetcher", FakeWiki), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(bd.references_main(["--show-tables", "--cache", os.path.join(d, "cache"), "--out", os.path.join(d, "r.json")]))
            self.assertFalse(os.path.exists(os.path.join(d, "r.json")))
        text = out.getvalue()
        self.assertIn("===== MEL: Superficial Spreading Melanoma / from 'Dermoscopic criteria': 1 tables", text)
        self.assertIn("[Dermoscopic criteria, 2 words] table text", text)
        self.assertIn("===== VASC: Angioma / (introduction): 0 tables", text)

    def test_empty_figure_references_are_removed(self):
        self.assertEqual(bd.text_paragraphs("<p>Blue dots (Figure ). This is BCC. . Then (see fig. ) more.</p>"),
                         ["Blue dots. This is BCC. Then more."])


class ReferenceBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs = build_references()

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
        refs = build_references(max_words=20, min_words=1)
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
            longest, limit = bd.token_check({"MEL": {"texts": [{"text": "a b c"}, {"text": " ".join("x" * 13)}]}},
                                            bd.get_logger("reference_texts"))
        self.assertEqual((longest, limit), (13, 12))
        self.assertIn("1 texts are longer", out.getvalue())
        self.assertIn("rebuild with a lower --max-words", out.getvalue())
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake}), contextlib.redirect_stdout(io.StringIO()) as out:
            bd.token_check({"MEL": {"texts": [{"text": "a b c"}]}}, bd.get_logger("reference_texts"))
        self.assertNotIn("rebuild", out.getvalue())


# ----------------------------------------------------------------------------------------------------------------------
# a whole run of run.py with a fake model, read by evaluate.py
# ----------------------------------------------------------------------------------------------------------------------


def run_cases(n, view_has_image):
    labels = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    return [{"isic_id": f"ISIC_{i:07d}", "split": "val", "label": labels[i % 8], "group_id": f"ISIC_{i:07d}",
             "clinical_text": "Patient: 50-year-old male. Skin lesion on the trunk.", "image_path": f"processed_images/ISIC_{i:07d}.jpg"}
            for i in range(n)]


class FakeClient:
    """Answers right for every second question, wrong for the others, with no letter for the third, and can fail on demand."""
    def __init__(self, questions, fail_at=None, refuse_at=None):
        self.by_text = {rl.user_text(q): q for q in questions}
        self.calls, self.images, self.fail_at, self.refuse_at = 0, [], fail_at, refuse_at

    def ask(self, system, text, image):
        self.calls += 1
        if self.fail_at and self.calls == self.fail_at:
            raise rl.ApiError("HTTP 402: no credit")
        if self.refuse_at and self.calls == self.refuse_at:
            raise rl.Refused("refused by the host's content moderation")
        self.images.append(image)
        q = self.by_text[text]
        k = int(q["id"][-3:])
        if k == 2:
            return {"text": "I cannot help with that.", "finish_reason": "refusal", "tokens_in": 10, "tokens_out": 5}
        letter = q["answer"] if k % 2 == 0 else next(x for x in q["options"] if x != q["answer"])
        others = [x for x in q["options"] if x not in (letter, q["answer"])][:2]
        if k == 1:
            others = [q["answer"], others[0]]        # wrong first, right second: right in the top 3
        flag = "yes" if k % 3 else "no"
        return {"text": f"Some reasoning.\nAnswer: {letter}, {others[0]}, {others[1]}\nEvidence sufficient: {flag}",
                "finish_reason": "stop", "tokens_in": 100, "tokens_out": 20}


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data, self.results = os.path.join(self.tmp.name, "data"), os.path.join(self.tmp.name, "results")
        self.make("text_only", 6)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, view, n, images=True):
        self.questions = [question(c, view) for c in run_cases(n, view != "text_only")]
        write_jsonl(os.path.join(self.data, question_file(view, "val")), self.questions)
        if images and view != "text_only":
            for q in self.questions:
                Path(self.data, q["image_path"]).parent.mkdir(parents=True, exist_ok=True)
                Path(self.data, q["image_path"]).write_bytes(b"\xff\xd8\xff fake jpeg")

    def args(self, view="text_only", **kw):
        base = dict(model="gpt-4o", view=view, split="val", style="rationale", limit=0, sleep=0.0, dry_run=False,
                    env_file=os.path.join(self.tmp.name, ".env"), data=self.data, results=self.results)
        base.update(kw)
        return argparse.Namespace(**base)

    def go(self, args, client):
        log = rl.get_logger(f"test.{id(self)}.{args.view}")
        with contextlib.redirect_stdout(io.StringIO()):               # keep the test output quiet
            return rl.run(args, client, log)

    def folder(self, view="text_only", partial=False):
        final = os.path.join(self.results, "llm_only", "gpt-4o", f"{view}_val")
        return final + ".partial" if partial else final

    def test_a_full_run_saves_every_answer_and_ends_complete(self):
        out = self.go(self.args(), FakeClient(self.questions))
        self.assertEqual(out, self.folder())
        self.assertFalse(os.path.exists(self.folder(partial=True)))
        preds = rl.read_jsonl(os.path.join(out, "predictions.jsonl"))
        self.assertEqual([p["id"] for p in preds], [q["id"] for q in rl.asking_order(self.questions)])
        self.assertEqual(sorted(p["id"] for p in preds), sorted(q["id"] for q in self.questions))
        info = json.loads(Path(out, "run.json").read_text(encoding="utf-8"))
        self.assertEqual((info["status"], info["answered"], info["of"]), ("complete", 6, 6))
        self.assertEqual(info["settings"]["prompt_version"], rl.PROMPT_VERSION)
        self.assertIn("42:<case id>", info["question_order"])
        refused = next(p for p in preds if p["id"].endswith("002"))
        self.assertEqual((refused["pred"], refused["valid"], refused["finish_reason"]), (None, False, "refusal"))
        first = preds[0]
        self.assertEqual((len(first["top3"]), first["pred"], first["evidence_sufficient"]), (3, first["top3"][0], "no"))

    def test_the_predictions_go_straight_into_evaluate(self):
        out = self.go(self.args(), FakeClient(self.questions))
        qa = {q["id"]: q for q in self.questions}
        pred = ev.load(os.path.join(out, "predictions.jsonl"))
        d = ev.diagnosis_metrics(qa, pred)
        self.assertEqual((d["accuracy"], d["invalid_answers"]), (0.3333, 1))    # ids 0 and 4 right; 2 has no letter
        self.assertEqual(d["top3_accuracy"], 0.5)                                  # ids 0, 4 first; id 1 second
        self.assertEqual((d["n_covered"], d["coverage"]), (3, 0.5))                # ids 1, 4, 5 say yes (k % 3); 2 is invalid
        self.assertEqual(d["selective_risk"], 0.6667)                              # ids 1 and 5 are wrong

    def test_the_asking_order_is_fixed_mixed_and_the_same_for_every_view(self):
        qs = [{"id": f"ISIC_{k:07d}", "gold_label": c} for k, c in enumerate(["AK"] * 10 + ["BCC"] * 10 + ["MEL"] * 10)]
        order = [q["id"] for q in rl.asking_order(qs)]
        self.assertEqual(order, [q["id"] for q in rl.asking_order(list(reversed(qs)))])    # the file order does not matter
        self.assertEqual(sorted(order), [q["id"] for q in qs])                              # every question once
        self.assertNotEqual(order, [q["id"] for q in qs])
        first = {q["gold_label"] for q in rl.asking_order(qs)[:5]}
        self.assertGreater(len(first), 1)                                                   # a trial of 5 is not one class
        self.assertEqual(order[:3], ["ISIC_0000003", "ISIC_0000026", "ISIC_0000015"])        # fixed: seed 42

    def test_a_limited_run_asks_the_first_questions_of_the_asking_order(self):
        self.go(self.args(limit=2), FakeClient(self.questions))
        preds = rl.read_jsonl(os.path.join(self.folder(partial=True), "predictions.jsonl"))
        self.assertEqual([p["id"] for p in preds], [q["id"] for q in rl.asking_order(self.questions)[:2]])
        info = json.loads(Path(self.folder(partial=True), "run.json").read_text(encoding="utf-8"))
        self.assertEqual(info["question_order"], rl.ORDER)

    def test_an_interrupted_run_continues_where_it_stopped_without_repeating(self):
        self.go(self.args(limit=2), FakeClient(self.questions))
        self.assertTrue(os.path.isdir(self.folder(partial=True)))
        info = json.loads(Path(self.folder(partial=True), "run.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status"], "unfinished")
        client = FakeClient(self.questions)
        self.go(self.args(), client)
        self.assertEqual(client.calls, 4)                                       # only the 4 left
        preds = rl.read_jsonl(os.path.join(self.folder(), "predictions.jsonl"))
        self.assertEqual(len(preds), 6)
        self.assertEqual(len({p["id"] for p in preds}), 6)

    def test_an_api_error_stops_the_run_and_keeps_what_was_answered(self):
        with self.assertRaises(SystemExit):
            self.go(self.args(), FakeClient(self.questions, fail_at=4))
        preds = rl.read_jsonl(os.path.join(self.folder(partial=True), "predictions.jsonl"))
        self.assertEqual(len(preds), 3)
        info = json.loads(Path(self.folder(partial=True), "run.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status"], "stopped")
        self.go(self.args(), FakeClient(self.questions))
        self.assertEqual(len(rl.read_jsonl(os.path.join(self.folder(), "predictions.jsonl"))), 6)

    def test_an_unfinished_run_is_not_continued_with_other_settings(self):
        self.go(self.args(limit=2), FakeClient(self.questions))
        with self.assertRaises(SystemExit):
            self.go(self.args(style="answer_only"), FakeClient(self.questions))

    def test_a_question_refused_by_the_host_is_saved_as_invalid_and_the_run_goes_on(self):
        self.go(self.args(), FakeClient(self.questions, refuse_at=2))
        preds = rl.read_jsonl(os.path.join(self.folder(), "predictions.jsonl"))
        self.assertEqual(len(preds), 6)
        refused = [p for p in preds if str(p["finish_reason"]).startswith("refused by host")]
        self.assertEqual(len(refused), 1)
        self.assertEqual((refused[0]["pred"], refused[0]["valid"]), (None, False))

    def test_a_complete_run_is_not_asked_again(self):
        self.go(self.args(), FakeClient(self.questions))
        client = FakeClient(self.questions)
        self.go(self.args(), client)
        self.assertEqual(client.calls, 0)

    def test_image_views_send_the_image_and_the_text_view_does_not(self):
        client = FakeClient(self.questions)
        self.go(self.args(), client)
        self.assertEqual(set(client.images), {None})
        self.make("multimodal", 3)
        client = FakeClient(self.questions)
        self.go(self.args(view="multimodal"), client)
        self.assertTrue(all(img and img.startswith("/9j") for img in client.images))   # base64 of a JPEG start

    def test_missing_images_stop_the_run_before_any_question(self):
        self.make("image_only", 3, images=False)
        client = FakeClient(self.questions)
        with self.assertRaises(SystemExit):
            self.go(self.args(view="image_only"), client)
        self.assertEqual(client.calls, 0)

    def test_a_dry_run_asks_nothing_and_writes_nothing(self):
        client = FakeClient(self.questions)
        self.go(self.args(dry_run=True), client)
        self.assertEqual(client.calls, 0)
        self.assertFalse(os.path.exists(self.results))

    def test_the_api_key_is_needed_and_never_written(self):
        argv = ["--model", "gpt-4o", "--view", "text_only", "--data", self.data, "--results", self.results,
                "--env-file", os.path.join(self.tmp.name, ".env")]
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit), \
                contextlib.redirect_stdout(io.StringIO()):
            rl.llm_only_main(argv)
        self.assertFalse(os.path.exists(self.results))                  # nothing is left behind
        Path(self.tmp.name, ".env").write_text("# keys\nOPENROUTER_API_KEY = 'SECRET123'\n", encoding="utf-8")
        fake = FakeClient(self.questions)
        made = []

        def client(cfg, key, log=None):
            made.append(key)
            return fake

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(rl, "OpenRouterClient", client), \
                contextlib.redirect_stdout(io.StringIO()):
            rl.llm_only_main(argv)
        self.assertEqual(made, ["SECRET123"])                         # read from .env, quotes removed
        written = "".join(p.read_text(encoding="utf-8") for p in Path(self.results).rglob("*") if p.is_file())
        self.assertIn("gpt-4o", written)
        self.assertNotIn("SECRET123", written)

    def test_claude_takes_its_key_from_either_name(self):
        with mock.patch.dict(os.environ, {"CLAUDE_API_KEY": "K1"}, clear=True):
            self.assertEqual(rl.api_key("anthropic"), "K1")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "K2", "CLAUDE_API_KEY": "K1"}, clear=True):
            self.assertEqual(rl.api_key("anthropic"), "K2")


# ----------------------------------------------------------------------------------------------------------------------
# the evaluate.py command
# ----------------------------------------------------------------------------------------------------------------------


def make_question(case_id, gold):
    options, letter = options_for(case_id)
    return {"id": case_id, "options": options, "answer": letter[gold], "gold_label": gold, "group_id": case_id,
            "question": f"Question {case_id}"}


class CommandTests(unittest.TestCase):
    def test_the_command_writes_a_report(self):
        qs = [make_question(f"ISIC_{i:07d}", c) for i, c in enumerate(CLASSES)]
        with tempfile.TemporaryDirectory() as d:
            paths = {k: os.path.join(d, f"{k}.jsonl") for k in ("qa", "pred")}
            with open(paths["qa"], "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(q) for q in qs) + "\n\n")  # a blank last line is ignored
            with open(paths["pred"], "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps({"id": q["id"], "pred": q["answer"]}) for q in qs[:4]))
            out = os.path.join(d, "report.json")
            with contextlib.redirect_stdout(io.StringIO()):
                ev.main(["--questions", paths["qa"], "--predictions", paths["pred"], "--out", out, "--skip-text"])
            with open(out, encoding="utf-8") as f:
                rep = json.load(f)
            sha = hashlib.sha256(Path(paths["qa"]).read_bytes()).hexdigest()
        self.assertEqual((rep["diagnosis"]["accuracy"], rep["diagnosis"]["missing_predictions"]), (0.5, 4))
        self.assertIsNone(rep["retrieval"])
        self.assertEqual(rep["inputs"]["questions"]["sha256"], sha)  # the report says which question file it used


if __name__ == "__main__":
    unittest.main()
