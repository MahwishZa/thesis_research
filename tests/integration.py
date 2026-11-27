"""Integration tests: several parts working together, on made-up folders and made-up answers of the outside services (no network,
no API call, no image needed). The corpus builder with fake downloads, the reference-text builder with a fake wiki, a whole run of
the run of a system with a fake model (answers saved, interrupted, resumed, then evaluated), the evaluate command, and status on
small folders. Run: python -m unittest"""

import argparse
import collections
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# ruff: noqa: E402  (the imports below need the line above)
import src.evaluation.evaluator
import src.evaluation.metrics
import src.evaluation.references
import src.main
import src.models.llm
import src.pipelines.base
import src.pipelines.llm_only
import src.pipelines.standard_rag
import src.retrieval.corpus
import src.retrieval.images
import src.retrieval.indexer
import src.retrieval.retriever
import src.utils
from src.data.views import options_for, question, question_file
from src.retrieval.indexer import read_index as common_read_index
from src.retrieval.retriever import reciprocal_rank_fusion
from src.utils import (
    ALL_CASES,
    CLASSES,
    IMAGE_LIBRARY,
    INDEX_DIR,
    PROCESSED_CASES,
    PROCESSED_IMAGES,
    REFERENCE_TEXTS,
    TEXT_LIBRARY,
    read_jsonl,
    write_jsonl,
)


def fake_hash(path):
    """A stand-in for the picture hash in tests whose "images" are text files: four unrelated 64-bit numbers per file name."""
    name = os.path.basename(path)
    return [int(hashlib.md5(f"{name}{i}".encode()).hexdigest()[:16], 16) for i in range(4)]


DATA = ROOT / "data"


# ----------------------------------------------------------------------------------------------------------------------
# the status command on small made-up folders
# ----------------------------------------------------------------------------------------------------------------------


class LibraryCommandTests(unittest.TestCase):
    """The library command runs the build steps in order (here with fake steps) and stops at the first one that fails."""

    def run_with(self, argv, fail_at=None, code=None):
        calls = []

        def fake_command(name):
            def step(options):
                calls.append((name, options))
                if name == fail_at:
                    raise RuntimeError("boom")
                return code if name == fail_at else None
            return step
        with mock.patch.object(src.main, "command", fake_command), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return src.main.library_main(argv), calls

    def test_every_step_runs_in_order_with_the_approved_options(self):
        status, calls = self.run_with(["--extra-papers", "40", "--dermoscopedia", "10"])
        self.assertEqual(status, 0)
        self.assertEqual([c[0] for c in calls], ["corpus", "audit", "visual", "visual-images", "visual-index", "index"])
        self.assertEqual(dict(calls)["visual"], ["--extra-papers", "40", "--dermoscopedia", "10"])
        self.assertEqual(dict(calls)["index"], ["--exclude-reference-pages", "--exclude-reference-overlap"])
        self.assertEqual(dict(calls)["audit"], [])
        self.assertNotIn("visual-calibrate", [c[0] for c in calls])               # a report about the filter, not a build step
        self.assertEqual(dict(self.run_with(["--online"])[1])["audit"], ["--online"])

    def test_a_failing_step_stops_the_build_and_from_resumes_at_a_step(self):
        status, calls = self.run_with([], fail_at="visual")
        self.assertEqual((status, [c[0] for c in calls]), (1, ["corpus", "audit", "visual"]))
        status, calls = self.run_with(["--from", "visual-images"])
        self.assertEqual((status, [c[0] for c in calls]), (0, ["visual-images", "visual-index", "index"]))
        status, calls = self.run_with([], fail_at="audit", code=2)                  # a step that returns an error exit code (it also raised here)
        self.assertEqual((status, [c[0] for c in calls]), (1, ["corpus", "audit"]))

    def test_the_command_is_listed_and_every_step_of_it_is_a_command(self):
        self.assertIn("library", src.main.COMMANDS)
        self.assertTrue(set(src.main.LIBRARY_STEPS) <= set(src.main.COMMANDS))


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data, self.results = Path(self.tmp.name, "data"), Path(self.tmp.name, "results")

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_empty_workspace_is_all_not_run(self):
        rows = {name: status for name, status, _ in src.main.check(str(self.data), str(self.results))}
        self.assertEqual(set(rows.values()), {src.main.NOT_RUN})

    def test_reference_texts_old_form_or_empty_are_a_problem(self):
        path = self.data / REFERENCE_TEXTS
        path.parent.mkdir(parents=True)
        good = {"texts": [{"section": "Dermoscopic criteria", "text": "x", "n_words": 120}]}
        path.write_text(json.dumps({**{c: good for c in CLASSES}, "BCC": {"texts": good["texts"] * 3}}), encoding="utf-8")
        status, note = src.main.step_references(str(self.data))
        self.assertEqual(status, src.main.DONE)
        self.assertEqual(note, "10 reference texts (MEL 1, NV 1, BCC 3, AK 1, BKL 1, DF 1, VASC 1, SCC 1)")
        path.write_text(json.dumps({**{c: good for c in CLASSES}, "NV": {"texts": []}}), encoding="utf-8")
        self.assertEqual(src.main.step_references(str(self.data)), (src.main.PROBLEM, "no text: ['NV']"))
        path.write_text(json.dumps({c: good for c in CLASSES if c != "DF"}), encoding="utf-8")
        self.assertEqual(src.main.step_references(str(self.data)), (src.main.PROBLEM, "missing ['DF']"))
        path.write_text(json.dumps({c: {"text": "x", "n_words": 120} for c in CLASSES}), encoding="utf-8")
        status, note = src.main.step_references(str(self.data))
        self.assertEqual(status, src.main.PROBLEM)
        self.assertIn("old form", note)

    def test_an_unbalanced_test_split_is_a_problem(self):
        cases = self.data / PROCESSED_CASES
        cases.mkdir(parents=True)
        for f in src.main.PROCESSED_FILES:
            (cases / f).write_text("isic_id\n", encoding="utf-8")
        rows = "".join(f"T_{c}_{k},{c},test\n" for c in CLASSES for k in range(31 if c == "DF" else 30))
        (self.data / ALL_CASES).write_text("isic_id,label,split\n" + rows, encoding="utf-8")
        status, note = src.main.step_clean(str(self.data))
        self.assertEqual(status, src.main.PROBLEM)
        self.assertIn("not balanced to 30 per class", note)
        self.assertIn("DF 31", note)

    def test_an_unbalanced_validation_split_is_a_problem(self):
        cases = self.data / PROCESSED_CASES
        cases.mkdir(parents=True)
        for f in src.main.PROCESSED_FILES:
            (cases / f).write_text("isic_id\n", encoding="utf-8")
        rows = "".join(f"T_{c}_{k},{c},test\n" for c in CLASSES for k in range(30))
        rows += "".join(f"V_{c}_{k},{c},val\n" for c in CLASSES for k in range(19 if c == "AK" else 20))
        (self.data / ALL_CASES).write_text("isic_id,label,split\n" + rows, encoding="utf-8")
        status, note = src.main.step_clean(str(self.data))
        self.assertEqual(status, src.main.PROBLEM)
        self.assertIn("the val split is not balanced to 20 per class", note)
        self.assertIn("AK 19", note)

    def test_missing_and_left_over_images_are_named(self):
        cases = self.data / PROCESSED_CASES
        cases.mkdir(parents=True)
        for f in src.main.PROCESSED_FILES:
            (cases / f).write_text("isic_id,image_path\n", encoding="utf-8")
        test_rows = "".join(f"T_{c}_{k},{c},test,processed/images/T_{c}_{k}.jpg\n" for c in CLASSES for k in range(30))
        val_rows = "".join(f"V_{c}_{k},{c},val,processed/images/V_{c}_{k}.jpg\n" for c in CLASSES for k in range(20))
        (self.data / ALL_CASES).write_text("isic_id,label,split,image_path\nISIC_1,MEL,train,processed/images/ISIC_1.jpg\n"
                                           + val_rows + test_rows, encoding="utf-8")
        images = self.data / PROCESSED_IMAGES
        images.mkdir(parents=True)
        for name in (["ISIC_1.jpg"] + [f"T_{c}_{k}.jpg" for c in CLASSES for k in range(30)]
                     + [f"V_{c}_{k}.jpg" for c in CLASSES for k in range(20)]):
            (images / name).write_bytes(b"")
        self.assertEqual(src.main.step_clean(str(self.data)), (src.main.DONE, "401 cases, 30 test and 20 validation images per class, every image present"))
        (images / "ISIC_9.jpg").write_bytes(b"")
        (images / "notes.txt").write_text("not an image", encoding="utf-8")
        status, note = src.main.step_clean(str(self.data))
        self.assertEqual(status, src.main.PROBLEM)
        self.assertIn("1 images not in the case table", note)
        self.assertIn("ISIC_9.jpg", note)
        self.assertNotIn("missing", note)
        (images / "V_NV_3.jpg").unlink()
        status, note = src.main.step_clean(str(self.data))
        self.assertIn("1 images missing", note)
        self.assertIn("V_NV_3", note)
        self.assertIn("ISIC_9.jpg", note)

    @unittest.skipUnless((DATA / ALL_CASES).is_file(), "data/processed/cases/all_cases.csv is not there")
    def test_views_that_differ_from_a_fresh_build_are_a_problem(self):
        shutil.copytree(DATA / PROCESSED_CASES, self.data / PROCESSED_CASES)
        shutil.copytree(DATA / "views", self.data / "views")
        self.assertEqual(src.main.step_views(str(self.data))[0], src.main.DONE)
        with open(self.data / question_file("text_only", "test"), "a", encoding="utf-8") as f:
            f.write("\n")
        self.assertEqual(src.main.step_views(str(self.data))[0], src.main.PROBLEM)

    def run_folder(self, partial=False, sha="x", report=False):
        folder = self.results / "llm_only" / "text_only" / ("gpt-4o_test" + (".partial" if partial else ""))
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps({"settings": {"questions_sha256": sha}, "status": "stopped",
                                                     "answered": 3, "of": 315}), encoding="utf-8")
        if report:
            (folder / "report.json").write_text("{}", encoding="utf-8")
        return folder

    def runs(self):
        return {n: (s, note) for n, s, note in src.main.step_runs(str(self.data), str(self.results))}

    def write_questions(self):
        q = self.data / question_file("text_only", "test")
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(b'{"id": "a"}\n')                          # bytes: the same file on Windows and Linux
        return src.utils.text_sha256(q)

    def test_no_run_asks_for_a_model_that_was_never_run_and_every_run_found_is_listed(self):
        rows = self.runs()
        self.assertEqual(len(rows), 12)                                          # 4 systems x 3 views, each "no run yet"
        self.assertEqual(set(rows.values()), {(src.main.NOT_RUN, "no run yet")})
        self.assertIn("standard_rag text_only", rows)
        sha = self.write_questions()
        self.run_folder(sha=sha, report=True)                                    # one finished llm_only run (gpt-4o, test)
        other = self.results / "llm_only" / "text_only" / "some-model_val"       # another model and split: listed although unasked
        other.mkdir()
        (other / "run.json").write_text(json.dumps({"settings": {"questions_sha256": "x"}}), encoding="utf-8")
        trial = self.results / "standard_rag_dense" / "image_only" / "gpt-4o_val.partial"
        trial.mkdir(parents=True)
        (trial / "run.json").write_text(json.dumps({"status": "stopped", "answered": 1, "of": 20}), encoding="utf-8")
        rows = self.runs()
        self.assertEqual(rows["llm_only gpt-4o text_only test"][0], src.main.DONE)
        self.assertIn("llm_only some-model text_only val", rows)
        self.assertEqual(rows["llm_only some-model text_only val"][0], src.main.PROBLEM)    # its question file is not there
        self.assertNotIn("llm_only text_only", rows)                             # a view with a run needs no "no run yet" line
        self.assertEqual(rows["standard_rag_dense gpt-4o image_only val"][0], src.main.PROBLEM)
        self.assertNotIn("llm_only claude-sonnet-5-5 text_only test", rows)

    def write_retrieval_result(self, view, **kw):
        path = self.results / "standard_rag" / view / "retrieval_validation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"approved": "bm25-rerank", "libraries": {"text_index_passages_sha256": "T", "image_items_sha256": None}, **kw}),
                        encoding="utf-8")

    def test_a_retrieval_result_must_belong_to_the_libraries_that_exist_now(self):
        check = lambda: src.main.step_retrieval_checks(str(self.data), str(self.results))             # noqa: E731
        self.assertEqual(check()[0], src.main.NOT_RUN)                                                 # nothing yet
        index = self.data / src.utils.INDEX_DIR / "index.json"
        index.parent.mkdir(parents=True)
        index.write_text(json.dumps({"passages_sha256": "T"}), encoding="utf-8")
        for view in src.utils.VIEWS:
            self.write_retrieval_result(view)
        self.assertEqual(check()[0], src.main.DONE)
        index.write_text(json.dumps({"passages_sha256": "another"}), encoding="utf-8")                 # the text library changed since
        status, note = check()
        self.assertEqual(status, src.main.PROBLEM)
        self.assertIn("another text library", note)
        index.write_text(json.dumps({"passages_sha256": "T"}), encoding="utf-8")
        self.write_retrieval_result("multimodal", approved="bm25")                                      # made for another approved search
        self.assertEqual(check()[0], src.main.PROBLEM)
        self.write_retrieval_result("multimodal")
        path = self.results / "standard_rag" / "text_only" / "retrieval_validation.json"
        path.write_text("{}", encoding="utf-8")                                                         # made before libraries were recorded
        status, note = check()
        self.assertEqual(status, src.main.NOT_RUN)
        self.assertIn("run retrieval-eval again", note)

    def test_an_unfinished_run_is_a_problem_with_its_progress(self):
        self.run_folder(partial=True)
        status, note = self.runs()["llm_only gpt-4o text_only test"]
        self.assertEqual(status, src.main.PROBLEM)
        self.assertIn("3/315", note)

    def test_a_run_on_old_questions_or_without_a_report_is_a_problem(self):
        sha = self.write_questions()
        folder = self.run_folder(sha="old")
        self.assertIn("question file changed", self.runs()["llm_only gpt-4o text_only test"][1])
        shutil.rmtree(folder)
        folder = self.run_folder(sha=sha)
        self.assertIn("not evaluated", self.runs()["llm_only gpt-4o text_only test"][1])
        (folder / "report.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.runs()["llm_only gpt-4o text_only test"][0], src.main.DONE)

    def test_a_run_on_questions_that_only_moved_matches_through_the_provenance_record(self):
        sha = self.write_questions()
        folder = self.run_folder(sha="before-the-move", report=True)
        self.assertIn("question file changed", self.runs()["llm_only gpt-4o text_only test"][1])
        record = self.data / "views" / "provenance.json"
        name = question_file("text_only", "test").replace("\\", "/")
        record.write_text(json.dumps({"files": {name: {"previous_sha256": "before-the-move", "sha256": sha}}}), encoding="utf-8")
        self.assertEqual(self.runs()["llm_only gpt-4o text_only test"][0], src.main.DONE)
        record.write_text(json.dumps({"files": {name: {"previous_sha256": "before-the-move", "sha256": "another"}}}), encoding="utf-8")
        self.assertEqual(self.runs()["llm_only gpt-4o text_only test"][0], src.main.PROBLEM)    # the file changed after the move
        self.assertTrue(folder.is_dir())

    def test_the_exit_code_is_1_only_when_a_step_has_a_problem(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(src.main.main(["status", "--data", str(self.data), "--results", str(self.results)]), 0)
            self.run_folder(partial=True)
            self.assertEqual(src.main.main(["status", "--data", str(self.data), "--results", str(self.results)]), 1)


# ----------------------------------------------------------------------------------------------------------------------
# the corpus builder with fake downloads (the corpus command)
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


CC_BY = ('<license license-type="open-access" xlink:href="https://creativecommons.org/licenses/by/4.0/">'
         '<license-p>This is an open access article under the CC BY license.</license-p></license>')
JATS = ('<pmc-articleset xmlns:xlink="http://www.w3.org/1999/xlink"><article article-type="{article_type}"><front><article-meta>'
        '<permissions><copyright-statement>(c) the authors</copyright-statement>{licence}</permissions>{related}'
        '</article-meta></front></article></pmc-articleset>')


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
        if url.endswith("efetch.fcgi") and params["db"] == "pmc":
            if params["id"] == "12":                          # a paper with no Creative Commons licence
                return JATS.format(licence="", article_type="research-article", related="")
            return JATS.format(licence=CC_BY, article_type="research-article", related="")
        if url.endswith("efetch.fcgi"):
            return PUBMED_XML
        if "BioC_json" in url:
            return json.dumps(BIOC_AK if "PMC13" in url else BIOC)
        if url == src.retrieval.corpus.DERMOSCOPEDIA_APIS[0]:
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
    base = dict(libraries=list(src.retrieval.corpus.LIBRARIES), per_class=50, max_words=200, min_words=5, pubmed_docs=10, pmc_docs=5,
                guideline_docs=5, textbook_pages=3)
    base.update(kw)
    return argparse.Namespace(**base)


class TextTests(unittest.TestCase):
    def test_passages_are_whole_sentences_within_the_word_limit_with_one_sentence_overlap(self):
        text = "One two three four. Five six seven eight. Nine ten eleven twelve. Thirteen fourteen."
        parts = src.retrieval.corpus.chunk(text, max_words=8)
        self.assertEqual(parts, ["One two three four. Five six seven eight.", "Five six seven eight. Nine ten eleven twelve.",
                                 "Nine ten eleven twelve. Thirteen fourteen."])
        self.assertTrue(all(len(p.split()) <= 8 for p in parts))

    def test_a_sentence_longer_than_the_limit_is_cut_and_nothing_is_lost(self):
        words = [f"w{i}" for i in range(25)]
        parts = src.retrieval.corpus.chunk(" ".join(words), max_words=10)
        self.assertEqual([len(p.split()) for p in parts], [10, 10, 5])
        self.assertEqual(" ".join(parts).split(), words)

    def test_a_short_text_is_one_passage_and_empty_text_is_none(self):
        self.assertEqual(src.retrieval.corpus.chunk("Dermatofibroma is benign.", 200), ["Dermatofibroma is benign."])
        self.assertEqual(src.retrieval.corpus.chunk("   ", 200), [])

    def test_tags_name_the_diagnoses_a_passage_talks_about(self):
        self.assertEqual(src.retrieval.corpus.tag("A seborrhoeic keratosis can mimic melanoma."), ["BKL", "MEL"])
        self.assertEqual(src.retrieval.corpus.tag("Bowen's disease is squamous cell carcinoma in situ."), ["SCC"])
        self.assertEqual(src.retrieval.corpus.tag("arborizing vessels", title="Basal cell carcinoma"), ["BCC"])
        self.assertEqual(src.retrieval.corpus.tag("nevirapine is a drug"), [])    # whole words only ("nevi" is inside "nevirapine")

    def test_passages_that_are_not_text_about_a_disease_are_found(self):
        bad = {
            "placeholder text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt.",
            "wiki template line": "Nevus, Body site Factors that influence nevus pattern - cite! Factors (message) Factors - participate!",
            "reference list entry": [
                'Granter SR, McKee PH, Calonje E, Mihm MC. "Melanoma associated with blue nevus". Am. J. Surg. Pathol. 25 (3): PMID 11224601.',
                "Costa J, Ortiz-Ibanez K, Salerni G. Dermoscopic patterns of melanoma metastases. Br J Dermatol. 2013 Jul;169(1):91-9",
                "Loghavi 1, Curry J1, Torres-Cabala CA. Melanoma arising in association with blue nevus. Mod Pathol.;27(11):1468-78.",
                "Kittler, H., Riedl, E., Rosendahl, C. & Cameron, A., 2008, Dermatoscopy of unpigmented lesions of the skin."]}
        for reason, texts in bad.items():
            for t in ([texts] if isinstance(texts, str) else texts):
                with self.subTest(t[:30]):
                    self.assertEqual(src.retrieval.corpus.junk_reason(t), reason)
        for good in ("A 47-year-old man presented with a pigmented lesion on his right upper back. The patient denied malignancies.",
                     "Dermoscopy can predict the diagnosis of sBCC with a sensitivity of 81.9% and specificity of 81.8%.",
                     "Smith et al. found arborizing vessels in 66% of nodular BCC (Jin et al., 2017)."):
            with self.subTest(good[:30]):
                self.assertIsNone(src.retrieval.corpus.junk_reason(good))

    def test_wiki_heading_marks_and_web_addresses_are_removed_from_a_passage(self):
        self.assertEqual(src.retrieval.corpus.strip_markup("===== Melanoma on sun damaged skin ===== https://dermoscopedia.org/05-Special_locations Identifying"),
                         "Melanoma on sun damaged skin. Identifying")

    def test_the_corpus_leaves_junk_passages_out_and_counts_them(self):
        import collections
        ns = argparse.Namespace(per_class=10, max_words=200, min_words=3)
        docs = [{"doc_id": "d1", "title": "T", "url": "u", "licence": "x", "paragraphs": [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "Melanomas show an atypical network on dermoscopy."]}]
        removed = collections.Counter()
        out = src.retrieval.corpus.passages_from(docs, "textbook", "MEL", ns, set(), set(), removed)
        self.assertEqual([p["text"] for p in out], ["Melanomas show an atypical network on dermoscopy."])
        self.assertEqual(removed["textbook: placeholder text"], 1)

    def test_spread_keeps_the_first_and_the_last_and_even_steps_between(self):
        s = src.retrieval.corpus.spread
        self.assertEqual(s(list(range(10)), 4), [0, 3, 6, 9])
        self.assertEqual(s(list(range(10)), 1), [0])
        self.assertEqual(s(list(range(3)), 8), [0, 1, 2])          # fewer than the cap: all
        self.assertEqual(s(list(range(10)), 0), list(range(10)))   # 0: no cap

    def test_a_document_gives_at_most_per_doc_passages_spread_over_it_and_the_rest_is_counted(self):
        import collections
        ns = argparse.Namespace(per_class=100, max_words=200, min_words=3, per_doc=3)
        paras = [f"Melanoma paragraph number {i} shows an atypical network." for i in range(9)]
        docs = [{"doc_id": "d1", "title": "T", "url": "u", "licence": "x", "paragraphs": paras}]
        removed = collections.Counter()
        out = src.retrieval.corpus.passages_from(docs, "pmc", "MEL", ns, set(), set(), removed)
        self.assertEqual([p["text"] for p in out], [paras[0], paras[4], paras[8]])
        self.assertEqual([p["passage_id"] for p in out], ["d1#0", "d1#1", "d1#2"])
        self.assertEqual(removed["pmc: beyond the 3 passages kept per document"], 6)
        ns.per_doc = 0
        self.assertEqual(len(src.retrieval.corpus.passages_from(docs, "pmc", "MEL", ns, set(), set(), collections.Counter())), 9)

    def test_only_full_papers_are_capped_per_document_guideline_and_textbook_pages_are_kept_whole(self):
        ns = argparse.Namespace(per_class=100, max_words=200, min_words=3, per_doc=3)
        docs = [{"doc_id": "d1", "title": "T", "url": "u", "licence": "x",
                 "paragraphs": [f"Melanoma paragraph number {i} shows an atypical network." for i in range(9)]}]
        counts = {lib: len(src.retrieval.corpus.passages_from(docs, lib, "MEL", ns, set(), set(), collections.Counter()))
                  for lib in ("pmc", "pubmed", "guidelines", "textbook")}
        self.assertEqual(counts, {"pmc": 3, "pubmed": 9, "guidelines": 9, "textbook": 9})

    def test_the_per_doc_cap_lets_more_documents_share_the_class_quota(self):
        import collections
        docs = [{"doc_id": f"d{j}", "title": "T", "url": "u", "licence": "x",
                 "paragraphs": [f"Document {j} melanoma paragraph {i} shows an atypical network." for i in range(6)]} for j in range(4)]
        ns = argparse.Namespace(per_class=6, max_words=200, min_words=3, per_doc=2)
        out = src.retrieval.corpus.passages_from(docs, "pmc", "MEL", ns, set(), set(), collections.Counter())
        self.assertEqual(sorted({p["doc_id"] for p in out}), ["d0", "d1", "d2"])
        self.assertEqual(len(out), 6)

    JATS = ('''<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta><title-group>
      <article-title>Dermoscopy of basal cell carcinoma</article-title></title-group></article-meta></front><body>
      <fig id="f1"><label>Figure 1</label><caption><title>Dermoscopy of a basal cell carcinoma.</title><p>Arborizing vessels and
        blue-gray ovoid nests are seen on the dermoscopic image of the lesion.</p></caption><graphic xlink:href="img-001"/></fig>
      <fig id="f2"><label>Figure 2</label><caption><p>Flow chart of the patients included in the basal cell carcinoma study and
        the follow-up.</p></caption><graphic xlink:href="img-002"/></fig>
      <fig id="f3"><label>Figure 3</label><caption><p>Haematoxylin and eosin stain of the basal cell carcinoma, original
        magnification x 200, showing nests of basaloid cells.</p></caption><graphic xlink:href="img-003"/></fig>
      <fig id="f4"><label>Figure 4</label><caption><p>Dermoscopy of the lesion.</p></caption><graphic xlink:href="img-004"/></fig>
      <fig id="f5"><label>Figure 5</label><caption><p>Clinical photograph of a pigmented basal cell carcinoma on the cheek of
        the patient.</p></caption></fig>
      <fig id="f6"><label>Figure 6</label><caption><p>Dermoscopy of a basal cell carcinoma. Arborizing vessels and blue-gray ovoid
        nests are seen on the dermoscopic image of the lesion!</p></caption><graphic xlink:href="img-006"/></fig>
      <fig id="f7"><label>Figure 7</label><caption><p>Clinical photograph of the lesion on the left forearm of a woman, taken
        before the visit.</p></caption><graphic xlink:href="img-007"/></fig>
      <fig id="f8"><label>Figure 8</label><caption><p>Distribution of the patients with basal cell carcinoma by age group and sex
        in the whole study cohort.</p></caption><graphic xlink:href="img-008"/></fig>
      </body></article>''')

    def test_the_figures_of_a_paper_are_read_with_label_caption_and_image_file(self):
        figs = src.retrieval.images.parse_pmc_figures(self.JATS)
        self.assertEqual(len(figs), 8)
        self.assertEqual((figs[0]["label"], figs[0]["href"]), ("Figure 1", "img-001"))
        self.assertTrue(figs[0]["caption"].startswith("Dermoscopy of a basal cell carcinoma. Arborizing vessels"))
        self.assertIsNone(figs[4]["href"])

    def test_only_real_skin_image_figures_about_one_of_the_diagnoses_are_kept_once(self):
        removed = collections.Counter()
        paper = {"doc_id": "PMC1", "title": "Dermoscopy of basal cell carcinoma", "licence": "CC BY 4.0", "query_class": "BCC",
                 "url": "https://x/PMC1"}
        items = src.retrieval.images.visual_items([paper], lambda doc_id: self.JATS, removed)
        self.assertEqual([it["item_id"] for it in items], ["PMC1#fig0", "PMC1#fig6"])
        self.assertEqual(dict(removed), {"chart, diagram or other imaging method": 1, "caption too short": 1, "no image file": 1,
                                         "duplicate caption": 1, "not a clinical or dermoscopic image": 2})
        hist = {"label": "", "href": "a", "caption": "Histology of the lesion (H&E, original magnification x200) showing basaloid nests."}
        self.assertEqual(src.retrieval.images.figure_reject_reason(hist, "basal cell carcinoma", set()), "histology only")
        real = {"label": "", "href": "a", "caption": "Traumatized nevi are a good example of the need for clinical-pathological correlation. "
                                                     "See the panoramic image of a nevus lesion on the thigh of a 44-year-old female patient."}
        self.assertEqual(src.retrieval.images.figure_reject_reason(real, "Dermoscopy of melanocytic nevi", set()), "histology only")
        mixed = {"label": "", "href": "a", "caption": "Clinical photograph of the lesion (a) and histology with H&E staining (b) in a case."}
        self.assertIsNone(src.retrieval.images.figure_reject_reason(mixed, "basal cell carcinoma", set()))
        self.assertEqual(items[0]["tags"], ["BCC"])
        self.assertEqual(items[0]["licence"], "CC BY 4.0")

    def test_a_paper_that_cannot_be_read_is_counted_and_does_not_stop_the_list(self):
        removed = collections.Counter()
        papers = [{"doc_id": "PMC1", "title": "t", "licence": "x", "query_class": "BCC", "url": "u"},
                  {"doc_id": "PMC2", "title": "Dermoscopy of basal cell carcinoma", "licence": "CC BY 4.0", "query_class": "BCC", "url": "u"}]

        def fetch_xml(doc_id):
            if doc_id == "PMC1":
                raise ValueError("no XML")
            return self.JATS
        with contextlib.redirect_stdout(io.StringIO()):                          # the expected message about PMC1
            items = src.retrieval.images.visual_items(papers, fetch_xml, removed)
        self.assertEqual(len(items), 2)
        self.assertEqual(removed["paper: figures not read"], 1)

    def test_a_figure_with_no_diagnosis_in_its_caption_or_title_is_left_out(self):
        fig = {"label": "", "href": "a", "caption": "Clinical photograph of a lesion on the left forearm of a woman, taken at a visit."}
        self.assertEqual(src.retrieval.images.figure_reject_reason(fig, "A case of itching", set()), "no diagnosis named")
        self.assertIsNone(src.retrieval.images.figure_reject_reason(fig, "A case of melanoma", set()))

    @staticmethod
    def article(licence_url="https://creativecommons.org/licenses/by/4.0/", title="Dermoscopy of basal cell carcinoma",
                extra="", article_type=""):
        return (f'<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="{article_type}"><front><article-meta><title-group>'
                f'<article-title>{title}</article-title></title-group><abstract><p>A report.{extra}</p></abstract>'
                f'<permissions><license xlink:href="{licence_url}"/></permissions></article-meta></front><body>'
                '<fig><label>Figure 1</label><caption><p>Dermoscopy of a basal cell carcinoma with arborizing vessels in a man.</p></caption>'
                '<graphic xlink:href="g1"/></fig></body></article>')

    def test_extra_papers_are_taken_from_the_pmc_search_with_the_same_licence_and_topic_rules(self):
        xmls = {"1": self.article(), "2": self.article(licence_url="https://example.org/closed"),
                "3": self.article(title="Deep learning for dermoscopy of basal cell carcinoma", extra=" We train a neural network."),
                "4": self.article(article_type="retraction"), "5": self.article(title="Another dermoscopy of basal cell carcinoma")}

        class Fake:
            def get(self, url, params=None, ncbi=False):
                if url.endswith("esearch.fcgi"):
                    return json.dumps({"esearchresult": {"idlist": ["1", "2", "3", "4", "5", "9"]}})
                if params["id"] == "9":
                    raise OSError("no XML")
                return xmls[params["id"]]
        removed = collections.Counter()
        with contextlib.redirect_stdout(io.StringIO()):                          # the expected message about PMC9
            papers = src.retrieval.images.extra_papers(Fake(), 10, {"PMC5"}, removed)                 # PMC5 is already a corpus paper
        self.assertEqual({p["doc_id"] for p in papers}, {"PMC1"})
        self.assertEqual(papers[0]["query_class"], "MEL")                           # the first diagnosis searched; later ones skip it
        self.assertEqual(dict(removed), {"extra paper: no open licence": 1, "extra paper: AI or image-dataset paper": 1,
                                         "extra paper: retracted": 1, "extra paper: not read": 1})

    def test_a_search_answered_with_an_error_is_asked_again_forgotten_and_never_stops_the_build(self):
        answers = [{"error": "API rate limit exceeded"}, {"esearchresult": {"ERROR": "Search Backend failed"}},
                   {"esearchresult": {"idlist": ["5"]}}]
        forgotten = []

        class Fake:
            def get(self, url, params=None, ncbi=False):
                return json.dumps(answers.pop(0))

            def forget(self, url, params=None, ncbi=False):
                forgotten.append(url)
        with contextlib.redirect_stdout(io.StringIO()):                          # the expected retry messages
            self.assertEqual(src.retrieval.corpus.esearch(Fake(), "pmc", "x", 5, wait=0), ["5"])
        self.assertEqual(len(forgotten), 2)                                     # the two error answers were not kept
        answers[:] = [{"error": "busy"}] * 3

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(src.retrieval.corpus.esearch(Fake(), "pmc", "x", 5, attempts=3, wait=0), [])     # still failing: no ids, no exception

    def test_the_download_cache_can_forget_an_answer(self):
        with tempfile.TemporaryDirectory() as d:
            fetch = src.retrieval.corpus.Fetcher(d)
            path = fetch.cache_path("https://x/y", {"a": 1}, ncbi=True)
            Path(path).write_text("bad", encoding="utf-8")
            self.assertEqual(fetch.get("https://x/y", {"a": 1}, ncbi=True), "bad")
            fetch.forget("https://x/y", {"a": 1}, ncbi=True)
            self.assertFalse(os.path.exists(path))
            fetch.forget("https://x/y", {"a": 1}, ncbi=True)                      # forgetting twice is harmless

    def test_the_visual_command_adds_extra_papers_when_asked_and_not_otherwise(self):
        xml = self.article()

        class Fake:
            def get(self, url, params=None, ncbi=False):
                return json.dumps({"esearchresult": {"idlist": ["7"]}}) if url.endswith("esearch.fcgi") else xml
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(os.path.join(d, "passages.jsonl"), [])
            lib = ["--corpus", d, "--images", os.path.join(d, "image_only")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(src.retrieval.images.visual_main(lib, fetch_xml=lambda x: xml), [])
                items = src.retrieval.images.visual_main([*lib, "--extra-papers", "5"], fetch=Fake())
            self.assertEqual({it["doc_id"] for it in items}, {"PMC7"})

    DERMO_PAGE = (
        '<div class="thumb tright"><div class="thumbinner"><a class="image"><img src="/images/thumb/a/ab/Bcc_1.jpg/220px-Bcc_1.jpg"></a>'
        '<div class="thumbcaption"><div class="magnify"></div>Dermoscopy of a basal cell carcinoma with arborizing vessels and '
        'blue-gray ovoid nests.</div></div></div>'
        '<figure><a><img src="/images/c/cd/Bcc_2.png"></a><figcaption>Dermoscopic image of a pigmented basal cell carcinoma with '
        'leaf-like areas on the cheek.</figcaption></figure>'
        '<div class="thumb"><img src="/images/thumb/e/ef/Chart.jpg/220px-Chart.jpg"><div class="thumbcaption">Flow chart of the study '
        'groups for the basal cell carcinoma analysis.</div></div>'
        '<div class="thumb"><img src="/images/thumb/1/12/Bcc_1.jpg/220px-Bcc_1.jpg"><div class="thumbcaption">Dermoscopy of a basal cell '
        'carcinoma with arborizing vessels and blue-gray ovoid nests.</div></div>')

    def test_dermoscopedia_pictures_are_paired_with_captions_and_given_the_full_size_address(self):
        figs = src.retrieval.images.dermoscopedia_figures(self.DERMO_PAGE)
        self.assertEqual([f["href"] for f in figs], ["https://dermoscopedia.org/images/a/ab/Bcc_1.jpg", "https://dermoscopedia.org/images/c/cd/Bcc_2.png",
                                                     "https://dermoscopedia.org/images/e/ef/Chart.jpg", "https://dermoscopedia.org/images/1/12/Bcc_1.jpg"])
        self.assertTrue(figs[0]["caption"].startswith("Dermoscopy of a basal cell carcinoma"))
        self.assertEqual(src.retrieval.images.dermoscopedia_figures(""), [])
        # a caption with no picture before it is not a figure (and a picture belongs to one caption only)
        self.assertEqual(src.retrieval.images.dermoscopedia_figures('<div class="thumbcaption">Dermoscopy of a basal cell carcinoma.</div>'), [])
        two = '<img src="/images/a/ab/X.jpg"><div class="thumbcaption">First caption.</div><div class="thumbcaption">Second caption.</div>'
        self.assertEqual(len(src.retrieval.images.dermoscopedia_figures(two)), 1)

    def test_dermoscopedia_items_follow_the_same_rules_as_pmc_figures_and_the_visual_command_adds_them_when_asked(self):
        page_html = self.DERMO_PAGE

        class Fake:
            def get(self, url, params=None, ncbi=False):
                if params.get("meta") == "siteinfo":
                    return "{}"
                if params.get("list") == "search":
                    return json.dumps({"query": {"search": [{"title": "Basal cell carcinoma"}]}})
                return json.dumps({"parse": {"text": page_html}})
        removed = collections.Counter()
        items = src.retrieval.images.dermoscopedia_items(Fake(), 3, set(), removed)
        self.assertEqual([it["item_id"] for it in items], ["dermoscopedia:Basal cell carcinoma#fig0", "dermoscopedia:Basal cell carcinoma#fig1"])
        self.assertEqual(items[0]["file_url"], "https://dermoscopedia.org/images/a/ab/Bcc_1.jpg")
        self.assertEqual((items[0]["licence"], items[0]["tags"], items[0]["library"]), ("CC BY-NC-SA 4.0", ["BCC"], "visual"))
        self.assertEqual(removed["dermoscopedia: chart, diagram or other imaging method"], 1)
        self.assertEqual(removed["dermoscopedia: duplicate caption"], 1)           # one page is read once, however many diagnoses find it
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(os.path.join(d, "passages.jsonl"), [])
            lib = ["--corpus", d, "--images", os.path.join(d, "image_only")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(src.retrieval.images.visual_main(lib, fetch_xml=lambda x: ""), [])
                self.assertEqual(len(src.retrieval.images.visual_main([*lib, "--dermoscopedia", "3"], fetch=Fake())), 2)
            card = json.loads(Path(d, "image_only", "visual_card.json").read_text(encoding="utf-8"))
            self.assertEqual(card["n_dermoscopedia_figures"], 2)

    def test_a_dermoscopedia_picture_is_downloaded_from_its_own_address(self):
        item = {"item_id": "dermoscopedia:Basal cell carcinoma#fig0", "doc_id": "dermoscopedia:Basal cell carcinoma", "href": "https://d/x.jpg",
                "file_url": "https://d/x.jpg"}
        asked = []

        def get(url):
            asked.append(url)
            return self.JPEG
        with tempfile.TemporaryDirectory() as d:
            kept, counts, worked = src.retrieval.images.download_images([item], os.path.join(d, "images"), get, pause=0)
            self.assertEqual((asked, dict(worked), kept[0]["image_url"]), (["https://d/x.jpg"], {"Dermoscopedia": 1}, "https://d/x.jpg"))
            self.assertTrue(os.path.exists(os.path.join(d, "images", kept[0]["image_file"])))
            kept, counts, _ = src.retrieval.images.download_images([item], os.path.join(d, "images2"), lambda u: b"<html>", pause=0)
            self.assertEqual((kept, dict(counts)), ([], {"no image at any address": 1}))

    def test_the_visual_list_command_writes_the_items_and_a_card_with_counts(self):
        with tempfile.TemporaryDirectory() as d:
            passage = {"passage_id": "PMC1#0", "doc_id": "PMC1", "title": "Dermoscopy of basal cell carcinoma", "licence": "CC BY 4.0",
                       "query_class": "BCC", "url": "https://x/PMC1"}
            write_jsonl(os.path.join(d, "passages.jsonl"), [passage, {**passage, "passage_id": "PMC1#1"},
                                                            {**passage, "passage_id": "m#0", "doc_id": "meditron:abc"}])
            with contextlib.redirect_stdout(io.StringIO()):
                items = src.retrieval.images.visual_main(["--corpus", d, "--images", os.path.join(d, "image_only")], fetch_xml=lambda doc_id: self.JATS)
            self.assertEqual(len(items), 2)                                  # one paper (the Meditron one is not a PMC paper)
            card = json.loads(Path(d, "image_only", "visual_card.json").read_text(encoding="utf-8"))
            self.assertEqual((card["n_papers_read"], card["n_figures_kept"], card["figures_per_tag"]["BCC"]), (1, 2, 2))
            self.assertFalse(card["images_downloaded"])
            self.assertEqual(len(src.utils.read_jsonl(os.path.join(d, "image_only", "items.jsonl"))), 2)

    JPEG = b"\xff\xd8\xff" + b"x" * 5000

    @staticmethod
    def bucket(files):
        """A fake PMC bucket: {key: bytes}. Answers a listing request with the S3 XML and a file request with the bytes."""
        def get(url):
            if url.startswith(src.retrieval.images.S3_BUCKET + "?"):
                q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                prefix, delimiter = q["prefix"][0], q.get("delimiter", [""])[0]
                keys = [k for k in files if k.startswith(prefix)]
                if delimiter:
                    subs = sorted({k[:k.index(delimiter, len(prefix)) + 1] for k in keys if delimiter in k[len(prefix):]})
                    body = "".join(f"<CommonPrefixes><Prefix>{x}</Prefix></CommonPrefixes>" for x in subs)
                else:
                    body = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
                return f'<ListBucketResult xmlns="{src.retrieval.images.S3_NS[1:-1]}">{body}</ListBucketResult>'.encode()
            if url.startswith(src.retrieval.images.S3_BUCKET) and url[len(src.retrieval.images.S3_BUCKET):] in files:
                return files[url[len(src.retrieval.images.S3_BUCKET):]]
            raise OSError("404 " + url)
        return get

    def test_a_figure_is_found_in_the_newest_version_folder_of_the_bucket_by_its_graphic_name(self):
        get = self.bucket({"PMC1.1/gr1.jpg": b"old", "PMC1.2/gr1.jpg": self.JPEG, "PMC1.2/gr2.png": b"\x89PNG" + b"x" * 5000,
                           "PMC1.2/article.xml": b"<xml/>", "PMC10.1/gr1.jpg": b"other article"})
        data, url = src.retrieval.images.fetch_image("PMC1", "gr1", get)
        self.assertEqual((data, url), (self.JPEG, src.retrieval.images.S3_BUCKET + "PMC1.2/gr1.jpg"))        # version 2, and PMC1 is not PMC10
        self.assertEqual(src.retrieval.images.fetch_image("PMC1", "gr2.png", get)[1], src.retrieval.images.S3_BUCKET + "PMC1.2/gr2.png")   # a name with its extension
        self.assertEqual(src.retrieval.images.fetch_image("PMC1", "gr9", get), (None, None))

    def test_the_older_address_is_tried_last_and_every_failure_is_explained_in_the_trace(self):
        europe = "https://europepmc.org/articles/PMC1/bin/gr1.jpg"
        trace = []

        def get(url):
            if url == europe:
                return self.JPEG
            if url.startswith(src.retrieval.images.S3_BUCKET + "?"):
                raise OSError("403 forbidden")
            raise OSError("404")
        self.assertEqual(src.retrieval.images.fetch_image("PMC1", "gr1", get, trace=trace), (self.JPEG, europe))
        self.assertTrue(any("bucket listing failed" in t and "403" in t for t in trace))
        trace = []
        empty_bucket = self.bucket({})

        def get_html(url):
            return b"<html>sorry</html>" if "europepmc" in url else empty_bucket(url)
        self.assertEqual(src.retrieval.images.fetch_image("PMC1", "gr1", get_html, trace=trace), (None, None))
        self.assertTrue(any("not an image file" in t for t in trace))
        self.assertEqual([src.retrieval.images.image_extension(x) for x in (b"\x89PNGrest", b"GIF89a", b"\xff\xd8\xffx", b"%PDF")],
                         [".png", ".gif", ".jpg", None])

    def test_images_are_saved_and_tiny_missing_and_identical_ones_are_left_out(self):
        items = [{"item_id": f"PMC1#fig{i}", "doc_id": "PMC1", "href": f"h{i}"} for i in range(5)]
        get = self.bucket({"PMC1.1/h0.jpg": self.JPEG, "PMC1.1/h1.jpg": b"\xff\xd8\xff" + b"x" * 100, "PMC1.1/h3.jpg": self.JPEG,
                           "PMC1.1/h4.jpg": self.JPEG + b"y"})
        with tempfile.TemporaryDirectory() as d:
            kept, counts, worked = src.retrieval.images.download_images(items, os.path.join(d, "images"), get, pause=0)
            self.assertEqual([k["item_id"] for k in kept], ["PMC1#fig0", "PMC1#fig4"])
            self.assertEqual(dict(counts), {"image too small": 1, "no image at any address": 1, "identical image already kept": 1})
            self.assertEqual(dict(worked), {"PMC bucket": 2})
            self.assertEqual(sorted(os.listdir(os.path.join(d, "images"))), ["PMC1_fig0.jpg", "PMC1_fig4.jpg"])
            self.assertEqual(kept[0]["image_bytes"], len(self.JPEG))
            probe_kept, _, _ = src.retrieval.images.download_images(items, os.path.join(d, "probe"), get, pause=0, probe=1)
            self.assertEqual(len(probe_kept), 1)
            self.assertFalse(os.path.exists(os.path.join(d, "probe", "PMC1_fig0.jpg")))           # a probe saves nothing

    def test_the_image_command_writes_the_list_with_images_and_updates_the_card(self):
        with tempfile.TemporaryDirectory() as d:
            folder = os.path.join(d, "image_only")
            write_jsonl(os.path.join(folder, "items.jsonl"), [{"item_id": "PMC1#fig0", "doc_id": "PMC1", "href": "h0"}])
            Path(folder, "visual_card.json").write_text(json.dumps({"n_figures_kept": 1, "images_downloaded": False}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                kept = src.retrieval.images.visual_images_main(["--images", folder], get=lambda url: self.JPEG)
            self.assertEqual(len(kept), 1)
            card = json.loads(Path(folder, "visual_card.json").read_text(encoding="utf-8"))
            self.assertEqual((card["images_downloaded"], card["n_images"], card["n_figures_kept"]), (True, 1, 1))
            self.assertEqual(len(src.utils.read_jsonl(os.path.join(folder, "items_with_images.jsonl"))), 1)

    def test_a_picture_that_is_the_same_by_its_hash_is_left_out_even_when_the_embedding_differs(self):
        def encode(paths):                                 # all vectors far apart: only the hash can catch the copy
            return [[1.0, 0.0] if "lib" in p else [0.0, 1.0] for p in paths]

        def hash_fn(path):                                 # the hash of an image is written in its file; a flipped copy is one variant
            h = int(Path(path).read_text())
            return [h, h ^ 0b1, h ^ 0b11, h ^ 0b111]
        with tempfile.TemporaryDirectory() as d:
            images, data = os.path.join(d, "image_only"), os.path.join(d, "data")
            os.makedirs(os.path.join(images, "images"))
            os.makedirs(os.path.join(data, PROCESSED_IMAGES))
            items = []
            for name, h in (("lib_copy", "1000"), ("lib_other", "7")):
                Path(images, "images", name + ".jpg").write_text(h)
                items.append({"item_id": f"PMC1#{name}", "passage_id": f"PMC1#{name}", "image_file": name + ".jpg", "title": "t",
                              "text": "x", "tags": ["MEL"]})
            write_jsonl(os.path.join(images, "items_with_images.jsonl"), items)
            Path(data, PROCESSED_IMAGES, "ISIC_0000001.jpg").write_text(str(1000 ^ 0b1))      # one bit away from a variant
            with contextlib.redirect_stdout(io.StringIO()):
                kept = src.retrieval.indexer.visual_index_main(["--images", images, "--data", data], encode=encode, hash_fn=hash_fn)
            self.assertEqual([k["item_id"] for k in kept], ["PMC1#lib_other"])
            card = json.loads(Path(images, "visual_card.json").read_text(encoding="utf-8"))["kaggle_check"]
            self.assertEqual((card["n_dropped_by_hash"], card["n_dropped_by_embedding"]), (1, 0))

    def test_the_hash_bits_follow_the_neighbour_rule_and_hamming_counts_the_different_bits(self):
        from src.retrieval import images as cm
        rising = [r * 9 + c for r in range(8) for c in range(9)]                     # every pixel darker than its right neighbour
        falling = [-v for v in rising]                                               # every pixel brighter than its right neighbour
        self.assertEqual(cm.dhash_bits(rising), 0)
        self.assertEqual(cm.dhash_bits(falling), 2 ** 64 - 1)
        self.assertEqual(cm.hamming(0b1011, 0b0001), 2)
        self.assertEqual(cm.hamming(2 ** 64 - 1, 0), 64)

    @unittest.skipUnless(importlib.util.find_spec("PIL") and importlib.util.find_spec("PIL._imaging"), "Pillow is not usable here")
    def test_the_picture_hash_is_the_same_for_a_resized_copy_and_far_for_another_picture(self):
        from PIL import Image, ImageDraw

        from src.retrieval import images as cm
        with tempfile.TemporaryDirectory() as d:
            img = Image.new("RGB", (200, 150), (200, 100, 60))
            ImageDraw.Draw(img).ellipse((40, 30, 150, 110), fill=(60, 20, 20))
            img.save(os.path.join(d, "a.png"))
            img.resize((100, 75)).save(os.path.join(d, "small.jpg"), quality=70)
            other = Image.new("RGB", (200, 150), (30, 30, 30))
            ImageDraw.Draw(other).rectangle((100, 0, 200, 150), fill=(250, 250, 250))
            other.save(os.path.join(d, "other.png"))
            a, small, far = (cm.dhash_variants(os.path.join(d, n)) for n in ("a.png", "small.jpg", "other.png"))
            self.assertLessEqual(cm.hamming(a[0], small[0]), 4)
            self.assertGreater(min(cm.hamming(x, y) for x in far for y in a), 10)
            flipped = Image.open(os.path.join(d, "a.png")).transpose(Image.FLIP_LEFT_RIGHT)
            flipped.save(os.path.join(d, "flip.png"))
            self.assertEqual(min(cm.hamming(x, a[0]) for x in cm.dhash_variants(os.path.join(d, "flip.png"))), 0)

    def test_the_calibration_counts_caught_same_lesion_pairs_and_false_alarms_on_different_lesions(self):
        # four photos: g1 has two photos of one lesion (near each other), g2 and g3 are other lesions
        vectors = [[1.0, 0.0], [0.99, 0.141], [0.0, 1.0], [0.6, 0.8]]
        hashes = [0b0000, 0b0001, 0b1111_0000, 0b0011]
        groups = ["g1", "g1", "g2", "g3"]
        r = src.retrieval.images.guard_calibration(vectors, hashes, groups, threshold=0.95, hash_threshold=1)
        same, diff = r["same_lesion_pairs"], r["most_similar_different_lesion"]
        self.assertEqual((same["n"], same["caught"]), (1, 1.0))                        # cosine 0.99 and 1 bit: caught
        self.assertEqual(diff["n"], 4)
        # no cosine between different lesions reaches 0.95; by the hash, photo 2 (0b0001) and photo 4 (0b0011) differ in 1 bit, so
        # both are flagged; photo 1 (0b0000) is 2 bits from photo 4 and photo 3 is 4 or more bits from all: not flagged -> 2 of 4
        self.assertEqual(diff["false_alarm"], 0.5)
        self.assertEqual(diff["hash_bits_different"]["min"], 1)
        r2 = src.retrieval.images.guard_calibration(vectors, hashes, groups, threshold=0.995, hash_threshold=0)
        self.assertEqual(r2["same_lesion_pairs"]["caught"], 0.0)                          # stricter thresholds miss the pair
        r3 = src.retrieval.images.guard_calibration(vectors, hashes, groups, threshold=0.995, hash_threshold=1)
        self.assertEqual(r3["same_lesion_pairs"]["caught"], 1.0)                          # the hash alone catches it

    def test_the_calibration_command_reads_the_saved_kaggle_numbers_and_writes_the_card(self):
        with tempfile.TemporaryDirectory() as d:
            images, data = os.path.join(d, "image_only"), os.path.join(d, "data")
            os.makedirs(images)
            os.makedirs(os.path.join(data, PROCESSED_CASES))
            names = ["ISIC_1.jpg", "ISIC_2.jpg", "ISIC_3.jpg"]
            Path(images, "kaggle_vectors.json").write_text(
                json.dumps({"names": names, "vectors": [[1, 0], [1, 0], [0, 1]]}), encoding="utf-8")
            Path(images, "kaggle_hashes.json").write_text(json.dumps({"names": names, "hashes": [0, 0, 255]}), encoding="utf-8")
            with open(os.path.join(data, ALL_CASES), "w", encoding="utf-8", newline="") as f:
                f.write("isic_id,group_id\nISIC_1,a\nISIC_2,a\nISIC_3,b\n")
            with contextlib.redirect_stdout(io.StringIO()):
                r = src.retrieval.images.visual_calibrate_main(["--images", images, "--data", data])
            self.assertEqual(r["same_lesion_pairs"]["caught"], 1.0)
            self.assertEqual(r["most_similar_different_lesion"]["false_alarm"], 0.0)
            card = json.loads(Path(images, "visual_card.json").read_text(encoding="utf-8"))
            self.assertIn("guard_calibration", card)

    def test_the_kaggle_vectors_are_saved_and_reused_only_for_the_same_images(self):
        calls = []

        def encode(paths):
            calls.append(len(paths))
            return [[1.0, 0.0] for _ in paths]
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(src.retrieval.images.kaggle_vectors(encode, ["a.jpg", "b.jpg"], d, d), [[1.0, 0.0]] * 2)
            self.assertEqual(src.retrieval.images.kaggle_vectors(encode, ["a.jpg", "b.jpg"], d, d), [[1.0, 0.0]] * 2)
            self.assertEqual(calls, [2])                                                   # the second time nothing was embedded
            src.retrieval.images.kaggle_vectors(encode, ["a.jpg", "b.jpg", "c.jpg"], d, d)       # another list of images: embedded again
            self.assertEqual(calls, [2, 3])

    def test_saved_kaggle_vectors_are_not_reused_when_what_was_loaded_has_changed(self):
        """encode.info (model, commit, preprocessing) is saved with the vectors: a changed setup must not reuse the old vectors."""
        def encoder(label):
            def encode(paths):
                return [[1.0, 0.0] if label == "a" else [0.0, 1.0] for _ in paths]
            encode.info = {"setup": label}
            return encode
        with tempfile.TemporaryDirectory() as d:
            first = src.retrieval.images.kaggle_vectors(encoder("a"), ["x.jpg"], d, d)
            self.assertEqual(src.retrieval.images.kaggle_vectors(encoder("a"), ["x.jpg"], d, d), first)               # same setup: reused
            second = src.retrieval.images.kaggle_vectors(encoder("b"), ["x.jpg"], d, d)                                # other setup: made again
            self.assertNotEqual(second, first)
            self.assertEqual(json.loads(Path(d, "kaggle_vectors.json").read_text(encoding="utf-8"))["info"], {"setup": "b"})

    def test_the_same_picture_in_two_papers_is_kept_once(self):
        def encode(paths):
            return [[float(x) for x in Path(p).read_text().split(",")] for p in paths]
        with tempfile.TemporaryDirectory() as d:
            images, data = os.path.join(d, "image_only"), os.path.join(d, "data")
            os.makedirs(os.path.join(images, "images"))
            os.makedirs(os.path.join(data, PROCESSED_IMAGES))
            items = []
            for name, vec in (("a", "0.6,0.8"), ("b", "0.6,0.8"), ("c", "0.8,0.6")):
                Path(images, "images", name + ".jpg").write_text(vec)
                items.append({"item_id": f"PMC1#{name}", "passage_id": f"PMC1#{name}", "image_file": name + ".jpg", "title": "t",
                              "text": "x", "tags": ["MEL"]})
            write_jsonl(os.path.join(images, "items_with_images.jsonl"), items)
            Path(data, PROCESSED_IMAGES, "ISIC_0000001.jpg").write_text("0,1")
            with contextlib.redirect_stdout(io.StringIO()):
                kept = src.retrieval.indexer.visual_index_main(["--images", images, "--data", data], encode=encode, hash_fn=fake_hash)
            self.assertEqual([k["item_id"] for k in kept], ["PMC1#a", "PMC1#c"])      # b is the same picture as a (cosine 1.0)
            card = json.loads(Path(images, "visual_card.json").read_text(encoding="utf-8"))["kaggle_check"]
            self.assertEqual(card["n_dropped_as_library_duplicates"], 1)

    def test_a_library_image_that_looks_like_a_kaggle_image_is_left_out_and_the_card_shows_the_numbers(self):
        def encode(paths):                                 # a fake encoder: the vector is written in the file
            return [[float(x) for x in Path(p).read_text().split(",")] for p in paths]
        with tempfile.TemporaryDirectory() as d:
            images, data = os.path.join(d, "image_only"), os.path.join(d, "data")
            os.makedirs(os.path.join(images, "images"))
            os.makedirs(os.path.join(data, PROCESSED_IMAGES))
            items = []
            for name, vec in (("near", "1,0"), ("far", "0,1"), ("mid", "0.6,0.8")):
                Path(images, "images", name + ".jpg").write_text(vec)
                items.append({"item_id": f"PMC1#{name}", "passage_id": f"PMC1#{name}", "image_file": name + ".jpg", "title": "t",
                              "text": "x", "tags": ["MEL"]})
            write_jsonl(os.path.join(images, "items_with_images.jsonl"), items)
            Path(data, PROCESSED_IMAGES, "ISIC_0000001.jpg").write_text("1,0")
            Path(data, PROCESSED_IMAGES, "ISIC_0000002.jpg").write_text("0,1")
            Path(images, "visual_card.json").write_text(json.dumps({"n_figures_kept": 3, "guard_calibration": {"old": True}}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                kept = src.retrieval.indexer.visual_index_main(["--images", images, "--data", data, "--threshold", "0.95"], encode=encode, hash_fn=fake_hash)
            # near and far each equal a Kaggle image (cosine 1.0); mid (0.6, 0.8) has cosine 0.8 at most with both
            self.assertEqual([k["item_id"] for k in kept], ["PMC1#mid"])
            card = json.loads(Path(images, "visual_card.json").read_text(encoding="utf-8"))["kaggle_check"]
            self.assertEqual((card["n_dropped"], card["n_kaggle_images"], card["threshold"]), (2, 2, 0.95))
            self.assertEqual(card["highest_20"][0]["similarity"], 1.0)
            self.assertEqual(sorted(d_["item_id"] for d_ in card["dropped"]), ["PMC1#far", "PMC1#near"])           # the card names what was left out
            whole = json.loads(Path(images, "visual_card.json").read_text(encoding="utf-8"))
            self.assertNotIn("guard_calibration", whole)                   # it measured the earlier check: visual-calibrate must be run again
            self.assertEqual(whole["image_encoder"], "chanwkim/monet")
            idx = common_read_index(os.path.join(images, "index"), os.path.join(images, "items_final.jsonl"))
            self.assertEqual([p["passage_id"] for p, _ in idx.search([0.6, 0.8], 1)], ["PMC1#mid"])
            self.assertEqual(json.loads(Path(images, "visual_card.json").read_text(encoding="utf-8"))["kaggle_check"]
                             ["n_dropped_as_library_duplicates"], 0)
            with self.assertRaises(SystemExit):                 # when every image is a look-alike nothing is indexed
                src.retrieval.indexer.visual_index_main(["--images", images, "--data", data, "--threshold", "0.5"], encode=encode, hash_fn=fake_hash)

    def test_plural_forms_are_tagged_too(self):
        self.assertEqual(src.retrieval.corpus.tag("Melanomas show asymmetry."), ["MEL"])
        self.assertEqual(src.retrieval.corpus.tag("Basal cell carcinomas and squamous cell carcinomas are common."), ["BCC", "SCC"])
        self.assertEqual(src.retrieval.corpus.tag("Haemangiomas and keratoacanthomas."), ["SCC", "VASC"])
        self.assertEqual(src.retrieval.corpus.tag("Melanomatous change"), [])     # a longer word is not a match

    def test_ai_and_image_dataset_papers_are_recognised(self):
        self.assertTrue(src.retrieval.corpus.is_ai_paper("Deep learning for skin cancer", ""))
        self.assertTrue(src.retrieval.corpus.is_ai_paper("A study", "We trained on HAM10000 images."))
        self.assertFalse(src.retrieval.corpus.is_ai_paper("Dermoscopy of basal cell carcinoma", LONG))

    def test_only_creative_commons_licences_count_as_open(self):
        for licence in ("CC BY", "CC0", "CC BY-NC-ND", "cc by-sa"):
            self.assertTrue(src.retrieval.corpus.open_licence(licence), licence)
        for licence in ("NO-CC CODE", "", None, "CC BY-XYZ"):
            self.assertFalse(src.retrieval.corpus.open_licence(licence), licence)

    def test_topics_of_a_passage(self):
        self.assertEqual(src.retrieval.corpus.topics(LONG), ["appearance", "basics", "differential", "patients"])
        self.assertEqual(src.retrieval.corpus.topics("Nothing here."), [])

    def test_wiki_html_keeps_paragraphs_and_list_items_only(self):
        paras = src.retrieval.corpus.html_paragraphs(WIKI_PAGE + "<ul><li>first<br/>line</li></ul>")
        self.assertEqual(len(paras), 2)
        self.assertNotIn("[1]", paras[0])
        self.assertNotIn("table text", " ".join(paras))
        self.assertEqual(paras[1], "first line")


class ParserTests(unittest.TestCase):
    def test_pubmed_xml(self):
        docs = src.retrieval.corpus.parse_pubmed_xml(PUBMED_XML)
        self.assertEqual([d["doc_id"] for d in docs], ["PMID:101", "PMID:102", "PMID:103", "PMID:104"])
        self.assertEqual((docs[0]["year"], docs[0]["url"]), ("2020", "https://pubmed.ncbi.nlm.nih.gov/101/"))
        self.assertEqual([d["retracted"] for d in docs], [False, False, True, False])

    def test_the_licence_and_retraction_of_a_pmc_paper_come_from_its_own_xml(self):
        def jats(licence="", article_type="research-article", related=""):
            return JATS.format(licence=licence, article_type=article_type, related=related)

        def lic(url, text=""):
            return f'<license xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="{url}"><license-p>{text}</license-p></license>'

        for url, name in (("https://creativecommons.org/licenses/by/4.0/", "CC BY 4.0"),
                          ("http://creativecommons.org/licenses/by-nc-sa/3.0/", "CC BY-NC-SA 3.0"),
                          ("https://creativecommons.org/publicdomain/zero/1.0/", "CC0 1.0"),
                          ("https://creativecommons.org/licenses/by-nd/4.0", "CC BY-ND 4.0")):
            self.assertEqual(src.retrieval.corpus.parse_pmc_licence(jats(lic(url))), (name, False), url)
            self.assertTrue(src.retrieval.corpus.open_licence(name))
        self.assertEqual(src.retrieval.corpus.parse_pmc_licence(jats()), (None, False))                                  # no licence element
        self.assertEqual(src.retrieval.corpus.parse_pmc_licence(jats('<license><license-p>All rights reserved.</license-p></license>')), (None, False))
        self.assertEqual(src.retrieval.corpus.parse_pmc_licence(jats(lic("", "see https://creativecommons.org/licenses/by/4.0/ for terms"))),
                         ("CC BY 4.0", False))                                                         # the link in the text
        self.assertEqual(src.retrieval.corpus.parse_pmc_licence(jats(lic("https://creativecommons.org/licenses/by/4.0/"), "retraction")
                                              .replace("research-article", "retraction"))[1], True)    # a retraction notice
        related = '<related-article related-article-type="retracted-article" xlink:href="x"/>'
        self.assertEqual(src.retrieval.corpus.parse_pmc_licence(jats(lic("https://creativecommons.org/licenses/by/4.0/"), related=related)),
                         ("CC BY 4.0", True))
        self.assertEqual(src.retrieval.corpus.parse_pmc_licence("<article/>"), (None, False))
        self.assertEqual(src.retrieval.corpus.parse_pmc_licence("<pmc-articleset/>"), (None, False))

    def test_bioc_keeps_text_sections_and_drops_references_and_tables(self):
        for answer in (BIOC, BIOC[0]):                        # a list of collections, or one collection
            title, year, paras = src.retrieval.corpus.parse_bioc(answer)
            self.assertEqual((title, year, len(paras)), ("Basal cell carcinoma: a review", "2021", 2))
            self.assertFalse([p for p in paras if "reference" in p or "table" in p])

    def test_meditron_guidelines_without_wikidoc_nice_and_unrelated_topics(self):
        removed = collections.Counter()
        docs = list(src.retrieval.corpus.meditron_guideline_docs(removed, MEDITRON))
        self.assertEqual([d["doc_id"] for d in docs], ["meditron:g1"])
        self.assertEqual(len(docs[0]["paragraphs"]), 3)
        self.assertEqual(removed, {"guidelines: source wikidoc excluded": 1, "guidelines: source nice excluded": 1})


class MeditronCacheTests(unittest.TestCase):
    def test_the_large_download_is_made_once_and_read_from_the_cache_afterwards(self):
        def refuse():
            raise AssertionError("downloaded again")

        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "meditron_skin_cancer.json")
            first = collections.Counter()
            docs = src.retrieval.corpus.meditron_guideline_docs(first, cache=cache, load=lambda: MEDITRON)
            self.assertEqual([x["doc_id"] for x in docs], ["meditron:g1"])
            self.assertTrue(os.path.isfile(cache))
            again = collections.Counter()
            self.assertEqual(src.retrieval.corpus.meditron_guideline_docs(again, cache=cache, load=refuse), docs)
            self.assertEqual(again, first)                                   # the removal counts come back with it
            self.assertEqual(first, {"guidelines: source wikidoc excluded": 1, "guidelines: source nice excluded": 1})
            other = os.path.join(d, "not_written.json")
            src.retrieval.corpus.meditron_guideline_docs(collections.Counter(), MEDITRON, cache=other)       # rows given by a caller: no cache is written
            self.assertFalse(os.path.exists(other))

    def test_an_interrupted_pass_saves_nothing(self):
        def broken():
            yield MEDITRON[0]
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "meditron_skin_cancer.json")
            with self.assertRaises(KeyboardInterrupt):
                src.retrieval.corpus.meditron_guideline_docs(collections.Counter(), cache=cache, load=broken)
            self.assertEqual(os.listdir(d), [])

    def test_the_hugging_face_library_gets_a_longer_timeout_and_no_symlink_warning(self):
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.dict(sys.modules, {"datasets": mock.Mock()}):
            os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)
            src.retrieval.corpus.meditron_rows_from_hub()
            self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "60")
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "5"
            src.retrieval.corpus.meditron_rows_from_hub()
            self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "5")        # a value the researcher set is kept
            del os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]


class BuildTests(unittest.TestCase):
    def build(self, **kw):
        fetch = FakeFetcher()
        with contextlib.redirect_stdout(io.StringIO()):
            passages, removed = src.retrieval.corpus.build_corpus(args(**kw), fetch, meditron_rows=MEDITRON)
        return passages, removed, fetch

    def test_every_library_contributes_and_every_passage_can_be_cited(self):
        passages, removed, _ = self.build()
        self.assertEqual({p["library"] for p in passages}, set(src.retrieval.corpus.LIBRARIES))
        for p in passages:
            self.assertTrue(p["url"] and p["licence"] and p["doc_id"] and p["text"], p["passage_id"])
        self.assertEqual(len({p["passage_id"] for p in passages}), len(passages))
        self.assertEqual(len({p["md5"] for p in passages}), len(passages))      # no duplicate text

    # ---- provenance: real sources, unchanged text ----
    def test_a_corpus_made_by_the_builder_passes_the_provenance_check_and_each_tampering_is_found(self):
        passages, _, _ = self.build()
        report = src.retrieval.corpus.provenance_offline(passages)
        self.assertEqual((report["records_with_a_problem"], report["problems"]), (0, {}))
        self.assertEqual(report["n_documents"], sum(report["documents_per_source"].values()))
        self.assertTrue({"PMID", "PMC", "dermoscopedia", "meditron"} <= set(report["documents_per_source"]))
        victim = next(p for p in passages if p["library"] == "pubmed")
        wrong_host = next(p for p in passages if p["library"] == "pmc")
        tampered = [{**victim, "text": victim["text"] + " An invented sentence."},                       # edited after the build
                    {**wrong_host, "url": "https://example.org/x"},                                        # not the source's own site
                    {**victim, "passage_id": "x1", "doc_id": "made-up-7"},                                 # not an id of a real source
                    {**victim, "passage_id": "x2", "licence": " "},
                    {**victim, "passage_id": "x3", "library": "pmc"}]                                      # a PubMed id in the PMC library
        found = src.retrieval.corpus.provenance_offline(tampered)["problem_counts"]
        self.assertEqual(found, {"the text is empty or no longer matches its fingerprint": 1,
                                 "no web address on the source's own site": 2, "the document id is not that of a real source of this library": 2,
                                 "no licence": 1})

    def test_a_guideline_without_an_address_is_given_the_collections_page_also_from_a_saved_download(self):
        rows = [{**MEDITRON[0], "id": "a", "url": "None"}, {**MEDITRON[0], "id": "b", "url": "http://www.scielo.br/x.pdf"},
                {**MEDITRON[0], "id": "c", "url": None}]
        docs = src.retrieval.corpus.meditron_guideline_docs(collections.Counter(), rows)
        collection = src.retrieval.corpus.MEDITRON_COLLECTION
        self.assertEqual([d["url"] for d in docs], [collection, "http://www.scielo.br/x.pdf", collection])        # its own address is kept
        with tempfile.TemporaryDirectory() as d:                                                                   # a download saved before this rule
            cache = os.path.join(d, "saved.json")
            Path(cache).write_text(json.dumps({"docs": [{**docs[0], "url": "None"}], "removed": {}}), encoding="utf-8")
            again = src.retrieval.corpus.meditron_guideline_docs(collections.Counter(), cache=cache, load=lambda: [])
        self.assertEqual(again[0]["url"], collection)
        passages = src.retrieval.corpus.passages_from(docs[:1], "guidelines", "MEL", args(), set(), set(), collections.Counter())
        self.assertEqual({src.retrieval.corpus.source_problems(p) == [] for p in passages}, {True})               # and the audit accepts it

    def test_a_guideline_without_an_address_is_a_problem_and_an_http_address_is_not(self):
        base = next(p for p in self.build()[0] if p["library"] == "guidelines" and p["doc_id"].startswith("meditron:"))
        problems = lambda **kw: src.retrieval.corpus.source_problems({**base, **kw})                      # noqa: E731
        self.assertEqual(problems(url="http://www.scielo.br/pdf/x.pdf"), [])                                 # a real guideline site that is not https
        for none in ("None", "", None, "nan", " null "):
            self.assertEqual(problems(url=none), ["no web address on the source's own site"], none)
        self.assertEqual(problems(url="ftp://x"), ["no web address on the source's own site"])
        self.assertEqual(src.retrieval.corpus.real_url("None"), "")                                         # the Meditron collection's word for "no address"
        self.assertEqual(src.retrieval.corpus.real_url("https://a.org/x"), "https://a.org/x")

    def test_the_audit_command_writes_the_provenance_and_names_the_records_with_a_problem(self):
        passages, _, _ = self.build()
        bad = {**next(p for p in passages if p["library"] == "guidelines"), "passage_id": "meditron:zz#0", "url": "http://old.example.org/g"}
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(os.path.join(d, "passages.jsonl"), [*passages, bad])
            refs = os.path.join(d, "refs.json")
            Path(refs).write_text(json.dumps({"MEL": {"texts": [{"text": "one two three four five six seven eight nine"}]}}), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = src.retrieval.corpus.audit_main(["--corpus", d, "--reference-texts", refs, "--images", os.path.join(d, "none")])
            saved = json.loads(Path(d, "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["provenance"]["problems"], {"no web address on the source's own site": ["meditron:zz#0"]})
            self.assertEqual(result["provenance"]["records_with_a_problem"], 1)
            self.assertIn("problem 'no web address on the source's own site': meditron:zz#0", out.getvalue())     # the ids are shown, not only counts

    def test_titles_agree_across_punctuation_signs_and_alphabets_and_a_bracketed_translation_is_not_another_title(self):
        agree = src.retrieval.corpus.titles_agree
        self.assertTrue(agree("Dermoscopy of very small basal cell carcinoma (≤3 mm)☆", "Dermoscopy of very small basal cell carcinoma (≤3mm)."))
        self.assertTrue(agree("Étude descriptive dermoscopique", "ÉTUDE descriptive dermoscopique."))                   # the audit's real cases
        self.assertTrue(agree("A long title with a subtitle: more words", "A long title with a subtitle"))              # a title cut at the colon
        self.assertFalse(agree("Dermoscopy of melanoma", "Surgery of the knee"))
        self.assertFalse(agree("", "Anything"))
        translated = src.retrieval.corpus.is_translated_title
        self.assertTrue(translated('[Characteristics of melanoma in Mexicans seen at "La Raza" National Medical Center].'))
        self.assertFalse(translated("Surgery of the knee."))
        self.assertFalse(translated(""))

    def test_the_online_check_finds_a_document_that_does_not_exist_and_one_that_has_another_title(self):
        passages, _, _ = self.build()
        titles = {p["doc_id"]: p["title"] for p in passages}
        real = sorted(d for d in titles if d.startswith(("PMID:", "PMC")))
        gone, other = real[0], real[-1]
        translated = real[1]

        class Ncbi:
            def get(self, url, params=None, ncbi=False):
                if params.get("meta") == "siteinfo":
                    return "{}"
                if "titles" in params:
                    return json.dumps({"query": {"pages": {"1": {"title": t, **({"missing": ""} if t == "gone page" else {})}
                                                           for t in params["titles"].split("|") + ["gone page"]}}})
                prefix = "PMID:" if params["db"] == "pubmed" else "PMC"      # (the same fake answers for both databases)
                rows = {i: {"uid": i, "title": ("Something else entirely" if prefix + i == other else "[An English translation].")
                            if prefix + i in (other, translated) else titles[prefix + i].upper() + "."}
                        for i in params["id"].split(",") if prefix + i in titles and prefix + i != gone}
                return json.dumps({"result": {"uids": list(rows), **rows}})
        r = src.retrieval.corpus.provenance_online(passages, Ncbi())
        self.assertEqual((r["not_found"], r["other_title"], r["unreachable"]), ([gone], [other], []))
        self.assertEqual(r["translated_title"], [translated])                          # a bracketed translation is counted apart, not as another title
        self.assertEqual(r["other_title_detail"][other]["ncbi"], "Something else entirely")                  # both titles are kept, to be compared
        self.assertEqual(r["other_title_detail"][other]["recorded"], titles[other])      # a case and full-stop change is no change
        self.assertGreater(r["checked"], 3)

        class Down:
            def get(self, url, params=None, ncbi=False):
                raise OSError("no internet")
        down = src.retrieval.corpus.provenance_online(passages, Down())
        self.assertEqual((down["not_found"], down["other_title"]), ([], []))          # no answer is never reported as "does not exist"
        self.assertTrue(down["unreachable"])

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

    def test_a_failed_licence_lookup_is_counted_and_does_not_stop_the_build(self):
        class Failing(FakeFetcher):
            def get(self, url, params=None, ncbi=False):
                if url.endswith("efetch.fcgi") and (params or {}).get("db") == "pmc":
                    raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
                return super().get(url, params, ncbi)

        with contextlib.redirect_stdout(io.StringIO()):
            passages, removed = src.retrieval.corpus.build_corpus(args(), Failing(), meditron_rows=MEDITRON)
        self.assertGreater(removed["pmc: licence lookup failed"], 0)
        self.assertEqual(removed["pmc: no open licence"], 0)
        self.assertEqual({p["library"] for p in passages} & {"pubmed", "textbook"}, {"pubmed", "textbook"})   # the rest goes on
        self.assertNotIn("pmc", {p["library"] for p in passages})                                           # nothing unverified is kept
        self.assertNotIn("PMC13", {p["doc_id"] for p in passages})

    def test_pmc_licences_are_asked_of_the_e_utilities_not_of_the_retired_oa_service(self):
        _, _, fetch = self.build()
        asked = [(u, p) for u, p, _ in fetch.calls if u.endswith("efetch.fcgi") and p.get("db") == "pmc"]
        self.assertTrue(asked)
        self.assertFalse([u for u, _, _ in fetch.calls if "oa.fcgi" in u])
        self.assertFalse(any(hasattr(m, "PMC_OA") for m in (src.retrieval.corpus,)))

    def test_no_meditron_skips_the_large_download_and_the_cached_download_is_used_by_the_build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with_it, _ = src.retrieval.corpus.build_corpus(args(libraries=["guidelines"]), FakeFetcher(), meditron_rows=MEDITRON)
            without, _ = src.retrieval.corpus.build_corpus(args(libraries=["guidelines"], no_meditron=True), FakeFetcher(), meditron_rows=MEDITRON)
        self.assertTrue([p for p in with_it if p["doc_id"].startswith("meditron:")])
        self.assertFalse([p for p in without if p["doc_id"].startswith("meditron:")])
        with tempfile.TemporaryDirectory() as d:
            fetch = FakeFetcher()
            fetch.cache = d
            src.utils.write_json(os.path.join(d, "meditron_skin_cancer.json"),
                          {"docs": src.retrieval.corpus.meditron_guideline_docs(collections.Counter(), MEDITRON), "removed": {}})
            with contextlib.redirect_stdout(io.StringIO()):
                passages, _ = src.retrieval.corpus.build_corpus(args(libraries=["guidelines"]), fetch)       # no rows given: the cache answers
        self.assertTrue([p for p in passages if p["doc_id"].startswith("meditron:")])

    def test_the_keys_come_from_the_env_file_and_are_never_printed(self):
        with tempfile.TemporaryDirectory() as d:
            env = os.path.join(d, ".env")
            Path(env).write_text("HF_TOKEN=hf_SECRET_VALUE\nNCBI_API_KEY=\n", encoding="utf-8")
            argv = ["--out", os.path.join(d, "corpus"), "--libraries", "textbook", "--textbook-pages", "1", "--env-file", env]
            seen = {}

            def build(args_, fetch):
                seen["token"] = os.environ.get("HF_TOKEN")
                return [], collections.Counter()

            with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(src.retrieval.corpus, "build_corpus", build), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                os.environ.pop("HF_TOKEN", None)
                os.environ.pop("NCBI_API_KEY", None)
                src.retrieval.corpus.corpus_main(argv)
                os.environ.pop("HF_TOKEN", None)
            log_text = out.getvalue()
            self.assertFalse(Path(d, "corpus", "build_corpus.log").exists())        # no log file (researcher, 2026-11-25)
            card = Path(d, "corpus", "corpus_card.json").read_text(encoding="utf-8")
        self.assertEqual(seen["token"], "hf_SECRET_VALUE")
        self.assertNotIn("hf_SECRET_VALUE", log_text + card)
        self.assertNotIn("env_file", card)                                  # the path of the key file is not recorded either
        self.assertIn("HF_TOKEN: found", log_text)
        self.assertIn("NCBI_API_KEY: not set", log_text)

    def test_a_broken_env_line_is_reported_by_its_number_only_in_the_corpus_build_and_in_a_run(self):
        with tempfile.TemporaryDirectory() as d:
            env = os.path.join(d, ".env")
            Path(env).write_text("# keys\nHF_TOKEN=hf_ok\nsk-or-a-value-without-a-name\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(src.retrieval.corpus, "build_corpus", lambda a, f: ([], collections.Counter())), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                os.environ.pop("HF_TOKEN", None)
                src.retrieval.corpus.corpus_main(["--out", os.path.join(d, "corpus"), "--env-file", env])
                os.environ.pop("HF_TOKEN", None)
            self.assertIn("lines [3] are not NAME=value", out.getvalue())
            self.assertNotIn("sk-or-a-value-without-a-name", out.getvalue())
            questions = os.path.join(d, "data", question_file("text_only", "val"))
            write_jsonl(questions, [])
            with mock.patch.object(src.pipelines.llm_only, "make_client", lambda model, log: object()), mock.patch.object(src.pipelines.llm_only, "run", lambda *a: "done"), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                src.pipelines.llm_only.llm_only_main(["--model", "gpt-4o", "--allow-paid", "--view", "text_only", "--split", "val", "--env-file", env,
                                  "--data", os.path.join(d, "data"), "--results", os.path.join(d, "results")])
            self.assertIn("lines [3] are not NAME=value", out.getvalue())
            self.assertNotIn("sk-or-a-value-without-a-name", out.getvalue())

    def test_an_empty_search_is_reported_with_the_way_ncbi_read_the_query(self):
        class Empty(FakeFetcher):
            def get(self, url, params=None, ncbi=False):
                if url.endswith("esearch.fcgi") and params["db"] == "pubmed":
                    return json.dumps({"esearchresult": {"count": "0", "idlist": [], "querytranslation": '"basal cell"[All Fields]',
                                                         "errorlist": {"phrasesnotfound": ["xyz"]}}})
                return super().get(url, params, ncbi)

        with contextlib.redirect_stdout(io.StringIO()) as out:
            passages, _ = src.retrieval.corpus.build_corpus(args(libraries=["pubmed"]), Empty(), meditron_rows=MEDITRON)
        self.assertEqual(passages, [])
        text = out.getvalue()
        self.assertEqual(text.count("pubmed search found nothing"), 8)          # once for each diagnosis
        self.assertIn("""query as read by NCBI: '"basal cell"[All Fields]'""", text)
        self.assertIn("phrasesnotfound", text)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            src.retrieval.corpus.build_corpus(args(libraries=["pubmed"]), FakeFetcher(), meditron_rows=MEDITRON)
        self.assertNotIn("found nothing", out.getvalue())                        # silent when there are results

        class Dropped(FakeFetcher):                       # results, but a quoted MeSH heading was not found (an OR kept it going)
            def get(self, url, params=None, ncbi=False):
                body = super().get(url, params, ncbi)
                if url.endswith("esearch.fcgi") and params["db"] == "pubmed":
                    r = json.loads(body)
                    r["esearchresult"]["warninglist"] = {"quotedphrasesnotfound": ['"old name"[MeSH Terms]']}
                    return json.dumps(r)
                return body

        with contextlib.redirect_stdout(io.StringIO()) as out:
            src.retrieval.corpus.build_corpus(args(libraries=["pubmed"]), Dropped(), meditron_rows=MEDITRON)
        self.assertIn("""did not find ['"old name"[MeSH Terms]']""", out.getvalue())
        self.assertNotIn("found nothing", out.getvalue())

    def test_the_basal_cell_query_does_not_depend_on_one_name_of_the_mesh_heading(self):
        query = src.retrieval.corpus.PUBMED_QUERIES["BCC"]
        for part in ('"carcinoma, basal cell"[MeSH Terms]', '"basal cell carcinoma"[MeSH Terms]', '"basal cell carcinoma"[tiab]'):
            self.assertIn(part, query)
        self.assertRegex(query, r"^\(.* OR .* OR .*\) AND \(dermoscopy")           # the three names are alternatives, not all required

    def test_no_library_gets_more_than_the_cap_per_diagnosis(self):
        passages, _, _ = self.build(per_class=1)
        counts = collections.Counter((p["library"], p["query_class"]) for p in passages)
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
        self.assertNotIn("--email", " ".join(src.retrieval.corpus.corpus_args([]).__dict__))

    def test_the_card_counts_and_the_manifest_has_no_text(self):
        passages, removed, _ = self.build()
        card = src.retrieval.corpus.corpus_card(passages, removed, args(out="x", cache="y"), "2026-11-22")
        self.assertEqual(card["n_passages"], len(passages))
        self.assertEqual(sum(sum(v.values()) for v in card["passages_per_library_and_class"].values()), len(passages))
        self.assertIn("VASC/appearance", card["empty_topic_boxes"])   # no made-up text is about vascular lesions
        self.assertNotIn("out", card["settings"])
        with tempfile.TemporaryDirectory() as d:
            src.retrieval.corpus.write_outputs(passages, card, d)
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
            passages, removed = src.retrieval.corpus.build_corpus(args(libraries=["guidelines"]), FakeFetcher(), meditron_rows=Broken())
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
            f = src.retrieval.corpus.Fetcher(d)
            self.assertEqual(f.get(src.retrieval.corpus.NCBI + "esearch.fcgi", {"db": "pubmed"}, ncbi=True), '{"ok": 1}')
            self.assertEqual(f.get(src.retrieval.corpus.NCBI + "esearch.fcgi", {"db": "pubmed"}, ncbi=True), '{"ok": 1}')
            stored = [Path(d, n).read_text(encoding="utf-8") + n for n in os.listdir(d)]
        self.assertEqual(len(seen), 1)                        # the second answer came from the cache
        self.assertIn("api_key=SECRET123", seen[0])
        self.assertIn("tool=" + src.retrieval.corpus.TOOL, seen[0])
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
                src.retrieval.corpus.Fetcher(d).get(src.retrieval.corpus.NCBI + "efetch.fcgi", {"id": "1"}, ncbi=True)
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch("urllib.request.urlopen", no_internet):
                self.assertEqual(src.retrieval.corpus.Fetcher(d).get(src.retrieval.corpus.NCBI + "efetch.fcgi", {"id": "1"}, ncbi=True), "cached answer")


# ----------------------------------------------------------------------------------------------------------------------
# the reference-text builder with a fake wiki (the references command)
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
            body = p["texts"].get(heading, f"<p>{params['page']} / {src.evaluation.references.clean_heading(heading)} text. {WORDS}</p>")
            html = f"<h2>{heading}</h2><p>{params['page']} – cite!</p>{body}<table><tr><td><p>table text</p></td></tr></table>"
            sub = p["sections"][p["sections"].index(next(s for s in p["sections"] if s["index"] == index)) + 1:]
            if p["texts"].get("nested") == heading and sub:      # the wiki returns a section with its subsections
                html += f"<h3>{sub[0]['line']}</h3><p>{params['page']} / {sub[0]['line']} text. {WORDS}</p>"
        out = {"text": html}
        if "links" in params["prop"]:
            out["links"] = [{"ns": 0, "title": t} for t in p["links"]]
        return json.dumps({"parse": out})


def build_references(max_words=200, min_words=20):
    with tempfile.TemporaryDirectory() as d, mock.patch.object(src.evaluation.references, "Fetcher", FakeWiki), \
            contextlib.redirect_stdout(io.StringIO()):
        out = os.path.join(d, "reference_texts.json")
        src.evaluation.references.references_main(["--out", out, "--cache", os.path.join(d, "cache"), "--max-words", str(max_words), "--min-words", str(min_words)])
        return json.loads(Path(out).read_text(encoding="utf-8"))


class ReferenceRuleTests(unittest.TestCase):
    def test_the_first_term_with_results_decides_and_an_equal_title_wins_within_it(self):
        self.assertEqual(src.evaluation.references.choose_page([("nevus", ["Nevi of the face", "Nevus"])]), ("Nevus", "title equals 'nevus'"))
        self.assertEqual(src.evaluation.references.choose_page([("a", []), ("b", ["B page", "Other"])]), ("B page", "first result for 'b'"))
        self.assertEqual(src.evaluation.references.choose_page([("a", ["A page"]), ("b", ["b"])]), ("A page", "first result for 'a'"))
        self.assertEqual(src.evaluation.references.choose_page([("a", [])]), (None, "no page found"))

    def test_sections_from_the_dermoscopy_heading_else_all_else_the_introduction(self):
        sections = [{"index": "1", "line": "History"}, {"index": "2", "line": "<b>Dermatoscopic</b> features"},
                    {"index": "3", "line": "Milia-like cysts"}]
        self.assertEqual(src.evaluation.references.choose_sections(sections), (["2", "3"], "from 'Dermatoscopic features'"))
        self.assertEqual(src.evaluation.references.choose_sections([{"index": "1", "line": "Large nests:"}])[0], ["1"])
        self.assertEqual(src.evaluation.references.choose_sections([]), ([0], "(introduction)"))

    def test_headings_and_template_lines_are_cleaned(self):
        self.assertEqual(src.evaluation.references.clean_heading("Dermoscopy features &#91;1&#93; &#91;2&#93;"), "Dermoscopy features")
        self.assertEqual(src.evaluation.references.text_paragraphs("<p>Nevus – cite!</p><p>Nevus (message)</p><p>A nevus is benign.</p>"),
                         ["A nevus is benign."])

    def test_texts_of_whole_sentences_up_to_the_word_limit_and_short_ones_left_out(self):
        paragraphs = ["One two three. Four five six seven. Eight nine."]
        self.assertEqual(src.evaluation.references.split_texts(paragraphs, 7, 1), (["One two three. Four five six seven.", "Eight nine."], []))
        self.assertEqual(src.evaluation.references.split_texts(paragraphs, 7, 3), (["One two three. Four five six seven."], [2]))
        self.assertEqual(src.evaluation.references.split_texts(["One two three four five."], 2, 1)[0][0], "One two")   # a too long sentence is cut
        self.assertEqual(src.evaluation.references.split_texts([], 10, 1), ([], []))

    def test_a_section_is_read_without_its_subsections(self):
        section = "<h2>Criteria</h2><p>own</p><h3>Sub</h3><p>sub</p>"
        self.assertEqual(src.evaluation.references.own_html(section), "<h2>Criteria</h2><p>own</p>")
        self.assertEqual(src.evaluation.references.own_html("<p>no heading</p>"), "<p>no heading</p>")

    def test_sections_not_about_the_disease_are_left_out(self):
        sections = [{"index": "1", "line": "Dermoscopy"}, {"index": "2", "line": "Differential diagnosis"},
                    {"index": "3", "line": "Vessels"}, {"index": "4", "line": "References"}, {"index": "5", "line": "See also"}]
        self.assertEqual(src.evaluation.references.choose_sections(sections)[0], ["1", "3"])

    def test_a_reference_list_is_cut_from_its_first_entry_and_a_citation_in_a_sentence_is_kept(self):
        text = ["Pigmented SCC shows blue structures (Kittler et al., 2008). Vessels are not seen.",
                "Argenziano, G., Zalaudek, I., 2004, Vascular structures. Arch Dermatol, 140, 1485–9.",
                "Kreusch, J. & Koch, F., 1996, Gefassmuster. Hautarzt, 47, 264–72."]
        self.assertEqual(src.evaluation.references.without_bibliography(text),
                         (["Pigmented SCC shows blue structures (Kittler et al., 2008). Vessels are not seen."], 21))
        self.assertEqual(src.evaluation.references.without_bibliography(["An Atlas of Dermoscopy, 2nd edition. Text after it."]), ([], 9))
        self.assertEqual(src.evaluation.references.without_bibliography(["Plain text. No list."]), (["Plain text. No list."], 0))
        self.assertTrue(src.retrieval.corpus.BIBLIOGRAPHY.search("Kreusch, J.F., 2002, Vascular patterns."))
        self.assertTrue(src.retrieval.corpus.BIBLIOGRAPHY.search("Kreusch, J. & Koch, F., 1996, Gefassmuster."))
        self.assertFalse(src.retrieval.corpus.BIBLIOGRAPHY.search("Types A. & B. are rare."))           # an author needs the comma
        self.assertFalse(src.retrieval.corpus.BIBLIOGRAPHY.search("Erythema: Structureless pale-red areas, as in AK."))
        self.assertFalse(src.retrieval.corpus.BIBLIOGRAPHY.search("Blue nevi, also known as Jadassohn-Tieche nevi, are common."))

    def test_the_text_of_each_table_of_a_section_without_its_subsections(self):
        section = ("<h2>Criteria</h2><p>prose</p><table><tr><td>Atypical network</td><td>Blue-white veil</td></tr>"
                   "<tr><td><table><tr><td>nested</td></tr></table></td></tr></table><p>more</p>"
                   "<table><tr><th>Regression</th></tr></table><h3>Sub</h3><table><tr><td>sub table</td></tr></table>")
        self.assertEqual(src.evaluation.references.table_texts(section), ["Atypical network Blue-white veil nested", "Regression"])
        self.assertEqual(src.evaluation.references.table_texts("<p>no table</p>"), [])

    def test_the_text_left_out_other_than_tables(self):
        section = ('<h2><span class="mw-headline">Criteria</span><span class="mw-editsection">edit</span></h2>'
                   '<p>read <b>bold</b></p><ul><li>item</li></ul><table><tr><td>table</td></tr></table>'
                   '<div class="thumb"><img alt="Atypical network" src="x.jpg"><div class="thumbcaption">Blue-white veil'
                   '<br>and regression</div></div><dl><dd>Irregular dots</dd></dl>loose text'
                   '<h3>Sub</h3><div>sub div</div>')
        self.assertEqual(src.evaluation.references.skipped_texts(section), [("img alt", "Atypical network"),
                                                     ("div.thumbcaption", "Blue-white veil and regression"),
                                                     ("dd", "Irregular dots"), ("(no element)", "loose text")])
        self.assertEqual(src.evaluation.references.skipped_texts("<p>only a paragraph</p>"), [])
        self.assertEqual(src.evaluation.references.skipped_texts('<div class="x">caption<span class="mw-editsection">[edit]</span></div>before'
                                          '<div>after</div>'), [("div.x", "caption"), ("(no element)", "before"), ("div", "after")])

    def test_show_skipped_prints_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(src.evaluation.references, "Fetcher", FakeWiki), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(src.evaluation.references.references_main(["--show-skipped", "--cache", os.path.join(d, "cache"), "--out", os.path.join(d, "r.json")]))
            self.assertFalse(os.path.exists(os.path.join(d, "r.json")))
        self.assertIn("===== MEL: Superficial Spreading Melanoma / from 'Dermoscopic criteria': 0 pieces of text left out",
                      out.getvalue())

    def test_show_tables_prints_the_tables_of_the_chosen_sections_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(src.evaluation.references, "Fetcher", FakeWiki), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(src.evaluation.references.references_main(["--show-tables", "--cache", os.path.join(d, "cache"), "--out", os.path.join(d, "r.json")]))
            self.assertFalse(os.path.exists(os.path.join(d, "r.json")))
        text = out.getvalue()
        self.assertIn("===== MEL: Superficial Spreading Melanoma / from 'Dermoscopic criteria': 1 tables", text)
        self.assertIn("[Dermoscopic criteria, 2 words] table text", text)
        self.assertIn("===== VASC: Angioma / (introduction): 0 tables", text)

    def test_empty_figure_references_are_removed(self):
        self.assertEqual(src.evaluation.references.text_paragraphs("<p>Blue dots (Figure ). This is BCC. . Then (see fig. ) more.</p>"),
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
        q = {"id": "ISIC_0000001", "gold_label": "BCC", "group_id": "g", "options": {"A": src.utils.OPTION_TEXT["BCC"]}}
        second = self.refs["BCC"]["texts"][1]["text"]
        r = src.evaluation.metrics.semantic_similarity({"ISIC_0000001": q}, {"ISIC_0000001": {"pred": "A", "rationale": second}}, self.refs,
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
            longest, limit = src.evaluation.references.token_check({"MEL": {"texts": [{"text": "a b c"}, {"text": " ".join("x" * 13)}]}},
                                            src.utils.get_logger("reference_texts"))
        self.assertEqual((longest, limit), (13, 12))
        self.assertIn("1 texts are longer", out.getvalue())
        self.assertIn("rebuild with a lower --max-words", out.getvalue())
        with mock.patch.dict(sys.modules, {"sentence_transformers": fake}), contextlib.redirect_stdout(io.StringIO()) as out:
            src.evaluation.references.token_check({"MEL": {"texts": [{"text": "a b c"}]}}, src.utils.get_logger("reference_texts"))
        self.assertNotIn("rebuild", out.getvalue())


# ----------------------------------------------------------------------------------------------------------------------
# a whole run with a fake model, then evaluated
# ----------------------------------------------------------------------------------------------------------------------


def run_cases(n, view_has_image):
    labels = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    return [{"isic_id": f"ISIC_{i:07d}", "split": "val", "label": labels[i % 8], "group_id": f"ISIC_{i:07d}",
             "clinical_text": "Patient: 50-year-old male. Skin lesion on the trunk.", "image_path": f"processed/images/ISIC_{i:07d}.jpg"}
            for i in range(n)]


class FakeClient:
    """Answers right for every second question, wrong for the others, with no letter for the third, and can fail on demand."""
    def __init__(self, questions, fail_at=None, refuse_at=None):
        self.by_text = {src.pipelines.base.user_text(q, view=v): q for q in questions for v in src.pipelines.base.FINDINGS_IN}
        self.calls, self.images, self.texts, self.fail_at, self.refuse_at = 0, [], [], fail_at, refuse_at

    def ask(self, system, text, image):
        self.calls += 1
        self.texts.append(text)
        if self.fail_at and self.calls == self.fail_at:
            raise src.models.llm.ApiError("HTTP 402: no credit")
        if self.refuse_at and self.calls == self.refuse_at:
            raise src.models.llm.Refused("refused by the host's content moderation")
        self.images.append(image)
        q = self.by_text[text]
        k = int(q["id"][-3:])
        if k == 2:
            return {"text": "I cannot help with that.", "finish_reason": "refusal", "tokens_in": 10, "tokens_out": 5}
        letter = q["answer"] if k % 2 == 0 else next(x for x in q["options"] if x != q["answer"])
        return {"text": f"Reasoning and evidence: Some findings of case {k}.\nAnswer: {letter} ({q['options'][letter]})",
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
        log = src.utils.get_logger(f"test.{id(self)}.{args.view}")
        with contextlib.redirect_stdout(io.StringIO()):               # keep the test output quiet
            return src.pipelines.base.run(args, client, log)

    def folder(self, view="text_only", partial=False):
        final = os.path.join(self.results, "llm_only", view, "gpt-4o_val")
        return final + ".partial" if partial else final

    def test_a_full_run_saves_every_answer_and_ends_complete(self):
        out = self.go(self.args(), FakeClient(self.questions))
        self.assertEqual(out, self.folder())
        self.assertFalse(os.path.exists(self.folder(partial=True)))
        preds = src.utils.read_jsonl(os.path.join(out, "predictions.jsonl"))
        self.assertEqual([p["id"] for p in preds], [q["id"] for q in src.pipelines.base.asking_order(self.questions)])
        self.assertEqual(sorted(p["id"] for p in preds), sorted(q["id"] for q in self.questions))
        info = json.loads(Path(out, "run.json").read_text(encoding="utf-8"))
        self.assertEqual((info["status"], info["answered"], info["of"]), ("complete", 6, 6))
        self.assertEqual(info["settings"]["prompt"], "llm_only")
        self.assertIn("(<the name of that option>)", info["settings"]["instruction"])      # the exact prompt text is saved with the run
        self.assertIn("42:<case id>", info["question_order"])
        refused = next(p for p in preds if p["id"].endswith("002"))
        self.assertEqual((refused["pred"], refused["valid"], refused["finish_reason"]), (None, False, "refusal"))
        first = preds[0]
        self.assertEqual((first["valid"], first["rationale"]), (True, f"Some findings of case {int(first['id'][-3:])}."))
        self.assertEqual((first["prompt"], first["model"]), ("llm_only", "gpt-4o"))                  # every answer says which prompt asked it
        self.assertFalse({"top3", "evidence_sufficient"} & set(first))                         # nothing of the old format is saved

    def test_the_predictions_go_straight_into_evaluate(self):
        out = self.go(self.args(), FakeClient(self.questions))
        qa = {q["id"]: q for q in self.questions}
        pred = src.evaluation.evaluator.load(os.path.join(out, "predictions.jsonl"))
        d = src.evaluation.metrics.diagnosis_metrics(qa, pred)
        self.assertEqual((d["accuracy"], d["invalid_answers"]), (0.3333, 1))    # ids 0 and 4 right; 2 has no letter
        self.assertFalse({"coverage", "selective_risk", "n_covered"} & set(d))      # no yes/no flag any more, so no such score
        rationale = [p["rationale"] for p in pred.values() if p["valid"]]
        self.assertEqual(len(rationale), 5)                                        # the reasoning of every valid answer is kept

    def test_the_asking_order_is_fixed_mixed_and_the_same_for_every_view(self):
        qs = [{"id": f"ISIC_{k:07d}", "gold_label": c} for k, c in enumerate(["AK"] * 10 + ["BCC"] * 10 + ["MEL"] * 10)]
        order = [q["id"] for q in src.pipelines.base.asking_order(qs)]
        self.assertEqual(order, [q["id"] for q in src.pipelines.base.asking_order(list(reversed(qs)))])    # the file order does not matter
        self.assertEqual(sorted(order), [q["id"] for q in qs])                              # every question once
        self.assertNotEqual(order, [q["id"] for q in qs])
        first = {q["gold_label"] for q in src.pipelines.base.asking_order(qs)[:5]}
        self.assertGreater(len(first), 1)                                                   # a trial of 5 is not one class
        self.assertEqual(order[:3], ["ISIC_0000003", "ISIC_0000026", "ISIC_0000015"])        # fixed: seed 42

    def test_a_limited_run_asks_the_first_questions_of_the_asking_order(self):
        self.go(self.args(limit=2), FakeClient(self.questions))
        preds = src.utils.read_jsonl(os.path.join(self.folder(partial=True), "predictions.jsonl"))
        self.assertEqual([p["id"] for p in preds], [q["id"] for q in src.pipelines.base.asking_order(self.questions)[:2]])
        info = json.loads(Path(self.folder(partial=True), "run.json").read_text(encoding="utf-8"))
        self.assertEqual(info["question_order"], src.pipelines.base.ORDER)

    def test_an_interrupted_run_continues_where_it_stopped_without_repeating(self):
        self.go(self.args(limit=2), FakeClient(self.questions))
        self.assertTrue(os.path.isdir(self.folder(partial=True)))
        info = json.loads(Path(self.folder(partial=True), "run.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status"], "unfinished")
        client = FakeClient(self.questions)
        self.go(self.args(), client)
        self.assertEqual(client.calls, 4)                                       # only the 4 left
        preds = src.utils.read_jsonl(os.path.join(self.folder(), "predictions.jsonl"))
        self.assertEqual(len(preds), 6)
        self.assertEqual(len({p["id"] for p in preds}), 6)

    def test_an_api_error_stops_the_run_and_keeps_what_was_answered(self):
        with self.assertRaises(SystemExit):
            self.go(self.args(), FakeClient(self.questions, fail_at=4))
        preds = src.utils.read_jsonl(os.path.join(self.folder(partial=True), "predictions.jsonl"))
        self.assertEqual(len(preds), 3)
        info = json.loads(Path(self.folder(partial=True), "run.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status"], "stopped")
        self.go(self.args(), FakeClient(self.questions))
        self.assertEqual(len(src.utils.read_jsonl(os.path.join(self.folder(), "predictions.jsonl"))), 6)

    def test_an_unfinished_run_is_not_continued_with_other_settings(self):
        self.go(self.args(limit=2), FakeClient(self.questions))
        with self.assertRaises(SystemExit):
            self.go(self.args(style="answer_only"), FakeClient(self.questions))

    def test_a_question_refused_by_the_host_is_saved_as_invalid_and_the_run_goes_on(self):
        self.go(self.args(), FakeClient(self.questions, refuse_at=2))
        preds = src.utils.read_jsonl(os.path.join(self.folder(), "predictions.jsonl"))
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

    def test_each_view_sends_and_records_its_own_wording_about_where_findings_may_come_from(self):
        for view, wording, absent in (("text_only", "in the patient facts, that support", "image"),
                                      ("image_only", "in the image, that support", "patient facts"),
                                      ("multimodal", "in the image and/or in the patient facts", None)):
            with self.subTest(view):
                self.make(view, 2)
                client = FakeClient(self.questions)
                self.go(self.args(view=view), client)
                self.assertTrue(client.texts and all(wording in t for t in client.texts))
                if absent:
                    self.assertFalse(any(absent in t.split("Reply in exactly this form")[1] for t in client.texts))
                info = json.loads(Path(self.folder(view), "run.json").read_text(encoding="utf-8"))
                self.assertIn(wording, info["settings"]["instruction"])      # the saved settings show what was really asked

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
        argv = ["--model", "gpt-4o", "--allow-paid", "--view", "text_only", "--data", self.data, "--results", self.results,
                "--env-file", os.path.join(self.tmp.name, ".env")]
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(SystemExit), \
                contextlib.redirect_stdout(io.StringIO()):
            src.pipelines.llm_only.llm_only_main(argv)
        self.assertFalse(os.path.exists(self.results))                  # nothing is left behind
        Path(self.tmp.name, ".env").write_text("# keys\nOPENROUTER_API_KEY = 'SECRET123'\n", encoding="utf-8")
        fake = FakeClient(self.questions)
        made = []

        def client(cfg, key, log=None):
            made.append(key)
            return fake

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(src.models.llm, "OpenRouterClient", client), \
                contextlib.redirect_stdout(io.StringIO()):
            src.pipelines.llm_only.llm_only_main(argv)
        self.assertEqual(made, ["SECRET123"])                         # read from .env, quotes removed
        written = "".join(p.read_text(encoding="utf-8") for p in Path(self.results).rglob("*") if p.is_file())
        self.assertIn("gpt-4o", written)
        self.assertNotIn("SECRET123", written)

    def test_claude_takes_its_key_from_either_name(self):
        with mock.patch.dict(os.environ, {"CLAUDE_API_KEY": "K1"}, clear=True):
            self.assertEqual(src.models.llm.api_key("anthropic"), "K1")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "K2", "CLAUDE_API_KEY": "K1"}, clear=True):
            self.assertEqual(src.models.llm.api_key("anthropic"), "K2")


# ----------------------------------------------------------------------------------------------------------------------
# the evaluate command
# ----------------------------------------------------------------------------------------------------------------------


def make_question(case_id, gold):
    options, letter = options_for(case_id)
    return {"id": case_id, "options": options, "answer": letter[gold], "gold_label": gold, "group_id": case_id,
            "question": f"Question {case_id}"}


class EvidenceCommandTests(unittest.TestCase):
    """A RAG run's predictions go through the evaluation with evidence, with and without a support model."""

    def files(self, d):
        qs = [make_question(f"ISIC_{i:07d}", c) for i, c in enumerate(["MEL", "NV", "BCC", "AK"])]
        letter = {q["id"]: next(k for k, v in q["options"].items() if v == src.utils.OPTION_TEXT[q["gold_label"]]) for q in qs}
        text = "Melanomas show asymmetry and irregular borders here. Nevi are usually symmetric and evenly coloured overall."
        preds = []
        for k, q in enumerate(qs):
            quote = ("Melanomas show asymmetry and irregular borders here" if k < 2 else
                     "Nothing in the passage says this at all")
            preds.append({"id": q["id"], "pred": letter[q["id"]], "contexts": [{"chunk_id": "c1", "text": text,
                                                                               "tags": [q["gold_label"]]}],
                          "evidence": {"chunk_id": "c1", "quote": quote}})
        paths = {k: os.path.join(d, f"{k}.jsonl") for k in ("qa", "pred")}
        write_jsonl(paths["qa"], qs)
        write_jsonl(paths["pred"], preds)
        return paths

    def run_report(self, extra):
        with tempfile.TemporaryDirectory() as d:
            paths = self.files(d)
            out = os.path.join(d, "report.json")
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(src.evaluation.evaluator, "relevance_metrics", lambda *a, **k: {}):
                src.evaluation.evaluator.report_main(["--questions", paths["qa"], "--predictions", paths["pred"], "--out", out, "--skip-text", *extra])
            return json.loads(Path(out).read_text(encoding="utf-8"))

    def test_the_report_has_an_evidence_section_with_the_quote_check(self):
        rep = self.run_report([])
        e = rep["evidence"]
        self.assertEqual((e["evidence_given"], e["n_covered"], e["selective_risk_verified"]), (4, 2, 0.0))
        self.assertEqual(e["coverage_verified"], 0.5)                      # 2 of the 4 answers have a quote that is in its passage

    def test_the_report_has_recall_at_k_when_the_corpus_is_given(self):
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(os.path.join(d, "passages.jsonl"), [{"tags": ["MEL"]}] * 4 + [{"tags": ["NV"]}] * 4)
            rep = self.run_report(["--passages", os.path.join(d, "passages.jsonl")])
        self.assertEqual(rep["retrieval"]["recall@1"], 0.125)        # 4 questions; MEL and NV each find 1 of 4: (.25 + .25 + 0 + 0) / 4
        self.assertNotIn("recall@1", self.run_report([])["retrieval"])

    def test_the_support_check_command_and_the_nli_options_are_gone(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            src.evaluation.evaluator.report_main(["--questions", "q", "--predictions", "p", "--support-model"])


class CollectTests(unittest.TestCase):
    """collect: a report for every finished run and one table of all runs (decided 2026-11-24)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.qs = [make_question(f"ISIC_{i:07d}", c) for i, c in enumerate(CLASSES)]
        self.qpath = os.path.join(self.root, "data", "val.jsonl")
        write_jsonl(self.qpath, self.qs)

    def tearDown(self):
        self.tmp.cleanup()

    def run_folder(self, model, name, status="complete", right=4, view="text_only"):
        run_view, _, split = name.partition(".")[0].rpartition("_")             # "text_only_val.partial" -> text_only, val
        folder = os.path.join(self.root, "results", "llm_only", run_view, f"{model}_{split}" + name[len(run_view) + 1 + len(split):])
        write_jsonl(os.path.join(folder, "predictions.jsonl"),
                    [{"id": q["id"], "pred": q["answer"] if k < right else "Z", "view": view, "rationale": "Patient facts only.",
                      "tokens_in": 100, "tokens_out": 20, "seconds": 1.5, "served_by": f"{model}@host"} for k, q in enumerate(self.qs)])
        Path(folder, "run.json").write_text(json.dumps({"settings": {"baseline": "llm_only", "model": model, "view": view, "split": "val",
                                                                    "questions": "data/val.jsonl", "questions_sha256": "abc",
                                                                    "model_settings": {"id": f"vendor/{model}"}},
                                                       "status": status, "answered": 8, "of": 8}), encoding="utf-8")
        return folder

    def collect(self, **kw):
        out = os.path.join(self.root, "results", "summary.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            return src.evaluation.evaluator.collect(os.path.join(self.root, "results"), out, self.root, skip_text=True, **kw), out

    def test_a_report_made_on_windows_is_reused_and_a_changed_answer_is_scored_again(self):
        folder = self.run_folder("gpt-4o", "text_only_val")
        ppath = os.path.join(folder, "predictions.jsonl")
        Path(ppath).write_bytes(Path(ppath).read_bytes().replace(b"\n", b"\r\n"))             # as Windows saved it
        self.collect()
        report = Path(folder, "report.json")
        crlf_sha = src.utils.file_sha256(ppath)
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["inputs"]["predictions"]["sha256"],
                         src.utils.text_sha256(ppath))                              # recorded with LF endings now
        saved = json.loads(report.read_text(encoding="utf-8"))
        saved["inputs"]["predictions"]["sha256"] = crlf_sha                                     # a report from before 2026-11-25
        saved["marker"] = "kept"
        report.write_text(json.dumps(saved), encoding="utf-8")
        Path(ppath).write_bytes(Path(ppath).read_bytes().replace(b"\r\n", b"\n"))             # the same file from Git (LF)
        with mock.patch.object(src.evaluation.evaluator, "evaluate", side_effect=AssertionError("scored again")):
            self.collect()
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["marker"], "kept")
        rows = read_jsonl(ppath)
        rows[0]["pred"] = "Z" if rows[0]["pred"] != "Z" else "A"
        write_jsonl(ppath, rows)
        self.collect()                                                                          # a real change: scored again
        self.assertNotIn("marker", json.loads(report.read_text(encoding="utf-8")))

    def test_a_row_for_every_finished_run_and_none_for_trials_or_unfinished_runs(self):
        done = self.run_folder("gpt-4o", "text_only_val")
        self.run_folder("gemini-3.8-flash", "text_only_val.partial")                  # a trial
        self.run_folder("llama-4-maverick", "text_only_val", status="stopped")         # unfinished
        rows, out = self.collect()
        self.assertEqual([(r["model"], r["view"], r["split"]) for r in rows], [("gpt-4o", "text_only", "val")])
        r = rows[0]
        self.assertEqual((r["accuracy"], r["questions"], r["invalid"], r["served_by"], r["model_id"]),
                         (0.5, 8, 4, "gpt-4o@host", "vendor/gpt-4o"))
        self.assertEqual((r["tokens_in"], r["tokens_out"], r["seconds"]), (800.0, 160.0, 12.0))
        self.assertTrue(os.path.isfile(os.path.join(done, "report.json")))
        self.assertEqual(r["text_only_image_claim_rate"], 0.0)
        table = Path(out).read_text(encoding="utf-8").splitlines()
        self.assertEqual(table[0].split(",")[:4], ["system", "model", "model_id", "view"])
        self.assertEqual(len(table), 2)
        self.assertLess(r["accuracy_ci_low"], r["accuracy"])
        self.assertGreater(r["accuracy_ci_high"], r["accuracy"])                      # the two ends of the interval are the right way round
        self.assertEqual(r["refusals"], 0)
        refused = self.run_folder("claude-sonnet-5-5", "text_only_val", right=6)
        lines = [json.loads(x) for x in Path(refused, "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        lines[7].update({"pred": None, "finish_reason": "refusal", "raw_output": "I cannot help with that."})
        write_jsonl(os.path.join(refused, "predictions.jsonl"), lines)
        by_model = {x["model"]: x for x in self.collect(force=True)[0]}
        self.assertEqual((by_model["claude-sonnet-5-5"]["refusals"], by_model["claude-sonnet-5-5"]["invalid"]), (1, 2))
        long = src.utils.read_csv(os.path.join(os.path.dirname(out), "metrics.csv"))       # every number of the report, one row each
        acc = [r for r in long if r["metric"] == "diagnosis.accuracy"]
        self.assertEqual([(r["model"], r["view"], r["split"], float(r["value"])) for r in acc], [("claude-sonnet-5-5", "text_only", "val", 0.75), ("gpt-4o", "text_only", "val", 0.5)])
        self.assertTrue(any(r["metric"] == "diagnosis.accuracy_ci_low" for r in long))
        self.assertTrue(any(r["metric"].startswith("diagnosis.per_class.MEL.") for r in long))
        self.assertFalse(any("ids" in r["metric"] or "confusion" in r["metric"] or "inputs" in r["metric"] for r in long))

    def test_the_report_is_reused_until_the_predictions_change(self):
        done = self.run_folder("gpt-4o", "text_only_val", right=4)
        self.collect()
        report = os.path.join(done, "report.json")
        first = Path(report).read_text(encoding="utf-8")
        before = os.path.getmtime(report)
        self.collect()
        self.assertEqual(os.path.getmtime(report), before)                            # nothing changed: not written again
        self.run_folder("gpt-4o", "text_only_val", right=6)
        rows, _ = self.collect()
        self.assertEqual(rows[0]["accuracy"], 0.75)
        self.assertNotEqual(Path(report).read_text(encoding="utf-8"), first)
        self.assertEqual(self.collect(force=True)[0][0]["accuracy"], 0.75)

    def test_a_missing_question_file_stops_the_collection(self):
        self.run_folder("gpt-4o", "text_only_val")
        os.remove(self.qpath)
        with self.assertRaises(FileNotFoundError):
            self.collect()

    def test_no_results_folder_gives_an_empty_table(self):
        rows, out = self.collect()
        self.assertEqual(rows, [])
        self.assertEqual(len(Path(out).read_text(encoding="utf-8").splitlines()), 1)

    def test_the_similarity_check_command_reads_finished_runs(self):
        SIM_WORD = {"MEL": "melanoma", "NV": "nevus", "BCC": "basal", "AK": "actinic", "BKL": "seborrheic", "DF": "dermatofibroma",
                    "VASC": "vascular", "SCC": "squamous"}

        def class_word_encoder(texts):
            return [[t.lower().split().count(SIM_WORD[c]) for c in CLASSES] for t in texts]
        self.run_folder("gpt-4o", "text_only_val")
        self.run_folder("gemini-3.8-flash", "text_only_val.partial")                  # a trial: not read
        refs = os.path.join(self.root, "refs.json")
        Path(refs).write_text(json.dumps({c: f"typical {w} appearance" for c, w in SIM_WORD.items()}), encoding="utf-8")
        out = os.path.join(self.root, "sim.json")
        with mock.patch.object(src.evaluation.evaluator, "sentence_encoder", lambda *a, **k: class_word_encoder), contextlib.redirect_stdout(io.StringIO()):
            res = src.evaluation.evaluator.similarity_check_main(["--results", os.path.join(self.root, "results"), "--root", self.root,
                           "--reference-texts", refs, "--out", out])
        self.assertEqual(list(res["runs"]), ["llm_only/gpt-4o/text_only_val"])
        self.assertEqual(json.loads(Path(out).read_text(encoding="utf-8"))["runs"], res["runs"])
        self.assertFalse(res["runs"]["llm_only/gpt-4o/text_only_val"]["passed"])      # the made-up reasoning "Patient facts only." names no disease

    def test_a_run_that_recorded_the_question_file_before_the_move_finds_it_through_the_provenance_record(self):
        folder = self.run_folder("gpt-4o", "text_only_val")
        info = json.loads(Path(folder, "run.json").read_text(encoding="utf-8"))
        info["settings"]["questions"] = "data/views/text_only/val.jsonl"                     # the name before 2026-11-25
        Path(folder, "run.json").write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            self.collect()
        now = os.path.join(self.root, "data", question_file("text_only", "val"))
        os.makedirs(os.path.dirname(now))
        shutil.copy(self.qpath, now)
        Path(self.root, "data", "views", "provenance.json").write_text(json.dumps({"files": {question_file("text_only", "val"): {
            "previous_path": "views/text_only/val.jsonl", "previous_sha256": "a", "sha256": "b"}}}), encoding="utf-8")
        rows, _ = self.collect()
        self.assertEqual([(r["model"], r["split"], r["accuracy"]) for r in rows], [("gpt-4o", "val", 0.5)])

    def test_the_command_runs_from_main(self):
        self.run_folder("gpt-4o", "text_only_val")
        with contextlib.redirect_stdout(io.StringIO()):
            rows = src.evaluation.evaluator.collect_main(["--results", os.path.join(self.root, "results"), "--out", os.path.join(self.root, "s.csv"),
                            "--root", self.root, "--skip-text"])
        self.assertEqual(len(rows), 1)


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
                src.evaluation.evaluator.report_main(["--questions", paths["qa"], "--predictions", paths["pred"], "--out", out, "--skip-text"])
            with open(out, encoding="utf-8") as f:
                rep = json.load(f)
            sha = src.utils.text_sha256(paths["qa"])          # LF endings, also on Windows
        self.assertEqual((rep["diagnosis"]["accuracy"], rep["diagnosis"]["missing_predictions"]), (0.5, 4))
        self.assertIsNone(rep["retrieval"])
        self.assertEqual(rep["inputs"]["questions"]["sha256"], sha)  # the report says which question file it used

    def test_the_report_counts_text_only_reasonings_that_describe_an_image(self):
        qs = [make_question(f"ISIC_{i:07d}", c) for i, c in enumerate(CLASSES)]
        said = ["Dermoscopy reveals a pigment network.", "The patient is 45 years old.", "The image shows a scar-like patch."]
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(os.path.join(d, "qa.jsonl"), qs)
            write_jsonl(os.path.join(d, "pred.jsonl"), [{"id": q["id"], "pred": q["answer"], "view": "text_only", "rationale": r}
                                                        for q, r in zip(qs, said)])
            out = os.path.join(d, "r.json")
            with contextlib.redirect_stdout(io.StringIO()):
                src.evaluation.evaluator.report_main(["--questions", os.path.join(d, "qa.jsonl"), "--predictions", os.path.join(d, "pred.jsonl"), "--out", out,
                         "--skip-text", "--answered-only"])
            claims = json.loads(Path(out).read_text(encoding="utf-8"))["text_only_image_claims"]
        self.assertEqual((claims["n_text_only_with_reasoning"], claims["n_claiming_an_image"]), (3, 2))

    def test_answered_only_scores_a_trial_on_the_questions_that_were_asked(self):
        qs = [make_question(f"ISIC_{i:07d}", c) for i, c in enumerate(CLASSES)]
        with tempfile.TemporaryDirectory() as d:
            write_jsonl(os.path.join(d, "qa.jsonl"), qs)
            write_jsonl(os.path.join(d, "pred.jsonl"), [{"id": q["id"], "pred": q["answer"]} for q in qs[:3]])
            reports = {}
            for name, extra in (("all", []), ("trial", ["--answered-only"])):
                out = os.path.join(d, f"{name}.json")
                with contextlib.redirect_stdout(io.StringIO()):
                    src.evaluation.evaluator.report_main(["--questions", os.path.join(d, "qa.jsonl"), "--predictions", os.path.join(d, "pred.jsonl"),
                             "--out", out, "--skip-text", *extra])
                reports[name] = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual((reports["all"]["diagnosis"]["accuracy"], reports["all"]["trial_answered_only"]), (0.375, False))
        trial = reports["trial"]
        self.assertEqual((trial["diagnosis"]["accuracy"], trial["diagnosis"]["n_questions"], trial["diagnosis"]["missing_predictions"]),
                         (1.0, 3, 0))
        self.assertTrue(trial["trial_answered_only"])                  # the report is marked, so a trial is never read as a result


# ----------------------------------------------------------------------------------------------------------------------
# standard RAG: a whole run with a fake model and a fake index, the index command, and the report on top (2026-11-24)
# ----------------------------------------------------------------------------------------------------------------------
RAG_WORDS = ["trunk", "head", "male", "female", "melanoma", "nevus", "vessels", "brown"]
RAG_CORPUS = [
    {"passage_id": f"p{i}", "title": t, "text": x, "tags": g, "url": u, "library": "pmc", "doc_id": f"d{i}", "licence": "CC BY 4.0",
     "query_class": g[0] if g else "MEL", "n_words": len(x.split()), "md5": f"m{i}", "year": "", "topics": []}
    for i, (t, x, g, u) in enumerate([
        ("Melanoma of the trunk", "Melanoma on the trunk of a male patient. It is brown.", ["MEL"], "https://x/a"),
        ("Head lesions", "Lesions of the head in a female patient.", ["SCC"], "https://x/b"),
        ("Vessels", "Vessels are seen in a nevus. A second sentence follows here.", ["NV"], "https://x/c"),
        ("Trunk nevus", "A nevus on the trunk of a male. It is brown.", ["NV"], "https://x/d"),
        ("Brown areas", "A brown area on the trunk is common.", [], "https://x/e")])]


def rag_vec(text):
    words = text.lower().replace(".", " ").replace(",", " ").split()
    counts = [words.count(w) for w in RAG_WORDS]
    norm = sum(c * c for c in counts) ** 0.5 or 1.0
    return [c / norm for c in counts]


def rag_enc(items):
    return [rag_vec(" ".join(x) if isinstance(x, list) else x) for x in items]


RAG_IMAGES = [{"passage_id": f"v{i}", "item_id": f"v{i}", "image_file": f"v{i}.jpg", "title": "Figure", "text": f"Dermoscopy of {name}. More.",
               "tags": [tag]} for i, (name, tag) in enumerate([("a melanoma", "MEL"), ("a nevus", "NV"), ("a carcinoma", "BCC")])]


class FakeImageVectors(dict):
    """The saved vector of every question image: [1, 0] for an even case number, [0, 1] for an odd one."""

    def __contains__(self, name):
        return True

    def __getitem__(self, name):
        return [1.0, 0.0] if int(name.split("_")[-1].split(".")[0]) % 2 == 0 else [0.0, 1.0]


def fake_visual():
    from array import array

    from src.retrieval.indexer import DenseIndex
    return (DenseIndex(list(RAG_IMAGES), array("f", [1, 0, 0, 1, 0.6, 0.8]), 2, {"encoder": "fake-image", "passages_sha256": "img"}),
            FakeImageVectors())


def make_rag(k=3, visual=True):
    """The standard RAG's retrieval (BM25 candidates, re-ordered by a fake re-ranker that keeps their order) with a fake text index and image index."""
    from array import array

    from src.retrieval.indexer import DenseIndex
    flat = array("f")
    for p in RAG_CORPUS:
        flat.extend(rag_vec(p["title"] + " " + p["text"]))
    return src.pipelines.standard_rag.StandardRag(DenseIndex(list(RAG_CORPUS), flat, len(RAG_WORDS), {"encoder": "fake", "passages_sha256": "abc"}),
                          rag_enc, k, src.pipelines.base.APPROVED_RETRIEVAL, rerank=lambda q, t: [float(-i) for i in range(len(t))],
                          visual=fake_visual() if visual else None)


class FakeRagClient:
    """Answers right for even case numbers. Evidence: the first sentence of passage 1 (case number 0, 4, 8...), none (odd numbers), a
    sentence that is not in the passage (2, 6, 10...), or no evidence line at all (every case number that ends in 2 or 7)."""

    def __init__(self, questions):
        self.heads, self.texts, self.images = {}, [], []
        for q in questions:
            options = "\n".join(f"{letter}. {text}" for letter, text in q["options"].items())
            self.heads[f"{q['question']}\n\nOptions:\n{options}"] = q
        self.questions = questions

    def ask(self, system, text, image):
        self.texts.append(text)
        self.images.append(image)
        q = self.heads[text.split("\n\n" + src.pipelines.base.PASSAGES_HEADER)[0]]
        k = int(q["id"][-3:])
        letter = q["answer"] if k % 2 == 0 else next(x for x in q["options"] if x != q["answer"])
        first = text.split("[1] ", 1)[1].split("\n", 1)[1].split(".")[0] + "."
        evidence = ('Evidence: none' if k % 2 else
                    'Evidence: [1] "A sentence that is not in the passage at all."' if k % 4 == 2 else f'Evidence: [1] "{first}"')
        body = f"Reasoning and evidence: Findings of case {k} [1].\nAnswer: {letter} ({q['options'][letter]})"
        if k % 10 not in (2, 7):
            body += "\n" + evidence
        return {"text": body, "finish_reason": "stop", "tokens_in": 1000, "tokens_out": 60, "served_by": "fake@host"}


class StandardRagRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data, self.results = os.path.join(self.tmp.name, "data"), os.path.join(self.tmp.name, "results")

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, view, n=8):
        self.questions = [question(c, view) for c in run_cases(n, view != "text_only")]
        write_jsonl(os.path.join(self.data, question_file(view, "val")), self.questions)
        if view != "text_only":
            for q in self.questions:
                Path(self.data, q["image_path"]).parent.mkdir(parents=True, exist_ok=True)
                Path(self.data, q["image_path"]).write_bytes(b"\xff\xd8\xff fake jpeg")

    def args(self, view="text_only", **kw):
        base = dict(model="gpt-4o", view=view, split="val", limit=0, sleep=0.0, dry_run=False, data=self.data, results=self.results,
                    env_file=os.path.join(self.tmp.name, ".env"))
        base.update(kw)
        return argparse.Namespace(**base)

    def go(self, args, client, rag):
        log = src.utils.get_logger(f"test.rag.{id(self)}.{args.view}")
        with contextlib.redirect_stdout(io.StringIO()):
            return src.pipelines.base.run(args, client, log, rag)

    def lines(self, folder):
        return [json.loads(x) for x in Path(folder, "predictions.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_a_run_saves_the_query_the_passages_and_the_evidence_of_every_answer(self):
        self.make("text_only")
        client = FakeRagClient(self.questions)
        out = self.go(self.args(), client, make_rag())
        self.assertEqual(os.path.relpath(out, self.results).replace("\\", "/"), "standard_rag/text_only/gpt-4o_val")
        rows = {r["id"]: r for r in self.lines(out)}
        self.assertEqual(len(rows), 8)
        first = rows["ISIC_0000000"]
        self.assertEqual((first["prompt"], first["view"], first["valid"]), ("standard_rag", "text_only", True))
        self.assertEqual(first["query"], "Patient: 50-year-old male. Skin lesion on the trunk.")
        self.assertEqual([c["rank"] for c in first["contexts"]], [1, 2, 3])
        self.assertTrue(all({"chunk_id", "title", "text", "tags", "score"} <= set(c) for c in first["contexts"]))
        self.assertEqual(first["evidence"]["chunk_id"], first["contexts"][0]["chunk_id"])
        self.assertIsNone(rows["ISIC_0000001"]["evidence"])                                 # an explicit "Evidence: none"
        self.assertEqual(rows["ISIC_0000002"]["evidence"], dict(src.pipelines.base.UNPARSED))               # no evidence line
        self.assertEqual(rows["ISIC_0000006"]["evidence"]["quote"], "A sentence that is not in the passage at all.")
        info = json.loads(Path(out, "run.json").read_text(encoding="utf-8"))
        self.assertEqual((info["status"], info["settings"]["baseline"], info["settings"]["prompt"]), ("complete", "standard_rag", "standard_rag"))
        self.assertEqual((info["settings"]["retrieval"]["k"], info["settings"]["retrieval"]["query"]), (3, "question"))
        self.assertIn("Evidence: none", info["settings"]["instruction"])
        self.assertTrue(all(i is None for i in client.images))                              # a text-only run sends no image

    def test_the_report_scores_retrieval_and_evidence_on_top_of_the_diagnosis(self):
        self.make("text_only")
        out = self.go(self.args(), FakeRagClient(self.questions), make_rag())
        rep = src.evaluation.evaluator.evaluate(os.path.join(self.data, question_file("text_only", "val")), os.path.join(out, "predictions.jsonl"), skip_text=True)
        self.assertEqual(rep["diagnosis"]["n_questions"], 8)
        self.assertEqual(rep["retrieval"]["n_questions_with_contexts"], 8)
        for key in ("hit@3", "precision@3", "ndcg@3", "mrr"):
            self.assertIn(key, rep["retrieval"])
        e = rep["evidence"]
        # cases 0 and 4 quote their passage; 1, 3, 5 say none (odd); 2 and 7 have no line; 6 quotes a sentence that is not there
        self.assertEqual((e["n_questions"], e["evidence_none"], e["evidence_unparsed"]), (8, 3, 2))
        self.assertEqual(e["n_covered"], 2)
        self.assertEqual(e["coverage_verified"], 0.25)

    def test_an_image_view_sends_the_image_and_the_view_wording(self):
        self.make("multimodal", 4)
        client = FakeRagClient(self.questions)
        out = self.go(self.args(view="multimodal"), client, make_rag())
        self.assertTrue(client.images and all(i and i.startswith("/9j") for i in client.images))
        self.assertTrue(all("in the image and/or in the patient facts" in t for t in client.texts))
        info = json.loads(Path(out, "run.json").read_text(encoding="utf-8"))
        self.assertIn("in the image and/or in the patient facts", info["settings"]["instruction"])

    def test_an_image_only_run_searches_the_image_library_by_the_question_image(self):
        """Researcher's decision (2026-11-24, connected 2026-11-25): the image views search by the image itself."""
        self.make("image_only", 4)
        out = self.go(self.args(view="image_only"), FakeRagClient(self.questions), make_rag())
        rows = {r["id"]: r for r in self.lines(out)}
        self.assertTrue(all(r["query"] is None and r["query_image"] == f"{i}.jpg" for i, r in rows.items()))
        self.assertTrue(all(c["chunk_id"].startswith("v") and c["image_file"] for r in rows.values() for c in r["contexts"]))
        self.assertEqual([c["chunk_id"] for c in rows["ISIC_0000000"]["contexts"]], ["v0", "v2", "v1"])     # vector [1, 0]
        self.assertEqual([c["chunk_id"] for c in rows["ISIC_0000001"]["contexts"]], ["v1", "v2", "v0"])     # vector [0, 1]
        s = json.loads(Path(out, "run.json").read_text(encoding="utf-8"))["settings"]["retrieval"]
        self.assertEqual((s["query"], s["image_search"]), ("image", {"encoder": "fake-image", "items_sha256": "img", "n_items": 3}))

    def test_an_image_and_text_run_fuses_the_text_and_the_image_search_as_validated(self):
        self.make("multimodal", 4)
        rag = make_rag()
        out = self.go(self.args(view="multimodal"), FakeRagClient(self.questions), rag)
        for r in self.lines(out):
            q = next(x for x in self.questions if x["id"] == r["id"])
            text = [p for p, _ in rag.retrieve(r["query"], src.pipelines.standard_rag.FUSION_LIST)]
            image = [p for p, _ in rag.image_search(q, src.pipelines.standard_rag.FUSION_LIST)]
            expected = [p["passage_id"] for p, _ in reciprocal_rank_fusion([text, image], top=3)]
            self.assertEqual([c["chunk_id"] for c in r["contexts"]], expected)
            self.assertEqual({c["chunk_id"][0] for c in r["contexts"]}, {"p", "v"})                # both libraries reach the top 3
        s = json.loads(Path(out, "run.json").read_text(encoding="utf-8"))["settings"]["retrieval"]
        self.assertEqual(s["fusion"], {"method": "reciprocal rank fusion", "k": 60, "list": 10})

    def test_a_caption_found_by_both_searches_reaches_the_model_once(self):
        from array import array

        from src.retrieval.indexer import DenseIndex
        rag = make_rag()
        copy = {**RAG_IMAGES[0], "passage_id": "d0#fig1", "text": RAG_CORPUS[0]["text"]}           # a figure caption that is passage p0
        rag.visual = (DenseIndex([copy] + RAG_IMAGES[1:], array("f", [1, 0, 0, 1, 0.6, 0.8]), 2, {}), FakeImageVectors())
        self.make("multimodal", 4)
        for q in self.questions:
            texts = [" ".join(p["text"].lower().split()) for p, _ in rag.ranked(q, "multimodal", 3)]
            self.assertEqual(len(texts), 3)
            self.assertEqual(len(set(texts)), 3)

    def test_an_image_view_without_the_image_index_stops_before_any_question_is_asked(self):
        self.make("image_only", 2)
        client = FakeRagClient(self.questions)
        with self.assertRaises(ValueError):
            self.go(self.args(view="image_only"), client, make_rag(visual=False))
        self.assertEqual(client.texts, [])
        self.assertFalse(os.path.exists(self.results))                                           # no run folder is made

    def test_an_interrupted_run_continues_with_the_same_settings_and_refuses_other_ones(self):
        self.make("text_only")
        client = FakeRagClient(self.questions)
        self.go(self.args(limit=3), client, make_rag())
        folder = os.path.join(self.results, "standard_rag", "text_only", "gpt-4o_val.partial")
        self.assertEqual(len(self.lines(folder)), 3)
        with self.assertRaises(SystemExit):
            self.go(self.args(), FakeRagClient(self.questions), make_rag(k=2))            # another k: not the same run
        out = self.go(self.args(), FakeRagClient(self.questions), make_rag())
        self.assertEqual(len(self.lines(out)), 8)
        self.assertEqual(len({r["id"] for r in self.lines(out)}), 8)                      # nothing asked twice

    def test_a_dry_run_shows_the_query_and_the_passages_and_writes_nothing(self):
        self.make("text_only", 4)
        client = FakeRagClient(self.questions)
        buf = io.StringIO()
        log = src.utils.get_logger("test.rag.dry")
        with contextlib.redirect_stdout(buf):
            self.assertIsNone(src.pipelines.base.run(self.args(dry_run=True, limit=2), client, log, make_rag()))
        shown = buf.getvalue()
        self.assertIn("[query] Patient: 50-year-old male. Skin lesion on the trunk.", shown)
        self.assertIn(src.pipelines.base.PASSAGES_HEADER, shown)
        self.assertEqual(client.texts, [])
        self.assertFalse(os.path.exists(self.results))

    def test_the_command_stops_with_a_clear_message_when_there_is_no_index(self):
        self.make("text_only", 2)
        with self.assertRaises(FileNotFoundError) as caught:
            src.pipelines.standard_rag.standard_rag_main(["--model", "gpt-4o", "--view", "text_only", "--dry-run", "--data", self.data,
                                  "--results", self.results])
        self.assertIn("python -m src.main index", str(caught.exception))

    def test_the_retrieval_comparison_scores_each_mode_on_validation_and_refuses_the_test_split(self):
        self.make("text_only", 6)
        from array import array

        from src.retrieval.indexer import DenseIndex
        flat = array("f")
        for p in RAG_CORPUS:
            flat.extend(rag_vec(p["title"] + " " + p["text"]))
        index = DenseIndex(list(RAG_CORPUS), flat, len(RAG_WORDS), {"encoder": "fake", "passages_sha256": "abc"})

        def rag_for(mode):
            return src.pipelines.standard_rag.StandardRag(index, rag_enc, 10, mode, rerank=lambda q, t: [float(-i) for i in range(len(t))])
        out = os.path.join(self.tmp.name, "cmp.json")
        with contextlib.redirect_stdout(io.StringIO()):
            res = src.pipelines.standard_rag.retrieval_eval_main(["--data", self.data, "--out", out], rag_for=rag_for)
        self.assertEqual(sorted(res), sorted(src.pipelines.standard_rag.RETRIEVAL_MODES))
        saved = json.loads(Path(out).read_text(encoding="utf-8"))
        self.assertEqual((saved["view"], saved["split"], saved["n_questions"]), ("text_only", "val", 6))
        self.assertTrue(all("hit@3" in saved["modes"][m] and "mrr" in saved["modes"][m] for m in src.pipelines.standard_rag.RETRIEVAL_MODES))
        for m in src.pipelines.standard_rag.RETRIEVAL_MODES:
            lo, hi = saved["intervals_95"][m]["hit@3"]
            self.assertLessEqual(lo, saved["modes"][m]["hit@3"] + 1e-9)
            self.assertGreaterEqual(hi, saved["modes"][m]["hit@3"] - 1e-9)
        self.assertEqual(set(saved["random_from_the_same_pool"]["dense"]), set(src.pipelines.standard_rag.INTERVAL_KEYS))
        self.assertEqual((saved["approved"], saved["libraries"]), ("bm25-rerank", {"text_index_passages_sha256": "abc", "image_items_sha256": None}))
        self.assertEqual(set(saved["hit3_by_question"]), set(saved["modes"]))          # each question's Hit@3, for a later paired comparison
        self.assertEqual(len(saved["hit3_by_question"]["bm25"]), 6)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            src.pipelines.standard_rag.retrieval_eval_args(["--split", "test"])

    def test_the_image_views_are_scored_by_image_search_and_by_text_and_image_fused(self):
        from array import array

        from src.retrieval.indexer import DenseIndex
        items = [{"passage_id": f"v{i}", "title": "fig", "text": f"caption {i}", "tags": [t]} for i, t in enumerate(["MEL", "NV", "BCC"])]
        flat = array("f", [1, 0, 0, 1, 0.7, 0.7])
        vindex = DenseIndex(items, flat, 2, {})
        for view in ("image_only", "multimodal"):
            self.make(view, 4)
            vectors = {os.path.basename(q["image_path"]): ([1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]) for i, q in enumerate(self.questions)}
            flat_t = array("f")
            for p in RAG_CORPUS:
                flat_t.extend(rag_vec(p["title"] + " " + p["text"]))
            index = DenseIndex(list(RAG_CORPUS), flat_t, len(RAG_WORDS), {"encoder": "fake", "passages_sha256": "abc"})

            def rag_for(mode, index=index):
                return src.pipelines.standard_rag.StandardRag(index, rag_enc, 10, mode, rerank=lambda q, t: [float(-i) for i in range(len(t))])
            out = os.path.join(self.tmp.name, f"{view}.json")
            with contextlib.redirect_stdout(io.StringIO()):
                res = src.pipelines.standard_rag.retrieval_eval_main(["--view", view, "--data", self.data, "--out", out], rag_for=rag_for,
                                             visual=(vindex, vectors))
            modes = list(src.pipelines.standard_rag.RETRIEVAL_MODES)
            expected = ["image"] if view == "image_only" else modes + ["image"] + [f"fused:{m}" for m in modes]
            self.assertEqual(list(res), expected)
            saved = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(set(saved["random_from_the_same_pool"]), set(expected))
            if view == "multimodal":                                    # the fused list holds text passages too, so it scores differently
                share = saved["share_of_top3_from_the_image_library"]
                self.assertEqual((share["image"], share["bm25-rerank"]), (1.0, 0.0))
                self.assertTrue(0.0 < share["fused:bm25"] < 1.0)            # both sources reach the top 3
                self.assertEqual(len(saved["random_from_the_same_pool"]), len(expected))
                paired = saved["paired_hit3_vs_approved"]                # every search against the approved one, same questions
                self.assertEqual(set(paired), set(expected) - {"fused:bm25-rerank"})
                self.assertTrue(all(r["against"] == "fused:bm25-rerank" and 0 <= r["p_value"] <= r["p_holm"] <= 1 for r in paired.values()))
        with self.assertRaises(KeyError):
            src.retrieval.retriever.image_query_vector({"image_path": "processed/images/ISIC_9.jpg"}, {})

    def test_pilot_check_judges_the_complete_validation_runs_only_and_by_model(self):
        self.make("text_only")
        self.go(self.args(split="val"), FakeRagClient(self.questions), make_rag())
        write_jsonl(os.path.join(self.data, question_file("text_only", "test")), self.questions)
        self.go(self.args(split="test"), FakeRagClient(self.questions), make_rag())              # a test run: never part of the pilot
        rows = src.evaluation.evaluator.pilot_rows(self.results, root=".")
        self.assertEqual([(r["system"], r["model"], r["view"]) for r in rows], [("standard_rag", "gpt-4o", "text_only")])
        self.assertEqual(set(rows[0]["gates"]), set(src.evaluation.evaluator.PILOT_GATES))          # a RAG run has all four gates
        self.assertEqual(src.evaluation.evaluator.pilot_rows(self.results, root=".", model="another"), [])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            src.evaluation.evaluator.pilot_main(["--results", self.results])
        self.assertRegex(out.getvalue(), r"(PASS|FAIL) standard_rag gpt-4o text_only: readable answers")

    def test_collect_reports_a_standard_rag_run_beside_an_llm_only_run(self):
        self.make("text_only")
        self.go(self.args(split="val"), FakeRagClient(self.questions), make_rag())
        rows = src.evaluation.evaluator.collect(self.results, os.path.join(self.results, "summary.csv"), root=".", skip_text=True)
        self.assertEqual([(r["system"], r["model"], r["view"]) for r in rows], [("standard_rag", "gpt-4o", "text_only")])


class PilotGateTests(unittest.TestCase):
    """The gates of the validation pilot (src/evaluation/evaluator.py): fixed before any pilot result, so a test pins them."""

    @staticmethod
    def report(invalid=0, missing=0, ci_low=0.30, n=160, evidence=None):
        r = {"diagnosis": {"n_questions": n, "invalid_answers": invalid, "missing_predictions": missing, "accuracy": 0.4, "accuracy_95ci": [ci_low, 0.5]}}
        if evidence:
            r["evidence"] = {"n_questions": n, **evidence}
        return r

    def test_the_gates_are_the_researchers_and_a_value_exactly_on_a_limit_is_judged_as_written(self):
        gates = src.evaluation.evaluator.PILOT_GATES
        self.assertEqual(gates, {"readable_answers": 0.90, "accuracy_ci_low": 0.125, "evidence_line": 0.80, "quotes_word_for_word": 0.50})
        row = src.evaluation.evaluator.pilot_row
        self.assertTrue(row(self.report(invalid=16))["gates"]["readable_answers"])                    # 144 of 160 = exactly 90%: at least 90% passes
        self.assertFalse(row(self.report(invalid=17))["gates"]["readable_answers"])
        self.assertFalse(row(self.report(invalid=8, missing=9))["gates"]["readable_answers"])         # an unanswered question is not a readable answer
        self.assertFalse(row(self.report(ci_low=0.125))["gates"]["accuracy_ci_low"])                  # the lower end must be above chance, not on it
        self.assertTrue(row(self.report(ci_low=0.126))["gates"]["accuracy_ci_low"])
        self.assertEqual(set(row(self.report())["gates"]), {"readable_answers", "accuracy_ci_low"})   # an LLM-only run has no evidence gates

    def test_a_rag_run_is_also_judged_on_its_evidence_line_and_its_quotes(self):
        row = src.evaluation.evaluator.pilot_row
        good = row(self.report(evidence={"evidence_unparsed": 32, "evidence_given": 100, "n_covered": 50}))        # 80% line, 50% quotes: both on the limit
        self.assertEqual((good["evidence_line"], good["quotes_word_for_word"], good["passed"]), (0.8, 0.5, True))
        self.assertFalse(row(self.report(evidence={"evidence_unparsed": 33, "evidence_given": 100, "n_covered": 50}))["passed"])
        self.assertFalse(row(self.report(evidence={"evidence_unparsed": 0, "evidence_given": 100, "n_covered": 49}))["passed"])
        none_given = row(self.report(evidence={"evidence_unparsed": 0, "evidence_given": 0, "n_covered": 0}))
        self.assertEqual((none_given["quotes_word_for_word"], none_given["passed"]), (0.0, False))                 # no quote given counts as failing, not as dividing by 0
        self.assertFalse(row(self.report(invalid=100, evidence={"evidence_unparsed": 0, "evidence_given": 10, "n_covered": 10}))["passed"])   # one failed gate fails the run


class IndexCommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = os.path.join(self.tmp.name, "corpus", "text_only")
        self.images = os.path.join(self.tmp.name, "corpus", "image_only")
        write_jsonl(os.path.join(self.corpus, "passages.jsonl"), RAG_CORPUS)
        self.out = os.path.join(self.tmp.name, "index")

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, *extra):
        seen = []

        def encode(items):
            seen.extend(items)
            return rag_enc(items)
        with contextlib.redirect_stdout(io.StringIO()):
            src.retrieval.indexer.index_main(["--corpus", self.corpus, "--images", self.images, "--out", self.out, *(extra or ["--no-captions"])], encode=encode)
        return seen

    def test_every_passage_is_embedded_as_its_title_and_text_and_the_index_can_be_read(self):
        seen = self.build()
        self.assertEqual(seen[0], ["Melanoma of the trunk", "Melanoma on the trunk of a male patient. It is brown."])
        self.assertEqual(len(seen), 5)
        from src.retrieval.indexer import read_index
        idx = read_index(self.out, os.path.join(self.corpus, "passages.jsonl"))
        self.assertEqual([p["passage_id"] for p, _ in idx.search(rag_vec("trunk male"), 2)], ["p0", "p3"])
        meta = json.loads(Path(self.out, "index.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["n_passages"], meta["exclude_reference_pages"], meta["n_excluded_passages"]), (5, False, 0))

    def test_the_caption_file_is_recorded_with_forward_slashes_also_when_windows_gives_backslashes(self):
        """The index records where the captions are relative to the text library ("../image_only/items_final.jsonl"); Windows writes
        that path with backslashes, and the same index must be read on any system."""
        write_jsonl(os.path.join(self.images, "items_final.jsonl"), [])
        with mock.patch.object(os.path, "relpath", return_value="..\\image_only\\items_final.jsonl"):
            self.build("--reference-texts", os.path.join(self.tmp.name, "none.json"))
        meta = json.loads(Path(self.out, "index.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["captions_file"], "../image_only/items_final.jsonl")

    def test_the_reference_pages_are_left_out_only_when_asked(self):
        refs = os.path.join(self.tmp.name, "reference_texts.json")
        Path(refs).write_text(json.dumps({"NV": {"url": "https://x/c", "texts": []}, "MEL": {"url": "https://x/a", "texts": []}}),
                              encoding="utf-8")
        seen = self.build("--exclude-reference-pages", "--reference-texts", refs, "--no-captions")
        self.assertEqual(len(seen), 3)
        meta = json.loads(Path(self.out, "index.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["n_passages"], meta["n_excluded_passages"], meta["exclude_reference_pages"]), (3, 2, True))
        self.assertEqual(meta["excluded_urls"], ["https://x/a", "https://x/c"])
        self.assertNotIn("p0", meta["passage_ids"])
        self.assertEqual(len(self.build("--no-captions", "--replace")), 5)                  # without the flag nothing is left out

    def test_the_caption_records_are_searched_as_text_once_and_a_changed_library_is_noticed(self):
        """Researcher's decision (2026-11-24, built 2026-11-25): the text search covers the picture-and-caption records too; a caption
        that is already a corpus passage of the same paper is not added twice."""
        long_caption = RAG_CORPUS[0]["text"] + " Seen on dermoscopy."                    # already passage p0 of paper d0
        items = [{"item_id": "d0#fig1", "passage_id": "d0#fig1", "doc_id": "d0", "title": "Melanoma of the trunk (Figure 1)",
                  "text": long_caption, "tags": ["MEL"], "image_file": "d0_fig1.jpg", "url": "https://x/a", "licence": "CC BY 4.0"},
                 {"item_id": "d9#fig2", "passage_id": "d9#fig2", "doc_id": "d9", "title": "Nevus (Figure 2)",
                  "text": "Dermoscopy of a brown nevus on the trunk of a female.", "tags": ["NV"], "image_file": "d9_fig2.jpg"}]
        write_jsonl(os.path.join(self.images, "items_final.jsonl"), items)
        seen = self.build("--reference-texts", os.path.join(self.tmp.name, "none.json"))
        self.assertEqual(len(seen), 6)                                                     # 5 passages + 1 new caption
        self.assertIn(["Nevus (Figure 2)", "Dermoscopy of a brown nevus on the trunk of a female."], seen)
        meta = json.loads(Path(self.out, "index.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["n_captions_indexed"], meta["captions_already_in_the_corpus"], meta["captions_file"]),
                         (1, ["d0#fig1"], "../image_only/items_final.jsonl"))
        from src.retrieval.indexer import read_index
        idx = read_index(self.out, os.path.join(self.corpus, "passages.jsonl"))
        record = next(p for p in idx.passages if p["passage_id"] == "d9#fig2")
        self.assertEqual((record["library"], record["image_file"]), ("visual", "d9_fig2.jpg"))
        write_jsonl(os.path.join(self.images, "items_final.jsonl"), items[1:])     # the library changed
        with self.assertRaises(ValueError):
            read_index(self.out, os.path.join(self.corpus, "passages.jsonl"))
        os.remove(os.path.join(self.images, "items_final.jsonl"))
        with self.assertRaises(SystemExit):                                                   # asked for, but not there
            self.build("--reference-texts", os.path.join(self.tmp.name, "none.json"))

    def test_a_rebuild_with_other_exclusions_is_refused_unless_asked_for(self):
        """Seen on the laptop (2026-11-25): a rebuild without --exclude-reference-overlap silently dropped that choice."""
        refs = os.path.join(self.tmp.name, "reference_texts.json")
        Path(refs).write_text(json.dumps({"NV": {"url": "https://x/c", "texts": [{"text": "unrelated words only here"}]}}), encoding="utf-8")
        self.build("--exclude-reference-pages", "--exclude-reference-overlap", "--reference-texts", refs, "--no-captions")
        with self.assertRaises(SystemExit) as caught:
            self.build("--exclude-reference-pages", "--reference-texts", refs, "--no-captions")
        self.assertIn("--exclude-reference-overlap was True, now False", str(caught.exception))
        self.build("--exclude-reference-pages", "--exclude-reference-overlap", "--reference-texts", refs, "--no-captions")   # same: fine
        self.build("--exclude-reference-pages", "--reference-texts", refs, "--no-captions", "--replace")                     # on purpose
        meta = json.loads(Path(self.out, "index.json").read_text(encoding="utf-8"))
        self.assertFalse(meta["exclude_reference_overlap"])

    def test_passages_that_copy_a_reference_text_are_left_out_only_when_asked(self):
        copy = "one two three four five six seven eight nine ten eleven twelve"
        passages = [{"passage_id": "a", "url": "u1", "text": copy, "title": ""},
                    {"passage_id": "b", "url": "u2", "text": "alpha beta gamma delta epsilon zeta eta theta iota kappa", "title": ""}]
        refs = src.retrieval.corpus.shingles(copy + " thirteen fourteen")
        kept, _, out = src.retrieval.indexer.build_index(passages, lambda x: [[1.0]] * len(x), (), refs)
        self.assertEqual(([p["passage_id"] for p in kept], [p["passage_id"] for p in out]), (["b"], ["a"]))
        kept, _, out = src.retrieval.indexer.build_index(passages, lambda x: [[1.0]] * len(x))
        self.assertEqual((len(kept), len(out)), (2, 0))

    def test_the_audit_finds_near_duplicates_reference_copies_and_tags_that_would_change(self):
        t = "melanomas of the trunk are seen in older men with dermoscopy showing a blue white veil and atypical network"

        def mk(i, text, tags):
            return {"passage_id": i, "doc_id": "d" + i, "library": "pmc", "title": "", "text": text,
                    "n_words": len(text.split()), "tags": tags}
        ps = [mk("a", t, []), mk("b", t + " too", ["MEL"]), mk("c", "unrelated words about something else entirely "
                                                           "that has no overlap at all with the rest", [])]
        r = src.retrieval.corpus.audit_corpus(ps, src.retrieval.corpus.shingles(ps[2]["text"]))
        self.assertEqual(r["near_duplicate_pairs"], [["a", "b"]])
        self.assertEqual(r["reference_copies"], ["c"])
        self.assertEqual(r["tags_that_would_change"], ["a"])
        self.assertEqual(r["by_library"]["pmc"]["untagged"], 2)

    def test_the_image_library_check_finds_a_missing_changed_or_repeated_picture_and_a_record_without_a_source(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "images"))
            files = {"a.jpg": b"\xff\xd8\xff" + b"a" * 5000, "b.jpg": b"\xff\xd8\xff" + b"b" * 5000, "c.txt": b"<html>not an image</html>"}
            for name, data in files.items():
                Path(d, "images", name).write_bytes(data)

            def item(i, **kw):
                name = kw.pop("image_file", f"{i}.jpg")
                return {"item_id": i, "doc_id": "PMC1", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/", "licence": "CC BY 4.0",
                        "caption": "Dermoscopy of a melanoma.", "tags": ["MEL"], "image_file": name,
                        "image_sha256": hashlib.sha256(files.get(name, b"")).hexdigest(), **kw}
            good = [item("g1", image_file="a.jpg"), item("g2", image_file="b.jpg", doc_id="dermoscopedia:Melanoma",
                                                         url="https://dermoscopedia.org/Melanoma")]
            report = src.retrieval.images.image_provenance(good, d)
            self.assertEqual((report["records_with_a_problem"], report["problem_counts"]), (0, {}))
            bad = [item("m", image_file="gone.jpg"), item("h", image_file="c.txt"), item("w", image_file="a.jpg", image_sha256="0" * 64),
                   item("t", image_file="a.jpg"), item("t2", image_file="a.jpg"), item("s", image_file="b.jpg", doc_id="made-up"),
                   item("u", image_file="b.jpg", url="https://example.org/x"), item("l", image_file="b.jpg", licence=""),
                   item("c", image_file="b.jpg", caption=" "), item("n", image_file="b.jpg", tags=[])]
            found = src.retrieval.images.image_provenance(bad, d)["problem_counts"]
            self.assertEqual(found, {"the picture file is missing": 1, "the file is not an image": 1,
                                     "the picture no longer matches its fingerprint": 1, "the same picture twice": 5,
                                     "the document id is not that of a real source": 1, "no web address on the source's own site": 1,
                                     "no licence": 1, "no caption": 1, "the caption names no diagnosis": 1})

    def test_the_tags_and_classes_are_never_embedded(self):
        seen = self.build()
        flat = " ".join(" ".join(x) for x in seen).lower()
        for hidden in ("mel", "scc", "query_class", "tags", "cc by"):
            self.assertNotIn(hidden, flat.replace("melanoma", "").replace("lesions", ""), hidden)

    def test_a_missing_corpus_stops_with_the_command_to_run(self):
        os.remove(os.path.join(self.corpus, "passages.jsonl"))
        with self.assertRaises(SystemExit) as caught:
            self.build()
        self.assertIn("python -m src.main corpus", str(caught.exception))


class ImageLibraryLoadTests(unittest.TestCase):
    """The image views of the standard RAG read the image index and the saved Kaggle vectors from data/corpus/image_only/ (made by visual-index)."""

    def test_load_visual_reads_the_index_and_the_vectors_from_the_image_library(self):
        from src.retrieval.indexer import write_index
        with tempfile.TemporaryDirectory() as d:
            folder = os.path.join(d, IMAGE_LIBRARY)
            records = os.path.join(folder, "items_final.jsonl")
            write_jsonl(records, RAG_IMAGES)
            with self.assertRaises(FileNotFoundError):                                          # no index yet
                src.pipelines.standard_rag.load_visual(d)
            write_index(os.path.join(folder, "index"), RAG_IMAGES, [rag_vec(p["text"]) for p in RAG_IMAGES],
                        {"encoder": "chanwkim/monet", "passages_sha256": src.utils.file_sha256(records)})
            with self.assertRaises(FileNotFoundError) as caught:                                # an index, but no Kaggle vectors yet
                src.pipelines.standard_rag.load_visual(d)
            self.assertIn("visual-index", str(caught.exception))
            Path(folder, "kaggle_vectors.json").write_text(json.dumps({"names": ["ISIC_1.jpg", "ISIC_2.jpg"], "vectors": [[1.0, 0.0], [0.0, 1.0]]}),
                                                           encoding="utf-8")
            index, vectors = src.pipelines.standard_rag.load_visual(d)
            self.assertEqual(len(index.passages), 3)
            self.assertEqual(vectors, {"ISIC_1.jpg": [1.0, 0.0], "ISIC_2.jpg": [0.0, 1.0]})
            Path(folder, "kaggle_vectors.json").write_text(json.dumps({"names": ["ISIC_1.jpg", "ISIC_2.jpg", "ISIC_3.jpg"], "vectors": [[1.0, 0.0]]}),
                                                           encoding="utf-8")
            with self.assertRaises(ValueError):                                                  # a damaged file must not be read half way
                src.pipelines.standard_rag.load_visual(d)


class LibraryStatusTests(unittest.TestCase):
    """The multimodal library is only a file that names the two libraries it combines; the status check must notice a wrong one."""

    GOOD = {"view": "multimodal", "sources": ["text_only", "image_only"],
            "fusion": {"method": "reciprocal rank fusion", "k": 60, "list": 10}}

    def check(self, library):
        with tempfile.TemporaryDirectory() as d:
            for folder in ("text_only", "image_only"):
                os.makedirs(os.path.join(d, "corpus", folder))
            if library is not None:
                Path(d, "corpus", "multimodal").mkdir()
                Path(d, "corpus", "multimodal", "library.json").write_text(json.dumps(library), encoding="utf-8")
            return src.main.step_libraries(d)[0]

    def test_a_missing_file_is_not_run_a_right_one_is_done_and_a_wrong_one_is_a_problem(self):
        self.assertEqual(self.check(None), src.main.NOT_RUN)
        self.assertEqual(self.check(self.GOOD), src.main.DONE)
        self.assertEqual(self.check({**self.GOOD, "sources": ["text_only"]}), src.main.PROBLEM)                  # a source is missing
        self.assertEqual(self.check({**self.GOOD, "sources": ["text_only", "multimodal"]}), src.main.PROBLEM)    # not a record library
        self.assertEqual(self.check({**self.GOOD, "fusion": {**self.GOOD["fusion"], "list": 5}}), src.main.PROBLEM)   # not what the RAG does
        self.assertEqual(self.check({**self.GOOD, "fusion": {**self.GOOD["fusion"], "k": 30}}), src.main.PROBLEM)

    def test_the_image_index_must_belong_to_the_image_records(self):
        """On the laptop the image records (items_final.jsonl) and the image index must match; the index is made by visual-index."""
        from src.retrieval.indexer import write_index
        with tempfile.TemporaryDirectory() as d:
            for folder in ("text_only", "image_only", "multimodal"):
                os.makedirs(os.path.join(d, "corpus", folder))
            Path(d, "corpus", "multimodal", "library.json").write_text(json.dumps(self.GOOD), encoding="utf-8")
            self.assertEqual(src.main.step_libraries(d)[0], src.main.DONE)                    # no records on this computer: nothing to compare
            records = os.path.join(d, "corpus", "image_only", "items_final.jsonl")
            write_jsonl(records, RAG_IMAGES)
            status, note = src.main.step_libraries(d)
            self.assertEqual(status, src.main.PROBLEM)                                         # records, but no index (the layout of an older version)
            self.assertIn("visual-index", note)
            write_index(os.path.join(d, "corpus", "image_only", "index"), RAG_IMAGES, [rag_vec(p["text"]) for p in RAG_IMAGES],
                        {"encoder": "chanwkim/monet", "passages_sha256": src.utils.file_sha256(records)})
            status, note = src.main.step_libraries(d)
            self.assertEqual(status, src.main.DONE)
            self.assertIn("3 records in the index, chanwkim/monet", note)
            write_jsonl(records, RAG_IMAGES[:1])                                               # the records changed after the index was made
            self.assertEqual(src.main.step_libraries(d)[0], src.main.PROBLEM)

    def test_a_library_folder_that_is_not_there_is_a_problem(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "corpus", "multimodal"))
            Path(d, "corpus", "multimodal", "library.json").write_text(json.dumps(self.GOOD), encoding="utf-8")
            status, note = src.main.step_libraries(d)
        self.assertEqual(status, src.main.PROBLEM)
        self.assertIn("text_only", note)


class IndexStatusTests(unittest.TestCase):
    def test_check_py_reports_the_index_and_notices_a_changed_corpus(self):
        from src.retrieval.indexer import write_index
        from src.utils import file_sha256
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(src.main.step_index(d)[0], src.main.NOT_RUN)
            corpus = os.path.join(d, TEXT_LIBRARY)
            write_jsonl(os.path.join(corpus, "passages.jsonl"), RAG_CORPUS)
            sha = file_sha256(os.path.join(corpus, "passages.jsonl"))
            write_index(os.path.join(d, INDEX_DIR), RAG_CORPUS, [rag_vec(p["text"]) for p in RAG_CORPUS],
                        {"encoder": "fake", "passages_sha256": sha, "exclude_reference_pages": True})
            status, note = src.main.step_index(d)
            self.assertEqual(status, src.main.DONE)
            self.assertIn("5 passages", note)
            self.assertIn("reference pages left out", note)
            write_jsonl(os.path.join(corpus, "passages.jsonl"), RAG_CORPUS[:3])               # the corpus changed after the index
            self.assertEqual(src.main.step_index(d)[0], src.main.PROBLEM)
            write_jsonl(os.path.join(corpus, "passages.jsonl"), RAG_CORPUS)
            with open(os.path.join(d, INDEX_DIR, "embeddings.f32"), "ab") as f:
                f.write(b"\0\0\0\0")                                                           # a damaged file
            self.assertEqual(src.main.step_index(d)[0], src.main.PROBLEM)
            captions = os.path.join(d, IMAGE_LIBRARY, "items_final.jsonl")
            write_jsonl(captions, RAG_IMAGES)
            write_index(os.path.join(d, INDEX_DIR), RAG_CORPUS, [rag_vec(p["text"]) for p in RAG_CORPUS],
                        {"encoder": "fake", "passages_sha256": sha, "captions_file": "../image_only/items_final.jsonl",
                         "captions_sha256": file_sha256(captions), "n_captions_indexed": 3})
            self.assertEqual(src.main.step_index(d), (src.main.DONE, "5 passages, 8 numbers each (fake); with 3 caption records; all passages"))
            write_jsonl(captions, RAG_IMAGES[:1])                                                # the image library changed
            self.assertEqual(src.main.step_index(d)[0], src.main.PROBLEM)


if __name__ == "__main__":
    unittest.main()
