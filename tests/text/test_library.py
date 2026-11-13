"""The reference library (src/text/library.py), with a local stand-in for NCBI (PubMed, PubMed Central and the Bookshelf's Open
Access subset), on a dataset built from the build's made-up cases, so no test reaches the internet."""

import contextlib
import csv
import io
import json
import tarfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from xml.sax.saxutils import escape

from src.text import library, prepare

from .test_prepare import BuildData

HEADINGS = {("melanoma", "melanoma"): "Melanoma", ("nevus", "nevus"): "Nevus, Pigmented",
            ("basal cell carcinoma", "basal cell carcinoma"): "Carcinoma, Basal Cell"}  # term_check.csv: the survey's headings
REVIEWS = {"Melanoma": ["5005", "5003", "5001"], "Nevus, Pigmented": ["5004", "5001"], "Carcinoma, Basal Cell": ["5006"]}
GUIDELINES = {"Melanoma": ["6001", "2008"]}  # 2008 is an article of the dataset's candidates: left out
NODULAR = ("Nodular melanoma grows vertically from the start, so it is often thick when it is found; it is usually a firm, "
           "symmetric, dark or red nodule that has changed within weeks or months.")
LEAK = ("In a typical history the patient noticed that a mole on his upper back had grown and darkened over six months before "
        "a first visit to a doctor, and nobody had looked at it.")  # 12 words in a row of the question of 2008-1
PUBMED = {  # pmid: (title, year, PMC number, abstract sections)
    "5001": ("Melanoma and naevi: a review", "2023", "PMC8001", [
        ("BACKGROUND", "Most melanomas arise on skin without a precursor, and only a minority arise within a melanocytic nevus "
                       "that was present before."),
        ("CONCLUSION", "Counting nevi and watching for change remain the simplest ways to find people at higher risk.")]),
    "5003": ("Nodular melanoma: a review", "2024", "PMC8003", [("", NODULAR)]),
    "5004": ("Nevi in children: a review", "2025", "", [("", "Children acquire most of their melanocytic nevi between the ages "
                                                             "of two and twelve years, mostly on skin that is exposed to the sun.")]),
    "5005": ("Melanoma for general practitioners", "2025", "PMC8005", [("", LEAK)]),
    "5006": ("Basal cell carcinoma in the elderly", "2022", "", []),  # no abstract: no text
    "6001": ("Guideline for the management of cutaneous melanoma", "2024", "PMC8601", [("", "The guideline summarises the "
                                                                                        "evidence on staging and treatment.")]),
}
PMC = {  # PMC number (without "PMC"): (licence, body); a number not here is not given in full
    "8001": ("https://creativecommons.org/licenses/by/4.0/",
             "<p>Melanocytic nevi are benign proliferations of melanocytes that appear in childhood and adolescence and may "
             "change slowly over the decades of adult life.</p>"
             "<sec><title>Diagnosis</title><p>Dermoscopy improves the accuracy of the diagnosis of pigmented lesions compared "
             "with the naked eye.</p><fig><caption><p>A figure caption that must not be read.</p></caption></fig>"
             "<p>Sequential digital images help to judge whether a flat lesion has changed in size, shape or colour.</p></sec>"
             "<sec><title>Treatment</title><p>Excision is needed only when a nevus has changed or looks atypical, or when "
             "the patient asks for its removal.</p></sec>"),
    "8005": ("https://creativecommons.org/licenses/by-nc/4.0/", f"<sec><title>Introduction</title><p>{NODULAR}</p></sec>"),
    "8601": ("https://creativecommons.org/licenses/by/4.0/",
             "<sec><title>Recommendations</title><p>Every suspicious pigmented lesion should be removed whole with a narrow "
             "margin for histology, never only sampled.</p><p>The width of the final margin depends on the thickness of the "
             "tumour measured by the pathologist.</p></sec>"),
}
MELANOMA_BODY = """
<sec id="s1"><title>Continuing Education Activity</title><p>This activity reviews melanoma and the role of the team in its care.</p></sec>
<sec id="s2"><title>Introduction</title><p>Melanoma is a malignant tumour of the melanocytes, the cells that make pigment, and it
causes most deaths from <italic>skin</italic> cancer.</p><p>It is most common in fair-skinned people with much exposure to the
sun.</p><fig id="f1"><caption><p>Figure: a figure caption that must not be read.</p></caption></fig>
<table-wrap id="t1"><table><tr><td>A table that must not be read.</td></tr></table></table-wrap></sec>
<sec id="s3"><title>Treatment / Management</title><list list-type="bullet"><list-item><p>Wide local excision, with a margin chosen
by the thickness of the tumour.</p></list-item><list-item><p>A sentinel lymph node biopsy for tumours thicker than 0.8
mm.</p></list-item></list><p>Follow-up visits:<list><list-item><p>every three months</p></list-item><list-item><p>for two
years.</p></list-item></list></p></sec>
<sec id="s4"><title>Review Questions</title><p>A review question that must not be read.</p></sec>
<sec id="s5"><title>Comment on this article</title><p>A comment form that must not be read, long enough to make a passage of
its own if it were read at all, which it must not be.</p></sec>"""
CHAPTERS = {  # file: (Bookshelf accession or "", title, summary, body); StatPearls' archive in the Open Access subset
    "article-101": ("NBK101", "Melanoma", "", MELANOMA_BODY),
    "article-102": ("", "Squamous Cell Skin Cancer", "Cutaneous squamous cell carcinoma is the second most common skin cancer.",
                    '<sec><title>Introduction</title><p>Cutaneous squamous cell carcinoma arises from the keratinocytes of the '
                    'epidermis, mostly on skin that has had much sun, and it is the second most common skin cancer.</p></sec>'),
    "article-103": ("", "Uveal Melanoma", "Uveal melanoma is the most common primary tumour of the eye.",
                    "<sec><title>Introduction</title><p>Text.</p></sec>"),  # a name that does not count, and another site
    "article-104": ("", "Oral Cancer", "Oral squamous cell carcinoma is the most common cancer of the mouth.",
                    "<sec><title>Introduction</title><p>Text.</p></sec>"),  # another site in the title
    "article-105": ("", "Asthma", "Asthma is a chronic disease of the airways.", "<sec><title>Introduction</title><p>Text.</p></sec>"),
}
FILE_LIST = ("File,Citation,Accession ID,Last Updated (YYYY-MM-DD HH:MM:SS)\n0a/1b/other_NBK1.tar.gz,Another book\u00ae,NBK1,"
             "2025-01-01 00:00:00\n3d/12/statpearls_NBK430685.tar.gz,StatPearls,NBK430685,2026-11-01 00:00:00\n")


def _nxml(accession: str, title: str, summary: str, body: str) -> str:
    summary = f"<sec><title>Continuing Education Activity</title><p>{escape(summary)}</p></sec>" if summary else ""
    ids = f'<book-part-id book-part-id-type="pmcid">{accession}</book-part-id>' if accession else ""
    licence = ('<license license-type="open-access" xlink:href="http://creativecommons.org/licenses/by-nc-nd/4.0/"/>' if accession
               else '<license license-type="open-access"><license-p>This book is distributed under the terms of the Creative '
                    'Commons licence (CC BY-NC-ND 4.0) ( http://creativecommons.org/licenses/by-nc-nd/4.0/ ).</license-p></license>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE book-part-wrapper PUBLIC "-//NLM//DTD BITS Book Interchange DTD v2.0 '
            '20151225//EN" "BITS-book2.dtd">\n<book-part-wrapper xmlns:xlink="http://www.w3.org/1999/xlink"><book-meta><book-title-'
            f'group><book-title>StatPearls</book-title></book-title-group><permissions>{licence}</permissions></book-meta><book-part '
            f'book-part-type="chapter"><book-part-meta>{ids}<title-group><title>{escape(title)}</title></title-group><pub-history><date date-'
            'type="created"><year>2017</year></date><date date-type="updated"><year>2024</year></date></pub-history></book-part-'
            f'meta><body>{summary}{body}</body><back><ref-list><ref><mixed-citation>A reference that must not be read.'
            '</mixed-citation></ref></ref-list></back></book-part></book-part-wrapper>')


def _archive() -> bytes:
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w:gz") as archive:
        files = {f"statpearls_NBK430685/{name}.nxml": _nxml(*chapter).encode("utf-8") for name, chapter in CHAPTERS.items()}
        files["statpearls_NBK430685/article-106.nxml"] = b"<book-part-wrapper><unclosed></book-part-wrapper>"  # cannot be read
        files["statpearls_NBK430685/article-101_f1.jpg"] = b"\xff\xd8 a figure"
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return content.getvalue()


ARCHIVE = _archive()


class LibraryTests(BuildData):
    def setUp(self):
        super().setUp()
        prepare.build(self.data, classes=self.CLASSES)
        terms = prepare.read_terms(self.data / prepare.TERMS, self.CLASSES)
        prepare._write_csv(self.data / prepare.CHECK, prepare.CHECK_COLUMNS, [
            {**{c: t[c] for c in ("class", "term", "counts")}, "in_mesh": "heading" if (t["class"], t["term"]) in HEADINGS else "no",
             "mesh_ids": "", "mesh_headings": HEADINGS.get((t["class"], t["term"]), ""), "pubmed_articles": "10"} for t in terms])
        self.ncbi, self.page_mode = [], "normal"
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body: bytes, kind: str):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path, _, query = self.path.partition("?")
                if path == "/esearch.fcgi":
                    params = dict(urllib.parse.parse_qsl(query))
                    test.ncbi.append(("esearch", params["term"], params.get("sort", ""), params.get("retmax")))
                    term = params["term"]
                    found = REVIEWS if "review[Publication Type] NOT" in term else GUIDELINES
                    ids = [p for heading, pmids in found.items() if f'"{heading}"[MeSH Major Topic]' in term for p in pmids]
                    ids = ids[:int(params["retmax"])]
                    self._send(json.dumps({"esearchresult": {"count": str(len(ids)), "idlist": ids}}).encode(),
                               "application/json")
                elif path.startswith("/pub/litarch/"):
                    test.ncbi.append(("open access", path.removeprefix("/pub/litarch/")))
                    if path == "/pub/litarch/file_list.csv":
                        text = FILE_LIST if test.page_mode == "normal" else FILE_LIST.replace("NBK430685", "NBK2")
                        self._send(text.encode("latin-1"), "text/csv")  # as NCBI's list: "®" as one byte, not UTF-8
                    elif path == "/pub/litarch/3d/12/statpearls_NBK430685.tar.gz":
                        self._send(ARCHIVE, "application/gzip")
                    else:
                        self.send_error(404)
                else:
                    self.send_error(404)

            def do_POST(self):
                form = dict(urllib.parse.parse_qsl(self.rfile.read(int(self.headers["Content-Length"])).decode()))
                ids = form["id"].split(",")
                test.ncbi.append(("efetch", form["db"], ids))
                if form["db"] == "pmc":
                    body = "<pmc-articleset>" + "".join(_pmc_xml(n) for n in ids) + "</pmc-articleset>"
                else:
                    body = "<PubmedArticleSet>" + "".join(_pubmed_xml(p) for p in ids) + "</PubmedArticleSet>"
                self._send(body.encode("utf-8"), "application/xml")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        for patcher in (mock.patch.object(prepare, "EUTILS", base + "/"),
                        mock.patch.object(library, "OPEN_ACCESS", base + "/pub/litarch/"),
                        mock.patch.object(library, "FULLTEXTS_PER_CLASS", 1), mock.patch.object(library, "FULLTEXT_BATCH", 1)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _library(self):
        with (self.data / "library.csv").open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, library.LIBRARY_COLUMNS)
            return list(reader)

    def test_four_corpora_are_found_by_fixed_rules_and_cut_word_for_word_into_passages(self):
        done = library.library(self.data, classes=self.CLASSES)
        rows = self._library()
        self.assertEqual([(r["passage_id"], r["source"], r["section"], r["topics"]) for r in rows], [
            ("pubmed-5001-1", "PMID 5001", "Abstract", "melanoma; nevus"), ("pubmed-5003-1", "PMID 5003", "Abstract", "melanoma"),
            ("pubmed-5004-1", "PMID 5004", "Abstract", "nevus"),  # 5005 shares 12 words with a question; 5006 has no abstract
            ("pmc-PMC8001-1", "PMC8001", "", "nevus"), ("pmc-PMC8001-2", "PMC8001", "Diagnosis", "nevus"),
            ("pmc-PMC8001-3", "PMC8001", "Treatment", "nevus"),  # PMC8005, the newest melanoma review: the same as 5003's
            ("guidelines-PMC8601-1", "PMC8601", "Recommendations", "melanoma"),
            ("textbook-NBK101-1", "NBK101", "Introduction", "melanoma"),
            ("textbook-NBK101-2", "NBK101", "Treatment / Management", "melanoma"),  # without a Bookshelf number: the file's name
            ("textbook-article-102-1", "article-102", "Introduction", "squamous cell carcinoma")])
        texts = {}
        with (self.data / "library" / "passages.jsonl").open(encoding="utf-8") as f:
            for line in f:
                passage = json.loads(line)
                texts[passage["passage_id"]] = passage["text"]
        self.assertEqual(list(texts), [r["passage_id"] for r in rows])
        for row in rows:
            self.assertEqual(row["sha256"], library.hashlib.sha256(texts[row["passage_id"]].encode("utf-8")).hexdigest())
            self.assertEqual(row["words"], str(len(texts[row["passage_id"]].split())))
        self.assertEqual(texts["pubmed-5001-1"], "\n\n".join(text for _, text in PUBMED["5001"][3]))
        self.assertEqual(texts["pmc-PMC8001-2"], "Dermoscopy improves the accuracy of the diagnosis of pigmented lesions compared "
                         "with the naked eye.\n\nSequential digital images help to judge whether a flat lesion has changed in "
                         "size, shape or colour.")  # the figure caption is left out
        self.assertEqual(texts["textbook-NBK101-1"], "Melanoma is a malignant tumour of the melanocytes, the cells that make pigment, "
                         "and it causes most deaths from skin cancer.\n\nIt is most common in fair-skinned people with much "
                         "exposure to the sun.")
        self.assertEqual(texts["textbook-NBK101-2"], "Wide local excision, with a margin chosen by the thickness of the tumour.\n\n"
                         "A sentinel lymph node biopsy for tumours thicker than 0.8 mm.\n\nFollow-up visits: every three months "
                         "for two years.")  # list items: each on its own, or within their paragraph
        everything = "\n".join(texts.values())
        for left_out in ("must not be read", "This activity reviews"):
            self.assertNotIn(left_out, everything)
        licences = {r["source"]: r["licence"] for r in rows}
        self.assertEqual((licences["PMID 5001"], licences["PMC8001"], licences["NBK101"], licences["article-102"]),
                         (library.ABSTRACT_LICENCE, "https://creativecommons.org/licenses/by/4.0/",
                          "http://creativecommons.org/licenses/by-nc-nd/4.0/", "http://creativecommons.org/licenses/by-nc-nd/4.0/"))
        self.assertEqual(dict(done["dropped"]), {"shares a run of words with a question": 1, "the same as an earlier passage": 1})
        searches = [entry for entry in self.ncbi if entry[0] == "esearch"]
        self.assertEqual(len(searches), 6)  # reviews and guidelines for the 3 classes with headings
        for _, term, sort, retmax in searches:
            self.assertIn('NOT "case reports"[Publication Type]', term)
            self.assertIn(prepare.HUMANS, term)
            self.assertEqual(sort, "pub_date")
        self.assertEqual([entry[2] for entry in self.ncbi if entry[:2] == ("efetch", "pmc")], [["8005"], ["8001"], ["8601"]])
        self.assertNotIn("2008", next(entry[2] for entry in self.ncbi if entry[:2] == ("efetch", "pubmed")))
        self.assertEqual([entry[1] for entry in self.ncbi if entry[0] == "open access"],
                         ["file_list.csv", "3d/12/statpearls_NBK430685.tar.gz"])
        meta = done["archive"]
        self.assertEqual((meta["file"], meta["bytes"], meta["chapters"], meta["unreadable"]),
                         ("statpearls_NBK430685.tar.gz", len(ARCHIVE), 6, ["statpearls_NBK430685/article-106.nxml"]))
        self.assertEqual(meta["sha256"], library.hashlib.sha256(ARCHIVE).hexdigest())

    def test_the_report_counts_each_class_and_corpus_and_a_second_run_reads_the_kept_archive(self):
        library.library(self.data, classes=self.CLASSES)
        first = (self.data / "library.csv").read_bytes()
        library.library(self.data, classes=self.CLASSES)
        self.assertEqual((self.data / "library.csv").read_bytes(), first)
        self.assertEqual(sum(entry[0] == "open access" for entry in self.ncbi), 2)  # not downloaded again
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertLess(report.index("<!-- step: build -->"), report.index("<!-- step: library -->"))
        for expected in ("## Library: the reference texts for retrieval", "| melanoma | Melanoma | 3 | 1 | 1 | 1 |",
                         "| nevus | Nevus, Pigmented | 2 | 1 | 0 | 0 |", "| basal cell carcinoma | Carcinoma, Basal Cell | 1 | 0 | 0 | 0 |",
                         "| squamous cell carcinoma | none | 0 | 0 | 0 | 1 |", "| PubMed: abstracts of reviews | 4 | 3 |",
                         "| Textbook: StatPearls chapters | 2 | 3 |", "share a run of 12 words with a question of the dataset: 1",
                         f"SHA-256 `{library.hashlib.sha256(ARCHIVE).hexdigest()}` (checked)", "chapters read: 6; chapter files "
                         "that could not be read: 1 (statpearls_NBK430685/article-106.nxml)",
                         "No person has read the passages"):
            self.assertIn(expected, report)

    def test_a_changed_archive_a_file_list_without_statpearls_and_a_missing_build_stop_the_step_with_a_message(self):
        self.page_mode = "no statpearls"
        with self.assertRaisesRegex(ValueError, "has no archive for StatPearls .NBK430685.; its columns are: File, Citation, "
                                                "Accession ID"):
            library.library(self.data, classes=self.CLASSES)
        self.assertFalse((self.data / "library.csv").exists())
        self.page_mode = "normal"
        library.library(self.data, classes=self.CLASSES)
        with (self.data / "sources" / "library" / "statpearls" / "statpearls_NBK430685.tar.gz").open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "statpearls_NBK430685.tar.gz has changed since it was downloaded"):
            library.library(self.data, classes=self.CLASSES)
        (self.data / "dataset.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "dataset.csv not found: run python -m src.text.prepare build first"):
            library.library(self.data, classes=self.CLASSES)

    def test_passages_are_whole_paragraphs_of_one_section_within_the_word_limit(self):
        with mock.patch.object(library, "PASSAGE_WORDS", 10), mock.patch.object(library, "SHORTEST", 3):
            found = library._passages([
                ["A", ["one two three four", "five six seven eight", "nine ten"]],  # exactly 10 words: one passage
                ["B", ["one two three four five six", "seven eight nine ten eleven"]],  # 11 words: two
                ["C", ["Alpha beta gamma delta epsilon. Zeta eta theta iota kappa lambda. Mu nu xi omicron pi.", "short"]],
                ["D", ["too short"]]])
        self.assertEqual(found, [("A", "one two three four\n\nfive six seven eight\n\nnine ten"),
                                 ("B", "one two three four five six"), ("B", "seven eight nine ten eleven"),
                                 ("C", "Alpha beta gamma delta epsilon."), ("C", "Zeta eta theta iota kappa lambda."),
                                 ("C", "Mu nu xi omicron pi.\n\nshort")])  # a long paragraph at its sentence ends; D too short

    def test_the_command_line_prints_each_corpus_and_what_to_commit(self):
        real = library.library
        with mock.patch.object(library, "library", lambda show=None: real(self.data, classes=self.CLASSES, show=show)), \
                contextlib.redirect_stdout(io.StringIO()) as shown:
            library.main([])
        for expected in ("  pubmed: 4 documents, 3 passages", "  textbook: 2 documents, 3 passages", "Passages left out: 2",
                         "(not in Git)", "Commit "):
            self.assertIn(expected, shown.getvalue())
        with self.assertRaises(SystemExit):
            library.main(["extra"])


def _pubmed_xml(pmid: str) -> str:
    title, year, pmcid, sections = PUBMED[pmid]
    abstract = "".join(f'<AbstractText Label="{label}">{escape(text)}</AbstractText>' if label else
                       f"<AbstractText>{escape(text)}</AbstractText>" for label, text in sections)
    return (f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><Journal><JournalIssue><PubDate><Year>{year}</Year>"
            f"</PubDate></JournalIssue></Journal><ArticleTitle>{escape(title)}</ArticleTitle>"
            + (f"<Abstract>{abstract}</Abstract>" if abstract else "") + "</Article></MedlineCitation><PubmedData><ArticleIdList>"
            f'<ArticleId IdType="pubmed">{pmid}</ArticleId>' + (f'<ArticleId IdType="pmc">{pmcid}</ArticleId>' if pmcid else "")
            + "</ArticleIdList></PubmedData></PubmedArticle>")


def _pmc_xml(number: str) -> str:
    front = f'<front><article-meta><article-id pub-id-type="pmc">PMC{number}</article-id>'
    if number not in PMC:  # PMC holds it, but its publisher does not allow the full text to be downloaded
        return f"<article>{front}</article-meta></front></article>"
    licence, body = PMC[number]
    return (f'<article xmlns:xlink="http://www.w3.org/1999/xlink">{front}<permissions><license xlink:href="{licence}"/>'
            f"</permissions></article-meta></front><body>{body}</body></article>")
