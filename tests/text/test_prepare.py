"""The text dataset's preparation (src/text/prepare.py), with a local stand-in for NCBI's E-utilities (MeSH records and PubMed
counts), so no test reaches the internet."""

import contextlib
import csv
import hashlib
import io
import json
import re
import shlex
import tempfile
import threading
import unittest
import urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree

from src.text import prepare

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["melanoma", "nevus"]
MESH = {  # uid: (MeSH ID, heading and entry terms)
    "1": ("D000001", ["Melanoma", "Melanomas", "Malignant Melanoma"]),
    "2": ("D000002", ["Nevus", "Nevi", "Nevus, Pigmented"]),
    "3": ("D000003", ["Uveal Neoplasms", "Uveal Melanoma"]),
}
TERMS = """class,term,counts,source,note
melanoma,melanoma,yes,ISIC records of the image dataset,class name
melanoma,malignant melanoma,yes,proposed: to be confirmed in MeSH,other name
melanoma,uveal melanoma,no,decision 2026-11-12,not skin
melanoma,made-up name,yes,proposed: to be confirmed in MeSH,a name MeSH does not have
nevus,nevus,yes,ISIC records of the image dataset,class name
nevus,pigmented naevus,yes,proposed: to be confirmed in MeSH,British spelling and another word order
"""


class EutilsServer(unittest.TestCase):
    """A local E-utilities: esearch on mesh finds the records whose names hold the phrase, esummary returns them, and esearch on
    pubmed returns a count that shows what was asked: for a name in titles and summaries, 7 (0 and a warning for the made-up name);
    for a class, 100 per heading plus 1 (2025 or later), 4 (reviews), 2 (free full text) or 3 (all case reports)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name) / "data" / "text"
        self.data.mkdir(parents=True)
        (self.data / prepare.TERMS).write_text(TERMS, encoding="utf-8")
        self.queries, self.mode = [], "normal"
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urllib.parse.urlparse(self.path)
                q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
                tool = url.path.rsplit("/", 1)[-1]
                if tool == "esearch.fcgi" and q["db"] == "mesh":
                    phrase = q["term"].strip('"').lower()
                    ids = [uid for uid, (_, names) in MESH.items()
                           if any(phrase in n.lower() or prepare._words(n) == prepare._words(phrase) for n in names)]
                    body = {"esearchresult": {"count": str(len(ids)), "idlist": ids}}
                elif tool == "esummary.fcgi":
                    ids = q["id"].split(",")
                    if test.mode == "odd summary":
                        body = {"result": {"uids": ids, **{i: {"uid": i, "title": "?"} for i in ids}}}
                    else:
                        body = {"result": {"uids": ids, **{i: {"uid": i, "ds_meshui": MESH[i][0], "ds_meshterms": MESH[i][1]}
                                                           for i in ids}}}
                elif tool == "esearch.fcgi" and q["db"] == "pubmed" and "[Title/Abstract]" in q["term"]:
                    if "made-up name" in q["term"]:
                        body = {"esearchresult": {"count": "0", "idlist": [], "warninglist": {"quotedphrasesnotfound": [q["term"]]}}}
                    else:
                        body = {"esearchresult": {"count": "7", "idlist": []}}
                elif tool == "esearch.fcgi" and q["db"] == "pubmed":
                    term = q["term"]
                    test.queries.append(term)
                    if test.mode == "not understood":
                        body = {"esearchresult": {"count": "0", "idlist": [], "errorlist": {"phrasesnotfound": ["free full text[sb]"]}}}
                    elif test.mode.startswith("backend") and (test.mode == "backend down" or test.queries.count(term) == 1):
                        body = {"esearchresult": {"ERROR": "Search Backend failed: read tcp: i/o timeout"}}  # NCBI's own failure
                    else:
                        extra = 1 if "2025:3000" in term else 4 if "review[" in term else 2 if "free full text" in term else 3
                        body = {"esearchresult": {"count": str(100 * term.count("[MeSH Major Topic]") + extra), "idlist": []}}
                else:
                    self.send_error(404)
                    return
                content = json.dumps(body).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        for patcher in (mock.patch.object(prepare, "EUTILS", f"http://127.0.0.1:{self.server.server_port}/entrez/eutils/"),
                        mock.patch("src.common.time.sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _mesh_rows(self):
        with (self.data / prepare.CHECK).open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, prepare.CHECK_COLUMNS)
            return {row["term"]: row for row in reader}


class SurveyTests(EutilsServer):
    def test_every_name_is_checked_in_mesh_and_each_class_is_counted_with_the_headings_of_its_names_that_count(self):
        done = prepare.survey(self.data, classes=CLASSES)
        rows = self._mesh_rows()
        self.assertEqual({t: (r["in_mesh"], r["mesh_ids"], r["mesh_headings"], r["pubmed_articles"]) for t, r in rows.items()}, {
            "melanoma": ("heading", "D000001", "Melanoma", "7"), "malignant melanoma": ("entry term", "D000001", "Melanoma", "7"),
            "uveal melanoma": ("entry term", "D000003", "Uveal Neoplasms", "7"), "made-up name": ("no", "", "", "0"),
            "nevus": ("heading", "D000002", "Nevus", "7"), "pigmented naevus": ("entry term", "D000002", "Nevus", "7")})
        self.assertEqual(dict(done["headings"]), {"melanoma": ["Melanoma"], "nevus": ["Nevus"]})  # not the eye tumours: they do not count
        self.assertEqual(done["counts"]["melanoma"], {"case reports": 103, "case reports, free full text": 102,
                                                     "case reports, free full text, 2025 or later": 101,
                                                     "reviews and guidelines, free full text": 104})
        self.assertEqual(len(self.queries), 8)  # four counts for each of the two classes
        for query in self.queries:
            self.assertIn(prepare.SKIN, query)  # every count is limited to skin lesions
            self.assertNotIn("Uveal", query)

    def test_the_report_lists_the_names_not_in_mesh_and_dates_the_counts(self):
        prepare.survey(self.data, classes=CLASSES)
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("# Preparation of the text dataset", "<!-- step: survey -->", "## Survey: the names checked in MeSH",
                         "- Names: 6; a MeSH heading: 2; an entry term of a heading (another name MeSH lists for it): 3; not in MeSH: 1 "
                         "(used in PubMed titles or summaries: 0; in none: 1)",
                         "| melanoma | made-up name | yes | 0 | proposed: to be confirmed in MeSH |",
                         "| melanoma | 103 | 102 | 101 | 104 |", "| nevus | 103 | 102 | 101 | 104 |",
                         "| melanoma | Melanoma |", "free full text[sb]", "No article, abstract or case text is downloaded"):
            self.assertIn(expected, text)
        self.assertRegex(text, r"## How much PubMed holds per class \(\d{4}-\d{2}-\d{2}\)")
        first = (self.data / prepare.CHECK).read_bytes()
        prepare.survey(self.data, classes=CLASSES)
        self.assertEqual((self.data / prepare.CHECK).read_bytes(), first)  # the same answers give the same table

    def test_a_query_pubmed_does_not_understand_stops_the_step_before_anything_is_written(self):
        self.mode = "not understood"
        with self.assertRaisesRegex(ValueError, "PubMed did not understand part of the query .*phrasesnotfound"):
            prepare.survey(self.data, classes=CLASSES)
        self.assertFalse((self.data / prepare.CHECK).exists())
        self.assertFalse((self.data / prepare.REPORT).exists())

    def test_a_search_answer_without_a_count_is_tried_again_and_its_error_is_shown_when_it_never_comes(self):
        self.mode = "backend once"  # each search's first answer is NCBI's backend error
        prepare.survey(self.data, classes=CLASSES)
        self.assertTrue((self.data / prepare.CHECK).exists())
        self.mode = "backend down"
        with self.assertRaisesRegex(ValueError, "after 4 attempts .*no count: .*Search Backend failed"):
            prepare.survey(self.data, classes=CLASSES)

    def test_an_unexpected_mesh_summary_stops_the_step_and_names_the_fields_it_had(self):
        self.mode = "odd summary"
        with self.assertRaisesRegex(ValueError, r"MeSH summary of 1 has an unexpected form \(fields: title, uid\)"):
            prepare.survey(self.data, classes=CLASSES)

    def test_the_command_line_prints_the_result_and_what_to_commit(self):
        real = prepare.survey
        with mock.patch.object(prepare, "survey", lambda show=None: real(self.data, classes=CLASSES, show=show)), \
                contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["survey"])
        for expected in ("6 of 6 names checked in MeSH and PubMed", "melanoma: case reports 103", "Names in MeSH: 2 as a heading, 3 as "
                         "an entry term; not in MeSH: 1 (of them in no PubMed title or summary: 1)", "Commit both files"):
            self.assertIn(expected, shown.getvalue())


def _patient(uid, title, text, age=70.0, sex="F", published="2019-05-01"):  # the fields of the real file (its first run)
    return {"patient_id": uid, "patient_uid": uid, "PMID": uid.split("-")[0], "file_path": "comm/x.xml", "title": title,
            "patient": text, "age": [[age, "year"]], "gender": sex, "pub_date": published}


PATIENTS = [
    _patient("1001-1", "Pigmented basal cell carcinoma of the scalp: a case report", "A man had a pigmented basal cell carcinoma."),
    _patient("1002-1", "Melanoma mimicking a seborrhoeic keratosis", "The lesion was a melanoma on biopsy."),
    _patient("1003-1", "Keratoacanthoma of the hand", "A fast-growing nodule on the hand."),
    _patient("1004-1", "Squamous cell carcinoma of the tongue", "A squamous cell carcinoma of the lateral tongue."),
    _patient("1005-1", "Basal cell nevus syndrome in a child", "A child with many basal cell carcinomas."),
    _patient("1006-1", "Multiple dermatofibromas in two sisters", "The elder sister had six dermatofibromas; each dermatofibroma was removed.",
             age=30.0),
    _patient("1006-2", "Multiple dermatofibromas in two sisters", "The younger sister had two dermatofibromas.", age=25.0),
    _patient("1007-1", "An unusual lesion of the back", "Histology showed a melanoma."),
    _patient("1008-1", "A giant congenital naevus", "A large hairy plaque since birth.", age=2.0, sex="M", published="2008"),
    _patient("1009-1", "Lentigo maligna on the nose", "A slowly growing brown patch: lentigo maligna.", published=None),
    _patient("1010-1", "Squamous cell carcinoma arising in a keratoacanthoma", "A squamous cell carcinoma on the forearm."),
    _patient("1011-1", "Cutaneous melanoma metastases: a case report", "A melanoma had spread within the skin."),
    _patient("1012-1", "Blue rubber bleb nevus syndrome in a girl", "Many soft blue nodules on the skin and in the gut."),
    _patient("1013-1", "Congenital sebaceous nevus of the scalp", "A yellow plaque on the scalp since birth."),
    _patient("1014-1", "Squamous cell carcinoma of the head and neck treated with cemiplimab",
             "A cutaneous SCC of the scalp spread to the neck; both SCCs were treated."),
    _patient("1015-1", "Actinic keratosis on the neck", "Two AKs on the neck of a farmer."),
]
CONTENT = json.dumps(PATIENTS, indent=1).encode("utf-8")
COMMIT = "0123456789abcdef0123456789abcdef01234567"


class HuggingFaceServer(unittest.TestCase):
    """A local Hugging Face: the repository's description (with its version) and the file of that version, sent in parts on
    request (HTTP ranges); it can cut the connection once in the middle, as a failing network does."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name) / "data" / "text"
        self.data.mkdir(parents=True)
        (self.data / prepare.TERMS).write_bytes((ROOT / "data" / "text" / prepare.TERMS).read_bytes())
        self.requests, self.mode, self.content, self.version = [], "normal", CONTENT, COMMIT
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                test.requests.append((self.path, self.headers.get("Range")))
                if self.path == "/api/datasets/zhengyun21/PMC-Patients":
                    body, status = json.dumps({"id": "zhengyun21/PMC-Patients", "sha": test.version}).encode(), 200
                elif self.path == f"/datasets/zhengyun21/PMC-Patients/resolve/{COMMIT}/PMC-Patients-V2.json":
                    self.send_response(302)  # as Hugging Face does: the file itself is served from its file store
                    self.send_header("Location", "/store/PMC-Patients-V2.json")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                elif self.path == "/store/PMC-Patients-V2.json":
                    start = int(self.headers["Range"].split("=")[1].rstrip("-")) if self.headers.get("Range") else 0
                    if test.mode == "no ranges":
                        start = 0
                    elif start >= len(test.content):
                        self.send_error(416)
                        return
                    body, status = test.content[start:], 206 if start else 200
                    if test.mode in ("cut once", "cut once, then no ranges"):  # promise the whole rest, send half of it and close
                        test.mode = "normal" if test.mode == "cut once" else "no ranges"
                        self.send_response(status)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body[: len(body) // 2])
                        return
                else:
                    self.send_error(404)
                    return
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        base = f"http://127.0.0.1:{self.server.server_port}"
        for patcher in (mock.patch.object(prepare, "HF_API", base + "/api/datasets/{repo}"),
                        mock.patch.object(prepare, "HF_FILE", base + "/datasets/{repo}/resolve/{commit}/{file}"),
                        mock.patch("src.common.time.sleep"), mock.patch("src.text.prepare.time.sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.folder = self.data / prepare.SOURCES / "pmc_patients"

    def _rows(self):
        with (self.data / prepare.CANDIDATES).open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, prepare.CANDIDATE_COLUMNS)
            return list(reader)


class CasesTests(HuggingFaceServer):
    CLASSES = ["actinic keratosis", "basal cell carcinoma", "dermatofibroma", "melanoma", "nevus", "pigmented benign keratosis",
               "squamous cell carcinoma", "vascular lesion"]

    def test_the_file_is_downloaded_once_from_a_fixed_version_and_every_patient_whose_title_names_a_class_is_listed(self):
        done = prepare.cases(self.data, classes=self.CLASSES)
        self.assertEqual((self.folder / "PMC-Patients-V2.json").read_bytes(), CONTENT)
        meta = json.loads((self.folder / "download.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["commit"], meta["bytes"], meta["sha256"], meta["licence"]),
                         (COMMIT, len(CONTENT), hashlib.sha256(CONTENT).hexdigest(), "CC BY-NC-SA 4.0"))
        self.assertEqual(done["read"], 16)
        self.assertEqual([(r["patient_uid"], r["class"], r["classes_named"], r["marks"]) for r in self._rows()], [
            ("1015-1", "actinic keratosis", "actinic keratosis", ""),  # "neck" alone is skin: only "head and neck" is marked
            ("1001-1", "basal cell carcinoma", "basal cell carcinoma", ""),
            ("1006-1", "dermatofibroma", "dermatofibroma", ""),
            ("1006-2", "dermatofibroma", "dermatofibroma", ""),
            ("1009-1", "melanoma", "melanoma", ""),  # lentigo maligna: a melanoma, not a lentigo
            ("1011-1", "melanoma", "melanoma", "a site that may not be skin: metastases"),  # the word as written, not its form
            ("1008-1", "nevus", "nevus", "the case text does not name the class"),  # British spelling in the title
            ("1004-1", "squamous cell carcinoma", "squamous cell carcinoma", "a site that may not be skin: tongue"),
            ("1010-1", "squamous cell carcinoma", "squamous cell carcinoma", "also names keratoacanthoma (does not count)"),
            ("1014-1", "squamous cell carcinoma", "squamous cell carcinoma",  # the text names it by its short form only
             "a site that may not be skin: head and neck; the title is about treatment: cemiplimab"),
            ("1002-1", "", "melanoma; pigmented benign keratosis", "names two or more classes")])
        # blue rubber bleb nevus and sebaceous nevus (the name "nevus sebaceous" in the other word order) are not moles
        self.assertEqual(done["read"] - len(self._rows()), 4 + 1)  # 1003, 1005, 1012, 1013 name no class; 1007 names nothing
        found = {r["patient_uid"]: (r["year"], r["text_full_names"], r["text_short_forms"], r["age"]) for r in self._rows()}
        self.assertEqual([found[uid] for uid in ("1001-1", "1006-1", "1008-1", "1009-1", "1014-1", "1015-1", "1002-1")], [
            ("2019", "1", "0", "70 year"), ("2019", "2", "0", "30 year"), ("2008", "0", "0", "2 year"), ("", "1", "0", "70 year"), ("2019", "0", "2", "70 year"),
            ("2019", "0", "1", "70 year"), ("2019", "", "", "70 year")])
        first = {name: (self.data / name).read_bytes() for name in (prepare.CANDIDATES,)}
        asked = len(self.requests)
        prepare.cases(self.data, classes=self.CLASSES)  # again: nothing is downloaded, the same list comes out
        self.assertEqual(len(self.requests), asked)
        self.assertEqual({name: (self.data / name).read_bytes() for name in first}, first)

    def test_a_cut_off_download_continues_where_it_stopped(self):
        self.mode = "cut once"
        prepare.cases(self.data, classes=self.CLASSES)
        self.assertEqual((self.folder / "PMC-Patients-V2.json").read_bytes(), CONTENT)
        ranges = [r for path, r in self.requests if path.startswith("/store/") and r]
        self.assertEqual(ranges, [f"bytes={len(CONTENT) // 2}-"])  # only the missing part was asked for again
        self.assertFalse((self.folder / "PMC-Patients-V2.json.part").exists())

    def test_a_server_that_sends_the_whole_file_again_or_has_nothing_left_to_send_still_gives_the_right_file(self):
        self.mode = "cut once, then no ranges"
        prepare.cases(self.data, classes=self.CLASSES)
        self.assertEqual((self.folder / "PMC-Patients-V2.json").read_bytes(), CONTENT)
        for name in ("PMC-Patients-V2.json", "download.json"):  # a run stopped after the last byte, before the file was named
            (self.folder / name).unlink()
        (self.folder / "PMC-Patients-V2.json.part").write_bytes(CONTENT)
        (self.folder / "download.part.json").write_text(json.dumps({"commit": COMMIT}), encoding="utf-8")
        self.mode, self.version, asked = "normal", "f" * 40, len(self.requests)  # meanwhile the repository has a newer version
        prepare.cases(self.data, classes=self.CLASSES)
        self.assertEqual((self.folder / "PMC-Patients-V2.json").read_bytes(), CONTENT)
        self.assertEqual(self.requests[asked:], [(f"/datasets/zhengyun21/PMC-Patients/resolve/{COMMIT}/PMC-Patients-V2.json",
                                                  f"bytes={len(CONTENT)}-"),  # the version the download started from
                                                 ("/store/PMC-Patients-V2.json", f"bytes={len(CONTENT)}-")])

    def test_a_changed_download_or_a_patient_without_the_needed_fields_stops_the_step(self):
        prepare.cases(self.data, classes=self.CLASSES)
        (self.folder / "PMC-Patients-V2.json").write_bytes(CONTENT + b" ")
        with self.assertRaisesRegex(ValueError, "has changed since it was downloaded"):
            prepare.cases(self.data, classes=self.CLASSES)
        for name in ("PMC-Patients-V2.json", "download.json"):
            (self.folder / name).unlink()
        self.content = json.dumps([{"patient_uid": "1-1", "title": "Melanoma"}]).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "patient 1 of .* lacks a field this step needs .*; it has: patient_uid, title"):
            prepare.cases(self.data, classes=self.CLASSES)

    def test_the_year_is_the_first_four_digit_year_in_the_publication_date(self):
        for value, year in (("2015-03-12", "2015"), ("20150312", "2015"), ("12/03/2015", "2015"), (2015, "2015"),
                            (["2015", "3"], "2015"), ("unknown", ""), (None, ""), ("", ""), ("1850", "")):
            self.assertEqual(prepare._year(value), year, value)

    def test_titles_naming_the_class_only_as_a_look_alike_an_animal_or_another_site_are_marked(self):
        judge = prepare._judge(prepare.read_terms(self.data / prepare.TERMS, self.CLASSES), self.CLASSES)
        cases = {
            "Sebaceoma on the nose mimicking basal cell carcinoma": ("basal cell carcinoma",
                                                                     f"{prepare.LOOKS_MARK}; {prepare.OTHER_MARK}: sebaceoma"),
            "Pigmented Mammary Paget Disease Misdiagnosed as Malignant Melanoma": ("melanoma", prepare.LOOKS_MARK),
            "Melanoma mimicking a cyst": ("melanoma", ""),  # the class comes first: it is the diagnosis
            "Melanoma misdiagnosed as a diabetic foot ulcer": ("melanoma", ""),
            "Cutaneous melanoma with a rhabdomyosarcomatous component in a dog": ("melanoma", "the title is about an animal: dog"),
            "Squamous cell carcinoma of the hoof in a mare": ("squamous cell carcinoma", "the title is about an animal: mare"),
            "Melanoma of the calf in a young woman": ("melanoma", ""),  # the calf of a leg, not an animal
            "Primary malignant melanoma of the trachea": ("melanoma", "a site that may not be skin: trachea"),
            "Primary Central Nervous System Melanoma": ("melanoma", "a site that may not be skin: central nervous system"),
            "Perianal squamous cell carcinoma in a man": ("squamous cell carcinoma", "a site that may not be skin: perianal"),
            "Pilar leiomyoma with dermatofibroma-like dermatoscopic features": (
                "dermatofibroma", f"{prepare.LOOKS_MARK}; {prepare.OTHER_MARK}: leiomyoma"),
            "Sebaceous carcinoma associated with seborrheic keratosis": (
                "pigmented benign keratosis", f"{prepare.LOOKS_MARK}; {prepare.OTHER_MARK}: carcinoma"),
            "Dermoscopy aids in differentiating cafe-au-lait macules from congenital melanocytic nevi": ("nevus", prepare.LOOKS_MARK),
            "Hidradenoma arising in a congenital nevus": ("nevus", f"{prepare.LOOKS_MARK}; {prepare.OTHER_MARK}: hidradenoma"),
            "Basal cell carcinoma arising in rhinophyma": ("basal cell carcinoma", ""),  # the class first: the diagnosis
            "Melanoma associated with pregnancy": ("melanoma", ""),
            "Mucinous nevus": ("nevus", prepare.BARE_MARK),  # the bare word: not always a mole
            "Dermoscopic changes in nevi during an atopic dermatitis flare-up": ("nevus", prepare.BARE_MARK),
            "Intradermal nevus of the scalp": ("nevus", ""), "Multiple acquired melanocytic nevi in a child": ("nevus", ""),
            "Intravenous lobular capillary hemangioma after needle insertion": ("vascular lesion",
                                                                                "a site that may not be skin: intravenous"),
            "Superficial spreading cervical squamous cell carcinoma as intrauterine nodules": (
                "squamous cell carcinoma", "a site that may not be skin: intrauterine"),
            "Basal cell carcinoma of the nipple-areola complex": ("basal cell carcinoma", ""),  # the nipple's skin
            "Cutaneous plasmacytoma adjacent to Bowenoid actinic keratosis on the scalp: Is there a link?": (
                "actinic keratosis", f"{prepare.OTHER_MARK}: plasmacytoma"),  # a second lesion
            "Fibroepithelial Polyp With Actinic Keratosis at Its Base and a Cutaneous Horn": (
                "actinic keratosis", f"{prepare.OTHER_MARK}: polyp"),
            "Composite Tumor Associating Trichoblastoma and Seborrheic Keratosis": (
                "pigmented benign keratosis", f"{prepare.OTHER_MARK}: trichoblastoma"),
            "Squamous cell carcinoma arising in mature cystic teratomas": (
                "squamous cell carcinoma", f"{prepare.OTHER_MARK}: teratoma"),  # arising in: a second finding
            "Synchronous Merkel cell carcinoma and primary cutaneous melanoma": ("melanoma", f"{prepare.OTHER_MARK}: carcinoma"),
            "Spectrum of disease from condylomata to squamous cell carcinoma": (
                "squamous cell carcinoma", f"{prepare.OTHER_MARK}: condylomata"),
            "A case of cellular neurothekeoma instead of basal cell carcinoma": (
                "basal cell carcinoma", f"{prepare.OTHER_MARK}: neurothekeoma"),
            "Large ulcerated melanoma mimicking extramedullary plasmacytoma": ("melanoma", ""),  # only what it looked like
            "A mimicker of dermatofibroma: cutaneous leiomyoma": (  # no class before the look-alike word: the diagnosis
                "dermatofibroma", f"{prepare.LOOKS_MARK}; {prepare.OTHER_MARK}: leiomyoma"),
            "Basal cell carcinoma misdiagnosed as trichoepithelioma": ("basal cell carcinoma", ""),
            "Atypical fibroxanthoma-like amelanotic melanoma": ("melanoma", ""),
            "Bowen's disease and human papilloma virus": ("squamous cell carcinoma", ""),  # no lesion
            "Squamous cell carcinoma with an anaplastic lymphoma kinase fusion": ("squamous cell carcinoma", ""),
            "Pseudovascular squamous cell carcinoma with eosinophil-rich stroma": ("squamous cell carcinoma", ""),
            "Lobular capillary hemangioma of the finger in a woman": ("vascular lesion", ""),  # a name of class_terms.csv
            "Cutaneous basaloid squamous cell carcinoma in a captive blue-fronted amazon (Amazona aestiva)": (
                "squamous cell carcinoma", "the title is about an animal: amazona"),
            "Melanoma in a pet bird": ("melanoma", "the title is about an animal: bird"),
        }
        for title, (label, mark) in cases.items():
            row, _ = judge(title, label)
            with self.subTest(title=title):
                self.assertEqual((row["class"], row["marks"]), (label, mark))
        self.assertEqual(judge("Syphilis resembling non-melanoma skin cancer", "x"), (None, True))  # not melanoma
        for title in ("Vitiligo coexistent with nevus depigmentosus", "Two cases of giant melanocytoma (hyperpigmented magnocellular "
                      "nevus)", "Multiple superficial lipomatous nevi with a verrucous nevus",  # not moles
                      "Acral pseudolymphomatous angiokeratoma of children (APACHE)"):  # a pseudolymphoma, not an angiokeratoma
            self.assertEqual(judge(title, "x"), (None, True), title)
        self.assertIn(prepare.HUMANS, prepare.ABSTRACT_QUERY)

    def test_the_reader_takes_items_split_across_reads_and_refuses_a_cut_off_file(self):
        path = Path(self.data) / "list.json"
        path.write_bytes(CONTENT)
        self.assertEqual(list(prepare._json_list(path, chunk=7)), PATIENTS)
        path.write_bytes(CONTENT[:-40])
        with self.assertRaisesRegex(ValueError, "cut off"):
            list(prepare._json_list(path, chunk=7))
        path.write_text('{"not": "a list"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "is not one JSON list"):
            list(prepare._json_list(path))

    def test_the_report_names_the_source_version_and_counts_the_candidates_per_class(self):
        prepare.cases(self.data, classes=self.CLASSES)
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("## Cases: candidate patient cases for each class", f"version (commit) `{COMMIT}`",
                         f"SHA-256 `{hashlib.sha256(CONTENT).hexdigest()}`", "licence CC BY-NC-SA 4.0", "- Patients read: 16;",
                         "- Publication years of the candidates (from `pub_date`): 2008–2019; candidates without a year: 1",
                         "- Candidates: 11 patients from 10 articles; titles naming only names that do not count (for example "
                         "keratoacanthoma), so no candidate: 4",
                         "| actinic keratosis | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 2019–2019 |\n",
                         "| dermatofibroma | 2 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 2019–2019 |\n",
                         "| melanoma | 3 | 3 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 2019–2019 |\n",
                         "| nevus | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 2008–2008 |\n",
                         "| squamous cell carcinoma | 3 | 3 | 0 | 0 | 1 | 2 | 1 | 0 | 1 | 2019–2019 |\n",
                         "| vascular lesion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | none |\n", "carboplatin, cemiplimab, cetuximab",
                         "head and neck", "squamous cell carcinoma: SCC, cSCC", "No case is read, chosen or cut"):
            self.assertIn(expected, text)

    def test_the_command_line_prints_the_candidates_per_class_and_what_to_commit(self):
        real = prepare.cases
        with mock.patch.object(prepare, "cases", lambda show=None: real(self.data, classes=self.CLASSES, show=show)), \
                contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["cases"])
        for expected in ("downloading PMC-Patients-V2.json (version 0123456789ab)", "Patients read: 16; candidates: 11 from 10 "
                         "articles (5 without marks)", "  dermatofibroma: 2 candidates, 2 without marks",
                         "  vascular lesion: 0 candidates, 0 without marks", "Commit both files"):
            self.assertIn(expected, shown.getvalue())


ABSTRACT_ARTICLES = {  # pmid: (main-topic headings, year, title, abstract sections); 3006 is already from PMC-Patients
    "3001": ({"Keratosis, Actinic"}, 2015, "Actinic keratosis on the scalp of a farmer",
             [["BACKGROUND", "Actinic keratosis is common in older people."],
              ["CASE PRESENTATION", "A 70-year-old farmer had a rough scaly patch on the scalp for a year. Histology showed an "
                                    "actinic keratosis."], ["CONCLUSIONS", "Sun protection matters."]]),
    "3002": ({"Keratosis, Actinic"}, 2015, "A hypertrophic actinic keratosis",
             [["", "We report a case of actinic keratosis on the hand of a sailor."]]),  # no labels: the whole abstract
    "3003": ({"Keratosis, Actinic"}, 2020, "Pigmented actinic keratosis mimicking melanoma",
             [["", "An actinic keratosis that looked like a melanoma."]]),  # two classes: marked
    "3004": ({"Keratosis, Seborrheic"}, 2012, "Seborrheic keratosis of the eyelid",
             [["CASE REPORT", "A 60-year-old man had a waxy brown papule on the eyelid: a seborrheic keratosis."]]),
    "3005": ({"Keratosis, Seborrheic"}, 2021, "Seborrheic keratosis in a child", []),  # no abstract
    "3006": ({"Keratosis, Actinic"}, 2016, "Actinic keratosis treated with light", [["", "An actinic keratosis."]]),
    "3007": ({"Keratosis, Actinic"}, 2022, "Actinic keratosis of the lip", [["", "An actinic keratosis."]]),  # PubMed returns none
    "3008": ({"Keratosis, Seborrheic"}, 2019, "Keratoacanthoma of the cheek", [["", "A keratoacanthoma, not a keratosis."]]),
    "3009": ({"Keratosis, Actinic"}, 2025, "Actinic keratosis on the ear of a gardener",
             [["", "We describe an actinic keratosis on the ear."]]),  # from 2025 with an open full text: its case section is used
    "3010": ({"Keratosis, Actinic"}, 2025, "Actinic keratosis of the nose", [["", "An actinic keratosis of the nose in a woman."]]),
    "3011": ({"Keratosis, Actinic"}, 2018, "Actinic keratosis of the cheek",
             [["", "A rough red patch on the cheek of an older man."]]),  # the abstract does not name it; the full text does
    "3012": (set(), 2025, "Seborrheic keratosis of the scalp",
             [["", "A 30-year-old woman had a waxy brown plaque on the scalp: a seborrheic keratosis."]]),  # not indexed: by title
}
PMC_NUMBERS = {"3001": "PMC9001", "3003": "PMC9003", "3009": "PMC9009", "3010": "PMC9010", "3011": "PMC9011"}  # 3003: two classes
FULL_CASE = ("A 72-year-old gardener had a rough, scaly pink patch on the left ear for two years. It bled after minor scratching "
             "(Figure 1). Histology showed an actinic keratosis.")
PMC_ARTICLES = {
    "9009": ('<article><front><article-meta><article-id pub-id-type="pmid">3009</article-id><article-id pub-id-type="pmc">PMC9009'
             '</article-id><permissions><license license-type="open-access" xlink:href="https://creativecommons.org/licenses/by/4.0/">'
             '<license-p>CC BY</license-p></license></permissions></article-meta></front><body><sec><title>Introduction</title>'
             '<p>Actinic keratosis is common.</p></sec><sec><title>Case Presentation</title><p>A 72-year-old gardener had a rough, '
             'scaly pink patch on the left ear for two years. It bled after minor scratching (<xref ref-type="fig">Figure 1</xref>).'
             '<fig><caption><p>The patch on the ear.</p></caption></fig></p><sec><title>Histology</title><p>Histology showed an '
             'actinic keratosis.</p></sec></sec><sec><title>Patient consent</title><p>Consent was obtained.</p></sec></body></article>'),
    "9001": '<article><front><article-meta><article-id pub-id-type="pmc">PMC9001</article-id></article-meta></front></article>',
    "9011": ('<article><front><article-meta><article-id pub-id-type="pmc">PMC9011</article-id></article-meta></front><body><sec>'
             '<title>Case report</title><p>A 77-year-old man had a rough red patch on the cheek for a year. Histology showed an '
             'actinic keratosis.</p></sec></body></article>'),
    "9010": '<article><front><article-meta><article-id pub-id-type="pmc">PMC9010</article-id></article-meta></front></article>'}
# PMC gives 9001 and 9010 without their body (not open): their abstracts are used


class AbstractsTests(unittest.TestCase):
    """A local PubMed for the abstracts step: esearch finds the made-up articles whose main topic is one of the query's headings,
    or, for the search of articles PubMed has not indexed, those without headings whose title holds one of the query's names,
    within its years; a name in no_title is in no PubMed title, so a search by title for it gets PubMed's warning that the
    phrase is not found; efetch (POST) returns their records as PubMed's XML, except 3007's."""
    CLASSES = CasesTests.CLASSES

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name) / "data" / "text"
        self.data.mkdir(parents=True)
        (self.data / prepare.TERMS).write_bytes((ROOT / "data" / "text" / prepare.TERMS).read_bytes())
        headings = {"actinic keratosis": "Keratosis, Actinic", "seborrheic keratosis": "Keratosis, Seborrheic"}
        terms = prepare.read_terms(self.data / prepare.TERMS, self.CLASSES)
        prepare._write_csv(self.data / prepare.CHECK, prepare.CHECK_COLUMNS, [
            {"class": t["class"], "term": t["term"], "counts": t["counts"], "in_mesh": "", "mesh_ids": "",
             "mesh_headings": headings.get(t["term"], ""), "pubmed_articles": 1} for t in terms])
        prepare._write_csv(self.data / prepare.CANDIDATES, prepare.CANDIDATE_COLUMNS, [
            {**{c: "" for c in prepare.CANDIDATE_COLUMNS}, "patient_uid": "9-1", "pmid": "3006", "class": "actinic keratosis"}])
        self.searches, self.fetches, self.pmc, self.mode, self.no_title = [], [], [], "normal", set()
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, content, kind):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def do_GET(self):
                q = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}
                test.searches.append(q["term"])
                unknown = [f'"{name}"[Title]' for name in re.findall(r'"([^"]+)"\[Title\]', q["term"]) if name in test.no_title]
                warning = {"warninglist": {"quotedphrasesnotfound": unknown}} if unknown else {}
                if "[Date - Publication]" not in q["term"]:  # which of the names some title uses
                    return self._send(json.dumps({"esearchresult": {"count": "5", "idlist": [], **warning}}).encode(), "application/json")
                first, last = map(int, re.search(r"(\d+):(\d+)\[Date - Publication\]", q["term"]).groups())
                if prepare.UNINDEXED_QUERY in q["term"]:
                    names = [n.lower() for n in re.findall(r'"([^"]+)"\[Title\]', q["term"])]
                    ids = [p for p, (h, year, title, _) in ABSTRACT_ARTICLES.items()
                           if not h and first <= year <= last and any(n in title.lower() for n in names)]
                else:
                    wanted = set(re.findall(r'"([^"]+)"\[MeSH Major Topic\]', q["term"]))
                    assert q["term"].count(prepare.ABSTRACT_QUERY) == 1
                    ids = [p for p, (h, year, *_) in ABSTRACT_ARTICLES.items() if h & wanted and first <= year <= last]
                listed = ids[:int(q["retmax"])][:-1 if test.mode == "short list" else None]
                body = {"esearchresult": {"count": str(len(ids)), "idlist": listed, **warning}}
                self._send(json.dumps(body).encode(), "application/json")

            def do_POST(self):
                form = {k: v[0] for k, v in urllib.parse.parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode()).items()}
                ids = form["id"].split(",")
                if form["db"] == "pmc":
                    test.pmc.append(ids)
                    articles = "".join(PMC_ARTICLES[i] for i in ids)
                    return self._send(f'<pmc-articleset xmlns:xlink="http://www.w3.org/1999/xlink">{articles}</pmc-articleset>'.encode(),
                                      "text/xml")
                test.fetches.append(ids)
                parts = []
                for pmid in ids:
                    if pmid == "3007":
                        continue
                    _, year, title, sections = ABSTRACT_ARTICLES[pmid]
                    abstract = "".join(f'<AbstractText{f" Label={chr(34)}{label}{chr(34)}" if label else ""}>{text}</AbstractText>'
                                       for label, text in sections)
                    parts.append(f"<PubmedArticle><MedlineCitation><PMID Version=\"1\">{pmid}</PMID><Article><Journal><JournalIssue>"
                                 f"<PubDate><Year>{year}</Year><Month>Mar</Month></PubDate></JournalIssue></Journal>"
                                 f"<ArticleTitle>{title.replace('Seborrheic', '<i>Seborrheic</i>')}</ArticleTitle>"
                                 f"{f'<Abstract>{abstract}</Abstract>' if sections else ''}</Article></MedlineCitation>"
                                 f"<PubmedData><ArticleIdList><ArticleId IdType=\"pubmed\">{pmid}</ArticleId>"
                                 + (f'<ArticleId IdType="pmc">{PMC_NUMBERS[pmid]}</ArticleId>' if pmid in PMC_NUMBERS else "")
                                 + "</ArticleIdList></PubmedData></PubmedArticle>")
                self._send(("<?xml version=\"1.0\"?><PubmedArticleSet>" + "".join(parts) + "</PubmedArticleSet>").encode(),
                           "text/xml")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        for patcher in (mock.patch.object(prepare, "EUTILS", f"http://127.0.0.1:{self.server.server_port}/entrez/eutils/"),
                        mock.patch.object(prepare, "ESEARCH_MAX", 3), mock.patch.object(prepare, "EFETCH_BATCH", 2),
                        mock.patch("src.common.time.sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_case_reports_not_in_pmc_patients_are_downloaded_and_judged_by_the_rules_of_cases(self):
        check = prepare._read_csv(self.data / prepare.CHECK, prepare.CHECK_COLUMNS, "survey")
        for row in check:  # a name PubMed has never used: not searched by title (PubMed refuses a phrase it does not know)
            row["pubmed_articles"] = 0 if row["term"] == "pigmented benign keratosis" else row["pubmed_articles"]
        prepare._write_csv(self.data / prepare.CHECK, prepare.CHECK_COLUMNS, check)
        self.no_title = {"solar lentigo"}  # used in abstracts, in no title: left out of the search by title
        done = prepare.abstracts(self.data, classes=self.CLASSES)
        with (self.data / prepare.ABSTRACT_CANDIDATES).open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, prepare.CANDIDATE_COLUMNS)
            rows = list(reader)
        self.assertEqual([(r["patient_uid"], r["class"], r["marks"], r["words"], r["year"]) for r in rows], [
            ("pubmed-3001", "actinic keratosis", "", "19", "2015"),  # only the case part: 2 sentences, 19 words
            ("pubmed-3002", "actinic keratosis", "", "13", "2015"),
            ("pubmed-3009", "actinic keratosis", "", "28", "2025"),  # the full text's case sections, not the abstract
            ("pubmed-3010", "actinic keratosis", "", "9", "2025"),  # PMC gave no body: the abstract
            ("pubmed-3011", "actinic keratosis", "", "19", "2018"),  # any year: the full text names the class, the abstract not
            ("pubmed-3004", "pigmented benign keratosis", "", "14", "2012"),  # the title's markup is read as text
            ("pubmed-3012", "pigmented benign keratosis", "", "14", "2025"),  # not indexed: found by its title
            ("pubmed-3003", "", "names two or more classes", "8", "2020")])
        self.assertEqual((done["downloaded"], done["fulltexts"]), (10, 2))  # 3001-3005, 3008-3012; 3006 left out, 3007 not returned
        unindexed = [t for t in self.searches if prepare.UNINDEXED_QUERY in t]
        self.assertEqual(len(unindexed), 8)  # one per class: every class has names that count
        self.assertTrue(unindexed[5].startswith('("seborrheic keratosis"[Title] OR '))
        self.assertNotIn('"solar lentigo"[Title]', unindexed[5])
        self.assertIn('"solar lentigo"[Title]', [t for t in self.searches if "[Date - Publication]" not in t][5])
        self.assertEqual(self.pmc, [["9001", "9009", "9010", "9011"]])  # every year, but only articles whose title passes (not 3003)
        self.assertTrue(all(len(ids) <= 2 for ids in self.fetches) and "3006" not in sum(self.fetches, []))
        years = {name: [re.search(r"(\d+:\d+)\[Date", t).group(1) for t in self.searches if name in t and "[Date" in t]
                 for name in ("Actinic", "Seborrheic")}
        self.assertEqual(years["Seborrheic"], ["1800:3000"])  # 3 articles: one search
        self.assertEqual(years["Actinic"][:2], ["1800:3000", "1800:2400"])  # 5 articles, more than 3: asked for in parts by year
        self.assertTrue({"2008:2016", "2017:2025"} <= set(years["Actinic"]))
        stored = [json.loads(line) for line in (self.data / "sources" / "pubmed" / "abstracts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["pmid"] for r in stored], ["3001", "3002", "3003", "3004", "3005", "3008", "3009", "3010", "3011", "3012"])
        self.assertEqual((stored[6]["fulltext"], stored[6]["licence"]), (FULL_CASE, "https://creativecommons.org/licenses/by/4.0/"))
        self.assertEqual(prepare._record_case(stored[6]), (FULL_CASE, "PMC full text"))  # no figure, introduction or consent
        self.assertEqual(prepare._record_case(stored[7]), ("An actinic keratosis of the nose in a woman.", "PubMed abstract"))
        self.assertEqual((stored[0]["pmcid"], stored[0]["fulltext"]), ("PMC9001", ""))  # no open full text: its abstract
        self.assertEqual(prepare._case_part(stored[0]["sections"]),
                         "A 70-year-old farmer had a rough scaly patch on the scalp for a year. Histology showed an actinic keratosis.")
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("## Abstracts: case reports PubMed holds as an abstract", "12 case reports found; already given by "
                         "PMC-Patients (left out): 1; downloaded: 10 (not returned by PubMed: 1); without an abstract: 1",
                         "- Candidate articles (title passes) with a PMC number, asked from PMC: 4; with an open full text and a "
                         "case section (used instead of the abstract): 2", "- Candidates: 8; titles naming only names that do not count, so no "
                         "candidate: 1", "| actinic keratosis | Keratosis, Actinic | 8 | 0 | 1 | 6 | 2 | 5 | 5 |",
                         "| pigmented benign keratosis | Keratosis, Seborrheic | 4 | 1 | 0 | 3 | 0 | 2 | 2 |",
                         "| melanoma | none | 0 | 0 | 0 | 0 | 0 | 0 | 0 |", "NOT medline[sb]"):
            self.assertIn(expected, text)

    def test_more_articles_in_one_year_than_pubmed_lists_or_a_list_shorter_than_its_count_stops_the_step(self):
        with mock.patch.object(prepare, "ESEARCH_MAX", 1), \
                self.assertRaisesRegex(ValueError, "more than 1 articles of 2015 .*: they cannot be listed. Nothing was written"):
            prepare.abstracts(self.data, classes=self.CLASSES)
        self.mode = "short list"
        with self.assertRaisesRegex(ValueError, "PubMed counted 3 articles but listed 2 for"):
            prepare.abstracts(self.data, classes=self.CLASSES)
        self.assertFalse((self.data / prepare.ABSTRACT_CANDIDATES).exists())

    def test_the_command_line_prints_the_candidates_per_class_and_what_to_commit(self):
        real = prepare.abstracts
        with mock.patch.object(prepare, "abstracts", lambda show=None: real(self.data, classes=self.CLASSES, show=show)), \
                contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["abstracts"])
        for expected in ("actinic keratosis: 8 case reports in PubMed", "11 of 11 abstracts downloaded", "4 of 4 full texts asked from "
                         "PMC", "Abstracts downloaded: 10 (with an open full text's case sections: 2); candidates: 8 (7 without marks)",
                         "  actinic keratosis: 5 candidates, 5 without marks", "Commit both files"):
            self.assertIn(expected, shown.getvalue())


MESH_CHECK = [  # term_check.csv for TERMS: the headings the survey found
    ("melanoma", "melanoma", "yes", "Melanoma"), ("melanoma", "malignant melanoma", "yes", "Melanoma"),
    ("melanoma", "uveal melanoma", "no", "Uveal Neoplasms"), ("melanoma", "made-up name", "yes", ""),
    ("nevus", "nevus", "yes", "Nevus"), ("nevus", "pigmented naevus", "yes", "Nevus, Pigmented")]
INDEX = {  # PubMed's subject headings per article (none: not indexed), and whether it is indexed under skin
    "11": ({"Melanoma"}, True), "12": ({"Melanoma"}, False), "13": ({"Nevus, Pigmented"}, True), "14": ({"Uveal Neoplasms"}, True),
    "15": ({"Melanoma", "Nevus"}, True), "16": ({"Nevus"}, True), "17": (set(), False), "18": ({"Melanoma"}, True)}


class MeshServer(unittest.TestCase):
    """A local E-utilities for the mesh step: epost (POST) keeps the article numbers sent under a query key; esearch on
    "#key AND medline[sb]" returns those of them with headings, and on "#key AND <class headings> AND <skin>" those the made-up
    index confirms. 17 is not indexed; 99 is unknown to PubMed."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data = Path(tmp.name) / "data" / "text"
        self.data.mkdir(parents=True)
        (self.data / prepare.TERMS).write_text(TERMS, encoding="utf-8")
        prepare._write_csv(self.data / prepare.CHECK, prepare.CHECK_COLUMNS, [
            {"class": c, "term": t, "counts": n, "in_mesh": "heading" if h else "no", "mesh_ids": "", "mesh_headings": h,
             "pubmed_articles": 7} for c, t, n, h in MESH_CHECK])
        rows = [("11", "melanoma"), ("12", "melanoma"), ("13", "nevus"), ("14", "nevus"), ("15", ""), ("16", "melanoma"),
                ("17", "melanoma"), ("99", "nevus")]
        prepare._write_csv(self.data / prepare.CANDIDATES, prepare.CANDIDATE_COLUMNS, [
            {**{c: "" for c in prepare.CANDIDATE_COLUMNS}, "patient_uid": f"{p}-1", "pmid": p, "class": c,
             "classes_named": c or "melanoma; nevus"} for p, c in rows])
        prepare._write_csv(self.data / prepare.ABSTRACT_CANDIDATES, prepare.CANDIDATE_COLUMNS, [  # from an abstract
            {**{c: "" for c in prepare.CANDIDATE_COLUMNS}, "patient_uid": "pubmed-18", "pmid": "18", "class": "melanoma"}])
        self.posts, self.searches, self.mode, sets = [], [], "normal", {}
        test = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, content, kind):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def do_POST(self):
                form = {k: v[0] for k, v in urllib.parse.parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode()).items()}
                ids = form["id"].split(",")
                test.posts.append(ids)
                if test.mode == "refused":
                    return self._send(b"<ePostResult><ERROR>Too many UIDs</ERROR></ePostResult>", "text/xml")
                key = str(len(sets) + 1)
                sets[key] = [i for i in ids if i != "99"]
                bad = "".join(f"<Id>{i}</Id>" for i in ids if i == "99")
                self._send(f"<ePostResult>{f'<InvalidIdList>{bad}</InvalidIdList>' if bad else ''}<QueryKey>{key}</QueryKey>"
                           f"<WebEnv>MCID_1</WebEnv></ePostResult>".encode(), "text/xml")

            def do_GET(self):
                q = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}
                test.searches.append(q)
                key = re.match(r"#(\d+) AND ", q["term"]).group(1)
                if q["term"].endswith(" AND medline[sb]"):  # 16 has headings but is outside the MEDLINE subset
                    ids = [i for i in sets[key] if INDEX[i][0] and i != "16"]
                else:
                    skin = q["term"].endswith(" AND " + prepare.SKIN)
                    headings = set(re.findall(r'"([^"]+)"\[MeSH Terms\]', q["term"].replace(prepare.SKIN, "")))
                    ids = [i for i in sets[key] if INDEX[i][0] & headings and INDEX[i][1] == skin]
                count = len(ids) + (1 if test.mode == "odd count" else 0)
                self._send(json.dumps({"esearchresult": {"count": str(count), "idlist": ids}}).encode(), "application/json")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        for patcher in (mock.patch.object(prepare, "EUTILS", f"http://127.0.0.1:{self.server.server_port}/entrez/eutils/"),
                        mock.patch.object(prepare, "EPOST_BATCH", 3), mock.patch("src.common.time.sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)


class MeshTests(MeshServer):
    def test_each_article_gets_whether_it_is_indexed_and_the_classes_its_headings_name_on_the_skin(self):
        done = prepare.mesh(self.data, classes=CLASSES)
        with (self.data / prepare.MESH_TABLE).open(encoding="utf-8", newline="") as f:
            rows = [(r["pmid"], r["indexed"], r["classes"]) for r in csv.DictReader(f)]
        self.assertEqual(rows, [("11", "yes", "melanoma"), ("12", "yes", ""), ("13", "yes", "nevus"), ("14", "yes", ""),
                                ("15", "yes", "melanoma; nevus"), ("16", "yes", "nevus"), ("17", "no", ""), ("18", "yes", "melanoma"),
                                ("99", "no", "")])
        # 12 is not on the skin; 14's heading is of a name that does not count; 17 has no headings at all
        self.assertEqual((done["unknown"], done["indexed"]), (["99"], 7))  # 16 counts as indexed: it has headings
        self.assertEqual(self.posts, [["11", "12", "13"], ["14", "15", "16"], ["17", "18", "99"]])  # in batches, in a POST
        self.assertEqual(len(self.searches), 3 * 3)
        self.assertTrue(all(q["WebEnv"] == "MCID_1" for q in self.searches))
        self.assertEqual(self.searches[0]["term"], "#1 AND medline[sb]")
        self.assertEqual(self.searches[2]["term"],
                         f'#1 AND ("Nevus"[MeSH Terms] OR "Nevus, Pigmented"[MeSH Terms]) AND {prepare.SKIN}')
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("## MeSH: the subject headings of each candidate article", "9 articles, 7 of them indexed; article "
                         "numbers PubMed does not know: 1 (99)", "| melanoma | Melanoma | 5 | 4 | 2 | 2 | 1 |",
                         "| nevus | Nevus; Nevus, Pigmented | 3 | 2 | 1 | 1 | 1 |"):
            self.assertIn(expected, text)

    def test_names_not_checked_by_the_survey_a_refusal_or_an_answer_that_does_not_fit_stop_the_step(self):
        (self.data / prepare.TERMS).write_text(TERMS + "nevus,Spitz nevus,yes,ISIC,\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not list the names of class_terms.csv as they are now: run python -m "
                                                "src.text.prepare survey again"):
            prepare.mesh(self.data, classes=CLASSES)
        (self.data / prepare.TERMS).write_text(TERMS, encoding="utf-8")
        for mode, message in (("refused", "PubMed did not take the article numbers: Too many UIDs"),
                              ("odd count", "for the indexed articles does not fit the 3 article numbers sent")):
            self.mode = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, message):
                prepare.mesh(self.data, classes=CLASSES)
            self.assertFalse((self.data / prepare.MESH_TABLE).exists())
        (self.data / prepare.ABSTRACT_CANDIDATES).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "abstract_candidates.csv not found: run python -m src.text.prepare abstracts "
                                                       "first"):
            prepare.mesh(self.data, classes=CLASSES)
        (self.data / prepare.CANDIDATES).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "case_candidates.csv not found: run python -m src.text.prepare cases first"):
            prepare.mesh(self.data, classes=CLASSES)

    def test_the_command_line_prints_the_confirmed_articles_per_class(self):
        real = prepare.mesh
        with mock.patch.object(prepare, "mesh", lambda show=None: real(self.data, classes=CLASSES, show=show)), \
                contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["mesh"])
        for expected in ("9 of 9 articles checked in PubMed", "  melanoma: 3 articles whose headings name it",
                         "  nevus: 3 articles whose headings name it", "Articles indexed by PubMed: 7 of 9; article numbers PubMed "
                         "does not know: 1", "Commit both files"):
            self.assertIn(expected, shown.getvalue())


BCC_TEXT = ("A 71-year-old woman presented with a slowly enlarging dark nodule on the left cheek that had been present for two "
            "years. Examination showed a 9 mm shiny papule with rolled borders, fine surface vessels and a small central ulcer "
            "that bled after minor trauma. Dermoscopy showed blue-grey ovoid nests and branching vessels (Fig. 1). A biopsy "
            "confirmed a pigmented basal cell carcinoma, which was excised.")
BUILD_PATIENTS = [  # made-up texts, one for each rule
    _patient("2001-1", "Pigmented basal cell carcinoma of the cheek", BCC_TEXT),
    _patient("2002-1", "Dermatofibroma of the leg", "A 40-year-old man had a dermatofibroma on the leg for many years. It was "
             "firm and brown, and it dimpled when pinched."),  # the first sentence names it: no question
    _patient("2003-1", "Squamous cell carcinoma of the scalp", "An 80-year-old farmer presented with a crusted, bleeding plaque on "
             "the bald scalp that had enlarged over four months. A BCC had been removed from his nose years ago. It measured 3 cm and was tender, with a keratotic surface and an "
             "indurated, raised edge. There was no palpable lymph node in the neck and he felt well otherwise. Keratoacanthoma was the "
             "main clinical suspicion. The excised SCC was well differentiated."),  # a name that does not count ends it too
    _patient("2004-1", "Melanoma of the arm", "A 60-year-old man noticed a changing dark spot on his arm. Histology showed a "
             "melanoma."),  # PubMed does not confirm it
    _patient("2005-1", "Congenital nevus of the back", "A 30-year-old woman came for a routine check-up of her skin. She had fair "
             "skin and many freckles since childhood, and her mother had been treated for skin cancer in the past. She worked "
             "outdoors as a gardener and rarely used sunscreen during the long summer months. Examination found a congenital "
             "nevus on the back."),  # no lesion described before the diagnosis
    _patient("2006-1", "Pyogenic granuloma in two siblings", "A 9-year-old boy had a rapidly growing red nodule on the skin of the "
             "chin for three weeks. It bled easily when touched and was 6 mm across, soft and moist, with a collarette of scale at "
             "its base. There had been a small cut at that site before. Histology showed a pyogenic granuloma.", age=9.0, sex="M"),
    _patient("2006-2", "Pyogenic granuloma in two siblings", "His 12-year-old sister had a similar red papule on the index finger "
             "that appeared two weeks after a thorn injury while gardening. The papule was 5 mm, shiny and friable, and it bled "
             "several times a day, staining her clothes. No other lesions were seen. A shave biopsy was taken. Histology showed a "
             "pyogenic granuloma.", age=12.0),  # a test result ends the question even when it names no diagnosis
    _patient("2007-1", "Basal cell carcinoma on the face of a woman", BCC_TEXT),  # the same question as 2001-1
    _patient("2008-1", "Superficial spreading melanoma of the back", "A 52-year-old man noticed that a mole on his upper back had "
             "grown and darkened over six months. He had a history of basal cell carcinoma on the nose. On examination there was an asymmetric 12 mm brown-black macule on the back with irregular "
             "borders and several colours, including blue-white areas. His wife had seen it bleed once after he scratched it. "
             "Histopathology showed a superficial spreading melanoma.", sex="M"),  # "mole" and another class's name stay
    _patient("2009-1", "Melanoma treated with nivolumab", "A 66-year-old man with a melanoma received nivolumab."),  # marked
    _patient("2010-1", "Actinic keratosis of the forehead", "A 75-year-old retired sailor presented with a rough, scaly pink patch "
             "on the forehead that had been present for a year. It was 8 mm across, slightly tender, and felt like sandpaper; there "
             "were several similar smaller spots nearby on sun-damaged skin. Clinically, an AK was diagnosed and treated with "
             "cryotherapy. A biopsy taken before the treatment confirmed an actinic keratosis."),
    _patient("2011-1", "Angiokeratoma of the thigh", "A 35-year-old man had a dark purple papule on the thigh for five years that "
             "bled after shaving and never itched or hurt. The papule measured 4 mm and had a rough, warty surface with dark red "
             "dots seen through a lens and a pale rim that was described as pyogenic. Granuloma formation was not considered. "
             "Histology confirmed an angiokeratoma."),  # "pyogenic" and "granuloma" in two sentences name its class: left out
    _patient("2012-1", "Cherry angioma of the chest", "A 45-year-old man had a red papule on the chest. Histology showed a cherry "
             "angioma."),  # too short a question
    _patient("2013-1", "Halo nevus on the back of a child", "A 10-year-old girl had a brown mole on her back that developed a white "
             "ring around it over the past year. The ring was 4 mm wide and symmetric, and the central papule was 5 mm, soft and "
             "evenly brown with a regular edge. She had vitiligo patches on both knees. Dermoscopy findings were consistent with "
             "a halo nevus. An excision biopsy confirmed a halo nevus.", age=10.0),  # "mole" is too vague to end a nevus question
]
BUILD_MESH = {  # indexed, classes: 2004's headings do not name melanoma (left out); 2010 is not indexed (title and text decide)
    "2001": ("yes", "basal cell carcinoma"), "2002": ("yes", "dermatofibroma"), "2003": ("yes", "squamous cell carcinoma"),
    "2004": ("yes", ""), "2005": ("yes", "nevus"), "2006": ("yes", "vascular lesion"), "2007": ("yes", "basal cell carcinoma"),
    "2008": ("yes", "melanoma"), "2009": ("yes", "melanoma"), "2010": ("no", ""), "2011": ("yes", "vascular lesion"),
    "2012": ("yes", "vascular lesion"), "2013": ("yes", "nevus"), "2101": ("yes", "actinic keratosis"), "2102": ("yes", "nevus")}
NEVUS_FULL = ("A 25-year-old woman had a small brown macule on her forearm that had been present since childhood and had not "
              "changed. It was 4 mm, round, evenly pigmented with a regular border, and dermoscopy showed a regular reticular network "
              "without atypical features. Excision "
              "was requested for cosmetic reasons. Histology showed a compound nevus.")
PG_CASE = ("A 17-month-old boy developed multiple eruptive red papules and nodules on the burned skin of his chest and arms two "
           "months after a scald injury, and they bled easily when touched and grew quickly over a few weeks. Several similar "
           "lesions of PG were seen on his back. Histology confirmed pyogenic granulomas.")
INSIDE_CASE = ("A 30-year-old pregnant woman had a red, bleeding nodule on the palatal gingiva that grew over two months and bled "
               "when she brushed her teeth; she had no pain, fever or weight loss. Histology confirmed a pyogenic granuloma.")
NO_PATIENT = ("Solitary firm tumours of the skin are common in adults and are usually found on the legs, where they often follow a "
              "minor injury such as an insect bite, and they dimple when pinched between two fingers. These growths are named "
              "dermatofibromas.")
NO_FEATURE = ("A 13-year-old girl came to the clinic because of lesions on her leg that had been present for two months; she had no "
              "other illness, no allergy and no family history of skin disease, and she took no medicines. The lesions were "
              "angiokeratomas.")
NODULE_ONLY = ("A 70-year-old man had a slowly growing nodule on his nose for two years; he had worked outdoors all his life as a "
               "farmer and had no other illness or family history of skin cancer. Histology confirmed a basal cell carcinoma.")
BLUE_NEVUS = ("A 40-year-old man had a slowly enlarging blue papule on the left cheek for ten years, without pain, bleeding or "
              "itching, and he had no history of trauma or of a tattoo at that site. Intraoral examination was normal. Histology "
              "showed a blue nevus.")
SHIN_CASE = ("A 60-year-old woman noticed a changing dark macule on her left shin over a year; it had grown from 4 mm to 9 mm, had "
             "two colours and an irregular border, and bled once after shaving. Her sister had been treated for skin cancer. "
             "Histology showed a melanoma.")
TWO_CASES = ("Case 1 is a 40-year-old woman with a red, scaly plaque on her left leg that had grown over two years and bled after "
             "minor trauma. Case 2 is her 42-year-old brother, with a crusted nodule on the right ear. Histology showed squamous cell "
             "carcinoma in both.")
GULLET = ("A 45-year-old man had a red, polypoid mass in the upper esophagus that bled and grew over three months. He had "
          "difficulty swallowing, lost weight during that time and had no history of skin disease. Histology showed a pyogenic "
          "granuloma.")  # a lesion, and the skin, but never in one sentence
LOOK_ALIKE = ("A 50-year-old woman had a firm brown papule on her right arm for six months that dimpled when pinched and had grown "
              "slowly to 7 mm in diameter over that time. It looked like a dermatofibroma. Histology showed mycosis fungoides.")
IHC_ONLY = ("A 30-year-old man had a red, bleeding papule on his left hand that grew quickly over three weeks after a cut while he "
            "was gardening. Immunohistochemistry (IHC) was performed. IHC was positive for CD31. He healed well. A year later the "
            "diagnosis was pyogenic granuloma.")
BY_EYE = ("A 63-year-old man had a stuck-on, warty brown plaque on his back for ten years that had slowly grown to 2 cm and itched "
          "at times, with no bleeding. The clinical and dermoscopic picture was typical of a seborrheic keratosis, which was frozen.")
HEADED = ("Case 2. A 58-year-old man had a nodule on his left cheek that had grown slowly over two years without any pain or "
          "itching. It was pink, pearly and 9 mm across. An excision was done. The margins were clear. It was a basal cell "
          "carcinoma.")  # a heading; what it looks like in the next sentence; the name two sentences after the result
WART_VIRUS = ("A 36-year-old woman had many brown, warty papules on her neck, back and chest that had appeared over four weeks "
              "and itched mildly at night. Cytology of the skin lesions showed no malignancy. A DNA test for HPV was positive. "
              "She was given a cream. Later, the papules were thought to be seborrheic keratoses by eye.")
TWO_AGES = ("We report a 40-year-old man and a 48-year-old man who presented with firm, reddish-brown plaques on a lower limb "
            "that had grown slowly over several years and dimpled when pinched. Histology showed a dermatofibroma in each.")
BY_ITS_SHORT_FORM = ("Angiokeratoma circumscriptum (AC) is a rare vascular malformation. A 19-year-old woman had an extensive AC "
                     "on her left leg, with dark red warty papules and plaques that had grown since birth and bled at times after "
                     "minor trauma. Histology showed an angiokeratoma.")
SHORTEST = ("A 52-year-old man had a firm red nodule on his left forearm that had grown slowly over three years without pain, "
            "itching or any bleeding.")  # MIN_WORDS words: the shortest question kept
AK_ABSTRACT = ("A 68-year-old fisherman presented with a rough, red, scaly patch on the back of the left hand that had persisted "
               "for eight months despite moisturisers. It was 1 cm across, flat, and tender when rubbed, and similar smaller rough "
               "spots were felt nearby. A shave biopsy was taken. Histology showed an actinic keratosis.")


class BuildData(HuggingFaceServer):
    """The earlier steps' tables and downloads for the build (and the audit, tests/text/test_audit.py): one made-up case for
    each rule."""
    CLASSES = CasesTests.CLASSES

    def setUp(self):
        super().setUp()
        self.content = json.dumps(BUILD_PATIENTS, indent=1).encode("utf-8")
        prepare.cases(self.data, classes=self.CLASSES)
        title = "Actinic keratosis on the hand of a fisherman"
        row, _ = prepare._judge(prepare.read_terms(self.data / prepare.TERMS, self.CLASSES), self.CLASSES)(title, AK_ABSTRACT)
        judge = prepare._judge(prepare.read_terms(self.data / prepare.TERMS, self.CLASSES), self.CLASSES)
        nevus = judge("Compound nevus of the forearm", NEVUS_FULL)[0]
        prepare._write_csv(self.data / prepare.ABSTRACT_CANDIDATES, prepare.CANDIDATE_COLUMNS, [
            {"patient_uid": "pubmed-2101", "pmid": "2101", "year": "2016", **row, "age": "", "sex": "", "words": 50},
            {"patient_uid": "pubmed-2102", "pmid": "2102", "year": "2025", **nevus, "age": "", "sex": "", "words": 47}])
        (self.data / "sources" / "pubmed").mkdir(parents=True)
        (self.data / "sources" / "pubmed" / "abstracts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (
            {"pmid": "2101", "pmcid": "", "year": "2016", "title": title, "fulltext": "", "licence": "",
             "sections": [["BACKGROUND", "Actinic keratosis is common."], ["CASE PRESENTATION", AK_ABSTRACT]]},
            {"pmid": "2102", "pmcid": "PMC2102", "year": "2025", "title": "Compound nevus of the forearm", "fulltext": NEVUS_FULL,
             "licence": "https://creativecommons.org/licenses/by/4.0/", "sections": [["", "A compound nevus."]]})), encoding="utf-8")
        self._mesh(BUILD_MESH)

    def _mesh(self, table):
        prepare._write_csv(self.data / prepare.MESH_TABLE, prepare.MESH_COLUMNS,
                           [{"pmid": p, "indexed": i, "classes": c} for p, (i, c) in table.items()])

    def _table(self):
        with (self.data / "dataset.csv").open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, prepare.DATASET_COLUMNS)
            return list(reader)


class BuildTests(BuildData):
    def test_only_cases_three_sources_agree_on_become_questions_cut_word_for_word_before_the_diagnosis(self):
        prepare.build(self.data, classes=self.CLASSES)
        rows = self._table()
        self.assertEqual([(r["split"], r["label"], r["patient_uid"]) for r in rows], [  # 60% of each class into test, rounded
            ("test", "actinic keratosis", "pubmed-2101"), ("test", "basal cell carcinoma", "2001-1"), ("test", "melanoma", "2008-1"),
            ("test", "nevus", "2013-1"), ("test", "squamous cell carcinoma", "2003-1"), ("train", "actinic keratosis", "2010-1"),
            ("train", "nevus", "pubmed-2102"),
            ("train", "vascular lesion", "2006-1"), ("train", "vascular lesion", "2006-2")])  # one article of 2: fits no target
        files = {r["patient_uid"]: (self.data / "dataset" / r["path"]).read_bytes() for r in rows}
        self.assertEqual(files["2001-1"].decode("utf-8"), BCC_TEXT.split(" A biopsy")[0])  # the figure's "Fig. 1" ends no sentence
        self.assertTrue(files["2008-1"].decode("utf-8").endswith("after he scratched it."))
        self.assertIn("a history of basal cell carcinoma", files["2008-1"].decode("utf-8"))  # another class's name stays
        self.assertTrue(files["2010-1"].decode("utf-8").endswith("on sun-damaged skin."))
        self.assertTrue(files["2006-2"].decode("utf-8").endswith("No other lesions were seen."))
        self.assertTrue(files["2003-1"].decode("utf-8").endswith("he felt well otherwise."))
        sources = {**{p["patient_uid"]: p["patient"] for p in BUILD_PATIENTS}, "pubmed-2101": AK_ABSTRACT, "pubmed-2102": NEVUS_FULL}
        for row in rows:
            question = files[row["patient_uid"]].decode("utf-8")
            self.assertTrue(sources[row["patient_uid"]].startswith(question))
            self.assertEqual(row["sha256"], hashlib.sha256(files[row["patient_uid"]]).hexdigest())
            self.assertEqual(row["words"], str(len(question.split())))
            self.assertEqual(row["path"], f"{row['split']}/{row['label']}/{row['patient_uid']}.txt")
        first = next(r for r in rows if r["patient_uid"] == "2001-1")
        self.assertEqual((first["pmid"], first["year"], first["age"], first["sex"]), ("2001", "2019", "70 year", "F"))
        self.assertEqual({r["patient_uid"]: r["pubmed"] for r in rows if r["pubmed"] != "agrees"}, {"2010-1": "not indexed"})
        self.assertEqual({r["patient_uid"]: r["source"] for r in rows if r["source"] != "PMC-Patients"},
                         {"pubmed-2101": "PubMed abstract", "pubmed-2102": "PMC full text"})
        self.assertEqual(files["pubmed-2102"].decode("utf-8"), NEVUS_FULL.split(" Excision")[0])  # the full text, not the abstract
        self.assertEqual(files["pubmed-2101"].decode("utf-8"), AK_ABSTRACT.split(" A shave")[0])  # the case part, cut

    def test_the_report_counts_every_rule_per_class_and_the_checks(self):
        prepare.build(self.data, classes=self.CLASSES)
        text = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("## Build: the text dataset", "| actinic keratosis | 2 | 2 | 0 | 2 | 2 | 0 | 1 | 1 | 1 | 1 | 0 | 1 |",
                         "| basal cell carcinoma | 2 | 2 | 0 | 2 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |",
                         "| dermatofibroma | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – |",
                         "| melanoma | 3 | 2 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |",
                         "| nevus | 3 | 3 | 0 | 2 | 2 | 1 | 0 | 2 | 0 | 1 | 0 | 1 |",
                         "| vascular lesion | 4 | 4 | 0 | 2 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 2 |",
                         "| all | 16 | 15 | 1 | 10 | 9 | 1 | 1 | 8 | 1 | 5 | 0 | 4 |", "across two sentences (left out): 1",
                         "60% of its cases into test (at most 40), 20% into validation (at most 15), the rest into train", "identical to an earlier one (left out): 1",
                         "Articles in more than one split: 0", f"SHA-256 `{hashlib.sha256(self.content).hexdigest()}` (checked)",
                         "No person has checked the labels"):
            self.assertIn(expected, text)

    def test_splits_keep_each_article_whole_within_the_caps_and_a_second_run_gives_the_same_files(self):
        with mock.patch.object(prepare, "TEST_PER_CLASS", 1), mock.patch.object(prepare, "VAL_PER_CLASS", 1):
            prepare.build(self.data, classes=self.CLASSES)
            first = (self.data / "dataset.csv").read_bytes()
            stale = self.data / "dataset" / "test" / "nevus" / "old.txt"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text("left from an earlier run", encoding="utf-8")
            prepare.build(self.data, classes=self.CLASSES)
        self.assertEqual((self.data / "dataset.csv").read_bytes(), first)
        self.assertFalse(stale.exists())
        rows = self._table()
        self.assertEqual({r["patient_uid"]: r["split"] for r in rows if r["label"] == "vascular lesion"},
                         {"2006-1": "train", "2006-2": "train"})  # two cases of one article fit neither cap of 1
        self.assertEqual(sorted(p.relative_to(self.data / "dataset").as_posix() for p in (self.data / "dataset").rglob("*.txt")),
                         sorted(r["path"] for r in rows))
        for split in ("test", "val"):
            self.assertLessEqual(max(Counter(r["label"] for r in rows if r["split"] == split).values(), default=0), 1)

    def test_a_missing_step_stops_the_build_and_names_the_command_to_run(self):
        self._mesh({"2001": ("yes", "")})
        with self.assertRaisesRegex(ValueError, "article_mesh.csv lacks 14 candidate article.s., for example 2002: run python -m "
                                                "src.text.prepare mesh again"):
            prepare.build(self.data, classes=self.CLASSES)
        (self.data / prepare.MESH_TABLE).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "article_mesh.csv not found: run python -m src.text.prepare mesh first"):
            prepare.build(self.data, classes=self.CLASSES)
        self._mesh(BUILD_MESH)
        (self.data / "sources" / "pubmed" / "abstracts.jsonl").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "abstracts.jsonl not found: run python -m src.text.prepare abstracts first"):
            prepare.build(self.data, classes=self.CLASSES)
        (self.folder / "PMC-Patients-V2.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "PMC-Patients-V2.json not found: run python -m src.text.prepare cases first"):
            prepare.build(self.data, classes=self.CLASSES)
        self.assertFalse((self.data / "dataset.csv").exists())

    def test_the_command_line_prints_the_cases_per_class_and_what_to_commit(self):
        real = prepare.build
        with mock.patch.object(prepare, "build", lambda show=None: real(self.data, classes=self.CLASSES, show=show)), \
                contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["build"])
        for expected in ("  vascular lesion: 2 cases (test 0, validation 0, train 2)", "  dermatofibroma: 0 cases (test 0,",
                         "the question texts in ", "(not in Git)", "Commit "):
            self.assertIn(expected, shown.getvalue())

    def test_each_class_gives_60_percent_to_test_and_20_to_validation_within_the_caps_keeping_articles_whole(self):
        self.assertEqual({n: tuple(prepare._targets(n).values()) for n in (1, 2, 3, 9, 30, 35, 44, 101)},
                         {1: (1, 0), 2: (1, 0), 3: (2, 1), 9: (5, 2), 30: (18, 6), 35: (21, 7), 44: (26, 9), 101: (40, 15)})
        rows = [{"label": "small", "pmid": f"s{i}"} for i in range(10)] + [{"label": "large", "pmid": f"l{i}"} for i in range(100)] + \
               [{"label": "mixed", "pmid": "m0"} for _ in range(3)] + [{"label": "mixed", "pmid": f"m{i}"} for i in (1, 2)]
        prepare._split(rows)
        counts = Counter((r["label"], r["split"]) for r in rows)
        self.assertEqual([counts[("small", s)] for s in ("test", "val", "train")], [6, 2, 2])
        self.assertEqual([counts[("large", s)] for s in ("test", "val", "train")], [40, 15, 45])
        self.assertEqual(len({r["split"] for r in rows if r["pmid"] == "m0"}), 1)  # an article's cases stay together
        self.assertLessEqual(counts[("mixed", "test")], 3)
        self.assertLessEqual(counts[("mixed", "val")], 1)

    def _add_abstract(self, pmid, title, sections):
        """Adds a PubMed record not indexed by PubMed, whose title and case text pass the rules of candidates, as abstracts would."""
        judge = prepare._judge(prepare.read_terms(self.data / prepare.TERMS, self.CLASSES), self.CLASSES)
        row, _ = judge(title, prepare._case_part(sections))
        self.assertEqual(row["marks"], "", title)
        rows = prepare._read_csv(self.data / prepare.ABSTRACT_CANDIDATES, prepare.CANDIDATE_COLUMNS, "abstracts")
        prepare._write_csv(self.data / prepare.ABSTRACT_CANDIDATES, prepare.CANDIDATE_COLUMNS, rows + [
            {"patient_uid": f"pubmed-{pmid}", "pmid": pmid, "year": "2024", **row, "age": "", "sex": "", "words": 50}])
        with (self.data / "sources" / "pubmed" / "abstracts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"pmid": pmid, "pmcid": "", "year": "2024", "title": title, "fulltext": "", "licence": "",
                                "sections": sections}) + "\n")
        table = {r["pmid"]: (r["indexed"], r["classes"]) for r in prepare._read_csv(self.data / prepare.MESH_TABLE,
                                                                                     prepare.MESH_COLUMNS, "mesh")}
        self._mesh({**table, pmid: ("no", "")})

    def _question_of(self, uid):
        row = next((r for r in self._table() if r["patient_uid"] == uid), None)
        return None if row is None else (self.data / "dataset" / row["path"]).read_text(encoding="utf-8")

    def test_a_short_form_the_abstract_defines_outside_its_case_part_ends_the_question(self):
        self._add_abstract("2103", "Eruptive pyogenic granuloma on burned skin",
                           [["BACKGROUND", "Pyogenic granuloma (PG) is a common benign vascular tumour of the skin."],
                            ["CASE PRESENTATION", PG_CASE]])
        prepare.build(self.data, classes=self.CLASSES)
        self.assertEqual(self._question_of("pubmed-2103"), PG_CASE.split(" Several")[0])  # not "PG"

    def test_a_question_without_a_patient_or_with_a_lesion_inside_the_body_is_left_out_and_counted(self):
        self._add_abstract("2104", "Pyogenic granuloma: a case report", [["", INSIDE_CASE]])
        self._add_abstract("2105", "Dermatofibroma: a review of a common growth", [["", NO_PATIENT]])
        self._add_abstract("2106", "Blue nevus of the cheek", [["", BLUE_NEVUS]])
        self._add_abstract("2109", "Solitary angiokeratoma in a girl", [["", NO_FEATURE]])  # says nothing of how they look
        self._add_abstract("2110", "Basal cell carcinoma of the nose", [["", NODULE_ONLY]])  # "a nodule" tells what it looks like
        prepare.build(self.data, classes=self.CLASSES)
        self.assertEqual(self._question_of("pubmed-2110"), NODULE_ONLY.split(" Histology")[0])
        self.assertEqual([self._question_of(f"pubmed-{n}") for n in ("2104", "2105", "2109")], [None, None, None])
        self.assertTrue(self._question_of("pubmed-2106").endswith("Intraoral examination was normal."))  # no lesion in it
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("| dermatofibroma | 1 | 0 | 0 | 0 | 0 | 0 | 0 |", "| vascular lesion | 0 | 0 | 1 | 0 | 1 | 0 | 0 |",
                         "| nevus | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"):
            self.assertIn(expected, report)

    def test_a_question_of_min_words_is_kept_and_one_word_fewer_is_left_out(self):
        self.assertEqual(len(SHORTEST.split()), prepare.MIN_WORDS)
        self._add_abstract("2111", "Dermatofibroma of the forearm", [["", f"{SHORTEST} Histology showed a dermatofibroma."]])
        self._add_abstract("2112", "Dermatofibroma of the arm", [["", f"{SHORTEST.replace(' left', '')} Histology showed a "
                                                                          "dermatofibroma."]])
        prepare.build(self.data, classes=self.CLASSES)
        self.assertEqual(self._question_of("pubmed-2111"), SHORTEST)
        self.assertIsNone(self._question_of("pubmed-2112"))

    def test_a_case_heading_a_look_in_the_next_sentence_and_a_name_two_sentences_after_the_result_are_kept(self):
        self._add_abstract("2118", "Basal cell carcinoma of the cheek", [["", HEADED]])
        self._add_abstract("2119", "Seborrheic keratosis and human papillomavirus (HPV) infection after fingolimod",
                           [["", WART_VIRUS]])  # the title's "(HPV)" is no short form of the class: no result names it
        self._add_abstract("2120", "Angiokeratoma circumscriptum of the leg", [["", BY_ITS_SHORT_FORM]])  # its question would
        # use "AC", the article's short form of a name of its class
        prepare.build(self.data, classes=self.CLASSES)
        self.assertEqual(self._question_of("pubmed-2118"), HEADED.split(" An excision")[0])
        self.assertIsNone(self._question_of("pubmed-2119"))
        self.assertIsNone(self._question_of("pubmed-2120"))
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        self.assertIn("| pigmented benign keratosis | 0 | 0 | 0 | 0 | 0 | 1 | 0 |", report)

    def test_the_title_rules_as_they_are_now_and_two_patients_ages_leave_a_case_out(self):
        self._add_abstract("2121", "Basal cell carcinoma of the cheek", [["", HEADED]])
        path = self.data / prepare.ABSTRACT_CANDIDATES  # listed before the rule on another lesion: no mark
        rows = prepare._read_csv(path, prepare.CANDIDATE_COLUMNS, "abstracts")
        for row in rows:
            if row["pmid"] == "2121":
                row["title"] = "Basal cell carcinoma occurring with a desmoplastic trichilemmoma"
        prepare._write_csv(path, prepare.CANDIDATE_COLUMNS, rows)
        self._add_abstract("2123", "Basal cell carcinoma of the face", [["", HEADED.replace("58-year-old", "61-year-old")]])
        rows = prepare._read_csv(path, prepare.CANDIDATE_COLUMNS, "abstracts")
        for row in rows:
            if row["pmid"] == "2123":
                row["title"] = "Basal cell nevus syndrome in a man"  # as if a name had been added since: it names no class now
        prepare._write_csv(path, prepare.CANDIDATE_COLUMNS, rows)
        self._add_abstract("2122", "Dermatofibroma of the leg", [["", TWO_AGES]])
        prepare.build(self.data, classes=self.CLASSES)
        self.assertEqual([self._question_of(f"pubmed-{n}") for n in ("2121", "2122", "2123")], [None] * 3)
        self.assertIn("| dermatofibroma | 0 | 1 | 0 | 0 | 0 | 0 | 0 |", (self.data / prepare.REPORT).read_text(encoding="utf-8"))

    def test_two_patients_a_lesion_not_on_the_skin_and_a_diagnosis_no_test_result_confirms_are_left_out_and_counted(self):
        self._add_abstract("2113", "Squamous cell carcinoma of the leg and the ear in two siblings", [["", TWO_CASES]])
        self._add_abstract("2114", "Pyogenic granuloma: an unusual case", [["", GULLET]])
        self._add_abstract("2115", "A dermatofibroma that was not one", [["", LOOK_ALIKE]])  # the result names another diagnosis
        self._add_abstract("2116", "Seborrheic keratosis of the back", [["", BY_EYE]])  # diagnosed by eye and by dermoscopy only
        self._add_abstract("2117", "Pyogenic granuloma of the hand", [["", IHC_ONLY]])  # a test's short form is not the class's
        prepare.build(self.data, classes=self.CLASSES)
        self.assertEqual([self._question_of(f"pubmed-{n}") for n in ("2113", "2114", "2115", "2116", "2117")], [None] * 5)
        report = (self.data / prepare.REPORT).read_text(encoding="utf-8")
        for expected in ("| squamous cell carcinoma | 0 | 1 | 0 | 0 | 0 | 0 | 0 |", "| vascular lesion | 0 | 0 | 0 | 1 | 0 | 1 | 0 |",
                         "| dermatofibroma | 0 | 0 | 0 | 0 | 0 | 1 | 0 |", "| pigmented benign keratosis | 0 | 0 | 0 | 0 | 0 | 1 | 0 |"):
            self.assertIn(expected, report)

    def test_a_result_confirms_in_its_own_sentence_or_the_next_and_only_by_a_name_that_counts(self):
        terms = prepare.read_terms(ROOT / "data" / "text" / prepare.TERMS, CasesTests.CLASSES)
        names = prepare._says(prepare._namer(terms, "squamous cell carcinoma"))
        for rest, confirmed in {" A biopsy was taken. Histology showed a squamous cell carcinoma.": True,
                                " Histology confirmed an SCC.": True,  # the class's short form
                                " A punch biopsy was taken. It showed a squamous cell carcinoma.": True,  # the next sentence names it
                                " Histology showed a keratoacanthoma.": False,  # a name that does not count: another diagnosis
                                " A squamous cell carcinoma was suspected. It was frozen.": False,  # no test result
                                " A biopsy was taken. It healed well. A squamous cell carcinoma was found.": True,  # two after
                                " A biopsy was taken. It healed. It was not sent. A squamous cell carcinoma was found.": False,
                                " A biopsy was taken. His father had a squamous cell carcinoma.": False,  # a relative's
                                " He was at risk of keratinocyte carcinoma. It was an SCC.": False,  # a group of diseases
                                " Histology showed atypical keratinocytes. It was an SCC.": True}.items():
            self.assertEqual(prepare._confirmed(rest, names), confirmed, rest)
        placed = {w for word in prepare.LESION_WORDS + prepare.PLACED for w in prepare._tokens(word)}
        where = {tuple(prepare._tokens(word)): word for word in prepare.SKIN_WORDS + prepare.SURFACE}
        looks = {tuple(prepare._tokens(word)): word for word in prepare.SKIN_WORDS + prepare.SEEN + prepare.FEATURES}
        for text, on_skin in {"A man had brown moles on both feet.": True, "A red nodule on the nasal tip.": True,
                              "A man had a brown mole. It was on both feet and grew.": False,  # what and where: not in one sentence
                              "A chest CT showed a small nodule in the right lung.": False,  # nothing seen or felt
                              "A pedunculated red tumor filled the nasal cavity.": False,  # "nasal" alone: not the skin
                              "A nodule grew on the left cheek. It was red and firm.": True,  # what it looks like: next sentence
                              "It was red and firm. A nodule grew on the left cheek.": True,  # or the sentence before
                              "A nodule grew on the left cheek. It was 5 mm. It was red.": False,  # but not two sentences away
                              "She had skin cancer. A nodule grew on her chest.": False,  # "skin" next to it: another matter
                              "A port-wine stain on the left cheek was purple.": True, "A wound on the left shin was red.": True,
                              "A red papule on the prepuce.": True,
                              "Abdominal CT showed a mass in the bile duct. It was cystic.": False,  # a scan's: not the skin
                              "A CT of the abdomen showed a polypoid lesion.": False}.items():
            self.assertEqual(prepare._on_skin(text, placed, where, looks), on_skin, text)
        text = "Immunohistochemistry (IHC) was done. Pyogenic granuloma (PG) is common."
        self.assertEqual(prepare._defined_short_forms(text, prepare._namer(terms, "vascular lesion"), last=True), ["PG"])  # confirms
        self.assertEqual(prepare._defined_short_forms(text, prepare._revealer(terms, "vascular lesion")), ["IHC", "PG"])  # ends a question
        keratosis, carcinoma = prepare._namer(terms, "pigmented benign keratosis"), prepare._namer(terms, "basal cell carcinoma")
        title = "Development of Seborrheic Keratosis and Human Papillomavirus (HPV) Infection"
        self.assertEqual(prepare._defined_short_forms(title, keratosis, last=True), [])  # the name does not end right before
        self.assertEqual(prepare._defined_short_forms(title, prepare._revealer(terms, "pigmented benign keratosis")), ["HPV"])  # a
        # question stops at it all the same: to be safe
        self.assertEqual(prepare._defined_short_forms("A nodular basal cell carcinoma (nBCC) and a nodular BCC (nBCCs).", carcinoma,
                                                      last=True), ["nBCC"])  # a name or the class's short form, right before
        self.assertEqual(prepare._defined_short_forms("A man had a nodular BCC (nBCC).", carcinoma, last=True), ["nBCC"])
        self.assertEqual(prepare._defined_short_forms("A BCC or not (NOT).", carcinoma, last=True), [])
        self.assertEqual(prepare._defined_short_forms("A BCC or not (NOT).", carcinoma), ["NOT"])
        self.assertEqual(prepare._defined_short_forms("Nevoid basal cell carcinoma syndrome (NBCCS) is rare.", carcinoma, last=True),
                         [])  # a syndrome, not the class
        vascular = prepare._revealer(terms, "vascular lesion")
        self.assertEqual(prepare._defined_short_forms("Angiokeratoma circumscriptum (AC) is rare.", vascular), ["AC"])  # the class
        self.assertEqual(prepare._defined_short_forms("Angiokeratoma circumscriptum (AC) is rare.", prepare._namer(
            terms, "vascular lesion"), last=True), [])
        self.assertEqual(prepare._defined_short_forms("Histopathological examination (HPE) was done.",
                                                      prepare._revealer(terms, "basal cell carcinoma")), ["HPE"])  # a test's
        for text, many in {"Two patients were seen.": True, "The mean age was 54 years.": True, "Case 2 is a man.": True,
                           "The patient's mother had a mole.": False, "A patient with a case of acne.": False}.items():
            self.assertEqual(bool(prepare.MANY.search(text)), many, text)
        for text, many in {"Case 2. A 40-year-old man.": False, "Patient 3: a woman.": False, "CASE II - A boy.": False,
                           "Case 1 is a woman; case 2, her son.": True, "A man. Case 2 is a woman.": True}.items():
            self.assertEqual(bool(prepare.MANY.search(prepare.HEADING.sub("", text, count=1))), many, text)  # a heading at the start
        self.assertEqual(prepare.HEADING.sub("", "Patient is a man.", count=1), "Patient is a man.")  # not a heading
        for text, ages in {"We report a 40-year-old man and a 48-year-old man.": {40, 48},
                           "One lesion was of a 53-year-old woman. The other, of a 20-year-old man.": {20, 53},
                           "A 45-year old African American woman; her 8-year-old daughter.": {45},  # a relative's: not a patient
                           "A 61 years old female. The 61-year-old female had a mole.": {61}}.items():
            self.assertEqual(prepare._ages(text), ages, text)

    def test_the_same_patient_in_two_articles_is_kept_once_from_the_article_pubmed_listed_first(self):
        self._add_abstract("2108", "A changing melanoma on the leg", [["", SHIN_CASE]])
        self._add_abstract("2107", "Melanoma of the shin in a woman", [["", "We describe a " + SHIN_CASE[2:].replace(
            " Her sister had been treated for skin cancer.", "")]])  # told again, a little differently
        prepare.build(self.data, classes=self.CLASSES)
        self.assertIsNotNone(self._question_of("pubmed-2107"))  # the lower PubMed number keeps the case
        self.assertIsNone(self._question_of("pubmed-2108"))
        self.assertIsNotNone(self._question_of("2008-1"))  # another melanoma, not the same patient
        self.assertIn("| melanoma | 0 | 0 | 0 | 0 | 0 | 0 | 1 |", (self.data / prepare.REPORT).read_text(encoding="utf-8"))

    def test_a_question_sharing_10_percent_of_its_runs_with_an_earlier_article_is_the_same_patient(self):
        case = SHIN_CASE.split(" Histology")[0]
        other = ("A 30-year-old man had a slowly growing firm red nodule on the right forearm for two years that itched at times "
                 "and bled once after a knock, with no change in its colour; no lymph node could be felt in the armpit.")
        words = case.split()
        rows = [{"pmid": "7", "patient_uid": "7-1", "question": case},
                {"pmid": "7", "patient_uid": "7-2", "question": f"{case} It was painless."},  # one article: never compared
                {"pmid": "9", "patient_uid": "9-1", "question": f"{' '.join(words[:16])} {other}"}]  # shares about a fifth
        self.assertEqual([r["patient_uid"] for r in prepare._same_patients(rows)], ["9-1"])  # the later article leaves

    def test_sentences_end_at_a_stop_before_a_capital_but_not_after_a_short_form(self):
        text = "A 3.5 cm nodule (Fig. 1) was seen, e.g. On the arm. Smith et al. Reported it. Was it new? Yes [2]. (A) note."
        starts = prepare._sentence_starts(text)
        self.assertEqual([text[a:b].strip() for a, b in zip(starts, starts[1:] + [len(text)])],
                         ["A 3.5 cm nodule (Fig. 1) was seen, e.g. On the arm.", "Smith et al. Reported it.", "Was it new?",
                          "Yes [2].", "(A) note."])


class QuestionTests(unittest.TestCase):
    """Where a question starts and ends, with the committed names."""

    def setUp(self):
        terms = prepare.read_terms(ROOT / "data" / "text" / prepare.TERMS, CasesTests.CLASSES)
        self.reveals = {label: prepare._revealer(terms, label) for label in CasesTests.CLASSES}

    def test_a_background_that_names_the_class_is_left_out_and_the_question_starts_at_the_patient(self):
        says = prepare._says(self.reveals["basal cell carcinoma"])
        patient = ("A 72-year-old man presented with a slowly growing pearly nodule with rolled borders on the left ala. Dermoscopy "
                   "showed arborizing vessels.")
        self.assertEqual(prepare._question("Basal cell carcinoma is the most common skin cancer. We describe an unusual case. "
                                           f"{patient} A biopsy confirmed a nodular BCC.", says), patient)
        self.assertEqual(prepare._question(f"Skin cancer is common in fair skin. {patient} A biopsy confirmed it.", says),
                         f"Skin cancer is common in fair skin. {patient}")  # a background that does not name it stays
        self.assertEqual(prepare._question("A 72-year-old man had a basal cell carcinoma on the nose. It was 5 mm.", says), "")

    def test_a_short_form_the_article_defines_names_the_class_as_written(self):
        vascular, keratosis = self.reveals["vascular lesion"], self.reveals["pigmented benign keratosis"]
        whole = f"Pyogenic granuloma (PG) is a common benign vascular tumour. {PG_CASE}"
        forms = prepare._defined_short_forms(whole, vascular)
        self.assertEqual(forms, ["PG"])
        self.assertEqual(prepare._question(PG_CASE, prepare._says(vascular, forms)), PG_CASE.split(" Several")[0])
        self.assertEqual(prepare._defined_short_forms("Seborrheic keratoses (SKs) and SK (seborrheic keratosis), not shown "
                                                      "(Fig. 2) nor on an MRI (MRI).", keratosis), ["SK"])
        self.assertEqual(prepare._defined_short_forms("PG (pyogenic granuloma) is common.", vascular), ["PG"])  # the other way
        says = prepare._says(vascular, ["PG"])
        self.assertTrue(says("Two PGs were seen."))
        self.assertFalse(says("The level was 5 pg per ml."))  # capitals and all: not another word

    def test_what_a_microscope_showed_ends_the_question_but_a_stained_shirt_does_not(self):
        says = prepare._says(self.reveals["basal cell carcinoma"])
        start = "A 48-year-old man had a firm nodule on the left cheek for a year, and it bled after shaving."
        for result in ("It showed cells with eccentric nuclei and abundant eosinophilic cytoplasm.", "Mitoses were frequent.",
                       "Microscopy showed nests of cells.", "The specimen was sent away.", "Dermatopathology was reviewed.",
                       "It was composed of fibroblastic and histiocytic cells and multinucleated giant cells.",
                       "Fine-needle aspiration showed a few cells.", "Thick collagen bundles were seen.",
                       "The upper dermis was thickened."):
            self.assertEqual(prepare._question(f"{start} {result}", says), start, result)
        self.assertEqual(prepare._question(f"{start} Blood was staining his shirt.", says), f"{start} Blood was staining his shirt.")

    def test_a_sentence_putting_a_lesion_inside_the_body_and_a_patient_described_by_an_age_or_a_sex(self):
        lesion = {w for word in prepare.LESION_WORDS for w in prepare._tokens(word)}
        inside = {tuple(prepare._tokens(site)): site for site in prepare.INSIDE}
        for text, found in (("She had a mushroom-shaped pigmented choroidal lesion.", True),
                            ("CT showed a large mass in the superior mediastinum.", True),
                            ("A dark mass was seen behind the tympanic membrane.", True),
                            ("A red papule on the cheek. Ophthalmic examination of the retina was normal.", False),
                            ("A black macule on the shin. Intraoral examination was normal.", False)):
            self.assertEqual(prepare._inside(text, lesion, inside), found, text)
        for text, found in (("A 55 yr old man had a nodule.", True), ("A newborn had many brown patches.", True),
                            ("This suggests antigen loss by this patient's tumor over six years.", False),
                            ("The patient had a nodule.", False)):
            self.assertEqual(bool(prepare.PATIENT.search(text)), found, text)

    def test_a_licence_is_read_from_its_link_its_text_or_its_type(self):
        def licence(xml):
            return prepare._licence(ElementTree.fromstring(xml))
        self.assertEqual(licence('<license xmlns:ali="http://www.niso.org/schemas/ali/1.0/"><ali:license_ref>https://creativecommons'
                                 '.org/licenses/by/4.0/</ali:license_ref><license-p>Open.</license-p></license>'),
                         "https://creativecommons.org/licenses/by/4.0/")  # as PMC gives it
        self.assertEqual(licence('<license license-type="open-access"><license-p>Free to read.</license-p></license>'), "open-access")
        self.assertEqual(prepare._licence(None), "")


class TermsTests(unittest.TestCase):
    """class_terms.csv is written by people: it must fit the image dataset's classes and give each name one class at most."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / prepare.TERMS

    def _read(self, text):
        self.path.write_text(text, encoding="utf-8")
        return prepare.read_terms(self.path, CLASSES)

    def test_a_list_that_would_need_a_guess_stops_with_the_line(self):
        head = "class,term,counts,source,note\n"
        good = head + "melanoma,melanoma,yes,ISIC,\nnevus,nevus,yes,ISIC,\n"
        self.assertEqual(len(self._read(good)), 2)
        for text, message in (
                ("class,term,counts\n", "must have the columns"),
                (good + "lentigo,lentigo,yes,ISIC,\n", "line 4: 'lentigo' is not a class of the image dataset"),
                (good + "melanoma,mole,maybe,ISIC,\n", "line 4: every row needs a term, counts .yes or no. and a source"),
                (good + "melanoma,,yes,ISIC,\n", "every row needs a term"),
                (good + "melanoma,Melanomas,yes,ISIC,\n", "line 4: 'Melanomas' is listed twice for melanoma"),
                (good + "melanoma,mole,yes,ISIC,\nnevus,moles,yes,ISIC,\n", "line 5: 'moles' counts for both melanoma and nevus"),
                (head + "melanoma,melanoma,yes,ISIC,\nnevus,mole,yes,ISIC,\n", "lacks the class name itself, counting yes, for: nevus")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self._read(text)
        self.assertEqual(len(self._read(good + "melanoma,lentigo maligna,yes,ISIC,\nnevus,lentigo maligna,no,ISIC,\n")), 4)  # no is fine
        self.path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "the list of names is part of the repository"):
            prepare.read_terms(self.path, CLASSES)

    def test_names_compare_without_regard_to_spelling_word_order_plurals_or_punctuation(self):
        for a, b in (("Keratoses, Actinic", "actinic keratosis"), ("Nevi, Pigmented", "pigmented naevus"), ("Lentigines", "lentigo"),
                     ("Bowen's Disease", "Bowen disease"), ("seborrhoeic keratosis", "Keratosis, Seborrheic"),
                     ("lobular capillary haemangioma", "Hemangioma, Capillary, Lobular")):
            self.assertEqual(prepare._words(a), prepare._words(b), (a, b))
        self.assertNotEqual(prepare._words("lentigo maligna"), prepare._words("lentigo"))

    @unittest.skipUnless((ROOT / "data" / "text" / prepare.TERMS).is_file() and (ROOT / prepare.IMAGE_TABLE).is_file(),
                         "the committed lists are not there")
    def test_the_committed_list_fits_the_committed_image_classes(self):
        classes = prepare.image_classes(prepare.read_table(ROOT / prepare.IMAGE_TABLE))
        rows = prepare.read_terms(ROOT / "data" / "text" / prepare.TERMS, classes)
        self.assertEqual(sorted({r["class"] for r in rows}), classes)

    def test_every_command_in_the_usage_documentation_is_accepted(self):
        usage = re.findall(r"^\s+python -m src\.text\.prepare (.+?)$", prepare.__doc__, re.M)
        self.assertGreaterEqual(len(usage), 1)
        for line in usage:
            prepare.build_parser().parse_args(shlex.split(line))


if __name__ == "__main__":
    unittest.main()
