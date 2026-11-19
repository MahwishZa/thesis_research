"""The knowledge index (src/build/knowledge/index.py), on a tiny invented text dataset and library: no network, no real data."""

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.build.knowledge import index
from src.build.published_cases.text import CLASSES


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class Project(unittest.TestCase):
    """data/published_cases/ with dataset.csv, build/class_terms.csv, build/term_check.csv and the download with each full text's licence, and
    data/knowledge/library.csv, as the text dataset and the library leave them; each test runs inside it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.text, self.knowledge = self.base / "data" / "published_cases", self.base / "data" / "knowledge"
        write_table(self.text / "dataset.csv", [
            {"split": split, "label": label, "path": f"{split}/{label}/{uid}.txt", "sha256": uid[-1] * 64, "patient_uid": uid, "pmid": pmid,
             "source": source}
            for split, label, uid, pmid, source in (
                ("train", "melanoma", "pubmed-11", "11", "PubMed abstract"), ("test", "melanoma", "12-1", "12", "PMC-Patients"),
                ("train", "melanoma", "pubmed-16", "16", "PMC full text"),
                ("train", "nevus", "13-1", "13", "PMC-Patients"), ("val", "nevus", "pubmed-14", "14", "PMC full text"),
                ("train", "basal cell carcinoma", "15-1", "15", "PMC-Patients"))])
        write_table(self.text / "build" / "class_terms.csv", [
            {"class": "melanoma", "term": "melanoma", "counts": "yes"}, {"class": "melanoma", "term": "melanoma in situ", "counts": "yes"},
            {"class": "nevus", "term": "nevus", "counts": "yes"}, {"class": "nevus", "term": "nevoid", "counts": "no"},
            {"class": "basal cell carcinoma", "term": "basal cell carcinoma", "counts": "yes"}])
        write_table(self.text / "build" / "term_check.csv", [
            {"class": "melanoma", "counts": "yes", "mesh_headings": "Melanoma; Hutchinson's Melanotic Freckle"},
            {"class": "nevus", "counts": "yes", "mesh_headings": "Nevus"},
            {"class": "basal cell carcinoma", "counts": "yes", "mesh_headings": "Carcinoma, Basal Cell"}])
        write_table(self.knowledge / "library.csv", [
            {"passage_id": "pmc-1-1", "source": "PMCID PMC1", "licence": "https://creativecommons.org/licenses/by/4.0/",
             "topics": "melanoma; nevus", "sha256": "a" * 64},
            {"passage_id": "book-2-1", "source": "StatPearls NBK2", "licence": "", "topics": "basal cell carcinoma", "sha256": "b" * 64}])
        (self.text / "downloads" / "pubmed").mkdir(parents=True)  # what the text dataset downloaded, with each full text's licence
        (self.text / "downloads" / "pubmed" / "abstracts.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in ({"pmid": "16", "pmcid": "PMC16", "licence": "https://creativecommons.org/licenses/by/4.0/"},
                                                  {"pmid": "14", "pmcid": "PMC14", "licence": ""})), encoding="utf-8")
        previous = os.getcwd()
        os.chdir(self.base)
        self.addCleanup(os.chdir, previous)

    def _table(self, name, folder=None):
        with ((folder or self.knowledge) / name).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))


class IndexTests(Project):
    def setUp(self):
        super().setUp()
        self.done = index.index()

    def test_the_index_holds_only_training_cases_and_library_passages_linked_by_disease(self):
        items = self._table("index.csv")
        self.assertEqual([(k["kind"], k["item_id"]) for k in items], [
            ("case", "case:15-1"), ("case", "case:pubmed-11"), ("case", "case:pubmed-16"), ("case", "case:13-1"),  # by path
            ("passage", "passage:pmc-1-1"), ("passage", "passage:book-2-1")])
        self.assertEqual({k["link"] for k in items}, {"disease"})
        self.assertNotIn("image", {k["kind"] for k in items})  # the photos of Version 1 are no longer indexed
        case = next(k for k in items if k["item_id"] == "case:pubmed-11")
        self.assertEqual((case["path"], case["source"], case["licence"]),
                         ("data/published_cases/files/text/train/melanoma/pubmed-11.txt", "PMID 11", "publisher's copyright (abstract in PubMed)"))
        self.assertEqual(next(k for k in items if k["item_id"] == "case:pubmed-16")["licence"],
                         "https://creativecommons.org/licenses/by/4.0/")  # the full text's own licence, as PMC gives it
        self.assertEqual(next(k for k in items if k["item_id"] == "case:13-1")["licence"], "CC BY-NC-SA 4.0 (PMC-Patients)")
        self.assertEqual(items[-1]["licence"], "(not recorded)")
        for held_out in ("12-1", "pubmed-14"):
            self.assertNotIn(held_out, (self.knowledge / "index.csv").read_text(encoding="utf-8"))

    def test_the_disease_table_has_a_row_for_each_of_the_eight_classes_and_counts_what_each_has(self):
        diseases = {d["class"]: d for d in self._table("diseases.csv")}
        self.assertEqual(sorted(diseases), sorted(CLASSES))
        self.assertEqual(list(next(iter(diseases.values()))), index.DISEASE_COLUMNS)
        self.assertNotIn("isic_diagnoses", index.DISEASE_COLUMNS)  # nothing of the ISIC photos is left
        melanoma = diseases["melanoma"]
        self.assertEqual({k: melanoma[k] for k in ("text_terms", "mesh_headings", "cases_train", "cases_val", "cases_test", "passages")},
                         {"text_terms": "melanoma; melanoma in situ", "mesh_headings": "Hutchinson's Melanotic Freckle; Melanoma",
                          "cases_train": "2", "cases_val": "0", "cases_test": "1", "passages": "1"})
        self.assertEqual(diseases["nevus"]["text_terms"], "nevus")  # a name that does not count is left out
        self.assertEqual(diseases["nevus"]["cases_val"], "1")
        self.assertEqual({k: diseases["vascular lesion"][k] for k in ("text_terms", "cases_train", "passages")},
                         {"text_terms": "", "cases_train": "0", "passages": "0"})  # a class without anything is still a row
        self.assertEqual((self.done["items"]["case"], self.done["items"]["passage"]), (4, 2))

    def test_the_report_counts_and_the_same_sources_give_the_same_files(self):
        report = (self.knowledge / "dataset_report.md").read_text(encoding="utf-8")
        for expected in ("<!-- step: index -->", "6 items: 4 training cases and 2 library passages", "| melanoma | 2 | 1 |",
                         "| basal cell carcinoma | 1 | 1 |", "| vascular lesion | 0 | 0 |"):
            self.assertIn(expected, report)
        files = [self.knowledge / "index.csv", self.knowledge / "diseases.csv", self.knowledge / "dataset_report.md"]
        before = {path: path.read_bytes() for path in files}
        index.index()
        self.assertEqual({path: path.read_bytes() for path in files}, before)

    def test_a_class_outside_the_eight_stops_the_step_and_a_missing_licence_file_is_named(self):
        path = self.text / "dataset.csv"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("basal cell carcinoma", "lentigo"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not among the eight of the dataset: lentigo"):
            index.index()
        path.write_text(text, encoding="utf-8")
        (self.text / "downloads" / "pubmed" / "abstracts.jsonl").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "holds the licence of each case taken from a PMC full text"):
            index.index()

    def test_a_validation_or_test_case_never_enters_the_index(self):
        written = (self.knowledge / "index.csv").read_text(encoding="utf-8")
        rows = self._table("index.csv")
        self.assertEqual(sum(r["item_id"].startswith("case:") for r in rows), sum(r["split"] == "train" for r in self._table("dataset.csv", self.text)))
        self.assertNotIn("/val/", written)
        self.assertNotIn("/test/", written)

    def test_the_report_has_the_four_steps_of_version_2_and_a_section_of_the_removed_cards_step_is_dropped(self):
        self.assertEqual(index.STEPS, ("library", "index", "appearance", "aspects"))
        report = self.knowledge / "dataset_report.md"
        report.write_text(report.read_text(encoding="utf-8") + "\n<!-- step: cards -->\n## Cards\n\nold\n", encoding="utf-8")
        index.index()
        self.assertNotIn("step: cards", report.read_text(encoding="utf-8"))
        self.assertIn("<!-- step: index -->", report.read_text(encoding="utf-8"))

    def test_a_held_out_case_among_the_items_is_found_and_stops_the_step(self):
        items = [{"item_id": "case:a", "path": "data/published_cases/files/text/train/nevus/a.txt"}, {"item_id": "case:b", "path": "data/published_cases/files/text/val/nevus/b.txt"},
                 {"item_id": "case:c", "path": "data/published_cases/files/text/test/nevus/c.txt"}, {"item_id": "passage:p", "path": "data/knowledge/texts/passages.jsonl#p"}]
        cases = [{"split": "train", "path": "train/nevus/a.txt"}, {"split": "val", "path": "val/nevus/b.txt"}, {"split": "test", "path": "test/nevus/c.txt"}]
        self.assertEqual(index.held_out_items(items, cases), ["case:b", "case:c"])
        self.assertEqual(index.held_out_items(items[:1] + items[3:], cases), [])
        with mock.patch.object(index, "held_out_items", return_value=["case:b"]):
            with self.assertRaisesRegex(ValueError, "validation or test items in the knowledge index: \\['case:b'\\]"):
                index.index()

    def test_the_command_line(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            index.main([])
        self.assertIn("Items: case 4, passage 2", shown.getvalue())


if __name__ == "__main__":
    unittest.main()
