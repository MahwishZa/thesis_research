"""The knowledge index (src/build/knowledge/index.py), on the tiny synthetic project of the image + metadata tests: no network."""

import contextlib
import io
import unittest

from src.build.image_metadata import prepare
from src.build.knowledge import index
from tests.build.image_metadata.test_prepare import Project


class IndexTests(Project):
    def setUp(self):
        super().setUp()
        prepare.build()
        self.done = index.index()

    def test_the_index_holds_only_training_items_and_library_passages_linked_by_disease(self):
        knowledge = self._table("index.csv", self.knowledge)
        self.assertEqual([(k["kind"], k["item_id"]) for k in knowledge], [
            ("image", "image:train/basal cell carcinoma/ISIC_0000004.jpg"), ("image", "image:train/melanoma/ISIC_0000002.jpg"),
            ("case", "case:15-1"), ("case", "case:pubmed-11"), ("case", "case:pubmed-16"), ("case", "case:13-1"),
            ("passage", "passage:pmc-1-1"), ("passage", "passage:book-2-1")])
        self.assertEqual({k["link"] for k in knowledge}, {"disease"})
        case = next(k for k in knowledge if k["item_id"] == "case:pubmed-11")
        self.assertEqual((case["path"], case["source"], case["licence"]),
                         ("data/text/dataset/train/melanoma/pubmed-11.txt", "PMID 11", "publisher's copyright (abstract in PubMed)"))
        self.assertEqual(next(k for k in knowledge if k["item_id"] == "case:pubmed-16")["licence"],
                         "https://creativecommons.org/licenses/by/4.0/")  # the full text's own licence, as PMC gives it
        self.assertEqual(next(k for k in knowledge if k["item_id"] == "case:13-1")["licence"], "CC BY-NC-SA 4.0 (PMC-Patients)")
        self.assertEqual(next(k for k in knowledge if k["kind"] == "image")["licence"], "CC-BY-NC")
        self.assertEqual(knowledge[-1]["licence"], "(not recorded)")
        for held_out in ("ISIC_0000001", "ISIC_0000003", "ISIC_0000005", "12-1", "pubmed-14"):
            self.assertNotIn(held_out, (self.knowledge / "index.csv").read_text(encoding="utf-8"))

    def test_the_disease_table_and_the_report_count_what_each_class_has(self):
        diseases = {d["class"]: d for d in self._table("diseases.csv", self.knowledge)}
        melanoma = diseases["melanoma"]
        self.assertEqual({k: melanoma[k] for k in ("isic_diagnoses", "text_terms", "mesh_headings", "questions_train", "questions_test",
                                                  "complete_test", "cases_train", "cases_test", "passages")},
                         {"isic_diagnoses": "Diagnosis of melanoma (2)", "text_terms": "melanoma; melanoma in situ",
                          "mesh_headings": "Hutchinson's Melanotic Freckle; Melanoma", "questions_train": "1", "questions_test": "1",
                          "complete_test": "1", "cases_train": "2", "cases_test": "1", "passages": "1"})
        self.assertEqual(diseases["nevus"]["text_terms"], "nevus")  # a name that does not count is left out
        self.assertEqual((self.done["items"]["image"], self.done["items"]["case"], self.done["items"]["passage"]), (2, 4, 2))
        report = (self.knowledge / "dataset_report.md").read_text(encoding="utf-8")
        for expected in ("<!-- step: index -->", "8 items: 2 training images, 4 training cases and 2 library passages",
                         "| melanoma | 1 | 2 | 1 |"):
            self.assertIn(expected, report)
        files = [self.knowledge / "index.csv", self.knowledge / "diseases.csv", self.knowledge / "dataset_report.md"]
        before = {path: path.read_bytes() for path in files}
        index.index()
        self.assertEqual({path: path.read_bytes() for path in files}, before)  # the same sources give the same files

    def test_an_out_of_date_image_metadata_table_or_different_classes_stop_the_step(self):
        table = self.data / "dataset.csv"
        text = table.read_text(encoding="utf-8")
        table.write_text("\n".join(line for line in text.splitlines() if "ISIC_0000005" not in line) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not hold exactly the image dataset's photos"):
            index.index()
        table.write_text(text, encoding="utf-8")
        path = self.base / "data" / "text" / "dataset.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("basal cell carcinoma", "dermatofibroma"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not have the same classes"):
            index.index()

    def test_the_command_line(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            index.main([])
        self.assertIn("Items: case 4, image 2, passage 2", shown.getvalue())


if __name__ == "__main__":
    unittest.main()
