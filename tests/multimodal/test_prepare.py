"""The multimodal dataset's preparation (src/multimodal/prepare.py) on a tiny synthetic image dataset: no network, no real data."""

import contextlib
import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.multimodal import prepare

COLLECTIONS = {"melanoma": "Anonymous", "nevus": "Anonymous", "basal cell carcinoma": "MILK study team"}


def record(number: str, label: str, clinical: dict) -> dict:
    """A record shaped like the ISIC Archive's (API v2): the diagnosis and its hints next to the patient's facts."""
    return {"isic_id": number, "attribution": COLLECTIONS[label], "copyright_license": "CC-BY-NC",
            "metadata": {"acquisition": {"image_type": "dermoscopic", "pixels_x": 600},
                         "clinical": {"diagnosis_1": "Malignant", "benign_malignant": "malignant", "melanocytic": True,
                                      "concomitant_biopsy": True, "lesion_id": f"IL_{number}", "family_hx_mm": False,
                                      "mel_thick_mm": 0.4, **clinical}}}


class Project(unittest.TestCase):
    """data/image/ with dataset.csv, raw_photos.csv and downloads/isic/, as the image dataset leaves them."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.image, self.data = self.base / "data" / "image", self.base / "data" / "multimodal"
        (self.image / "downloads" / "isic").mkdir(parents=True)
        cases = [("ISIC_0000001", "test", "melanoma", {"age_approx": 60, "sex": "male", "anatom_site_general": "back",
                                                        "anatom_site_1": "Trunk", "anatom_site_2": "Posterior trunk"}),
                 ("ISIC_0000002", "train", "melanoma", {"age_approx": 70, "sex": "female", "anatom_site_general": "lower extremity",
                                                         "anatom_site_1": "Lower extremity", "clin_size_long_diam_mm": 8.0}),
                 ("ISIC_0000003", "test", "nevus", {"age_approx": 30, "sex": "female"}),
                 ("ISIC_0000004", "train", "basal cell carcinoma", {"age_approx": 75, "sex": "male", "anatom_site_general": "head/neck",
                                                                     "anatom_site_1": "Head and neck"}),
                 ("ISIC_0000005", "val", "basal cell carcinoma", {})]
        dataset, photos = [], []
        for number, split, label, clinical in cases:
            entry = {"isic_id": number, "status": "found", "record": record(number, label, clinical)}
            (self.image / "downloads" / "isic" / f"{number}.json").write_text(json.dumps(entry), encoding="utf-8")
            dataset.append({"split": split, "label": label, "path": f"{split}/{label}/{number}.jpg", "sha256": hashlib.sha256(number.encode()).hexdigest(),
                            "lesion_group": number, "photo_id": number})
            photos.append({"photo_id": number, "isic_ids": number, "attribution": COLLECTIONS[label], "isic_diagnosis": f"Diagnosis of {label}",
                           "licence": "CC-0" if COLLECTIONS[label] == "Anonymous" else "CC-BY-NC",
                           "isic_record_sha256": hashlib.sha256(json.dumps(entry["record"], sort_keys=True).encode("utf-8")).hexdigest()})
        for name, rows in (("dataset.csv", dataset), ("raw_photos.csv", photos)):
            with (self.image / name).open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        text = self.base / "data" / "text"
        text.mkdir(parents=True)
        tables = {
            "dataset.csv": [{"split": split, "label": label, "path": f"{split}/{label}/{uid}.txt", "sha256": uid[-1] * 64,
                             "patient_uid": uid, "pmid": pmid, "source": source}
                            for split, label, uid, pmid, source in (
                                ("train", "melanoma", "pubmed-11", "11", "PubMed abstract"), ("test", "melanoma", "12-1", "12", "PMC-Patients"),
                                ("train", "melanoma", "pubmed-16", "16", "PMC full text"),
                                ("train", "nevus", "13-1", "13", "PMC-Patients"), ("val", "nevus", "pubmed-14", "14", "PMC full text"),
                                ("train", "basal cell carcinoma", "15-1", "15", "PMC-Patients"))],
            "class_terms.csv": [{"class": "melanoma", "term": "melanoma", "counts": "yes"}, {"class": "melanoma", "term": "melanoma in situ", "counts": "yes"},
                                {"class": "nevus", "term": "nevus", "counts": "yes"}, {"class": "nevus", "term": "nevoid", "counts": "no"},
                                {"class": "basal cell carcinoma", "term": "basal cell carcinoma", "counts": "yes"}],
            "term_check.csv": [{"class": "melanoma", "counts": "yes", "mesh_headings": "Melanoma; Hutchinson's Melanotic Freckle"},
                               {"class": "nevus", "counts": "yes", "mesh_headings": "Nevus"},
                               {"class": "basal cell carcinoma", "counts": "yes", "mesh_headings": "Carcinoma, Basal Cell"}],
            "library.csv": [{"passage_id": "pmc-1-1", "source": "PMCID PMC1", "licence": "https://creativecommons.org/licenses/by/4.0/",
                             "topics": "melanoma; nevus", "sha256": "a" * 64},
                            {"passage_id": "book-2-1", "source": "StatPearls NBK2", "licence": "", "topics": "basal cell carcinoma",
                             "sha256": "b" * 64}]}
        for name, rows in tables.items():
            with (text / name).open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        (text / "sources" / "pubmed").mkdir(parents=True)  # what the text dataset downloaded, with each full text's licence
        (text / "sources" / "pubmed" / "abstracts.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in ({"pmid": "16", "pmcid": "PMC16", "licence": "https://creativecommons.org/licenses/by/4.0/"},
                                                  {"pmid": "14", "pmcid": "PMC14", "licence": ""})), encoding="utf-8")
        previous = os.getcwd()
        os.chdir(self.base)
        self.addCleanup(os.chdir, previous)

    def _table(self, name):
        with (self.data / name).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))


class RecordsTests(Project):
    def test_fields_that_point_to_the_diagnosis_or_identify_a_lesion_are_never_offered_as_facts(self):
        for field, kind in (("age_approx", "fact"), ("sex", "fact"), ("anatom_site_general", "fact"), ("clin_size_long_diam_mm", "fact"),
                            ("fitzpatrick_skin_type", "fact"), ("diagnosis_1", "hint"), ("diagnosis_confirm_type", "hint"),
                            ("benign_malignant", "hint"), ("melanocytic", "hint"), ("mel_thick_mm", "hint"), ("mel_class", "hint"),
                            ("nevus_type", "hint"), ("concomitant_biopsy", "hint"), ("family_hx_mm", "hint"),
                            ("personal_hx_mm", "hint"), ("lesion_id", "identifier"), ("patient_id", "identifier")):
            with self.subTest(field):
                self.assertEqual(prepare.kind_of(f"metadata.clinical.{field}"), kind)

    def test_the_report_counts_each_fact_by_class_and_collection_and_lists_its_values(self):
        done = prepare.records()
        self.assertEqual(done["images"], 5)
        self.assertEqual(done["fields"]["metadata.clinical.anatom_site_general"], ("fact", 3))
        self.assertEqual(done["fields"]["metadata.clinical.melanocytic"], ("hint", 5))
        self.assertNotIn("metadata.acquisition.image_type", done["fields"])  # how the photo was taken, not the patient
        text = (self.data / "dataset_report.md").read_text(encoding="utf-8")
        for expected in ("<!-- step: records -->", "| basal cell carcinoma | 0 | 2 | 2 |", "| melanoma | 2 | 0 | 2 |",
                         "| `metadata.clinical.anatom_site_general` | fact | 3 of 5 (60%) | 2 of 3 (67%) | 1 of 2 (50%) |",
                         "| `metadata.clinical.melanocytic` | hint | 5 of 5 (100%) |",
                         "| `anatom_site_general` | 1 of 2 (50%) | 2 of 2 (100%) | 0 of 1 (0%) |",
                         "| Anonymous | back (1), lower extremity (1) |", "| MILK study team | head/neck (1) |",
                         "| melanoma | 65 | 60 to 70 |", "| basal cell carcinoma | 75 | 75 to 75 |"):
            self.assertIn(expected, text)
        self.assertNotIn("| `melanocytic` |", text)  # a hint never gets a row among the facts
        before = text
        prepare.records()
        self.assertEqual((self.data / "dataset_report.md").read_text(encoding="utf-8"), before)  # the same files give the same report

    def test_a_photo_uploaded_twice_counts_a_field_only_where_its_records_agree(self):
        second = record("ISIC_0000006", "melanoma", {"age_approx": 60, "sex": "female", "anatom_site_general": "back"})
        entry = {"isic_id": "ISIC_0000006", "status": "found", "record": second}
        (self.image / "downloads" / "isic" / "ISIC_0000006.json").write_text(json.dumps(entry), encoding="utf-8")
        table = self.image / "raw_photos.csv"
        with table.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rows[0]["isic_ids"] += " ISIC_0000006"
        rows[0]["isic_record_sha256"] += " " + hashlib.sha256(json.dumps(second, sort_keys=True).encode("utf-8")).hexdigest()
        with table.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        first = next(r for r in prepare.read_images() if r["isic_id"] == "ISIC_0000001 ISIC_0000006")
        self.assertEqual({k.split(".")[-1]: v for k, v in first["fields"].items() if prepare.kind_of(k) == "fact"},
                         {"age_approx": 60, "anatom_site_general": "back", "anatom_site_1": "Trunk",
                          "anatom_site_2": "Posterior trunk"})  # sex differs; a field only one record has is taken from it
        self.assertEqual(first["conflicts"], {"metadata.clinical.sex": ["male", "female"],
                                              "metadata.clinical.lesion_id": ["IL_ISIC_0000001", "IL_ISIC_0000006"]})
        prepare.records()
        text = (self.data / "dataset_report.md").read_text(encoding="utf-8")
        self.assertIn("Photos with more than one ISIC record (uploaded more than once): 1 (ISIC_0000001 ISIC_0000006)", text)
        self.assertIn("- Records disagree, so left out: `ISIC_0000001 ISIC_0000006` `metadata.clinical.sex`: male, female", text)
        self.assertNotIn("lesion_id`: IL_", text)  # only facts are listed
        (self.image / "downloads" / "isic" / "ISIC_0000006.json").unlink()  # a second record is checked like the first
        with self.assertRaisesRegex(ValueError, "1 ISIC records missing"):
            prepare.records()

    def test_a_missing_or_changed_record_stops_the_step_before_anything_is_written(self):
        path = self.image / "downloads" / "isic" / "ISIC_0000003.json"
        entry = json.loads(path.read_text(encoding="utf-8"))
        entry["record"]["metadata"]["clinical"]["age_approx"] = 35
        path.write_text(json.dumps(entry), encoding="utf-8")
        (self.image / "downloads" / "isic" / "ISIC_0000005.json").unlink()
        with self.assertRaisesRegex(ValueError, "1 ISIC records missing and 1 differing"):
            prepare.records()
        self.assertFalse(self.data.exists())

    def test_the_command_line_says_what_it_read_and_wrote(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["records"])
        for expected in ("ISIC records read and checked: 5", "metadata.clinical.sex: fact, in 4 records",
                         "metadata.clinical.melanocytic: hint, in 5 records", "data/multimodal/dataset_report.md"):
            self.assertIn(expected, shown.getvalue())
        (self.image / "raw_photos.csv").unlink()
        with self.assertRaisesRegex(SystemExit, "raw_photos.csv not found"):
            prepare.main(["records"])


class BuildTests(Project):
    def test_each_image_becomes_one_question_with_the_chosen_facts_and_nothing_else(self):
        prepare.build()
        questions = {q["isic_id"]: q for q in self._table("questions.csv")}
        self.assertEqual(len(questions), 5)
        first = questions["ISIC_0000001"]
        self.assertEqual({k: first[k] for k in ("split", "label", "path", "sha256", "lesion_group", "collection")},
                         {"split": "test", "label": "melanoma", "path": "test/melanoma/ISIC_0000001.jpg",
                          "sha256": hashlib.sha256(b"ISIC_0000001").hexdigest(),
                          "lesion_group": "ISIC_0000001", "collection": "Anonymous"})  # as in the image dataset
        self.assertEqual((first["facts"], first["complete"]), ("Age: about 60 years. Sex: male. Body site: trunk.", "yes"))
        self.assertEqual((questions["ISIC_0000003"]["facts"], questions["ISIC_0000003"]["complete"]),
                         ("Age: about 30 years. Sex: female. Body site: not recorded.", "no"))
        self.assertEqual(questions["ISIC_0000005"]["facts"], "Age: not recorded. Sex: not recorded. Body site: not recorded.")
        everything = (self.data / "questions.csv").read_text(encoding="utf-8")
        for never in ("Posterior trunk", "8.0", "back", "head/neck", "Malignant", "IL_ISIC"):  # other sites, size, hints, identifiers
            self.assertNotIn(never, everything)

    def test_the_knowledge_holds_only_training_items_and_library_passages_linked_by_disease(self):
        prepare.build()
        knowledge = self._table("knowledge.csv")
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
            self.assertNotIn(held_out, (self.data / "knowledge.csv").read_text(encoding="utf-8"))

    def test_the_disease_table_and_the_report_count_what_each_class_has(self):
        prepare.records()
        done = prepare.build()
        diseases = {d["class"]: d for d in self._table("diseases.csv")}
        melanoma = diseases["melanoma"]
        self.assertEqual({k: melanoma[k] for k in ("isic_diagnoses", "text_terms", "mesh_headings", "questions_train", "questions_test",
                                                  "complete_test", "cases_train", "cases_test", "passages")},
                         {"isic_diagnoses": "Diagnosis of melanoma (2)", "text_terms": "melanoma; melanoma in situ",
                          "mesh_headings": "Hutchinson's Melanotic Freckle; Melanoma", "questions_train": "1", "questions_test": "1",
                          "complete_test": "1", "cases_train": "2", "cases_test": "1", "passages": "1"})
        self.assertEqual(diseases["nevus"]["text_terms"], "nevus")  # a name that does not count is left out
        self.assertEqual((done["questions"], done["knowledge"]["passage"], done["complete"]["test"]), (5, 2, 1))
        report = (self.data / "dataset_report.md").read_text(encoding="utf-8")
        for expected in ("<!-- step: records -->", "<!-- step: build -->", "`questions.csv`: 5 questions", "train 2, val 1, test 2",
                         "2 training images, 4 training cases and 2 library passages", "| nevus | 0 of 0 | 0 of 0 | 0 of 1 | 0 | 0 | 1 |",
                         '- "Age: about 60 years. Sex: male. Body site: trunk."'):
            self.assertIn(expected, report)
        self.assertLess(report.index("<!-- step: records -->"), report.index("<!-- step: build -->"))
        before = {name: (self.data / name).read_bytes() for name in ("questions.csv", "knowledge.csv", "diseases.csv", "dataset_report.md")}
        prepare.build()
        self.assertEqual({name: (self.data / name).read_bytes() for name in before}, before)  # the same sources give the same files

    def test_classes_that_differ_between_the_datasets_stop_the_build(self):
        path = self.base / "data" / "text" / "dataset.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("basal cell carcinoma", "dermatofibroma"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not have the same classes"):
            prepare.build()
        self.assertFalse((self.data / "questions.csv").exists())

    def test_the_command_line_runs_the_build(self):
        with contextlib.redirect_stdout(io.StringIO()) as shown:
            prepare.main(["build"])
        for expected in ("Questions: 5 (train 2, val 1, test 2); with all three facts: train 2, val 0, test 1",
                         "Knowledge items: case 4, image 2, passage 2"):
            self.assertIn(expected, shown.getvalue())


if __name__ == "__main__":
    unittest.main()
