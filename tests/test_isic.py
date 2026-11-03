"""The ISIC label check, with the ISIC Archive replaced by fixed records (no network)."""

import csv
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from src import isic

LEGACY = {"isic_id": "ISIC_1", "copyright_license": "CC-BY-NC",
          "metadata": {"clinical": {"diagnosis": "actinic keratosis", "benign_malignant": "benign", "age_approx": 70}}}
HIERARCHY = {"isic_id": "ISIC_2", "copyright_license": "CC-0",
             "metadata": {"clinical": {"diagnosis_1": "Benign", "diagnosis_3": "Seborrheic keratosis", "lesion_id": "IL_9"}}}


class IsicTests(unittest.TestCase):
    def test_fields_from_either_record_layout(self):
        self.assertEqual(isic.diagnosis_fields(LEGACY), {"copyright_license": "CC-BY-NC",
                                                         "metadata.clinical.benign_malignant": "benign",
                                                         "metadata.clinical.diagnosis": "actinic keratosis"})
        self.assertIn("metadata.clinical.diagnosis_3", isic.diagnosis_fields(HIERARCHY))

    def test_supported_labels(self):
        self.assertEqual(isic.supported_labels(isic.diagnosis_fields(LEGACY), ["actinic keratosis", "nevus"]), ["actinic keratosis"])
        self.assertEqual(isic.supported_labels(isic.diagnosis_fields(HIERARCHY), ["melanoma", "seborrheic keratosis"]),
                         ["seborrheic keratosis"])
        self.assertEqual(isic.supported_labels({"x.diagnosis": "Dermatofibroma"}, ["melanoma"]), [])

    def test_fetch_caches_and_reports_missing_records(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(LEGACY).encode()
            with mock.patch("urllib.request.urlopen", return_value=response) as call, mock.patch("time.sleep"):
                self.assertEqual(isic.fetch("ISIC_1", cache), LEGACY)
                self.assertEqual(isic.fetch("ISIC_1", cache), LEGACY)  # second time from the cache
            self.assertEqual(call.call_count, 1)
            missing = urllib.error.HTTPError("u", 404, "Not Found", {}, io.BytesIO(b""))
            with mock.patch("urllib.request.urlopen", side_effect=missing), mock.patch("time.sleep"):
                self.assertIn("404", isic.fetch("ISIC_3", cache)["_error"])

    def test_check_covers_conflicts_and_test_images(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = Path(d) / "manifest.csv"
            rows = [("train", "actinic keratosis", "ISIC_1"), ("train", "nevus", "ISIC_1"),   # conflict, training only
                    ("test", "seborrheic keratosis", "ISIC_2"), ("train", "melanoma", "ISIC_2"),  # conflict with a test copy
                    ("test", "melanoma", "ISIC_3"),                                          # test image, no conflict
                    ("train", "melanoma", "ISIC_4")]                                         # neither: not checked by default
            with open(manifest, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["split", "label", "path", "isic_id"])
                w.writerows((split, label, f"{split}/{label}/{i}.jpg", i) for split, label, i in rows)
            records = {"ISIC_1": LEGACY, "ISIC_2": HIERARCHY, "ISIC_3": {"_error": "not found in the ISIC Archive (404)"}}
            with mock.patch("src.isic.fetch", side_effect=lambda i, *a, **k: records[i]):
                report = isic.check(manifest, Path(d) / "out", Path(d) / "cache")
            with open(Path(d) / "out" / "isic_check.csv", encoding="utf-8") as f:
                table = {r["isic_id"]: r for r in csv.DictReader(f)}
            self.assertEqual(sorted(table), ["ISIC_1", "ISIC_2", "ISIC_3"])
            self.assertEqual(table["ISIC_1"]["verdict"], "ISIC supports: actinic keratosis")
            self.assertEqual(table["ISIC_2"]["verdict"], "ISIC supports: seborrheic keratosis")
            self.assertTrue(table["ISIC_3"]["verdict"].startswith("not checked"))
            text = report.read_text(encoding="utf-8")
            self.assertIn("conflict: melanoma vs seborrheic keratosis (a copy is in test)", text)
            self.assertIn("| CC-BY-NC | 1 |", text)
            with open(Path(d) / "out" / "label_corrections.csv", encoding="utf-8") as f:
                self.assertEqual(list(csv.DictReader(f)), [])  # ISIC_2 confirms its test label; ISIC_3 is undecided

    def test_corrections_list_test_images_that_isic_contradicts(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, checked, out = Path(d) / "manifest.csv", Path(d) / "isic_check.csv", Path(d) / "label_corrections.csv"
            manifest.write_text("split,label,path,isic_id\n"
                                "test,seborrheic keratosis,Test/sk/ISIC_2.jpg,ISIC_2\n"   # ISIC: melanoma -> corrected
                                "train,melanoma,Train/mel/ISIC_2.jpg,ISIC_2\n"            # training copy: not a test image
                                "test,melanoma,Test/mel/ISIC_3.jpg,ISIC_3\n"              # ISIC confirms: unchanged
                                "test,nevus,Test/nev/ISIC_5.jpg,ISIC_5\n"                 # record not fetched: undecided
                                "test,nevus,Test/nev/x.jpg,\n", encoding="utf-8")          # no ISIC number: undecided
            checked.write_text("isic_id,verdict,isic_fields\n"
                               "ISIC_2,ISIC supports: melanoma,diagnosis_3=Melanoma in situ\n"
                               "ISIC_3,ISIC supports: melanoma,diagnosis_3=Melanoma\n"
                               "ISIC_5,not checked: HTTP 503,\n", encoding="utf-8")
            fixes, undecided = isic.write_corrections(manifest, checked, out)
            self.assertEqual([(f["path"], f["folder_label"], f["corrected_label"]) for f in fixes],
                             [("Test/sk/ISIC_2.jpg", "seborrheic keratosis", "melanoma")])
            self.assertEqual(undecided, ["Test/nev/ISIC_5.jpg", "Test/nev/x.jpg"])
            labels = ["melanoma", "nevus", "seborrheic keratosis"]
            self.assertEqual(isic.read_corrections(out, labels, {"Test/sk/ISIC_2.jpg": "seborrheic keratosis"}),
                             {"Test/sk/ISIC_2.jpg": "melanoma"})
            self.assertEqual(isic.read_corrections(out, labels, {"Test/mel/ISIC_3.jpg": "melanoma"}), {})
            with self.assertRaises(ValueError):  # the dataset now labels the image differently
                isic.read_corrections(out, labels, {"Test/sk/ISIC_2.jpg": "nevus"})
            with self.assertRaises(ValueError):  # melanoma is not an answer option
                isic.read_corrections(out, ["nevus", "seborrheic keratosis"], {})


if __name__ == "__main__":
    unittest.main()
