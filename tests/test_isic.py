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
                w.writerow(["split", "label", "isic_id"])
                w.writerows(rows)
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


if __name__ == "__main__":
    unittest.main()
