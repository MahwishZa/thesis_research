"""The command that builds the main benchmark from a local clone of the MedChange release (a small synthetic release)."""

import contextlib
import csv
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import build_benchmark as BB
from tests.fixtures import fixture


class BuildBenchmarkTests(unittest.TestCase):
    def _release(self, tmp, released_labels=None):
        medrev, groups = fixture()
        base = Path(tmp) / "mc" / "Datasets"
        base.mkdir(parents=True)
        fields = ["", "Label", "Question", "objectives", "conclusions", "PMID", "DOI_Date"]
        with open(base / "MedRevQA.csv", "w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            w.writerows(medrev.values())
        with open(base / "AllStudyGroups.csv", "w", encoding="utf-8", newline="") as h:
            w = csv.writer(h)
            w.writerow(["Group_ID", "Study_ID"])
            for gid, keys in groups.items():
                for j, k in enumerate(keys):
                    w.writerow([gid if j == 0 else "", k])
        pairs = released_labels or [("SUPPORTED", "NOT ENOUGH INFORMATION"), ("SUPPORTED", "NOT ENOUGH INFORMATION")]
        with open(base / "MedChangeQA.csv", "w", encoding="utf-8", newline="") as h:
            w = csv.writer(h)
            w.writerow(["Newest Label", "Outdated Label"])
            w.writerows(pairs)
        return Path(tmp) / "mc"

    def _args(self, tmp, release):
        return ["--medchange-dir", str(release), "--out-dir", str(Path(tmp) / "data"), "--manifest", str(Path(tmp) / "manifest.json"),
                "--n-unchanged", "5", "--dev-fraction", "0.5", "--seed", "3"]

    def test_the_benchmark_file_and_the_manifest_are_written_and_a_rerun_gives_the_same_files(self):
        with TemporaryDirectory() as tmp:
            release = self._release(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(BB.main(self._args(tmp, release)), 0)
            bench = (Path(tmp) / "data" / "benchmark.jsonl").read_bytes()
            manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["items"], 3)                             # two changed groups and one unchanged control
            self.assertEqual(manifest["counts"]["likely_label_noise_excluded"], 1)       # group 3: a label change between near-identical conclusions
            self.assertEqual(manifest["medchangeqa_rebuilt_and_verified"], 2)
            self.assertEqual(set(manifest["input_sha256"]), {"MedRevQA.csv", "AllStudyGroups.csv", "MedChangeQA.csv"})
            self.assertEqual(sorted(i for ids in manifest["splits"].values() for i in ids), ["MC-00001", "MC-00002", "MC-00003"])
            self.assertNotIn("Question", json.dumps(manifest))                           # the release is not redistributed
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(BB.main(self._args(tmp, release)), 0)
            self.assertEqual((Path(tmp) / "data" / "benchmark.jsonl").read_bytes(), bench)

    def test_a_reconstruction_that_differs_from_the_release_is_refused(self):
        with TemporaryDirectory() as tmp:
            release = self._release(tmp, released_labels=[("SUPPORTED", "NOT ENOUGH INFORMATION"), ("REFUTED", "SUPPORTED")])
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(BB.main(self._args(tmp, release)), 2)
            self.assertIn("benchmark build refused", err.getvalue())
            self.assertFalse((Path(tmp) / "data" / "benchmark.jsonl").exists())

    def test_a_missing_input_file_is_reported(self):
        with TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(BB.main(self._args(tmp, Path(tmp) / "absent")), 2)
            self.assertIn("missing input", err.getvalue())


if __name__ == "__main__":
    unittest.main()
