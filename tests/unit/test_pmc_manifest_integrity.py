"""The finalized PMC manifest, verified against itself.

`metadata/pmc.csv` is the repository's first EXECUTED artifact (ledger D-42):
114,256 rows, produced by `--finalize` after the log-evidence fix in D-41 and
independently checked here rather than trusted from the run's own log line.

This is not a corpus-content test - it does not know what PMC actually
contains - it locks internal consistency: every row accounted for exactly
once, statuses summing to the total, and path fields present exactly where
each status requires them. A future accidental re-finalization, a hand edit,
or a merge conflict resolved the wrong way would trip one of these.

Skips cleanly wherever the finalized manifest is not present (e.g. a fresh
clone before Step 1 runs) rather than failing the suite.
"""

import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "alzheimer_corpus" / "metadata" / "pmc.csv"

#: Locked to the verified 2026-09-18 finalization (ledger D-42). A
#: legitimate re-finalization (e.g. after the corpus grows) will change
#: these and this test should be updated deliberately alongside it - not
#: silently, which is why the numbers are asserted exactly rather than as
#: a lower bound.
EXPECTED_TOTAL_ROWS = 114256
EXPECTED_VERIFIED_ROWS = 114157
EXPECTED_UNAVAILABLE_ROWS = 99


def load_rows():
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@unittest.skipUnless(MANIFEST.exists(), "PMC manifest not finalized yet")
class PMCManifestIntegrityTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows = load_rows()

    def test_row_count_matches_the_verified_finalization(self):
        self.assertEqual(len(self.rows), EXPECTED_TOTAL_ROWS)

    def test_every_pmcid_version_pair_is_unique(self):
        pairs = [(r["pmcid"], r["version"]) for r in self.rows]
        self.assertEqual(len(pairs), len(set(pairs)),
                         "duplicate (pmcid, version) rows in the manifest")

    def test_status_counts_sum_to_the_total_and_match_expectations(self):
        statuses = {}
        for row in self.rows:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        self.assertEqual(statuses.get("already_verified", 0),
                         EXPECTED_VERIFIED_ROWS)
        self.assertEqual(statuses.get("unavailable_current_dataset", 0),
                         EXPECTED_UNAVAILABLE_ROWS)
        self.assertEqual(sum(statuses.values()), EXPECTED_TOTAL_ROWS)

    def test_verified_rows_carry_both_paths_unavailable_rows_carry_neither(self):
        for row in self.rows:
            if row["status"] == "already_verified":
                self.assertTrue(row["json_path"], row)
                self.assertTrue(row["xml_path"], row)
                self.assertTrue(row["version"], row)
            elif row["status"] == "unavailable_current_dataset":
                self.assertFalse(row["json_path"], row)
                self.assertFalse(row["xml_path"], row)
                self.assertFalse(row["version"], row)
            else:
                self.fail(f"unexpected status: {row['status']!r}")

    def test_no_row_is_missing_a_pmcid(self):
        self.assertTrue(all(row["pmcid"] for row in self.rows))

    def test_header_matches_the_scripts_declared_schema(self):
        with MANIFEST.open(encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
        self.assertEqual(header, [
            "pmid", "pmcid", "version", "doi", "title", "citation",
            "is_pmc_openaccess", "is_manuscript", "license_code",
            "is_retracted", "json_path", "xml_path", "status",
        ])


if __name__ == "__main__":
    unittest.main()
