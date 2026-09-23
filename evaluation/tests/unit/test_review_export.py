"""Tests for the reviewer-facing export.

Two kinds here. Most build synthetic records so the logic is exercised
anywhere. A few read the committed pool directly, because "the file the
reviewer will actually open is neutral and complete" is the claim that matters
and it can only be checked against the real file.
"""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.questions import (
    REVIEW_COLUMNS, WITHHELD_FROM_REVIEW, REVIEW_DECISIONS,
)
from experiments.shared.questions import export_review as ex

POOL = Path("experiments/shared/questions")


def record(**kw):
    base = {
        "question_id": "ADQ-000000000001",
        "question": "Does donepezil improve cognition in mild Alzheimer disease?",
        "topic": "treatment",
        "subtopic": "pharmacological",
        "reference_answer": "Donepezil produced a small improvement in cognition.",
        "reference_source": "Cochrane Database of Systematic Reviews; CD001190",
        "reference_locator": "PMID:12345678 | CD001190.pub3",
        "reference_date": "2018-06",
        "AD_anchor": True,
        "determinate": True,
        "corpus_support_expected": True,
        "temporal_candidate": True,
        "ambiguity_candidate": True,
        "status": "validated",
        "rejection_reason": None,
        "metadata": {
            "reference_source_type": "peer_reviewed_evidence_synthesis",
            "source_verdict_label": "SUPPORTED",
            "verification_required": True,
        },
    }
    base.update(kw)
    return base


def write_pool(tmp, records):
    d = Path(tmp) / "pool"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "candidates.jsonl", "w", encoding="utf-8") as h:
        for r in records:
            h.write(json.dumps(r, sort_keys=True) + "\n")
    return d


class NeutralityTests(unittest.TestCase):

    def test_export_columns_are_exactly_the_agreed_set(self):
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, [record()])
            ex.run(d)
            rows = list(csv.DictReader(open(d / "review.csv", encoding="utf-8")))
        self.assertEqual(list(rows[0]), list(REVIEW_COLUMNS))

    def test_internal_classifications_never_appear(self):
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, [record()])
            ex.run(d)
            header = open(d / "review.csv", encoding="utf-8").readline()
        for field in WITHHELD_FROM_REVIEW:
            self.assertNotIn(field, header)

    def test_verdict_label_does_not_leak_into_any_cell(self):
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, [record()])
            ex.run(d)
            body = (d / "review.csv").read_text(encoding="utf-8")
        self.assertNotIn("source_verdict_label", body)
        self.assertNotIn("verification_required", body)

    def test_flagged_and_unflagged_candidates_are_indistinguishable(self):
        """Two candidates differing only in internal flags export identically."""
        flagged = record(question_id="ADQ-a", temporal_candidate=True,
                         ambiguity_candidate=True)
        plain = record(question_id="ADQ-b", temporal_candidate=False,
                       ambiguity_candidate=False)
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, [flagged, plain])
            ex.run(d)
            rows = list(csv.DictReader(open(d / "review.csv", encoding="utf-8")))
        a, b = ({k: v for k, v in r.items() if k != "question_id"} for r in rows)
        self.assertEqual(a, b)

    def test_export_refuses_a_row_carrying_a_withheld_field(self):
        with self.assertRaises(ex.ExportError) as ctx:
            ex.assert_neutral([{"question_id": "x", "temporal_candidate": True}])
        self.assertIn("temporal_candidate", str(ctx.exception))

    def test_export_refuses_unexpected_columns(self):
        with self.assertRaises(ex.ExportError):
            ex.assert_neutral([{c: "" for c in REVIEW_COLUMNS} | {"extra": 1}])

    def test_four_decisions_are_defined(self):
        self.assertEqual(REVIEW_DECISIONS,
                         ("ACCEPT", "REVISE", "REJECT", "HOLD"))


class CompletenessTests(unittest.TestCase):

    def test_only_validated_candidates_reach_review(self):
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, [
                record(question_id="ADQ-a", status="validated"),
                record(question_id="ADQ-b", status="rejected",
                       rejection_reason="near_duplicate_of_earlier_candidate"),
            ])
            result = ex.run(d)
            rows = list(csv.DictReader(open(d / "review.csv", encoding="utf-8")))
        self.assertEqual(result["reviewable"], 1)
        self.assertEqual([r["question_id"] for r in rows], ["ADQ-a"])

    def test_duplicate_export_rows_are_refused(self):
        rows = [{c: "" for c in REVIEW_COLUMNS} for _ in range(2)]
        for r in rows:
            r["question_id"] = "ADQ-dup"
        with self.assertRaises(ex.ExportError) as ctx:
            ex.assert_complete(rows, ["ADQ-dup"])
        self.assertIn("duplicate", str(ctx.exception))

    def test_missing_candidate_is_refused(self):
        rows = [{c: "" for c in REVIEW_COLUMNS}]
        rows[0]["question_id"] = "ADQ-a"
        with self.assertRaises(ex.ExportError):
            ex.assert_complete(rows, ["ADQ-a", "ADQ-b"])

    def test_rerun_with_no_prior_decisions_overwrites_freely(self):
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, [record(question_id="ADQ-a")])
            ex.run(d)
            ex.run(d)  # still blank, so a second run is not a hazard

    def test_rerun_after_a_decision_is_recorded_is_refused_without_force(self):
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, [record(question_id="ADQ-a")])
            ex.run(d)
            rows = list(csv.DictReader(open(d / "review.csv", encoding="utf-8")))
            rows[0]["review_decision"] = "ACCEPT"
            with open(d / "review.csv", "w", encoding="utf-8", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(REVIEW_COLUMNS))
                w.writeheader()
                w.writerows(rows)
            with self.assertRaises(ex.ExportError):
                ex.run(d)
            ex.run(d, force=True)  # explicit override still works
            after = list(csv.DictReader(open(d / "review.csv", encoding="utf-8")))
            self.assertEqual(after[0]["review_decision"], "")

    def test_missing_provenance_is_reported_not_filled(self):
        rows = [{c: "" for c in REVIEW_COLUMNS}]
        rows[0]["question_id"] = "ADQ-a"
        gaps = ex.missing_provenance(rows)
        self.assertEqual(gaps[0]["question_id"], "ADQ-a")
        self.assertIn("source_locator", gaps[0]["missing"])
        # Nothing was invented to fill the gap.
        self.assertEqual(rows[0]["source_locator"], "")

    def test_round_trip_preserves_provenance(self):
        original = record()
        questions = ex.to_questions([original])
        self.assertEqual(questions[0].reference_locator,
                         original["reference_locator"])
        self.assertEqual(questions[0].metadata["source_verdict_label"],
                         "SUPPORTED")
        self.assertTrue(questions[0].temporal_candidate)


class CommittedPoolTests(unittest.TestCase):
    """Checks against the real file the reviewer will open."""

    @classmethod
    def setUpClass(cls):
        if not (POOL / "candidates.jsonl").exists():
            raise unittest.SkipTest("pool not present")
        cls.pool = [json.loads(l) for l in
                    (POOL / "candidates.jsonl").read_text(
                        encoding="utf-8").splitlines() if l.strip()]
        cls.rows = list(csv.DictReader(
            open(POOL / "review.csv", encoding="utf-8")))

    def test_all_validated_candidates_present_exactly_once(self):
        expected = [r["question_id"] for r in self.pool
                    if r["status"] == "validated"]
        actual = [r["question_id"] for r in self.rows]
        self.assertEqual(len(expected), 123)
        self.assertEqual(len(actual), 123)
        self.assertEqual(sorted(actual), sorted(expected))
        self.assertEqual(len(set(actual)), 123)

    def test_no_candidate_was_altered(self):
        by_id = {r["question_id"]: r for r in self.pool}
        for row in self.rows:
            source = by_id[row["question_id"]]
            self.assertEqual(row["question"], source["question"])
            self.assertEqual(row["reference_answer"], source["reference_answer"])
            self.assertEqual(row["reference_source"], source["reference_source"])
            self.assertEqual(row["reference_date"], source["reference_date"])
            self.assertEqual(row["source_locator"], source["reference_locator"])

    def test_provenance_is_complete_for_every_row(self):
        self.assertEqual(ex.missing_provenance(self.rows), [])

    def test_committed_review_file_is_neutral(self):
        self.assertEqual(list(self.rows[0]), list(REVIEW_COLUMNS))
        header = open(POOL / "review.csv", encoding="utf-8").readline()
        for field in WITHHELD_FROM_REVIEW:
            self.assertNotIn(field, header)

    def test_decision_columns_are_now_fully_recorded(self):
        """review.csv started empty; human review of all 123 is now complete
        (see _archive/docs_legacy/status_and_decisions.md). Every row must carry a valid
        decision and reviewer/date provenance for it."""
        for row in self.rows:
            self.assertIn(row["review_decision"], REVIEW_DECISIONS)
            self.assertTrue(row["reviewer_id"].strip())
            self.assertTrue(row["review_date"].strip())

    def test_rerunning_export_refuses_to_erase_recorded_decisions(self):
        """A blank re-export would silently destroy the completed human
        review; ex.run() must refuse unless explicitly forced."""
        with self.assertRaises(ex.ExportError):
            ex.run(POOL)

    def test_internal_flags_are_still_preserved_in_the_pool(self):
        """Neutrality is about the review file, not about losing the metadata."""
        self.assertTrue(any(r["temporal_candidate"] for r in self.pool))
        self.assertTrue(any(r["ambiguity_candidate"] for r in self.pool))
        self.assertTrue(all("reference_source_type" in r["metadata"]
                            for r in self.pool))


if __name__ == "__main__":
    unittest.main()
