"""Tests for the validation/test split over the reviewed question pool.

Two kinds. Most build a synthetic reviewed pool so the split logic is
exercised anywhere. A few read the committed ``review.csv`` and
``splits.json`` directly, because "the real 123-question review actually
produces a valid, reproducible split" is the claim that matters and only the
real files can check it.
"""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.questions import split as sp

POOL = Path("experiments/questions")

CANDIDATE_FIELDS = ("question_id", "question", "topic", "subtopic",
                     "reference_answer", "reference_source", "reference_date",
                     "status", "temporal_candidate")


def candidate(qid, topic, temporal, **kw):
    base = {
        "question_id": qid,
        "question": f"Does treatment X help condition Y for {qid}?",
        "topic": topic,
        "subtopic": "sub",
        "reference_answer": "Some verbatim answer.",
        "reference_source": "Cochrane Database of Systematic Reviews; CD000000",
        "reference_date": "2020-01",
        "status": "validated",
        "temporal_candidate": temporal,
    }
    base.update(kw)
    return base


def review_row(qid, decision, note=""):
    return {
        "question_id": qid, "question": "", "reference_answer": "",
        "reference_source": "", "reference_source_type": "", "reference_date": "",
        "source_locator": "", "topic": "", "subtopic": "", "AD_anchor": "",
        "review_decision": decision, "reviewer_note": note,
        "reviewer_id": "R1", "review_date": "2026-09-20",
    }


def write_pool(tmp, candidates, review_rows):
    d = Path(tmp) / "pool"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "candidates.jsonl", "w", encoding="utf-8") as h:
        for c in candidates:
            h.write(json.dumps(c, sort_keys=True) + "\n")
    with open(d / "review.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(review_row("x", "ACCEPT")))
        w.writeheader()
        for r in review_rows:
            w.writerow(r)
    return d


def make_pool(n_per_bucket=6):
    """A small synthetic pool: 3 topics x {temporal, non-temporal}."""
    candidates, rows = [], []
    i = 0
    for topic in ("treatment", "diagnosis", "management"):
        for temporal in (True, False):
            for _ in range(n_per_bucket):
                qid = f"ADQ-{i:04d}"
                candidates.append(candidate(qid, topic, temporal))
                decision = "ACCEPT" if i % 3 else "REVISE"
                rows.append(review_row(qid, decision))
                i += 1
    # A couple of rejects and a hold, so exclusion has something to exclude.
    candidates.append(candidate("ADQ-reject-1", "treatment", False))
    rows.append(review_row("ADQ-reject-1", "REJECT", "bad answer"))
    candidates.append(candidate("ADQ-hold-1", "treatment", True))
    rows.append(review_row("ADQ-hold-1", "HOLD", "needs source check"))
    return candidates, rows


class LoadingAndFilteringTests(unittest.TestCase):

    def test_reject_and_hold_are_excluded(self):
        candidates, rows = make_pool()
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, candidates, rows)
            reviewed = sp.load_reviewed_pool(d)
        usable_ids = {r["question_id"] for r in sp.usable_records(reviewed)}
        self.assertNotIn("ADQ-reject-1", usable_ids)
        self.assertNotIn("ADQ-hold-1", usable_ids)

    def test_review_row_with_no_matching_candidate_is_refused(self):
        candidates, rows = make_pool()
        rows.append(review_row("ADQ-ghost", "ACCEPT"))
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, candidates, rows)
            with self.assertRaises(sp.SplitError):
                sp.load_reviewed_pool(d)

    def test_missing_files_are_refused_clearly(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(sp.SplitError):
                sp.load_reviewed_pool(Path(tmp) / "nowhere")


class AssignmentTests(unittest.TestCase):

    def setUp(self):
        candidates, rows = make_pool()
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, candidates, rows)
            self.reviewed = sp.load_reviewed_pool(d)
        self.usable = sp.usable_records(self.reviewed)

    def test_deterministic_given_same_seed(self):
        a = sp.assign_splits(self.usable, seed=42)
        b = sp.assign_splits(self.usable, seed=42)
        self.assertEqual(a, b)

    def test_different_seed_can_change_the_assignment(self):
        a = sp.assign_splits(self.usable, seed=1)
        b = sp.assign_splits(self.usable, seed=2)
        self.assertNotEqual(a, b)

    def test_covers_the_usable_pool_exactly_once(self):
        assignment = sp.assign_splits(self.usable)
        usable_ids = {r["question_id"] for r in self.usable}
        self.assertEqual(set(assignment), usable_ids)
        self.assertEqual(len(assignment), len(usable_ids))

    def test_validation_and_test_are_disjoint(self):
        assignment = sp.assign_splits(self.usable)
        validation = {q for q, s in assignment.items() if s == "validation"}
        test = {q for q, s in assignment.items() if s == "test"}
        self.assertEqual(validation & test, set())
        self.assertEqual(validation | test, set(assignment))

    def test_only_known_split_names_are_produced(self):
        assignment = sp.assign_splits(self.usable)
        self.assertTrue(set(assignment.values()) <= set(sp.SPLIT_NAMES))

    def test_validation_fraction_roughly_respected(self):
        assignment = sp.assign_splits(self.usable, validation_fraction=0.2)
        n_val = sum(1 for s in assignment.values() if s == "validation")
        expected = round(len(self.usable) * 0.2)
        self.assertLessEqual(abs(n_val - expected), 2)

    def test_every_topic_present_in_both_splits_when_large_enough(self):
        assignment = sp.assign_splits(self.usable)
        by_id = {r["question_id"]: r for r in self.usable}
        topics = {r["topic"] for r in self.usable}
        for topic in topics:
            splits_seen = {assignment[qid] for qid, r in by_id.items()
                           if r["topic"] == topic}
            self.assertEqual(splits_seen, set(sp.SPLIT_NAMES),
                              f"topic {topic!r} missing from a split")

    def test_temporal_and_non_temporal_both_present_in_both_splits(self):
        assignment = sp.assign_splits(self.usable)
        by_id = {r["question_id"]: r for r in self.usable}
        for split in sp.SPLIT_NAMES:
            ids = [q for q, s in assignment.items() if s == split]
            temporal = [by_id[q]["temporal_candidate"] for q in ids]
            self.assertIn(True, temporal, f"{split} has no temporal item")
            self.assertIn(False, temporal, f"{split} has no non-temporal item")

    def test_temporal_share_is_close_between_splits(self):
        """Guards against the topic-only stratification this module replaced,
        which let validation's temporal share drift far from test's."""
        assignment = sp.assign_splits(self.usable)
        by_id = {r["question_id"]: r for r in self.usable}
        shares = {}
        for split in sp.SPLIT_NAMES:
            ids = [q for q, s in assignment.items() if s == split]
            shares[split] = sum(by_id[q]["temporal_candidate"] for q in ids) / len(ids)
        self.assertLess(abs(shares["validation"] - shares["test"]), 0.15)

    def test_rejects_out_of_range_fraction(self):
        with self.assertRaises(sp.SplitError):
            sp.assign_splits(self.usable, validation_fraction=0.0)
        with self.assertRaises(sp.SplitError):
            sp.assign_splits(self.usable, validation_fraction=1.0)


class ValidationCheckTests(unittest.TestCase):

    def test_assert_valid_passes_a_correct_assignment(self):
        candidates, rows = make_pool()
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, candidates, rows)
            reviewed = sp.load_reviewed_pool(d)
        usable = sp.usable_records(reviewed)
        assignment = sp.assign_splits(usable)
        sp.assert_valid(reviewed, assignment)  # must not raise

    def test_assert_valid_catches_a_leaked_reject(self):
        candidates, rows = make_pool()
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, candidates, rows)
            reviewed = sp.load_reviewed_pool(d)
        usable = sp.usable_records(reviewed)
        assignment = sp.assign_splits(usable)
        assignment["ADQ-reject-1"] = "test"  # smuggle a REJECT in
        with self.assertRaises(sp.SplitError):
            sp.assert_valid(reviewed, assignment)

    def test_assert_valid_catches_incomplete_coverage(self):
        candidates, rows = make_pool()
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, candidates, rows)
            reviewed = sp.load_reviewed_pool(d)
        usable = sp.usable_records(reviewed)
        assignment = sp.assign_splits(usable)
        del assignment[next(iter(assignment))]
        with self.assertRaises(sp.SplitError):
            sp.assert_valid(reviewed, assignment)


class WriteSplitTests(unittest.TestCase):

    def test_writes_a_valid_json_file(self):
        candidates, rows = make_pool()
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, candidates, rows)
            out = Path(tmp) / "splits.json"
            result = sp.write_split(d, out)
            on_disk = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["items"], result["items"])

    def test_refuses_to_silently_change_an_existing_split(self):
        candidates, rows = make_pool()
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, candidates, rows)
            out = Path(tmp) / "splits.json"
            sp.write_split(d, out)
            # Change the pool: flip one REJECT to ACCEPT.
            for r in rows:
                if r["question_id"] == "ADQ-reject-1":
                    r["review_decision"] = "ACCEPT"
            write_pool(tmp, candidates, rows)
            with self.assertRaises(sp.SplitError):
                sp.write_split(d, out)
            sp.write_split(d, out, force=True)  # explicit override succeeds

    def test_rerunning_with_no_pool_change_is_a_no_op(self):
        candidates, rows = make_pool()
        with TemporaryDirectory() as tmp:
            d = write_pool(tmp, candidates, rows)
            out = Path(tmp) / "splits.json"
            first = sp.write_split(d, out)
            second = sp.write_split(d, out)  # must not raise
        self.assertEqual(first["items"], second["items"])


class CommittedSplitTests(unittest.TestCase):
    """Checks against the real, committed 123-question review and split."""

    @classmethod
    def setUpClass(cls):
        if not (POOL / "review.csv").exists() or not (POOL / "splits.json").exists():
            raise unittest.SkipTest("committed pool/split not present")
        cls.reviewed = sp.load_reviewed_pool(POOL)
        cls.on_disk = json.loads((POOL / "splits.json").read_text(encoding="utf-8"))

    def test_committed_split_is_internally_valid(self):
        assignment = {item["question_id"]: item["split"]
                      for item in self.on_disk["items"]}
        sp.assert_valid(self.reviewed, assignment)

    def test_committed_split_matches_a_fresh_recomputation(self):
        fresh = sp.build_split(
            POOL,
            validation_fraction=self.on_disk["validation_fraction"],
            seed=self.on_disk["seed"],
        )
        self.assertEqual(fresh["items"], self.on_disk["items"])

    def test_committed_split_excludes_every_rejected_question(self):
        rejected = {r["question_id"] for r in self.reviewed
                    if r["review_decision"] == "REJECT"}
        split_ids = {item["question_id"] for item in self.on_disk["items"]}
        self.assertEqual(rejected & split_ids, set())
        self.assertGreater(len(rejected), 0)  # the check is actually exercised

    def test_committed_split_totals_match_review_counts(self):
        decisions = [r["review_decision"] for r in self.reviewed]
        usable = sum(1 for d in decisions if d in sp.USABLE_DECISIONS)
        self.assertEqual(len(self.on_disk["items"]), usable)
        self.assertEqual(self.on_disk["total_reviewed"], len(self.reviewed))
        self.assertEqual(self.on_disk["total_usable"], usable)

    def test_revise_items_are_flagged_pending_revision(self):
        for item in self.on_disk["items"]:
            expected = item["review_decision"] == "REVISE"
            self.assertEqual(item["pending_revision"], expected)


if __name__ == "__main__":
    unittest.main()
