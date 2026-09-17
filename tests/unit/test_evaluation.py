"""Tests for the corpus-independent evaluation infrastructure."""

import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from evaluation import annotation as ann
from evaluation import freezing as fz
from evaluation import questions as qs
from evaluation import stats as st


def question(**kw):
    base = dict(
        question="Do cholinesterase inhibitors improve cognition in mild "
                 "Alzheimer disease?",
        topic="treatment",
        reference_answer="Yes, with a modest average benefit.",
        reference_source="Cochrane Database Syst Rev CD001190",
        reference_date="2018-06",
        AD_anchor=True,
        determinate=True,
    )
    base.update(kw)
    return qs.EvaluationQuestion(**base)


class QuestionSchemaTests(unittest.TestCase):

    def test_id_is_stable_and_content_derived(self):
        self.assertEqual(question().question_id, question().question_id)
        self.assertNotEqual(
            question().question_id,
            question(reference_source="Other source").question_id,
        )

    def test_missing_reference_is_refused(self):
        for field in ("reference_answer", "reference_source", "reference_date"):
            with self.subTest(field=field), self.assertRaises(qs.QuestionError):
                question(**{field: "  "})

    def test_bad_date_refused(self):
        with self.assertRaises(qs.QuestionError):
            question(reference_date="June 2018")

    def test_unknown_status_refused(self):
        with self.assertRaises(qs.QuestionError):
            question(status="probably_fine")

    def test_rejected_question_must_say_why(self):
        with self.assertRaises(qs.QuestionError):
            question(status="rejected")

    def test_validation_promotes_and_demotes(self):
        self.assertEqual(qs.validate(question()).status, "validated")
        failed = qs.validate(question(AD_anchor=False))
        self.assertEqual(failed.status, "rejected")
        self.assertIn("not_ad_anchored", failed.rejection_reason)

    def test_non_interrogative_is_caught(self):
        self.assertIn("not_interrogative",
                      qs.validation_failures(question(question="Donepezil use.")))

    def test_only_approved_statuses_are_usable(self):
        self.assertFalse(question().is_usable)
        self.assertTrue(question(status="approved").is_usable)


class DuplicateTests(unittest.TestCase):

    def test_exact_duplicate_found(self):
        dups = qs.find_duplicates([question(), question()])
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0][2], 1.0)

    def test_near_duplicate_found(self):
        near = question(
            question="Do cholinesterase inhibitors improve cognition in mild "
                     "Alzheimer disease patients?",
            reference_source="Another source",
        )
        self.assertTrue(qs.find_duplicates([question(), near], threshold=0.8))

    def test_distinct_questions_not_flagged(self):
        other = question(
            question="What imaging biomarkers support an amyloid diagnosis?",
            reference_source="Guideline X",
        )
        self.assertEqual(qs.find_duplicates([question(), other]), ())

    def test_review_export_has_blank_decision_columns(self):
        row = qs.review_export([question()])[0]
        self.assertEqual(row["reviewer_decision"], "")
        self.assertIn("automatic_failures", row)


def candidates(n=3):
    return tuple(
        fz.FrozenCandidate(f"E{i}", f"passage {i}", retrieval_rank=i,
                           rerank_rank=i, rerank_score=1.0 / i,
                           publication_date="2020-01")
        for i in range(1, n + 1)
    )


def frozen_item(**kw):
    base = dict(
        question_id="ADQ-1", question="q?", reference_answer="a",
        reference_source="src", reference_date="2020",
        candidates=candidates(),
    )
    base.update(kw)
    return fz.FrozenItem(**base)


class FreezingTests(unittest.TestCase):

    def test_hash_is_stable(self):
        self.assertEqual(frozen_item().candidate_set_hash,
                         frozen_item().candidate_set_hash)

    def test_hash_is_order_sensitive(self):
        reversed_item = frozen_item(candidates=tuple(reversed(candidates())))
        self.assertNotEqual(frozen_item().candidate_set_hash,
                            reversed_item.candidate_set_hash)

    def test_hash_changes_when_text_changes(self):
        altered = list(candidates())
        altered[0] = fz.FrozenCandidate("E1", "DIFFERENT", 1, rerank_rank=1)
        self.assertNotEqual(frozen_item().candidate_set_hash,
                            frozen_item(candidates=tuple(altered)).candidate_set_hash)

    def test_source_metadata_does_not_affect_hash(self):
        tagged = tuple(
            fz.FrozenCandidate(c.evidence_id, c.text, c.retrieval_rank,
                               rerank_rank=c.rerank_rank,
                               rerank_score=c.rerank_score,
                               publication_date=c.publication_date,
                               source_metadata={"note": "provenance only"})
            for c in candidates()
        )
        self.assertEqual(frozen_item().candidate_set_hash,
                         frozen_item(candidates=tagged).candidate_set_hash)

    def test_empty_or_duplicated_candidates_refused(self):
        with self.assertRaises(fz.FreezeError):
            frozen_item(candidates=())
        dup = candidates() + (candidates()[0],)
        with self.assertRaises(fz.FreezeError):
            frozen_item(candidates=dup)

    def test_identical_sets_pass(self):
        h = frozen_item().candidate_set_hash
        fz.assert_same_candidate_sets({"ADQ-1": h}, {"ADQ-1": h})

    def test_differing_hashes_fail_the_experiment(self):
        with self.assertRaises(fz.FreezeError) as ctx:
            fz.assert_same_candidate_sets({"ADQ-1": "aaa"}, {"ADQ-1": "bbb"})
        self.assertIn("differing_hashes", str(ctx.exception))

    def test_missing_item_fails(self):
        with self.assertRaises(fz.FreezeError):
            fz.assert_same_candidate_sets({"ADQ-1": "a", "ADQ-2": "b"},
                                          {"ADQ-1": "a"})

    def test_config_hash_detects_change(self):
        self.assertNotEqual(fz.config_hash({"temperature": 0.0}),
                            fz.config_hash({"temperature": 0.7}))
        self.assertEqual(fz.config_hash({"a": 1, "b": 2}),
                         fz.config_hash({"b": 2, "a": 1}))

    def test_firewall_blocks_reference_evidence_in_candidates(self):
        leaky = frozen_item(reference_evidence_ids=("E2",))
        self.assertEqual(leaky.firewall_violations(), ("E2",))
        with self.assertRaises(fz.FreezeError):
            fz.assert_firewall([leaky])

    def test_firewall_allows_independent_reference(self):
        clean = frozen_item(reference_evidence_ids=("REF-9",))
        self.assertEqual(clean.firewall_violations(), ())
        fz.assert_firewall([clean])

    def test_manifest_roundtrip_and_tamper_detection(self):
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "frozen.jsonl")
            digest = fz.write_manifest([frozen_item()], path)
            restored = fz.verify_manifest(path, digest)
            self.assertEqual(restored[0].candidate_set_hash,
                             frozen_item().candidate_set_hash)
            Path(path).write_text(
                Path(path).read_text(encoding="utf-8").replace("passage 1",
                                                               "tampered"),
                encoding="utf-8")
            with self.assertRaises(fz.FreezeError):
                fz.verify_manifest(path, digest)


class AnnotationTests(unittest.TestCase):

    def test_hallucinated_answer_needs_a_claim_and_subtype(self):
        with self.assertRaises(ann.AnnotationError):
            ann.Annotation("K", "m", hallucinated=1, n_hallucinated_claims=0,
                           subtype="temporal")
        with self.assertRaises(ann.AnnotationError):
            ann.Annotation("K", "m", hallucinated=1, n_hallucinated_claims=1)

    def test_clean_answer_cannot_carry_claims(self):
        with self.assertRaises(ann.AnnotationError):
            ann.Annotation("K", "m", hallucinated=0, n_hallucinated_claims=2)

    def test_abstention_cannot_be_hallucinated(self):
        with self.assertRaises(ann.AnnotationError):
            ann.Annotation("K", "m", hallucinated=1, n_hallucinated_claims=1,
                           subtype="other", abstained=1)

    def test_unknown_subtype_refused(self):
        with self.assertRaises(ann.AnnotationError):
            ann.Annotation("K", "m", 1, 1, subtype="vibes")

    def answers(self):
        return [
            {"run_id": "r1", "question_id": "ADQ-1", "system": "baseline",
             "question": "q?", "generated_answer": "A"},
            {"run_id": "r1", "question_id": "ADQ-1", "system": "proposed",
             "question": "q?", "generated_answer": "B"},
        ]

    def test_packet_hides_system_identity(self):
        packet, key = ann.build_blinded_packet(self.answers(), seed="s")
        for row in packet:
            self.assertNotIn("system", row)
            self.assertIn(row["shown_as"], ("System A", "System B"))
        self.assertEqual(len(key), 2)
        self.assertEqual({v["system"] for v in key.values()},
                         {"baseline", "proposed"})

    def test_blinding_is_reproducible_under_seed(self):
        a, _ = ann.build_blinded_packet(self.answers(), seed="s")
        b, _ = ann.build_blinded_packet(self.answers(), seed="s")
        c, _ = ann.build_blinded_packet(self.answers(), seed="different")
        self.assertEqual([r["answer_key"] for r in a],
                         [r["answer_key"] for r in b])
        self.assertEqual({r["answer_key"] for r in a},
                         {r["answer_key"] for r in c})

    def test_second_annotator_sample_is_reproducible(self):
        keys = [f"ANS-{i:03d}" for i in range(100)]
        first = ann.sample_for_second_annotator(keys, fraction=0.2, seed="x")
        self.assertEqual(len(first), 20)
        self.assertEqual(first,
                         ann.sample_for_second_annotator(keys, fraction=0.2,
                                                         seed="x"))

    def test_agreement_counts_only_shared_answers(self):
        a = [ann.Annotation("K1", "med", 1, 1, "temporal"),
             ann.Annotation("K2", "med", 0)]
        b = [ann.Annotation("K1", "doc", 1, 2, "factuality"),
             ann.Annotation("K2", "doc", 1, 1, "faithfulness"),
             ann.Annotation("K3", "doc", 0)]
        result = ann.agreement(a, b)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["raw_agreement"], 0.5)
        self.assertEqual(result["disagreements"], ["K2"])

    def test_no_overlap_reports_honestly(self):
        result = ann.agreement([ann.Annotation("K1", "med", 0)],
                               [ann.Annotation("K9", "doc", 0)])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["kappa"])


class StatsTests(unittest.TestCase):

    def test_mcnemar_uses_only_discordant_pairs(self):
        b = {"a": 1, "b": 1, "c": 0, "d": 1}
        p = {"a": 0, "b": 1, "c": 0, "d": 0}
        r = st.mcnemar(b, p)
        self.assertEqual((r.baseline_only, r.proposed_only), (2, 0))
        self.assertEqual((r.both, r.neither), (1, 1))
        self.assertEqual(r.discordant, 2)

    def test_no_discordant_pairs_gives_p_one(self):
        same = {"a": 1, "b": 0}
        self.assertEqual(st.mcnemar(same, dict(same)).p_value, 1.0)

    def test_mismatched_question_sets_refused(self):
        with self.assertRaises(st.StatsError):
            st.mcnemar({"a": 1}, {"b": 1})

    def test_large_consistent_difference_is_detected(self):
        b = {f"q{i}": 1 if i < 30 else 0 for i in range(100)}
        p = {f"q{i}": 1 if i < 15 else 0 for i in range(100)}
        self.assertLess(st.mcnemar(b, p).p_value, 0.01)

    def test_bootstrap_ci_brackets_the_point_estimate(self):
        b = {f"q{i}": 1.0 if i < 40 else 0.0 for i in range(100)}
        p = {f"q{i}": 1.0 if i < 20 else 0.0 for i in range(100)}
        ci = st.paired_bootstrap_ci(b, p, iterations=2000, seed="t")
        self.assertAlmostEqual(ci["difference"], -0.20, places=6)
        self.assertLessEqual(ci["ci_low"], ci["difference"])
        self.assertGreaterEqual(ci["ci_high"], ci["difference"])

    def test_har_reports_both_denominators_and_abstention(self):
        h = {"a": 0, "b": 1, "c": 0, "d": 0}
        abstain = {"a": 0, "b": 0, "c": 1, "d": 1}
        r = st.har(h, abstain)
        self.assertEqual(r["n_abstained"], 2)
        self.assertEqual(r["abstention_rate"], 0.5)
        self.assertEqual(r["har_conditional"], 0.5)   # 1 of 2 answered
        self.assertEqual(r["har_all_items"], 0.25)    # 1 of 4 items

    def test_abstaining_everything_does_not_look_perfect(self):
        h = {"a": 0, "b": 0}
        r = st.har(h, {"a": 1, "b": 1})
        self.assertIsNone(r["har_conditional"])
        self.assertEqual(r["abstention_rate"], 1.0)

    def test_crosstab_surfaces_proposed_only_failures(self):
        b = {"a": 1, "b": 0, "c": 1, "d": 0}
        p = {"a": 0, "b": 1, "c": 1, "d": 0}
        t = st.outcome_crosstab(b, p)
        self.assertEqual(t["baseline_only"], ["a"])
        self.assertEqual(t["proposed_only"], ["b"])
        self.assertEqual(t["both"], ["c"])
        self.assertEqual(t["neither"], ["d"])

    def test_subtype_distribution_sorted_by_frequency(self):
        d = st.subtype_distribution(["temporal", "temporal", "factuality", None])
        self.assertEqual(list(d), ["temporal", "factuality"])
        self.assertEqual(d["temporal"], 2)

    def test_holm_is_monotone_and_bounded(self):
        adj = st.holm({"har": 0.01, "accuracy": 0.04})
        self.assertGreaterEqual(adj["har"], 0.01)
        self.assertLessEqual(adj["accuracy"], 1.0)
        self.assertGreaterEqual(adj["accuracy"], adj["har"])


if __name__ == "__main__":
    unittest.main()
