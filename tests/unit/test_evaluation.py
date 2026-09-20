"""Tests for the corpus-independent evaluation infrastructure."""

import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from experiments.evaluation import annotation as ann
from experiments.evaluation import freezing as fz
from experiments.evaluation import questions as qs
from experiments.evaluation import stats as st


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
        self.assertEqual(row["review_decision"], "")
        self.assertEqual(row["reviewer_id"], "")
        self.assertEqual(row["review_date"], "")

    def test_review_export_withholds_internal_classifications(self):
        row = qs.review_export([
            question(temporal_candidate=True, ambiguity_candidate=True)
        ])[0]
        for field in qs.WITHHELD_FROM_REVIEW:
            self.assertNotIn(field, row)
        self.assertEqual(set(row), set(qs.REVIEW_COLUMNS))


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
        corpus_snapshot="test-corpus@fixture",
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


class AbstentionAccountingTests(unittest.TestCase):
    """Abstention must never be a route to a low hallucination rate."""

    def setUp(self):
        self.q = [f"Q{i}" for i in range(10)]

    def test_abstention_is_counted_separately(self):
        h = {k: 0 for k in self.q}
        a = {k: 1 if i < 3 else 0 for i, k in enumerate(self.q)}
        cov = st.coverage(h, a)
        self.assertEqual(cov.abstained, 3)
        self.assertEqual(cov.answered, 7)
        self.assertAlmostEqual(cov.answer_coverage, 0.7)

    def test_abstention_cannot_also_be_hallucinated(self):
        with self.assertRaises(st.StatsError):
            st.coverage({"Q0": 1}, {"Q0": 1})

    def test_mismatched_records_refused(self):
        with self.assertRaises(st.StatsError):
            st.coverage({"Q0": 0}, {"Q1": 0})

    def test_har_denominators_are_explicit_and_differ(self):
        h = {k: 1 if i < 2 else 0 for i, k in enumerate(self.q)}
        a = {k: 1 if i >= 6 else 0 for i, k in enumerate(self.q)}
        r = st.har(h, a)
        self.assertEqual(r["n_answered"], 6)
        self.assertEqual(r["n_abstained"], 4)
        self.assertAlmostEqual(r["har_conditional"], 2 / 6, places=5)
        self.assertAlmostEqual(r["har_all_items"], 2 / 10)
        self.assertNotEqual(r["har_conditional"], r["har_all_items"])

    def test_abstaining_on_everything_is_not_interpretable(self):
        baseline_h = {k: 1 if i < 4 else 0 for i, k in enumerate(self.q)}
        baseline_a = {k: 0 for k in self.q}
        proposed_h = {k: 0 for k in self.q}
        proposed_a = {k: 1 for k in self.q}

        report = st.compare_systems(baseline_h, baseline_a,
                                    proposed_h, proposed_a)
        self.assertFalse(report["interpretable"])
        self.assertIn("warning", report)
        self.assertNotIn("delta_har_all_items", report)
        self.assertEqual(report["proposed"]["coverage"]["answer_coverage"], 0.0)
        self.assertTrue(report["proposed"]["coverage"]["is_degenerate"])

    def test_coverage_loss_is_always_visible_beside_the_rate(self):
        baseline_h = {k: 1 if i < 5 else 0 for i, k in enumerate(self.q)}
        baseline_a = {k: 0 for k in self.q}
        # Proposed abstains on the hard half and hallucinates on none.
        proposed_h = {k: 0 for k in self.q}
        proposed_a = {k: 1 if i < 5 else 0 for i, k in enumerate(self.q)}

        report = st.compare_systems(baseline_h, baseline_a,
                                    proposed_h, proposed_a, iterations=500)
        self.assertTrue(report["interpretable"])
        self.assertEqual(report["coverage_difference"], -0.5)
        self.assertEqual(report["proposed"]["coverage"]["abstained"], 5)

    def test_full_coverage_comparison_reports_effect_and_test(self):
        baseline_h = {k: 1 if i < 6 else 0 for i, k in enumerate(self.q)}
        proposed_h = {k: 1 if i < 2 else 0 for i, k in enumerate(self.q)}
        none_abstain = {k: 0 for k in self.q}
        report = st.compare_systems(baseline_h, none_abstain,
                                    proposed_h, none_abstain, iterations=500)
        self.assertTrue(report["interpretable"])
        self.assertEqual(report["coverage_difference"], 0.0)
        self.assertLess(report["delta_har_all_items"]["difference"], 0)
        self.assertIn("p_value", report["mcnemar_all_items"])


class AbstentionPolicyConfigTests(unittest.TestCase):
    """The policy is configurable and its default preserves cross-arm parity."""

    def test_default_is_answer_always(self):
        from systems.proposed.admission import AbstentionPolicy, AdmissionConfig
        import datetime
        cfg = AdmissionConfig(admit_threshold=0.5,
                              question_date=datetime.date(2024, 1, 1))
        self.assertIs(cfg.abstention_policy, AbstentionPolicy.ANSWER_ALWAYS)

    def test_abstention_remains_available_as_a_declared_condition(self):
        from systems.proposed.admission import AbstentionPolicy, AdmissionConfig
        import datetime
        cfg = AdmissionConfig(
            admit_threshold=0.5,
            question_date=datetime.date(2024, 1, 1),
            abstention_policy=AbstentionPolicy.ABSTAIN_WHEN_EMPTY,
        )
        self.assertIs(cfg.abstention_policy,
                      AbstentionPolicy.ABSTAIN_WHEN_EMPTY)

    def test_policy_is_recorded_in_run_metadata(self):
        from systems.proposed.admission import AbstentionPolicy
        self.assertEqual(AbstentionPolicy.ANSWER_ALWAYS.value, "answer_always")
        self.assertEqual(AbstentionPolicy.ABSTAIN_WHEN_EMPTY.value,
                         "abstain_when_empty")


class QuestionPoolTests(unittest.TestCase):

    def test_ambiguity_flag_is_recorded(self):
        q = question(ambiguity_candidate=True)
        self.assertTrue(q.to_dict()["ambiguity_candidate"])
        self.assertFalse(question().to_dict()["ambiguity_candidate"])

    def test_thin_reference_answer_is_flagged(self):
        self.assertIn("reference_answer_too_thin",
                      qs.validation_failures(question(reference_answer="Yes")))

    def test_no_expected_corpus_support_is_flagged(self):
        self.assertIn("no_corpus_support_expected",
                      qs.validation_failures(
                          question(corpus_support_expected=False)))

    def test_answerability_screen_flags_opinion_questions(self):
        self.assertTrue(qs.looks_indeterminate(
            "In your opinion should I prescribe donepezil?"))
        self.assertFalse(qs.looks_indeterminate(question().question))

    def test_pool_summary_counts_without_deciding_sample_size(self):
        pool = [question(), question(temporal_candidate=True,
                                     reference_source="Guideline 2024",
                                     ambiguity_candidate=True)]
        s = qs.summarise_pool(pool)
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["ad_anchored"], 2)
        self.assertEqual(s["temporal_candidates"], 1)
        self.assertEqual(s["ambiguity_candidates"], 1)
        self.assertEqual(s["distinct_sources"], 2)
        self.assertEqual(s["usable"], 0)  # nothing approved yet


class FreezeFromQuestionTests(unittest.TestCase):
    """Step 3: building a FrozenItem from an approved question."""

    def test_shared_fields_are_copied_from_the_question(self):
        q = question(status="approved")
        item = fz.from_question(
            q, candidates(), corpus_snapshot="alzheimer_corpus@fixture",
        )
        self.assertEqual(item.question_id, q.question_id)
        self.assertEqual(item.question, q.question)
        self.assertEqual(item.reference_answer, q.reference_answer)
        self.assertEqual(item.reference_source, q.reference_source)
        self.assertEqual(item.reference_date, q.reference_date)
        self.assertEqual(item.corpus_snapshot, "alzheimer_corpus@fixture")

    def test_temporal_candidate_is_carried_from_the_question(self):
        """Diagnostic-only metadata (specification SS13), but it has to
        survive freezing for run_end_to_end.py's temporal-subgroup
        breakdown to be able to use it at all."""
        temporal_item = fz.from_question(
            question(status="approved", temporal_candidate=True),
            candidates(), corpus_snapshot="c@v1",
        )
        plain_item = fz.from_question(
            question(status="approved", temporal_candidate=False),
            candidates(), corpus_snapshot="c@v1",
        )
        self.assertTrue(temporal_item.temporal_candidate)
        self.assertFalse(plain_item.temporal_candidate)

    def test_reference_evidence_ids_pass_through_to_the_firewall(self):
        q = question(status="approved")
        item = fz.from_question(
            q, candidates(), corpus_snapshot="c@v1",
            reference_evidence_ids=("E1",),
        )
        self.assertEqual(item.firewall_violations(), ("E1",))

    def test_corpus_snapshot_is_required(self):
        with self.assertRaises(fz.FreezeError):
            fz.FrozenItem(
                question_id="Q1", question="q?", reference_answer="a",
                reference_source="s", reference_date="2020",
                corpus_snapshot="",
                candidates=candidates(),
            )

    def test_corpus_snapshot_round_trips_through_the_manifest(self):
        item = frozen_item()
        with TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "m.jsonl")
            digest = fz.write_manifest([item], path)
            restored = fz.verify_manifest(path, digest)
        self.assertEqual(restored[0].corpus_snapshot, item.corpus_snapshot)


class RunnerResultsTests(unittest.TestCase):
    """Step 6: reading results back and reshaping for comparison."""

    def records(self):
        return [
            {"run_id": "r1", "question_id": "Q1", "system": "baseline",
             "generated_answer": "a"},
            {"run_id": "r1", "question_id": "Q1", "system": "proposed",
             "generated_answer": "b"},
            {"run_id": "r1", "question_id": "Q2", "system": "baseline",
             "generated_answer": "c"},
        ]

    def test_read_results_round_trips_jsonl(self):
        from experiments.evaluation.runner import read_results
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text("".join(
                json.dumps(r) + "\n" for r in self.records()))
            back = read_results(str(path))
        self.assertEqual(back, self.records())

    def test_group_by_system_reshapes_for_comparison(self):
        from experiments.evaluation.runner import group_by_system
        grouped = group_by_system(self.records())
        self.assertEqual(set(grouped), {"baseline", "proposed"})
        self.assertEqual(set(grouped["baseline"]), {"Q1", "Q2"})
        self.assertEqual(grouped["baseline"]["Q1"]["generated_answer"], "a")

    def test_duplicate_answer_for_one_question_is_refused(self):
        from experiments.evaluation.runner import RunnerError, group_by_system
        dup = self.records() + [
            {"run_id": "r1", "question_id": "Q1", "system": "baseline",
             "generated_answer": "second"},
        ]
        with self.assertRaises(RunnerError):
            group_by_system(dup)


class AnnotationImportTests(unittest.TestCase):
    """Step 7: reading annotations back and unblinding them."""

    def test_read_annotations_validates_each_row(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            good = ann.Annotation("K1", "med", 1, 1, "temporal")
            path.write_text(json.dumps(good.to_dict()) + "\n")
            [back] = ann.read_annotations(str(path))
        self.assertEqual(back, good)

    def test_read_annotations_rejects_an_incoherent_row(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            # hallucinated=1 with no subtype is invalid.
            path.write_text(json.dumps({
                "answer_key": "K1", "annotator_id": "med", "hallucinated": 1,
                "n_hallucinated_claims": 1, "subtype": None, "abstained": 0,
                "note": None,
            }) + "\n")
            with self.assertRaises(ann.AnnotationError):
                ann.read_annotations(str(path))

    def test_unblind_recovers_system_and_question(self):
        answers = [
            {"run_id": "r", "question_id": "Q1", "system": "baseline",
             "question": "q?", "generated_answer": "A"},
            {"run_id": "r", "question_id": "Q1", "system": "proposed",
             "question": "q?", "generated_answer": "B"},
        ]
        packet, key = ann.build_blinded_packet(answers, seed="s")
        annotations = [
            ann.Annotation(row["answer_key"], "med", 0)
            for row in packet
        ]
        unblinded = ann.unblind_annotations(annotations, key)
        self.assertEqual(set(unblinded), {"baseline", "proposed"})
        self.assertIn("Q1", unblinded["baseline"])
        self.assertIn("Q1", unblinded["proposed"])

    def test_unblind_refuses_an_unknown_answer_key(self):
        stray = ann.Annotation("ANS-not-in-key", "med", 0)
        with self.assertRaises(ann.AnnotationError):
            ann.unblind_annotations([stray], {})

    def test_unblind_refuses_a_duplicate_question(self):
        key = {
            "K1": {"system": "baseline", "question_id": "Q1", "run_id": "r"},
            "K2": {"system": "baseline", "question_id": "Q1", "run_id": "r"},
        }
        dup = [ann.Annotation("K1", "med", 0), ann.Annotation("K2", "doc", 0)]
        with self.assertRaises(ann.AnnotationError):
            ann.unblind_annotations(dup, key)


class HallucinationOutcomesTests(unittest.TestCase):
    """The glue between Step 7 annotations and Step 8 rate calculation."""

    def test_extracts_hallucinated_and_abstained_from_annotations(self):
        by_question = {
            "Q1": ann.Annotation("K1", "med", 1, 1, "temporal"),
            "Q2": ann.Annotation("K2", "med", 0, abstained=1),
            "Q3": ann.Annotation("K3", "med", 0),
        }
        hallucinated, abstained = st.hallucination_outcomes(by_question)
        self.assertEqual(hallucinated, {"Q1": 1, "Q2": 0, "Q3": 0})
        self.assertEqual(abstained, {"Q1": 0, "Q2": 1, "Q3": 0})

    def test_output_feeds_har_directly(self):
        by_question = {
            "Q1": ann.Annotation("K1", "med", 1, 1, "temporal"),
            "Q2": ann.Annotation("K2", "med", 0),
        }
        hallucinated, abstained = st.hallucination_outcomes(by_question)
        rates = st.har(hallucinated, abstained)
        self.assertEqual(rates["n_hallucinated"], 1)
        self.assertEqual(rates["har_all_items"], 0.5)


class ErrorAnalysisTests(unittest.TestCase):

    def test_subtype_breakdown_per_crosstab_cell(self):
        crosstab = {
            "baseline_only": ["Q1", "Q2"],
            "proposed_only": ["Q3"],
            "both": [],
            "neither": ["Q4"],
        }
        subtypes = {"Q1": "temporal", "Q2": "temporal", "Q3": "factuality"}
        report = st.error_analysis(crosstab, subtypes)
        self.assertEqual(report["baseline_only"], {"temporal": 2})
        self.assertEqual(report["proposed_only"], {"factuality": 1})
        self.assertEqual(report["neither"], {})

    def test_manufactures_no_conclusions_only_counts(self):
        crosstab = {"baseline_only": ["Q1"], "proposed_only": [],
                    "both": [], "neither": []}
        report = st.error_analysis(crosstab, {"Q1": "ambiguity"})
        self.assertEqual(report, {
            "baseline_only": {"ambiguity": 1},
            "proposed_only": {}, "both": {}, "neither": {},
        })


class QAAccuracyStatsTests(unittest.TestCase):

    def test_totals_and_accuracy(self):
        correct = {"Q1": 1, "Q2": 1, "Q3": 0, "Q4": 1}
        result = st.qa_accuracy(correct)
        self.assertEqual(result["n_evaluated"], 4)
        self.assertEqual(result["n_correct"], 3)
        self.assertEqual(result["n_incorrect"], 1)
        self.assertEqual(result["accuracy"], 0.75)

    def test_empty_input_refused(self):
        with self.assertRaises(st.StatsError):
            st.qa_accuracy({})

    def test_paired_comparison_reuses_mcnemar_and_bootstrap(self):
        """No new statistical test is needed: accuracy is paired-binary too."""
        baseline = {f"Q{i}": 1 if i < 5 else 0 for i in range(10)}
        proposed = {f"Q{i}": 1 if i < 8 else 0 for i in range(10)}
        result = st.mcnemar(baseline, proposed)
        self.assertEqual(result.proposed_only, 3)
        self.assertEqual(result.baseline_only, 0)
        ci = st.paired_bootstrap_ci(
            {k: float(v) for k, v in baseline.items()},
            {k: float(v) for k, v in proposed.items()},
            iterations=500,
        )
        self.assertAlmostEqual(ci["difference"], 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
