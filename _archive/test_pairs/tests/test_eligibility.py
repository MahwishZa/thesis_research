"""Stage-2 temporal eligibility rules."""

import unittest

from experiments.test_pairs.scripts.eligibility import (
    EligibilityConfig,
    evaluate,
    separation,
)
from experiments.test_pairs.scripts.schema import (
    POOL_PRIMARY_EXTERNAL,
    EvidenceRef,
    Provenance,
    TestPair,
)


def ref(eid="E", doc="D", date=None, **kwargs):
    return EvidenceRef(evidence_id=eid, document_id=doc,
                       publication_date=date, **kwargs)


def pair(older_date="2019-05", newer_date="2024-03", question_date="2024-03",
         older=None, newer=None, **kwargs):
    return TestPair(
        pair_id="P-1",
        question_id="Q-1",
        question_text="Does X help?",
        provenance=Provenance(
            pool=POOL_PRIMARY_EXTERNAL,
            question_source="ExternalQA",
            answer_source="ExternalQA",
            evidence_source="corpus@test",
            extraction_date="2026-09-15",
        ),
        older=older or ref("E-old", "D-1", older_date),
        newer=newer or ref("E-new", "D-2", newer_date),
        question_date=question_date,
        reference_answer="No",
        **kwargs,
    )


class SeparationTests(unittest.TestCase):

    def test_point_and_guaranteed_separation(self):
        point, guaranteed = separation(
            ref("a", "d", "2023-01"), ref("b", "d", "2024-01")
        )
        self.assertEqual(point, 365)
        # 2024-01-01 minus 2023-01-31
        self.assertEqual(guaranteed, 335)

    def test_missing_date_yields_none(self):
        self.assertEqual(separation(ref("a", "d", None), ref("b", "d", "2024")),
                         (None, None))


class TemporalEligibilityTests(unittest.TestCase):

    def setUp(self):
        self.config = EligibilityConfig()

    def test_well_dated_pair_is_eligible(self):
        result = evaluate(pair(), self.config)
        self.assertTrue(result.temporal_eligible)
        self.assertTrue(result.is_eligible)
        self.assertEqual(result.exclusion_reasons, ())
        self.assertGreater(result.separation_days, 0)

    def test_missing_publication_date_excludes(self):
        result = evaluate(pair(older_date=None), self.config)
        self.assertFalse(result.is_eligible)
        self.assertIn("missing_publication_date", result.exclusion_reasons)

    def test_missing_question_date_excludes(self):
        result = evaluate(pair(question_date=None), self.config)
        self.assertIn("missing_question_date", result.exclusion_reasons)

    def test_unparseable_question_date_excludes(self):
        result = evaluate(pair(question_date="whenever"), self.config)
        self.assertIn("missing_question_date", result.exclusion_reasons)

    def test_overlapping_year_dates_are_ambiguous(self):
        # Both sides dated only to the year: which is older is not established.
        result = evaluate(pair(older_date="2023", newer_date="2023"),
                          self.config)
        self.assertIn("ambiguous_temporal_order", result.exclusion_reasons)

    def test_coarse_dates_far_apart_remain_orderable(self):
        result = evaluate(pair(older_date="2019", newer_date="2024"),
                          self.config)
        self.assertTrue(result.is_eligible)

    def test_reversed_order_is_ambiguous(self):
        result = evaluate(pair(older_date="2024-03", newer_date="2019-05"),
                          self.config)
        self.assertIn("ambiguous_temporal_order", result.exclusion_reasons)

    def test_min_separation_enforced_only_when_configured(self):
        item = pair(older_date="2023-01", newer_date="2023-06")
        self.assertTrue(evaluate(item, EligibilityConfig()).is_eligible)

        strict = EligibilityConfig(min_separation_days=365)
        result = evaluate(item, strict)
        self.assertIn("insufficient_temporal_separation",
                      result.exclusion_reasons)

    def test_enforced_rules_reports_what_was_applied(self):
        self.assertNotIn("min_separation_days",
                         EligibilityConfig().enforced_rules())
        self.assertIn("min_separation_days",
                      EligibilityConfig(min_separation_days=30).enforced_rules())


class OtherRuleTests(unittest.TestCase):

    def test_retracted_side_excluded_by_default(self):
        item = pair(newer=ref("E-new", "D-2", "2024-03", retracted=True))
        result = evaluate(item, EligibilityConfig())
        self.assertIn("retracted_or_withdrawn", result.exclusion_reasons)

    def test_retraction_rule_can_be_disabled(self):
        item = pair(newer=ref("E-new", "D-2", "2024-03", withdrawn=True))
        result = evaluate(item, EligibilityConfig(exclude_retracted=False))
        self.assertNotIn("retracted_or_withdrawn", result.exclusion_reasons)

    def test_persistent_id_rule_off_by_default(self):
        self.assertTrue(evaluate(pair(), EligibilityConfig()).is_eligible)
        result = evaluate(pair(), EligibilityConfig(require_persistent_id=True))
        self.assertIn("missing_persistent_identifier", result.exclusion_reasons)

    def test_source_tier_match_rule(self):
        item = pair(
            older=ref("E-old", "D-1", "2019-05", source_tier="journal"),
            newer=ref("E-new", "D-2", "2024-03", source_tier="guideline"),
        )
        self.assertTrue(evaluate(item, EligibilityConfig()).is_eligible)
        result = evaluate(item, EligibilityConfig(require_same_source_tier=True))
        self.assertIn("source_tier_mismatch", result.exclusion_reasons)

    def test_length_ratio_rule(self):
        item = pair(
            older=ref("E-old", "D-1", "2019-05", length_tokens=10),
            newer=ref("E-new", "D-2", "2024-03", length_tokens=100),
        )
        result = evaluate(item, EligibilityConfig(max_length_ratio=3.0))
        self.assertIn("length_mismatch", result.exclusion_reasons)

    def test_prior_exclusions_are_preserved(self):
        item = pair(exclusion_reasons=("insufficient_contradiction",))
        result = evaluate(item, EligibilityConfig())
        self.assertIn("insufficient_contradiction", result.exclusion_reasons)
        self.assertFalse(result.is_eligible)

    def test_evaluation_is_deterministic(self):
        item, config = pair(), EligibilityConfig(min_separation_days=100)
        self.assertEqual(evaluate(item, config), evaluate(item, config))


if __name__ == "__main__":
    unittest.main()
