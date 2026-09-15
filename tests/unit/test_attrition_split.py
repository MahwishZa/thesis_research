"""Attrition accounting, partitioning, Check A stratification, and sizing."""

import unittest

from experiments.test_pairs.scripts import attrition, check_a, power, split
from experiments.test_pairs.scripts.schema import (
    POOL_PRIMARY_EXTERNAL,
    EvidenceRef,
    Provenance,
    TestPair,
)


def pair(pair_id, question_id="Q-1", reasons=(), eligible=False,
         separation_days=None, newer_date="2024-03", change_point=None):
    return TestPair(
        pair_id=pair_id,
        question_id=question_id,
        question_text="Does X help?",
        provenance=Provenance(
            pool=POOL_PRIMARY_EXTERNAL,
            question_source="ExternalQA",
            answer_source="ExternalQA",
            evidence_source="corpus@test",
            extraction_date="2026-09-15",
        ),
        older=EvidenceRef(evidence_id=f"{pair_id}-old", document_id="D-1",
                          publication_date="2019-05"),
        newer=EvidenceRef(evidence_id=f"{pair_id}-new", document_id="D-2",
                          publication_date=newer_date),
        question_date="2024-03",
        reference_answer="No",
        change_point_date=change_point,
        exclusion_reasons=tuple(reasons),
        temporal_eligible=eligible,
        separation_days=separation_days,
    )


class AttritionTests(unittest.TestCase):

    def test_counts_and_yield(self):
        pairs = [
            pair("P-1", eligible=True),
            pair("P-2", reasons=("missing_question_date",)),
            pair("P-3", reasons=("missing_question_date",
                                 "ambiguous_temporal_order")),
        ]
        report = attrition.compute(pairs)

        self.assertEqual(report.candidates, 3)
        self.assertEqual(report.eligible, 1)
        self.assertEqual(report.excluded, 2)
        self.assertAlmostEqual(report.yield_rate, 1 / 3)

    def test_sole_reason_separates_recoverable_losses(self):
        pairs = [
            pair("P-1", reasons=("missing_question_date",)),
            pair("P-2", reasons=("missing_question_date",
                                 "ambiguous_temporal_order")),
        ]
        report = attrition.compute(pairs)

        # Both carry the reason ...
        self.assertEqual(report.pairs_with_reason["missing_question_date"], 2)
        # ... but fixing it alone only recovers one.
        self.assertEqual(report.sole_reason["missing_question_date"], 1)
        self.assertEqual(report.sole_reason.get("ambiguous_temporal_order", 0), 0)

    def test_empty_input(self):
        report = attrition.compute([])
        self.assertEqual(report.candidates, 0)
        self.assertEqual(report.yield_rate, 0.0)
        self.assertEqual(report.rows(), [])

    def test_separation_summary_uses_eligible_pairs_only(self):
        pairs = [
            pair("P-1", eligible=True, separation_days=100),
            pair("P-2", eligible=True, separation_days=300),
            pair("P-3", eligible=True, separation_days=500),
            pair("P-4", reasons=("missing_question_date",),
                 separation_days=9999),
        ]
        summary = attrition.separation_summary(pairs)
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["min_days"], 100)
        self.assertEqual(summary["median_days"], 300)
        self.assertEqual(summary["max_days"], 500)

    def test_separation_summary_empty(self):
        self.assertEqual(attrition.separation_summary([]), {"n": 0})


class SplitTests(unittest.TestCase):

    def setUp(self):
        self.config = split.SplitConfig(seed="unit-test")

    def test_assignment_is_deterministic(self):
        first = split.assign("Q-42", self.config)
        for _ in range(5):
            self.assertEqual(split.assign("Q-42", self.config), first)

    def test_seed_changes_assignment_distribution(self):
        other = split.SplitConfig(seed="different")
        questions = [f"Q-{i}" for i in range(200)]
        same = sum(
            split.assign(q, self.config) == split.assign(q, other)
            for q in questions
        )
        self.assertLess(same, len(questions))

    def test_all_pairs_of_a_question_stay_together(self):
        pairs = [
            pair("P-1", question_id="Q-1"),
            pair("P-2", question_id="Q-1"),
            pair("P-3", question_id="Q-2"),
        ]
        partitions = split.split_pairs(pairs, self.config)
        located = {
            name: {p.question_id for p in items}
            for name, items in partitions.items()
        }
        for question in ("Q-1", "Q-2"):
            holders = [n for n, qs in located.items() if question in qs]
            self.assertEqual(len(holders), 1, f"{question} in {holders}")

    def test_every_pair_is_assigned_exactly_once(self):
        pairs = [pair(f"P-{i}", question_id=f"Q-{i}") for i in range(50)]
        partitions = split.split_pairs(pairs, self.config)
        ids = [p.pair_id for items in partitions.values() for p in items]
        self.assertEqual(sorted(ids), sorted(p.pair_id for p in pairs))
        self.assertEqual(len(ids), len(set(ids)))

    def test_partitions_are_sorted(self):
        pairs = [pair("P-3", "Q-3"), pair("P-1", "Q-1"), pair("P-2", "Q-2")]
        for items in split.split_pairs(pairs, self.config).values():
            self.assertEqual([p.pair_id for p in items],
                             sorted(p.pair_id for p in items))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            split.SplitConfig(seed="").validate()
        with self.assertRaises(ValueError):
            split.SplitConfig(seed="s", dev=0.5, val=0.5, test=0.5).validate()

    def test_summarise_reports_realised_counts(self):
        pairs = [pair(f"P-{i}", question_id=f"Q-{i}", eligible=(i % 2 == 0))
                 for i in range(20)]
        summary = split.summarise(split.split_pairs(pairs, self.config))
        self.assertEqual(sum(s["pairs"] for s in summary.values()), 20)


class CheckATests(unittest.TestCase):

    def test_stratifies_by_cutoff(self):
        pairs = [
            pair("P-1", change_point="2022-06"),
            pair("P-2", change_point="2024-06"),
            pair("P-3", change_point="2025-01"),
        ]
        stratum = check_a.stratify(pairs, model="m", cutoff="2023-12")
        self.assertEqual(stratum.post_cutoff, 2)
        self.assertEqual(stratum.pre_cutoff, 1)
        self.assertAlmostEqual(stratum.post_cutoff_share, 2 / 3)

    def test_falls_back_to_newer_publication_date(self):
        pairs = [pair("P-1", change_point=None, newer_date="2025-02")]
        stratum = check_a.stratify(pairs, model="m", cutoff="2023-12")
        self.assertEqual(stratum.post_cutoff, 1)

    def test_undated_change_point_counted_separately(self):
        pairs = [pair("P-1", change_point=None, newer_date=None)]
        stratum = check_a.stratify(pairs, model="m", cutoff="2023-12")
        self.assertEqual(stratum.undated_change_point, 1)
        self.assertEqual(stratum.pairs_dated, 0)
        self.assertIsNone(stratum.post_cutoff_share)

    def test_bad_cutoff_rejected(self):
        with self.assertRaises(ValueError):
            check_a.stratify([], model="m", cutoff="soon")


class PowerTests(unittest.TestCase):

    def test_requires_measured_estimates(self):
        with self.assertRaises(ValueError):
            power.required_pairs(discordant_rate=None, older_share=0.7)
        with self.assertRaises(ValueError):
            power.required_pairs(discordant_rate=0.3, older_share=None)

    def test_null_effect_has_no_finite_sample_size(self):
        with self.assertRaises(ValueError):
            power.required_pairs(discordant_rate=0.3, older_share=0.5)

    def test_smaller_effect_needs_more_pairs(self):
        big = power.required_pairs(discordant_rate=0.3, older_share=0.80)
        small = power.required_pairs(discordant_rate=0.3, older_share=0.60)
        self.assertLess(big.total_pairs_needed, small.total_pairs_needed)

    def test_lower_discordant_rate_needs_more_pairs(self):
        high = power.required_pairs(discordant_rate=0.5, older_share=0.7)
        low = power.required_pairs(discordant_rate=0.1, older_share=0.7)
        self.assertLess(high.total_pairs_needed, low.total_pairs_needed)

    def test_yield_rate_scales_candidate_requirement(self):
        result = power.required_pairs(discordant_rate=0.4, older_share=0.7,
                                      yield_rate=0.25)
        self.assertEqual(result.candidates_needed,
                         result.total_pairs_needed * 4)

    def test_symmetric_in_direction_of_effect(self):
        older = power.required_pairs(discordant_rate=0.3, older_share=0.7)
        newer = power.required_pairs(discordant_rate=0.3, older_share=0.3)
        self.assertEqual(older.discordant_pairs_needed,
                         newer.discordant_pairs_needed)

    def test_unsupported_levels_rejected(self):
        with self.assertRaises(ValueError):
            power.required_pairs(discordant_rate=0.3, older_share=0.7,
                                 alpha=0.123)
        with self.assertRaises(ValueError):
            power.required_pairs(discordant_rate=0.3, older_share=0.7,
                                 power=0.42)


if __name__ == "__main__":
    unittest.main()
