"""The Temporal Filter's recency term and admission score (stage 1, a result of record).

``src/temporal_filter/temporal.py`` computes T(s, q, t_q) = 2^(-age / H); ``scorer.py`` combines it with a rank-normalised
relevance, A(s) = (1 - lambda) * rho(s) + lambda * T(s). ``experiments/medchange/arms.py`` calls both.
"""

import unittest
from datetime import date

from src.common.evidence import Candidate, Evidence
from src.temporal_filter import AdmissionScorer, TemporalPolicy, TemporalState

TQ = date(2026, 1, 1)


def evidence(eid, *, published=TQ, **kwargs):
    return Evidence(evidence_id=eid, text=f"text of {eid}",
                    source_tier="journal", publication_date=published,
                    **kwargs)


def candidate(eid, rank, **kwargs):
    return Candidate(evidence=evidence(eid, **kwargs),
                     rerank_score=1.0 / rank, rerank_rank=rank)


class TemporalScoreTests(unittest.TestCase):

    def setUp(self):
        self.temporal = TemporalPolicy(half_life_days=365.0, undated_score=0.5)

    def score(self, published):
        return self.temporal.score(evidence("e", published=published),
                                   question="q", question_date=TQ)

    def test_same_day_scores_one(self):
        self.assertAlmostEqual(self.score(TQ).score, 1.0)

    def test_halves_over_one_half_life(self):
        self.assertAlmostEqual(self.score(date(2025, 1, 1)).score, 0.5,
                               places=3)
        self.assertAlmostEqual(self.score(date(2024, 1, 1)).score, 0.25,
                               places=2)

    def test_newer_always_scores_higher(self):
        older = self.score(date(2020, 6, 1)).score
        newer = self.score(date(2024, 6, 1)).score
        self.assertLess(older, newer)

    def test_future_dates_clamp_rather_than_exceed_one(self):
        self.assertAlmostEqual(self.score(date(2030, 1, 1)).score, 1.0)

    def test_score_is_always_in_unit_interval(self):
        for year in range(1990, 2027):
            value = self.score(date(year, 1, 1)).score
            self.assertGreater(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_undated_uses_the_configured_value_and_is_labelled(self):
        result = self.score(None)
        self.assertEqual(result.score, 0.5)
        self.assertIs(result.state, TemporalState.UNDATED)

    def test_undated_score_has_no_default(self):
        with self.assertRaises(ValueError):
            TemporalPolicy(half_life_days=365.0)

    def test_half_life_has_no_default(self):
        with self.assertRaises(ValueError):
            TemporalPolicy(half_life_days=None, undated_score=0.5).score(
                evidence("e"), question="q", question_date=TQ)


class SecondaryRulesAreOffByDefaultTests(unittest.TestCase):
    """The main score is plain age decay and nothing else."""

    def test_no_secondary_rules_by_default(self):
        temporal = TemporalPolicy(half_life_days=365.0, undated_score=0.5)
        self.assertEqual(temporal.secondary_rules_enabled, ())

    def test_retracted_evidence_is_scored_by_age_unless_asked_otherwise(self):
        temporal = TemporalPolicy(half_life_days=365.0, undated_score=0.5)
        result = temporal.score(evidence("e", retracted=True), question="q",
                                question_date=TQ)
        self.assertIs(result.state, TemporalState.DATED)
        self.assertAlmostEqual(result.score, 1.0)

    def test_retraction_rule_is_opt_in_and_recorded(self):
        temporal = TemporalPolicy(half_life_days=365.0, undated_score=0.5,
                                  exclude_retracted=True)
        result = temporal.score(evidence("e", retracted=True), question="q",
                                question_date=TQ)
        self.assertEqual(result.score, 0.0)
        self.assertIn("exclude_retracted", temporal.secondary_rules_enabled)

    def test_supersession_is_opt_in_and_recorded(self):
        plain = TemporalPolicy(half_life_days=365.0, undated_score=0.5)
        item = evidence("e", published=date(2025, 1, 1),
                        supersession_pointer="other")
        self.assertIs(
            plain.score(item, question="q", question_date=TQ).state,
            TemporalState.DATED,
        )

        secondary = TemporalPolicy(half_life_days=365.0, undated_score=0.5,
                                   superseded_factor=0.5)
        result = secondary.score(item, question="q", question_date=TQ)
        self.assertIs(result.state, TemporalState.SUPERSEDED)
        self.assertIn("supersession_discount",
                      secondary.secondary_rules_enabled)


class ScoringRuleTests(unittest.TestCase):

    def test_single_weight_interpolates_between_the_two_signals(self):
        item = candidate("a", 1)
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(weight=weight):
                score = AdmissionScorer(weight).score(
                    item, temporal=0.2, candidate_count=2)
                expected = (1 - weight) * score.relevance + weight * 0.2
                self.assertAlmostEqual(score.total, expected)

    def test_weight_zero_is_pure_relevance_the_built_in_ablation(self):
        score = AdmissionScorer(0.0).score(candidate("a", 1), temporal=0.0,
                                           candidate_count=4)
        self.assertAlmostEqual(score.total, score.relevance)

    def test_score_stays_in_unit_interval(self):
        for rank in range(1, 6):
            for temporal in (0.0, 0.5, 1.0):
                score = AdmissionScorer(0.5).score(
                    candidate("a", rank), temporal=temporal, candidate_count=5)
                self.assertGreaterEqual(score.total, 0.0)
                self.assertLessEqual(score.total, 1.0)

    def test_rank_normalisation_endpoints(self):
        self.assertAlmostEqual(AdmissionScorer.normalize_rank(1, 5), 1.0)
        self.assertAlmostEqual(AdmissionScorer.normalize_rank(5, 5), 0.0)
        self.assertAlmostEqual(AdmissionScorer.normalize_rank(1, 1), 1.0)

    def test_weight_must_be_set_explicitly(self):
        with self.assertRaises(ValueError):
            AdmissionScorer(None)

    def test_weight_out_of_range_rejected(self):
        for bad in (-0.1, 1.1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                AdmissionScorer(bad)


if __name__ == "__main__":
    unittest.main()
