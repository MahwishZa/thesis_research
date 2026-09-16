"""The reduced recency-aware admission policy, and the three-arm fairness controls.

`systems/` had no committed tests before this file: the earlier checks lived
in a scratch script and did not survive. These lock the properties the
primary experiment depends on.
"""

import unittest
from datetime import date

from systems.baseline import (
    HELPFUL,
    NOT_HELPFUL,
    MockRAG2Filter,
    NoFilterSystem,
    RAG2Config,
    RAG2System,
)
from systems.interfaces.evidence import Candidate, Evidence
from systems.interfaces.generator import GenerationResult, Generator
from systems.proposed import (
    AdmissionConfig,
    AdmissionScorer,
    OutputState,
    RecencyAwareAdmissionPolicy,
    RecencyAwareSystem,
    RecencyPolicy,
    RecencyState,
)


TQ = date(2026, 1, 1)


class EchoGenerator(Generator):
    """Records what the generator was actually given."""

    def __init__(self):
        self.calls = []

    def generate(self, question, evidence=(), *, prompt=None):
        self.calls.append((question, tuple(e.evidence_id for e in evidence),
                           prompt))
        return GenerationResult(text="|".join(e.evidence_id for e in evidence))


def evidence(eid, *, published=TQ, **kwargs):
    return Evidence(evidence_id=eid, text=f"text of {eid}",
                    source_tier="journal", publication_date=published,
                    **kwargs)


def candidate(eid, rank, **kwargs):
    return Candidate(evidence=evidence(eid, **kwargs),
                     rerank_score=1.0 / rank, rerank_rank=rank)


def policy(*, weight=0.5, threshold=0.5, half_life=365.0, limit=None,
           undated=0.5, contested=None, **recency_kwargs):
    return RecencyAwareAdmissionPolicy(
        scorer=AdmissionScorer(weight),
        recency=RecencyPolicy(half_life_days=half_life,
                              undated_score=undated, **recency_kwargs),
        config=AdmissionConfig(admit_threshold=threshold, question_date=TQ,
                               max_admitted_passages=limit),
        contested=contested,
    )


class RecencyScoreTests(unittest.TestCase):

    def setUp(self):
        self.recency = RecencyPolicy(half_life_days=365.0, undated_score=0.5)

    def score(self, published):
        return self.recency.score(evidence("e", published=published),
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
        self.assertIs(result.state, RecencyState.UNDATED)

    def test_undated_score_has_no_default(self):
        with self.assertRaises(ValueError):
            RecencyPolicy(half_life_days=365.0)

    def test_half_life_has_no_default(self):
        with self.assertRaises(ValueError):
            RecencyPolicy(half_life_days=None, undated_score=0.5).score(
                evidence("e"), question="q", question_date=TQ)


class SecondaryRulesAreOffByDefaultTests(unittest.TestCase):
    """The primary score is plain age decay and nothing else."""

    def test_no_secondary_rules_by_default(self):
        recency = RecencyPolicy(half_life_days=365.0, undated_score=0.5)
        self.assertEqual(recency.secondary_rules_enabled, ())

    def test_retracted_evidence_is_scored_by_age_unless_asked_otherwise(self):
        recency = RecencyPolicy(half_life_days=365.0, undated_score=0.5)
        result = recency.score(evidence("e", retracted=True), question="q",
                               question_date=TQ)
        self.assertIs(result.state, RecencyState.DATED)
        self.assertAlmostEqual(result.score, 1.0)

    def test_retraction_rule_is_opt_in_and_recorded(self):
        recency = RecencyPolicy(half_life_days=365.0, undated_score=0.5,
                                exclude_retracted=True)
        result = recency.score(evidence("e", retracted=True), question="q",
                               question_date=TQ)
        self.assertEqual(result.score, 0.0)
        self.assertIn("exclude_retracted", recency.secondary_rules_enabled)

    def test_supersession_is_opt_in_and_recorded(self):
        plain = RecencyPolicy(half_life_days=365.0, undated_score=0.5)
        item = evidence("e", published=date(2025, 1, 1),
                        supersession_pointer="other")
        self.assertIs(
            plain.score(item, question="q", question_date=TQ).state,
            RecencyState.DATED,
        )

        secondary = RecencyPolicy(half_life_days=365.0, undated_score=0.5,
                                  superseded_factor=0.5)
        result = secondary.score(item, question="q", question_date=TQ)
        self.assertIs(result.state, RecencyState.SUPERSEDED)
        self.assertIn("supersession_discount",
                      secondary.secondary_rules_enabled)

    def test_metadata_reports_no_secondary_rules_in_a_primary_run(self):
        system = RecencyAwareSystem(answer_generator=EchoGenerator(),
                                    admission_policy=policy())
        result = system.run(sample_id="s", experiment_id="e", question="q",
                            candidates=[candidate("a", 1)])
        self.assertEqual(result.metadata["recency_secondary_rules"], [])
        self.assertFalse(result.metadata["contested_detection_enabled"])


class ScoringRuleTests(unittest.TestCase):

    def test_single_weight_interpolates_between_the_two_signals(self):
        item = candidate("a", 1)
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(weight=weight):
                score = AdmissionScorer(weight).score(
                    item, recency=0.2, candidate_count=2)
                expected = (1 - weight) * score.relevance + weight * 0.2
                self.assertAlmostEqual(score.total, expected)

    def test_weight_zero_is_pure_relevance_the_built_in_ablation(self):
        score = AdmissionScorer(0.0).score(candidate("a", 1), recency=0.0,
                                           candidate_count=4)
        self.assertAlmostEqual(score.total, score.relevance)

    def test_score_stays_in_unit_interval(self):
        for rank in range(1, 6):
            for recency in (0.0, 0.5, 1.0):
                score = AdmissionScorer(0.5).score(
                    candidate("a", rank), recency=recency, candidate_count=5)
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

    def test_unresolved_config_raises(self):
        with self.assertRaises(ValueError):
            AdmissionConfig(question_date=TQ).validate()
        with self.assertRaises(ValueError):
            AdmissionConfig(admit_threshold=0.5).validate()


class AdmissionBehaviourTests(unittest.TestCase):

    def test_recency_changes_which_passage_is_admitted(self):
        # The whole point: with weight 0 the older, better-ranked passage
        # wins; with weight 1 the newer one does.
        candidates = [
            candidate("old", 1, published=date(2016, 1, 1)),
            candidate("new", 2, published=date(2025, 12, 1)),
        ]
        by_relevance = policy(weight=0.0, threshold=0.6, limit=1)
        by_recency = policy(weight=1.0, threshold=0.6, limit=1)

        admitted = lambda p: [d.candidate.evidence.evidence_id
                              for d in p.decide("q", candidates) if d.admitted]

        self.assertEqual(admitted(by_relevance), ["old"])
        self.assertEqual(admitted(by_recency), ["new"])

    def test_budget_is_never_exceeded(self):
        candidates = [candidate(f"e{i}", i + 1) for i in range(6)]
        decisions = policy(threshold=0.0, limit=2).decide("q", candidates)
        self.assertEqual(sum(d.admitted for d in decisions), 2)

    def test_budget_drops_are_recorded(self):
        candidates = [candidate(f"e{i}", i + 1) for i in range(4)]
        decisions = policy(threshold=0.0, limit=2).decide("q", candidates)
        self.assertEqual(sum(d.dropped_for_budget for d in decisions), 2)

    def test_selection_is_deterministic(self):
        candidates = [candidate(f"e{i}", i + 1) for i in range(5)]
        first = policy(threshold=0.0, limit=3).decide("q", candidates)
        second = policy(threshold=0.0, limit=3).decide("q", list(reversed(
            candidates)))
        self.assertEqual(
            sorted(d.candidate.evidence.evidence_id for d in first
                   if d.admitted),
            sorted(d.candidate.evidence.evidence_id for d in second
                   if d.admitted),
        )

    def test_rejected_passages_carry_no_state(self):
        decisions = policy(threshold=1.1).decide("q", [candidate("a", 1)])
        self.assertTrue(all(d.state is None for d in decisions))

    def test_no_admitted_evidence_abstains(self):
        system = RecencyAwareSystem(answer_generator=EchoGenerator(),
                                    admission_policy=policy(threshold=1.1))
        result = system.run(sample_id="s", experiment_id="e", question="q",
                            candidates=[candidate("a", 1)])
        self.assertEqual(result.output_state, OutputState.ABSTAIN.value)
        self.assertIsNone(result.prediction)

    def test_empty_candidate_set(self):
        self.assertEqual(policy().decide("q", []), ())


class ThreeArmFairnessTests(unittest.TestCase):
    """Controls that make the three-arm comparison interpretable."""

    def setUp(self):
        self.candidates = [candidate(f"e{i}", i + 1) for i in range(5)]
        self.budget = 3
        self.prompt = "Q: {question}\n\nEvidence:\n{context}"

    def arms(self):
        generators = {name: EchoGenerator()
                      for name in ("no_filter", "rag2", "recency")}
        return generators, [
            NoFilterSystem(answer_generator=generators["no_filter"],
                           max_admitted_passages=self.budget,
                           context_prompt=self.prompt),
            RAG2System(
                answer_generator=generators["rag2"],
                admission_filter=MockRAG2Filter(
                    {c.evidence.evidence_id: HELPFUL for c in self.candidates}
                ),
                config=RAG2Config(max_admitted_passages=self.budget,
                                  context_prompt=self.prompt),
            ),
            RecencyAwareSystem(
                answer_generator=generators["recency"],
                admission_policy=policy(threshold=0.0, limit=self.budget),
                context_prompt=self.prompt,
            ),
        ]

    def test_every_arm_gets_the_same_candidate_set_unmutated(self):
        before = [(c.evidence.evidence_id, c.rerank_rank)
                  for c in self.candidates]
        _, systems = self.arms()
        for system in systems:
            system.run(sample_id="s", experiment_id="e", question="q",
                       candidates=self.candidates)
        after = [(c.evidence.evidence_id, c.rerank_rank)
                 for c in self.candidates]
        self.assertEqual(before, after)

    def test_arms_have_distinct_variant_ids(self):
        _, systems = self.arms()
        names = [s.name for s in systems]
        self.assertEqual(len(names), len(set(names)))

    def test_no_arm_exceeds_the_shared_budget(self):
        _, systems = self.arms()
        for system in systems:
            result = system.run(sample_id="s", experiment_id="e", question="q",
                                candidates=self.candidates)
            self.assertLessEqual(len(result.admitted_evidence_ids),
                                 self.budget, system.name)

    def test_every_arm_reports_the_same_candidate_count(self):
        _, systems = self.arms()
        counts = {
            system.run(sample_id="s", experiment_id="e", question="q",
                       candidates=self.candidates).candidate_count
            for system in systems
        }
        self.assertEqual(counts, {5})

    def test_prompt_parity_across_arms(self):
        generators, systems = self.arms()
        for system in systems:
            system.run(sample_id="s", experiment_id="e", question="q",
                       candidates=self.candidates)
        # Same template, so every arm's prompt has the same shape and differs
        # only in which evidence was admitted.
        for generator in generators.values():
            prompt = generator.calls[0][2]
            self.assertTrue(prompt.startswith("Q: q"))
            self.assertIn("Evidence:", prompt)

    def test_no_arm_sees_the_reference_answer(self):
        generators, systems = self.arms()
        for system in systems:
            system.run(sample_id="s", experiment_id="e",
                       question="q", candidates=self.candidates)
        for name, generator in generators.items():
            question, evidence_ids, prompt = generator.calls[0]
            self.assertEqual(question, "q", name)
            self.assertNotIn("answer", prompt.lower(), name)

    def test_context_order_is_identical_across_arms(self):
        """Every arm presents surviving passages in candidate-list order.

        Regression test. The control arm used to emit rank-sorted context
        while the other two emitted candidate order, so the arms diverged
        whenever the cached candidate list was not already rank-ordered -
        which is what Stage 3 produces when it injects the evaluation pair
        into a retrieved set. Position in the context window affects the
        generator, so that is an admission-unrelated difference between arms.
        """
        # Candidate list deliberately NOT in rank order.
        shuffled = [
            candidate("e-c", 3), candidate("e-a", 1), candidate("e-d", 4),
            candidate("e-b", 2),
        ]
        expected = ["e-c", "e-a", "e-b"]  # candidate order, best 3 by rank

        generators, systems = self.arms()
        self.candidates = shuffled

        no_filter = NoFilterSystem(answer_generator=EchoGenerator(),
                                   max_admitted_passages=3)
        self.assertEqual(
            [c.evidence.evidence_id for c in no_filter.select(shuffled)],
            expected,
        )

        rag2 = RAG2System(
            answer_generator=EchoGenerator(),
            admission_filter=MockRAG2Filter(
                {c.evidence.evidence_id: HELPFUL for c in shuffled}),
            config=RAG2Config(max_admitted_passages=3),
        )
        self.assertEqual(
            list(rag2.run(sample_id="s", experiment_id="e", question="q",
                          candidates=shuffled).admitted_evidence_ids),
            expected,
        )

        recency = RecencyAwareSystem(
            answer_generator=EchoGenerator(),
            admission_policy=policy(threshold=0.0, limit=3, weight=0.0),
        )
        self.assertEqual(
            list(recency.run(sample_id="s", experiment_id="e", question="q",
                             candidates=shuffled).admitted_evidence_ids),
            expected,
        )

    def test_no_filter_admits_everything_within_budget(self):
        generators, systems = self.arms()
        result = systems[0].run(sample_id="s", experiment_id="e", question="q",
                                candidates=self.candidates)
        self.assertEqual(len(result.admitted_evidence_ids), self.budget)
        self.assertIsNone(result.output_state)
        self.assertEqual(result.metadata["filtering"], "none")

    def test_no_filter_without_a_budget_admits_all(self):
        system = NoFilterSystem(answer_generator=EchoGenerator())
        result = system.run(sample_id="s", experiment_id="e", question="q",
                            candidates=self.candidates)
        self.assertEqual(len(result.admitted_evidence_ids),
                         len(self.candidates))

    def test_rag2_filter_is_unchanged_by_the_proposed_arm(self):
        # The baseline still admits exactly its [HELPFUL] passages.
        generators = EchoGenerator()
        labels = {"e0": HELPFUL, "e1": NOT_HELPFUL, "e2": HELPFUL}
        result = RAG2System(
            answer_generator=generators,
            admission_filter=MockRAG2Filter(labels),
            config=RAG2Config(),
        ).run(sample_id="s", experiment_id="e", question="q",
              candidates=self.candidates)
        self.assertEqual(set(result.admitted_evidence_ids), {"e0", "e2"})


if __name__ == "__main__":
    unittest.main()
