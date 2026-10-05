"""The Temporal Filter admission policy, and the three-arm fairness controls.

`src/` (then named `systems/`) had no committed tests before this file: the earlier checks lived
in a scratch script and did not survive. These lock the properties the
main experiment depends on.
"""

import unittest
from datetime import date

from _archive.alzheimers_framework.src.baseline import (
    HELPFUL,
    NOT_HELPFUL,
    MockRAG2Filter,
    NoFilterSystem,
    RAG2Config,
    RAG2System,
)
from src.common.evidence import Candidate, Evidence
from _archive.alzheimers_framework.src.common.generator import GenerationResult, Generator
from _archive.alzheimers_framework.src.temporal_filter import (
    AdmissionConfig,
    OutputState,
    TemporalFilterPolicy,
    TemporalFilterSystem,
)
from src.temporal_filter import AdmissionScorer, TemporalPolicy


TQ = date(2026, 1, 1)

#: Old enough that its temporal score is well below 1.0, which is how
#: a test builds the "nothing clears theta" condition using an
#: in-range theta.
STALE = date(2016, 1, 1)


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
           undated=0.5, **temporal_kwargs):
    return TemporalFilterPolicy(
        scorer=AdmissionScorer(weight),
        temporal=TemporalPolicy(half_life_days=half_life,
                                undated_score=undated, **temporal_kwargs),
        config=AdmissionConfig(admit_threshold=threshold, question_date=TQ,
                               max_admitted_passages=limit),
    )



class SecondaryRulesAreOffByDefaultTests(unittest.TestCase):
    """The main score is plain age decay and nothing else."""

    def test_metadata_reports_no_secondary_rules_in_a_main_run(self):
        system = TemporalFilterSystem(answer_generator=EchoGenerator(),
                                      admission_policy=policy())
        result = system.run(sample_id="s", experiment_id="e", question="q",
                            candidates=[candidate("a", 1)])
        self.assertEqual(result.metadata["temporal_secondary_rules"], [])


class ScoringRuleTests(unittest.TestCase):

    def test_unresolved_config_raises(self):
        with self.assertRaises(ValueError):
            AdmissionConfig(question_date=TQ).validate()
        with self.assertRaises(ValueError):
            AdmissionConfig(admit_threshold=0.5).validate()


class AdmissionBehaviourTests(unittest.TestCase):

    def test_temporal_score_changes_which_passage_is_admitted(self):
        # The whole point: with weight 0 the older, better-ranked passage
        # wins; with weight 1 the newer one does.
        candidates = [
            candidate("old", 1, published=date(2016, 1, 1)),
            candidate("new", 2, published=date(2025, 12, 1)),
        ]
        by_relevance = policy(weight=0.0, threshold=0.6, limit=1)
        by_temporal = policy(weight=1.0, threshold=0.6, limit=1)

        admitted = lambda p: [d.candidate.evidence.evidence_id
                              for d in p.decide("q", candidates) if d.admitted]

        self.assertEqual(admitted(by_relevance), ["old"])
        self.assertEqual(admitted(by_temporal), ["new"])

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
        # theta=1.0 with a STALE sole candidate under pure temporal scoring:
        # A < 1.0, so nothing is admitted. (A theta above 1.0 would be
        # simpler but is now rejected as degenerate - it can never admit
        # anything at all, whatever the evidence.)
        decisions = policy(weight=1.0, threshold=1.0).decide(
            "q", [candidate("a", 1, published=STALE)])
        self.assertTrue(all(d.state is None for d in decisions))
        self.assertTrue(all(not d.admitted for d in decisions))

    def test_no_admitted_evidence_answers_ungrounded_by_default(self):
        system = TemporalFilterSystem(answer_generator=EchoGenerator(),
                                      admission_policy=policy(weight=1.0,
                                                              threshold=1.0))
        result = system.run(sample_id="s", experiment_id="e", question="q",
                            candidates=[candidate("a", 1, published=STALE)])
        # Default policy is ANSWER_ALWAYS: the baseline generates from an
        # empty evidence block in this situation, so this arm must too, or
        # the hallucination rates are not comparable.
        self.assertEqual(result.output_state, OutputState.UNGROUNDED.value)
        self.assertIsNotNone(result.prediction)
        self.assertEqual(result.admitted_evidence_ids, ())

    def test_empty_candidate_set(self):
        self.assertEqual(policy().decide("q", []), ())


class RAG2ConfigValidationTests(unittest.TestCase):
    """RAG2Config must reject an invalid budget exactly like
    NoFilterSystem and AdmissionConfig already do for the same field -
    before this, a non-positive max_admitted_passages here silently
    degraded to "keep everything" instead of being rejected."""

    def test_rejects_zero_budget(self):
        with self.assertRaises(ValueError):
            RAG2Config(max_admitted_passages=0)

    def test_rejects_negative_budget(self):
        with self.assertRaises(ValueError):
            RAG2Config(max_admitted_passages=-1)

    def test_accepts_a_positive_budget(self):
        self.assertEqual(RAG2Config(max_admitted_passages=3).max_admitted_passages, 3)

    def test_accepts_no_budget(self):
        self.assertIsNone(RAG2Config().max_admitted_passages)


class ThreeArmFairnessTests(unittest.TestCase):
    """Controls that make the three-arm comparison interpretable."""

    def setUp(self):
        self.candidates = [candidate(f"e{i}", i + 1) for i in range(5)]
        self.budget = 3
        self.prompt = "Q: {question}\n\nEvidence:\n{context}"

    def arms(self):
        generators = {name: EchoGenerator()
                      for name in ("no_filter", "rag2", "temporal")}
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
            TemporalFilterSystem(
                answer_generator=generators["temporal"],
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

        temporal = TemporalFilterSystem(
            answer_generator=EchoGenerator(),
            admission_policy=policy(threshold=0.0, limit=3, weight=0.0),
        )
        self.assertEqual(
            list(temporal.run(sample_id="s", experiment_id="e", question="q",
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
