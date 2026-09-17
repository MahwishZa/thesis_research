"""The runner's cross-arm parity gates.

Three controls are declared throughout `systems/`: the arms must share one
context prompt, one context budget, and one generator. Prompt parity was
already enforced; budget and generator parity were documented but unchecked,
so an arm configured with a larger budget - or with its own generator - would
have produced a comparison confounded by something other than the admission
policy, with no sign of it in the output.

These tests lock the gates themselves, not the systems: each one builds arms
that differ in exactly one of the three controls and asserts the runner
refuses before any answer is written.
"""

import unittest
from datetime import date

from experiments.evaluation.runner import (
    RunnerError,
    assert_budget_parity,
    assert_generator_parity,
    context_budget,
)
from systems.baseline.admission import MockRAG2Filter
from systems.baseline.no_filter import NoFilterSystem
from systems.baseline.rag2 import RAG2Config, RAG2System
from systems.interfaces.generator import CallableGenerator, GenerationResult
from systems.proposed.admission import (
    AdmissionConfig, RecencyAwareAdmissionPolicy, RecencyAwareSystem,
)
from systems.proposed.recency import RecencyPolicy
from systems.proposed.scorer import AdmissionScorer

TQ = date(2026, 1, 1)


def fake_generate(question, evidence, prompt):
    return GenerationResult(text="answer")


def no_filter(generator, budget):
    return NoFilterSystem(answer_generator=generator,
                          max_admitted_passages=budget)


def rag2(generator, budget):
    return RAG2System(
        answer_generator=generator,
        admission_filter=MockRAG2Filter({}),
        config=RAG2Config(max_admitted_passages=budget),
    )


def proposed(generator, budget):
    return RecencyAwareSystem(
        answer_generator=generator,
        admission_policy=RecencyAwareAdmissionPolicy(
            scorer=AdmissionScorer(recency_weight=0.5),
            recency=RecencyPolicy(half_life_days=365.0, undated_score=0.5),
            config=AdmissionConfig(admit_threshold=0.5, question_date=TQ,
                                   max_admitted_passages=budget),
        ),
    )


class ContextBudgetLookupTests(unittest.TestCase):
    """The budget lives in a different place in each arm."""

    def setUp(self):
        self.gen = CallableGenerator(fake_generate)

    def test_reads_it_off_the_system_itself(self):
        self.assertEqual(context_budget(no_filter(self.gen, 3)), 3)

    def test_reads_it_off_config(self):
        self.assertEqual(context_budget(rag2(self.gen, 3)), 3)

    def test_reads_it_off_the_admission_policy_config(self):
        self.assertEqual(context_budget(proposed(self.gen, 3)), 3)

    def test_unset_budget_reads_as_none_rather_than_a_number(self):
        """An uncapped arm must not be mistaken for one capped at some value."""
        self.assertIsNone(context_budget(no_filter(self.gen, None)))


class BudgetParityTests(unittest.TestCase):

    def setUp(self):
        self.gen = CallableGenerator(fake_generate)

    def test_matching_budgets_pass_across_all_three_arm_shapes(self):
        assert_budget_parity({
            "no_filter": no_filter(self.gen, 3),
            "baseline": rag2(self.gen, 3),
            "proposed": proposed(self.gen, 3),
        })

    def test_a_larger_budget_on_one_arm_is_refused(self):
        with self.assertRaises(RunnerError) as ctx:
            assert_budget_parity({"baseline": rag2(self.gen, 3),
                                  "proposed": proposed(self.gen, 5)})
        self.assertIn("context budgets differ", str(ctx.exception))

    def test_an_uncapped_arm_against_a_capped_one_is_refused(self):
        """The confound the smoke fixture used to encode."""
        with self.assertRaises(RunnerError):
            assert_budget_parity({"baseline": no_filter(self.gen, None),
                                  "proposed": rag2(self.gen, 2)})


class GeneratorParityTests(unittest.TestCase):

    def test_one_shared_instance_passes(self):
        gen = CallableGenerator(fake_generate)
        assert_generator_parity({"baseline": rag2(gen, 3),
                                 "proposed": proposed(gen, 3)})

    def test_two_instances_are_refused_even_when_behaviour_matches(self):
        """Identical behaviour is not the point.

        `RunConfig` stamps one model name and one generation config onto every
        record, so two generators would be recorded as if they were the same.
        Identity is what the `Generator` interface makes checkable.
        """
        with self.assertRaises(RunnerError) as ctx:
            assert_generator_parity({
                "baseline": rag2(CallableGenerator(fake_generate), 3),
                "proposed": proposed(CallableGenerator(fake_generate), 3),
            })
        self.assertIn("do not share one generator instance", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
