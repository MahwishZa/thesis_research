"""End-to-end check that the real baseline and real proposed system run
together through the shared runner (Steps 4, 5 and 6).

The earlier smoke test only exercised ``NoFilterSystem`` against itself. This
exercises ``RAG2System`` (the baseline, using ``MockRAG2Filter`` - never the
real Flan-T5 checkpoint, exactly as that class's own docstring requires for
anything that is not a thesis result) alongside ``RecencyAwareSystem`` (the
proposed solution), over one frozen candidate set, through the same
``run_experiment`` call.

This is engineering verification, not a scientific run: the generator is a
deterministic stand-in, the filter labels are hand-assigned, and nothing here
produces a number that could be mistaken for a result.
"""

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.evaluation import freezing as fz
from experiments.evaluation.runner import (
    RunConfig, group_by_system, read_results, run_experiment,
)
from systems.baseline.admission import MockRAG2Filter
from systems.baseline.rag2 import RAG2Config, RAG2System
from systems.interfaces.generator import CallableGenerator, GenerationResult
from systems.proposed.admission import (
    AdmissionConfig, RecencyAwareAdmissionPolicy, RecencyAwareSystem,
)
from systems.proposed.recency import RecencyPolicy
from systems.proposed.scorer import AdmissionScorer

TQ = date(2026, 1, 1)


def fake_generate(question, evidence, prompt):
    """Deterministic stand-in. No model is loaded anywhere in this test."""
    return GenerationResult(text=f"answer citing {len(evidence)} passages")


def make_items(n=3, k=4):
    items = []
    for i in range(n):
        candidates = tuple(
            fz.FrozenCandidate(
                evidence_id=f"E{i}-{j}", text=f"passage {i}-{j}",
                retrieval_rank=j, rerank_rank=j, rerank_score=1.0 / j,
                publication_date="2021-05",
            )
            for j in range(1, k + 1)
        )
        items.append(fz.FrozenItem(
            question_id=f"ADQ-{i:03d}",
            question=f"Synthetic Alzheimer question {i}?",
            reference_answer="synthetic reference",
            reference_source="synthetic source",
            reference_date="2024-01",
            corpus_snapshot="test-corpus@fixture",
            candidates=candidates,
        ))
    return items


def make_baseline(items, *, prompt):
    """RAG2System with a hand-labelled mock filter - never the real checkpoint."""
    labels = {}
    for item in items:
        for i, candidate in enumerate(item.candidates):
            labels[candidate.evidence_id] = "[HELPFUL]" if i % 2 == 0 else "[NOT_HELPFUL]"
    return RAG2System(
        answer_generator=CallableGenerator(fake_generate),
        admission_filter=MockRAG2Filter(labels),
        config=RAG2Config(context_prompt=prompt),
    )


def make_proposed(*, prompt):
    policy = RecencyAwareAdmissionPolicy(
        scorer=AdmissionScorer(recency_weight=0.5),
        recency=RecencyPolicy(half_life_days=365.0, undated_score=0.5),
        config=AdmissionConfig(admit_threshold=0.5, question_date=TQ),
    )
    return RecencyAwareSystem(
        answer_generator=CallableGenerator(fake_generate),
        admission_policy=policy,
        context_prompt=prompt,
    )


class BaselineAndProposedEndToEndTests(unittest.TestCase):

    def setUp(self):
        self.items = make_items()
        self.prompt = "Answer {question} using {context}"
        self.systems = {
            "baseline": make_baseline(self.items, prompt=self.prompt),
            "proposed": make_proposed(prompt=self.prompt),
        }
        self.config = RunConfig(
            run_id="e2e-fixture-001",
            model="fixture-generator",
            model_version="n/a",
            generation_config={"temperature": 0.0},
            system_config_hash=fz.config_hash({"fixture": True}),
        )

    def test_both_real_systems_run_over_the_same_frozen_set(self):
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "results.jsonl")
            summary = run_experiment(self.items, self.systems, self.config, out)

            self.assertEqual(summary["n_records"], len(self.items) * 2)
            self.assertEqual(summary["n_errors"], 0)

            records = [json.loads(l) for l in Path(out).read_text().splitlines()]
            for record in records:
                self.assertEqual(record["status"], "ok")
                self.assertIsNotNone(record["generated_answer"])

    def test_baseline_and_proposed_admit_different_subsets(self):
        """The systems differ only in admission rule - candidates stay identical."""
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "results.jsonl")
            run_experiment(self.items, self.systems, self.config, out)
            grouped = group_by_system(read_results(out))

        for qid in grouped["baseline"]:
            self.assertEqual(
                grouped["baseline"][qid]["candidate_set_hash"],
                grouped["proposed"][qid]["candidate_set_hash"],
            )
        admitted_counts = {
            name: [len(r["admitted_evidence_ids"]) for r in by_q.values()]
            for name, by_q in grouped.items()
        }
        self.assertNotEqual(admitted_counts["baseline"], admitted_counts["proposed"])

    def test_results_reshape_for_question_by_question_comparison(self):
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "results.jsonl")
            run_experiment(self.items, self.systems, self.config, out)
            grouped = group_by_system(read_results(out))

        self.assertEqual(set(grouped), {"baseline", "proposed"})
        self.assertEqual(set(grouped["baseline"]), {i.question_id for i in self.items})
        # Every field needed downstream is present with no manual copying.
        sample = grouped["baseline"][self.items[0].question_id]
        for field in ("run_id", "question_id", "system", "question",
                     "candidate_evidence_ids", "candidate_set_hash",
                     "admitted_evidence_ids", "generated_answer", "model",
                     "model_version", "generation_config",
                     "system_config_hash", "status", "error"):
            self.assertIn(field, sample)

    def test_a_failing_generator_is_recorded_not_silently_dropped(self):
        def broken_generate(question, evidence, prompt):
            raise RuntimeError("simulated generation failure")

        systems = {
            "baseline": make_baseline(self.items, prompt=self.prompt),
            "proposed": RecencyAwareSystem(
                answer_generator=CallableGenerator(broken_generate),
                admission_policy=RecencyAwareAdmissionPolicy(
                    scorer=AdmissionScorer(recency_weight=0.5),
                    recency=RecencyPolicy(half_life_days=365.0,
                                          undated_score=0.5),
                    config=AdmissionConfig(admit_threshold=0.5,
                                           question_date=TQ),
                ),
                context_prompt=self.prompt,
            ),
        }
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "results.jsonl")
            summary = run_experiment(self.items, systems, self.config, out)
            records = [json.loads(l) for l in Path(out).read_text().splitlines()]

        self.assertEqual(summary["n_errors"], len(self.items))
        failed = [r for r in records if r["system"] == "proposed"]
        for record in failed:
            self.assertEqual(record["status"], "error")
            self.assertIn("simulated generation failure", record["error"])
            self.assertIsNone(record["generated_answer"])

    def test_output_is_never_overwritten(self):
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "results.jsonl")
            run_experiment(self.items, self.systems, self.config, out)
            with self.assertRaises(Exception):
                run_experiment(self.items, self.systems, self.config, out)

    def test_reproducible_given_the_same_frozen_manifest(self):
        """Same items, same (deterministic) generator -> same admitted sets."""
        with TemporaryDirectory() as tmp:
            out_a = str(Path(tmp) / "a.jsonl")
            out_b = str(Path(tmp) / "b.jsonl")
            run_experiment(self.items,
                           {"baseline": make_baseline(self.items, prompt=self.prompt),
                            "proposed": make_proposed(prompt=self.prompt)},
                           self.config, out_a)
            run_experiment(self.items,
                           {"baseline": make_baseline(self.items, prompt=self.prompt),
                            "proposed": make_proposed(prompt=self.prompt)},
                           self.config, out_b)
            a = group_by_system(read_results(out_a))
            b = group_by_system(read_results(out_b))
        for system in ("baseline", "proposed"):
            for qid in a[system]:
                self.assertEqual(a[system][qid]["admitted_evidence_ids"],
                                 b[system][qid]["admitted_evidence_ids"])
                self.assertEqual(a[system][qid]["generated_answer"],
                                 b[system][qid]["generated_answer"])


if __name__ == "__main__":
    unittest.main()
