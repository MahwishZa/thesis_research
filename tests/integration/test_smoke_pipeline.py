"""Engineering smoke test over synthetic fixtures.

This is NOT a pilot and produces no scientific result. It checks that the
pieces connect: frozen manifest -> runner -> JSONL -> blinded packet ->
statistics, and that the guards actually fire.

It uses the real no-filter arm with two different context budgets, which
gives two arms that receive an identical candidate set but admit different
subsets - exactly the condition the experiment relies on.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.evaluation import annotation as ann
from experiments.evaluation import freezing as fz
from experiments.evaluation import stats as st
from experiments.evaluation.runner import (
    RunConfig, RunnerError, assert_prompt_parity, run_experiment, to_candidates,
)
from systems.baseline.admission import HELPFUL, NOT_HELPFUL, MockRAG2Filter
from systems.baseline.no_filter import NoFilterSystem
from systems.baseline.rag2 import RAG2Config, RAG2System
from systems.interfaces.generator import CallableGenerator, GenerationResult


def fake_generate(question, evidence, prompt):
    """Deterministic stand-in for a real model. No model is loaded here."""
    return GenerationResult(
        text=f"answer[{len(evidence)} passages]",
        metadata={"prompt_len": len(prompt or "")},
    )


def make_items(n=3, k=4):
    items = []
    for i in range(n):
        items.append(fz.FrozenItem(
            question_id=f"ADQ-{i:03d}",
            question=f"Synthetic Alzheimer question {i}?",
            reference_answer="synthetic reference",
            reference_source="synthetic source",
            reference_date="2024-01",
            corpus_snapshot="test-corpus@fixture",
            candidates=tuple(
                fz.FrozenCandidate(
                    evidence_id=f"E{i}-{j}", text=f"passage {i}-{j}",
                    retrieval_rank=j, rerank_rank=j, rerank_score=1.0 / j,
                    publication_date="2021-05",
                )
                for j in range(1, k + 1)
            ),
            reference_evidence_ids=(f"REF-{i}",),
        ))
    return items


#: One budget for every arm. The arms must differ by admission *rule*, never
#: by how much evidence they are allowed to admit - an arm given a bigger
#: budget answers from more context, which would confound the comparison.
BUDGET = 3


def make_systems():
    """Two arms: admit-everything, and admit-what-the-filter-labels-helpful.

    Same generator instance, same budget, same prompt; only the admission
    rule differs. That is the shape the real comparison has to take, so the
    smoke fixture takes it too.
    """
    gen = CallableGenerator(fake_generate)
    helpful = {
        f"E{i}-{j}": HELPFUL if j % 2 else NOT_HELPFUL
        for i in range(3) for j in range(1, 5)
    }
    return {
        "baseline": NoFilterSystem(answer_generator=gen,
                                   max_admitted_passages=BUDGET),
        "proposed": RAG2System(
            answer_generator=gen,
            admission_filter=MockRAG2Filter(helpful),
            config=RAG2Config(max_admitted_passages=BUDGET),
        ),
    }


def make_config():
    return RunConfig(
        run_id="smoke-001",
        model="synthetic-fixture-generator",
        model_version="n/a",
        generation_config={"temperature": 0.0},
        system_config_hash=fz.config_hash({"budget": BUDGET}),
    )


class SmokeTest(unittest.TestCase):

    def test_end_to_end_runs_and_serialises(self):
        items = make_items()
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "results.jsonl")
            summary = run_experiment(items, make_systems(), make_config(), out)

            self.assertEqual(summary["n_records"], 6)
            self.assertEqual(summary["n_errors"], 0)

            records = [json.loads(l) for l in Path(out).read_text().splitlines()]
            required = {
                "run_id", "question_id", "system", "question",
                "reference_answer", "candidate_evidence_ids",
                "candidate_set_hash", "admitted_evidence_ids",
                "admitted_evidence_text", "generated_answer", "model",
                "model_version", "generation_config", "system_config_hash",
                "timestamp", "status", "error",
            }
            for record in records:
                self.assertTrue(required.issubset(record))
                self.assertEqual(record["status"], "ok")
                self.assertIsNotNone(record["generated_answer"])

    def test_candidate_sets_identical_while_admitted_sets_differ(self):
        items = make_items()
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "r.jsonl")
            run_experiment(items, make_systems(), make_config(), out)
            records = [json.loads(l) for l in Path(out).read_text().splitlines()]

        by_system = {}
        for r in records:
            by_system.setdefault(r["system"], {})[r["question_id"]] = r

        for qid in by_system["baseline"]:
            self.assertEqual(
                by_system["baseline"][qid]["candidate_set_hash"],
                by_system["proposed"][qid]["candidate_set_hash"],
                "candidate sets must be identical across arms",
            )
        self.assertNotEqual(
            [len(r["admitted_evidence_ids"]) for r in by_system["baseline"].values()],
            [len(r["admitted_evidence_ids"]) for r in by_system["proposed"].values()],
            "the two arms should admit different subsets in this fixture",
        )

    def test_refuses_to_overwrite_raw_output(self):
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "r.jsonl")
            run_experiment(make_items(), make_systems(), make_config(), out)
            with self.assertRaises(RunnerError):
                run_experiment(make_items(), make_systems(), make_config(), out)

    def test_prompt_parity_break_is_caught_before_running(self):
        gen = CallableGenerator(fake_generate)
        mismatched = {
            "baseline": NoFilterSystem(answer_generator=gen),
            "proposed": NoFilterSystem(
                answer_generator=gen,
                context_prompt="Different: {question} {context}"),
        }
        with self.assertRaises(RunnerError) as ctx:
            assert_prompt_parity(mismatched)
        self.assertIn("prompt templates differ", str(ctx.exception))

    def test_single_arm_is_refused(self):
        gen = CallableGenerator(fake_generate)
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RunnerError):
                run_experiment(make_items(),
                               {"only": NoFilterSystem(answer_generator=gen)},
                               make_config(), str(Path(tmp) / "r.jsonl"))

    def test_firewall_blocks_a_leaky_manifest(self):
        leaky = make_items(n=1)
        bad = fz.FrozenItem(
            question_id=leaky[0].question_id, question=leaky[0].question,
            reference_answer="r", reference_source="s", reference_date="2024",
            corpus_snapshot="test-corpus@fixture",
            candidates=leaky[0].candidates,
            reference_evidence_ids=(leaky[0].candidates[0].evidence_id,),
        )
        with TemporaryDirectory() as tmp:
            with self.assertRaises(fz.FreezeError):
                fz.write_manifest([bad], str(Path(tmp) / "m.jsonl"))

    def test_candidate_conversion_preserves_order(self):
        item = make_items(n=1)[0]
        converted = to_candidates(item)
        self.assertEqual(
            [c.evidence.evidence_id for c in converted],
            list(item.candidate_evidence_ids),
        )

    def test_annotation_and_statistics_consume_the_output(self):
        items = make_items()
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "r.jsonl")
            run_experiment(items, make_systems(), make_config(), out)
            records = [json.loads(l) for l in Path(out).read_text().splitlines()]

            packet, key = ann.build_blinded_packet(records, seed="smoke")
            self.assertEqual(len(packet), 6)
            self.assertTrue(all("system" not in row for row in packet))

            # Synthetic labels: engineering check only, not a finding.
            labels = {}
            for akey, meta in key.items():
                labels.setdefault(meta["system"], {})[meta["question_id"]] = 0

            result = st.mcnemar(labels["baseline"], labels["proposed"])
            self.assertEqual(result.discordant, 0)
            self.assertEqual(result.p_value, 1.0)

            rates = st.har(labels["baseline"])
            self.assertEqual(rates["har_all_items"], 0.0)


if __name__ == "__main__":
    unittest.main()
