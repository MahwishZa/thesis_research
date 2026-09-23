"""The controlled validation pathway (Step F).

One test class walks the whole chain a real run will take - corpus → index →
retrieval → reranking → freeze → baseline + proposed → runner → JSONL - and
asserts, at each joint, the property the comparison depends on.

**This is software validation, not a pilot.** Every model is a deterministic
stand-in: a hashing encoder with no semantics, a lexical-overlap reranker, a
hand-labelled mock filter, and a generator that echoes how much evidence it
was given. Nothing here produces a number that could be read as a result, and
the fixtures are named so that a reader cannot mistake one for data.

What it proves is narrower and worth stating exactly: that the plumbing is
correct and the guards fire. Whether the proposed system reduces hallucination
is not knowable from any of this.
"""

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation import freezing as fz
from evaluation.questions import EvaluationQuestion
from evaluation.runner import (
    RunConfig, RunnerError, group_by_system, read_results, run_experiment,
)
from experiments.shared.retrieval.corpus import CorpusPassage
from experiments.shared.retrieval.encoders import HashingEncoder, LexicalOverlapReranker
from experiments.shared.retrieval.index import build_index
from experiments.shared.retrieval.pipeline import RetrievalConfig, RetrievalPipeline
from src.baseline.admission import HELPFUL, NOT_HELPFUL, MockRAG2Filter
from src.baseline.rag2 import RAG2Config, RAG2System
from src.common.generator import CallableGenerator, GenerationResult
from src.proposed.admission import (
    AdmissionConfig, TemporalFilterPolicy, TemporalFilterSystem,
)
from src.proposed.temporal import TemporalPolicy
from src.proposed.scorer import AdmissionScorer

TQ = date(2026, 1, 1)
BUDGET = 5
CANDIDATES = 20

#: Fitted-on-validation values do not exist yet. These are fixture values and
#: exist only so the policy can run; they are not a suggestion for the real
#: run, which fits them on the validation split.
FIXTURE_LAMBDA = 0.5
FIXTURE_THETA = 0.45
FIXTURE_HALF_LIFE = 730.0


def fixture_generate(question, evidence, prompt):
    """Echoes what it was given. Obviously not an answer."""
    return GenerationResult(
        text=f"FIXTURE ANSWER from {len(evidence)} passages",
        metadata={"prompt_chars": len(prompt or "")},
    )


def make_corpus(n=40):
    """Synthetic passages spread across years, so the temporal score can discriminate."""
    return tuple(
        CorpusPassage(
            chunk_id=f"FIXTURE-{i:03d}",
            document_id=f"DOC-{i // 4}",
            text=f"fixture passage {i} concerning donepezil and memantine",
            retrieval_text=f"fixture passage {i} donepezil memantine",
            publication_date=f"{2010 + (i % 15)}-06",
            source_tier="peer_reviewed",
            retracted=False,
        )
        for i in range(n)
    )


def make_question(i):
    return EvaluationQuestion(
        question_id=f"ADQ-{i:03d}",
        question=f"fixture question {i} about donepezil in Alzheimer's?",
        topic="treatment",
        reference_answer="fixture reference answer",
        reference_source="fixture source",
        reference_date="2024-01",
        AD_anchor=True,
        determinate=True,
        status="candidate",
    )


class ControlledValidationTests(unittest.TestCase):
    """Walks the pathway once in setUp; each test checks one property."""

    @classmethod
    def setUpClass(cls):
        passages = make_corpus()
        encoder = HashingEncoder(dim=32)
        index = build_index(passages, encoder,
                            corpus_snapshot="fixture-corpus@validation")
        pipeline = RetrievalPipeline(
            index=index, passages=passages, query_encoder=encoder,
            reranker=LexicalOverlapReranker(),
            config=RetrievalConfig(retrieval_depth=40,
                                   candidate_count=CANDIDATES),
        )

        cls.questions = [make_question(i) for i in range(4)]
        cls.items = []
        for question in cls.questions:
            retrieved = pipeline.build_candidate_set(
                question_id=question.question_id, question=question.question)
            cls.items.append(fz.from_question(
                question, retrieved.candidates,
                corpus_snapshot=retrieved.corpus_snapshot))

        # One generator object, shared. This is what a real run does.
        cls.generator = CallableGenerator(fixture_generate)
        cls.prompt = "Answer {question} using {context}"

        labels = {
            candidate.evidence_id: (HELPFUL if i % 2 == 0 else NOT_HELPFUL)
            for item in cls.items
            for i, candidate in enumerate(item.candidates)
        }
        cls.systems = {
            "baseline": RAG2System(
                answer_generator=cls.generator,
                admission_filter=MockRAG2Filter(labels),
                config=RAG2Config(context_prompt=cls.prompt,
                                  max_admitted_passages=BUDGET),
            ),
            "proposed": TemporalFilterSystem(
                answer_generator=cls.generator,
                admission_policy=TemporalFilterPolicy(
                    scorer=AdmissionScorer(temporal_weight=FIXTURE_LAMBDA),
                    temporal=TemporalPolicy(half_life_days=FIXTURE_HALF_LIFE,
                                            undated_score=0.5),
                    config=AdmissionConfig(admit_threshold=FIXTURE_THETA,
                                           question_date=TQ,
                                           max_admitted_passages=BUDGET),
                ),
                context_prompt=cls.prompt,
            ),
        }
        cls.config = RunConfig(
            run_id="controlled-validation-fixture",
            model="fixture-generator",
            model_version="n/a",
            generation_config={"do_sample": False, "max_new_tokens": 256},
            system_config_hash=fz.config_hash(
                {"budget": BUDGET, "candidates": CANDIDATES}),
        )

        cls._tmp = TemporaryDirectory()
        out = str(Path(cls._tmp.name) / "results.jsonl")
        cls.summary = run_experiment(cls.items, cls.systems, cls.config, out)
        cls.records = read_results(out)
        cls.grouped = group_by_system(cls.records)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # 1-2. One question in, the same question to both arms.
    def test_every_question_reaches_both_arms_exactly_once(self):
        self.assertEqual(set(self.grouped), {"baseline", "proposed"})
        expected = {q.question_id for q in self.questions}
        for arm in ("baseline", "proposed"):
            self.assertEqual(set(self.grouped[arm]), expected)
        self.assertEqual(self.summary["n_records"], len(self.questions) * 2)
        self.assertEqual(self.summary["n_errors"], 0)

    def test_both_arms_receive_identical_question_text(self):
        for qid in self.grouped["baseline"]:
            self.assertEqual(self.grouped["baseline"][qid]["question"],
                             self.grouped["proposed"][qid]["question"])

    # 3. The same candidate evidence set.
    def test_both_arms_receive_a_byte_identical_candidate_set(self):
        for qid in self.grouped["baseline"]:
            self.assertEqual(
                self.grouped["baseline"][qid]["candidate_set_hash"],
                self.grouped["proposed"][qid]["candidate_set_hash"])
            self.assertEqual(
                self.grouped["baseline"][qid]["candidate_evidence_ids"],
                self.grouped["proposed"][qid]["candidate_evidence_ids"])

    def test_every_question_has_the_same_candidate_set_size(self):
        """rho is a within-set rank, so theta needs constant N."""
        for item in self.items:
            self.assertEqual(len(item.candidates), CANDIDATES)

    # 4. Retrieval artifacts are reproducible.
    def test_rebuilding_the_index_reproduces_the_same_candidate_sets(self):
        passages = make_corpus()
        encoder = HashingEncoder(dim=32)
        pipeline = RetrievalPipeline(
            index=build_index(passages, encoder,
                              corpus_snapshot="fixture-corpus@validation"),
            passages=passages, query_encoder=encoder,
            reranker=LexicalOverlapReranker(),
            config=RetrievalConfig(retrieval_depth=40,
                                   candidate_count=CANDIDATES))
        for question, original in zip(self.questions, self.items):
            rebuilt = pipeline.build_candidate_set(
                question_id=question.question_id, question=question.question)
            item = fz.from_question(question, rebuilt.candidates,
                                    corpus_snapshot=rebuilt.corpus_snapshot)
            self.assertEqual(item.candidate_set_hash,
                             original.candidate_set_hash)

    # 5-7. Each arm applies its own rule; that difference is the factor.
    def test_the_arms_admit_different_subsets_of_one_shared_set(self):
        admitted = {
            arm: [tuple(r["admitted_evidence_ids"]) for r in by_q.values()]
            for arm, by_q in self.grouped.items()
        }
        self.assertNotEqual(admitted["baseline"], admitted["proposed"])

    def test_neither_arm_exceeds_the_shared_context_budget(self):
        for arm, by_q in self.grouped.items():
            for qid, record in by_q.items():
                self.assertLessEqual(
                    len(record["admitted_evidence_ids"]), BUDGET,
                    f"{arm}/{qid} admitted more than the budget")

    def test_admitted_evidence_is_always_drawn_from_the_frozen_set(self):
        """An arm inventing or re-retrieving evidence would break everything."""
        for item in self.items:
            allowed = set(item.candidate_evidence_ids)
            for arm in self.grouped:
                record = self.grouped[arm][item.question_id]
                self.assertTrue(set(record["admitted_evidence_ids"]) <= allowed)

    # 8-9. Same generator, same generation settings.
    def test_one_generator_instance_serves_both_arms(self):
        self.assertIs(self.systems["baseline"].answer_generator,
                      self.systems["proposed"].answer_generator)

    def test_every_record_carries_the_same_generation_config(self):
        configs = {json.dumps(r["generation_config"], sort_keys=True)
                   for r in self.records}
        self.assertEqual(len(configs), 1)
        self.assertEqual({r["model"] for r in self.records},
                         {"fixture-generator"})

    def test_decoding_is_recorded_as_greedy(self):
        for record in self.records:
            self.assertFalse(record["generation_config"]["do_sample"])

    # 10. Raw output is never overwritten.
    def test_a_second_run_cannot_overwrite_raw_output(self):
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "r.jsonl")
            run_experiment(self.items, self.systems, self.config, out)
            with self.assertRaises(RunnerError):
                run_experiment(self.items, self.systems, self.config, out)

    # 11-12. What generation saw, and what annotation will need.
    def test_the_evidence_supplied_to_generation_is_recorded(self):
        for record in self.records:
            self.assertEqual(len(record["admitted_evidence_text"]),
                             len(record["admitted_evidence_ids"]))

    def test_records_carry_everything_annotation_and_scoring_need(self):
        required = {
            "run_id", "question_id", "system", "question", "reference_answer",
            "candidate_evidence_ids", "candidate_set_hash",
            "admitted_evidence_ids", "admitted_evidence_text",
            "generated_answer", "output_state", "model", "model_version",
            "generation_config", "system_config_hash", "timestamp", "status",
            "error",
        }
        for record in self.records:
            self.assertTrue(required.issubset(record))
            self.assertEqual(record["status"], "ok")
            self.assertIsNotNone(record["generated_answer"])

    # 13. No test-set tuning.
    def test_the_policy_refuses_to_run_with_unfitted_parameters(self):
        """theta has no default that could be silently inherited."""
        with self.assertRaises(Exception):
            AdmissionConfig(question_date=TQ).validate()

    def test_the_proposed_arm_never_abstains_by_default(self):
        """The baseline cannot abstain, so parity requires answering."""
        for record in self.grouped["proposed"].values():
            self.assertNotEqual(record["output_state"], "ABSTAIN")


class ValidationIsNotAPilotTests(unittest.TestCase):
    """Guards against this fixture ever being read as data."""

    def test_the_stand_in_models_announce_that_they_are_stand_ins(self):
        self.assertIn("never", MockRAG2Filter.__doc__.lower())
        self.assertIn("never a thesis result",
                      HashingEncoder.__doc__.lower())
        self.assertIn("never a thesis result",
                      LexicalOverlapReranker.__doc__.lower())

    def test_fixture_answers_are_not_mistakable_for_answers(self):
        result = fixture_generate("q", [], "p")
        self.assertIn("FIXTURE", result.text)


if __name__ == "__main__":
    unittest.main()
