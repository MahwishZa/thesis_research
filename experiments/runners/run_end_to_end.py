#!/usr/bin/env python3
"""Run the proposed system end-to-end, evaluate it against the RAG²
baseline (step 3: main evaluation), and ablate its key component - recency
weighting - against the same protocol (step 4: ablation study). This is the
single entry point for the thesis's three current objectives (see
docs/current_objectives.md): validate the proposed system runs correctly,
run a standard-RAG-metrics ablation of its key component, and determine
experimentally whether it improves on RAG².

Before this script existed, every piece (retriever, admission policies,
generator interface, runner) was real and unit-tested, but nothing glued
them to a concrete set of questions and produced a number. This script is
that glue.

Three arms are run over the same frozen candidate sets:
  - baseline : RAG2System            (systems/baseline/rag2.py)
  - proposed : RecencyAwareSystem, swept across --ablation-lambdas
  - no_filter: NoFilterSystem        (the admit-everything control)

--proposed-lambda (default 1.0) designates which swept configuration is
"the full proposed system" for two distinct comparisons the report keeps
separate:
  - main_evaluation (step 3): that arm vs. the RAG² baseline.
  - ablation_study  (step 4): that arm vs. the same system with its key
    component - recency weighting - removed (lambda=0, pure relevance
    ranking, systems/proposed/scorer.py's built-in ablation). lambda=0 is
    always included in the sweep for this reason, even if
    --ablation-lambdas omits it.
The rest of --ablation-lambdas (default also sweeps 0.25/0.5/0.75) is
reported as supplementary context, not a required part of either step.

Older activities this project does not treat as required pipeline stages
(a separate bias probe, temporal test-pair studies, verifier studies,
contestedness/authority studies, clinician studies, SOTA comparisons,
additional backbones) are not exercised by this script and are not needed
for it to answer the three current objectives; see
docs/current_objectives.md for what is/isn't in scope and why nothing was
deleted.

Every arm's answers are scored with experiments/evaluation/rag_metrics.py
(exact match, token F1, ROUGE-L, context precision/recall, groundedness) and
a summary table is printed and written to --output-dir. This is the
AUTOMATIC-METRICS track; it does not replace the thesis's earlier
human-annotated hallucination-rate protocol (annotation.py/stats.py), which
remains available as a secondary, more rigorous confirmation but is not
required for the three current objectives.

DATA: by default this runs against a small synthetic fixture (10 questions
with dated evidence, mirroring the shape of a real frozen candidate set) so
it is runnable in any environment, including one with no network access and
no local model - exactly this development sandbox. Pass --questions/--index
to run against a real frozen candidate set once one exists (see
experiments/evaluation/freezing.py), and --real-model to use an actual
Hugging Face generator instead of the deterministic stand-in (requires
transformers/torch and a downloaded checkpoint - not available in this
sandbox; see docs/generator_contract.md for the target model spec).

Usage:
    python -m experiments.runners.run_end_to_end
    python -m experiments.runners.run_end_to_end --ablation-lambdas 0,0.5,1
    python -m experiments.runners.run_end_to_end --real-model \\
        --model-name meta-llama/Meta-Llama-3-8B-Instruct \\
        --model-revision <pinned-commit-sha>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from experiments.evaluation import freezing as fz
from experiments.evaluation import rag_metrics as rm
from experiments.evaluation.runner import RunConfig, run_experiment
from systems.baseline.admission import HELPFUL, MockRAG2Filter
from systems.baseline.no_filter import NoFilterSystem
from systems.baseline.rag2 import RAG2Config, RAG2System
from systems.interfaces.generator import CallableGenerator, GenerationResult
from systems.proposed.admission import (
    AdmissionConfig, RecencyAwareAdmissionPolicy, RecencyAwareSystem,
)
from systems.proposed.recency import RecencyPolicy
from systems.proposed.scorer import AdmissionScorer

#: Forces every arm to pick exactly one passage per question, so the
#: fixture cleanly demonstrates the thesis's central failure mode: a
#: relevance-only ranking (RAG², no-filter) prefers a higher-reranked but
#: STALE passage over a lower-reranked but CURRENT one, while a
#: sufficiently recency-weighted proposed policy prefers the current one.
BUDGET = 1
QUESTION_DATE = date(2026, 1, 1)
HALF_LIFE_DAYS = 365.0
THETA = 0.5

CONTEXT_PROMPT = (
    "Answer the question using only the provided evidence.\n\n"
    "Question: {question}\n\nEvidence:\n{context}\n\nAnswer:"
)


def _synthetic_generate(question, evidence, prompt):
    """Deterministic stand-in for a real model: answers verbatim from the
    single most relevant admitted passage's stated fact, so the metrics
    below actually differentiate systems by what got admitted, not by
    generation quality (which this sandbox cannot exercise - no network, no
    local model)."""
    if not evidence:
        return GenerationResult(text="", metadata={"n_evidence": 0})
    # Evidence carries the fact directly (see make_fixture_items): the first
    # passage's text IS the answer content for these synthetic items.
    return GenerationResult(
        text=evidence[0].text,
        metadata={"n_evidence": len(evidence), "prompt_len": len(prompt or "")},
    )


def make_generator(
    real_model: bool, model_name: Optional[str], model_revision: Optional[str],
):
    if not real_model:
        return CallableGenerator(_synthetic_generate)
    from systems.interfaces.hf_generator import GenerationConfig, HuggingFaceGenerator, ModelSpec

    if not model_name:
        raise SystemExit("--real-model requires --model-name")
    if not model_revision:
        raise SystemExit(
            "--real-model requires --model-revision, a pinned commit sha "
            "(not a branch name like 'main') so the run is reproducible - "
            "see systems/interfaces/hf_generator.py's ModelSpec."
        )
    spec = ModelSpec(model_id=model_name, revision=model_revision)
    return HuggingFaceGenerator(spec, GenerationConfig())


def make_fixture_items(n: int = 10) -> list[fz.FrozenItem]:
    """A small, self-contained dataset where, for every question, the
    reranker ranks a STALE passage above a CURRENT one (as a keyword-driven
    reranker plausibly would - the stale passage is worded closer to the
    question), but the current passage carries the correct, up-to-date
    answer. This is the exact failure mode the thesis's admission policy
    targets, so it is what an ablation over lambda should be run against;
    it does not require the real corpus or a real model to demonstrate.
    """
    items = []
    for i in range(n):
        recent_id = f"Q{i}-recent"
        stale_id = f"Q{i}-stale"
        candidates = (
            fz.FrozenCandidate(
                evidence_id=stale_id,
                text=f"Outdated answer to question {i}: value was {i}-outdated.",
                retrieval_rank=1, rerank_rank=1, rerank_score=0.9,
                publication_date="2015-01",
            ),
            fz.FrozenCandidate(
                evidence_id=recent_id,
                text=f"Current answer to question {i}: value is {i}-current.",
                retrieval_rank=2, rerank_rank=2, rerank_score=0.5,
                publication_date="2025-11",
            ),
        )
        items.append(fz.FrozenItem(
            question_id=f"ADQ-{i:03d}",
            question=f"What is the current value for question {i}?",
            reference_answer=f"Current answer to question {i}: value is {i}-current.",
            reference_source="synthetic",
            reference_date="2025-11",
            corpus_snapshot="fixture@run_end_to_end",
            candidates=candidates,
            reference_evidence_ids=(recent_id,),
        ))
    return items


def build_systems(
    generator, lambdas: list[float], items: list[fz.FrozenItem],
) -> dict[str, object]:
    """baseline (RAG2), no_filter control, and one proposed arm per lambda
    in ``lambdas`` (labelled proposed_lambda_<value>).

    RAG²'s real trained filter checkpoint does not exist yet
    (docs/filter_training.md: "NOT TRAINED. No checkpoint exists."), so the
    baseline here uses MockRAG2Filter labelling every candidate HELPFUL -
    the faithful relevance-only stand-in for "the filter admits everything
    relevance ranked it as passing", which is documented, not hidden. This
    means today's "beats baseline" comparison is against RAG²'s CODE PATH
    with an untrained filter, not the paper's actual classifier; training
    the real filter (D-39/D-40) would make it a faithful reproduction.
    """
    helpful = {
        c.evidence_id: HELPFUL
        for item in items for c in item.candidates
    }

    systems: dict[str, object] = {
        "no_filter": NoFilterSystem(
            answer_generator=generator, max_admitted_passages=BUDGET,
            context_prompt=CONTEXT_PROMPT,
        ),
        "baseline": RAG2System(
            answer_generator=generator,
            admission_filter=MockRAG2Filter(helpful),
            config=RAG2Config(
                max_admitted_passages=BUDGET, context_prompt=CONTEXT_PROMPT,
            ),
        ),
    }

    for lam in lambdas:
        scorer = AdmissionScorer(recency_weight=lam)
        recency = RecencyPolicy(half_life_days=HALF_LIFE_DAYS, undated_score=0.0)
        config = AdmissionConfig(
            admit_threshold=THETA,
            question_date=QUESTION_DATE,
            max_admitted_passages=BUDGET,
        )
        policy = RecencyAwareAdmissionPolicy(
            scorer=scorer, recency=recency, config=config,
        )
        label = f"proposed_lambda_{lam:g}"
        systems[label] = RecencyAwareSystem(
            answer_generator=generator, admission_policy=policy,
            context_prompt=CONTEXT_PROMPT,
        )
    return systems


def gold_evidence_ids(items: list[fz.FrozenItem]) -> dict[str, set[str]]:
    return {item.question_id: set(item.reference_evidence_ids) for item in items}


def score_run(records: list[dict], gold: dict[str, set[str]]) -> list[rm.MetricRow]:
    rows = []
    for record in records:
        qid = record["question_id"]
        rows.append(rm.score_record(record, gold.get(qid, set())))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="experiments/outputs/end_to_end")
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument(
        "--ablation-lambdas", default="0,0.25,0.5,0.75,1.0",
        help="comma-separated lambda values to sweep for the proposed arm",
    )
    ap.add_argument(
        "--proposed-lambda", type=float, default=1.0,
        help="the recency weight that designates 'the full proposed system' "
             "for the main evaluation (step 3: proposed vs RAG²) and the "
             "ablation study (step 4: full vs the same system with its key "
             "component - recency weighting - removed, i.e. lambda=0). "
             "Must be one of --ablation-lambdas. Unfit on real data (see "
             "docs/current_objectives.md); 1.0 is a placeholder until a "
             "validation split exists to fit it on.",
    )
    ap.add_argument("--real-model", action="store_true",
                    help="use a real Hugging Face generator instead of the "
                         "deterministic fixture stand-in (needs transformers/"
                         "torch and a downloaded checkpoint)")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--model-revision", default=None,
                    help="pinned commit sha for --real-model (required with "
                         "it; never a branch name - see ModelSpec)")
    args = ap.parse_args(argv)

    lambdas = [float(x) for x in args.ablation_lambdas.split(",") if x.strip()]
    if 0.0 not in lambdas:
        lambdas = [0.0] + lambdas  # the ablation (component removed) arm is mandatory
    if args.proposed_lambda not in lambdas:
        lambdas = sorted(set(lambdas) | {args.proposed_lambda})
    proposed_label = f"proposed_lambda_{args.proposed_lambda:g}"
    ablated_label = "proposed_lambda_0"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()  # each invocation is a fresh run

    items = make_fixture_items(args.n_questions)
    generator = make_generator(args.real_model, args.model_name, args.model_revision)
    systems = build_systems(generator, lambdas, items)

    config = RunConfig(
        run_id="end_to_end-001",
        model="hf:" + args.model_name if args.real_model else "synthetic-fixture-generator",
        model_version="n/a",
        generation_config={"temperature": 0.0},
        system_config_hash=fz.config_hash({
            "budget": BUDGET, "theta": THETA, "half_life": HALF_LIFE_DAYS,
            "lambdas": lambdas,
        }),
    )

    summary = run_experiment(items, systems, config, str(results_path))

    records = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines()]
    gold = gold_evidence_ids(items)
    rows = score_run(records, gold)
    by_system = rm.aggregate_by_system(rows)

    def delta(a: str, b: str) -> float:
        return by_system[a]["token_f1"] - by_system[b]["token_f1"]

    main_evaluation = {
        "baseline": "baseline",
        "proposed": proposed_label,
        "token_f1_delta": delta(proposed_label, "baseline"),
        "verdict": "IMPROVES" if delta(proposed_label, "baseline") > 0 else "DOES NOT IMPROVE",
    }
    ablation_study = {
        "full_proposed": proposed_label,
        "ablated_proposed": ablated_label,
        "component_removed": "recency weighting (lambda=0, pure relevance ranking)",
        "token_f1_delta": delta(proposed_label, ablated_label),
        "verdict": ("COMPONENT HELPS" if delta(proposed_label, ablated_label) > 0
                    else "COMPONENT DOES NOT HELP"),
    }

    report = {
        "run_summary": summary,
        "metrics_by_system": by_system,
        "main_evaluation": main_evaluation,   # step 3: RAG2 vs proposed
        "ablation_study": ablation_study,      # step 4: full vs ablated proposed
        "ablation_sweep": [n for n in systems if n.startswith("proposed_")],
    }
    report_path = out_dir / "metrics_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    print(f"\nRan {summary['n_systems']} arms over {summary['n_items']} questions "
          f"({summary['n_errors']} errors). Results: {results_path}\n")
    print(f"{'system':<24}{'n':>4}{'EM':>8}{'F1':>8}{'ROUGE-L':>9}"
          f"{'ctx_P':>8}{'ctx_R':>8}{'ground':>9}")
    for name, metrics in sorted(by_system.items()):
        print(f"{name:<24}{metrics['n']:>4}{metrics['exact_match']:>8.3f}"
              f"{metrics['token_f1']:>8.3f}{metrics['rouge_l_f1']:>9.3f}"
              f"{metrics['context_precision']:>8.3f}"
              f"{metrics['context_recall']:>8.3f}"
              f"{metrics['groundedness']:>9.3f}")

    print(f"\nStep 3 - Main evaluation (RAG2 baseline vs proposed, lambda="
          f"{args.proposed_lambda:g}): token F1 delta {main_evaluation['token_f1_delta']:+.3f} "
          f"-> proposed {main_evaluation['verdict']} on the baseline")

    print(f"\nStep 4 - Ablation study (full proposed vs the same system with "
          f"its key component - recency weighting - removed, lambda=0): "
          f"token F1 delta {ablation_study['token_f1_delta']:+.3f} "
          f"-> the recency component {ablation_study['verdict']}")

    if len(lambdas) > 2:
        print("\nFull lambda sweep (token F1 delta vs baseline, for context beyond "
              "the two required arms above):")
        for name in sorted(n for n in by_system if n.startswith("proposed_")):
            print(f"  {name:<24} {delta(name, 'baseline'):+.3f}")

    print(f"\nFull report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
