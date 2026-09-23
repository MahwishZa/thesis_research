#!/usr/bin/env python3
"""Run the proposed system end-to-end, evaluate it against the RAG²
baseline (step 3: main evaluation), and ablate its key component -
temporal weighting - against the same protocol (step 4: ablation study). This is the
single entry point for the thesis's three current objectives (see
_archive/docs_legacy/current_objectives.md): validate the proposed system runs correctly,
run a standard-RAG-metrics ablation of its key component, and determine
experimentally whether it improves on RAG².

Before this script existed, every piece (retriever, admission policies,
generator interface, runner) was real and unit-tested, but nothing glued
them to a concrete set of questions and produced a number. This script is
that glue.

Three arms are run over the same frozen candidate sets:
  - baseline : RAG2System            (systems/baseline/rag2.py) - RAG²
  - proposed : TemporalFilterSystem, swept across --ablation-lambdas -
               RAG² + the Temporal Filter
  - no_filter: NoFilterSystem        (the admit-everything control)

--proposed-lambda (default 1.0) designates which swept configuration is
"the full proposed system" for two distinct comparisons the report keeps
separate:
  - main_evaluation (step 3): that arm vs. the RAG² baseline.
  - ablation_study  (step 4): that arm vs. the same system with its key
    component - temporal weighting - removed (lambda=0, pure relevance
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
_archive/docs_legacy/current_objectives.md for what is/isn't in scope and why nothing was
deleted.

Every arm's answers are scored with evaluation/rag_metrics.py
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
evaluation/freezing.py), and --real-model to use an actual
Hugging Face generator instead of the deterministic stand-in (requires
transformers/torch and a downloaded checkpoint - not available in this
sandbox; see _archive/docs_legacy/research_experimental_specification.md §9 for the target
model spec).

The RAG² baseline arm runs the real ``FlanT5RAG2Filter`` when
``--rag2-checkpoint`` supplies a trained checkpoint. Without one it runs a
documented all-HELPFUL stand-in, and both the console output and the
report's ``main_evaluation.baseline_filter`` /
``baseline_is_trained_rag2`` fields say which - a verdict against the
stand-in is a weaker claim than one against the paper's classifier, and
nothing should have to infer that from context.

Usage:
    python -m experiments.shared.runners.run_end_to_end
    python -m experiments.shared.runners.run_end_to_end --ablation-lambdas 0,0.5,1
    python -m experiments.shared.runners.run_end_to_end --real-model \\
        --model-revision <pinned-commit-sha> \\
        --rag2-checkpoint /path/to/trained/rag2-filter
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Optional

from evaluation import freezing as fz
from evaluation import rag_metrics as rm
from evaluation.runner import RunConfig, run_experiment
from src.baseline.admission import HELPFUL, MockRAG2Filter
from src.baseline.no_filter import NoFilterSystem
from src.baseline.rag2 import RAG2Config, RAG2System
from src.common.generator import CallableGenerator, GenerationResult
from src.common.hf_generator import RAG2_GENERATOR_ID
from src.proposed.admission import (
    AdmissionConfig, TemporalFilterPolicy, TemporalFilterSystem,
)
from src.proposed.temporal import TemporalPolicy
from src.proposed.scorer import AdmissionScorer

#: Forces every arm to pick exactly one passage per question, so the
#: fixture cleanly demonstrates the thesis's central failure mode: a
#: relevance-only ranking (RAG², no-filter) prefers a higher-reranked but
#: STALE passage over a lower-reranked but CURRENT one, while a
#: sufficiently temporal-weighted proposed policy prefers the current one.
#: FIXTURE defaults - not fitted values, and not the specification's
#: engineering constants. theta and the half-life must be fitted on the
#: validation split before any real run (specification S8.1), and the real
#: context budget is 5 (S8.2); 1 is used here only to force the
#: one-passage demonstration described above. All three are overridable
#: from the CLI (--theta/--half-life/--budget) so a real run does not have
#: to edit this file: AdmissionConfig.validate() deliberately refuses to
#: default theta, and a hard-coded placeholder here would defeat that.
DEFAULT_BUDGET = 1
QUESTION_DATE = date(2026, 1, 1)
DEFAULT_HALF_LIFE_DAYS = 365.0
DEFAULT_THETA = 0.5

#: 4-bit NF4 is what the generator contract specifies, and what makes an
#: 8B model fit the target GPU at all - a full-precision load is the one
#: configuration the documented hardware cannot run. "none" disables it.
DEFAULT_QUANTIZATION = "nf4"

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
    quantization: Optional[str] = DEFAULT_QUANTIZATION,
):
    if not real_model:
        return CallableGenerator(_synthetic_generate)
    from src.common.hf_generator import GenerationConfig, HuggingFaceGenerator, ModelSpec

    if not model_name:
        raise SystemExit("--real-model requires --model-name")
    if not model_revision:
        raise SystemExit(
            "--real-model requires --model-revision, a pinned commit sha "
            "(not a branch name like 'main') so the run is reproducible - "
            "see systems/interfaces/hf_generator.py's ModelSpec."
        )
    spec = ModelSpec(
        model_id=model_name,
        revision=model_revision,
        quantization=None if quantization == "none" else quantization,
    )
    return HuggingFaceGenerator(spec, GenerationConfig())


def make_rag2_filter(
    checkpoint: Optional[str], items: list[fz.FrozenItem],
    *, device: Optional[str] = None,
):
    """The RAG² baseline's admission filter, and a label saying which one.

    With ``--rag2-checkpoint`` this is the real ``FlanT5RAG2Filter`` over the
    supplied trained checkpoint. Without it, no trained checkpoint exists yet
    (the released weights are not distributed - see the filter-training
    contract), so the baseline necessarily runs on a stand-in that labels
    every candidate HELPFUL - i.e. RAG²'s code path with its relevance
    ordering and shared budget, but no learned filtering.

    The label is returned with the filter and stamped into the report
    because the difference matters for what the comparison means: a
    "proposed IMPROVES on baseline" verdict against the stand-in is not the
    same claim as one against the paper's classifier, and a reader of
    metrics_report.json must not have to guess which they are looking at.

    ``device`` defaults to ``FlanT5RAG2Filter``'s own auto-pick (cuda if
    available), which is right for an actual run but can contend with a GPU
    something else is already using during a readiness check - pass
    ``"cpu"`` for that case (see ``--rag2-device``).
    """
    if checkpoint:
        from src.baseline.admission import FlanT5RAG2Filter
        return (
            FlanT5RAG2Filter(checkpoint, device=device),
            f"FlanT5RAG2Filter({checkpoint})",
        )
    helpful = {
        c.evidence_id: HELPFUL
        for item in items for c in item.candidates
    }
    return (
        MockRAG2Filter(helpful),
        "MockRAG2Filter(all-HELPFUL stand-in; NO trained checkpoint)",
    )


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
    generator, lambdas: list[float], rag2_filter, *,
    theta: float = DEFAULT_THETA,
    half_life: float = DEFAULT_HALF_LIFE_DAYS,
    budget: int = DEFAULT_BUDGET,
    question_date: date = QUESTION_DATE,
) -> dict[str, object]:
    """baseline (RAG2), no_filter control, and one proposed arm per lambda
    in ``lambdas`` (labelled proposed_lambda_<value>).

    ``rag2_filter`` comes from make_rag2_filter() - the real trained
    checkpoint when one is supplied, the documented stand-in otherwise.

    ``question_date`` defaults to this module's fixture placeholder
    (2026-01-01) so existing fixture-based callers are unaffected; a caller
    scoring real, dated evidence against a real "as of" date (real-world
    passage ages, not ages relative to a demo constant) must pass its own.
    """
    systems: dict[str, object] = {
        "no_filter": NoFilterSystem(
            answer_generator=generator, max_admitted_passages=budget,
            context_prompt=CONTEXT_PROMPT,
        ),
        "baseline": RAG2System(
            answer_generator=generator,
            admission_filter=rag2_filter,
            config=RAG2Config(
                max_admitted_passages=budget, context_prompt=CONTEXT_PROMPT,
            ),
        ),
    }

    for lam in lambdas:
        scorer = AdmissionScorer(temporal_weight=lam)
        temporal = TemporalPolicy(half_life_days=half_life, undated_score=0.0)
        config = AdmissionConfig(
            admit_threshold=theta,
            question_date=question_date,
            max_admitted_passages=budget,
        )
        # Validate up front, so a bad theta/budget fails before any
        # generation rather than part-way through a run.
        config.validate()
        policy = TemporalFilterPolicy(
            scorer=scorer, temporal=temporal, config=config,
        )
        label = f"proposed_lambda_{lam:g}"
        systems[label] = TemporalFilterSystem(
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


def temporal_flags(items: list[fz.FrozenItem]) -> dict[str, bool]:
    return {item.question_id: item.temporal_candidate for item in items}


def temporal_subgroup_breakdown(
    rows: list[rm.MetricRow], flags: dict[str, bool],
) -> dict[str, Any]:
    """Aggregate metrics separately for temporal-candidate questions (a
    Cochrane review cited at .pub2+, so its conclusion has been revisited
    at least once - specification SS13) versus the rest.

    This is the direct, minimal test of whether the Temporal Filter's
    effect is concentrated where it should matter, rather than flat
    across the whole pool - and it costs nothing new to compute:
    ``temporal_candidate`` already exists on every ``EvaluationQuestion``
    and now survives freezing (``FrozenItem.temporal_candidate``); this
    only groups rows that are already scored.

    A real contrast needs both subgroups non-empty. On this module's own
    demo fixture every item is synthetic (not sourced from an actual
    Cochrane republication), so ``temporal_candidate`` is left at its
    default ``False`` there rather than set to an unearned ``True`` -
    the temporal-candidate group reports empty on the fixture, honestly,
    not populated to make the breakdown look exercised.
    """
    temporal_ids = {qid for qid, flag in flags.items() if flag}
    temporal_rows = [r for r in rows if r.question_id in temporal_ids]
    other_rows = [r for r in rows if r.question_id not in temporal_ids]
    # Counts come from the rows actually scored, not from len(flags): a
    # question absent from flags entirely still gets scored (as "other",
    # never dropped), and the two counts must always sum to the number of
    # distinct questions these rows cover.
    return {
        "n_temporal_candidate_questions": len({r.question_id for r in temporal_rows}),
        "n_other_questions": len({r.question_id for r in other_rows}),
        "temporal_candidate_questions": rm.aggregate_by_system(temporal_rows),
        "other_questions": rm.aggregate_by_system(other_rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="experiments/results/end_to_end")
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument(
        "--ablation-lambdas", default="0,0.25,0.5,0.75,1.0",
        help="comma-separated lambda values to sweep for the proposed arm",
    )
    ap.add_argument(
        "--proposed-lambda", type=float, default=1.0,
        help="the temporal weight (lambda) that designates 'the full "
             "proposed system' for the main evaluation (step 3: proposed vs "
             "RAG²) and the ablation study (step 4: full vs the same system "
             "with its key component - temporal weighting - removed, i.e. "
             "lambda=0). "
             "Must be one of --ablation-lambdas. Unfit on real data (see "
             "_archive/docs_legacy/current_objectives.md); 1.0 is a placeholder until a "
             "validation split exists to fit it on.",
    )
    ap.add_argument("--real-model", action="store_true",
                    help="use a real Hugging Face generator instead of the "
                         "deterministic fixture stand-in (needs transformers/"
                         "torch and a downloaded checkpoint)")
    ap.add_argument("--model-name", default=RAG2_GENERATOR_ID,
                    help="generator model id (default: the contract's "
                         f"{RAG2_GENERATOR_ID})")
    ap.add_argument("--model-revision", default=None,
                    help="pinned commit sha for --real-model (required with "
                         "it; never a branch name - see ModelSpec)")
    ap.add_argument("--quantization", default=DEFAULT_QUANTIZATION,
                    choices=("nf4", "int8", "none"),
                    help="generator quantization for --real-model "
                         f"(default: {DEFAULT_QUANTIZATION}, per the "
                         "generator contract)")
    ap.add_argument("--theta", type=float, default=DEFAULT_THETA,
                    help="admission threshold. MUST be a value fitted on "
                         "the validation split for a real run; the default "
                         f"({DEFAULT_THETA}) is a fixture placeholder, not "
                         "a fitted value. Must be in [0, 1].")
    ap.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE_DAYS,
                    help="temporal half-life H in days. Same status as "
                         f"--theta: the default ({DEFAULT_HALF_LIFE_DAYS:g}) "
                         "is a fixture placeholder. Must be positive.")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="shared context budget, applied identically to "
                         f"every arm. Default {DEFAULT_BUDGET} is the "
                         "fixture's one-passage demonstration; the "
                         "specification's value for a real run is 5.")
    ap.add_argument("--rag2-checkpoint", default=None,
                    help="path/id of a TRAINED RAG² filter checkpoint, to "
                         "run the baseline arm as the real classifier. "
                         "Without it the baseline runs on a documented "
                         "all-HELPFUL stand-in and the report says so.")
    ap.add_argument("--rag2-device", default=None,
                    help="device for --rag2-checkpoint (e.g. cpu, cuda). "
                         "Omit to auto-pick (cuda if available). Flan-T5 "
                         "filter inference is small enough to run on CPU - "
                         "pass 'cpu' to sanity-check a checkpoint without "
                         "touching a GPU something else is already using "
                         "(e.g. a concurrent index build).")
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
    generator = make_generator(args.real_model, args.model_name,
                               args.model_revision, args.quantization)
    rag2_filter, rag2_filter_label = make_rag2_filter(
        args.rag2_checkpoint, items, device=args.rag2_device
    )
    systems = build_systems(generator, lambdas, rag2_filter,
                            theta=args.theta, half_life=args.half_life,
                            budget=args.budget)

    config = RunConfig(
        run_id="end_to_end-001",
        model=("hf:" + args.model_name if args.real_model
               else "synthetic-fixture-generator"),
        # The pinned revision IS the reproducibility anchor - ModelSpec
        # refuses a branch name precisely so this field can identify the
        # exact weights. Recording "n/a" would throw that away.
        model_version=(args.model_revision or "n/a") if args.real_model else "n/a",
        generation_config={"temperature": 0.0},
        system_config_hash=fz.config_hash({
            "budget": args.budget, "theta": args.theta,
            "half_life": args.half_life,
            "lambdas": lambdas, "proposed_lambda": args.proposed_lambda,
            "quantization": args.quantization if args.real_model else None,
            "rag2_filter": rag2_filter_label,
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
        "baseline_filter": rag2_filter_label,
        "baseline_is_trained_rag2": bool(args.rag2_checkpoint),
        "proposed": proposed_label,
        "token_f1_delta": delta(proposed_label, "baseline"),
        "verdict": "IMPROVES" if delta(proposed_label, "baseline") > 0 else "DOES NOT IMPROVE",
    }
    ablation_study = {
        "full_proposed": proposed_label,
        "ablated_proposed": ablated_label,
        "component_removed": "temporal weighting (lambda=0, pure relevance ranking)",
        "token_f1_delta": delta(proposed_label, ablated_label),
        "verdict": ("COMPONENT HELPS" if delta(proposed_label, ablated_label) > 0
                    else "COMPONENT DOES NOT HELP"),
    }
    temporal_subgroup = temporal_subgroup_breakdown(rows, temporal_flags(items))

    report = {
        # The config hash detects a changed setting but cannot tell a
        # reader WHAT theta or the half-life was, and neither appears
        # anywhere else in the outputs. Recording them in readable form is
        # what lets a run be described (and reproduced) from its own
        # report rather than from whatever the CLI history happened to be.
        "system_config": {
            "theta": args.theta,
            "half_life_days": args.half_life,
            "context_budget": args.budget,
            "proposed_lambda": args.proposed_lambda,
            "ablation_lambdas": lambdas,
            "question_date": QUESTION_DATE.isoformat(),
            "theta_and_half_life_are_fitted": False,
            "fitted_note": (
                "theta and half_life are NOT fitted values unless a "
                "validation-split procedure set them; the CLI defaults are "
                "fixture placeholders."
            ),
        },
        "run_summary": summary,
        "metrics_by_system": by_system,
        "main_evaluation": main_evaluation,   # step 3: RAG2 vs proposed
        "ablation_study": ablation_study,      # step 4: full vs ablated proposed
        "ablation_sweep": [n for n in systems if n.startswith("proposed_")],
        # Does the Temporal Filter's effect concentrate on questions whose
        # evidence base has actually been revised over time, or is it flat
        # across the pool regardless of temporal_candidate? Zero new data
        # collection: temporal_candidate is already on every question.
        "temporal_subgroup": temporal_subgroup,
    }
    report_path = out_dir / "metrics_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    print(f"\nRan {summary['n_systems']} arms over {summary['n_items']} questions "
          f"({summary['n_errors']} errors). Results: {results_path}")
    print(f"RAG2 baseline filter: {rag2_filter_label}")
    if not args.rag2_checkpoint:
        print("  ^ NOT the paper's trained classifier. A verdict against this "
              "stand-in is a weaker claim;\n    pass --rag2-checkpoint once a "
              "trained checkpoint exists.")
    print()

    def fmt(value) -> str:
        """None means the metric was not applicable (no gold annotation),
        which must not print as 0.000."""
        return "     n/a" if value is None else f"{value:>8.3f}"

    print(f"{'system':<24}{'n':>4}{'EM':>8}{'F1':>8}{'ROUGE-L':>9}"
          f"{'ctx_P':>8}{'ctx_R':>8}{'ground':>9}")
    for name, metrics in sorted(by_system.items()):
        print(f"{name:<24}{metrics['n']:>4}{metrics['exact_match']:>8.3f}"
              f"{metrics['token_f1']:>8.3f}{metrics['rouge_l_f1']:>9.3f}"
              f"{fmt(metrics['context_precision'])}"
              f"{fmt(metrics['context_recall'])}"
              f"{metrics['groundedness']:>9.3f}")

    print(f"\nStep 3 - Main evaluation (RAG2 baseline vs proposed, lambda="
          f"{args.proposed_lambda:g}): token F1 delta {main_evaluation['token_f1_delta']:+.3f} "
          f"-> proposed {main_evaluation['verdict']} on the baseline")

    print(f"\nStep 4 - Ablation study (full proposed vs the same system with "
          f"its key component - temporal weighting - removed, lambda=0): "
          f"token F1 delta {ablation_study['token_f1_delta']:+.3f} "
          f"-> the Temporal Filter {ablation_study['verdict']}")

    if len(lambdas) > 2:
        print("\nFull lambda sweep (token F1 delta vs baseline, for context beyond "
              "the two required arms above):")
        for name in sorted(n for n in by_system if n.startswith("proposed_")):
            print(f"  {name:<24} {delta(name, 'baseline'):+.3f}")

    print(f"\nFull report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
