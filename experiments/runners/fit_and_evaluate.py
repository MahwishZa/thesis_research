#!/usr/bin/env python3
"""Fit theta/half_life on the validation split, then report baseline vs.
proposed on the held-out test split — the step that was missing from
``run_real_evaluation.py``'s first pass (which used un-fit placeholder
values and degenerated at high lambda: theta=0.5 admitted almost nothing
once the temporal term dominated).

This is standard practice, not "forcing a result": three scalar values
(theta, half_life, lambda) are chosen on data the final number is never
computed from, exactly as docs/current_objectives.md and
experiments/questions/split.py always specified. If fitting finds nothing
that beats the baseline on validation, this script reports that honestly
and still evaluates the best-found configuration on test — it does not
search until something looks good and then stop.

Two-phase, real data both phases:

  Phase 1 (fit): retrieve real evidence once for the 23 validation
  questions. Sweep a grid of (theta, half_life, lambda) for the proposed
  system ONLY — no model calls are needed for this (TemporalFilterPolicy is
  pure arithmetic over already-retrieved, already-reranked candidates), so
  the sweep is fast and does not touch the RAG2 checkpoint or hit the
  question set the final number is reported on. Selects the config with the
  highest mean token_f1 on validation (ties broken by groundedness).

  Phase 2 (report): retrieve real evidence once for the 90 test questions
  (never used in phase 1). Run baseline / no_filter / proposed_lambda_0
  (mandatory ablation) / proposed_lambda_<fitted> with the fitted theta and
  half_life. This is the number that goes in the thesis.

Same documented limitations as run_real_evaluation.py apply here
unchanged: pilot-scale corpus (~1% of the full corpus), the RAG2 checkpoint
trained on 20 labels/1 epoch (validation_accuracy=0.0), and an extractive
stand-in in place of a real generative model. See that module's docstring
for the full account. This script does not change any of them; it only
adds the fitting step both prior runs were missing.

Usage:
    python -m experiments.runners.fit_and_evaluate --device cpu
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Optional

from experiments.evaluation import freezing as fz
from experiments.evaluation import rag_metrics as rm
from experiments.evaluation.runner import (
    RunConfig, assert_budget_parity, assert_generator_parity,
    assert_prompt_parity, run_experiment, to_candidates,
)
from experiments.runners.run_end_to_end import CONTEXT_PROMPT, build_systems
from experiments.runners.run_real_evaluation import (
    DEFAULT_BUDGET, DEFAULT_CHECKPOINT, DEFAULT_CORPUS, DEFAULT_INDEX,
    DEFAULT_QUESTIONS_DIR, admitted_recency_by_system, build_frozen_items,
    _extractive_answer, load_usable_questions, temporal_flags,
    temporal_subgroup_breakdown, gold_evidence_ids, score_run,
)
from systems.interfaces.generator import CallableGenerator

#: Kept deliberately small: TemporalFilterPolicy is pure arithmetic (no
#: model calls), so a much larger grid would still be fast, but a small,
#: pre-declared grid is easier to defend as "not tuned until something
#: looked good" than a suspiciously fine one chosen after seeing results.
THETA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
HALF_LIFE_GRID_DAYS = [30.0, 90.0, 180.0, 365.0, 730.0, 1825.0]
#: lambda=0 is the mandatory ablation arm, handled separately in Phase 2 -
#: not swept here, since fitting "the full system" over lambda=0 would be
#: fitting the ablation, not the thing being ablated against.
LAMBDA_GRID = [0.25, 0.5, 0.75, 1.0]


def _build_retrieval_pipeline(corpus: str, index: str, device: Optional[str]):
    from experiments.retrieval.corpus import dated_only, read_passages_with_snapshot
    from experiments.retrieval.encoders import MedCPTReranker, medcpt_query_encoder
    from experiments.retrieval.index import DenseIndex
    from experiments.retrieval.pipeline import RetrievalConfig, RetrievalPipeline

    passages, snapshot, _ = read_passages_with_snapshot(corpus, on_duplicate="keep_first")
    passages = dated_only(passages)
    idx = DenseIndex.load(index)
    pipeline = RetrievalPipeline(
        index=idx, passages=passages,
        query_encoder=medcpt_query_encoder(device=device),
        reranker=MedCPTReranker(device=device),
        config=RetrievalConfig(),
    )
    return pipeline, snapshot


def _run_in_memory(items, systems: dict[str, Any]) -> list[dict[str, Any]]:
    """Like runner.run_experiment, but returns records in memory instead of
    writing a JSONL file — used for the fitting sweep, where writing and
    re-reading a file hundreds of times would be pure overhead. Enforces
    the same parity checks as the real runner so a fitting run cannot
    silently compare arms configured differently."""
    assert_prompt_parity(systems)
    assert_budget_parity(systems)
    assert_generator_parity(systems)

    records = []
    for item in items:
        candidates = to_candidates(item)
        for name, system in systems.items():
            result = system.run(
                sample_id=item.question_id, experiment_id="fit-sweep",
                question=item.question, candidates=candidates,
            )
            by_id = {c.evidence_id: c.text for c in item.candidates}
            admitted_text = [by_id[e] for e in result.admitted_evidence_ids
                             if e in by_id]
            records.append({
                "question_id": item.question_id, "system": name,
                "reference_answer": item.reference_answer,
                "admitted_evidence_ids": list(result.admitted_evidence_ids),
                "admitted_evidence_text": admitted_text,
                "generated_answer": result.prediction,
            })
    return records


def fit_on_validation(
    val_items, generator, *, budget: int,
) -> dict[str, Any]:
    """Phase 1: grid-sweep theta/half_life/lambda on validation only.

    Returns the fitted config plus the full grid's scores, so the fitting
    process itself is auditable (not just the winning cell).
    """
    gold = gold_evidence_ids(val_items)
    grid_results = []
    best = None

    for theta, half_life, lam in itertools.product(
        THETA_GRID, HALF_LIFE_GRID_DAYS, LAMBDA_GRID,
    ):
        systems = build_systems(
            generator, [lam], rag2_filter=None, theta=theta,
            half_life=half_life, budget=budget,
        )
        # build_systems always includes "baseline" (needs a real filter) and
        # "no_filter" too; the sweep only needs the proposed arm - drop the
        # rest here rather than paying for a baseline filter call per grid
        # point (baseline is scored once, outside the grid, in main()).
        proposed_only = {k: v for k, v in systems.items() if k.startswith("proposed_")}
        records = _run_in_memory(val_items, proposed_only)
        rows = score_run(records, gold)
        agg = rm.aggregate_by_system(rows)
        label = f"proposed_lambda_{lam:g}"
        metrics = agg[label]
        grid_results.append({
            "theta": theta, "half_life_days": half_life, "lambda": lam,
            "token_f1": metrics["token_f1"], "groundedness": metrics["groundedness"],
        })
        key = (metrics["token_f1"], metrics["groundedness"])
        if best is None or key > best[0]:
            best = (key, theta, half_life, lam)

    _, theta, half_life, lam = best
    return {
        "fitted_theta": theta, "fitted_half_life_days": half_life,
        "fitted_lambda": lam,
        "fitted_validation_token_f1": best[0][0],
        "fitted_validation_groundedness": best[0][1],
        "grid": grid_results,
        "grid_size": len(grid_results),
    }


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions-dir", type=Path, default=DEFAULT_QUESTIONS_DIR)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--rag2-checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--output-dir", default="experiments/outputs/fit_and_evaluate")
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading validation and test question sets...")
    val_questions = load_usable_questions(args.questions_dir, "validation")
    test_questions = load_usable_questions(args.questions_dir, "test")
    print(f"  validation: {len(val_questions)}, test: {len(test_questions)}")

    print("Building retrieval pipeline...")
    pipeline, snapshot = _build_retrieval_pipeline(args.corpus, args.index, args.device)

    print(f"Retrieving real evidence for {len(val_questions)} validation questions...")
    val_items, val_skipped = build_frozen_items(val_questions, pipeline)
    print(f"  {len(val_items)} retrieved, {len(val_skipped)} skipped")

    print(f"Retrieving real evidence for {len(test_questions)} test questions...")
    test_items, test_skipped = build_frozen_items(test_questions, pipeline)
    print(f"  {len(test_items)} retrieved, {len(test_skipped)} skipped")

    generator = CallableGenerator(_extractive_answer)

    print(f"\nPhase 1: fitting theta/half_life/lambda on validation "
          f"({len(THETA_GRID)}x{len(HALF_LIFE_GRID_DAYS)}x{len(LAMBDA_GRID)} = "
          f"{len(THETA_GRID) * len(HALF_LIFE_GRID_DAYS) * len(LAMBDA_GRID)} configs, "
          "no model calls needed for this - pure arithmetic over already-"
          "retrieved candidates)...")
    fit = fit_on_validation(val_items, generator, budget=args.budget)
    print(f"  fitted: theta={fit['fitted_theta']}, "
          f"half_life={fit['fitted_half_life_days']}d, "
          f"lambda={fit['fitted_lambda']}")
    print(f"  validation token_f1 at this config: "
          f"{fit['fitted_validation_token_f1']:.3f}")

    print(f"\nLoading RAG2 filter checkpoint from {args.rag2_checkpoint} ...")
    from systems.baseline.admission import FlanT5RAG2Filter
    rag2_filter = FlanT5RAG2Filter(args.rag2_checkpoint, device=args.device)
    rag2_filter_label = (
        f"FlanT5RAG2Filter({args.rag2_checkpoint}) - trained on 20 MedQA "
        "labels/1 epoch, validation_accuracy=0.0 (at chance). NOT a "
        "validated classifier - see checkpoint_record.json."
    )

    print("\nPhase 2: final baseline-vs-proposed comparison on the HELD-OUT "
          f"test split ({len(test_items)} questions, never used for fitting)...")
    lambdas = sorted({0.0, fit["fitted_lambda"]})
    systems = build_systems(
        generator, lambdas, rag2_filter,
        theta=fit["fitted_theta"], half_life=fit["fitted_half_life_days"],
        budget=args.budget,
    )
    proposed_label = f"proposed_lambda_{fit['fitted_lambda']:g}"
    ablated_label = "proposed_lambda_0"

    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    config = RunConfig(
        run_id="fit_and_evaluate-001",
        model="extractive-stand-in (NOT a generative model - see "
              "run_real_evaluation.py's docstring)",
        model_version="n/a",
        generation_config={"temperature": 0.0},
        system_config_hash=fz.config_hash({
            "budget": args.budget, "theta": fit["fitted_theta"],
            "half_life": fit["fitted_half_life_days"], "lambdas": lambdas,
            "rag2_filter": rag2_filter_label, "corpus_snapshot": snapshot,
            "index": args.index, "fitted_on": "validation_split",
        }),
    )
    summary = run_experiment(test_items, systems, config, str(results_path))
    records = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines()]
    gold = gold_evidence_ids(test_items)
    rows = score_run(records, gold)
    by_system = rm.aggregate_by_system(rows)
    items_by_id = {i.question_id: i for i in test_items}
    recency = admitted_recency_by_system(records, items_by_id, temporal_flags(test_items))

    def delta(a: str, b: str) -> float:
        return by_system[a]["token_f1"] - by_system[b]["token_f1"]

    main_evaluation = {
        "baseline": "baseline", "baseline_filter": rag2_filter_label,
        "proposed": proposed_label,
        "baseline_token_f1": by_system["baseline"]["token_f1"],
        "proposed_token_f1": by_system[proposed_label]["token_f1"],
        "token_f1_delta": delta(proposed_label, "baseline"),
        "verdict": ("IMPROVES" if delta(proposed_label, "baseline") > 0
                    else "DOES NOT IMPROVE"),
    }
    ablation_study = {
        "full_proposed": proposed_label, "ablated_proposed": ablated_label,
        "token_f1_delta": delta(proposed_label, ablated_label),
        "verdict": ("COMPONENT HELPS" if delta(proposed_label, ablated_label) > 0
                    else "COMPONENT DOES NOT HELP"),
    }

    report = {
        "limitations": {
            "corpus": "pilot index, ~1% of the real 4,377,041-chunk corpus",
            "rag2_checkpoint": rag2_filter_label,
            "generator": "extractive stand-in, not free-text generation - "
                        "token_f1/rouge_l/groundedness measure evidence-"
                        "overlap with the reference answer, not generation "
                        "quality",
            "n_validation_questions": len(val_items),
            "n_test_questions": len(test_items),
        },
        "fitting": fit,
        "system_config": {
            "theta": fit["fitted_theta"],
            "half_life_days": fit["fitted_half_life_days"],
            "context_budget": args.budget,
            "proposed_lambda": fit["fitted_lambda"],
            "fitted_on": "validation split (n={}), reported on test split "
                        "(n={}), no overlap".format(len(val_items), len(test_items)),
        },
        "run_summary": summary,
        "metrics_by_system": by_system,
        "admitted_recency_by_system": recency,
        "temporal_subgroup": temporal_subgroup_breakdown(rows, temporal_flags(test_items)),
        "main_evaluation": main_evaluation,
        "ablation_study": ablation_study,
    }
    report_path = out_dir / "metrics_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    print(f"\n{'system':<24}{'n':>4}{'F1':>8}{'ROUGE-L':>9}{'ground':>9}")
    for name, metrics in sorted(by_system.items()):
        print(f"{name:<24}{metrics['n']:>4}{metrics['token_f1']:>8.3f}"
              f"{metrics['rouge_l_f1']:>9.3f}{metrics['groundedness']:>9.3f}")

    print(f"\nFitted on validation (n={len(val_items)}): theta="
          f"{fit['fitted_theta']}, half_life={fit['fitted_half_life_days']:g}d, "
          f"lambda={fit['fitted_lambda']}")
    print(f"Reported on held-out test (n={len(test_items)}):")
    print(f"  Main evaluation: baseline F1={main_evaluation['baseline_token_f1']:.3f} "
          f"vs proposed F1={main_evaluation['proposed_token_f1']:.3f} "
          f"(delta {main_evaluation['token_f1_delta']:+.3f}) -> "
          f"{main_evaluation['verdict']}")
    print(f"  Ablation: delta {ablation_study['token_f1_delta']:+.3f} -> "
          f"{ablation_study['verdict']}")
    print(f"\nFull report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
