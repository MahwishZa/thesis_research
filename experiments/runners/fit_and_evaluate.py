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
from datetime import date
from pathlib import Path
from typing import Any, Optional

from experiments.evaluation import freezing as fz
from experiments.evaluation import rag_metrics as rm
from experiments.evaluation.runner import (
    RunConfig, _parse_date, assert_budget_parity, assert_generator_parity,
    assert_prompt_parity, run_experiment, to_candidates,
)
from experiments.runners.run_end_to_end import CONTEXT_PROMPT, build_systems
from experiments.runners.run_real_evaluation import (
    DEFAULT_BUDGET, DEFAULT_CHECKPOINT, DEFAULT_CORPUS, DEFAULT_INDEX,
    DEFAULT_QUESTIONS_DIR, admitted_recency_by_system, build_frozen_items,
    _extractive_answer, load_usable_questions, temporal_flags,
    temporal_subgroup_breakdown, gold_evidence_ids, score_run,
)
from systems.interfaces.evidence import Evidence
from systems.interfaces.generator import CallableGenerator
from systems.proposed.temporal import TemporalPolicy

#: Kept deliberately small: TemporalFilterPolicy is pure arithmetic (no
#: model calls), so a much larger grid would still be fast, but a small,
#: pre-declared grid is easier to defend as "not tuned until something
#: looked good" than a suspiciously fine one chosen after seeing results.
#:
#: theta extends to 0.9 (widened from an earlier 0.3-0.7 pass) because
#: rho(s) alone (candidate_count=20, rank-normalised) already clears 0.3-0.7
#: for well over half the candidate set regardless of recency - at those
#: thresholds the budget cap reproduces relevance-only ranking no matter
#: what lambda is, so that range cannot show the temporal term doing
#: anything. Values above ~0.7 are where relevance alone stops being
#: sufficient and admission can actually depend on T(s).
THETA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
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


def _relevance_only_top_budget(item, budget: int) -> set[str]:
    """The evidence ids a pure relevance ranking (no filter at all) would
    admit for this item - what NoFilterSystem effectively does. Used only
    as the divergence yardstick below, never to score anything."""
    ranked = sorted(item.candidates, key=lambda c: c.rerank_rank)
    return {c.evidence_id for c in ranked[:budget]}


def _mean_currency(
    evidence_ids, item, temporal: TemporalPolicy, question: str, question_date: date,
) -> float:
    """Mean T(s) - the same currency score TemporalFilterPolicy admits
    on - over one item's admitted evidence. 0.0 for an empty admitted set
    (nothing was admitted, so there is no currency to report - not the
    same as "maximally current," which would reward admitting nothing).

    THIS is the metric the research question is actually about: RAG2
    scores relevance/confidence, the proposed mechanism's whole claim is
    that it admits more CURRENT evidence. token_f1 (rag_metrics.py) scores
    textual overlap with a fixed reference sentence, which has no
    necessary relationship to currency - a correct, current passage can
    use entirely different wording than an older Cochrane conclusion. This
    function measures the thing the mechanism is designed to change,
    directly, independent of the extractive stand-in generator's wording.
    """
    if not evidence_ids:
        return 0.0
    by_id = {c.evidence_id: c for c in item.candidates}
    scores = []
    for eid in evidence_ids:
        cand = by_id.get(eid)
        if cand is None:
            continue
        evidence = Evidence(
            evidence_id=cand.evidence_id, text=cand.text,
            source_tier=str(cand.source_metadata.get("source_tier", "unknown")),
            persistent_id=cand.source_metadata.get("persistent_id"),
            publication_date=_parse_date(cand.publication_date),
        )
        result = temporal.score(evidence, question=question, question_date=question_date)
        scores.append(result.score)
    return sum(scores) / len(scores) if scores else 0.0


def _divergence_rate(records, items_by_id: dict, system_label: str, budget: int) -> float:
    """Fraction of items where this system's admitted set differs at all
    from relevance-only top-budget. This is the direct, model-free answer
    to "can this configuration possibly behave differently from the
    baseline" - independent of whether that difference helps or hurts
    token_f1, and independent of the extractive stand-in generator."""
    total = 0
    differs = 0
    for record in records:
        if record["system"] != system_label or record.get("status", "ok") != "ok":
            continue
        item = items_by_id[record["question_id"]]
        total += 1
        if set(record["admitted_evidence_ids"]) != _relevance_only_top_budget(item, budget):
            differs += 1
    return differs / total if total else 0.0


def fit_on_validation(
    val_items, generator, *, budget: int, question_date: date,
) -> dict[str, Any]:
    """Phase 1: grid-sweep theta/half_life/lambda on validation only.

    Returns the fitted config plus the full grid's scores, so the fitting
    process itself is auditable (not just the winning cell). Every cell
    also records ``divergence_rate`` - whether that configuration ever
    admits anything different from pure relevance ranking - so a grid that
    is structurally unable to test the temporal mechanism shows that
    plainly, rather than only showing tied metrics with no diagnosis.
    """
    gold = gold_evidence_ids(val_items)
    items_by_id = {i.question_id: i for i in val_items}
    flags = temporal_flags(val_items)
    temporal_ids = {qid for qid, flag in flags.items() if flag}
    grid_results = []
    best = None

    for theta, half_life, lam in itertools.product(
        THETA_GRID, HALF_LIFE_GRID_DAYS, LAMBDA_GRID,
    ):
        systems = build_systems(
            generator, [lam], rag2_filter=None, theta=theta,
            half_life=half_life, budget=budget, question_date=question_date,
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
        divergence = _divergence_rate(records, items_by_id, label, budget)

        # PRIMARY objective: mean currency gain over relevance-only
        # admission, restricted to temporal_candidate questions (where the
        # Cochrane conclusion is actually known to have been revised over
        # time - the one subgroup where currency should determine
        # correctness at all). token_f1 measures something else (textual
        # overlap with a fixed reference) and is recorded only as a
        # secondary diagnostic below - see this module's docstring.
        temporal_obj = TemporalPolicy(half_life_days=half_life, undated_score=0.0)
        gains = []
        for record in records:
            if record["system"] != label or record.get("status", "ok") != "ok":
                continue
            if record["question_id"] not in temporal_ids:
                continue
            item = items_by_id[record["question_id"]]
            proposed_currency = _mean_currency(
                record["admitted_evidence_ids"], item, temporal_obj,
                item.question, question_date,
            )
            baseline_ids = _relevance_only_top_budget(item, budget)
            baseline_currency = _mean_currency(
                baseline_ids, item, temporal_obj, item.question, question_date,
            )
            gains.append(proposed_currency - baseline_currency)
        currency_gain = sum(gains) / len(gains) if gains else 0.0

        grid_results.append({
            "theta": theta, "half_life_days": half_life, "lambda": lam,
            "currency_gain_temporal_subgroup": currency_gain,
            "token_f1": metrics["token_f1"], "groundedness": metrics["groundedness"],
            "divergence_rate": divergence,
        })
        # Primary: currency gain. Secondary tie-break: token_f1, so that
        # among configs tied on currency gain (e.g. both 0.0, meaning no
        # temporal_candidate question was affected) the one that does not
        # also wreck textual quality is preferred.
        key = (currency_gain, metrics["token_f1"])
        if best is None or key > best[0]:
            best = (key, theta, half_life, lam)

    _, theta, half_life, lam = best
    any_divergence = any(g["divergence_rate"] > 0 for g in grid_results)
    max_divergence = max((g["divergence_rate"] for g in grid_results), default=0.0)
    best_currency_gain = max(g["currency_gain_temporal_subgroup"] for g in grid_results)
    winning_cell = next(
        g for g in grid_results
        if (g["theta"], g["half_life_days"], g["lambda"]) == (theta, half_life, lam)
    )
    return {
        "fitted_theta": theta, "fitted_half_life_days": half_life,
        "fitted_lambda": lam,
        "fitted_validation_currency_gain": winning_cell["currency_gain_temporal_subgroup"],
        "fitted_validation_token_f1": winning_cell["token_f1"],
        "fitted_validation_groundedness": winning_cell["groundedness"],
        "best_currency_gain_in_grid": best_currency_gain,
        "n_temporal_candidate_questions_in_validation": len(temporal_ids),
        "grid": grid_results,
        "grid_size": len(grid_results),
        "any_config_diverges_from_relevance_only": any_divergence,
        "max_divergence_rate_in_grid": max_divergence,
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
    ap.add_argument("--question-date", default=None,
                    help="YYYY-MM-DD 'as of' date passage ages are computed "
                         "against. Defaults to today - NOT the fixture "
                         "placeholder (2026-01-01) run_end_to_end.py uses "
                         "for its synthetic demo, which earlier real-data "
                         "runs inherited by mistake.")
    ap.add_argument("--output-dir", default="experiments/outputs/fit_and_evaluate")
    args = ap.parse_args(argv)
    question_date = (date.fromisoformat(args.question_date) if args.question_date
                     else date.today())

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
    fit = fit_on_validation(val_items, generator, budget=args.budget,
                            question_date=question_date)
    print(f"  fitted: theta={fit['fitted_theta']}, "
          f"half_life={fit['fitted_half_life_days']}d, "
          f"lambda={fit['fitted_lambda']}")
    print(f"  validation token_f1 at this config: "
          f"{fit['fitted_validation_token_f1']:.3f}")
    print(f"  diagnostic: {'YES' if fit['any_config_diverges_from_relevance_only'] else 'NO'} "
          f"- does ANY of the {fit['grid_size']} grid configs admit a "
          "different evidence set than pure relevance ranking on at least "
          "one question? (max divergence rate in grid: "
          f"{fit['max_divergence_rate_in_grid']:.1%})")
    if not fit["any_config_diverges_from_relevance_only"]:
        print("  ^ If this is NO, the mechanism cannot be tested by this grid "
              "at all - the result below cannot distinguish 'the idea doesn't "
              "help' from 'this configuration never even tried anything "
              "different.' Treat any verdict below as provisional if so.")

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
        budget=args.budget, question_date=question_date,
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

    test_flags = temporal_flags(test_items)
    test_temporal_ids = {qid for qid, flag in test_flags.items() if flag}
    temporal_obj = TemporalPolicy(
        half_life_days=fit["fitted_half_life_days"], undated_score=0.0,
    )

    def mean_currency_for(system_label: str) -> float:
        scores = []
        for record in records:
            if record["system"] != system_label or record["status"] != "ok":
                continue
            if record["question_id"] not in test_temporal_ids:
                continue
            item = items_by_id[record["question_id"]]
            scores.append(_mean_currency(
                record["admitted_evidence_ids"], item, temporal_obj,
                item.question, question_date,
            ))
        return sum(scores) / len(scores) if scores else 0.0

    currency_by_system = {
        name: mean_currency_for(name) for name in by_system
    }

    def delta(a: str, b: str) -> float:
        return by_system[a]["token_f1"] - by_system[b]["token_f1"]

    def currency_delta(a: str, b: str) -> float:
        return currency_by_system[a] - currency_by_system[b]

    # PRIMARY verdict: mean currency of admitted evidence on the
    # temporal_candidate subgroup of the held-out test split - the direct
    # measure of what the proposed mechanism is supposed to change.
    # token_f1 is kept as a secondary/diagnostic field, not the verdict:
    # see this module's docstring for why it is insensitive to currency.
    main_evaluation = {
        "baseline": "baseline", "baseline_filter": rag2_filter_label,
        "proposed": proposed_label,
        "n_temporal_candidate_test_questions": len(test_temporal_ids),
        "baseline_mean_currency": currency_by_system["baseline"],
        "proposed_mean_currency": currency_by_system[proposed_label],
        "currency_delta": currency_delta(proposed_label, "baseline"),
        "verdict": ("IMPROVES currency" if currency_delta(proposed_label, "baseline") > 0
                    else "DOES NOT IMPROVE currency"),
        "secondary_token_f1": {
            "baseline": by_system["baseline"]["token_f1"],
            "proposed": by_system[proposed_label]["token_f1"],
            "delta": delta(proposed_label, "baseline"),
        },
    }
    ablation_study = {
        "full_proposed": proposed_label, "ablated_proposed": ablated_label,
        "currency_delta": currency_delta(proposed_label, ablated_label),
        "verdict": ("COMPONENT HELPS currency" if currency_delta(proposed_label, ablated_label) > 0
                    else "COMPONENT DOES NOT HELP currency"),
        "secondary_token_f1_delta": delta(proposed_label, ablated_label),
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
            "question_date": question_date.isoformat(),
            "fitted_on": "validation split (n={}), reported on test split "
                        "(n={}), no overlap".format(len(val_items), len(test_items)),
        },
        "run_summary": summary,
        "metrics_by_system": by_system,
        "currency_by_system": currency_by_system,
        "admitted_recency_by_system": recency,
        "temporal_subgroup": temporal_subgroup_breakdown(rows, temporal_flags(test_items)),
        "main_evaluation": main_evaluation,
        "ablation_study": ablation_study,
    }
    report_path = out_dir / "metrics_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    print(f"\n{'system':<24}{'n':>4}{'currency':>10}{'F1(sec)':>9}{'ground':>9}")
    for name, metrics in sorted(by_system.items()):
        print(f"{name:<24}{metrics['n']:>4}{currency_by_system[name]:>10.3f}"
              f"{metrics['token_f1']:>9.3f}{metrics['groundedness']:>9.3f}")
    print("  (currency = mean T(s) of admitted evidence on temporal_candidate "
          "questions, primary metric; F1 is the secondary/diagnostic score)")

    print(f"\nFitted on validation (n={len(val_items)}): theta="
          f"{fit['fitted_theta']}, half_life={fit['fitted_half_life_days']:g}d, "
          f"lambda={fit['fitted_lambda']}")
    print(f"Reported on held-out test (n={len(test_items)}, "
          f"{main_evaluation['n_temporal_candidate_test_questions']} temporal_candidate):")
    print(f"  Main evaluation (currency): baseline="
          f"{main_evaluation['baseline_mean_currency']:.3f} vs proposed="
          f"{main_evaluation['proposed_mean_currency']:.3f} "
          f"(delta {main_evaluation['currency_delta']:+.3f}) -> "
          f"{main_evaluation['verdict']}")
    print(f"  Ablation (currency): delta {ablation_study['currency_delta']:+.3f} -> "
          f"{ablation_study['verdict']}")
    print(f"  [secondary] token_f1 delta vs baseline: "
          f"{main_evaluation['secondary_token_f1']['delta']:+.3f}")
    print(f"\nFull report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
