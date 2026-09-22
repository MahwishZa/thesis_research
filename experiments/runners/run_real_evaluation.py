#!/usr/bin/env python3
"""Run the three-arm comparison (RAG2 baseline / proposed Temporal Filter /
no-filter) over REAL, human-reviewed Alzheimer's questions, retrieved
against a REAL (if reduced-scope) corpus index - instead of
``run_end_to_end.py``'s synthetic 10-question fixture.

This is a TEMPORARY, reduced-scope run, assembled to get a first real-data
result under a hard deadline. Every simplification below is deliberate and
documented so it can be strengthened later; none of it is hidden in the
report this script writes.

Known, tracked simplifications (see "limitations" in the written report):

  1. CORPUS: retrieval runs against whatever index --index points to. The
     index built so far (experiments/outputs/index_pilot_reduced_scope)
     covers roughly the first 40,000 of the corpus's 4,377,041 chunks -
     about 1% - not the full frozen corpus. Retrieval quality is bounded by
     what that slice actually contains.
  2. RAG2 CHECKPOINT: whatever --rag2-checkpoint points to. The checkpoint
     trained so far was trained on 20 MedQA labels for 1 epoch and reported
     validation accuracy 0.0 (at chance) - see checkpoints/rag2_filter/
     checkpoint_record.json - meaning train.py's own usability check
     (record.is_usable()) says this checkpoint has NOT learned the label
     function. It is used anyway here as the best available real classifier
     for a first look, not because it is trusted.
  3. GENERATOR: no free-text generative model runs here. Every arm answers
     by returning the single highest-priority ADMITTED passage's text
     verbatim (see ``_extractive_answer`` below) - the same documented
     pattern ``run_end_to_end.py`` uses for its fixture, applied here to
     real retrieved evidence instead of synthetic text. This means
     token_f1/rouge_l/groundedness measure evidence-overlap with the
     reference answer, NOT free-text generation quality or hallucination
     from a real model - that measurement still requires wiring in a real
     generator (Llama-3-8B per the spec, or a smaller documented substitute)
     and is NOT done by this script.
  4. HYPERPARAMETERS: theta/half_life/lambda are the same specification
     placeholders run_end_to_end.py uses by default - NOT fit on the
     validation split. A real fitting procedure over
     experiments/questions/splits.json's "validation" split does not exist
     yet.
  5. QUESTION SET: by default this runs BOTH the validation and test splits
     combined (113 usable questions), because no fitting is happening
     tonight (see 4) - there is no validation/test leakage to protect yet.
     Once theta/half_life/lambda are actually fit on validation, a reported
     final result must run test only (--split test).

Because of (3) especially, treat context/F1/groundedness numbers here as a
proxy for "did this arm's admission policy surface evidence that overlaps
with the real reference answer", not as a hallucination measurement in the
full sense. The ADDITIONAL "admitted_recency" metric this script computes
(does an arm admit the most-recently-published candidate available, on
questions flagged temporal_candidate) does not depend on the generator
substitution at all, and is the more direct real-data signal for whether
the Temporal Filter is doing what it is meant to.

Usage (run on a machine with the real corpus, the pilot index, torch and
transformers installed - not this development sandbox):

    python -m experiments.runners.run_real_evaluation \\
        --rag2-checkpoint checkpoints/rag2_filter/final \\
        --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from experiments.evaluation import freezing as fz
from experiments.evaluation import rag_metrics as rm
from experiments.evaluation.runner import RunConfig, run_experiment
from experiments.retrieval.corpus import dated_only, read_passages_with_snapshot
from experiments.retrieval.index import DenseIndex
from experiments.retrieval.pipeline import RetrievalConfig, RetrievalPipeline, RetrievalError
from experiments.runners.run_end_to_end import (
    CONTEXT_PROMPT, build_systems, gold_evidence_ids, score_run,
    temporal_flags, temporal_subgroup_breakdown,
)
from systems.interfaces.generator import CallableGenerator, GenerationResult

DEFAULT_QUESTIONS_DIR = Path("experiments/questions")
DEFAULT_CORPUS = "alzheimer_corpus_pilot"
DEFAULT_INDEX = "experiments/outputs/index_pilot_reduced_scope"
DEFAULT_CHECKPOINT = "checkpoints/rag2_filter/final"

#: Real specification value (docs/research_experimental_specification.md
#: S8.2), unlike run_end_to_end.py's fixture DEFAULT_BUDGET=1.
DEFAULT_BUDGET = 5
DEFAULT_THETA = 0.5
DEFAULT_HALF_LIFE_DAYS = 365.0


def load_usable_questions(
    questions_dir: Path, split: str,
) -> list[dict[str, Any]]:
    """Join candidates.jsonl (full question text/answer) with splits.json
    (which questions are usable and which split they are in).

    Returns one dict per usable question with question/reference_answer/
    reference_date/reference_source/temporal_candidate/question_id/split.
    """
    candidates_by_id: dict[str, dict[str, Any]] = {}
    with open(questions_dir / "candidates.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidates_by_id[row["question_id"]] = row

    splits = json.loads((questions_dir / "splits.json").read_text(encoding="utf-8"))
    out = []
    for item in splits["items"]:
        if split != "all" and item["split"] != split:
            continue
        full = candidates_by_id.get(item["question_id"])
        if full is None:
            raise SystemExit(
                f"question_id {item['question_id']!r} is in splits.json but "
                "not in candidates.jsonl - the two files are out of sync"
            )
        out.append({
            **full,
            "split": item["split"],
            "review_decision": item["review_decision"],
            "pending_revision": item["pending_revision"],
        })
    return out


def _extractive_answer(question, evidence, prompt):
    """Documented stand-in generator: NOT a free-text model.

    Returns the highest-priority admitted passage's text verbatim. See the
    module docstring's simplification (3) for exactly what this does and
    does not measure. Mirrors run_end_to_end.py's ``_synthetic_generate``,
    applied to real retrieved evidence instead of synthetic fixture text.
    """
    if not evidence:
        return GenerationResult(text="", metadata={"n_evidence": 0})
    return GenerationResult(
        text=evidence[0].text,
        metadata={"n_evidence": len(evidence), "prompt_len": len(prompt or "")},
    )


def build_frozen_items(
    questions: list[dict[str, Any]],
    pipeline: RetrievalPipeline,
    *, verbose: bool = True,
) -> tuple[list[fz.FrozenItem], list[str]]:
    """Retrieve real evidence for every question and freeze it.

    A question that fails retrieval (RetrievalError) is skipped, not
    fatal - its id is returned in ``skipped`` so the caller can report how
    many of the intended set actually ran. Skipping is expected to be rare
    (retrieval_depth=50 > candidate_count=20 over a 40k-passage index
    almost always yields enough candidates) but must not crash a run that
    is otherwise fine over the other ~112 questions.
    """
    items: list[fz.FrozenItem] = []
    skipped: list[str] = []
    for i, q in enumerate(questions, 1):
        try:
            retrieved = pipeline.build_candidate_set(
                question_id=q["question_id"], question=q["question"],
            )
        except RetrievalError as exc:
            skipped.append(q["question_id"])
            print(f"  SKIPPED {q['question_id']}: {exc}", file=sys.stderr)
            continue
        items.append(fz.FrozenItem(
            question_id=q["question_id"],
            question=q["question"],
            reference_answer=q["reference_answer"],
            reference_source=q["reference_source"],
            reference_date=q["reference_date"],
            candidates=retrieved.candidates,
            corpus_snapshot=retrieved.corpus_snapshot,
            # No passage-level gold annotation exists for this real corpus:
            # the Cochrane reference answer is not tied to a specific chunk
            # id in alzheimer_corpus. Left empty (not guessed), matching
            # rag_metrics.context_scores()'s documented "no gold -> None"
            # behaviour rather than inventing a relevance judgement.
            reference_evidence_ids=(),
            temporal_candidate=bool(q.get("temporal_candidate", False)),
            metadata={
                "topic": q.get("topic"), "subtopic": q.get("subtopic"),
                "review_decision": q.get("review_decision"),
                "pending_revision": q.get("pending_revision"),
                "split": q.get("split"),
            },
        ))
        if verbose and (i % 10 == 0 or i == len(questions)):
            print(f"  ...retrieved {i}/{len(questions)} questions "
                  f"({len(skipped)} skipped)")
    return items, skipped


def admitted_recency_by_system(
    records: list[dict[str, Any]], items_by_id: dict[str, fz.FrozenItem],
    flags: dict[str, bool],
) -> dict[str, Any]:
    """Does each arm admit the most-recently-published available candidate?

    Computed directly from publication_date strings (ISO-prefixed, so plain
    string comparison orders them correctly) - no generator or reference
    answer involved, so this is unaffected by simplification (3) in the
    module docstring. This is the more direct real-data test of whether the
    Temporal Filter changes WHAT EVIDENCE SURVIVES in the direction it is
    meant to; token_f1/groundedness below test something adjacent to that.
    """
    by_system: dict[str, list[bool]] = {}
    by_system_temporal: dict[str, list[bool]] = {}
    for record in records:
        if record["status"] != "ok":
            continue
        item = items_by_id[record["question_id"]]
        dated = [c for c in item.candidates if c.publication_date]
        if not dated:
            continue
        most_recent_id = max(dated, key=lambda c: c.publication_date).evidence_id
        admitted_most_recent = most_recent_id in set(record["admitted_evidence_ids"])
        by_system.setdefault(record["system"], []).append(admitted_most_recent)
        if flags.get(record["question_id"], False):
            by_system_temporal.setdefault(record["system"], []).append(
                admitted_most_recent)

    def _rate(bools: list[bool]) -> Optional[float]:
        return (sum(bools) / len(bools)) if bools else None

    return {
        "all_questions": {
            sys_name: {"n": len(v), "admitted_most_recent_rate": _rate(v)}
            for sys_name, v in sorted(by_system.items())
        },
        "temporal_candidate_questions": {
            sys_name: {"n": len(v), "admitted_most_recent_rate": _rate(v)}
            for sys_name, v in sorted(by_system_temporal.items())
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions-dir", type=Path, default=DEFAULT_QUESTIONS_DIR)
    ap.add_argument("--split", choices=("validation", "test", "all"),
                    default="all",
                    help="which question split to run. Default 'all' "
                         "(validation+test combined) is correct ONLY while "
                         "theta/half_life/lambda are unfit placeholders - "
                         "see simplification (5) in this module's docstring. "
                         "Once they are fit on 'validation', pass "
                         "--split test for any reported final result.")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--rag2-checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cpu",
                    help="device for the RAG2 filter and MedCPT models. "
                         "Defaults to cpu (recommended tonight - GPU memory "
                         "has been unreliable in this environment); pass "
                         "cuda to try the GPU.")
    ap.add_argument("--output-dir", default="experiments/outputs/real_evaluation_pilot")
    ap.add_argument("--ablation-lambdas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--proposed-lambda", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=DEFAULT_THETA)
    ap.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE_DAYS)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = ap.parse_args(argv)

    lambdas = [float(x) for x in args.ablation_lambdas.split(",") if x.strip()]
    if 0.0 not in lambdas:
        lambdas = [0.0] + lambdas
    if args.proposed_lambda not in lambdas:
        lambdas = sorted(set(lambdas) | {args.proposed_lambda})
    proposed_label = f"proposed_lambda_{args.proposed_lambda:g}"
    ablated_label = "proposed_lambda_0"

    print(f"Loading usable questions (split={args.split})...")
    questions = load_usable_questions(args.questions_dir, args.split)
    print(f"  {len(questions)} usable questions loaded")
    if not questions:
        print("no usable questions for this split", file=sys.stderr)
        return 1

    print(f"Loading corpus from {args.corpus} ...")
    passages, snapshot, _dupes = read_passages_with_snapshot(
        args.corpus, on_duplicate="keep_first")
    passages = dated_only(passages)
    print(f"  {len(passages)} dated passages, snapshot={snapshot}")

    print(f"Loading index from {args.index} ...")
    index = DenseIndex.load(args.index)
    print(f"  {len(index.passage_ids)} x {index.dim} ({index.encoder_name})")

    from experiments.retrieval.encoders import medcpt_query_encoder, MedCPTReranker
    query_encoder = medcpt_query_encoder(device=args.device)
    reranker = MedCPTReranker(device=args.device)
    retrieval_config = RetrievalConfig()  # spec defaults: depth=50, count=20
    pipeline = RetrievalPipeline(
        index=index, passages=passages, query_encoder=query_encoder,
        reranker=reranker, config=retrieval_config,
    )

    print(f"Retrieving real evidence for {len(questions)} questions "
          "(this is the slow step - MedCPT encode + rerank per question)...")
    items, skipped = build_frozen_items(questions, pipeline)
    print(f"  {len(items)} frozen items built, {len(skipped)} skipped")
    if not items:
        print("no items survived retrieval", file=sys.stderr)
        return 1

    print(f"Loading RAG2 filter checkpoint from {args.rag2_checkpoint} ...")
    from systems.baseline.admission import FlanT5RAG2Filter
    rag2_filter = FlanT5RAG2Filter(args.rag2_checkpoint, device=args.device)
    rag2_filter_label = (
        f"FlanT5RAG2Filter({args.rag2_checkpoint}) - trained on 20 MedQA "
        "labels/1 epoch, validation_accuracy=0.0 (at chance) - see "
        "checkpoint_record.json. NOT a validated classifier."
    )

    generator = CallableGenerator(_extractive_answer)
    systems = build_systems(generator, lambdas, rag2_filter,
                            theta=args.theta, half_life=args.half_life,
                            budget=args.budget)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    config = RunConfig(
        run_id="real_evaluation_pilot-001",
        model="extractive-stand-in (NOT a generative model - see module docstring)",
        model_version="n/a",
        generation_config={"temperature": 0.0},
        system_config_hash=fz.config_hash({
            "budget": args.budget, "theta": args.theta,
            "half_life": args.half_life, "lambdas": lambdas,
            "proposed_lambda": args.proposed_lambda,
            "rag2_filter": rag2_filter_label,
            "corpus_snapshot": snapshot, "index": args.index,
        }),
    )

    print("Running all arms...")
    summary = run_experiment(items, systems, config, str(results_path))

    records = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines()]
    gold = gold_evidence_ids(items)
    rows = score_run(records, gold)
    by_system = rm.aggregate_by_system(rows)
    items_by_id = {item.question_id: item for item in items}
    recency = admitted_recency_by_system(records, items_by_id, temporal_flags(items))

    def delta(a: str, b: str) -> float:
        return by_system[a]["token_f1"] - by_system[b]["token_f1"]

    def recency_delta(a: str, b: str, group: str) -> Optional[float]:
        ra = recency[group].get(a, {}).get("admitted_most_recent_rate")
        rb = recency[group].get(b, {}).get("admitted_most_recent_rate")
        if ra is None or rb is None:
            return None
        return ra - rb

    main_evaluation = {
        "baseline": "baseline", "baseline_filter": rag2_filter_label,
        "proposed": proposed_label,
        "token_f1_delta": delta(proposed_label, "baseline"),
        "admitted_most_recent_rate_delta_temporal_subgroup": recency_delta(
            proposed_label, "baseline", "temporal_candidate_questions"),
        "verdict_token_f1": ("IMPROVES" if delta(proposed_label, "baseline") > 0
                             else "DOES NOT IMPROVE"),
    }
    ablation_study = {
        "full_proposed": proposed_label, "ablated_proposed": ablated_label,
        "token_f1_delta": delta(proposed_label, ablated_label),
        "admitted_most_recent_rate_delta_temporal_subgroup": recency_delta(
            proposed_label, ablated_label, "temporal_candidate_questions"),
        "verdict_token_f1": ("COMPONENT HELPS" if delta(proposed_label, ablated_label) > 0
                             else "COMPONENT DOES NOT HELP"),
    }

    report = {
        "limitations": {
            "corpus": f"pilot index, {len(passages)} of 4,377,041 real corpus "
                      "passages (~1%) - see module docstring (1)",
            "rag2_checkpoint": rag2_filter_label,
            "generator": "extractive stand-in, NOT free-text generation - "
                        "see module docstring (3). token_f1/rouge_l/"
                        "groundedness below measure evidence-overlap with "
                        "the reference answer, not generation quality.",
            "hyperparameters_fit": False,
            "question_split_used": args.split,
            "n_questions_requested": len(questions),
            "n_questions_retrieved": len(items),
            "n_questions_skipped": len(skipped),
            "skipped_question_ids": skipped,
        },
        "system_config": {
            "theta": args.theta, "half_life_days": args.half_life,
            "context_budget": args.budget,
            "proposed_lambda": args.proposed_lambda,
            "ablation_lambdas": lambdas,
        },
        "run_summary": summary,
        "metrics_by_system": by_system,
        "admitted_recency_by_system": recency,
        "temporal_subgroup": temporal_subgroup_breakdown(rows, temporal_flags(items)),
        "main_evaluation": main_evaluation,
        "ablation_study": ablation_study,
    }
    report_path = out_dir / "metrics_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    print(f"\nRan {summary['n_systems']} arms over {summary['n_items']} real "
          f"questions ({summary['n_errors']} errors, {len(skipped)} skipped "
          f"at retrieval). Results: {results_path}")
    print(f"\n{'system':<24}{'n':>4}{'F1':>8}{'ROUGE-L':>9}{'ground':>9}"
          f"{'recent%(temp)':>15}")
    for name, metrics in sorted(by_system.items()):
        rec = recency["temporal_candidate_questions"].get(name, {})
        rec_rate = rec.get("admitted_most_recent_rate")
        rec_str = f"{rec_rate:>14.3f}" if rec_rate is not None else "           n/a"
        print(f"{name:<24}{metrics['n']:>4}{metrics['token_f1']:>8.3f}"
              f"{metrics['rouge_l_f1']:>9.3f}{metrics['groundedness']:>9.3f}"
              f"{rec_str}")

    print(f"\nMain evaluation (RAG2 baseline vs proposed lambda="
          f"{args.proposed_lambda:g}): token F1 delta "
          f"{main_evaluation['token_f1_delta']:+.3f} -> "
          f"{main_evaluation['verdict_token_f1']}")
    print(f"Ablation (full vs lambda=0): token F1 delta "
          f"{ablation_study['token_f1_delta']:+.3f} -> "
          f"{ablation_study['verdict_token_f1']}")
    print(f"\nFull report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
