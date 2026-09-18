# Current Objectives — canonical reference

**Adopted 2026-09-18, refined 2026-09-18.** This document is the canonical
statement of what this repository is for. Where any other doc
(`frozen_scope.md`, `research_ledger.md`, `next_steps.md`, or anything else
under `docs/`) disagrees with this one about scope, priority, or what counts
as the primary contribution, **this one governs**. Everything else under
`docs/` is kept for provenance — the design history that led here — not as
an alternative active scope.

---

## The three objectives

1. **Proposed-system validation.** Run the proposed system end-to-end,
   evaluate its overall performance, identify and correct genuine
   implementation/experimental errors, and verify it works correctly.

2. **Ablation study.** Using standard RAG evaluation metrics, determine the
   contribution of the proposed system's key component (recency weighting,
   `lambda`) by comparing the full system against the same system with that
   component removed (`lambda=0`, pure relevance ranking).

3. **RAG² comparison.** Evaluate whether the proposed system improves the
   RAG² baseline under comparable experimental conditions (same questions,
   same frozen candidate set, same context budget, same prompt, same
   generator).

**Main contribution:** determine, through systematic experiments, whether
the proposed system improves RAG² performance. This is the primary research
question everything else serves — do not assume the answer; the pipeline
below exists to determine it experimentally, honestly reporting a negative
result if that's what the numbers show.

## The pipeline

1. **Experimental setup** — finalize evaluation data, corpus, retrieval,
   generator, RAG², proposed system, metrics and experiment configuration;
   ensure both systems are evaluated under comparable conditions.
2. **Proposed-system validation** — run the proposed system end-to-end,
   detect and correct genuine implementation/evaluation errors, re-run until
   results are valid and complete.
3. **Main evaluation** — run RAG² and the proposed system on the same setup,
   compare with the selected standard RAG metrics. Determined experimentally,
   never assumed.
4. **Ablation study** — disable/remove the key proposed component (recency
   weighting) and compare the full system against the ablated version, same
   protocol and metrics as step 3.
5. **Analysis and write-up** — analyze proposed-system performance, RAG² vs.
   proposed results, and ablation results; report positive or negative
   findings honestly.

**The single entry point for all of steps 2-4** is
`experiments/runners/run_end_to_end.py`. It runs RAG², the no-filter
control, and the proposed system (swept across `--ablation-lambdas`,
default `0, 0.25, 0.5, 0.75, 1.0`), scores every arm with
`experiments/evaluation/rag_metrics.py`, and reports two distinct
comparisons in its JSON report and console output:
- `main_evaluation` (step 3): RAG² baseline vs. the full proposed system
  (`--proposed-lambda`, default `1.0`).
- `ablation_study` (step 4): the full proposed system vs. the same system
  with recency weighting removed (`lambda=0`, always included in the sweep).

The rest of the lambda sweep is reported as supplementary context, not a
required part of either step.

## Removed from the primary pipeline

Do not treat the following as independent thesis stages going forward,
unless a specific one becomes demonstrably required by the three objectives
above: a separate temporal/recency bias probe, temporal test-pair studies
(`experiments/test_pairs/`), verifier studies (`systems/proposed/verifier.py`),
contestedness/authority studies (`systems/proposed/contested.py`), clinician
studies / human annotation (`experiments/evaluation/annotation.py`,
`stats.py`), SOTA comparisons, or additional generator backbones. None of
this code is deleted — it is real, tested, and may still support secondary
analysis or a later write-up section — but it is not part of the critical
path for objectives 1-3, and no future work here should assume it needs to
run before a main result can be reported.

## What this does NOT change

- The corpus pipeline (`alzheimer_corpus/scripts/`), the retriever
  (`experiments/retrieval/`), the admission policy implementations
  (`systems/proposed/`, `systems/baseline/`), and the runner
  (`experiments/evaluation/runner.py`) are unchanged and are exactly what
  `run_end_to_end.py` orchestrates.
- The human-annotated hallucination-rate protocol is not removed, only
  moved out of the critical path — see "Removed from the primary pipeline."

## What this DOES change, going forward

- **The standard-metrics implementation** lives in
  `experiments/evaluation/rag_metrics.py`. It is deliberately separate from
  `experiments/evaluation/accuracy.py`, whose design decision to exclude
  automatic string-overlap scoring as the PRIMARY correctness signal still
  stands — `rag_metrics.py` is the ablation study's metrics track, not a
  replacement for that decision.
- **Git workflow**: this repository develops directly on `main`. Feature
  branches and pull requests are not used; the GitHub repository carries
  only `main`.

## Known blockers to a real (non-fixture) run

These are pre-existing, not introduced by this realignment — see
`docs/filter_training.md`, `docs/generator_contract.md`,
`docs/question_review.md`, and `alzheimer_corpus/README.md`'s "Open
decisions" for detail:

- RAG²'s real trained admission-filter checkpoint does not exist
  (`docs/filter_training.md`). Until it is trained, a "beats RAG²" result is
  against RAG²'s code path with an all-HELPFUL stand-in filter, not the
  paper's actual classifier.
- The proposed system's `lambda`/`theta`/half-life are unfit — they must be
  chosen on a validation split before a real (non-ablation-sweep) run.
  `--proposed-lambda 1.0` is a placeholder, not a fitted value.
- No real gold question/evidence evaluation set exists yet
  (`experiments/test_pairs/data/` is empty; the curated question pool is
  still under human review — `docs/question_review.md`).
- No real model has ever been invoked in this repository (only deterministic
  stand-ins in tests and in `run_end_to_end.py`'s default fixture run) —
  this sandbox has no network access to download one.
- `04_normalize.py`'s licensing gate is not enforced for PMC records
  (`alzheimer_corpus/README.md`, "Open decisions") — affects what the real
  corpus contains, not this script, but relevant before a real corpus-wide
  run.

None of these block objectives 1-2 as stated (run it, check performance, fix
errors, ablate with standard metrics) against the fixture; they block
treating the fixture's numbers as a real result. `run_end_to_end.py
--real-model` plus a real corpus-derived question set and a fitted
`--proposed-lambda` is the path to a real result once these are resolved.
