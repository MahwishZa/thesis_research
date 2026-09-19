# Current Objectives — canonical reference

**Adopted 2026-09-18, refined 2026-09-19.** This document is the canonical
statement of what this repository is for. Where any other doc under `docs/`
disagrees with this one about scope, priority, or what counts as the primary
contribution, **this one governs**.

There are three other documents, and only three:

- `docs/research_experimental_specification.md` — **the method**: what each
  arm does, what is held constant, the parameters, the generator contract,
  filter training, metrics, statistics, question provenance, and how to run
  it reproducibly.
- `docs/status_and_decisions.md` — **the record**: what has actually been
  executed, decided, measured, blocked and limited.
- `docs/question_review.md` — reviewer instructions for the question pool.

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

## Superseded designs — provenance only

This repository has changed research question twice. Neither earlier question
is an alternative active scope, and neither produces a thesis outcome. They
are recorded here so the repository's history is traceable and so the code
retained under `experiments/test_pairs/` is explicable.

**First design — admission asymmetry (superseded 2026-09-18 by the
hallucination framing, and again by this document).**

> *(superseded, not current)* Does a confidence-derived evidence admission
> mechanism exhibit recency asymmetry, and does explicitly incorporating
> evidence recency reduce that asymmetry without simply degrading answer
> quality?

Its primary measurement was `Δ = P(admit | older) − P(admit | newer)` over
matched temporal-counterfactual pairs. **Δ is no longer an outcome.** The
pivot narrowed the thesis from "measure a bias, then correct it" to "does the
correction help". The proximate cause was measured, not stylistic: the
matched-pair instrument needed AD-domain pairs, and the Cochrane census
contains only 9 usable ones (exact power 0.000) — a census ceiling, not a
sampling shortfall (decision D-35). What did **not** survive: the asymmetry
measurement, the matched-pair construction as a primary instrument, the
negative control built for Δ, and the multi-backbone replication question.

**Second design — hallucination rate primary (superseded 2026-09-18 by this
document).**

> *(superseded, not current)* Does the proposed system reduce the rate of
> hallucinated answers relative to the baseline, under identical question and
> evidence conditions, while maintaining comparable QA accuracy?

The human-annotated hallucination protocol it defined is not deleted — it is
implemented, tested, and moved out of the critical path (see "Removed from
the primary pipeline"). Under the current scope the main evaluation and the
ablation study are scored with **standard automatic RAG metrics**
(`experiments/evaluation/rag_metrics.py`), which the first design explicitly
forbade as outcomes. That inversion is guarded by
`tests/unit/test_scope_invariants.py`.

**What survives both pivots is the proposed method itself**, unchanged: the
same `A(s) = (1 − λ)·ρ(s) + λ·R(s, q, t_q)` scoring rule served all three
framings. `experiments/test_pairs/` is retained rather than deleted —
deleting it would destroy provenance and it breaks nothing — but nothing in
it produces a thesis outcome and it is not part of the five-step pipeline.

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
`docs/status_and_decisions.md` §3.1 for their current state and
`docs/research_experimental_specification.md` §§9–10 for what resolving each
one requires:

- RAG²'s real trained admission-filter checkpoint does not exist
  (specification §10). Until it is trained, a "beats RAG²" result is against
  RAG²'s code path with an all-HELPFUL stand-in filter, not the paper's
  actual classifier. `run_end_to_end.py` records which filter actually ran
  (`baseline_is_trained_rag2`) so this cannot be inferred wrongly.
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
  (`docs/status_and_decisions.md` §2.1) — affects
  what the real corpus contains, not this script, but relevant before a
  real corpus-wide run; the real corpus has already been normalized
  without this gate enforced.

None of these block objectives 1-2 as stated (run it, check performance, fix
errors, ablate with standard metrics) against the fixture; they block
treating the fixture's numbers as a real result. `run_end_to_end.py
--real-model` plus a real corpus-derived question set and a fitted
`--proposed-lambda` is the path to a real result once these are resolved.
