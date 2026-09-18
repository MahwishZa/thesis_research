# Current Objectives — canonical reference

**Adopted 2026-09-18.** This document is now the canonical statement of what
this repository is for. Where any other doc (`frozen_scope.md`,
`research_ledger.md`, `next_steps.md`, or anything else under `docs/`)
disagrees with this one about scope, priority, or what counts as the primary
contribution, **this one governs**. Everything else under `docs/` is kept for
provenance — the design history that led here — not as an alternative active
scope.

The prior design (hallucination-rate as primary outcome, human-annotated,
multi-arm contested-evidence/currency-state framework, sample-size power
analysis, judge validation guards, a multi-year phased plan) is not wrong,
but it is far more elaborate than the three objectives below require. It
remains recorded in `docs/research_ledger.md` and `docs/frozen_scope.md` for
provenance; nothing there is deleted, and none of it blocks the objectives
below.

---

## The three objectives

1. **Run the proposed system end-to-end and check its overall performance.**
   Identify and fix any errors that prevent it from running. There must be a
   single entry point that takes a question, runs retrieval → admission →
   generation, and produces an answer, for both the proposed system and the
   RAG² baseline.

2. **Add an ablation study using standard RAG evaluation metrics.**
   Standard, automatic metrics — exact match, token F1, ROUGE-L, context
   precision/recall, a groundedness/faithfulness proxy — computed across a
   sweep of the proposed system's configuration (its recency weight,
   `lambda`, at minimum), so the effect of the admission policy is visible
   without a human annotator in the loop for every ablation cell.

3. **Assess whether the proposed system improves on the baseline (RAG²).**
   The main contribution of the thesis is that the proposed system
   outperforms RAG² on the standard metrics above. This is now the headline
   comparison; hallucination rate and the earlier human-annotation protocol
   remain available as a secondary, more rigorous confirmation once a real
   evaluation run is possible, but they are not required to state the main
   result.

## What this does NOT change

- The corpus pipeline (`alzheimer_corpus/scripts/`), the retriever
  (`experiments/retrieval/`), the admission policy implementations
  (`systems/proposed/`, `systems/baseline/`), and the runner
  (`experiments/evaluation/runner.py`) are unchanged and are exactly what
  objective 1 runs.
- The human-annotated hallucination-rate protocol
  (`experiments/evaluation/annotation.py`, `stats.py`) is not removed. It is
  no longer the thing that has to happen before a headline result can be
  reported; the standard-metrics ablation is.

## What this DOES change, going forward

- **The single entry point for objective 1** is
  `experiments/runners/run_end_to_end.py`. It runs the RAG² baseline,
  the no-filter control, and the proposed system across a sweep of
  `lambda`, scores every arm with standard RAG metrics, and reports whether
  the proposed system improves on the baseline. See its module docstring for
  usage, including `--real-model` for a real Hugging Face generator once one
  is available (this development sandbox has no network access to download
  a model, so its default run uses a deterministic stand-in generator over
  a small synthetic fixture — real corpus data and a real model are needed
  for a result that means anything scientifically; see the script's
  docstring and `docs/generator_contract.md`).
- **The standard-metrics implementation** lives in
  `experiments/evaluation/rag_metrics.py`. It is deliberately separate from
  `experiments/evaluation/accuracy.py`, whose design decision to exclude
  automatic string-overlap scoring as the PRIMARY correctness signal still
  stands — `rag_metrics.py` is the ablation study's metrics track, not a
  replacement for that decision.
- **Git workflow**: this repository now develops directly on `main`. Feature
  branches and pull requests are no longer used; the GitHub repository has
  been reduced to `main` only.

## Known blockers to a real (non-fixture) run of objective 1/3

These are pre-existing, not introduced by this realignment — see
`docs/filter_training.md`, `docs/generator_contract.md`,
`docs/question_review.md` for detail:

- RAG²'s real trained admission-filter checkpoint does not exist
  (`docs/filter_training.md`). Until it is trained, a "beats RAG²" result is
  against RAG²'s code path with an all-HELPFUL stand-in filter, not the
  paper's actual classifier.
- The proposed system's `lambda`/`theta`/half-life are unfit — they must be
  chosen on a validation split before a real (non-ablation-sweep) run.
- No real gold question/evidence evaluation set exists yet
  (`experiments/test_pairs/data/` is empty; the curated question pool is
  still under human review — `docs/question_review.md`).
- No real model has ever been invoked in this repository (only deterministic
  stand-ins in tests and in `run_end_to_end.py`'s default fixture run) —
  this sandbox has no network access to download one.

None of these block objective 1 as stated (run it, check performance, fix
errors that prevent running) or objective 2 (ablate with standard metrics)
against the fixture; they block treating the fixture's numbers as a real
result. `run_end_to_end.py --real-model` plus a real corpus-derived question
set is the path to a real result once these are resolved.
