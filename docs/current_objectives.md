# Current Objectives — canonical reference

**Adopted 2026-09-18, last simplified 2026-09-20.** This document says what
this repository is for. If another doc disagrees about scope or priority,
**this one wins**.

There are three other active documents:

- `docs/research_experimental_specification.md` — **the method**: what each
  arm does, the parameters, the generator, the metrics, and how to run it.
- `docs/status_and_decisions.md` — **the record**: what has actually been
  run, decided, measured, and blocked.
- `docs/question_review.md` — instructions for reviewing the question pool.

Earlier, larger research directions were moved to `_archive/` — see
`_archive/README.md`. They are not part of this project anymore.

---

## The research question

> **Does adding a Temporal Filter to RAG² improve question answering for
> Alzheimer's disease, compared to RAG² alone?**

- **RAG²** is the published baseline (Sohn et al., NAACL 2025): a filter
  decides which retrieved passages an LLM gets to see, based only on how
  helpful the text looks.
- **The Temporal Filter** is this thesis's one contribution: the same idea,
  but the filter also considers *how old each passage is* relative to the
  question. Code: `systems/proposed/`.
- **No Filter** is a third arm that skips filtering entirely — it shows
  whether filtering (of either kind) helps at all.

**Main contribution:** determine, through experiments, whether the Temporal
Filter improves RAG². Do not assume the answer. The point of the experiment is to find out, and to
report honestly if the Temporal Filter does **not** help.

## The three objectives

1. **Run it.** Get the Temporal Filter working end-to-end and fix real bugs.
2. **Ablate it.** Show whether the temporal part specifically is what helps,
   by comparing the full Temporal Filter against the same system with the
   temporal part switched off (`lambda=0`).
3. **Compare it to RAG².** Under identical questions, evidence, and
   generator, does the Temporal Filter score better than RAG² alone?

## The pipeline

```
RAG² baseline  →  Temporal Filter  →  No-Filter control
        ↓                ↓                    ↓
                  same questions
                  same retrieved evidence
                  same generator
                        ↓
                    Evaluation
                        ↓
                     Ablation
                        ↓
              Statistical comparison
```

One command runs steps 2-4 of this: `experiments/runners/run_end_to_end.py`.
It runs all three arms, sweeps the Temporal Filter's weight `lambda` across
`--ablation-lambdas` (default `0, 0.25, 0.5, 0.75, 1.0`), scores every arm
with `experiments/evaluation/rag_metrics.py`, and prints two comparisons:

- `main_evaluation` — RAG² vs. the full Temporal Filter (`--proposed-lambda`,
  default `1.0`).
- `ablation_study` — the full Temporal Filter vs. the same system with
  `lambda=0` (temporal weighting off).

The rest of the lambda sweep is extra context, not a required result.

## What is archived, and why

`_archive/` holds an earlier, more complex research direction: matched pairs
of old vs. new evidence, contested-evidence detection, and answer
verification. None of it answers the current research question, so it was
moved out of the active folders. It still works and is not deleted — see
`_archive/README.md`.

## What stays the same

- The corpus pipeline (`alzheimer_corpus/scripts/`) is untouched.
- The retriever (`experiments/retrieval/`), the admission code
  (`systems/proposed/`, `systems/baseline/`), and the runner
  (`experiments/evaluation/runner.py`) are unchanged.
- The human-annotated hallucination protocol
  (`experiments/evaluation/annotation.py`, `stats.py`) still exists and
  works, but is not required for the three objectives above.

## Standing rules

- **Metrics:** `experiments/evaluation/rag_metrics.py` (automatic) is
  separate from `experiments/evaluation/accuracy.py` (judged correctness).
  They are not meant to agree; they measure different things.
- **Git:** everything happens on `main`. No feature branches, no PRs.

## What is blocking a real (non-fixture) run

See `docs/status_and_decisions.md` §3.1 for full detail. In short:

- **No trained RAG² filter checkpoint yet.** Until one is trained, RAG² runs
  with an all-HELPFUL stand-in, and the report says so
  (`baseline_is_trained_rag2`).
- **`lambda`, `theta`, the half-life are unfit.** They must be chosen on a
  validation split before a real run. `--proposed-lambda 1.0` is a
  placeholder.
- **No approved question set yet.** The 123-question pool is still under
  human review (`docs/question_review.md`).
- **No real generator run yet** in this environment — a smaller model has
  confirmed the pipeline works; Llama-3-8B-Instruct itself has not run here.
- **PMC licensing gate is not enforced** (`docs/status_and_decisions.md`
  §2.1) — affects redistribution of the corpus text, not research use.

None of this blocks running, checking, or ablating the system against the
built-in fixture. It blocks treating fixture numbers as a real result.
