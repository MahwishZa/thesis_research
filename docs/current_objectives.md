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

## Current status (2026-09-23)

Objectives 1–3 have each been exercised end to end on real (pilot-scale)
data: the Temporal Filter runs, has been ablated, and has been compared to
the RAG² baseline under a validation-fitted configuration, with a paired
significance test — not a bare number — deciding the verdict. This
confirms the pipeline and statistical procedure both work correctly on real
data; see `docs/status_and_decisions.md` §1.2 for the engineering record.
**The comparison this thesis reports is the one obtained after the
pilot-scale reductions below are removed**, not the pilot run itself.

## What would change the setup from "pilot-scale" to "full"

See `docs/status_and_decisions.md` §3.1/§3.2 for full detail. The current
pilot-scale run was obtained with:

- **A ~1% pilot slice of the corpus**, not the full built index.
- **An untrained (chance-level) RAG² filter checkpoint.**
- **An extractive stand-in generator**, not a real generative model —
  `token_f1`/groundedness reflect evidence overlap, not free-text answer
  quality.
- **PMC licensing gate is not enforced** (`docs/status_and_decisions.md`
  §2.1) — affects redistribution of the corpus text, not research use.

Removing these is scoped, already implemented end-to-end, and does not
require new methodology.
