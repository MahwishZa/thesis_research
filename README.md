# Does Recency-Aware Evidence Admission Improve RAG²?

MS thesis implementation. **Research code, not a clinical system** — nothing
here is validated for, or usable in, patient care.

## The question

Retrieval-augmented pipelines filter retrieved passages before answering.
RAG² ([Sohn et al., NAACL 2025](https://github.com/dmis-lab/RAG2)) trains that
filter on labels derived from *how much a passage raises the model's
confidence*, and it represents time nowhere — not in the corpus, the index,
the retriever, the reranker or the filter. In a domain where guidance
changes, the evidence that reaches the generator may therefore be superseded.

> **Does a recency-aware admission policy improve on RAG², determined
> experimentally rather than assumed?**

This is the main contribution: not a claim that the proposed system is
better, but a systematic experiment to find out. See
`docs/current_objectives.md` for the canonical, current statement of scope —
where any other document disagrees with it, it governs.

## The three objectives

1. **Proposed-system validation** — run the proposed system end-to-end,
   evaluate its performance, and fix genuine implementation/experimental
   errors.
2. **Ablation study** — using standard RAG evaluation metrics, determine the
   contribution of the proposed system's key component (recency weighting,
   `lambda`) by comparing the full system against the same system with that
   component removed (`lambda=0`).
3. **RAG² comparison** — evaluate whether the proposed system improves RAG²
   under comparable experimental conditions.

## The experiment

| Arm | What it does | Code |
|---|---|---|
| **Baseline (RAG²)** | RAG²-style adaptation: a Flan-T5 `[HELPFUL]` / `[NOT_HELPFUL]` filter | `systems/baseline/` |
| **Proposed system** | Recency-aware admission | `systems/proposed/` |
| *No-filter control* | Admits everything, up to the budget — a reference point, not an outcome | `systems/baseline/no_filter.py` |

Retrieval and reranking run **once per item**, are frozen, and are replayed
byte-identically to every arm, so a difference in the answer is attributable
to the admission step and not to what was retrieved. Prompt, context budget,
generator instance and decoding settings are identical and checked in code
before a run starts (`experiments/evaluation/runner.py`).

## The proposed method

Deliberately small — an admission policy, not a framework:

```
A(s) = (1 − λ)·ρ(s)  +  λ·R(s, q, t_q)          admit if A(s) ≥ θ
```

- `ρ(s)` — rank-normalised reranker score, **the same signal the baseline gets**
- `R(s, q, t_q)` — recency: `2^(−age_days / H)`, where age is measured from the
  question's as-of date `t_q`
- `λ`, `θ`, `H` — the only tunable quantities, fitted on the validation split

`λ = 0` recovers pure relevance — the ablation study's "component removed"
condition (objective 2).

## The pipeline

```
1 Experimental setup → 2 Proposed-system validation
→ 3 Main evaluation: proposed vs RAG² → 4 Ablation study
→ 5 Analysis and write-up
```

**The single entry point for steps 2-4** is
`experiments/runners/run_end_to_end.py`. It runs RAG², the no-filter
control, and the proposed system (swept across `--ablation-lambdas`), scores
every arm with `experiments/evaluation/rag_metrics.py` (exact match, token
F1, ROUGE-L, context precision/recall, a groundedness proxy), and reports
two distinct comparisons: `main_evaluation` (RAG² vs. the full proposed
system) and `ablation_study` (full vs. component-removed).

## What is out of the critical path

Not deleted, not required before a main result can be reported: entailment
support, source authority, contested-evidence handling, supersession,
answer verification, temporal test-pair studies
(`experiments/test_pairs/`), clinician rating / human annotation
(`experiments/evaluation/annotation.py`, `stats.py`), and comparison against
further backbones or a further state-of-the-art filter. See
`docs/current_objectives.md`'s "Removed from the primary pipeline" for the
full list and why each is kept rather than deleted.

## Layout

```
docs/                four authoritative documents — see below
alzheimer_corpus/    the evidence corpus (PMC-based), COMPLETE / FROZEN
systems/             the arms: baseline (RAG²), proposed, no-filter control
experiments/         retrieval, question pool, evaluation infrastructure,
                     runners/run_end_to_end.py (steps 2-4's entry point)
tests/               unit + integration
```

## Documentation

| Document | Read it for |
|---|---|
| [`docs/current_objectives.md`](docs/current_objectives.md) | **Canonical scope** — the three objectives, the pipeline, what is out of the critical path, the known blockers. Governs every other document. |
| [`docs/research_experimental_specification.md`](docs/research_experimental_specification.md) | **The method** — what each arm does, what is held constant, the parameters, the generator contract, filter training, metrics, statistics, question provenance, and how to run it reproducibly. |
| [`docs/status_and_decisions.md`](docs/status_and_decisions.md) | **The record** — what has been executed, decided, measured, blocked and limited, plus the decision log and change log. |
| [`docs/question_review.md`](docs/question_review.md) | Reviewer instructions for the candidate question pool. |

## Status

**Alzheimer's corpus: COMPLETE / FROZEN** (verified 2026-09-19, see
`docs/status_and_decisions.md` §2). All seven
pipeline stages have run for real against the live corpus and their
outputs are independently cross-checked consistent end to end: 676 PubMed
records; 114,256-row PMC manifest, 114,157 verified and normalized;
111,315 unique after dedup; 4,377,041 chunks via the real MedCPT
tokenizer; every chunk tagged by claim classification (all 43
topical/evidence-level classes matched at least once). Confirmed
consumable by the retrieval layer via a real end-to-end test. Do not
rerun stages 01-07.

Questions (Step 2) are in human review. Steps 2-4 above (proposed system
validation, main evaluation, ablation) have tested infrastructure
(`run_end_to_end.py`) but no real run yet — see `docs/current_objectives.md`
for what's still needed (a real generator, real evaluation data, a fitted
`lambda`). No experimental result exists yet. See
`docs/status_and_decisions.md` for full component readiness and blockers.

```bash
python -m unittest discover -s tests -t .
```
