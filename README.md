# Reducing Hallucination by Recency-Aware Evidence Admission

MS thesis implementation. **Research code, not a clinical system** — nothing
here is validated for, or usable in, patient care.

## The question

Retrieval-augmented pipelines filter retrieved passages before answering.
RAG² ([Sohn et al., NAACL 2025](https://github.com/dmis-lab/RAG2)) trains that
filter on labels derived from *how much a passage raises the model's
confidence*, and it represents time nowhere — not in the corpus, the index,
the retriever, the reranker or the filter. In a domain where guidance changes,
the evidence that reaches the generator may therefore be superseded, and an
answer built on superseded evidence can be fluent, confident and wrong.

> **Does the proposed solution/system reduce the rate of hallucinated answers
> in Alzheimer's disease question answering, relative to the baseline system,
> under identical question and evidence conditions, while maintaining
> comparable QA accuracy?**

Two outcomes, and only two:

| | Outcome |
|---|---|
| **Primary** | Hallucination rate (HAR) — an answer scores 1 if any claim is unsupported by, or contradicted by, the evidence actually supplied to the generator |
| **Secondary** | QA accuracy against the question's reference answer |

## The experiment

| Arm | What it does | Code |
|---|---|---|
| **Baseline** | RAG²-style adaptation: a Flan-T5 `[HELPFUL]` / `[NOT_HELPFUL]` filter | `systems/baseline/` |
| **Proposed solution/system** | Recency-aware admission | `systems/proposed/` |
| *No-filter control* | Admits everything, up to the budget — a reference point, not an outcome | `systems/baseline/no_filter.py` |

Retrieval and reranking run **once per item**, are frozen, and are replayed
byte-identically to every arm, so a difference in the answer is attributable
to the admission step and not to what was retrieved. Prompt, context budget,
generator instance and decoding settings are identical and checked in code
before a run starts.

## The proposed method

Deliberately small — an admission policy, not a framework:

```
A(s) = (1 − λ)·ρ(s)  +  λ·R(s, q, t_q)          admit if A(s) ≥ θ
```

- `ρ(s)` — rank-normalised reranker score, **the same signal the baseline gets**
- `R(s, q, t_q)` — recency: `2^(−age_days / H)`, where age is measured from the
  question's as-of date `t_q`
- `λ`, `θ`, `H` — the only tunable quantities, all fitted on the validation
  split, never on the test set

`λ = 0` recovers pure relevance, which is the built-in ablation isolating what
the recency signal contributes.

The smallness is the point. With one added signal and one weight, an observed
difference is attributable to the temporal component. With four weighted
components it would not be.

## What is deliberately *not* in the experiment

Entailment-derived support, source authority, contested-evidence handling,
supersession, answer verification, clinician rating, and comparison against a
further state-of-the-art filter. `contested.py` and `verifier.py` remain in the
repository, marked SECONDARY, off by default. **None is required for the
result.**

No additional research objective is reported — no hallucination subtype
analysis, error taxonomy, retrieval-quality ranking, temporal or ambiguity
analysis, and no ROUGE / BLEU / BERTScore.

An earlier design made *admission asymmetry* the primary question. It is
superseded; see `docs/frozen_scope.md` §7 for what that was and what survived
of it. `experiments/test_pairs/` is its infrastructure, retained for
provenance and not part of the pipeline below.

## The pipeline

```
1 corpus → 2 questions → 3 freeze evidence → 4 baseline → 5 proposed
→ 6 collect → 7 annotate → 8 HAR → 9 accuracy → 10 statistics
→ 11 analyse → 12 write up
```

## Canonical scope

`docs/frozen_scope.md` states exactly what is primary, what is secondary, and
what the experiment must satisfy. Where any other document disagrees with it
about what is primary, it governs.

## Layout

```
docs/                frozen_scope (canonical scope), system_specification
                     (what each arm does), research_ledger (decisions) —
                     see docs/repository_structure.md for the full index
alzheimer_corpus/    Step 1 — retrieval corpus (in progress, do not modify)
systems/             the arms
experiments/         retrieval, question pool, evaluation infrastructure
tests/               unit + integration
```

## Status

Step 1 (corpus): PMC source retrieval executed and verified; other sources
and normalize/deduplicate/chunk still in progress. Step 2 (questions) in
human review. Steps 3–12 have tested infrastructure and are waiting on real
data. No experimental result exists. See `docs/next_steps.md`.

```bash
python -m unittest discover -s tests -t .
```
