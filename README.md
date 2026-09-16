# Measuring Recency Bias in Evidence Admission

MS thesis implementation. **Research code, not a clinical system** — nothing
here is validated for, or usable in, patient care.

## The question

Retrieval-augmented pipelines filter retrieved passages before answering.
RAG² ([Sohn et al., NAACL 2025](https://github.com/dmis-lab/RAG2)) trains that
filter on labels derived from *how much a passage raises the model's
confidence*. Confidence gain is not the same thing as evidential support, and
it may be asymmetric with respect to evidence age: a passage agreeing with the
model's pre-training-era priors raises confidence, while newer, dissonant
guidance lowers it.

> **Does a confidence-derived evidence admission mechanism exhibit recency
> asymmetry, and does explicitly incorporating evidence recency reduce that
> asymmetry without simply degrading answer quality?**

The filter never sees a publication date — at inference it receives only the
question and the passage text. So any age asymmetry it shows must come from
*content* correlates of era, which is what makes the measurement interesting.

## The experiment

Three arms, one shared candidate set:

| Arm | What it does | Code |
|---|---|---|
| **No-filter control** | Admits everything, up to the context budget | `systems/baseline/no_filter.py` |
| **RAG²** | The reproduced baseline: a Flan-T5 `[HELPFUL]` / `[NOT_HELPFUL]` filter | `systems/baseline/` |
| **Recency-aware admission** | The proposed method | `systems/proposed/` |

Retrieval and reranking run **once per item** and are replayed byte-identically
to all three arms, so a difference in the answer is attributable to the
admission step and not to what was retrieved.

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

## What is deliberately *not* in the primary experiment

Entailment-derived support, source authority, contested-evidence handling,
supersession, answer verification, clinician rating, and comparison against a
further state-of-the-art filter. These were part of an earlier, larger design
and were cut to fit one student's compute and time budget. `contested.py` and
`verifier.py` remain in the repository, marked SECONDARY, off by default; they
may support a qualitative analysis or future work. **None of them is required
for the primary result.** See ledger decisions D-25 and D-26.

## Canonical scope

`docs/FROZEN_SCOPE.md` states exactly what is primary, what is secondary, and
what Stage 3 must satisfy. Where any other document disagrees with it about
what is primary, it governs.

## Layout

```
docs/                FROZEN_SCOPE (canonical scope), RESEARCH_LEDGER (decisions),
                     _SPECIFICATION (method), _UNDERSTANDING_REPORT (audit)
alzheimer_corpus/    Stage 1 — retrieval corpus (in progress)
systems/             the three arms
experiments/         Stage 2 — test-pair construction
tests/               unit + integration
```

## Status

Stage 1 (corpus) in progress. Stage 2 (test pairs) built and frozen, waiting
on one external dependency: the evaluation dataset must be acquired manually
(see `experiments/test_pairs/data/external/README.md`). Stage 3 not started.

```bash
python -m unittest discover -s tests -t .
```
