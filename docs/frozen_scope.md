# Frozen Scope — canonical reference

**Rewritten 2026-09-17** to state the current research question. Where any
other document disagrees with this one about what is *primary*, this one
governs. The design it replaces is recorded in §7 as superseded history, for
provenance only — it is not an alternative active scope and no part of it is a
thesis outcome.

---

## 1. Primary research question

> **Does the proposed solution/system reduce the rate of hallucinated answers
> in Alzheimer's disease question answering, relative to the baseline system,
> under identical question and evidence conditions, while maintaining
> comparable QA accuracy?**

## 2. The two evaluation objectives

| | Outcome | Definition |
|---|---|---|
| **PRIMARY** | Hallucination rate (HAR) | An answer scores 1 if it contains at least one claim unsupported by, or contradicted by, the evidence actually supplied to the generator; 0 otherwise. HAR = hallucinated answers ÷ evaluated answers. |
| **SECONDARY** | QA accuracy | Whether the answer is correct against the question's reference answer. |

Primary comparison, paired by question:

```
ΔHAR = HAR_proposed − HAR_baseline
```

**There is no third objective, and no diagnostic objective.** Specifically
excluded as thesis outcomes: hallucination subtype analysis, error taxonomy,
retrieval-quality measurement, temporal analysis, ambiguity analysis,
admission asymmetry, and ROUGE / BLEU / BERTScore. Fields serving these may
exist inside the implementation where the code needs them; none is reported
as a research finding.

## 3. The comparison

| Arm | Admission rule | Role |
|---|---|---|
| `B2_RAG2` | Flan-T5 `[HELPFUL]` / `[NOT_HELPFUL]` | **The baseline.** One half of the primary comparison. |
| `P_RECENCY` | `A(s) ≥ θ` | **The proposed solution/system.** The other half. |
| `B1_NO_FILTER` | Admit everything, up to the budget | Optional reference control. Not required for the primary result and not a research objective; it bounds what filtering does at all. |

What differs between arms is the **admission rule and nothing else**.
Identical across arms and enforced in code: question, candidate evidence set
and its order, reranker scores, context budget, prompt template, generator
instance and decoding settings (`experiments/evaluation/runner.py`:
`assert_prompt_parity`, `assert_budget_parity`, `assert_generator_parity`,
`assert_same_candidate_sets`).

## 4. The proposed method

Unchanged by this rewrite — the scoring mechanism is the thesis's method and
is not reopened:

```
A(s) = (1 − λ)·ρ(s) + λ·R(s, q, t_q)        admit if A(s) ≥ θ
```

* `ρ(s)` — rank-normalised reranker score, the signal the baseline already
  produces. In [0, 1].
* `R(s, q, t_q) = 2^(−age_days / H)` where `age_days = t_q − publication_date(s)`,
  clamped at zero. In (0, 1]. **Plain age decay and nothing else.**
* `λ ∈ [0, 1]` — how much of the score is recency.
* `θ ∈ [0, 1]` — admission threshold, on the same scale as `A`.

Three tunable quantities: **λ, θ, H**. All fitted on the validation split,
never on test. `λ = 0` is an internal ablation, not a fourth arm.

### Why this could affect the primary outcome

The causal pathway the thesis tests: admission decides which passages reach
the generator; a generator given superseded evidence can produce a
well-formed claim that the current evidence contradicts; weighting recency at
admission changes *which* evidence is present to be contradicted. The
mechanism acts on the evidence supplied, and HAR is defined against exactly
that evidence.

## 5. Secondary / future components

Present in the repository, off by default, never part of a primary result:
contested-evidence detection, supersession discounting, retraction exclusion,
time-invariance (ψ), answer verification, entailment-derived support, source
authority. Enabling any of them is a secondary analysis and must be declared
as such; each run records which were active.

## 6. Explicitly outside scope

Clinician rating study · comparison against a further state-of-the-art filter ·
agentic reference arm · supersession table · claim-class-dependent inclusion
rules · retrieval-quality optimisation · a second filter backbone · any
additional research objective.

---

## 7. Superseded design — provenance only

An earlier design made **admission asymmetry** the primary question:

> *(superseded, not current)* Does a confidence-derived evidence admission
> mechanism exhibit recency asymmetry, and does explicitly incorporating
> evidence recency reduce that asymmetry without simply degrading answer
> quality?

with primary measurement `Δ = P(admit | older) − P(admit | newer)` over
matched temporal-counterfactual pairs.

**This is no longer the research question and Δ is no longer an outcome.**
It is recorded here so the repository's history is traceable and so the
Stage-2 artifacts under `experiments/test_pairs/` are explicable.

The pivot narrowed the thesis from "measure a bias, then correct it" to
"does the correction reduce hallucination". What survives is the proposed
method itself, which is unchanged: the same `A(s)` scoring rule serves both
framings. What does not survive is the asymmetry measurement, the matched-pair
construction as a *primary* instrument, the negative control built for Δ, and
the multi-backbone replication question.

`experiments/test_pairs/` and `docs/research_experimental_specification.md`
belong to that superseded design. The code is retained rather than deleted —
deleting it would destroy provenance and it breaks nothing — but **nothing in
it produces a thesis outcome**, and it is not run as part of the 12-step
pipeline.

## 8. Interface requirements the experiment must satisfy

1. **Candidate sets are frozen once and replayed byte-identically** to both
   arms. Neither arm retrieves for itself during the primary comparison.
2. **N is identical for every item.** `ρ` is a within-set rank, so `θ` is only
   comparable across items at fixed candidate-set size.
3. **Candidate sets contain only dated passages.** Applied identically to both
   arms at construction, so the undated branch of the recency score never
   fires and `undated_score` is not a tunable. Every run reports its count of
   `UNDATED` passages; in a valid run that count is zero.
4. **Context order is candidate-list order in both arms.** Rank decides
   *which* passages survive the budget; it does not reorder what the generator
   sees.
5. **The frozen manifest's hash is asserted before the run**, so the
   evaluation set cannot change after outcomes are seen.
6. **λ, θ and H are fitted on the validation split and frozen before any test
   run.** `AdmissionConfig.validate()` raises rather than supply a default.

## 9. Reporting requirements

* Every rate is reported both conditional on answering and with abstentions
  counted, alongside answer coverage.
* The baseline cannot abstain, so the proposed system answers always by
  default (`AbstentionPolicy.ANSWER_ALWAYS`); see
  `docs/experimental_parity_audit.md` §2.
* Statistical comparison: exact McNemar on the paired discordant answers,
  paired bootstrap CI resampling questions, Holm correction across the two
  outcomes.
