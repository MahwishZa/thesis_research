# Frozen Scope — canonical reference for Stage 3 onward

Frozen 2026-09-16 at the close of the design review. Where any other document
disagrees with this one about what is *primary*, this one governs. Stage 3
begins from here; it does not reopen the architecture.

---

## Primary research question

> Does a confidence-derived evidence admission mechanism exhibit recency
> asymmetry, and does explicitly incorporating evidence recency reduce that
> asymmetry without simply degrading answer quality?

## Primary measurement (C1)

Admission asymmetry over matched temporal-counterfactual pairs:

```
Δ = P(admit | older passage) − P(admit | newer passage)
```

measured per arm, with the matched pair as the unit of analysis. Δ > 0 means
older evidence is preferentially admitted.

The RAG² filter receives only `(question, passage text)` at inference — no
dates, no ranks, no metadata. Any Δ it shows must therefore come from content
correlates of era, which is what makes the measurement meaningful and what
insulates the primary claim from rank and ordering artifacts.

## Three primary arms

One candidate set per item, built once, frozen, replayed byte-identically.

| Arm | Admission rule | Sees dates? |
|---|---|---|
| `B1_NO_FILTER` | Admit everything, up to the budget | no |
| `B2_RAG2` | Flan-T5 `[HELPFUL]` / `[NOT_HELPFUL]` | no |
| `P_RECENCY` | `A(s) ≥ θ` | **yes** |

What differs between arms is the admission rule and nothing else. Identical
across all three: question, candidate set and its order, reranker scores,
context budget, prompt template, generator and decoding settings.

## Primary proposed method

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
never on test. `λ = 0` is an **internal ablation**, not a fourth arm.

## Secondary / future components

Present in the repository, off by default, never part of a primary result:
contested-evidence detection, supersession discounting, retraction exclusion,
time-invariance (ψ), answer verification, entailment-derived support,
source authority. Enabling any of them is a secondary analysis and must be
declared as such; each run records which were active.

## Explicitly outside primary scope

Clinician rating study · comparison against a further state-of-the-art filter ·
agentic reference arm · the supersession table · claim-class-dependent
inclusion rules · retrieval-quality optimisation.

---

## Interface requirements Stage 3 must satisfy

1. **Ranks are 1..N contiguous after injection.** The evaluation pair is
   injected into the retrieved set, so the combined set must be re-ranked;
   the pair does not keep ranks from elsewhere. `normalize_rank` raises
   otherwise.
2. **N is identical for every item.** `ρ` is a within-set rank, so `θ` is only
   comparable across items at fixed candidate-set size.
3. **Candidate sets contain only dated passages.** Applied identically to all
   three arms at construction. This keeps the undated branch of the recency
   score from ever firing, so `undated_score` is not a primary parameter and
   the tunable count stays at three. Every run reports its count of `UNDATED`
   passages; in a valid primary run that count is zero.
4. **Context order is candidate-list order in all three arms.** Rank decides
   *which* passages survive the budget; it does not reorder what the generator
   sees.
5. **The frozen pair file's SHA-256 is asserted before the run**
   (`verify_frozen`), so the evaluation set cannot change after outcomes are
   seen.

## Controls

* **Negative control (blocking):** the unchanged-claim set. Same construction,
  same matching, but the claim did not change across the window. Δ ≈ 0 there
  alongside Δ > 0 on changed-claim pairs is what separates claim recency from
  prose-era style. The permutation control it replaces is invalid here: no
  filter under test receives a date, so permuting dates changes no output and
  drives Δ to zero by construction rather than by evidence.
* **Ablation (λ = 0):** pure relevance at the same θ, isolating what the
  recency signal contributes over thresholding relevance alone.
* **Rank-balance diagnostic (free):** report the rerank-rank distribution of
  older vs. newer passages. Bears on the recency-aware arm and on budget
  competition; the RAG² measurement is rank-blind by construction.
* **Dual reporting:** every rate reported both conditional on answering and
  with abstentions counted as failures.

## Backbones

**One filter backbone is the primary experiment.** A second is an *optional*
robustness check. The central question is whether this confidence-derived
signal shows recency asymmetry; one faithful reproduction answers it. A second
backbone would upgrade the claim from "this filter, as trained here" to "the
signal family", which is a generalisation, not the thesis. The RAG²
checkpoint is not distributed, so each backbone costs a full filter training.

If only one is run, the claim is narrowed explicitly and the single-backbone
limitation stated — not quietly dropped.
