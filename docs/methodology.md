# Methodology

Answers **what is being compared, how, and under what conditions.** For the
research question and objectives, see the root `README.md`. For corpus and
question-pool provenance, see `data.md`. For metrics and statistics, see
`evaluation.md`. For how to run any of this, see `reproducibility.md`.

## 1. The comparison in one line

Both systems receive **the same question and the same frozen candidate
evidence**, and both answer with **the same generator under the same
decoding settings and the same prompt**. They differ only in **which subset
of that evidence reaches the generator**.

## 2. Shared upstream (identical by construction)

Performed once per question and frozen before either system runs:

```
question → MedCPT retrieval → MedCPT reranking → candidate set → frozen
```

Code: `experiments/shared/retrieval/` builds it, `src/evaluation/freezing.py`
freezes it. The frozen item carries the candidate list, its order, each
passage's rerank score and rank, publication date, a corpus-snapshot id, and
an order-sensitive candidate-set hash. Neither system retrieves, re-ranks, or
re-scores anything at comparison time.

Retrieval is MedCPT at both stages (dual-encoder for retrieval,
cross-encoder for reranking) over this thesis's Alzheimer's corpus, a
domain-scoped substitution for RAG²'s own general-medical corpus. The index
is a flat exact inner-product search, not an approximate one, so recall
cannot depend on index build order.

## 3. Baseline — RAG²

```
frozen candidates
  → for each candidate: Flan-T5 filter → [HELPFUL] | [NOT_HELPFUL]
  → keep the [HELPFUL] ones
  → budget: keep the best rerank_rank, ties broken by evidence_id
  → build context in candidate-list order
  → generate
```

The filter (`src/baseline/admission.py`) scores one `(question, passage)`
pair at a time using RAG²'s published prompt template and label tokens,
taking a two-way softmax over the `[HELPFUL]`/`[NOT_HELPFUL]` logits. It
sees **no date, no rank, no metadata** — only the question and passage
text. That is faithful to the published method, and it is what makes the
comparison interpretable: the baseline cannot respond to temporal
information even in principle.

Budget ordering is by reranker rank, because the filter emits a binary
label rather than a ranking.

**Filter training.** RAG²'s checkpoint is not distributed, so the filter is
retrained by this project on general-medical MedQA data (never on the
thesis's own Alzheimer's questions), using the paper's own label-generation
recipe (correctness-flip decision tree with a perplexity-differential
tie-break) and Flan-T5 hyperparameters, with any deviation stated where it
occurs. Code: `experiments/baseline/filter_training/`.

## 4. Proposed system — the Temporal Filter

```
frozen candidates
  → for each candidate: A(s) = (1 − λ)·ρ(s) + λ·T(s, q, t_q)
  → admit if A(s) ≥ θ
  → budget: keep the highest A(s), ties broken by evidence_id
  → build context in candidate-list order
  → generate
```

| Symbol | Meaning |
|---|---|
| `ρ(s)` | Relevance — rank-normalised reranker score in [0, 1], the same signal the baseline's upstream produces |
| `T(s, q, t_q)` | Temporal score — `2^(−age_days / H)`, in (0, 1] |
| `λ` | Weight on the temporal term vs. relevance (0 to 1) |
| `θ` | Admission threshold |
| `H` | Half-life in days |

One weight, two components — a single weight removes a redundant degree of
freedom and makes `λ = 0` a built-in, pure-relevance ablation rather than a
fourth arm. Code: `src/proposed/`.

## 5. Control — No Filter

A third arm (`src/baseline/no_filter.py`) that admits every retrieved
passage up to the context budget, with no filtering at all. It is the
honest floor: if a trained filter does not beat admitting everything, an
advantage for the proposed system must be read against that, not against
the baseline alone.

## 6. What is held constant, and how it's enforced

| Held constant | Enforced by |
|---|---|
| Question and question id | one value passed to every arm by the runner |
| Candidate evidence and its order | candidate-set-hash assertion |
| Context budget | budget-parity assertion |
| Prompt template | prompt-parity assertion |
| Generator instance and decoding | generator-parity assertion + one run config |
| Context ordering | all arms emit candidate-list order; rank decides *which* passages survive, never *where* they sit |
| Output schema | all arms return the same result type |

All assertions run before the first item in a run, not after. Code:
`src/evaluation/runner.py`; tests: `tests/unit/test_runner_parity.py`.

## 7. Parameters

`λ`, `θ`, `H` are fitted on a held-out **validation** split and frozen
before any test-split run — there is no default value they could silently
inherit, and no code path reads test outcomes while fitting. `λ = 0` is the
ablation's "component removed" condition.

Engineering constants (decided, not fitted): context budget = 5 admitted
passages (a control, not a treatment — identical for both arms); candidate
set size N = 20 (fixed so `θ` is comparable across questions); retrieval
depth before reranking = 50.

## 8. Ablation

`λ = 0` removes the temporal term and keeps everything else the same,
isolating whether the temporal signal specifically is what helps, as
opposed to filtering in general (which the No-Filter control already
separates out).

## 9. Generator

Target model: `meta-llama/Meta-Llama-3-8B-Instruct` (RAG²'s own generator),
4-bit NF4 quantised, greedy decoding (no sampling, no seed — a repeat run
reproduces exactly), on a free-tier remote GPU. A pinned commit sha is
required for the model revision; `"main"` is refused, since a branch name
resolves to different weights over time. Declared fallback if the licence
cannot be obtained: `Qwen/Qwen2.5-7B-Instruct` (ungated, same parameter
scale), stated as a substitution rather than made silently. Interface:
`src/common/hf_generator.py`.

## 10. Statistical procedure

See `evaluation.md` for the full metrics and statistics design; the
headline point here is methodological: whether a difference between systems
is real is decided by a **paired significance test** over per-question
outcomes, not by the size of an average gap. An average can look large or
small while still being indistinguishable from chance; a test that looks at
every question's direction independently can tell the two apart.
