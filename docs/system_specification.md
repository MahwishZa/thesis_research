# System Specification — what each arm does, and what differs

**Written 2026-09-17** by reading the implementation, not the design
documents. Every claim below was checked against the code it names.

This is the answer to "what exactly is being compared", stated once, so the
thesis's fairness argument does not have to be reassembled from three
documents and a test suite.

---

## 1. The comparison in one line

Both arms receive **the same question and the same frozen candidate evidence**,
and both answer with **the same generator under the same decoding settings and
the same prompt**. They differ in **which subset of that evidence reaches the
generator**, and in nothing else.

## 2. Shared upstream (identical by construction)

Performed **once per question** and serialised before either arm runs:

```
question → rationale → MedCPT retrieval → MedCPT reranking → candidate set → FROZEN
```

Implementation: `experiments/retrieval/` builds it, `experiments/evaluation/
freezing.py` freezes it. The frozen `FrozenItem` carries the candidate list,
its order, each passage's `rerank_score` and `rerank_rank`, publication dates,
the `corpus_snapshot` id, and an order-sensitive `candidate_set_hash`.

Neither arm retrieves, re-ranks, or re-scores anything. `RAG2System` records
`retrieval_external: True` precisely because stages 1–2 are not its job.

## 3. Baseline — `systems/baseline/rag2.py`, `name = "B2_RAG2"`

```
frozen candidates
  → for each candidate: Flan-T5 filter → [HELPFUL] | [NOT_HELPFUL]
  → keep the [HELPFUL] ones
  → budget: keep the best rerank_rank, ties broken by evidence_id
  → build context in candidate-list order
  → generate
```

**The filter** (`systems/baseline/admission.py`) scores one `(question,
passage)` pair at a time using RAG²'s verbatim prompt template, takes a
two-way softmax over the `[HELPFUL]` / `[NOT_HELPFUL]` label-token logits at
the first decoder position, and labels by argmax. It sees **no date, no rank,
no metadata** — only the question and the passage text. That is faithful to
the published method and it matters here: the baseline cannot respond to
recency even in principle, which is what makes the contrast interpretable.

**Budget ordering is by reranker rank** because RAG²'s filter emits a binary
label, not a ranking, so rank is the only ordering available to it.

## 4. Proposed solution/system — `systems/proposed/admission.py`, `name = "P_RECENCY"`

```
frozen candidates
  → for each candidate: A(s) = (1 − λ)·ρ(s) + λ·R(s, q, t_q)
  → admit if A(s) ≥ θ
  → budget: keep the highest A(s), ties broken by evidence_id
  → build context in candidate-list order
  → generate
```

* `ρ(s)` — `AdmissionScorer.normalize_rank`: rank-normalised reranker score in
  [0, 1], best rank scoring 1.0. **The same signal the baseline's upstream
  produced.**
* `R(s, q, t_q)` — `RecencyPolicy`: `2^(−age_days / H)`, age clamped at zero,
  in (0, 1].
* `A(s)` is therefore in [0, 1] and θ is directly interpretable on that scale.

**Verified unchanged.** The implementation matches the formula in
`docs/frozen_scope.md` §4 exactly: one weight, two components, no hidden
rescaling. Secondary machinery (contested evidence, supersession, retraction,
ψ, verification) is present, `None`/off by default, and records itself in run
metadata when enabled.

## 5. The independent experimental factor

**The admission rule, and only that.**

| | Baseline | Proposed |
|---|---|---|
| Decides admission by | learned binary label from passage text | `A(s) ≥ θ` over rank + recency |
| Can see publication date | **no** | **yes** |
| Orders the budget by | reranker rank | `A(s)` |

Everything a difference in HAR could otherwise be attributed to is held
constant — and, since the earlier parity audit, checked in code rather than
assumed:

| Held constant | Enforced by |
|---|---|
| Question and question id | one value passed to every arm by the runner |
| Candidate evidence and its order | `assert_same_candidate_sets` + `candidate_set_hash` |
| Context budget | `assert_budget_parity` |
| Prompt template | `assert_prompt_parity` (defaults are byte-identical) |
| Generator instance and decoding | `assert_generator_parity` + one `RunConfig` |
| Context ordering | both arms emit candidate-list order; rank decides *which*, never *where* |
| Abstention behaviour | `ANSWER_ALWAYS` by default, since the baseline cannot abstain |
| Output schema | both return `ExperimentResult` |

## 6. What reaches generation

Both arms build context as `[evidence_id] text`, joined by blank lines, **in
candidate-list order**, substituted into the shared `{question}` / `{context}`
template. The runner records `admitted_evidence_ids` and
`admitted_evidence_text` per answer, so the annotator judges each answer
against the evidence that answer actually saw — which is what the HAR
definition requires.

## 7. Causal pathway under test

1. Admission decides which passages reach the generator.
2. A generator given superseded evidence can produce a fluent claim that
   current evidence contradicts.
3. Weighting recency at admission changes which evidence is present.
4. HAR is defined against the evidence supplied, so a change in (3) can move
   the primary outcome.

This is the whole mechanism. The thesis does not claim the pathway is the only
one, and does not measure the intermediate steps as outcomes.

## 8. Parameters

### Fitted on the validation split, never on test

| Parameter | Meaning | Status |
|---|---|---|
| `λ` | recency weight | **unresolved by design** — `AdmissionConfig.validate()` raises rather than default |
| `θ` | admission threshold | same |
| `H` | recency half-life (days) | same |

There is no default value any of these could silently inherit, and no code
path reads test outcomes. `λ = 0` is the built-in pure-relevance ablation.

### Decided here (engineering constants, not fitted quantities)

| Parameter | Value | Reasoning |
|---|---|---|
| `max_admitted_passages` (context budget) | **5** | The budget is a *control*, not a treatment: it exists so both arms answer from the same amount of evidence. 5 matches the context budget the earlier design already specified across arms and is comfortable for a 7B generator's context window alongside the prompt. Applied identically to both arms and asserted by `assert_budget_parity`. |
| Candidate-set size `N` | **20** | Fixed and identical for every item, because `ρ` is a within-set rank: θ is only comparable across questions at constant N. 20 gives the admission rule something to discriminate at a budget of 5 without making the reranker pass expensive. |
| Retrieval depth before reranking | **50** | Reranked down to the 20 kept. Deep enough that reranking is doing real work, shallow enough to stay cheap. |
| `undated_score` | not a primary parameter | Candidate sets contain only dated passages, so the undated branch never fires. Every run reports its `UNDATED` count; in a valid run it is zero. |

These four are engineering choices with defensible reasons, not empirical
findings, and are recorded as such. They are set **before** any outcome is
seen and do not change between arms or between runs.

## 9. Unresolved

1. **λ, θ, H are unfitted.** They are fitted on the validation split once real
   candidate evidence exists. This is the intended pre-experiment state.
2. **The baseline filter checkpoint does not exist yet** — see
   `docs/filter_training.md`. Until it does, the baseline arm runs only
   against `MockRAG2Filter`, which is engineering validation and never a
   result.
3. **Corpus is still building**, so no real candidate set exists to freeze.

None of these is a design ambiguity. Each is a dependency with a known
resolution.
