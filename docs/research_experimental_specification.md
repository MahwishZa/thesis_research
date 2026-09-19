# Research and Experimental Specification

**Authoritative method specification. Rewritten 2026-09-19** by consolidating
the eight separate design documents that previously described it
(`system_specification.md`, `generator_contract.md`, `filter_training.md`,
`rag2_classifier_feasibility.md`, `experimental_parity_audit.md`,
`methodology.md`, `question_sources.md`, `external_evaluation_data.md`) plus
the still-live interface and reporting requirements from `frozen_scope.md`.
Every claim below was checked against the code it names.

This document answers **what is being compared, how, and under what
conditions**. It contains no results — none exist. For *what the thesis is
for*, see `docs/current_objectives.md`, which governs scope. For *what has
actually been executed, decided and measured*, see
`docs/status_and_decisions.md`.

---

## 1. The comparison in one line

Both arms receive **the same question and the same frozen candidate
evidence**, and both answer with **the same generator under the same decoding
settings and the same prompt**. They differ in **which subset of that evidence
reaches the generator**, and in nothing else.

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

RAG² uses MedCPT at both stages — a dual-encoder for retrieval, a
cross-encoder for reranking (established fact E6 in
`docs/status_and_decisions.md`). The thesis keeps both, over its own
Alzheimer's corpus rather than RAG²'s 564 GB general-medical one. That
substitution of *corpus* is the declared adaptation; the *method* is
unchanged. The index is a **flat exact** inner-product search, not an
approximate one (decision D-37): an approximate index introduces
build-order-dependent recall, and a domain-slice corpus is small enough that
exact search removes that reproducibility hazard for free.

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

A no-filter control (`systems/baseline/no_filter.py`, `name = "B1_NO_FILTER"`)
admits everything up to the budget. It is not a research objective; it is the
**honest floor** — if a trained filter does not beat admitting everything, the
baseline was weak, and any advantage for the proposed system must be read
against that rather than celebrated.

## 4. Proposed system — `systems/proposed/admission.py`, `name = "P_RECENCY"`

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
  produced.** Rank-based rather than min-max normalisation, so a single global
  θ is well defined across questions (D-12).
* `R(s, q, t_q)` — `RecencyPolicy`: `2^(−age_days / H)` where
  `age_days = t_q − publication_date(s)`, clamped at zero, in (0, 1].
  **Plain age decay and nothing else** (D-28).
* `A(s)` is therefore in [0, 1] and θ is directly interpretable on that scale.

One weight, two components, no hidden rescaling — a single weight removes the
redundant degree of freedom two free weights would give, and makes `λ = 0` a
built-in pure-relevance ablation rather than a fourth arm (D-27, D-30).

Secondary machinery (contested evidence, supersession, retraction, ψ,
verification) is present, `None`/off by default, and records itself in run
metadata when enabled. Enabling any of it is a declared secondary analysis,
never part of a primary result.

## 5. The independent experimental factor

**The admission rule, and only that.**

| | Baseline | Proposed |
|---|---|---|
| Decides admission by | learned binary label from passage text | `A(s) ≥ θ` over rank + recency |
| Can see publication date | **no** | **yes** |
| Orders the budget by | reranker rank | `A(s)` |

Everything a difference in outcome could otherwise be attributed to is held
constant — and checked in code rather than assumed:

| Held constant | Enforced by |
|---|---|
| Question and question id | one value passed to every arm by the runner (`sample_id` set once per item) |
| Candidate evidence and its order | `assert_same_candidate_sets` + `candidate_set_hash` |
| Context budget | `assert_budget_parity` |
| Prompt template | `assert_prompt_parity` (defaults are byte-identical) |
| Generator instance and decoding | `assert_generator_parity` + one `RunConfig` |
| Context ordering | both arms emit candidate-list order; rank decides *which*, never *where* |
| Abstention behaviour | `ANSWER_ALWAYS` by default — see §7 |
| Output schema | all arms return `ExperimentResult` |

All four assertions run in `run_experiment` **before the first item**, not
after. Tests: `tests/unit/test_runner_parity.py`.

Two of these were documented but previously unchecked, and both were real
hazards rather than hypothetical ones:

* **Context budget.** Each arm stores its cap in a different place — on the
  system itself (`NoFilterSystem`), on `config` (`RAG2System`), on
  `admission_policy.config` (`RecencyAwareSystem`). Nothing compared them. An
  arm allowed more passages answers from more context, so a difference in
  outcome would be attributable to context volume rather than to the admission
  rule. `context_budget()` resolves the value through all three shapes; an arm
  with no cap reads as `None` rather than as some number, so an uncapped arm
  cannot be mistaken for a capped one. The repository's own smoke fixture had
  obtained "arms that admit different subsets" precisely by encoding this
  confound; it now differs by admission rule instead.
* **Generator.** `RunConfig` records one model, one model version and one
  generation config and stamps them onto every result record. Different
  generator objects would make that metadata silently mislabel which model
  produced which answer — worse than an unchecked difference, because the
  output would look correct. `assert_generator_parity` compares object
  identity, which is what the `Generator` interface exposes and what a real
  run does: load once, share.

## 6. What reaches generation

Both arms build context as `[evidence_id] text`, joined by blank lines, **in
candidate-list order**, substituted into the shared `{question}` / `{context}`
template. The runner records `admitted_evidence_ids` and
`admitted_evidence_text` per answer, so every answer can be scored against the
evidence that answer actually saw.

## 7. Abstention — the one resolved asymmetry

`RecencyAwareSystem.run()` used to return `prediction=None`,
`output_state=ABSTAIN` when no passage cleared θ. `RAG2System.run()` has no
empty-evidence guard: when its filter admits nothing it builds a prompt with an
empty evidence block and generates anyway.

**Abstention is not an intended component.** It fires only in the degenerate
case where the threshold admits nothing, it is not part of the scoring rule,
and the original RAG² has no abstention mechanism at all. Left in place it is
metric gaming by construction: an abstention makes no claims, so a θ set high
enough drives any faithfulness rate to zero while answering nothing.

**Decision.** `AbstentionPolicy.ANSWER_ALWAYS` is the **default** — the
proposed system generates from whatever it admitted, including nothing,
exactly as the baseline does. `ABSTAIN_WHEN_EMPTY` remains available as a
declared secondary condition, never the primary comparison. Under
`ANSWER_ALWAYS`, an answer produced from an empty evidence block is recorded
as `output_state=UNGROUNDED` — not `GROUNDED`, which would be false, and not
`ABSTAIN`, because an answer was produced. `experiments/evaluation/stats.py`
always reports `answered`, `abstained`, `hallucinated` and `non_hallucinated`
separately with `answer_coverage` beside every rate, and `compare_systems()`
sets `interpretable: false` and refuses a headline difference when either
system answered nothing.

Two alternatives were considered and rejected: allowing abstention with
coverage reported separately (sound, but makes the outcome a pair with no
pre-declared exchange rate), and counting abstention as failure (ungameable,
but conflates "declined to answer" with "answered wrongly" in one
denominator). Option A is the smallest change that makes the comparison fair,
and Option B's accounting is reported alongside it regardless.

Tests: `AbstentionAccountingTests`, `AbstentionPolicyConfigTests` — including
that a system abstaining on every question yields `interpretable: false` and
no headline difference.

## 8. Parameters

### 8.1 Fitted on the validation split, never on test

| Parameter | Meaning | Status |
|---|---|---|
| `λ` | recency weight | **unresolved by design** — `AdmissionConfig.validate()` raises rather than default |
| `θ` | admission threshold | same |
| `H` | recency half-life (days) | same |

`validate()` raises with *"Fit it on the validation split before any test
run"* if θ is unresolved. There is no default value any of these could
silently inherit, and no code path reads test outcomes. **They are not tuned
on the test set, and cannot be by accident.** `λ = 0` is the built-in
pure-relevance ablation.

The evaluation question pool is likewise never used to select θ, λ, H,
generation settings, prompt wording or retrieval parameters.

### 8.2 Engineering constants (decided, not fitted)

| Parameter | Value | Reasoning |
|---|---|---|
| `max_admitted_passages` (context budget) | **5** | The budget is a *control*, not a treatment: it exists so both arms answer from the same amount of evidence. Comfortable for a 7–8B generator's context window alongside the prompt. Applied identically to both arms and asserted by `assert_budget_parity`. |
| Candidate-set size `N` | **20** | Fixed and identical for every item, because `ρ` is a within-set rank: θ is only comparable across questions at constant N. 20 gives the admission rule something to discriminate at a budget of 5 without making the reranker pass expensive. |
| Retrieval depth before reranking | **50** | Reranked down to the 20 kept. Deep enough that reranking is doing real work, shallow enough to stay cheap. |
| `undated_score` | not a primary parameter | Candidate sets contain only dated passages (§15.3), so the undated branch never fires. Every run reports its `UNDATED` count; in a valid run it is zero. |

These are engineering choices with defensible reasons, not empirical findings.
They are set **before** any outcome is seen and do not change between arms or
between runs. Cited from code at `experiments/retrieval/pipeline.py`.

## 9. Generator execution contract

**Decided 2026-09-17 (D-38). Both arms run under it; nothing in it differs
between them.**

**Model: `meta-llama/Meta-Llama-3-8B-Instruct` — RAG²'s own generator, kept.
Venue: free-tier remote GPU (Colab or Kaggle T4, 16 GB). Precision: 4-bit
NF4.**

The local machine (RTX 2050, 4 GB VRAM) cannot run an 8B model at any
precision — BF16 ≈ 18–20 GB, 8-bit ≈ 10 GB, 4-bit NF4 ≈ 6–8 GB including KV
cache at RAG-length contexts, all against 4 GB. CPU-only inference is
technically possible and practically unusable for a run that has to be
repeatable.

**That rules out the venue, not the model.** The tempting move is to swap in a
model small enough for 4 GB; it is the wrong one, because it would trade away
fidelity to RAG² to solve a problem a free T4 already solves. Quantisation is
the one deviation from the paper, it is a precision change applied
**identically to both arms**, and it is reported as a limitation.

**Declared fallback:** Llama-3 is gated on Hugging Face. If the licence cannot
be obtained, `Qwen/Qwen2.5-7B-Instruct` (ungated, same parameter scale)
substitutes, and the substitution is stated in the thesis rather than made
quietly. Accepting the licence is a student action, not a technical blocker.

### 9.1 Greedy decoding, and why it is not a knob

`do_sample=False`, no temperature, no top-p, no top-k.

With sampling, one run per arm would be a *sample* from a distribution, and
separating a real difference from decoding variance would need many runs per
question — which the compute budget does not allow. With greedy decoding, one
run per arm **is** the measurement, and a repeat run reproduces it exactly.

`GenerationConfig` raises if `do_sample=True` rather than permitting it
quietly, and raises if temperature or top-p are set while greedy, because
recording parameters the run ignored would misdescribe it.

There is no random seed, because nothing samples. A seed that does not matter
is worse than no seed: it implies a control that is not there.

### 9.2 The contract

| Field | Value | Recorded in |
|---|---|---|
| Model id | `meta-llama/Meta-Llama-3-8B-Instruct` | `ModelSpec.model_id` |
| Revision | **a commit sha — pinned at download** | `ModelSpec.revision` |
| Quantization | `nf4`, double quant, bf16 compute | `ModelSpec.quantization` |
| Prompt template | shared; `assert_prompt_parity` | `RunConfig` |
| Chat template | the checkpoint's own, applied to the built prompt | generator metadata |
| Decoding | greedy, `max_new_tokens=256` | `GenerationConfig` |
| Context budget | 5 admitted passages, both arms | `assert_budget_parity` |
| Generator instance | **one object, shared** | `assert_generator_parity` |
| Software versions | transformers, torch, bitsandbytes, python | run manifest |

**`revision` must be a commit sha and `ModelSpec` refuses `"main"`.** A branch
name resolves to different weights over time, so a result recorded against one
could not be reproduced. The sha is read off the Hub at download time and
written into the run manifest. It is deliberately **not** pre-filled in the
code: a placeholder would be a fabricated provenance record.

`experiments/runners/run_end_to_end.py --real-model` requires
`--model-revision` for this reason and exits rather than defaulting it;
`--quantization` defaults to `nf4`, because a full-precision 8B load is the
one configuration the documented venue cannot run.

### 9.3 Execution procedure

1. Accept the Llama-3 licence on Hugging Face; create a read token.
2. Open a T4 session. Install pinned `transformers`, `bitsandbytes`,
   `accelerate`.
3. Download the model **recording the resolved commit sha**.
4. **Timing check:** generate 5 answers, record wall-clock and tokens/second.
   This is the first measurement and it decides whether the full run fits the
   session limit. It is engineering measurement, not a result.
5. Fit λ, θ, H on the **validation split only**.
6. Run every arm over the frozen test manifest, one shared generator object.
7. Download the raw JSONL. It is never overwritten (`run_experiment` refuses
   an existing path).

**No tokens/second figure appears anywhere in this repository until one is
produced on the actual venue.** Generation speed has not been measured.

## 10. RAG² filter training

**Decided 2026-09-17 (D-39, D-40). DECIDED AND IMPLEMENTED. NOT TRAINED —
no checkpoint exists and no label has been generated.**

### 10.1 Why the filter must be trained at all

RAG²'s contribution is a filter trained on perplexity-derived helpfulness
labels. The checkpoint is not distributed (E9), so an identical baseline is
impossible for anyone, not just for this thesis. The baseline is therefore an
**adaptation**, and the thesis says so.

The official repository's `classifier/data/medqa/llama3_cot/5%-train.json`
looks like the paper's 5% training split. **It was downloaded and inspected:
it contains 5 examples.** The ids run to `llama3_5%_23600`, so the real split
held roughly 23,600 — the released file is an illustration of the format. This
settles the strategy: labels cannot be obtained, only regenerated.

### 10.2 The label function — kept exactly

`experiments/filter_training/labeling.py` implements the paper's decision tree
(§3.2, Fig. 2, Eq. 3):

1. Answer the question **without** the passage → correct or not.
2. Answer it **with** the passage → correct or not.
3. **Correctness flip decides:** wrong → right is `[HELPFUL]`; right → wrong is
   `[NOT_HELPFUL]`.
4. **Unchanged correctness falls back to the perplexity differential** of the
   generated *rationale* (not the query — E3). Top τ = 0.25 of reductions is
   `[HELPFUL]`.

Two details that are easy to get wrong and are locked by tests: **a flip
always outranks perplexity** (a large perplexity gain cannot rescue a passage
that turned a right answer wrong), and **the τ quantile is computed only over
the pairs the tie-break actually judges** — the unchanged ones.

**τ is a property of the reference method, not a thesis parameter.** Fixed at
0.25, never fitted, tuned or swept. Fitting it would make the baseline
something this thesis chose rather than something the paper specifies.

MedQA is multiple-choice, so correctness is checked automatically. **No human
annotation is involved and none is planned.**

### 10.3 The recipe, read from the official repository

Source: `classifier/README.md`, `classifier/run_classifier.py`,
`classifier/run/run_large_train_xl_000.sh`, read directly on 2026-09-17.
Nothing here is from memory.

The filter is a **Flan-T5 seq2seq model** fine-tuned with HF Accelerate. The
two labels are added as special tokens and the embedding matrix is resized:

```python
new_tokens = ["[HELPFUL]", "[NOT_HELPFUL]"]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
```

| Setting | Value | Source |
|---|---|---|
| Base model | a Flan-T5 checkpoint with the label tokens added | `MODEL=/classifier/model/updated_flan_t5_model` |
| Learning rate | **3e-5** | launch script |
| Optimizer | **AdamW** | `run_classifier.py:528` |
| Max sequence length | **512** | launch script |
| Doc stride | **128** | launch script |
| Train batch size per device | **16** | launch script |
| Epochs | **40** | launch script |
| Checkpointing | **per epoch** | launch script |
| Weight decay | **0.0** (default) | `run_classifier.py:283` |
| Warmup steps | **0** (default) | `run_classifier.py:306` |
| Gradient accumulation | configurable, default **1** | `run_classifier.py:293` |
| LR scheduler | configurable (HF default linear) | `run_classifier.py:299` |
| Precision | **not hardcoded** — from `accelerate config` | `accelerator.use_fp16`, line 499 |

Training data format — a JSON list, one object per example, with `id`,
`answer` (the label token), `dataset_name` and `question` (the rendered
prompt). **Inference** takes a softmax over the two label-token logits and
predicts the higher-probability label — exactly what
`systems/baseline/admission.py` implements.

**One ambiguity, recorded rather than resolved.** The launch script sets
`MODELNAME=flant5` and points at a generic directory; its filename mentions
both "large" and "xl". The paper record states Flan-T5-large (≈770–780 M).
**The released script does not pin the base size.** Confirm against the paper
before stating a size in the thesis.

### 10.4 Deviations, each with its reason

`experiments/filter_training/config.py`. Every value there is either the
paper's or a deviation that states itself, side by side in one object, so a
deviation cannot be made without appearing in the training report.

| Deviation | Reason |
|---|---|
| Per-device batch **4**, accumulation **4** (paper: 16 on one device) | A free 16 GB T4 cannot hold batch 16 at seq 512 for a 770M seq2seq model. **The effective batch stays 16** — `validate()` refuses a configuration where it does not, so accumulation cannot be used to quietly shrink it. |
| Epochs **below 40** | Free-session limits. The count is deliberately **unset in code**: `validate()` raises rather than default it, because what fits depends on the labelled-set size. The count actually run is reported. |
| **Subsampled training set** | Label generation needs two rationale generations per (question, passage) pair, so cost is linear in set size. The size is chosen from the §9.3 timing measurement — a budget, not a target. |

**The base model is not reduced.** Dropping to Flan-T5-base is the obvious
economy and is not taken: the venue is already a free T4, where the paper's
own size trains, so shrinking it would give up fidelity to solve a problem
that no longer exists.

Training Flan-T5-large locally is ruled out by arithmetic, not tuning: ≈3.1 GB
fp32 master weights + ≈3.1 GB gradients + ≈6.2 GB AdamW moments ≈ **12.4 GB
before a single activation**, against 4 GB of VRAM — and batch 1 with
accumulation does not reduce the optimizer footprint. CPU training on 16 GB
RAM would not crash but would take days to weeks for 40 epochs. **Filter
*inference* runs locally** (≈1.6 GB fp16 fits 4 GB); only the one-off training
moves to a remote GPU.

### 10.5 Before a checkpoint may produce thesis numbers

`CheckpointRecord` requires all of: base model, training and validation set
sizes, validation accuracy, epochs actually run, label distribution.

* `is_usable()` returns False at or below chance (0.5). A binary classifier at
  chance has not learned the label function, and a baseline built on one would
  be **broken rather than weak** — which changes what a difference means.
* `label_distribution()` is checked **before** training: a set that is 95% one
  label teaches the prior, and that is better caught in the data than
  diagnosed afterwards from a bad validation number.
* `FlanT5RAG2Filter` refuses to run without an explicit checkpoint path, and
  validates that each label token maps to exactly one token id.
* `MockRAG2Filter` states in its own docstring that it must never produce
  thesis performance results, and `run_end_to_end.py` labels every report it
  appears in `MockRAG2Filter(all-HELPFUL stand-in; NO trained checkpoint)`
  with `baseline_is_trained_rag2: false`.

### 10.6 Procedure and contamination control

1. Open a T4 session; install pinned dependencies.
2. Run the generator timing check (§9.3).
3. Choose the labelled-set size from that measurement.
4. Generate labels over a MedQA subsample with the paper's decision tree.
   Write with `write_training_file()`, which refuses to overwrite — labels
   cost GPU hours and a silent rerun would destroy a checkpoint's provenance.
5. Check `label_distribution()` before training.
6. Train Flan-T5-large with the config above; record epochs actually run.
7. Record validation accuracy in a `CheckpointRecord`.
8. Download the checkpoint; run inference locally.

The filter is trained on **general-medical** MedQA, never on the thesis's
Alzheimer's evaluation questions. This is the paper's own setup (D-13) and it
also removes any suspicion that the baseline was tuned on the evaluation set.
The 123-candidate question pool plays no part in filter training.

**What the thesis must state:** the checkpoint was unavailable · the filter
was retrained by the student · the base size used · the epoch count and
validation accuracy actually reached · the training-set size and that it is a
subsample · that training ran on different hardware from the rest of the
pipeline · **that this makes the baseline an adaptation of RAG², not a
reproduction.**

## 11. Evaluation metrics

`experiments/evaluation/rag_metrics.py`, stdlib only (matching `stats.py`'s
no-SciPy convention). These are the metrics the main evaluation (objective 3)
and the ablation study (objective 2) are scored with.

| Metric | Definition |
|---|---|
| `exact_match` | Normalized string equality against the reference answer (SQuAD normalization: lowercase, strip punctuation and articles, collapse whitespace). |
| `token_f1` | SQuAD-style unigram precision/recall/F1 over multiset token overlap. 1.0 when both sides are empty, 0.0 when exactly one is. |
| `rouge_l_f1` | Standard ROUGE-L: F1 over the longest common in-order subsequence. |
| `context_precision` / `context_recall` / `context_f1` | How well the admitted evidence ids match the question's gold-relevant evidence ids. |
| `groundedness` | Fraction of the answer's content tokens (stopwords removed) that appear in the admitted evidence text. |

**`groundedness` is an automatic PROXY for faithfulness, not an entailment
judgement.** Token overlap has no access to entailment, negation or
paraphrase, so a high score does not certify that a claim is supported and a
low score does not certify that it is not. It is a cheap, deterministic
diagnostic for comparing arms against each other.

**Context metrics return `None`, never 0.0, when a question carries no
gold-evidence annotation.** This is load-bearing rather than a nicety: under
the provenance firewall (§15.7) an externally-authored question's reference
answer is deliberately independent of the retrieved candidate set, so absent
gold evidence ids are the **expected** case on real data. Scoring that 0.0
would report "context precision 0.000" for every arm — which reads as a real,
uniformly terrible result — and would do so identically for baseline and
proposed, quietly diluting the comparison. `aggregate()` skips `None`,
averages over only the annotated rows, and reports `context_scored_n` so a
reader can tell "0.0 across 10 annotated questions" from "not measurable
here". `gold_evidence_ids` must come from the question/evidence pool; it is
never guessed from the run.

`experiments/evaluation/accuracy.py` remains the separate judged-correctness
track and deliberately computes no automatic score of its own: `QAJudgment`
records the generated answer, the reference answer, a binary `correct`
outcome, and **who or what decided** (a human annotator id, or a named rule
for closed-form questions). `rag_metrics.py` owns automatic scoring so the two
cannot drift into two competing judges of the same thing.

### 11.1 Human-judged hallucination protocol (out of the critical path)

Retained, implemented and tested, but **not required before a main result** —
see `docs/current_objectives.md`, "Removed from the primary pipeline".
Recorded here because it defines terms the write-up uses.

An answer is hallucinated if it contains **at least one claim unsupported by,
or contradicted by, the evidence supplied to the system that produced it** — a
faithfulness judgement against the frozen candidate set, not a clinical
correctness judgement. Schema and validation:
`experiments/evaluation/annotation.py`. Diagnostic subtypes, recorded but not
separately weighted: faithfulness, factuality, temporal, misinterpretation,
ambiguity, other.

Generated answers are exported as a blinded packet (`build_blinded_packet`)
with system identity replaced by "System A" / "System B" and order randomised;
the unblinding key is kept separate from what the annotator sees. Completed
annotations are re-validated on import (`read_annotations`) and recombined
with system identity only after review (`unblind_annotations`). Arm identity
must be hidden and order randomised, with the mapping stored separately —
without it the result is not defensible.

## 12. Statistical procedures

`experiments/evaluation/stats.py`, standard library only.

* **Exact McNemar** (`mcnemar`) on the paired discordant answers — a two-sided
  exact binomial test, no normal approximation, no SciPy dependency.
* **Paired bootstrap 95% CI** (`paired_bootstrap_ci`) on the absolute
  difference, resampling **questions** as units so each question's paired
  outcome stays together.
* **Holm correction** (`holm`) across the pre-declared comparisons.
* `har()` reports every rate two ways — conditional on answering, and over
  every item with abstentions counted — because a bare rate over zero answers
  is not a rate. `coverage()` and `compare_systems()` refuse a headline
  difference when either system answered nothing.
* `outcome_crosstab()` splits paired outcomes into baseline-only /
  proposed-only / both / neither, keeping question ids so representative
  examples — including cases where the proposed system does worse — can be
  pulled directly. `error_analysis()` breaks each cell down by diagnostic
  subtype. Both only count what annotation already recorded; neither infers a
  pattern.

No separate test was written for QA accuracy: it is the same paired-binary
shape, so the same functions apply (verified in
`tests/unit/test_evaluation.py::QAAccuracyStatsTests`).

Dev/validation/test assignment is a pure function of `question_id` and a
recorded seed, with the **question** — not the pair — as the unit (D-18), so
adding questions cannot quietly change the test partition's composition.

## 13. Evaluation-question provenance protocol

**Established 2026-09-17, before any question was sourced.** It keeps a single
property true: **every reference answer is traceable to a real record that
this thesis did not author.**

```
inspected external source record → factual proposition → candidate question
→ verbatim reference answer → citation + stable locator + date
→ automatic validation and deduplication
→ human review        ← authority to approve lives here
→ approved candidate → corpus-support check → final evaluation question
```

**Forbidden.** A language model inventing a question, an answer or a citation
that is then treated as ground truth. Search snippets as reference evidence.
An LLM cited as a source. A paraphrase presented as a quotation.

**What automation does:** parsing, normalising, keyword classification,
deduplication, completeness checks, formatting a review file. It does not
decide what is true. No LLM wrote any question or answer in the current pool.

| # | Rule |
|---|---|
| Source of the question | The source record itself — a Cochrane review states its own review question, an NIH page its own section question. |
| Source of the answer | One **verbatim** sentence of the source's own conclusion or section text. Methodological preamble and restated headings are skipped: they cite correctly but answer nothing. |
| Independent verification | **Not done yet.** Every record carries `verification_required: true` and `status: candidate`; PubMed E-utilities and doi.org were blocked in the build environment, so identifiers were transcribed, not resolved. This is the first review task. |
| Date recorded | The source's publication date where it has one. MedQuAD carries none, so the retrieval date is recorded and flagged `reference_date_is_retrieval_date: true`. |
| AD relevance | Cochrane: the record must name Alzheimer's *somewhere* and the question or objectives must be about Alzheimer's, dementia or cognition. NIH: UMLS CUI **C0002395**, assigned by NLM indexers. Never a bare keyword match on the question. |
| Determinacy | Cochrane items carry an explicit verdict label. `NOT ENOUGH INFORMATION` is still a determinate finding *about the evidence base*, flagged ambiguous for the reviewer. Final determinacy is the reviewer's call. |
| Corpus support | `corpus_support_expected` is an expectation, not a check. The real check runs against the completed corpus. |
| Temporal flag | `temporal_candidate` is set when the Cochrane citation carries `.pub2` or higher. **It does not assert the verdict changed** — diagnostic only. |
| Ambiguity flag | Recorded, not acted on. Ambiguity is a *cause* of hallucination, so these are the cases most likely to expose the behaviour under study; discarding them would remove the signal. |
| Duplicates | Content-derived ids catch exact repeats; Jaccard over normalised tokens at 0.85 catches near-duplicates. First occurrence kept, later ones marked `rejected` with a reason so counts reconcile. Deliberately crude and deterministic — an embedding model would make the set depend on an unreproducible judgement. |
| Provenance stored | `reference_source`, `reference_source_type`, `reference_locator`, `reference_date`, the source dataset and citation, and how the answer was extracted. Locators are concrete, never a bare title. |
| Approval | **A human reviewer, and only a human.** The builder emits `candidate` or `rejected`; no code path produces `approved` or `final`. |

### 13.1 Source categories

| Category | Status | Acceptable for | Not acceptable for |
|---|---|---|---|
| **A. Peer-reviewed evidence synthesis** (Cochrane) | **In use** | treatment, diagnosis, prevention, prognosis | facts the review does not state |
| **D. Government / public health** (NIH via MedQuAD) | **In use** | disease characteristics, genetics, symptoms, epidemiology | fine-grained treatment efficacy |
| **C. Guidelines / consensus documents** | Not yet used | diagnostic criteria, recommendations, temporal facts | — |
| **B. Research organisations** | Not yet used | epidemiology, general characteristics | efficacy claims |
| **E. Existing ADRD QA datasets** | Not used | — | licensing and provenance unverified |
| **F. The thesis corpus itself** | **Excluded by design** | — | **anything** — a question written from a passage later shown to the model is circular |

Where sources conflict, a dated peer-reviewed synthesis beats an undated
public-health page.

### 13.2 Review

Materials: `experiments/questions/review.csv` and `docs/question_review.md`.
Outcomes: `ACCEPT` · `REVISE` · `REJECT` · `HOLD`. No numeric score.

**The review file is deliberately neutral.** It carries the question, the
reference answer and everything needed to trace that answer to a source — and
none of the classifications this pipeline assigned. Internal flags
(`temporal_candidate`, `ambiguity_candidate`, source verdict label, automatic
validation outcomes) stay in `candidates.jsonl`, because a column saying a
candidate looked weak invites confirmation rather than assessment — and the
automated judgement is precisely what needs independent checking.
`export_review.py` enforces this: it emits exactly `REVIEW_COLUMNS` and raises
rather than write a file carrying a withheld field.

The reviewer fills `review_decision`, `reviewer_note`, `reviewer_id` and
`review_date`, and edits nothing else; a reference-answer change is described
in the note and applied afterwards, so the original wording and its provenance
stay recoverable.

## 14. External evaluation data — input contract

Used by the superseded temporal test-pair design (`experiments/test_pairs/`,
out of the critical path per `docs/current_objectives.md`). The contract is
recorded here because `validate_external.py` and `build_pairs.py` cite it at
runtime and refuse to run without it.

**Nothing in this repository generates that material**, and
`build_pairs.py --pool primary_external` fails with a message naming the
dependency rather than falling back to thesis-written questions.

**Dependency:** MedChangeQA (Vladika, Dhaini & Matthes, *Facts Fade Fast*,
Findings of EMNLP 2025), <https://github.com/jvladika/MedChange>. **No
`LICENSE` file is published**, and the underlying text is Cochrane Library
abstract content (Wiley copyright), so **do not commit the dataset** — fetch
it at build time, record the commit SHA and the file SHA-256, and cite the
paper.

Released files (inspected directly, 2026-09-16): `MedChangeQA.csv` (512 rows:
`Question`, `Newest Label`, `Outdated Label`), `MedRevQA.csv` (16,501 rows,
including `conclusions`, `DOI_Date`, `PMID`), `AllStudyGroups.csv` (4,379
rows: `Group_ID`, `Study_ID`, `Label`).

**The join (D-34), deterministic and verified:** `MedChangeQA.csv` alone has
no PMIDs, dates or evidence text. `AllStudyGroups.csv` supplies the linkage —
`Group_ID` is sparse and must be forward-filled (1,535 groups of size 2–9);
`Study_ID` is a **0-based row index into `MedRevQA.csv`**, verified by label
agreement on 4,379/4,379 rows. Groups holding more than one distinct `Label`
number exactly **512** and align 1:1 in file order with `MedChangeQA.csv`
(`Newest Label` agreement 512/512). Per side: evidence text = `conclusions`,
id = `PMID`, date = year parsed from `DOI_Date`. **No lexical matching, no
embeddings, no semantic retrieval.** Do not try to recover the sides by
matching question text: it was tried and only 6 of 512 recover both labels.

### 14.1 JSONL contract

One JSON object per line. The loader validates these and supplies **no
defaults**, because a silently defaulted reference answer is
indistinguishable from a real one.

**Required** — without them the record does not identify an evaluation item:
`question_id` (the partition unit, so all pairs for one question stay
together), `question_text`, `reference_answer`, `older_document_id`,
`newer_document_id`.

**Optional** — absence is counted as a visible exclusion, never filled in:
`older_text`, `newer_text` (absent ⇒ `missing_evidence_text`),
`older_publication_date`, `newer_publication_date` (absent ⇒
`missing_publication_date`), `question_date`, `change_point_date`,
`older_evidence_id`, `newer_evidence_id`, `claim_class`, `pair_category`,
`contradiction_status`, and per-side `*_source_tier`, `*_persistent_id`,
`*_length_tokens`.

`evidence_id` is minted as `EXT:<dataset>:<question_id>:<side>` when the
dataset supplies none.

A missing **required** field makes the file unusable: the validator exits
non-zero and the builder refuses it. Neither fills the gap.

### 14.2 Four different dates, kept apart

| Field | Meaning |
|---|---|
| `question_date` (t_q) | The information state the admission decision is evaluated against |
| `*_publication_date` | When each passage appeared |
| `change_point_date` | When the clinical verdict moved |
| `label_model_cutoffs` (config) | Pre-training cutoffs of the label-generating models |

`question_date` comes from the dataset when it supplies one, otherwise from a
single configured `evaluation_as_of_date`. It is **never** derived from a
passage in the pair, and the schema rejects such a value: setting t_q to the
newer passage's publication date gives that passage `R = 1` by construction
and inflates the recency contrast the proposed method is measured on (D-21). A
`change_point_date` the dataset does not supply is left absent, not
synthesised from a publication date.

### 14.3 Acquisition

```bash
BASE=https://raw.githubusercontent.com/jvladika/MedChange/main/Datasets
for f in MedChangeQA.csv MedRevQA.csv AllStudyGroups.csv; do
  curl -fsS -o "experiments/test_pairs/data/external/$f" "$BASE/$f"
done

# Convert to the §14.1 contract via the §14 join, then check it BEFORE building:
python -m experiments.test_pairs.scripts.validate_external \
    --input experiments/test_pairs/data/external/pairs_input.jsonl \
    --dataset "MedChangeQA @ <commit-sha>" \
    --output experiments/outputs/stage2_pilot/acquisition.json

python -m experiments.test_pairs.scripts.build_pairs \
    --pool primary_external \
    --input experiments/test_pairs/data/external/pairs_input.jsonl \
    --output-dir experiments/outputs/stage2_pilot
```

`validate_external` records the raw file's SHA-256 and the dataset name, so
the evaluation set can be traced back to the exact bytes it came from.

## 15. Interface requirements the experiment must satisfy

1. **Candidate sets are frozen once and replayed byte-identically** to every
   arm. No arm retrieves for itself during the comparison.
2. **N is identical for every item.** `ρ` is a within-set rank, so `θ` is only
   comparable across items at fixed candidate-set size.
3. **Candidate sets contain only dated passages.** Applied identically to
   every arm at construction (`experiments/retrieval/corpus.py::dated_only`),
   so the undated branch of the recency score never fires and `undated_score`
   is not a tunable. Every run reports its count of `UNDATED` passages; in a
   valid run that count is zero.
4. **Context order is candidate-list order in every arm.** Rank decides
   *which* passages survive the budget; it does not reorder what the generator
   sees.
5. **The frozen manifest's hash is asserted before the run**, so the
   evaluation set cannot change after outcomes are seen.
6. **λ, θ and H are fitted on the validation split and frozen before any test
   run.** `AdmissionConfig.validate()` raises rather than supply a default.
7. **The provenance firewall.** `assert_firewall` refuses a manifest where a
   question's reference evidence also appears among its candidates, which
   would make the comparison circular. `FrozenItem.provenance_leak()` treats
   any such overlap as a **defect**, not a goal.
8. **Corpus version identification.** Every `FrozenItem` carries a
   `corpus_snapshot` id: a manifest frozen against one corpus build is not
   comparable to one frozen against another.

## 16. Reporting requirements

* Every rate is reported both conditional on answering and with abstentions
  counted, alongside answer coverage. **Never report a bare rate.**
* The report must record which RAG² filter the baseline actually used
  (`baseline_filter`, `baseline_is_trained_rag2`): "proposed improves on
  baseline" against an all-HELPFUL stand-in is a different claim from the same
  verdict against the paper's classifier, and a reader must not have to infer
  which.
* Context metrics that were not measurable are reported as such, with
  `context_scored_n`, never as 0.000.
* Statistical comparison: exact McNemar on paired discordant answers, paired
  bootstrap CI resampling questions, Holm correction across the pre-declared
  comparisons.
* The corpus date distribution is reported alongside results (corpus recency
  skew is a condition of the experiment, not a finding).
* The generator's knowledge cutoff relative to the reference time is recorded
  and treated as a documented condition: the generator may answer correctly
  from parametric memory without using the evidence, which is not
  hallucination but does compress the difference between arms.

## 17. How to run it

```bash
# Full test suite (no network, no model required)
python -m unittest discover -s tests -t .

# Fixture end-to-end run: every arm, both comparisons, standard metrics
python -m experiments.runners.run_end_to_end --output-dir experiments/outputs/smoke

# A real run, once the §18 dependencies are resolved
python -m experiments.runners.run_end_to_end \
    --questions <frozen manifest> --index <built index> \
    --real-model --model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --model-revision <commit-sha> --quantization nf4 \
    --rag2-checkpoint <trained Flan-T5 checkpoint> \
    --proposed-lambda <fitted λ> \
    --output-dir experiments/outputs/<run id>
```

Reproducibility rests on: a pinned model revision (§9.2), greedy decoding with
no seed because nothing samples (§9.1), an order-sensitive
`candidate_set_hash` asserted before the run (§15.5), a recorded
`corpus_snapshot` (§15.8), a `RunConfig` hash covering model, revision,
quantization, prompt, budget, λ and the baseline filter identity, and an
output directory that `run_experiment` refuses to overwrite.

## 18. Unresolved dependencies

Each is a dependency with a known resolution, not a design ambiguity. Current
state of each: `docs/status_and_decisions.md`.

1. **λ, θ, H are unfitted**, by design, until a validation split exists.
2. **The baseline filter checkpoint does not exist** (§10). Until it does, the
   baseline arm runs only against `MockRAG2Filter`, which is engineering
   validation and never a result.
3. **No approved evaluation question set exists**: the 123-candidate pool is
   under human review (§13.2).
4. **No model has been downloaded or run** (§9.3 step 3), so no
   tokens/second figure exists anywhere in this repository.
