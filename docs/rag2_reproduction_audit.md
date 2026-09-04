# RAG² reproduction audit

**Question this answers:** does the code in `rag2/` actually implement the method
the original RAG² paper describes (Sohn et al., NAACL 2025,
[2025.naacl-long.635](https://aclanthology.org/2025.naacl-long.635/))?

**Verdict: MOSTLY VERIFIED** — see [§7](#7-verdict) for the reasoning and the
priority fix list.

| | |
| --- | --- |
| Audited commit | `746b611`, corrected by the six fixes in [§5](#5-defects-found-and-fixed) |
| Automated checks | 42, reproducible with `python -m rag2_audit.run` |
| Result | 34 PASS · 0 FAIL · 2 PARTIAL · 3 UNKNOWN · 2 APPROXIMATION · 1 MANUAL |
| Machine-readable | `audit/audit_results.json` |
| End-to-end trace | `audit/trace_synthetic.json` (`python -m rag2_audit.trace`) |
| Quantitative tests | `tests/test_audit_quantitative.py` (28), `tests/test_audit_module.py` (12) |
| Full suite | 222 passed, 1 skipped |

Classification used throughout: **PASS** matches the original work · **FAIL**
contradicts it · **PARTIAL** broadly matches but differs in an important detail ·
**UNKNOWN** cannot be verified from available information · **APPROXIMATION** the
original resource is unavailable and an alternative was used · **MANUAL** needs a
GPU run or a human to settle.

> **Self-audit disclosure.** The implementation and this audit were produced in
> the same engagement. That is a real bias risk, and the mitigation is that the
> checks are executable and adversarial rather than assertions of correctness:
> `tests/test_audit_module.py` contains six mutation tests that deliberately
> break the sign of ΔPPL, the Figure 2 truth table, the admission direction, the
> τ percentile, the leakage guard and the balance guarantee, and asserts that the
> corresponding check turns FAIL. A check that cannot fail proves nothing; those
> six can. An independent re-audit of §3 by a second reader is still advisable
> before the thesis relies on the baseline.

---

## 1. What was inspected

| Area | Location |
| --- | --- |
| Entry points | `scripts/01`–`06`, `scripts/smoke_test.py` |
| Configuration | `configs/*.yaml`, `rag2/config.py` |
| Rationale generation | `rag2/rationale.py`, `rag2/prompts.py` |
| Retrieval | `rag2/retrieval/{encoder,index,balanced}.py`, `rag2/corpora/` |
| Reranking | `rag2/retrieval/rerank.py` |
| Filtering | `rag2/filtering/{base,rag2_filter,passthrough}.py` |
| Filter training | `rag2/filter_training/{perplexity,labeling,build_labels,train}.py` |
| Answer generation | `rag2/generation.py` |
| Evaluation | `rag2/evaluation.py` |
| Cached evidence | `rag2/cache.py` |
| Orchestration | `rag2/pipeline.py` |
| Reference (unmodified) | `retriever/`, `classifier/` at release commit `86add43` |

Against: the paper (all 15 pages including Appendix A, Figures 1–4 and A1–A3),
and the authors' released repository including its full git history. The paper's
machine-checkable claims are transcribed with citations in `rag2_audit/paper.py`,
so the constants themselves can be checked against the source independently of
the code.

---

## 2. Component-by-component result

### Architecture and scope

| Component | Status | Note |
| --- | --- | --- |
| Four-stage pipeline present | PASS | STR-01: all 15 required callables |
| No SCAF / thesis machinery in the baseline | PASS | STR-02: AST-level scan of `rag2/`, executable tokens only |
| Authors' release unmodified | PASS | STR-03: byte-identical to `86add43` |
| Configuration hygiene | PASS | STR-04, **after the fix in §5.5**: no silent no-ops remain |

### Rationale and retrieval

| Component | Status | Note |
| --- | --- | --- |
| Rationale prompt | PASS | verbatim from §3.3; `tests/test_prompts.py` pins the wording |
| Rationale used as query, question excluded | PASS | §3.3; trace §03 confirms |
| MedCPT checkpoints, 768-dim, 512-token truncation | PASS | RET-01 |
| Exact inner-product MIPS | PASS | RET-02: top-k equals brute force on every probe |
| Balanced retrieval (equal per corpus) | PASS | RET-03 |
| Sharding preserves the corpus top-k | PASS | RET-04: score-merge ≡ single index; `concat` reproduces the release's imbalance |
| Retrieval sources | APPROXIMATION | RET-07 → [§4.6](#46-ret-07--the-corpus-is-unavailable) |
| Rerank ordering and tie-breaking | PASS | RET-05: stable, descending, 1-based ranks |
| Which query the reranker sees | PARTIAL | RET-06 → [§4.3](#43-ret-06--paper-and-released-code-disagree-on-the-rerank-query) |

### Filtering — the audit's focus

| Component | Status | Note |
| --- | --- | --- |
| Filter input template | PASS | FLT-01: reconstructs all 5 released records **byte-for-byte** |
| Filter sees the initial question, not the rationale | PASS | FLT-02, matching Figure 1 |
| No leakage of rank/score/source/date into the filter | PASS | FLT-03: 7 provenance fields checked absent |
| Two-way softmax ≡ the release's `argmax` | PASS | FLT-04, including the tie case → `[HELPFUL]` |
| Admission direction | PASS | FLT-05: keeps `[HELPFUL]`, rejects `[NOT_HELPFUL]` |
| Per-passage independence | PASS | FLT-06: batched decisions ≡ one-at-a-time |
| Filter model and checkpoint | APPROXIMATION | MOD-03 → [§4.7](#47-mod-03--the-filter-checkpoint-must-be-retrained) |

### Perplexity and labels

| Component | Status | Note |
| --- | --- | --- |
| PPL = exp(−mean log p), length-normalised | PASS | PPL-01, Equation 4 |
| **ΔPPL sign** | PASS | PPL-02: `PPL(x) − PPL(x,d)`; helpful ⇒ positive |
| Controlled comparison (same continuation, doc is the only difference) | PASS | PPL-03 |
| Which tokens are scored | PARTIAL | PPL-04 → [§4.1](#41-ppl-04--equation-4-contradicts-the-papers-own-prose) |
| Truncation preserves the scored length | PASS | PPL-05, **after the fix in §5.3** |
| Figure 2 decision tree | PASS | LBL-01: all 8 branches, both `[DISCARD]` leaves |
| τ = top 25%, inclusive `≥` | PASS | LBL-02: admits exactly 25/100 |
| Non-finite ΔPPL | PASS | LBL-03, **after the fix in §5.2** |
| Training-record schema | PASS | LBL-04: the release's four fields |
| Which rationale is scored | PASS | LBL-05, **after the fix in §5.4**: the cached stage-1 rationale is reused |
| Training hyperparameters | PASS | MOD-04: all 11 specified values |

### Generation and evaluation

| Component | Status | Note |
| --- | --- | --- |
| Backbone model identifier | PASS | MOD-01: the URL the paper gives explicitly |
| Meerkat checkpoint | UNKNOWN | MOD-02 → [§4.8](#48-mod-02--the-meerkat-checkpoint-is-inferred) |
| Greedy decoding at temperature 0 | PASS | GEN-01 |
| Answer-generation prompt | UNKNOWN | GEN-02 → [§4.4](#44-gen-02--eva-02--the-answer-prompt-and-extraction-rule-were-never-published) |
| Context = kept passages in rank order | PASS | GEN-03 |
| Generation length keys | PASS | GEN-04, **after the fix in §5.6**: passed explicitly from `llm.*` |
| Accuracy metric | PASS | EVA-01 |
| Answer extraction | UNKNOWN | EVA-02 → [§4.4](#44-gen-02--eva-02--the-answer-prompt-and-extraction-rule-were-never-published) |
| Table 1 split sizes | PASS | EVA-03 |
| Seeding | PASS | DET-01 (note: the authors ran **unseeded**) |
| Deterministic hashing | PASS | DET-02, **after the fix in §5.1** |
| Cache replay is fingerprint-verified | PASS | DET-03 |
| Runtime weight/tokenizer fidelity | MANUAL | MOD-05 → [§6](#6-what-this-audit-could-not-check) |

---

## 3. The filter, traced line by line

The audit brief asks for this component specifically. Each question, and what the
code actually does:

| Question | Answer |
| --- | --- |
| What model produces the perplexity values? | The **backbone LLM** (`rag2/llm/base.LLM.score`), not the filter. Correct: §3.2 measures the utility of a document *for the base model*. |
| Which model measures perplexity? | The same backbone used for rationale generation and QA, per §3.3. `vLLM` and `openai` backends raise rather than silently approximating — neither exposes teacher-forced log-probabilities. |
| What is the conditioning context? | `PPL(x)`: the answer prompt with **no** evidence block. `PPL(x,d)`: the identical prompt plus that one passage. PPL-03 asserts the document is the only difference. |
| What text is scored? | The **rationale** (`filter_training.ppl_target`). Prompt tokens are masked; only continuation tokens contribute to `L`. See [§4.1](#41-ppl-04--equation-4-contradicts-the-papers-own-prose) for the ambiguity. |
| How is perplexity calculated? | `exp(−(1/L)·Σ log p)` over continuation tokens, float32 accumulation, teacher-forced. Matches Equation 4 on hand-computable inputs (`−ln 2` per token ⇒ PPL 2.0). |
| How is the difference defined? | `ΔPPL = PPL(x) − PPL(x,d)`, Equation 3. |
| **Is the sign correct?** | **Yes.** A document that lowers perplexity gives ΔPPL > 0, which Figure 2 reads as "Lower Perplexity → Yes". Verified end-to-end in `test_a_helpful_document_yields_a_positive_delta_end_to_end`, and the mutation test confirms a reversed sign turns PPL-02 FAIL. |
| How are positive/negative examples created? | Figure 2's tree over three binary tests. Correctness flips settle the label alone; ΔPPL decides only when correctness is unchanged. |
| How are labels generated? | `rag2/filter_training/labeling.py`, reconstructed from Figure 2 because the release's `preprocess.py` is an **empty file**. All 8 branches pinned. |
| How is the threshold obtained? | 75th percentile of the ΔPPL distribution (linear interpolation, numpy-equivalent). |
| Is top-25% implemented correctly? | **Yes** — admits exactly 25/100 on a uniform grid, and 50/200, and 10/40. Boundary inclusive per Equation 3's `≥`. |
| Is filtering per-passage? | **Yes.** FLT-06 proves batched ≡ individual decisions. Matches the paper's Limitations. |
| Are passages ordered correctly before selection? | **Yes.** Descending cross-encoder logit, stable on ties, 1-based ranks; top-k slices that order. |
| Any information leakage? | **None found.** Only `Evidence.text` reaches the filter; rank, retrieval/rerank score, corpus, `doc_id`, `passage_id` and publication metadata are all absent from the prompt (FLT-03), and the whole-pipeline test confirms stripping or permuting dates changes no decision. |

---

## 4. Findings that need a decision

Ordered by how much they could move the baseline.

### 4.1 PPL-04 — Equation 4 contradicts the paper's own prose

**Paper.** Equation 4 sums `log P(x_i | x_<i)` over **x**, the query. But the
abstract says "perplexity-based labels **of rationales**"; §2.1 says "we measure
perplexity differences in the **rationales generated by the base LLM**"; and in
Figure 2 the "Lower Perplexity" node is tagged **Rationale**. Three statements
say rationale, the equation as written says query.

**Implementation.** Scores the rationale (`ppl_target: rationale`). The literal
reading is implemented and selectable (`ppl_target: query`).

**Why it matters.** These measure different things. Under the query reading,
ΔPPL asks whether the document makes the *question* more predictable; under the
rationale reading, whether it makes the model's *reasoning* more confident. Every
training label differs, so the reproduced filter would be trained on a different
signal. This is the single largest interpretive risk in the filter.

**Correction.** Not resolvable from the artefacts — it is a defect in the source,
not in this code. Ask the authors. Both readings are implemented; if the question
becomes load-bearing, build labels both ways and compare filter behaviour.

### 4.2 *(resolved)* LBL-05 — the scored rationale was regenerated

Fixed in [§5.4](#54-lbl-05--the-scored-rationale-was-regenerated-was-partial).
Labeling now reuses the cached stage-1 rationale, so the passage a rationale
retrieved is judged by that same rationale.

### 4.3 RET-06 — paper and released code disagree on the rerank query

**Paper.** Twice: "cross-encoding the **initial query** and each snippet"
(Figure 1 caption) and "encodes the **original query** along with each document"
(§3.4).

**Released code.** `retriever/main.py:124` passes `input_list` — the *rationale*
file — to `combine_query_evidence`.

**Implementation.** Defaults to `initial` (the paper);
`configs/ablation_release_retrieval.yaml` reproduces the code path.

**Why it matters.** One of the two does not describe the run that produced
Table 2, and they give different final top-k orderings from the same pool. Which
the authors used is unresolvable from the artefacts.

**Correction.** Ask the authors. Meanwhile run the ablation config alongside the
main config and report both; the difference is measurable, so it should be
measured rather than argued about.

### 4.4 GEN-02 / EVA-02 — the answer prompt and extraction rule were never published

**Paper.** Figure 1 shows only the structure (snippets + initial query). The
prompt wording appears nowhere. Neither does any answer-extraction rule; §3.3's
prompt asks for "your explanation and single option ... as the final answer", and
Figure 4's worked example ends "Therefore, the answer is (C) Intubation".

**Implementation.** Reuses the paper's chain-of-thought prompt with an evidence
block prepended (justified: §3.3 says the same LLM does both jobs, and Meerkat
was instruction-tuned on that prompt). Extraction is an ordered regex list,
last match wins, falling back to option-text matching; it parses the paper's own
Figure 4 example correctly. Unparsed generations count as incorrect, and
`num_unparsed` is reported alongside accuracy.

**Why it matters.** Prompt wording materially moves multiple-choice accuracy, and
extraction strictness moves it directly. These are the **first two things to vary**
if the reproduction misses Table 2 — before concluding anything about the method.

**Correction.** Unresolvable. Keep reporting `num_unparsed`; if it is
non-trivial, the extraction rule is doing real work and the number is fragile.

### 4.5 *(resolved)* STR-04 / GEN-04 — dead and ambiguously-scoped configuration

Fixed in [§5.5](#55-str-04--configuration-keys-that-were-silent-no-ops-was-partial)
and [§5.6](#56-gen-04--rationale-and-answer-length-governed-by-look-alike-keys-was-partial).

### 4.6 RET-07 — the corpus is unavailable

**Paper.** Appendix A.3 / Table A1: the Self-BioRAG corpus — PubMed (69.7M
passages), PMC (46.3M), CPG (607k), Textbooks (134k); 564.2 GB indexed.

**Implementation.** Four configured corpus slots with no data; the medical corpus
is being prepared separately.

**Why it matters.** **This is the dominant expected source of divergence from
Table 2.** Retrieval quality dominates a RAG pipeline's accuracy; a different
corpus can move results by several points on its own.

**Correction.** Not fixable here. Supply the corpus, re-run, record the delta.
Do not attribute a gap to the method before this is controlled.

### 4.7 MOD-03 — the filter checkpoint must be retrained

**Paper.** Flan-T5-large (770M) trained on the perplexity labels.

**Implementation.** Correct base model; `filter.checkpoint` empty, and the filter
**refuses to run** without one rather than silently falling back.

**Why it matters.** The reproduced filter is retrained from reconstructed labels.
The authors ran unseeded (`--seed` defaults to `None`), so an exact checkpoint
match is impossible **in principle**, not merely inconvenient.

**Correction.** Train via `scripts/03` + `scripts/04`; report filter accuracy and
per-class counts alongside downstream accuracy, so a filter that keeps everything
or nothing is visible immediately.

### 4.8 MOD-02 — the Meerkat checkpoint is inferred

**Paper.** Cites Kim et al. (2024) and describes the model (Mistral-7B init,
GPT-4-rationale instruction tuning, MedQA+MedMCQA fine-tuning) but gives **no
checkpoint path**.

**Implementation.** `dmis-lab/meerkat-7b-v1.0`, inferred from that description.

**Why it matters.** If the inference is wrong, every Meerkat row is produced by a
different model than the paper's and is not comparable.

**Correction.** Confirm against the Meerkat paper/release before the final run.

---

## 5. Defects found and fixed

Three defects were unambiguous bugs rather than interpretive choices, so they
were fixed. Each was confirmed by probe first, then fixed, then re-verified.

### 5.1 DET-02 — non-deterministic retrieval in the smoke test *(was FAIL)*

`scripts/smoke_test.py` seeded its stand-in encoder from
`hash(tuple(queries))`. Python randomises string hashing per process unless
`PYTHONHASHSEED` is set, so the retrieval draw differed on **every run** — four
fresh processes produced four different seeds. The test still passed, because its
assertions do not depend on which passages come back, which is exactly why this
would have gone unnoticed. Now seeded from `sha256`. Two consecutive runs are
byte-identical.

### 5.2 LBL-03 — non-finite ΔPPL silently labelled *(was PARTIAL)*

`percentile()` drops non-finite values when computing τ, but the comparison still
ran: `inf >= τ` is `True` (labelled via the "lower perplexity" branch) and
`nan >= τ` is `False` (labelled via the *other* branch, indistinguishable from a
genuine low-ΔPPL observation). These arise from a degenerate rationale — an empty
generation, or a scoring failure — and became training examples whose label was an
artefact of IEEE comparison semantics rather than evidence utility. Now excluded
before labeling, counted in `num_non_finite_excluded`, and listed in the stats so
degenerate generations stay visible.

### 5.3 PPL-05 — truncation could shorten the scored continuation *(was PARTIAL)*

`rag2/llm/hf.py:score` truncated from the left, and its docstring claimed the
continuation "always survives intact". It did not: once the cut exceeded the
prompt, `prompt_ids` became `[]` and the removed tokens came out of the
continuation, so `PPL(x)` and `PPL(x,d)` would normalise by different `L` and
Equation 3's difference would stop being a controlled comparison. **This was
latent** — it needs a rationale longer than the whole context window, and
`max_new_tokens` is 512 against a ≥2048 window, so no shipped config could reach
it. Fixed anyway because the false docstring invited a future change to
`max_input_tokens` to break Equation 3 silently: the cut is now clamped to the
prompt and an over-long continuation raises with an actionable message.

### 5.4 LBL-05 — the scored rationale was regenerated *(was PARTIAL)*

`build_labels.py` called the LLM again with the closed-book prompt and scored
*that* generation, discarding the `CandidateSet.rationale` stage 1 had written.
The paper has **one** rationale per question: generated by the base LLM, used as
the retrieval query (§3.3), and scored for perplexity (Figure 2). The two agreed
only while decoding was exactly deterministic and the same backbone and prompt
were configured for both stages — and the paper itself reports residual
nondeterminism at temperature 0 (A.3).

Now `filter_training.rationale_source: cached` (the default) reuses the stage-1
rationale and reads the closed-book correctness signal off that same string. It
falls back to regeneration, **recording why in the diagnostics**, when there is
no cached rationale or when the closed-book answer prompt has been overridden so
that the cached text is a completion of a different prompt. Side effect: one
fewer LLM call per question in the most expensive stage.

### 5.5 STR-04 — configuration keys that were silent no-ops *(was PARTIAL)*

Ten keys were declared but never read. Resolved by category rather than
wholesale:

- **Wired** — `generation.stop` (now truncates at the earliest stop sequence,
  applied post-generation so it behaves identically on every backend);
  `evaluation.open_ended_metrics` and `evaluation.bertscore_model` (now drive
  `scripts/06_evaluate.py --references`, the ROUGE-L/BERTScore path of
  appendix A.4.1).
- **Removed** — `retrieval.shard_size` (sharding comes from the corpus, not the
  config), `evaluation.metric` (accuracy is fixed by Table 2), `cache.enabled`
  (caching is controlled by whether `--candidates` is passed).
- **Declared provenance-only** — `experiment.notes`,
  `retrieval.article_encoder`, `corpus.chunk_size`, `corpus.chunk_overlap` are
  recorded in the manifest but legitimately unread, because the value they
  document is supplied out-of-band. They are now listed in
  `rag2.config.MANIFEST_ONLY_FIELDS`, and the audit's dead-key check consults
  that allowlist — so a *new* unread key still fails the check.

### 5.6 GEN-04 — rationale and answer length governed by look-alike keys *(was PARTIAL)*

`llm.max_new_tokens` governed the rationale and `generation.max_new_tokens` the
answer: same name, same default, different stages. Setting the latter to study
answer length would have left the rationale — and therefore the retrieval query
*and every perplexity label* — unchanged. `generate_rationales` now takes
`max_new_tokens` and `temperature` explicitly, wired from `llm.*` at every call
site, so which key governs which stage is visible in the code rather than
implied by construction order.

**Deliberately not changed:** the four findings in §4.1, §4.3, §4.4 and §4.8.
None is an error in this code — each is missing or self-contradictory information
in the source. "Fixing" them would mean inventing an answer the paper does not
give and presenting a guess as a reproduction. They stay documented and, where
the artefacts disagree, switchable.

---

## 6. What this audit could **not** check

Stated plainly rather than left as an implied PASS.

- **Runtime weight and tokenizer fidelity (MOD-05, MANUAL).** All model
  identifiers are verified statically. Actual weight loading, tokenizer
  vocabulary (including that `[HELPFUL]`/`[NOT_HELPFUL]` become *single* tokens),
  dtype and device placement are exercised only with `torch`/`transformers`
  installed — neither is present in this environment, and there is no GPU. A
  correct identifier can still resolve to an upstream revision that has changed.
  **Pin `llm.revision` and re-run the audit on the training machine.**
- **Anything requiring the corpus or a trained filter.** No end-to-end accuracy
  number exists yet, so §10 of the brief — do not judge correctness from accuracy
  alone — is satisfied trivially here: this audit is *entirely* methodological.
- **Whether the reconstructed labels resemble the authors'.** The only released
  artifact is 5 records, all `[NOT_HELPFUL]`. That is enough to pin the input
  template exactly (FLT-01) and nothing else — no label distribution, no τ
  behaviour. The `5%` in its filename remains unexplained against the paper's
  τ = 25%.
- **The paper's own internal consistency.** Where the paper contradicts itself
  (§4.1) or its released code (§4.3), this audit records both readings. It cannot
  determine which produced Table 2.

---

## 7. Verdict

### MOSTLY VERIFIED

**Reasoning.** Every component that defines the *method* is verified against the
paper and, where the paper is silent, against the authors' released code:

- The **filter** — the audit's focus and the thesis's dependency — is correct in
  every mechanical respect that can be checked without weights. The ΔPPL sign is
  right, the threshold admits exactly the top 25%, Figure 2's tree is transcribed
  branch-for-branch including both `[DISCARD]` leaves, decisions are per-passage
  and independent, the admission direction is not inverted, and the scoring rule
  reproduces the release's `argmax` including its tie behaviour. The filter input
  template reconstructs all five released records **byte-for-byte**, which is
  strong external evidence rather than self-assessment. No leakage was found.
- **Retrieval and reranking** match the release's checkpoints, geometry,
  truncation and exact-MIPS semantics, and balanced retrieval holds.
- **Training hyperparameters** match all eleven specified values.
- The three defects found were fixed; **no FAIL remains**.

It is not **VERIFIED** because three things are unresolvable from the available
artefacts, and honest reporting requires saying so rather than rounding up:
Equation 4 contradicts the paper's own prose about what is scored (§4.1); the
paper and the authors' code disagree about the rerank query (§4.3); and the
answer-generation prompt and extraction rule were never published (§4.4). None is
a defect in this code — each is missing or contradictory information in the
source — but each could change results, and no amount of care here can settle
them.

It is not merely **PARTIALLY VERIFIED** because nothing found contradicts the
paper. The gaps are absences and ambiguities, not errors.

**A caution the verdict does not capture:** this audit establishes methodological
fidelity, not empirical agreement. The baseline has never been run on real data —
there is no corpus and no trained filter yet. A later accuracy match would not
retroactively resolve §4.1–§4.4, and a mismatch should be investigated in the
order given in `docs/rag2_reproduction.md` §11 before anything is concluded about
the implementation.

### Highest-priority items before this becomes the thesis baseline

| # | Item | Why it blocks | Owner |
| --- | --- | --- | --- |
| 1 | Confirm the **Meerkat checkpoint** (§4.8) | a wrong model invalidates every Meerkat row | verify against Kim et al. (2024) |
| 2 | Decide the **ΔPPL scored text** (§4.1) | changes every training label | ask the authors; else run both and compare |
| 3 | Fix **rationale reuse** in labeling (§4.2) | silent divergence between the retrieval query and the scored rationale | one-line change, recommended |
| 4 | Run the audit **with models installed** (MOD-05) | single-token label ids and revision pinning are unverified | on the training machine |
| 5 | Settle the **rerank query** (§4.3) | changes final top-k ordering | run both configs, report both |
| 6 | Supply the **corpus** and train the **filter** (§4.6, §4.7) | no empirical result exists without them | blocked on the dataset |
| 7 | Wire or delete **dead config keys** (§4.5) | silent no-ops invite undetected deviation | housekeeping |

### Reproducing this audit

```bash
python -m rag2_audit.run                      # 42 checks, human-readable
python -m rag2_audit.run --json audit.json    # machine-readable
python -m rag2_audit.trace                    # end-to-end trace, no models needed
pytest tests/test_audit_quantitative.py       # 25 math and edge-case tests
pytest tests/test_audit_module.py             # 12 tests, incl. 6 mutation tests
```

`rag2_audit.run` exits non-zero on any FAIL, so a regression in the filter's
direction, the τ percentile or the paper-specified constants breaks CI.
