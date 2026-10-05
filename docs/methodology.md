# Methodology

What is compared, how, and under what conditions. For the research question and objectives see
the root `README.md`; for the exact protocol, gates and statistics see `experimentation.md`; for
data provenance see `data.md`; for metrics, statistics and results see `experimentation.md` §6–§7 and §12–§14; for how to run things see
`reproducibility.md`.

## 1. The comparison in one line

Every arm receives **the same question and the same frozen candidate pool**, and answers with **the
same generator, prompt template and decoding**. Arms differ only in **which passages they admit**
from that pool (at most five, in a fixed order).

Stage 2 adds a layer that never changes what the generator reads: it takes the first eight candidates
of the same pool in cross-encoder order, judges each paper separately, and combines the judgements with
B1's verdict (§4). Every stage-2 arm is therefore a function of the same frozen pool and the same B1
answer.

The realigned study (`experimentation.md`) changes one thing at a time. The adapted RAG² baseline (R2)
changes retrieval only: a rationale as the dense query, balancing across evidence types and a zero-shot
LLM filter, then the same answer prompt as B0 and B1. R2C, R2V (proposed) and R2V-ND admit exactly R2's
evidence for every question and change only how it is read: with explicit evidence criteria, design and
date labels, and (R2V, R2V-ND) R2's answer as a draft to check.

## 2. Two evaluation settings

| | Primary: MedChange as-of benchmark | Secondary: Alzheimer's/dementia test set (split `ad`) |
|---|---|---|
| Questions | 754 usable Cochrane questions (504 whose verdict changed between review versions, 250 unchanged controls); dev 226, held-out ("confirm") 528; `experiments/medchange/` | 212 MedRevQA questions that name dementia, Alzheimer's disease, mild cognitive impairment or cognitive decline, from 163 reviews that are in neither dev nor confirm; **all 212 are unchanged-verdict questions** (206 have a single review version); `experiments/medchange/ad_benchmark.py` |
| Question date t_q | the newest review's publication date | the same |
| Evidence | PubMed abstracts first public strictly before t_q, fetched per question | the same |
| Retrieval | PubMed best match → MedCPT dense rank → MedCPT cross-encoder rerank | the same |

The Alzheimer's-specific design that preceded this one (a local Alzheimer's corpus and a 113-question reviewed
pool) could not test a temporal claim or reach useful power: only 5 of its 113 questions are known verdict changes
and 71% of the corpus is from 2020 or later while the questions cite mostly older reviews (`log.md` Phase 21). It
is archived (`_archive/alzheimers_framework/`) and not used. The realigned study replaces it with the `ad` set,
built by the same rules as the main benchmark, run once after the freeze (`experimentation.md` §11). Because every
`ad` question is an unchanged-verdict question, changed-question and update-window results cannot be computed for
it; the secondary reading is on its overall verdict accuracy.

## 3. Shared upstream (identical by construction)

Per question, once, before any arm runs:

```
question → candidate generation (as-of) → MedCPT dense rank → MedCPT cross-encoder rerank
         → frozen pool of 20 (date bounds, order-sensitive hash)
```

MedCPT is used at both stages (dual encoder for similarity, cross-encoder for reranking). Neither
the arms nor the generator retrieves, reranks or re-scores anything afterwards. The as-of rule
(only records provably public before t_q; the source review and its versions excluded) is what makes
the comparison leakage-safe.

The realigned systems start one step earlier, from the same as-of candidate records (up to 200 per
question, cached locally): R2 ranks them by MedCPT similarity to the rationale, takes up to 8 of each
evidence type (systematic review or meta-analysis, trial, other design, from PubMed publication types),
re-ranks that union with the MedCPT cross-encoder against the question and keeps the top 8 for the
filter (`rag2.build_lists`). B0 and B1 keep the frozen pool of 20, so their committed answers are reused.

## 4. Arms

| Arm | Admission from the pool of 20 |
|---|---|
| B0 | nothing (no evidence block in the prompt) |
| B1 | top 5 by cross-encoder rank (standard RAG; equals the "No-Filter" control of the original design) |
| B2 | top 5 by zero-shot helpfulness P(yes) |
| B3 | top 5 by A(s) with ρ from the cross-encoder rank |
| P (proposed) | top 5 by A(s) with ρ from the helpfulness rank |
| C1 | as P but with publication dates shuffled within the pool |

**The proposed system ("Temporal Filter").** A(s) = (1 − λ)·ρ(s) + λ·T(s, q, t_q), where ρ is the
rank-normalised relevance signal within the pool (best = 1, worst = 0; ties broken by cross-encoder
rank) and T = 2^(−age_days / H) is plain age decay clamped at zero. λ is the temporal weight and H the
half-life. The formula and the decay live in **one place**, `src/temporal_filter/` (`AdmissionScorer`,
`TemporalPolicy`), which the MedChange arms call; `λ = 0` recovers pure relevance. In the MedChange
experiment λ = 0.5 and H = 1,095 days are **fixed, not fitted**, and passages are admitted by top-5
score with no θ threshold. The threshold-and-budget form (admit if A ≥ θ, then cap at the budget)
is archived with the Alzheimer's framework it served
(`_archive/alzheimers_framework/src/temporal_filter/admission.py`, still tested there); the validation-split
fitting of λ, θ and H belonged to the superseded Alzheimer's pilot (`_archive/alzheimers_pilot_v1/`) and is not
part of the current experiments. Since the realignment the Temporal Filter is a stage-1 result of record, not the
proposed system: the proposed system is R2V (below).

**What the RAG²-inspired arms are and are not.** RAG² (Sohn et al., NAACL 2025) trains a Flan-T5
filter on perplexity-derived labels, keeps passages labelled [HELPFUL], and caps by reranker rank.
Its checkpoint is not distributed (the authors' README says so) and only a 5-example sample of its
labels is released. A local retraining attempt (500 labels from a 4-bit Llama-3-8B on CPU) learned only
the class prior, and control experiments showed the labels carry almost no passage-specific signal at
that scale (archived: `_archive/rag2_filter_reproduction/`, `log.md` Phases 13–18). B2 and P therefore use
the **unmodified Flan-T5-large** asked RAG²'s prompt plus "Answer yes or no.", scored as P(yes) and
used as a ranking, not a threshold. They are untrained, text-only, helpfulness-ranked stand-ins and
must be described that way. The faithful RAG² admission slot (`FlanT5RAG2Filter`, which needs an externally
trained checkpoint) was part of the archived framework (`_archive/alzheimers_framework/src/baseline/`) and has
never been run on real data.

**Relation to published work.** B3 corresponds to adding a temporal score to a retriever's ranking, as in
TempRALM (Gade & Jetcheva). The proposed arm's only difference from B3 is the relevance signal; the
2 × 2 of helpfulness × recency is what the comparison can legitimately say about.

**Stage 2 arms and the evidence-synthesis layer.** Stage 2 was pre-specified in the stage-2 protocol (in
the repository history, commit 92e3aaf) and run on dev (`stance.py`, `synthesis.py`); gate 2 failed, so it was
not run on the held-out split (`experimentation.md` §13).

| Arm | What it is |
|---|---|
| B0, B1 | as above (no evidence; cross-encoder top-5) |
| B1R | B1's verdict through the same logistic fitting on dev as the hybrids; separates "stance helps" from "any fitting helps" |
| S0 | stance features only, papers weighted equally |
| S1 / S2 / S3 | S0 with recency weights / study-type weights / both |
| H0–H3 | hybrid: B1's verdict (one-hot) plus the stance features of S0–S3 |
| H1C, H3C | H1 and H3 with publication dates shuffled within the pool |

*Stance of one paper.* The model is shown the question read as a claim, then one paper's title and its
RESULTS and CONCLUSIONS sections (at most 200 words; an abstract without labelled sections gives its last
three sentences) and answers with one letter: supports / contradicts / neither. The three probabilities come
from the first output token's distribution over the three letters. Two wordings with different letter
orders exist; both are run and their probabilities averaged (no human chooses). This replaces reading five abstracts at once (which is
sensitive to order: the same five papers in another order change 14% of verdicts) by eight independent,
order-free judgements.

*Four features* from the eight stance probabilities and paper weights w_i (W = Σ w_i): signed stance
Σ w_i (p_sup − p_con) / W; the share without a clear stance Σ w_i p_nei / W; conflict
2·min(Σ w_i p_sup, Σ w_i p_con) / W; informative mass ln(1 + Σ w_i (p_sup + p_con)). Weights: none; recency
2^(−age/H) with H = 1,095 days measured to t_q only; study type 3 for a systematic review or meta-analysis,
2 for a randomized or other controlled trial, 1 otherwise (PubMed publication types); or the product.

*The fitted layer.* A multinomial logistic regression on standardised features with a fixed L2 penalty (5.0),
fitted on the 226 dev items and frozen; no feature uses an item's kind, change type, previous-version date or
label. The selected hybrid is H0 unless a weighted variant's dev cross-validated accuracy is at least 1.0
pp higher.

**Realigned systems: adapted RAG² and evidence-criteria verification** (`rag2.py`, `rag2_run.py`).

| Arm | What it reads | How it answers |
|---|---|---|
| R2 (baseline) | the rationale-ranked, type-balanced, re-ranked top 8, filtered: abstracts with P(yes) ≥ 0.5 in cross-encoder order, at most 5 | the standard verdict prompt (`prompts.build_prompt`) |
| R2-RQ / R2-BR / R2-NF | as R2 with the question as the dense query / without balancing / without the filter (top 5) | as R2 |
| R2C | R2's admitted abstracts, labelled with design and year | one pass with the evidence criteria; three lines ending in FINAL VERDICT |
| R2V (proposed) | the same, plus R2's answer as a draft | checks the draft against the criteria; same three lines |
| R2V-ND | as R2V without years | criteria without the currency clause |

*Filter.* The generator is asked whether the abstract (title, results and conclusions, ≤ 200 words) contains
information that helps answer the question; P(yes) is the first token's probability renormalised over Yes and
No. An invalid judgement (Yes + No probability < 0.5) keeps the paper; if nothing passes, R2 answers without
evidence and the criteria arms keep R2's answer.

*Evidence criteria* (verbatim in `rag2.CRITERIA`): directness (only studies of the question's intervention,
population and outcome count; other interventions, populations and surrogate measures do not); design weight
(randomised trials and systematic reviews first); SUPPORTED when the direct studies show at least partial
benefit even with low certainty; REFUTED when they show no benefit, an effect similar to placebo or the
comparison, or harm; NOT ENOUGH INFORMATION only without direct studies; when direct studies disagree, the
weight of the direct randomised evidence, and (dated version) newer evidence takes precedence over older
evidence it may have superseded. The class definitions restate the benchmark's labelling rubric; R2C measures
what they achieve without verification. A verification output without a FINAL VERDICT line keeps the draft
verdict and is counted as invalid.

## 5. What is held constant, and how it is enforced

| Held constant | Enforced by |
|---|---|
| Question and gold labels | one benchmark file; arms receive the item, never its labels |
| Candidate pool and order | one frozen pool per item with an order-sensitive hash; arms select from it |
| Budget | `arms.BUDGET`, hashed into every answer record (`arms.settings_hash`) |
| Prompt template | one function (`prompts.build_prompt`); B0 differs only by omitting the evidence block; a unit test checks passages appear without dates |
| Generator and decoding | one llama.cpp call (`generate_answers.llama_generator`), greedy, 160 new tokens |
| Context order | the order `admit` returns (score, then cross-encoder rank, then PMID) |
| Stance prompt, wording and decoding (stage 2) | `stance.build_messages`; one output token, temperature 0; the model file's SHA-256, wording hashes and settings are recorded beside the stance file and a resume under a different configuration is refused |
| Stance inputs (stage 2) | the first 8 pool candidates by cross-encoder rank; `stance.study_snippet` (title, RESULTS, CONCLUSIONS, ≤ 200 words) |
| The fitted layer (stage 2) | one frozen model file written by `synthesis fit` on dev; `synthesis predict` refuses to run without it and records its hash with the confirmatory predictions |
| Evidence of R2, R2C, R2V, R2V-ND | `rag2_run.answer_one` admits for every criteria arm through R2's list and R2's filter judgements, so the four arms read the same abstracts in the same order |
| Realigned settings and prompts | `rag2.SETTINGS` and every prompt text are hashed into each output file's configuration; a resume under a different configuration is refused; the design record (`results/rag2_design.json`) must be on origin/main and equal to the current design before the held-out run starts |

The original three-arm framework (archived: `_archive/alzheimers_framework/evaluation/runner.py`) enforced
parity with assertions that ran before the first item (candidate-set hash, budget, prompt, generator identity);
its tests are `_archive/alzheimers_framework/tests/unit/test_runner_parity.py` and the integration suite beside
it. The current pipeline enforces the same properties through the table above.

## 6. Generator

Meta-Llama-3-8B-Instruct, Q4_K_M GGUF (bartowski), run with llama.cpp on CPU on a laptop with 15.2 GB
RAM and a 4 GB GPU that cannot hold the model. Greedy decoding, no sampling. A quantised model on CPU
is a disclosed deviation from RAG²'s own full-precision GPU generator. Qwen2.5-7B-Instruct, a different
model family, is used only as an independent judge (label audit, consistency check, optional directness of
admitted evidence), never as a generator. In the realigned study the same Llama file writes the rationales
(1,024-token context, 128 new tokens), judges the filter (1,536-token context, one token, top-20
log-probabilities) and writes the answers and verifications (6,144-token context, 160 new tokens). The GGUF file's SHA-256 and the decoding settings are recorded
automatically beside the answers (`reproducibility.md` §8). The stage-2 stance step uses the same GGUF
file with a 1,536-token context, one output token and the top-20 log-probabilities of that token
(llama-cpp-python needs `logits_all`, ≈ 0.5 GB extra memory; a hard-label mode skips it). One declared
fallback exists: the same pilot with Flan-T5-large as the stance model.

## 7. Parameters

**Stage 1:** nothing is fitted. λ, H and the budget are fixed (`arms.py`) and recorded in every answer. A
grid, if ever run, is exploratory and on dev data only. (The superseded pilot fitted λ, θ and H on a
validation split by maximising currency, which made its primary metric circular; see `log.md` Phase 19.) **Stage 2:** the logistic layer's coefficients are fitted on the dev split (226 items) and frozen
before any confirmatory stance output exists; its penalty (5.0), the top-k (8), the snippet length, the
half-life (1,095 days) and the study-type weights (3 / 2 / 1) are fixed in advance and not tuned. The only
data-dependent choice is which of H0–H3 is selected (dev
cross-validation, 1.0 pp margin). **Realigned study:** nothing is fitted. The quota (8 per evidence type),
the re-ranked list (8), the budget (5), the filter threshold (0.5), the rationale length (128 new tokens) and
the answer length (160) are fixed in `rag2.SETTINGS` before any output exists.

## 8. Ablation

`λ = 0` (relevance only) is the "component removed" condition; in the MedChange design it is B2 (for
the helpfulness signal) and B1 (for the cross-encoder signal), and C1 tests whether any gain depends
on the dates being real. In stage 2: S0 against S1/S2/S3 and H0 against H1/H2/H3 isolate the recency and
study-type weights; H1C and H3C test whether real dates matter; B1R against H0 isolates what the stance
features add beyond the same fitting; and an irrelevant-paper control in the pilot checks that the stance
step reads the paper rather than the question. In the realigned study: R2-RQ, R2-BR and R2-NF each remove one
RAG² component from the baseline; R2C removes the draft from the verification (criteria only); R2V-ND removes
the dates and the currency criterion.

## 9. Deviation register

| Deviation | From | Reason | Effect on comparability |
|---|---|---|---|
| Untrained zero-shot Flan-T5 helpfulness instead of RAG²'s trained filter | RAG² | checkpoint unavailable; local retraining failed | B2/P are not a RAG² reproduction |
| No rationale-as-query (stages 1 and 2) | RAG² | one shared pool per question | arms isolate admission; R2 restores it |
| Rationale ranks the question's as-of PubMed candidates (≤ 200), not a 564 GB index | RAG² | compute; as-of dating is available for PubMed only | R2 is an adapted, not a reproduced, RAG² |
| Balance across evidence types, not across four corpora | RAG² | one dated corpus | keeps RAG²'s purpose (no dominant source crowds out the rest) |
| Zero-shot filter by the local 8B generator instead of the trained Flan-T5 filter | RAG² | checkpoint unavailable; local retraining failed; RAG² reports a GPT-4o filter matched its trained one | a weaker judge than GPT-4o; reported as a substitute |
| Three-way verdict task instead of four-option exam questions | RAG² | the benchmark's task; the only dated medical benchmark available | results are not comparable with RAG²'s accuracies |
| Class definitions in the criteria arms restate the benchmark's labelling rubric | the benchmark's answering prompt | the dev errors show the two definitions differ | measured separately by R2C |
| Fixed-budget top-5, no θ | original Temporal Filter | removes a tuning degree of freedom | none among the arms |
| 4-bit GGUF generator on CPU | RAG²'s generator | hardware | applies to all arms alike |
| PubMed abstracts, as-of | local full-text Alzheimer's corpus | per-question as-of retrieval across medicine | the local corpus is archived and not used |
| Model-generated gold labels | human-verified labels | the MedChange release | reproducibility of the labels measured by an independent model (`label_audit.py`); clinician validation unavailable (stated limitation) |
| Stance judged paper by paper by the same 4-bit 8B model, from title + RESULTS + CONCLUSIONS | holistic reading of five full abstracts | removes order sensitivity; conclusion-focused inputs are much shorter than full abstracts (measured in the P0 diagnostics) | stage 2 only; quality checked by machine checks, a negative control and its predictive value on dev (no human validation) |
| Logistic layer fitted on dev and frozen | fixed-rule admission (stage 1) | combines stance with the RAG answer | learns the dev class mix; B1R receives the same fitting |
| Pooled all-items primary outcome | changed-items-only primary (stage 1) | power: 528 instead of 353 items | key secondary reports changed items alone |
