# Methodology

What is compared, how, and under what conditions. For the research question and objectives see
the root `README.md`; for the exact protocol, gates and statistics see `experiment_plan.md`; for
data provenance see `data.md`; for metrics see `evaluation.md`; for how to run things see
`reproducibility.md`.

## 1. The comparison in one line

Every arm receives **the same question and the same frozen candidate pool**, and answers with **the
same generator, prompt template and decoding**. Arms differ only in **which passages they admit**
from that pool (at most five, in a fixed order).

Stage 2 adds a layer that never changes what the generator reads: it takes the first eight candidates
of the same pool in cross-encoder order, judges each paper separately, and combines the judgements with
B1's verdict (§4). Every stage-2 arm is therefore a function of the same frozen pool and the same B1
answer.

## 2. Two evaluation settings

| | Primary: MedChange as-of benchmark | Secondary: Alzheimer's disease case study |
|---|---|---|
| Questions | 754 usable Cochrane questions (504 whose verdict changed between review versions, 250 unchanged controls); `experiments/medchange/` | 113 usable questions reviewed earlier by the researcher (99 with a verdict label; not extended, see `experiment_plan.md` §11); `experiments/shared/questions/` |
| Question date t_q | the newest review's publication date | the cited review's date (**planned**; the earlier pilot used the run date) |
| Evidence | PubMed abstracts first public strictly before t_q, fetched per question | local Alzheimer's corpus restricted to passages before t_q (**planned**) |
| Retrieval | PubMed best match → MedCPT dense rank → MedCPT cross-encoder rerank | MedCPT dense retrieval over the 4.4M-chunk local index → rerank |
| Status | benchmark, dev pools, six stage-1 arms and the dev run done; stage-2 code built, pilot passed, dev run pending; confirmatory pools not built | corpus, question pool and index built; as-of case-study run **not implemented** |

The primary setting exists because the Alzheimer's pool alone cannot test a temporal claim (only 5 of
113 questions are known verdict changes) or reach useful power; see `experiment_plan.md` §13.

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
half-life. The formula and the decay live in **one place**, `src/proposed/` (`AdmissionScorer`,
`TemporalPolicy`), which the MedChange arms call; `λ = 0` recovers pure relevance. In the MedChange
experiment λ = 0.5 and H = 1,095 days are **fixed, not fitted**, and passages are admitted by top-5
score with no θ threshold. The threshold-and-budget form (admit if A ≥ θ, then cap at the budget)
is implemented in `src/proposed/admission.py` and exercised by the framework's tests and fixture
demo; the validation-split fitting of λ, θ and H belonged to the superseded Alzheimer's pilot
(`_archive/alzheimers_pilot_v1/`) and is not part of the current experiments.

**What the RAG²-inspired arms are and are not.** RAG² (Sohn et al., NAACL 2025) trains a Flan-T5
filter on perplexity-derived labels, keeps passages labelled [HELPFUL], and caps by reranker rank.
Its checkpoint is not distributed (the authors' README says so) and only a 5-example sample of its
labels is released. A local retraining attempt (500 labels from a 4-bit Llama-3-8B on CPU) learned only
the class prior, and control experiments showed the labels carry almost no passage-specific signal at
that scale (archived: `_archive/rag2_filter_reproduction/`, `log.md` Phases 13–18). B2 and P therefore use
the **unmodified Flan-T5-large** asked RAG²'s prompt plus "Answer yes or no.", scored as P(yes) and
used as a ranking, not a threshold. They are untrained, text-only, helpfulness-ranked stand-ins and
must be described that way. `src/baseline/` still contains the faithful RAG² admission slot
(`FlanT5RAG2Filter`, which needs an externally trained checkpoint); it is exercised only by the
fixture demo and tests.

**Relation to published work.** B3 corresponds to adding a temporal score to a retriever's ranking, as in
TempRALM (Gade & Jetcheva). The proposed arm's only difference from B3 is the relevance signal; the
2 × 2 of helpfulness × recency is what the comparison can legitimately say about.

**Stage 2 arms and the evidence-synthesis layer.** Stage 2 is pre-specified in `experiment_plan.md` (§1,
§4–§10) and its code is built (`stance.py`, `synthesis.py`); no real stance output exists yet.

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

The original three-arm framework (`src/`, `evaluation/runner.py`) enforces parity with assertions
that run before the first item (candidate-set hash, budget, prompt, generator identity); its tests
are `test_runner_parity.py` and the integration suite.

## 6. Generator

Meta-Llama-3-8B-Instruct, Q4_K_M GGUF (bartowski), run with llama.cpp on CPU on a laptop with 15.2 GB
RAM and a 4 GB GPU that cannot hold the model. Greedy decoding, no sampling. A quantised model on CPU
is a disclosed deviation from RAG²'s own full-precision GPU generator. Qwen2.5-7B-Instruct is the
planned second generator for robustness. The GGUF file's SHA-256 and the decoding settings are recorded
automatically beside the answers (`reproducibility.md` §8). The stage-2 stance step uses the same GGUF
file with a 1,536-token context, one output token and the top-20 log-probabilities of that token
(llama-cpp-python needs `logits_all`, ≈ 0.5 GB extra memory; a hard-label mode skips it). One declared
fallback exists: the same pilot with Flan-T5-large as the stance model.

## 7. Parameters

**Stage 1:** nothing is fitted. λ, H and the budget are fixed (`arms.py`) and recorded in every answer. A
grid, if ever run, is exploratory and on dev data only. (The superseded pilot fitted λ, θ and H on a
validation split by maximising currency, which made its primary metric circular; see `experiment_plan.md`
§13.) **Stage 2:** the logistic layer's coefficients are fitted on the dev split (226 items) and frozen
before any confirmatory stance output exists; its penalty (5.0), the top-k (8), the snippet length, the
half-life (1,095 days) and the study-type weights (3 / 2 / 1) are fixed in advance and not tuned. The only
data-dependent choice is which of H0–H3 is selected (dev
cross-validation, 1.0 pp margin).

## 8. Ablation

`λ = 0` (relevance only) is the "component removed" condition; in the MedChange design it is B2 (for
the helpfulness signal) and B1 (for the cross-encoder signal), and C1 tests whether any gain depends
on the dates being real. In stage 2: S0 against S1/S2/S3 and H0 against H1/H2/H3 isolate the recency and
study-type weights; H1C and H3C test whether real dates matter; B1R against H0 isolates what the stance
features add beyond the same fitting; and an irrelevant-paper control in the pilot checks that the stance
step reads the paper rather than the question.

## 9. Deviation register

| Deviation | From | Reason | Effect on comparability |
|---|---|---|---|
| Untrained zero-shot Flan-T5 helpfulness instead of RAG²'s trained filter | RAG² | checkpoint unavailable; local retraining failed | B2/P are not a RAG² reproduction |
| No rationale-as-query | RAG² | one shared pool per question | arms isolate admission |
| Fixed-budget top-5, no θ | original Temporal Filter | removes a tuning degree of freedom | none among the arms |
| 4-bit GGUF generator on CPU | RAG²'s generator | hardware | applies to all arms alike |
| PubMed abstracts, as-of | local full-text corpus | per-question as-of retrieval across medicine | the local corpus is used only for the case study |
| Model-generated gold labels | human-verified labels | the MedChange release | reproducibility of the labels measured by an independent model (`label_audit.py`); clinician validation unavailable (stated limitation) |
| Stance judged paper by paper by the same 4-bit 8B model, from title + RESULTS + CONCLUSIONS | holistic reading of five full abstracts | removes order sensitivity; conclusion-focused inputs are much shorter than full abstracts (measured in the P0 diagnostics) | stage 2 only; quality checked by machine checks, a negative control and its predictive value on dev (no human validation) |
| Logistic layer fitted on dev and frozen | fixed-rule admission (stage 1) | combines stance with the RAG answer | learns the dev class mix; B1R receives the same fitting |
| Pooled all-items primary outcome | changed-items-only primary (stage 1) | power: 528 instead of 353 items | key secondary reports changed items alone |
