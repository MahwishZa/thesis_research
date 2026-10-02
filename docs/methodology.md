# Methodology

What is compared, how, and under what conditions. For the research question and objectives see
the root `README.md`; for the exact protocol, gates and statistics see `experiment_plan.md`; for
data provenance see `data.md`; for metrics see `evaluation.md`; for how to run things see
`reproducibility.md`.

## 1. The comparison in one line

Every arm receives **the same question and the same frozen candidate pool**, and answers with **the
same generator, prompt template and decoding**. Arms differ only in **which passages they admit**
from that pool (at most five, in a fixed order).

## 2. Two evaluation settings

| | Primary: MedChange as-of benchmark | Secondary: Alzheimer's disease case study |
|---|---|---|
| Questions | 754 usable Cochrane questions (504 whose verdict changed between review versions, 250 unchanged controls); `experiments/medchange/` | 113 usable human-reviewed questions (99 with a verdict label); `experiments/shared/questions/` |
| Question date t_q | the newest review's publication date | the cited review's date (**planned**; the earlier pilot used the run date) |
| Evidence | PubMed abstracts first public strictly before t_q, fetched per question | local Alzheimer's corpus restricted to passages before t_q (**planned**) |
| Retrieval | PubMed best match → MedCPT dense rank → MedCPT cross-encoder rerank | MedCPT dense retrieval over the 4.4M-chunk local index → rerank |
| Status | benchmark, dev pools and arms built; dev generation pending | corpus, question pool and index built; as-of case-study run **not implemented** |

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

## 5. What is held constant, and how it is enforced

| Held constant | Enforced by |
|---|---|
| Question and gold labels | one benchmark file; arms receive the item, never its labels |
| Candidate pool and order | one frozen pool per item with an order-sensitive hash; arms select from it |
| Budget | `arms.BUDGET`, hashed into every answer record (`arms.settings_hash`) |
| Prompt template | one function (`prompts.build_prompt`); B0 differs only by omitting the evidence block; a unit test checks passages appear without dates |
| Generator and decoding | one llama.cpp call (`generate_answers.llama_generator`), greedy, 160 new tokens |
| Context order | the order `admit` returns (score, then cross-encoder rank, then PMID) |

The original three-arm framework (`src/`, `evaluation/runner.py`) enforces parity with assertions
that run before the first item (candidate-set hash, budget, prompt, generator identity); its tests
are `test_runner_parity.py` and the integration suite.

## 6. Generator

Meta-Llama-3-8B-Instruct, Q4_K_M GGUF (bartowski), run with llama.cpp on CPU on a laptop with 15.2 GB
RAM and a 4 GB GPU that cannot hold the model. Greedy decoding, no sampling. A quantised model on CPU
is a disclosed deviation from RAG²'s own full-precision GPU generator. Qwen2.5-7B-Instruct is the
planned second generator for robustness. The GGUF file's SHA-256 and the decoding settings are recorded
automatically beside the answers (`reproducibility.md` §8).

## 7. Parameters

Nothing is fitted. λ, H and the budget are fixed (`arms.py`) and recorded in every answer. The
planned Alzheimer's case study reuses the same settings. A grid, if ever run, is exploratory and on dev
data only. (The superseded pilot fitted λ, θ and H on a validation split by maximising currency, which
made its primary metric circular; see `experiment_plan.md` §13.)

## 8. Ablation

`λ = 0` (relevance only) is the "component removed" condition; in the MedChange design it is B2 (for
the helpfulness signal) and B1 (for the cross-encoder signal), and C1 tests whether any gain depends
on the dates being real.

## 9. Deviation register

| Deviation | From | Reason | Effect on comparability |
|---|---|---|---|
| Untrained zero-shot Flan-T5 helpfulness instead of RAG²'s trained filter | RAG² | checkpoint unavailable; local retraining failed | B2/P are not a RAG² reproduction |
| No rationale-as-query | RAG² | one shared pool per question | arms isolate admission |
| Fixed-budget top-5, no θ | original Temporal Filter | removes a tuning degree of freedom | none among the arms |
| 4-bit GGUF generator on CPU | RAG²'s generator | hardware | applies to all arms alike |
| PubMed abstracts, as-of | local full-text corpus | per-question as-of retrieval across medicine | the local corpus is used only for the case study |
| Model-generated gold labels | human-verified labels | the MedChange release | ~100 labels to be human-checked |
