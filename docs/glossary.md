# Research Glossary

Terms as used consistently throughout this repository, code, and
documentation.

| Term | Meaning |
|---|---|
| **RAG²** | The published baseline method (Sohn et al., NAACL 2025): a trained filter decides which retrieved passages an LLM sees, based on passage text alone. |
| **Temporal Filter** | This thesis's proposed system: RAG²'s idea plus one additional signal — how old each passage is relative to the question. |
| **No-Filter control** | A third arm that admits every retrieved passage up to the context budget. The honest floor a filtering method must beat. |
| **Admission rule** | The function deciding which retrieved passages reach the generator. The one thing that differs between arms; everything else is held constant. |
| **`ρ(s)`** (relevance score) | Rank-normalised reranker score for passage `s`, in [0, 1]. The same signal RAG²'s own upstream produces. |
| **`T(s, q, t_q)`** (temporal score) | `2^(−age_days / H)` — 1.0 for a passage published on the question date, halving every `H` days. Plain age decay, nothing else. |
| **`λ`** (lambda) | Weight on the temporal term vs. relevance in the admission score, 0 to 1. `λ = 0` is the ablation's "component removed" condition. |
| **`θ`** (theta) | Admission threshold — a passage is kept only if its admission score is at least `θ`. |
| **`H`** (half-life) | Days for the temporal score to halve. |
| **Admission score, `A(s)`** | `(1 − λ)·ρ(s) + λ·T(s, q, t_q)` — the single scalar the proposed system thresholds against `θ`. |
| **Currency** | The mean temporal score of the evidence a system actually admitted. The primary outcome metric for the main comparison. |
| **`temporal_candidate`** | A question flag: its cited source (a Cochrane review) has been revised at least once. Diagnostic only — does not assert the verdict changed. The subgroup where a temporal signal should matter most. |
| **Ablation** | Comparing the full proposed system against the same system with the temporal component switched off (`λ = 0`), to isolate whether *that specific signal* is what helps. |
| **Validation split** | The held-out question subset used only to fit `λ`/`θ`/`H`. Never used to report a result. |
| **Test split** | The held-out question subset used only to report a result. Never used to fit parameters. |
| **Frozen candidate set / `FrozenItem`** | The retrieved-and-reranked evidence for one question, cached once and replayed byte-identically to every arm, so no arm can retrieve differently from another. |
| **Corpus snapshot** | An id identifying exactly which corpus/index build a frozen item was produced against. |
| **Context budget** | The maximum number of passages any arm may admit — a control, not a treatment, identical across arms. |
| **Extractive stand-in generator** | A placeholder "generator" that returns the top-admitted passage's own text verbatim, used when a real generative model is not yet in the loop. Explicitly not a thesis-quality generation result. |
| **Groundedness** | Fraction of an answer's content tokens that appear in the evidence it was given — an automatic proxy for faithfulness, not an entailment judgement. |
| **Context precision / recall** | How well a system's admitted evidence matches the question's known gold-relevant evidence, where that annotation exists. |
| **Token F1** | SQuAD-style unigram overlap between a generated answer and the reference answer. |
| **Paired sign test** | The significance test deciding whether a per-question difference between two systems is directionally consistent, rather than driven by a couple of outlier questions. |
| **Provenance firewall** | The rule (and its automated check) that a question's reference evidence must never also appear among its retrieval candidates, which would make the comparison circular. |
