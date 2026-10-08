# Related work and positioning

How this study relates to the literature it builds on, and what it does and does not compare against. Bibliographic
details were written from memory of the cited works and must be checked against the sources before submission. For the
design see `methodology.md`; for the protocol and results see `experimentation.md`.

## 1. Retrieval-augmented generation for medical questions

* **RAG² (Sohn et al., NAACL 2025)** is the base work. It improves retrieval-augmented medical question answering with
  three components: a rationale written by the model as the retrieval query, retrieval balanced across corpora, and a
  trained filter that removes unhelpful passages. It reports gains on multiple-choice exam benchmarks (e.g. MedQA) with
  full-precision models, a 564 GB index and a trained filter that is not distributed. This study keeps the three components
  in kind (R2: rationale query, balance across evidence types, zero-shot LLM filter) and measures them on a different task
  and scale. **R2 is an adapted baseline, not a reproduction**, and its accuracies are not comparable with RAG²'s.
* **MedRAG / benchmarking RAG for medicine (Xiong et al., Findings of ACL 2024)** evaluates retrievers, corpora and
  language models for medical question answering; it motivates the use of MedCPT (Jin et al., 2023) for dense retrieval
  and re-ranking, as used here.

## 2. Time and outdated knowledge

* **MedChange / MedRevQA (Vladika et al., Findings of EMNLP 2025)** provides the benchmark: Cochrane-derived questions with
  labels that change between review versions, and the finding that language models often repeat outdated conclusions.
  This study adds the as-of protocol (evidence first public before the question date) and reports verdict accuracy on
  the changed and unchanged questions. Its gold labels are model-generated; their reproducibility is measured, not assumed
  (`data.md`).
* **TempRALM (Gade and Jetcheva, 2024)** adds a temporal score to a retriever's ranking. The stage-1 Temporal Filter is the
  same idea (relevance plus recency); no novelty is claimed for it, and it did not beat standard retrieval here
  (`experimentation.md` §13).
* Related temporal question-answering benchmarks (e.g. RealTime QA, StreamingQA) concern general news-style knowledge; they
  are cited as context and not used.

## 3. Verification and self-correction

* **Chain-of-Verification (Dhuliawala et al., 2024)**, **Self-Refine (Madaan et al., 2023)** and **Self-RAG (Asai et al.,
  2024)** have the model critique or revise its own output; **corrective RAG (Yan et al., 2024)** evaluates retrieved
  evidence before use. R2V belongs to this family: a second pass checks a draft verdict against admitted evidence under
  explicit criteria (directness, study design, the meaning of each verdict, currency).
* **Huang et al. (ICLR 2024)** show that language models without external feedback often fail to self-correct reasoning.
  This is the relevant caution for R2V, which has no external feedback beyond the evidence it re-reads, and is consistent
  with the small, unconfirmed effect reported here.
* The criteria follow evidence-appraisal practice in systematic reviews (directness and study design in the spirit of
  GRADE) and restate the benchmark's labelling rubric; R2C (criteria without a draft) separates what the criteria alone
  achieve from what verifying a draft adds.

## 4. What this study compares against

| Comparison | Basis | Status |
|---|---|---|
| No evidence (B0), standard retrieval (B1), adapted RAG² (R2) | same generator, questions and decoding | controlled; paired tests |
| R2C, R2V, R2V-ND | same evidence as R2, only the reading differs | controlled; paired tests; the pre-declared requirement is read on R2V − R2 |
| Recency-aware retrieval (TempRALM-style), helpfulness filter, evidence synthesis | stage-1 and stage-2 arms | controlled; negative results of record |
| Other language models (Qwen2.5-7B, Mistral-24B, Llama-3.3-70B, GPT-4o-mini, DeepSeek-V3) | the benchmark authors' released closed-book answers on the same questions | context only: different size, training and prompt, no retrieval |
| RAG² as published | not comparable (different task, scale, corpus and filter) | described, not compared |

Further models were not run: each extra generator costs days of CPU time on the study's hardware, larger ones do not fit,
and the released answers already place the local 8B system among existing models. A stronger generator, a trained filter
and a larger sample are the natural next steps (`experimentation.md` §15).

## 5. References (to verify before submission)

Asai, A. et al. (2024). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. ICLR.
Dhuliawala, S. et al. (2024). Chain-of-Verification reduces hallucination in large language models. Findings of ACL.
Gade, A., Jetcheva, J. (2024). It's about time: Incorporating temporality in retrieval augmented language models. arXiv.
Huang, J. et al. (2024). Large language models cannot self-correct reasoning yet. ICLR.
Jin, Q. et al. (2023). MedCPT: Contrastive pre-trained transformers with large-scale PubMed search logs. Bioinformatics.
Madaan, A. et al. (2023). Self-Refine: Iterative refinement with self-feedback. NeurIPS.
Sohn, J. et al. (2025). RAG²: Rationale-guided retrieval augmented generation for medical question answering. NAACL.
Vladika, J., Dhaini, M., Matthes, F. (2025). Facts fade fast: Evaluating memorization of outdated medical knowledge in large
language models. Findings of EMNLP.
Xiong, G. et al. (2024). Benchmarking retrieval-augmented generation for medicine. Findings of ACL.
Yan, S. et al. (2024). Corrective retrieval augmented generation. arXiv.
