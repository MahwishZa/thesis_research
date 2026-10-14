# Methodology

Status: **Phase 1 specification of the RAG² baseline, for supervisor approval (A2).** Nothing here is built yet. It records what
the original method does, taken from the paper and the authors' code, and what must be decided before each part is rebuilt.
Choices of this thesis (datasets, scale, anything that departs from the paper) are listed as decisions, not assumed.

## 1. Sources and how facts are marked

- **Paper:** Sohn et al., *Rationale-Guided Retrieval Augmented Generation for Medical Question Answering*, NAACL 2025,
  arXiv 2411.00300v2 (22 Sep 2025). Page numbers refer to that PDF. The NAACL camera-ready may differ slightly.
- **Code:** `github.com/dmis-lab/RAG2`, commit `86add43` (retriever and filter-training code, five example labels). The same
  group's earlier project Self-BioRAG (`github.com/dmis-lab/self-biorag`) is the origin of the corpus.
- **MedCPT:** `github.com/ncbi/MedCPT` (public-domain notice), models on Hugging Face.
- Marks: **[Paper p.N]** stated in the paper; **[Code]** read in the released code; **[Open]** not stated anywhere read,
  or the sources disagree; **[Decision]** a choice this thesis must make.

## 2. The original pipeline

Four stages [Paper p.1, p.4; Code README]:

1. The language model writes a chain-of-thought **rationale** for the question.
2. The rationale is the retrieval query. **Balanced retrieval** takes an equal number of snippets from each of four corpora
   (PubMed, PMC, clinical guidelines, textbooks); a **re-ranker** orders the pooled snippets.
3. A small trained **filter** labels each snippet helpful or not helpful; only helpful snippets are kept.
4. The language model answers the question with the kept snippets.

Only stage 3 involves training (the filter). All other models are used as released.

## 3. Part 1: rationale-based query formulation

| Item | Specification | Source |
|---|---|---|
| Rationale prompt | "The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output your explanation and single option from the given options as the final answer. Here is the question: [initial_query]" | Paper p.4 |
| Query used for search | The rationale alone. The initial question is **not** added: both together exceed the retriever's maximum length and gave worse results in the authors' tests. | Paper p.4–5 |
| Who writes it | The same language model that answers (Llama-3-8B-Instruct in the main experiments). | Paper p.5 |
| Query encoder | `ncbi/MedCPT-Query-Encoder`, first-token vector, **`max_length=512`**. (MedCPT's own example uses 64 for short queries.) | Code `retriever/query_encode.py`; MedCPT README |
| Decoding | Greedy, temperature 0, served with vLLM; no quantisation described. | Paper p.13 (A.3) |
| Maximum rationale length | Not stated. | **[Open]** |

## 4. Part 2: balanced retrieval

| Item | Specification | Source |
|---|---|---|
| Corpora | PubMed abstracts, PMC full text, clinical practice guidelines (Meditron collection, 8 of 16 sources public), 18 medical textbooks. Same corpus as Self-BioRAG. | Paper p.13 (A.3) |
| Size in the paper | PubMed 36.5 M documents / 69.7 M snippets; PMC 1.1 M / 46.3 M; guidelines 35.7 k / 607 k; textbooks 18 / 134 k. Total 37.6 M documents, 116.7 M snippets, 564.2 GB. | Paper p.13 (Table A1) |
| Chunking | Sliding window with overlap. Self-BioRAG's released corpus: 128 words, 32-word overlap. | Paper p.13; Self-BioRAG README |
| Search | Exact maximum-inner-product search (768-dimension MedCPT article vectors) in each corpus separately. | Paper p.4 (Eq. 2); Code |
| Balance | An equal number of candidates from each corpus, then pooled. The balance holds before re-ranking only. | Paper p.5; Code |
| Candidates per corpus | Not stated in the paper. Code default: 100; README example: 10. Self-BioRAG: 10 per index group, PubMed split into four groups (40 PubMed candidates against 10 for each other corpus). The RAG² README repeats the four groups, but the code's default uses one group. | **[Open]** |
| Re-ranker | `ncbi/MedCPT-Cross-Encoder`, `max_length=512`. The paper says it scores **the original query** with each snippet. The released code scores the text used for retrieval (the rationale). | Paper p.5; Code `retriever/main.py`, `rerank.py` — **[Open: they disagree]** |
| Final number of snippets | The "optimal top-k", chosen on validation data; the value is not stated in the text. Single-component tests use k = 1. | Paper p.7–8; **[Open]** |

## 5. Part 3: rationale-guided filtering

**Labels** (one label per question–snippet pair; Figure 2, p.4):

| Answer without retrieval | Answer with the snippet | Rationale perplexity lower with the snippet? | Label |
|---|---|---|---|
| correct | correct | yes | helpful |
| correct | correct | no | discarded |
| correct | wrong | (not asked) | not helpful |
| wrong | correct | (not asked) | helpful |
| wrong | wrong | yes | not helpful |
| wrong | wrong | no | discarded |

- Perplexity change: ΔPPL = PPL(x) − PPL(x, d) ≥ τ, with PPL(x, d) = exp(−(1/L) Σ log P(xᵢ | x<ᵢ, d)) [Paper p.4, Eq. 3–4].
  "Lower perplexity" in Figure 2 means ΔPPL ≥ τ.
- τ is the **top 25%** of perplexity differentials, chosen on validation data and fixed for every experiment [Paper p.4].
- Each snippet is labelled on its own, one snippet at a time; the authors name this as a limitation [Paper p.9].
- **[Open]** Eq. 4 defines the perplexity over "the input query x", while the text and Figure 2 say the *rationale's*
  perplexity. The likely reading is the rationale's tokens given the question, with and without the snippet. Which set of
  pairs the 25% is taken over is not stated.
- **[Open]** Which snippet is labelled for a question (the top-ranked one is assumed), how many pairs per question, and the
  size of the training set. The one released example id, `llama3_5%_23600`, implies at least 23,600 examples.

**Model and training:**

| Item | Specification | Source |
|---|---|---|
| Model | Flan-T5-large, 770 million parameters; new tokens `[HELPFUL]`, `[NOT_HELPFUL]`. The launch script's name contains "xl" and does not fix the size. | Paper p.4, p.6; Code |
| Input | "Given the following evidence, determine whether it helps answer the provided question. Evidence: … Question: …". The rationale is not an input. | Code (released sample) |
| Training data | Question–answer pairs from the MedQA and MedMCQA training sets; MedMCQA also for MMLU-Med, which has no training set. | Paper p.7 |
| Settings | Learning rate 3e-5, per-device batch 16, 40 epochs, maximum input 512, stride 128, AdamW, one H100 80 GB. A few checkpoints chosen on validation data, then tested. | Paper p.13; Code launch script |
| Hardware note | The paper says the model can be trained on one RTX 3090 24 GB. | Paper p.6 |
| Use | Softmax over the two label tokens; keep the helpful snippets. | Code README |

## 6. What the paper reports (targets for a directional comparison)

Llama-3-8B-Instruct, accuracy in percent [Paper p.6, p.8]. A restored system cannot match these numbers exactly (different corpora
and scale); the aim is the same direction of effect.

| System | MedQA | MedMCQA | MMLU-Med | Average |
|---|---|---|---|---|
| No retrieval (0-shot) | 57.7 | 53.5 | 69.5 | 60.2 |
| MedCPT, top-1 | 55.3 | 51.3 | 65.8 | 57.5 |
| MedCPT + rationale query, top-1 | 58.0 | 52.1 | 70.3 | 60.1 |
| MedRAG | 56.4 | 56.6 | 69.2 | 60.7 |
| RAG² | 64.6 | 59.4 | 74.8 | 66.3 |

| Filter ablation, top-1, original query (Table 3) | MedQA | MedMCQA |
|---|---|---|
| No filter | 55.3 | 51.3 |
| Adaptive-RAG classifier | 57.3 | 53.1 |
| GPT-4o as filter | 58.5 | 55.8 |
| Rationale-guided filter | 58.6 | 55.8 |

| Balanced retrieval, top-1 (Table 4) | MedQA | MedMCQA | MMLU-Med |
|---|---|---|---|
| MedRAG retrieval | 51.9 | 49.0 | 65.1 |
| Balanced retrieval | 55.3 | 51.3 | 65.8 |

Benchmarks (all four-option multiple choice) [Paper p.5, Table 1]: MedQA 10,178 train / 1,272 validation / 1,273 test; MedMCQA
182,822 / 4,183 / 6,150; MMLU-Med (six subjects) 1,089 test only. Note that plain retrieval (MedCPT, top-1) scores below no
retrieval on MedQA and MedMCQA in the paper itself.

## 7. Decisions required before building

| # | Decision | Recommended default |
|---|---|---|
| D1 | Task format of the restored baseline | Multiple choice, as in the paper |
| D2 | Where Alzheimer's-related exam questions go: training or evaluation | Evaluation, if the evaluation dataset is approved; the filter trains on general questions |
| D3 | The library used to make training labels | Textbook and guideline corpora at reduced scale, if licences allow |
| D4 | Dated (as-of) evidence as a requirement | No; an optional later study |
| D5 | How to obtain the filter | Ask the authors for the labelled file and prepare regeneration on a rented GPU in parallel |
| D6 | What "matches the paper" means | Directional agreement with the table in section 6 |

The evaluation dataset is **not decided here**; its approval is a separate step.

## 8. Questions for the authors

1. The full labelled training file and the validation file (only five examples are released).
2. The label-building script (`classifier/data/preprocess.py` is empty) and the exact rule for ΔPPL and τ.
3. Which questions were used for training (MedQA only, or MedMCQA too) and how many pairs.
4. The Flan-T5 size, and whether the checkpoint still exists.
5. Candidates per corpus, PubMed grouping, and whether the re-ranker scored the question or the rationale.
6. The final top-k and the maximum rationale length.
7. Whether the guideline and textbook corpora are Self-BioRAG's `CPG_128` and `Textbook_128`, and the terms for reuse.

## 9. Acceptance tests to write with each part

Each test is written when its part is built and uses small fixed examples, no models.

- **Query:** encoding at 512 tokens matches the authors' `query_encode.py` output on fixed texts; the share of rationales cut off
  by the length limit is reported.
- **Retrieval:** on a toy four-corpus example with fixed scores, pooling and re-ranking return the same ids in the same order as
  the authors' `main.py` logic; the share each corpus contributes to the final list is reported.
- **Labels:** one hand-made example per row of the table in section 5, including both discard rows.
- **Filter:** the input text format equals the released sample; the decision uses the softmax over the two label tokens; held-out
  accuracy and AUC are compared with the majority rate and with a relevance-score baseline, with margins fixed beforehand.
- **Control test of the labels:** before any training, a retrieved passage must differ from an irrelevant passage in the label
  signal, with the rule and sample size fixed in advance.

## 10. Status of each part

| Part | Faithful (F), adapted (A) or substitute (S) | State |
|---|---|---|
| 1 Rationale query | to be set when built | not built |
| 2 Balanced retrieval | to be set when built | not built |
| 3 Rationale-guided filter | to be set when built | not built |

A part is called "RAG²" only if it is F, or A with its deviation written down. An S part is always named a substitute.
