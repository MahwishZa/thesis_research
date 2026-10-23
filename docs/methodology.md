# Methodology

Status: **specification of the RAG² baseline and of its adaptation.** Revised 2026-10-15 after the researcher
approved the direction of the evidence report (restore the baseline faithfully first, then test one extension), and again on
2026-10-23, when the researcher accepted the dataset roles, the k grid, the filter plan and the working defaults recorded in
sections 5, 7.6, 7.7, 7.9 and 9. The goal is to finish the baseline as faithfully and accurately as the evidence and the hardware
allow; the extension waits until the baseline is finished (D25). Supervisor approval (A2) is pending, and the corpus choice
(D8, D9) waits for it. Part 1 (rationale-based query formulation) is built in `src/rationale_query/` and Part 2 (balanced
retrieval) in `src/balanced_retrieval/`; Part 3 is not. The requirements below are written as executable acceptance tests in `tests/spec/test_*.py`; each test skips until the
part it checks exists (section 11).

The evaluation datasets and their roles are decided in D12. Where the paper and the released code leave a point open, this document
states a **working default**, the least-deviating choice, and says so; the authors' answer, if it comes, replaces it.

## 1. Sources and how facts are marked

- **Paper:** Sohn et al., *Rationale-Guided Retrieval Augmented Generation for Medical Question Answering*, NAACL 2025,
  arXiv 2411.00300v2 (22 Sep 2025). Page numbers refer to that PDF; the NAACL camera-ready may differ slightly.
- **Code:** `github.com/dmis-lab/RAG2`, commit `86add43` (retriever and filter-training code, five example labels). The same
  group's earlier project Self-BioRAG (`github.com/dmis-lab/self-biorag`) is the origin of the corpus and of the retriever.
- **MedCPT:** `github.com/ncbi/MedCPT` (public-domain notice); models on Hugging Face.
- Marks: **[Paper p.N]** stated in the paper; **[Code]** read in the released code; **[Figure]** read off a figure, so
  approximate; **[Inferred]** my reasoning from the facts around it; **[Open]** not stated anywhere read, or the sources
  disagree; **[Decision]** a choice this thesis must make; **[Rule]** an engineering rule of this project, not of the paper;
  **[Working default]** the least-deviating choice where the paper and the code are silent, accepted by the researcher on
  2026-10-23 and open to the authors' answer; **[Researcher]** a fact the researcher checked on the dataset itself (a dataset card
  or the data), not seen by the author of this document.
- Sources read on 2026-10-23 for the open points: the released RAG² code (`86add43`), Self-BioRAG (`ad348ba`, a different method
  from the same group) and MedRAG (`7599a72`, a baseline of the paper). Neither of the last two is RAG²'s answer code, so
  their prompts are not used.

## 2. The original pipeline

Four stages [Paper p.1, p.4; Code README]:

1. The language model writes a chain-of-thought **rationale** for the question.
2. The rationale is the retrieval query. **Balanced retrieval** takes the same number of candidates from each of four corpora
   (PubMed, PMC, clinical guidelines, textbooks); a **re-ranker** orders the pooled candidates and the best k are kept.
3. A small trained **filter** judges each kept snippet on its own, helpful or not helpful; only helpful snippets are kept.
4. The language model answers the question with the kept snippets (greedy decoding).

The number of snippets k is chosen on validation data for each dataset. For Llama-3-8B-Instruct the best k is **32 on MedQA**,
**16 on MedMCQA** and about **8 on MMLU-Med** [Paper p.7 Figure 3; p.14 Figure A1; Figure]. Only stage 3 involves training.
All other models are used as released.

## 3. Part 1: rationale-based query formulation

| Item | Specification | Source |
|---|---|---|
| Rationale prompt | "The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output your explanation and single option from the given options as the final answer. Here is the question: [initial_query]". The exact line breaks are not recoverable from the PDF. | Paper p.4 |
| Query used for search | The rationale alone. The initial question is **not** added: both together exceed the retriever's maximum length and gave worse results in the authors' tests. | Paper p.4–5 |
| Who writes it | The same language model that answers (Llama-3-8B-Instruct in the main experiments). | Paper p.5 |
| Query encoder | `ncbi/MedCPT-Query-Encoder`, first-token vector, **`max_length=512`**. (MedCPT's own example uses 64 for short queries.) | Code `retriever/query_encode.py`; MedCPT README |
| Sentence separators | `main.py` inserts `[SEP]` only into the instruction-set queries, and only when `--use_spacy` is given; a rationale query is encoded as written. Whether the corpora were embedded with separators is not stated. This implementation never inserts one. | Code; **[Open]** for the corpora |
| Decoding | Greedy, temperature 0, served with vLLM; no quantisation described. | Paper p.13 (A.3) |
| Maximum rationale length | Not stated. | **[Open]** |
| Answer-generation prompt | Not published; the README step 4 is prose only. A search of the released code and of Self-BioRAG found no RAG² answer template. Working default D14. | Code README; **[Open]**; **[Working default]** |
| Empty rationale | A failed rationale is an error and is reported as such; it is never silently replaced by the question, because that would change the method. | **[Rule]** |

How Part 1 is built here (`src/rationale_query/`; the experiment that runs it is `experiments/01_rationale_writers/`):

| Item | Choice | Status |
|---|---|---|
| Generator | Llama-3-8B-Instruct as a 4-bit GGUF file, run with llama.cpp through llama-cpp-python (0.3.35 was used) | Declared deviation (allowed if declared, section 8) |
| Prompt | The paper's text with one line break before "Here is the question:", sent as one user turn with no system prompt, in the Llama-3-Instruct chat format. This code writes that format itself and hands llama.cpp the token list, so the start-of-text token appears once; the template stored in the model file is rendered and compared, and a file whose template differs is refused. | **[Open]** line break and system prompt; **[Rule]** the rest |
| Decoding | Every sampling value is written out (temperature 0, top-k 1, neutral penalties) because the library's own defaults are not greedy; the cache is emptied before every question | **[Rule]** |
| Rationale length | At most 512 new tokens (the encoder reads at most 512 tokens); a rationale cut at the limit is flagged | **[Decision]**; the paper is silent |
| Context | 8192 tokens, the window of Llama-3; a prompt that does not fit is an error and is never shortened | **[Rule]** |
| Encoder | One query at a time, first-token vector, 512 tokens, 32-bit, not normalised; the token count and whether the text was cut are recorded | **[Code]** |
| Records | One line per question, written and flushed at once; the settings that can change a rationale are recorded and bind the output folder, so a run with other settings or a question whose text changed is refused; a failure is recorded, never hidden, and five failures in a row stop the run | **[Rule]** |
| Other writers | Any instruction-tuned model in GGUF form can write the rationales with the chat template stored in its file; the writers compared are listed in `experiments/01_rationale_writers/settings.json` | **[Decision]** |

## 4. Part 2: balanced retrieval

| Item | Specification | Source |
|---|---|---|
| Corpora | PubMed abstracts, PMC full text, clinical practice guidelines (Meditron collection, 8 of 16 sources public), 18 medical textbooks. Same corpus as Self-BioRAG, kept "due to its already proven effectiveness". | Paper p.13 (A.3) |
| Size in the paper | PubMed 36.5 M documents / 69.7 M snippets; PMC 1.1 M / 46.3 M; guidelines 35.7 k / 607 k; textbooks 18 / 134 k. Total 37.6 M documents, 116.7 M snippets, 564.2 GB. | Paper p.13 (Table A1) |
| Chunking | Sliding window with overlap. Self-BioRAG's released corpus: 128 words, 32-word overlap. | Paper p.13; Self-BioRAG README |
| Storage and search | Everything is local and offline: precomputed 768-number MedCPT article vectors, one exact inner-product index (`faiss.IndexFlatIP`) per corpus. PubMed is 38 chunk files searched in groups to save memory; PMC has two files (main text, abstracts); guidelines and textbooks one each. No web service is called. | Code `retriever/main.py`, `retrieve.py` |
| Candidates per corpus | **One parameter, `--top_k`, sets both the number of candidates taken from each corpus and the final number kept** (code default 100, README example 10). With k = 32 and one PubMed group the pool holds 128 candidates. | Code `main.py`, `rerank.py`, README |
| PubMed groups | The README says four groups (10/10/10/8 chunks), giving up to 7 × k pooled candidates; Self-BioRAG used 40 PubMed and 10 each from the other corpora (70 pooled, 10 kept). The RAG² code's default groups all 38 chunks into one. | Code; **[Open: which was used]** |
| Pooling | The four lists are concatenated in the order PubMed, PMC, guidelines, textbooks, with **no duplicate removal**. A corpus that holds fewer snippets than asked for adds fewer; the gap is not filled from another corpus. An unknown corpus name, or a search that fails, is an error, never a silently dropped corpus. | Code `rerank.py`; the last two sentences are a **[Rule]** |
| Balance | Equal candidates per corpus before re-ranking only; nothing keeps the final list balanced. | Paper p.5; Code |
| Re-ranker | `ncbi/MedCPT-Cross-Encoder`, `max_length=512`, scores the pair [query, snippet]; descending order, equal scores keep pool order (a stable sort); the best k are kept. The paper says the query is **the original question**; the released code uses the retrieval text (the rationale). | Paper p.5; Code — **[Open: they disagree]** |

How Part 2 is built here (`src/balanced_retrieval/`; the plan was approved by the researcher on 2026-10-17, the supervisor's approval is pending):

| Item | Choice | Status |
|---|---|---|
| Corpora | A hybrid: PubMed and PMC scoped to Alzheimer's disease and its neighbours (decision D9); clinical guidelines and textbooks general and whole, from the paper's own sources (the public part of the Meditron guideline set; the 18 MedQA textbooks). Licences of the last two are unchecked. | **[Decision]**, declared adaptation of scope and size |
| Fetching | `fetch pubmed --as-of DATE` searches the scope of `scope.py` (Alzheimer's disease, other dementias, mild cognitive impairment, amyloid and tau disease, and the neighbouring conditions only when the record is also about dementia or cognition), English, with an abstract, retracted records removed, by publication date up to DATE, through NCBI's E-utilities; windows of dates are halved until each holds fewer than 10,000 hits. `fetch pmc` takes the full text of the same records that carry a PMC identifier (open-access articles only; tables, figures, formulas and references left out; the licence group of each is kept). `fetch guidelines` and `fetch textbooks` take the open Meditron guideline file and the 18 textbooks of `MedRAG/textbooks` (pieces joined back into books). Every download records its settings and, for the hub, the dataset revision; PubMed and PMC are resumable | A declared adaptation of scope and size; the extraction of PMC text, the joining of textbook pieces and the choice of the guideline file are **[Rule]** (the authors' own preprocessing is **[Open]**); the licences of the guidelines and the textbooks are unchecked |
| Passages | 128 words with 32 words of overlap, cut by `chunking.py`; a word is a run between white space and the last window is the one that reaches the end of the text | Self-BioRAG's sizes **[Paper p.13]**; how it split words and ended the last window is **[Open]** |
| Vectors | `ncbi/MedCPT-Article-Encoder`, first-token vector, 512 tokens, 32-bit, not normalised (16-bit storage is an option and is recorded); a 16-bit arithmetic option (`--compute-dtype float16`) exists, is recorded in the embedding state and is measured by `precision-check`; it is not used unless the measurement is accepted; a passage is encoded as one text, whereas MedCPT's own example passes title and abstract as a pair | Usage as the query encoder **[Code]**; the authors' choice for snippets is **[Open]**; the stored PMC vectors have unit length and are not reused (D8) |
| Search | Exact inner product over a memory-mapped matrix, block by block, so a corpus larger than memory can be searched; ties go to the lower row number | Same arithmetic as `faiss.IndexFlatIP` **[Code]**; the block-wise reading replaces the authors' in-memory index **[Rule]** |
| k | One value for the candidates per corpus and for the final list, required on the command line (no default) | **[Code]** |
| Pooling and re-ranking | As in the table above: fixed order, nothing removed, a short corpus not topped up, stable ordering, no quota; the cross-encoder reads the pairs in batches, which gives the same scores up to rounding | **[Code]**, **[Rule]** |
| Re-ranker input | The original question (the paper); the rationale (the released code) is selectable and is a labelled sensitivity check | **[Open: they disagree]**; D21 follows the paper |
| Output | One record per question in `retrieved.jsonl`: the final passages with corpus, document key, chunk, row, dense score, re-ranking score and text; settings, corpora hashes and the hash of the rationale are bound to the folder, so another setting is refused | **[Rule]** |
| Diagnostics | Share of each corpus in the final lists and the cross-corpus duplicate rate (section 7.4), reported by `report` | **[Rule]** |

Corpus ablation (MedQA, Llama-3-8B-Instruct, best accuracy over k, **[Figure]**, p.15): PubMed alone about 59.8 (k = 32), PMC
about 58.4 (k = 16), guidelines about 58, textbooks about 55.5 (k = 16); stacked retrieval about 59.5, stacked with re-ranking
about 60.9, balanced retrieval with re-ranking 63.4, and with the filter 64.6. The authors conclude that "no single corpus
contained sufficient information" (p.15). A single-corpus design therefore corresponds to the weaker condition of this ablation.

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
  "Lower perplexity" in Figure 2 means ΔPPL ≥ τ, so a value equal to τ counts as lower.
- τ is the **top 25%** of perplexity differentials, chosen on validation data and fixed for every experiment [Paper p.4].
- Each snippet is labelled on its own; the authors name this as a limitation [Paper p.9].
- **[Rule]** Inputs are checked before the rule is applied: a perplexity that is missing, not finite or not above zero, a τ that
  is not finite, or an answer that is neither correct nor wrong is an error, never a label. How an answer that cannot be read
  counts is not stated in the paper; the **[Working default]** is D16 (wrong when scoring, the pair skipped when labelling),
  settled before the label function is called.
- **[Open]** Eq. 4 defines the perplexity over "the input query x", while the text and Figure 2 say the *rationale's*
  perplexity. Eq. 4 uses the same x and the same length L in both lines [Paper p.4], so the same text is scored without and with
  the snippet. **[Working default]** D17: that text is the rationale written without evidence. Which set of pairs the 25% is
  taken over, and how a quarter is rounded, are not stated; **[Working default]** D18 fixes the pool, and the code of this
  project takes the pool as an explicit argument so that the choice is visible.
- **[Open]** Which snippets are labelled for a question, how many pairs per question, and what "5%" in the file name means. The
  five released ids run up to 23600 while MedQA has 10,178 training questions [Paper p.5, Table 1], so if the file holds MedQA
  only it has at least about 2.3 pairs per question; it may also mix datasets **[Inferred]**. **[Working default]** D19 (top 4)
  and D20 (the query that retrieves them).

**Model and training:**

| Item | Specification | Source |
|---|---|---|
| Model | Flan-T5-large, 770 million parameters; new tokens `[HELPFUL]`, `[NOT_HELPFUL]`. The launch script's name contains "xl" and does not fix the size. | Paper p.4, p.6; Code |
| Input | `Given the following evidence, determine whether it helps answer the provided question.` + blank line + `Evidence: ` + snippet + blank line + `Question: ` + question, with both parts stripped of surrounding whitespace. Identical in all five released examples. The question text carries the four options inline (`A) … B) … C) … D) …`). The rationale is **not** an input. | Code (released sample, checked) |
| Snippets | Short passages of about 128 words that may end mid-sentence. | Code (released sample); Self-BioRAG README |
| Training data | Question–answer pairs from the MedQA and MedMCQA training sets; MedMCQA also for MMLU-Med, which has no training set. The launch script trains on `llama3_5%-train.json`; the five sample ids run up to 23600, so the file holds at least about 23,600 examples. | Paper p.7; Code; **[Inferred]** size |
| Settings | Learning rate 3e-5, per-device batch 16, 40 epochs, maximum input 512 tokens with overflow windows of stride 128, AdamW, one H100 80 GB. A few checkpoints chosen on validation data, then tested. | Paper p.13; Code launch script |
| Learning | Ordinary sequence-to-sequence cross-entropy on the label token; no class weighting. An input longer than 512 tokens becomes several overlapping examples with the same label (inherited from a question-answering script). | Code `utils.py`, `run_classifier.py` |
| Hardware note | The paper says the model can be trained on one RTX 3090 24 GB. | Paper p.6 |
| Decision rule | Softmax over the logits of `[HELPFUL]` and `[NOT_HELPFUL]` at the first generated position; the larger wins. An exact tie counts as helpful, because the authors' `argmax` picks index 0. Each snippet is judged alone, from its own filter input; a failed judgement is an error, never a default. Only helpful snippets reach the answering model, in their original order. | Code `run_classifier.py`, README; the failed-judgement sentence is a **[Rule]** |

**How the filter is built here** (decisions D22 and D23; nothing is built yet):

- **One filter per training dataset [Inferred].** The paper trains "using QA pairs from MedQA and MedMCQA" and, because MMLU-Med
  has no training data, uses the MedMCQA training data for it [Paper p.7]. The plan: a MedQA filter scores MedQA; a MedMCQA
  filter scores MedMCQA and MMLU-Med. Whether the authors trained one filter on both is a question for them (section 10).
- **Size.** Flan-T5-base first, as a declared smaller stand-in (about 250 million parameters **[Inferred]**), then Flan-T5-large as
  in the paper when a GPU with 24 GB is available [Paper p.6: "can be trained on a single RTX 3090 24G"]. The paper's settings
  are kept (learning rate 3e-5, per-device batch 16, 40 epochs, maximum input 512 tokens, stride 128). No other method is added
  (no LoRA, no other size). Results with the base model are named "base", never "reproduction".
- **Memory [Inferred].** Full fine-tuning with AdamW needs about 16 bytes per parameter: roughly 4 GB for the base model and 12 GB
  for the large one, before activations. The laptop GPU has 4 GB, so even the base model may not fit; a one-batch memory test
  decides, and it has not been run. Filter training needs only the finished label file, so it may run on another machine.
  **[Open]** If the per-device batch of 16 does not fit, the effective batch of 16 can be reached by gradient accumulation; this
  is a hardware measure to be confirmed after the memory test.
- **Label scale (D23).** MedQA: all 10,178 training questions, labelled in resumable stages and checked after each stage.
  MedMCQA: a staged subset, extended as far as compute allows, with the final size reported. Measured: about 60 s per generation
  at a 256-token cap (git history `46ac455^:docs/log.md`, Phase 13); not measured at longer caps. **[Inferred]** at 45 to 120 s
  per generation, about 5 generations per question: MedQA about 51,000 generations (about 640 to 1,700 GPU hours), MedMCQA in full
  about 914,000 (about 11,400 to 30,500 hours). The full MedMCQA set is therefore out of reach on the laptop whatever the
  deadline; the timeline is for the supervisor.

## 6. What the paper reports (targets for a directional comparison)

Llama-3-8B-Instruct, accuracy in percent [Paper p.6, p.8]. A restored system cannot match these numbers exactly (different
corpora and scale); the aim is the same direction of effect.

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

| Who writes the rationale (Table 5, MedQA) | Llama-3-8B backbone | Meerkat-7B backbone | GPT-4o backbone |
|---|---|---|---|
| Llama-3-8B rationales | 63.4 | 71.3 | 87.4 |
| Meerkat-7B rationales | 71.5 | 74.6 | 88.3 |
| GPT-4o rationales | 73.6 | 78.8 | 89.8 |

Benchmarks (all four-option multiple choice) [Paper p.5, Table 1]: MedQA 10,178 train / 1,272 validation / 1,273 test; MedMCQA
182,822 / 4,183 / 6,150; MMLU-Med (six subjects) 1,089 test only. The MedMCQA test answers are not public (section 7.7), so the
4,183-question split is the only scorable one besides training; the paper does not say which MedMCQA split its table reports, and
the same group's Self-BioRAG ships a 4,183-row file tagged `MedMCQA_dev` as its test file [Code, Self-BioRAG]; so the authors most
likely scored the 4,183 questions **[Inferred]**.

Notes: plain retrieval (MedCPT, top-1) scores below no retrieval on MedQA and MedMCQA in the paper itself. The filter's gain
depends on the dataset: at each dataset's best k about +1.2 points on MedQA (between about −0.1 and +1.5 across k), roughly
+2 to +3 on MedMCQA and about +3.9 on MMLU-Med [Figure, p.7 and p.14]. Better rationales raise the Llama-3-8B backbone from 63.4
to 73.6 on MedQA, partly because the stronger model's own knowledge enters the query.

## 7. The Alzheimer's-scoped adaptation (declared deviations from the paper)

Status (2026-10-23): sections 7.1 to 7.5 describe the corpora as built (PubMed and PMC scoped to Alzheimer's disease). Whether that
scope stays is the supervisor's decision (D8, D9) and is on hold; sections 7.6 to 7.9 (k, data roles, deviations, analysis) do not
depend on it.

### 7.1 Principle: build with services, evaluate from frozen local copies

Four terms are kept apart. A **corpus** is a fixed, versioned library stored on disk. An **index** is vectors computed from it
and can be rebuilt. **Live API retrieval** is searching a web service at question time; its answers change with the day and the
service, so it is never used during an evaluation. A **cache** holds past API answers; it is not a corpus. Services and bulk
downloads are used only to *create* a corpus version; every experiment then runs offline against a named version. This is also
how the authors' retriever works.

### 7.2 Scope [Decision D9]

"Alzheimer's disease and its neighbours": Alzheimer's disease; other dementias; mild cognitive impairment; the conditions that
must be told apart from it or that coexist (delirium, depression, normal-pressure hydrocephalus, vitamin B12 and thyroid causes,
Parkinson-related dementia, stroke); biomarkers and imaging; drugs and their adverse effects; prevention and risk factors; care,
capacity and ethics. The boundary is checked by the share of development questions for which a collection holds nothing relevant
(no snippet above a fixed relevance score); that share is reported.

### 7.3 The four collections

Every passage carries: corpus, document key, chunk index and text. What a document is made from is kept in the downloads beside
the corpus (`retrieval_corpora\documents\`): the date (PubMed), the licence group of each article (PMC), the source of each guideline,
and a state file with the search terms, the snapshot date and the identifiers, or a `.source.json` with the dataset revision. PMC
holds only articles of the Open Access subset, and their licence groups (cc-by, cc-by-nc, cc-by-nc-nd, other) are kept, not
filtered. The guidelines are the part of the Meditron collection that its authors release for redistribution; the textbooks are
copyrighted books, used for research and not redistributed. The licences of the guidelines and the textbooks were **not**
verified and are stated as a limitation. The authors' own downloads (`CPG_128`, `Textbook_128`, `PMC_128`, `PubMed_128` of Self-BioRAG) were not reachable from the
researcher's laptop on 2026-10-15.

| Corpus | Source and keys | Version and maintenance | Counts on file |
|---|---|---|---|
| PubMed | Title and abstract (a structured abstract keeps its section labels) of the records that match the scope of `scope.py` (MeSH and title/abstract terms, section 7.2), English, with an abstract, retracted records removed; the date, DOI and PMC number are kept beside the text. Key: PMID. Obtained once through the E-utilities service. | A named snapshot by publication date (`--as-of`); PubMed adds records for old dates later, so the identifiers kept with the download, not the date, define the corpus. A new version only on purpose. The core query grows by about 17,000 to 20,000 records a year. | 411,751 records and 1,048,208 passages of 128 words (scope of `scope.py`, publications up to 2026-10-18; 413,232 identifiers, of which 215 had no abstract and 589 were retracted, and 677 were not returned as journal articles); the first design's query bank found 255,907 on 2026-09-13 |
| PMC | Open-access full text for the PubMed set (PMC Open Access subset). Keys: PMCID, PMID, DOI; licence group kept per article; only the latest version of an article. | Add open-access articles whose PMIDs join the PubMed set; keep the version table. | 158,574 open-access articles from the 187,257 PubMed records of the 2026-10-18 set that carry a PMC number (28,619 had no full text, 64 were not in English; downloaded 2026-10-19), which give 8,205,229 passages of 128 words. The first design held 114,256 articles, 4,377,041 chunks of 256 tokens (32-token overlap) and a 12.5 GB vector file on the researcher's laptop. The stored vectors have unit length (norms 0.9999998 to 1.0000001, checked 2026-10-15) while the authors' retriever uses the unnormalised MedCPT vectors, so inner-product fidelity needs re-embedding (D8). |
| Guidelines | General and whole (hybrid plan): the open file of the Meditron guideline collection (`epfl-llm/guidelines`), as the paper's corpus. Key: the address of the guideline, else its identifier. | A named dataset revision. | 37,970 documents and 721,364 passages of 128 words (open file of `epfl-llm/guidelines`, revision `a8f0269`, 2026-10-18); the paper has 35.7 k documents and 607 k snippets, so this file holds 6% more documents and 19% more passages. The paper's exact release is **[Open]** |
| Textbooks | General and whole (hybrid plan): the 18 MedQA textbooks as published in `MedRAG/textbooks`, joined back into one document per book. Key: the book. | Static editions; a named dataset revision. | 18 books and 133,903 passages of 128 words (`MedRAG/textbooks`, revision `9c72838`, 2026-10-18); the paper has about 134,000 |

The **document key** is one canonical identity per document (PMID when known, otherwise PMCID, DOI or normalised title). It is
assigned when a library is built and is the same in every corpus that holds the same document. A snippet whose identity is
unknown has an empty key and never counts as a duplicate.

### 7.4 Overlap policy

1. Inside a corpus keep one record per key; keep only the latest version of a PMC article.
2. First **reproduce the paper**: no removal across corpora. Measure the duplicate rate (the share of final snippets whose key also
   occurs in a snippet of another corpus) and each corpus's share of the final list. Report both.
3. Then run a deduplicated arm as a sensitivity check.
4. Remove from the library any document that an evaluation question was written from, where such a link exists.

### 7.5 Two libraries [Decision D3]

The **deployment library** is the four collections of the hybrid plan (PubMed and PMC scoped to Alzheimer's disease and its neighbours, the guidelines and textbooks general and whole), searched when answering. The **label-building library** is a
reduced general textbook and guideline set (the closest match to exam questions), used only to create filter-training labels from
general training questions. If it cannot be assembled within the licences, labels are built from the Alzheimer's collections with
Alzheimer's-related questions only, as a declared small-scale variant.

### 7.6 Number of snippets [Decisions D7 and D13]

**k grid.** The paper's grid, k in {1, 2, 4, 8, 16, 32}, is kept whole; no value is dropped [Paper p.7, Figure 3]. k is chosen per
dataset on validation data only: MedQA on the 1,272 validation questions, MedMCQA on the 4,183 questions of its validation split.
The paper says its k was "determined through validation" [Paper p.7]; it does not say whether the curves of Figures 3 and A1 are
validation or test accuracy, and its Table 2 values equal the peaks of those curves (MedQA 64.6 at k = 32, MedMCQA 59.4 at k = 16,
MMLU-Med 74.8 at about k = 8) [Figure; Paper p.6, p.7, p.14]. **[Open]**
**MMLU-Med.** It has no validation set in the paper, so the authors' k for it was read from a test curve **[Inferred]**. Here its
k is the k chosen on MedMCQA validation, because the MedMCQA filter is what scores it. This is a declared deviation. The 123
validation and 30 dev questions of the dataset copy are not used (too few to choose k, and the paper does not use them).

The paper's best k (32 on MedQA) means 128 pooled candidates, 32 filter decisions (about 43 s per question on the researcher's CPU,
derived from 1.33 s each) and a long answering prompt. Time per question is measured in a calibration run (a 20-question timing
pilot on the researcher's laptop, not yet run); the measurement informs the plan and never removes a value of k.

### 7.7 Data roles and leakage rules

| Role | Meaning |
|---|---|
| Model-training data | Rows (question id, snippet, label) **built** from training questions, a library and the generator. |
| Validation questions | Used to choose τ, k and checkpoints; never to learn. |
| Test questions | Used once at the end; never to train, tune or choose. |
| Retrieval library | Passages, not questions; separate from all question sets. |

**Datasets and roles (D12).** The roles are those of the paper's Table 1 [Paper p.5], except where the data do not allow it.

| Dataset (source) | Training | Validation | Test | Notes |
|---|---|---|---|---|
| MedQA, US 4-option (`awinml/medqa`, configuration `questions`) | 10,178 (labels) | 1,272 (k, checkpoint) | 1,273, scored once | A re-upload of the official release with its official split; options A to D, `answer_idx`; MIT licence, inherited **[Researcher]** |
| MedMCQA (`openlifescienceai/medmcqa`) | 182,822 (labels; a staged subset, D23) | 4,183 (k, checkpoint) | **none**: the 6,150 test answers are withheld | The `cop` field of the test split is -1 in every row **[Researcher]**; the stored `cop` values are 0 to 3 (A = 0 … D = 3), not 1 to 4 as the card says **[Researcher]**. The dataset authors state that they do not release the test ground truth and score submitted predictions through a form (`github.com/medmcqa/medmcqa` README, read 2026-10-23). No MedMCQA accuracy is reported as a test result. |
| MMLU-Med (`Layered-Labs/benchbase-mmlu-medical`, from `cais/mmlu`) | none | none | 1,089 (six subjects), scored once, with the MedMCQA filter | The copy also holds 123 validation and 30 dev rows, which are not used **[Researcher]**. The licence field of the copy is unclear |

The blind submission of MedMCQA predictions is **not used** for now. If it is used later, with the supervisor's approval: one final
submission per system, fixed before any test result, the answer index written as 1 to 4 as the README requires, and no paired
test (only an overall score comes back, if anything). The MedMCQA test labels are not reconstructed from other splits: the
authors withhold them on purpose, and such labels would belong to questions that have look-alikes in training.
No additional evaluation dataset is used for now.

1. Split by question: all rows of a question stay in one role. A question's role is a function of its id and a recorded seed
   only; it does not depend on row order, on the Python process, or on which other questions are present, so adding or removing
   questions never moves a question to another role.
2. Remove exact and near duplicates across roles, including across MedQA and MedMCQA (not yet done). The MedMCQA authors removed
   similar questions between their splits [MedMCQA card and README]; a read-only train-versus-validation check on the laptop
   shows whether the validation split is clean for choosing k.
3. Fit and tune on training and validation data only; keep a manifest of ids and hashes made before any run.
4. Labels made by the generator under evaluation are part of the method, but the questions that produced them must be disjoint from
   every test question.
5. Pretraining contamination of the generator cannot be removed; compare systems that share one generator and say so.
6. **Test discipline (D24).** Everything is chosen on validation data: k, the checkpoint, the unreadable-answer rate and every
   threshold check. All settings are frozen and written down before a test set is touched. Each test set is scored once per
   system. No second run follows a score. A fault found afterwards is reported with both results, never by replacing the first.

### 7.8 Declared deviations expected

Scale and scope of the corpora (the paper has 116.7 M passages; the corpus choice waits for the supervisor, D8 and D9, and the
built corpora hold about 10.1 M passages with PubMed and PMC scoped to Alzheimer's disease and its neighbours); 16-bit arithmetic of the article
encoder for PubMed and PMC (D11); a 4-bit generator (a 4 GB GPU; the paper describes no quantisation); the filter first as
Flan-T5-base (a stand-in) and a staged MedMCQA label set (D22, D23); no MedMCQA test result (D12); the k of MMLU-Med taken from
MedMCQA validation (D13); the evidence layout, the generation cap, the answer-reading rule, the perplexity text, the τ pool, the
snippets per question and the training retrieval query of D14 to D20, where the paper and the code are silent; pretraining
contamination.

### 7.9 Analysis (decision D24)

Systems are compared on the same questions with paired methods: the exact McNemar test (a two-sided binomial test on the
questions where the two systems differ), a paired bootstrap interval of the difference in accuracy (questions resampled as units),
and Holm's adjustment for the comparisons declared in advance. The arms are no retrieval, the retrieval system without the
filter, and the full system, over the k grid. The statistics are declared before any run. The paper reports plain accuracy only
[Paper Tables 2 to 5], so this adds to the analysis and does not change the method. They apply to MedQA and MMLU-Med, whose
answers are public; there is no MedMCQA test. The archived `stats.py` of `_archive/rag2_adaptation_v1/` implements all three; its
nine tests passed in an isolated copy on 2026-10-23 (the binomial p-values were not compared with a second library). **[Open]**
Which comparisons form the Holm family is fixed before the first test run.

## 8. Fidelity

**Wording rule.** Call the baseline "a faithful re-implementation of RAG² with scale-reduced corpora" (the scope of the corpora
follows the supervisor's choice, D8 and D9), with a deviation table. Use "reproduction" only if the authors' labels or checkpoint are obtained and used. No percentage of fidelity is
stated: no source defines one and the items below cannot be added up.

| Tier | Items | Rule |
|---|---|---|
| **Essential (must match)** | E1 the rationale alone is the dense query, written by the model that answers, with the paper's prompt; E2 separate exact search in four corpora, an equal number of candidates from each, one pooled cross-encoder re-ranking against the original question; E3 a small sequence-to-sequence filter trained on labels from the Figure 2 rule with τ at the top 25%, reading question and snippet only, applied to every kept snippet; E4 Llama-3-8B-Instruct without fine-tuning and greedy decoding; E5 k, τ and checkpoints chosen on validation data only | A change makes the part an *adaptation* or a *substitute*; name it |
| **Allowed if declared** | corpus scope and size; label-set size; filter size or tuning method; benchmark; quantised generator | Record the reason; measure the effect where possible |
| **Replaceable** | serving software, exact-search software, hardware, seeds, batch sizes, file formats, checkpoint-selection details | List them |

Checklist (pass, fail or declared; no score):

1. The rationale prompt equals the paper's text and only the rationale is encoded, at 512 tokens.
2. Four separately indexed corpora, exact inner-product search, the same candidate count from each, no keyword pre-filtering.
3. One parameter sets the candidates per corpus and the final count; k is chosen on validation data.
4. Re-ranking uses the original question; the code's rationale variant is run as a labelled sensitivity check.
5. The label rule matches Figure 2 including both discard branches, with one hand-made case per branch.
6. The filter input equals the released sample format; the decision is the softmax over the two label tokens.
7. Filter and generator are trained or tuned on data disjoint from every test item, split by question.
8. Models and files are pinned by revision or hash; every deviation appears in one table.
9. The restored system is judged on directional agreement with the paper's ablations (rationale helps, balance helps, filter
   helps), with intervals, and is never tuned to improve that agreement.

## 9. Decisions

| # | Decision | Status |
|---|---|---|
| D1 | Task format of the restored baseline | **Adopted:** multiple choice, as in the paper |
| D2 | Where Alzheimer's-related exam questions go (training or evaluation) | **Open**, and not urgent: no Alzheimer's evaluation set is used for now (D12, D25); revisit if one is added |
| D3 | The library used to make training labels | **Adopted default** (section 7.5); licence screen pending |
| D4 | Dated (as-of) evidence as a requirement | **Adopted:** not required; an optional later study |
| D5 | How to obtain the filter | **Adopted:** ask the authors (labels first) and prepare a scale-reduced build in parallel. The researcher emailed Dr Sohn, Dr Kim and Prof. Kang (the labelled files or the label-building script, the perplexity definition and pool, a checkpoint and the model size, the meaning of "5%", the answer prompt, the rationale length, the re-ranker query); no reply by 2026-10-23 |
| D6 | What "matches the paper" means | **Adopted:** directional agreement with section 6 |
| D7 | k and evaluation size | **Adopted** (2026-10-23): the paper's k grid, chosen on validation (D13); every test set in full, scored once (D12, D24) |
| D8 | PMC chunks: keep the existing 256-token chunks or re-chunk to 128-word windows | **Adopted default** (hybrid plan, 2026-10-17): re-chunk to 128 words with 32 overlap and re-embed with raw vectors; supervisor approval pending. **On hold (2026-10-23):** the corpus choice itself (all four corpora with general PubMed and PMC samples; textbooks and guidelines only; those plus a general PubMed/PMC sample; or the Alzheimer's-scoped PubMed and PMC already built) waits for the supervisor. The PubMed grouping (10/10/10/8 chunks as in the paper, or one group) waits with it |
| D9 | Boundary of "neighbours" of Alzheimer's disease | **Open**, and only relevant if the Alzheimer's-scoped corpora are kept (D8) |
| D10 | Evaluation dataset | **Adopted** (2026-10-23): the paper's three datasets in the roles of D12 |
| D11 | Arithmetic of the article encoder for the PubMed and PMC passages: 32-bit as for the textbooks and guidelines, or 16-bit | **Adopted default** (2026-10-18), supervisor approval pending: on 1,000 guideline passages on the researcher's RTX 2050, 16-bit ran at 105 passages a second against 22 in 32-bit (the 32-bit embedding of the guidelines ran at 27 to 34), the cosine between the two vectors of a passage was 0.999999 on average and 0.999998 at the lowest, the same 10 nearest passages were found 99.8% of the time and the same nearest passage 100%. The vectors are still stored as 32-bit numbers; the setting is recorded in each corpus |
| D12 | Datasets and roles | **Adopted** (2026-10-23): section 7.7. MedQA as in the paper; MedMCQA train and validation only (no test result, option A); MMLU-Med test only. No additional dataset for now |
| D13 | k | **Adopted**: {1, 2, 4, 8, 16, 32}, chosen on validation; MMLU-Med takes the k chosen on MedMCQA validation (section 7.6) |
| D14 | How the evidence enters the answer prompt | **[Working default]**: the kept snippets in an "Evidence" section before the question, the paper's printed prompt wording otherwise unchanged, the same layout for every condition. The answer without retrieval is the final option of the rationale itself, since the rationale prompt already asks for one [Paper p.4, Figure 2, shows the model's output as answer and rationale; **Inferred**]. The paper prints no such template (searched: RAG² code, Self-BioRAG, MedRAG; the last two belong to other methods) |
| D15 | Generation cap | **[Working default]**: 768 new tokens, fixed before any test run and checked on validation (the share of generations cut off should be near zero; the 2026-09 pilot cut about a quarter at 256 tokens and 0 to 1.25% at 768, git history `46ac455^:docs/log.md`, Phases 17 and 18). Both sides of a label comparison (without and with the snippet) use the same cap. **[Open]** Part 1 stops at 512 new tokens (the encoder's reading limit, section 3); before labels are made, either Part 1's cap is raised to the answer cap or a cut-off generation counts as unreadable (D16) |
| D16 | Reading the final answer | **[Working default]**: a generation is readable when it names exactly one option letter as its final answer, in the form the paper's case study shows ("Therefore, the answer is (D) …", Figure 4, p.8). Anything else is unreadable: wrong when scoring, the pair skipped when labelling. The unreadable rate is measured on validation and the rule is widened, if needed, before any test run, never after. The authors' parsing is not stated (U) |
| D17 | Perplexity (Eq. 4) | **[Working default]**: the rationale written without evidence, scored twice by the same model, without and with the snippet. Same text, same length L, as Eq. 4 has it. The alternative reading, the question text, is kept as a possible sensitivity check |
| D18 | The pool of the top 25% (τ) | **[Working default]**: all labelled pairs from the training questions of one dataset, taken before discarding, one τ per filter; a difference equal to τ counts (section 5) |
| D19 | Snippets labelled per question | **[Working default]**: the first 4 of the reranked list. The paper does not say; more snippets give more labels at one generation each |
| D20 | The query that retrieves the snippets of the training questions | **[Working default]**: the rationale, as at inference, so that the filter sees the kind of snippet it will judge. The released code also has a training-set mode whose query is the instruction and input text joined (`--instruction_preprocess`, `query_encode.py`), and the paper does not say which was used for the labelled pairs. **[Open]** |
| D21 | The re-ranker's query | **Adopted**: the original question, as the paper says; the code's rationale variant is a labelled sensitivity check (section 4) |
| D22 | Filter model | **Adopted**: Flan-T5-base first (declared stand-in), Flan-T5-large when a 24 GB GPU is available; the paper's settings; no other method added (section 5). The one-batch memory test is not yet run |
| D23 | Label scale | **Adopted**: MedQA, the full training set, in resumable stages; MedMCQA, a staged subset with the final size reported (section 5) |
| D24 | Test discipline and analysis | **Adopted**: sections 7.7 (item 6) and 7.9 |
| D25 | Extensions and extra datasets | **Adopted:** deferred until the baseline is finished: the control test with an irrelevant passage, set-conditioned labels, further evaluation datasets, further filter sizes |
| D26 | Time | No deadline is applied in the baseline plan; the schedule is for the supervisor. The compute figures are estimates until the timing pilot has run |

## 10. Questions for the authors

Status: the researcher emailed the authors with items 1 to 3, 5 and parts of 6 and 7 below; no reply by 2026-10-23. The items marked
"new" were found afterwards and were not in that email.

1. The full labelled training file and the validation file (only five examples are released).
2. The label-building script (`classifier/data/preprocess.py` is empty), the exact perplexity definition, and the set of pairs the
   top 25% is taken over.
3. What "5%" means, how many snippets per question were labelled, and which.
4. Whether one filter was trained per backbone model.
5. The Flan-T5 size, and whether a checkpoint still exists.
6. The answer-generation prompt, the maximum rationale length, and whether a system prompt or one or two line breaks before
   "Here is the question:" were used.
7. PubMed grouping at retrieval, whether sentence separators were used when embedding, and whether the re-ranker scored the
   question or the rationale.
8. Which MedMCQA split is reported (Table 1 lists 6,150 test questions, whose answers are not public; Self-BioRAG's test file has
   4,183 rows tagged `MedMCQA_dev`).
9. Whether the guideline and textbook corpora are Self-BioRAG's `CPG_128` and `Textbook_128`, and the terms for reuse.
10. New. How final answers were read (D16) and the generation cap for the answers (D15).
11. New. How many snippets per question were labelled, and which query retrieved them for the training questions: the rationale or
    the question (D19, D20).
12. New. Whether the curves of Figures 3 and A1 show validation or test accuracy, and how k was chosen for MMLU-Med, which has no
    validation set (D13).
13. New. Whether one filter was trained on MedQA and MedMCQA together or one per dataset (section 5).
14. New. What the "5%" file holds: how many pairs, from which datasets, and what it is a fraction of.

Resolved by reading the paper and code: the rationale prompt (printed, p.4), the candidates per corpus (equal to k), the final k per
dataset (section 2) and the input format of the filter.

**Open for the supervisor** (the researcher will take these to the supervisor; none is decided here):

- The corpus (D8, D9): which of the four options, the scale, the source of any general PubMed or PMC sample, the licences of the
  textbooks and the guidelines, and what may be published (identifiers and hashes, never the corpora).
- Compute: a 24 GB GPU for training Flan-T5-large and, if possible, a faster way to run the labelling; if there is none,
  whether Flan-T5-base may stay as the filter, as a declared deviation.
- Whether a smaller-scale, method-faithful reproduction with the declared deviations of section 7.8 is acceptable, and that success
  means directional agreement with the paper (D6), not equal numbers.
- Whether MedQA as the only paper dataset with a public test answer key, plus MMLU-Med, suffices; whether a MedMCQA blind
  submission or a split of its validation set is wanted (D12).
- The k of MMLU-Med (D13) and the working defaults D14 to D20; how long to wait for the authors before relying on them.
- The labelled-set size for MedMCQA, the schedule that follows from the compute figures, and any ethics or registration
  requirement of the department (only public exam questions and published literature are used; no patient data).

## 11. Acceptance tests

The requirements above that can be checked without models are written as tests in `tests/spec/test_*.py`. They fix behaviour, not
code: each test calls a function that `tests/spec/targets.py` maps to a module under `src/`; that file also lists the arguments of
every function. The names of the parts that are not built yet are provisional; those of `rationale_query` are final. When one
changes, change `tests/spec/targets.py` and this table together; a guard test fails if they differ.

| Requirement | Module and function | Test file |
|---|---|---|
| Label rule (section 5): six cases, a changed answer outranks perplexity, a difference equal to τ counts as lower, bad input is an error | `rationale_filter.labels`: `label_pair`, `to_training_token` | `test_labels.py` |
| τ as the top fraction of the perplexity differentials | `rationale_filter.labels`: `perplexity_threshold` | `test_labels.py` |
| Filter input format, decision rule (a tie is helpful), judging one snippet at a time, keeping the helpful snippets in order | `rationale_filter.inference`: `build_filter_input`, `decide_helpful`, `judge_snippets`, `keep_helpful` | `test_filter.py` |
| The paper's rationale prompt | `rationale_query.prompt`: `build_rationale_prompt` | `test_query.py` |
| The rationale alone as the dense query; the re-ranking query (the question by default, the rationale as a labelled variant); an empty rationale is an error | `rationale_query.search_text`: `dense_query`, `rerank_query` | `test_query.py` |
| The same number from each corpus, fixed order, no deduplication, no borrowing; stable re-ranking without a quota | `balanced_retrieval.pool`: `pool_candidates`, `rerank` | `test_retrieval.py` |
| The wiring: the rationale alone to every search, one k for every search and for the final list, one pooled re-ranking | `balanced_retrieval.retrieve`: `balanced_retrieve` | `test_retrieval.py` |
| Share per corpus and cross-corpus duplicate rate of a final list | `balanced_retrieval.diagnostics`: `corpus_share`, `duplicate_rate` | `test_retrieval.py` |
| Split by question (independent of row order, process and other questions); disjoint roles | `data_roles.splits`: `split_by_question`, `assert_disjoint` | `test_data_roles.py` |
| The tests themselves: the target map, the skip rule and the link to this document | none | `test_guard.py` |

**Tests of the built parts.** `src/rationale_query` and `src/balanced_retrieval` have their own tests, which check the code rather than the
requirements. They sit in `tests/unit/<package>/` (with stand-ins, no model needed) and `tests/integration/` (real libraries, real
files); `tests/README.md` explains the folders. For `rationale_query`:
`test_generator.py` (prompt tokens, greedy settings, limits, template checks, with a stand-in for llama.cpp),
`test_prompt.py` (the exact layout of the prompt, which the acceptance tests squeeze), `test_store.py`
and `test_batch.py` (records, resuming, refusals), `test_encoder.py`, `test_diagnostics.py`,
`test_cli.py`, `test_rationale_writers.py` (the settings and runner of the experiment). For `balanced_retrieval`: `test_pool.py` and
`test_retrieve.py` (argument handling and the order of calls, beyond the acceptance tests), `test_diagnostics.py`, `test_chunking.py`
(the windows), `test_corpus.py` (the corpus folder), `test_embedding.py` (the resumable embedding), `test_search.py` (the exact
search against a brute-force check), `test_scorer.py`, `test_records.py`, `test_sources.py` (the four downloads, with a stand-in for NCBI's service and for the Hugging Face hub) and `test_cli.py` (the commands end to end with stand-in
models). For `data_roles`: `test_splits.py` (the hash rule pinned, the order of the role names, the messages), `test_questions.py` (the three readers
and what they refuse, the inline form of a question), `test_overlap.py` (exact and near repeats between splits) and `test_cli.py` (the
commands end to end). For `answer_evaluation`: `test_prompt.py` (the layout of the answer prompt), `test_reader.py` (the strict reading of the
final answer), `test_paired_tests.py` (exact McNemar, the bootstrap, Holm, with exact values and the archived tests), `test_ledger.py`
(a test set is scored once per system), `test_records.py` (the record and its store), `test_runner.py` (every system on every question,
resuming, the refusals and the ledger on a test split) and `test_results.py` (accuracy per system, the paired comparison, k chosen on validation). For `rationale_filter`:
`test_labels.py` and `test_inference.py` (types, blank parts and messages beyond the acceptance tests, which share the first name; the unit
files are in `tests/unit/rationale_filter/`), `test_perplexity.py` (the formula and the pairing of the two prompts), `test_scorer.py` (the
llama.cpp scorer against a stand-in whose outputs the test fixes, so the alignment of the scores is checked), `test_label_records.py`
(the record, the store, τ over the pool), `test_labelling.py` (one pair and the resumable loop), `test_training_data.py` (the released format,
the matching of labels to pairs, refusing to overwrite) and `test_filter_eval.py` (accuracy per class, the majority rate, the AUC);
`tests/integration/test_real_scoring.py` runs the scorer against the real llama.cpp on the tiny synthetic model. `tests/repository/test_layout.py` checks every package (every module has a docstring and a test, importing the commands loads no heavy
package, and this section names every test file). `test_real_libraries.py` runs the real llama.cpp and a real torch encoder on tiny
synthetic models written by `tests/integration/tiny_models.py`; `test_real_retrieval.py` runs the article encoder, the cross-encoder and
the whole `balanced_retrieval` command chain on tiny synthetic BERT models, and compares them with the authors' own lines of code. They need
llama-cpp-python and gguf, or torch and transformers, and skip a class whose packages are missing. `test_real_models.py` checks the real
Llama-3 file and the MedCPT models; it runs only when `RATIONALE_LLAMA3_GGUF`, `RATIONALE_MEDCPT` or `RETRIEVAL_MEDCPT` is set (see
`experiments/01_rationale_writers/NOTES.md`).

**Skipping.** A test is skipped only while its part has not been started, that is, while no Python file exists under
`src/<package>/` for its module. Once the folder holds code, a missing module, a missing function or a failing check is an error,
never a skip. **Strict run.** `SPEC_STRICT=1` (PowerShell: `$env:SPEC_STRICT = "1"`) turns every skip into a failure. Baseline
acceptance requires a strict run in which every acceptance test passes. `python -m unittest discover -s tests -t . -v` lists the
skipped tests with the name of the function each one waits for.

**Checks outside the unit tests.** They need models or a numerical library and are run when those exist: encoding at 512 tokens
against the authors' `query_encode.py` output on fixed texts (cosine at least 0.999); the exact-search check (the index returns the
same ids as a brute-force inner product on a small random matrix); the share of rationales cut by the length limit; and the
held-out label gates (accuracy and AUC above the majority rate and above a relevance-score baseline, margins fixed beforehand).
A **control test of the labels** (the label signal for a retrieved passage compared with that of an irrelevant passage, with
the rule and sample size fixed in advance) was part of this list. It is **deferred** (D25): it is not part of the baseline and
runs, if at all, after the baseline is finished, on non-test questions only.

## 12. Status of each part

| Part | Faithful (F), adapted (A) or substitute (S) | State |
|---|---|---|
| 1 Rationale query | **F**, with the allowed deviation of a 4-bit generator and two open items (the line break before the question; the length limit) | built; the tests above pass, including the real llama.cpp and a real encoder on tiny stand-in models; the checks on the real Llama-3 file and MedCPT are written and wait for the researcher's laptop |
| 2 Balanced retrieval | **F** in the arithmetic, the pooling and the re-ranking; **A** in the scope and size of PubMed and PMC and in the block-wise search; two open items (how the authors embedded snippets; question or rationale for the re-ranker) | built and run on the real corpora (2026-10-20); the acceptance and unit tests pass, including real torch on tiny stand-in models; the real MedCPT checks pass on the researcher's laptop. The four corpora are built and embedded: textbooks 133,903, guidelines 721,364, PubMed 1,048,208 and PMC 8,205,229 passages. A trial of two questions at k = 32 gave a pool of 128 and a list of 32 each, with the final list shared between PubMed 0.344, PMC 0.328, guidelines 0.219 and textbooks 0.109 and a cross-corpus duplicate rate of 0.062; the best passages were on the topic of each question (a trial of two, not a measurement) |
| 3 Rationale-guided filter | to be set when built; the plan is D17 to D20 for the labels and D22 and D23 for the filter | not built |
| 4 Answer stage and evaluation runner | to be set when built; the plan is D13 to D16 and section 7.9 | not built. Not yet run on the laptop: the 20-question timing pilot, the one-batch filter memory test |

A part is called "RAG²" only if it is F, or A with its deviation written down. An S part is always named a substitute.

Acceptance tests (section 11): written for the three parts and for the data roles. Before they were committed they were run
against a throwaway reference implementation, which they accept, and against planted faults (a wrong tie rule, a dropped discard
branch, a changed order, a removed check, a split that depends on row order), which they reject. That implementation is not part
of the repository. The tests of Parts 1 and 2 now run against `src/rationale_query` and `src/balanced_retrieval` and pass; the others skip until their part is started.

## Revision note (2026-10-23)

The researcher accepted the plan for the baseline. Added or changed: the status lines; the marks `[Working default]` and
`[Researcher]` and the sources read for the open points (section 1); the answer prompt row (section 3); the re-ranker input
(section 4); the open items of the label rule replaced by working defaults and a description of how the filter is built (section
5); the MedMCQA split note (section 6); the k grid and the MMLU-Med rule (section 7.6); the dataset roles, the MedMCQA facts
checked by the researcher and the test discipline (sections 7.7); the declared deviations and the analysis (sections 7.8 and 7.9);
decisions D2, D5, D7 to D10 updated and D12 to D26 added (section 9); questions for the authors 10 to 14 and the open points for
the supervisor (section 10); the control test of the labels deferred (section 11); the state of Parts 3 and 4 (section 12).
Nothing was built. The corpus choice (D8, D9) is on hold for the supervisor; the extension and further datasets are deferred (D25).

## Revision note (2026-10-15)

Changes against the first version of this document: candidates per corpus resolved (one `top_k` for both); best k per dataset
added; the training file identified as the "5%" split; the exact filter input format and the tie rule added; the corpus ablation
and Table 5 added; the filter's gain stated per dataset; the Alzheimer's-scoped adaptation, fidelity checklist, data-role rules and
acceptance-test map added; decisions D7 to D10 added.

Added with the acceptance tests: the rules for pooling (no borrowing; unknown corpus or failed search is an error), for the label
inputs, for the document key and for the stability of the split; section 11 rewritten (skip rule, strict run, checks outside the
unit tests); the tests validated as described in section 12.

Part 2 built (2026-10-17): the implementation table in section 4, decision D8 adopted as a default, the tests of the built parts in section 11 and the state
of Part 2 in section 12.

Part 1 built (2026-10-15): the implementation table in section 3, the unit-level facts in section 7.3 from the researcher's own
checks, the `[SEP]` row made precise, the tests of the built part in section 11 and the state of Part 1 in section 12.
