# Methodology

Status: **specification of the RAG² baseline and of its Alzheimer's-scoped adaptation.** Revised 2026-10-15 after the researcher
approved the direction of the evidence report (restore the baseline faithfully first, then test one extension). Supervisor
approval (A2) is pending. Nothing is built yet. The requirements below are written as executable acceptance tests in
`tests/test_spec_*.py`; each test skips until the part it checks exists (section 11).

The evaluation dataset is **not decided here**; its approval is a separate step.

## 1. Sources and how facts are marked

- **Paper:** Sohn et al., *Rationale-Guided Retrieval Augmented Generation for Medical Question Answering*, NAACL 2025,
  arXiv 2411.00300v2 (22 Sep 2025). Page numbers refer to that PDF; the NAACL camera-ready may differ slightly.
- **Code:** `github.com/dmis-lab/RAG2`, commit `86add43` (retriever and filter-training code, five example labels). The same
  group's earlier project Self-BioRAG (`github.com/dmis-lab/self-biorag`) is the origin of the corpus and of the retriever.
- **MedCPT:** `github.com/ncbi/MedCPT` (public-domain notice); models on Hugging Face.
- Marks: **[Paper p.N]** stated in the paper; **[Code]** read in the released code; **[Figure]** read off a figure, so
  approximate; **[Inferred]** my reasoning from the facts around it; **[Open]** not stated anywhere read, or the sources
  disagree; **[Decision]** a choice this thesis must make; **[Rule]** an engineering rule of this project, not of the paper.

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
| Sentence separators | The code can insert `[SEP]` between sentences before encoding; the flag defaults to off in RAG² and on in Self-BioRAG. Whether the corpora were embedded with separators is not stated. | Code; **[Open]** |
| Decoding | Greedy, temperature 0, served with vLLM; no quantisation described. | Paper p.13 (A.3) |
| Maximum rationale length | Not stated. | **[Open]** |
| Answer-generation prompt | Not published; the README step 4 is prose only. | Code README; **[Open]** |
| Empty rationale | A failed rationale is an error and is reported as such; it is never silently replaced by the question, because that would change the method. | **[Rule]** |

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
  is not finite, or an answer that is neither correct nor wrong is an error, never a label. **[Open]** How an answer that cannot
  be read counts (as wrong, or the pair is dropped) is not stated; it is settled before the label function is called.
- **[Open]** Eq. 4 defines the perplexity over "the input query x", while the text and Figure 2 say the *rationale's*
  perplexity. The likely reading is the rationale's tokens given the question, with and without the snippet. Which set of
  pairs the 25% is taken over, and how a quarter is rounded, are not stated; the code of this project takes the pool as an
  explicit argument so that the choice is visible.
- **[Open]** Which snippet is labelled for a question, how many pairs per question, and what "5%" in the file name means.

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
182,822 / 4,183 / 6,150; MMLU-Med (six subjects) 1,089 test only.

Notes: plain retrieval (MedCPT, top-1) scores below no retrieval on MedQA and MedMCQA in the paper itself. The filter's gain
depends on the dataset: at each dataset's best k about +1.2 points on MedQA (between about −0.1 and +1.5 across k), roughly
+2 to +3 on MedMCQA and about +3.9 on MMLU-Med [Figure, p.7 and p.14]. Better rationales raise the Llama-3-8B backbone from 63.4
to 73.6 on MedQA, partly because the stronger model's own knowledge enters the query.

## 7. The Alzheimer's-scoped adaptation (declared deviations from the paper)

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

Every snippet carries: corpus, document key, chunk index, text, source date, licence group, corpus version. Only sources whose
licence a script can read are included; anything else is left out and stated as a limitation. Licences were **not** verified.

| Corpus | Source and keys | Version and maintenance | Counts on file |
|---|---|---|---|
| PubMed | Title, abstract, date, MeSH and publication types for records matching the archived query bank (MeSH and title/abstract terms), English, with an abstract, retractions removed. Keys: PMID, DOI. Obtained once through the E-utilities service or from NLM's baseline and update files. | A named snapshot; a new version only on purpose. The core query grows by about 17,000 to 20,000 records a year. | 255,907 records on 2026-09-13 (archived log) |
| PMC | Open-access full text for the PubMed set (PMC Open Access subset). Keys: PMCID, PMID, DOI; licence group kept per article; only the latest version of an article. | Add open-access articles whose PMIDs join the PubMed set; keep the version table. | 114,256 articles, 4,377,041 chunks of 256 tokens (32-token overlap) and a 12.5 GB vector file on the researcher's laptop |
| Guidelines | Dementia and neighbouring guidance, current versions only; candidate sources: the public Meditron guideline set and publishers' own files. Keys: DOI or URL with version, supersedes, superseded by. | Check yearly. | none yet; the first design's registry was empty |
| Textbooks | Dementia and neighbouring chapters; edition recorded. Keys: edition, chapter, chunk. | Static editions. | none yet; the first design's registry was empty |

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

The **deployment library** is the four Alzheimer's-scoped collections, searched when answering. The **label-building library** is a
reduced general textbook and guideline set (the closest match to exam questions), used only to create filter-training labels from
general training questions. If it cannot be assembled within the licences, labels are built from the Alzheimer's collections with
Alzheimer's-related questions only, as a declared small-scale variant.

### 7.6 Number of snippets and evaluation size [Decision D7]

The paper's best k (32 on MedQA) means 128 pooled candidates, 32 filter decisions (about 43 s per question on the researcher's CPU,
derived from 1.33 s each) and a long answering prompt. Time per question is measured in a calibration run before k and the
evaluation size are fixed. k is then chosen on validation data within the affordable range, and the limit is reported.

### 7.7 Data roles and leakage rules

| Role | Meaning |
|---|---|
| Model-training data | Rows (question id, snippet, label) **built** from training questions, a library and the generator. |
| Validation questions | Used to choose τ, k and checkpoints; never to learn. |
| Test questions | Used once at the end; never to train, tune or choose. |
| Retrieval library | Passages, not questions; separate from all question sets. |

1. Split by question: all rows of a question stay in one role. A question's role is a function of its id and a recorded seed
   only; it does not depend on row order, on the Python process, or on which other questions are present, so adding or removing
   questions never moves a question to another role.
2. Remove exact and near duplicates across roles, including across MedQA, MedMCQA and any Alzheimer's set (not yet done).
3. Fit and tune on training and validation data only; keep a manifest of ids and hashes made before any run.
4. Labels made by the generator under evaluation are part of the method, but the questions that produced them must be disjoint from
   every test question.
5. Pretraining contamination of the generator cannot be removed; compare systems that share one generator and say so.

### 7.8 Declared deviations expected

Scale (Alzheimer's-scoped collections instead of 116.7 M snippets); scope; possibly partial guideline and textbook collections;
a smaller label set or a different filter size or method if only the laptop is used; a 4-bit generator unless a GPU is
available; the answer prompt and rationale length if the authors' values cannot be had; pretraining contamination.

## 8. Fidelity

**Wording rule.** Call the baseline "a faithful re-implementation of RAG² with scale-reduced, Alzheimer's-scoped corpora", with a
deviation table. Use "reproduction" only if the authors' labels or checkpoint are obtained and used. No percentage of fidelity is
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
| D2 | Where Alzheimer's-related exam questions go (training or evaluation) | **Open**; depends on the evaluation dataset |
| D3 | The library used to make training labels | **Adopted default** (section 7.5); licence screen pending |
| D4 | Dated (as-of) evidence as a requirement | **Adopted:** not required; an optional later study |
| D5 | How to obtain the filter | **Adopted:** ask the authors (labels first) and prepare a scale-reduced build in parallel; request not yet sent |
| D6 | What "matches the paper" means | **Adopted:** directional agreement with section 6 |
| D7 | k and evaluation size | **Open** until the calibration run |
| D8 | PMC chunks: keep the existing 256-token chunks or re-chunk to 128-word windows | **Open** |
| D9 | Boundary of "neighbours" of Alzheimer's disease | **Open** |
| D10 | Evaluation dataset | **Not decided**; separate approval |

## 10. Questions for the authors

1. The full labelled training file and the validation file (only five examples are released).
2. The label-building script (`classifier/data/preprocess.py` is empty), the exact perplexity definition, and the set of pairs the
   top 25% is taken over.
3. What "5%" means, how many snippets per question were labelled, and which.
4. Whether one filter was trained per backbone model.
5. The Flan-T5 size, and whether a checkpoint still exists.
6. The answer-generation prompt and the maximum rationale length.
7. PubMed grouping at retrieval, whether sentence separators were used when embedding, and whether the re-ranker scored the
   question or the rationale.
8. Which MedMCQA split is reported (Table 1 lists 6,150 test questions; Self-BioRAG's test file has 4,183).
9. Whether the guideline and textbook corpora are Self-BioRAG's `CPG_128` and `Textbook_128`, and the terms for reuse.

Resolved by reading the paper and code: the rationale prompt (printed, p.4), the candidates per corpus (equal to k), the final k per
dataset (section 2) and the input format of the filter.

## 11. Acceptance tests

The requirements above that can be checked without models are written as tests in `tests/test_spec_*.py`. They fix behaviour, not
code: each test calls a function that `tests/spec_targets.py` maps to a module under `src/`; that file also lists the arguments of
every function. The names below are provisional. When one changes, change `tests/spec_targets.py` and this table together; a
guard test fails if they differ.

| Requirement | Module and function | Test file |
|---|---|---|
| Label rule (section 5): six cases, a changed answer outranks perplexity, a difference equal to τ counts as lower, bad input is an error | `rationale_filter.labels`: `label_pair`, `to_training_token` | `test_spec_labels.py` |
| τ as the top fraction of the perplexity differentials | `rationale_filter.labels`: `perplexity_threshold` | `test_spec_labels.py` |
| Filter input format, decision rule (a tie is helpful), judging one snippet at a time, keeping the helpful snippets in order | `rationale_filter.inference`: `build_filter_input`, `decide_helpful`, `judge_snippets`, `keep_helpful` | `test_spec_filter.py` |
| The paper's rationale prompt | `rationale_query.prompt`: `build_rationale_prompt` | `test_spec_query.py` |
| The rationale alone as the dense query; the re-ranking query (the question by default, the rationale as a labelled variant); an empty rationale is an error | `rationale_query.query`: `dense_query`, `rerank_query` | `test_spec_query.py` |
| The same number from each corpus, fixed order, no deduplication, no borrowing; stable re-ranking without a quota | `balanced_retrieval.pool`: `pool_candidates`, `rerank` | `test_spec_retrieval.py` |
| The wiring: the rationale alone to every search, one k for every search and for the final list, one pooled re-ranking | `balanced_retrieval.retrieve`: `balanced_retrieve` | `test_spec_retrieval.py` |
| Share per corpus and cross-corpus duplicate rate of a final list | `balanced_retrieval.diagnostics`: `corpus_share`, `duplicate_rate` | `test_spec_retrieval.py` |
| Split by question (independent of row order, process and other questions); disjoint roles | `data_roles.splits`: `split_by_question`, `assert_disjoint` | `test_spec_data_roles.py` |
| The tests themselves: the target map, the skip rule and the link to this document | none | `test_spec_guard.py` |

**Skipping.** A test is skipped only while its part has not been started, that is, while no Python file exists under
`src/<package>/` for its module. Once the folder holds code, a missing module, a missing function or a failing check is an error,
never a skip. **Strict run.** `SPEC_STRICT=1` (PowerShell: `$env:SPEC_STRICT = "1"`) turns every skip into a failure. Baseline
acceptance requires a strict run in which every acceptance test passes. `python -m unittest discover -s tests -t . -v` lists the
skipped tests with the name of the function each one waits for.

**Checks outside the unit tests.** They need models or a numerical library and are run when those exist: encoding at 512 tokens
against the authors' `query_encode.py` output on fixed texts (cosine at least 0.999); the exact-search check (the index returns the
same ids as a brute-force inner product on a small random matrix); the share of rationales cut by the length limit; and the
held-out label gates (accuracy and AUC above the majority rate and above a relevance-score baseline, margins fixed beforehand).
Before any training, a **control test of the labels**: the label signal for a retrieved passage must differ from that of an
irrelevant passage, with the rule and sample size fixed in advance.

## 12. Status of each part

| Part | Faithful (F), adapted (A) or substitute (S) | State |
|---|---|---|
| 1 Rationale query | to be set when built | not built |
| 2 Balanced retrieval | to be set when built | not built |
| 3 Rationale-guided filter | to be set when built | not built |

A part is called "RAG²" only if it is F, or A with its deviation written down. An S part is always named a substitute.

Acceptance tests (section 11): written for the three parts and for the data roles. Before they were committed they were run
against a throwaway reference implementation, which they accept, and against planted faults (a wrong tie rule, a dropped discard
branch, a changed order, a removed check, a split that depends on row order), which they reject. That implementation is not part
of the repository, so every test skips until its part is started.

## Revision note (2026-10-15)

Changes against the first version of this document: candidates per corpus resolved (one `top_k` for both); best k per dataset
added; the training file identified as the "5%" split; the exact filter input format and the tie rule added; the corpus ablation
and Table 5 added; the filter's gain stated per dataset; the Alzheimer's-scoped adaptation, fidelity checklist, data-role rules and
acceptance-test map added; decisions D7 to D10 added.

Added with the acceptance tests: the rules for pooling (no borrowing; unknown corpus or failed search is an error), for the label
inputs, for the document key and for the stability of the split; section 11 rewritten (skip rule, strict run, checks outside the
unit tests); the tests validated as described in section 12.
