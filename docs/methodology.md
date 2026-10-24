# The RAG² baseline model

This document explains the baseline model and its components, as the paper and the authors' released code describe them.
It states what each component does, with which model and settings. It does not describe this repository's code, plans or decisions.

**Sources.** Paper: Sohn et al., *Rationale-Guided Retrieval Augmented Generation for Medical Question Answering*, NAACL 2025
(<https://aclanthology.org/2025.naacl-long.635.pdf>). Code: <https://github.com/dmis-lab/RAG2>, commit `86add43`.
Each fact is marked **[Paper]** (stated in the paper), **[Code]** (read in the released code) or **[Figure]** (read off a figure, so approximate).
**[Not stated]** means neither source says it. The authors' repository is a research release: the four corpora and the trained
filter checkpoint are not included (its README says so), but the retriever code, the filter-training code and a five-example label sample are.

## 1. What the model does

RAG² answers a multiple-choice medical question in four stages. Each stage's output is the next stage's input.

```
question
  -> [1] rationale: the LLM writes a step-by-step explanation (chain of thought); the rationale is the search query
  -> [2] balanced retrieval: the same number of snippets from each of four corpora, pooled, then re-ranked; the best k are kept
  -> [3] filter: a small trained model judges each kept snippet, [HELPFUL] or [NOT_HELPFUL]; only helpful snippets stay
  -> [4] answer: the LLM answers the question with the helpful snippets in its prompt
```

Only stage 3 is trained. Every other model is used as released [Paper §3, §4.2].

| Stage | Model |
|---|---|
| Rationale and answer | Llama-3-8B-Instruct, used as is, no fine-tuning [Paper §4.2]. The same LLM writes the rationale and answers [Paper §3.3]. |
| Query encoder, article encoder, re-ranker | MedCPT: `ncbi/MedCPT-Query-Encoder`, `ncbi/MedCPT-Article-Encoder`, `ncbi/MedCPT-Cross-Encoder` [Paper §3.4, A.3; Code]. |
| Filter | Flan-T5-large (770 million parameters), fine-tuned [Paper §3.2, §4.2]. |

The paper also applies the method to Meerkat-7B and GPT-4o. The baseline here is the Llama-3-8B-Instruct system.

## 2. Component 1: rationale-based query

- The LLM gets this prompt, with the question in place of `[initial_query]`: "The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, starting by summarizing the available information. Output your explanation and single option from the given options as the final answer. Here is the question: [initial_query]" [Paper §3.3]. The exact line breaks are not recoverable from the PDF.
- The search query is the rationale alone. The question is not added, because both together exceed the retriever's maximum length and gave worse results in the authors' tests [Paper §3.3].
- The query encoder is `ncbi/MedCPT-Query-Encoder`. The query vector is the first-token vector (`last_hidden_state[:, 0, :]`), with truncation at 512 tokens [Code `query_encode.py`].
- The code adds `[SEP]` between sentences only for the instruction-set (training) queries, and only if `--use_spacy` is set. A rationale query is encoded as written [Code `main.py`].
- Generation is greedy (temperature 0), run with vLLM on one H100 [Paper A.3].
- **[Not stated]:** the maximum rationale length, and the exact system prompt and chat template.

## 3. Component 2: balanced retrieval

**Corpora** [Paper A.3, Table A1]: PubMed abstracts, PMC full texts, clinical practice guidelines (CPG, the Meditron collection, 8 of 16 sources public), and 18 medical textbooks. This is the corpus of Self-BioRAG.

| Corpus | Documents | Snippets | Index size |
|---|---|---|---|
| PubMed | 36.5M | 69.7M | 400 GB |
| PMC | 1.1M | 46.3M | 160 GB |
| CPG | 35.7k | 607.0k | 3.5 GB |
| Textbooks | 18 | 134.0k | 0.7 GB |
| Total | 37.6M | 116.7M | 564.2 GB |

Documents are cut into snippets with a sliding window with overlap [Paper A.3]. **[Not stated]** in the paper: the window size and overlap.

**Search** [Code `retrieve.py`, `main.py`]:
- Every snippet has a 768-number vector from `ncbi/MedCPT-Article-Encoder`.
- Each corpus has an exact inner-product index (`faiss.IndexFlatIP`). PubMed is stored in 38 files, searched in groups; the default groups all 38 into one. PMC has two files (main text and abstracts); CPG and textbooks have one each.
- One parameter, `top_k`, sets both the number of candidates taken from each corpus and the number of snippets kept after re-ranking.

**Balanced pooling** [Paper §3.4; Code `rerank.py`]:
- The top `top_k` candidates of each corpus are taken. The four lists are joined in the order PubMed, PMC, CPG, textbooks. Duplicates are not removed. The pool therefore holds up to 4 × `top_k` candidates.
- Only the candidate step is balanced. Nothing keeps the final list balanced.

**Re-ranking** [Paper §3.4; Code `rerank.py`]:
- `ncbi/MedCPT-Cross-Encoder` scores each (query, snippet) pair, with truncation at 512 tokens. Snippets are sorted from the highest score down, and the best `top_k` are kept.
- **The paper and the code differ on the query.** The paper says the cross-encoder reads the original question with each snippet. The released code passes the same retrieval text as the dense search, which is the rationale.

**Corpus ablation** (MedQA, Llama-3-8B-Instruct, best accuracy over k) [Figure A3]: PubMed alone about 59.8, PMC about 58.4, CPG about 58, textbooks about 55.5; stacked retrieval about 59.5, stacked with re-ranking about 60.9, balanced retrieval with re-ranking 63.4, balanced with the filter 64.6. The authors conclude that no single corpus held enough information [Paper A.5].

## 4. Component 3: rationale-guided filter

### 4.1 Training labels

Each label belongs to one (question, snippet) pair. The model answers the question twice, without retrieval and with the snippet, and the rationale's perplexity is compared in the two settings [Paper §3.2, Figure 2].

| Answer without snippet | Answer with snippet | Rationale perplexity lower with the snippet? | Label |
|---|---|---|---|
| correct | correct | yes | helpful |
| correct | correct | no | discarded |
| correct | wrong | not asked | not helpful |
| wrong | correct | not asked | helpful |
| wrong | wrong | yes | not helpful |
| wrong | wrong | no | discarded |

- Perplexity (Eq. 4): PPL(x) = exp(−(1/L) Σ log P(xᵢ | x<ᵢ)) and PPL(x, d) = exp(−(1/L) Σ log P(xᵢ | x<ᵢ, d)), where d is the snippet.
- The perplexity difference is ΔPPL = PPL(x) − PPL(x, d) ≥ τ (Eq. 3). "Lower perplexity" in Figure 2 means ΔPPL ≥ τ.
- τ is set to the top 25% of the perplexity differences. It was chosen on validation data and kept fixed for all experiments [Paper §3.2].
- Each snippet is labelled on its own. The authors name this as a limitation [Paper, Limitations].
- **[Not stated]:** whether x in Eq. 4 is the question or the rationale (the text and Figure 2 say the rationale; Eq. 4 says "the input query"), which snippets are labelled per question, how many, over which pool of pairs the 25% is taken, and how an unreadable answer counts.

### 4.2 The filter model and its training

**Input** (identical in all five released examples) [Code, `5%-train.json`]:

```
Given the following evidence, determine whether it helps answer the provided question.

Evidence: <snippet>

Question: <question with the four options inline>
```

The target is the single token `[HELPFUL]` or `[NOT_HELPFUL]`. The rationale is not an input. The two tokens are added to the tokenizer, and the embedding matrix is resized to the new tokenizer length [Code `classifier/README.md`, `token_add.ipynb`].

**Training** [Paper A.3; Code `run_classifier.py`, launch script]:
- Flan-T5-large, 770 million parameters. It can be trained on one RTX 3090 with 24 GB [Paper §4.2]. The authors trained on one H100 with 80 GB.
- AdamW, learning rate 3e-5, weight decay 0, linear schedule, no warmup, per-device batch size 16, 40 epochs, shuffled batches.
- Maximum input 512 tokens. A longer input becomes several overlapping windows with stride 128, each with the same label.
- Labels: the target tokens, with padding set to −100 so the loss ignores it. Maximum target length 30.
- A few candidate checkpoints are chosen on the validation set and then evaluated on the test set [Paper A.3].
- Training data: question–answer pairs from the MedQA and MedMCQA training sets. MMLU-Med has no training set, so the filter trained on MedMCQA is used for it [Paper §4.2, A.1]. The released sample has five examples. Its ids run up to 23,600, and its file name is `5%-train.json`; the meaning of "5%" is not explained.

**Decision at answer time** [Code `run_classifier.py`]:
- The model generates once. At the first generated position the logits of `[HELPFUL]` and `[NOT_HELPFUL]` are compared with a softmax. The larger wins. On an exact tie the authors' `argmax` picks index 0, which is `[HELPFUL]`.
- Each snippet is judged alone, from its own filter input. Only helpful snippets go to the LLM, in their original order [Paper §3.2, Limitations].
- The evaluation code builds its features with the same overflowing windows as training, makes one prediction per window, and asserts that the number of predictions equals the number of examples. An input longer than 512 tokens would therefore stop the evaluation. How such an input is judged is **[Not stated]**.

## 5. Component 4: answer generation

- The LLM receives the question and the helpful snippets in its prompt and answers the question [Paper §3, Figure 1]. In the case study the answer ends "Therefore, the answer is (A) …" [Paper Figure 4].
- Decoding is greedy, temperature 0, with vLLM [Paper A.3]. The paper notes that small run-to-run differences can remain because of hardware and software.
- **[Not stated]:** the answer prompt. The authors' README step 4 is prose only, and a search of the released code found no answer template.

## 6. Data and evaluation

| Dataset | Training | Validation | Test | Use |
|---|---|---|---|---|
| MedQA (USMLE, four options) | 10,178 | 1,272 | 1,273 | filter training; evaluation |
| MedMCQA | 182,822 | 4,183 | 6,150 | filter training; evaluation |
| MMLU-Med (six medical subjects) | none | none | 1,089 | evaluation only |

[Paper Table 1]. All three are closed-book multiple choice with four options [Paper §4.1]. The metric is accuracy [Paper §4.3].

**Number of snippets k.** k takes the values 1, 2, 4, 8, 16 and 32. It is chosen on validation data for each dataset [Paper §4.2, Figure 3, Figure A1]. For Llama-3-8B-Instruct the best k read off the figures is 32 on MedQA, 16 on MedMCQA and 8 on MMLU-Med [Figure]. The comparison systems MedCPT and MedCPT+Rationale use k = 1; Adaptive-RAG uses k = 1 and InstructRAG-ICL uses k = 5 [Paper §4.2].

**Reference results** (Llama-3-8B-Instruct, accuracy, [Paper Table 2]):

| System | MedQA | MedMCQA | MMLU-Med | Average |
|---|---|---|---|---|
| No retrieval (0-shot) | 57.7 | 53.5 | 69.5 | 60.2 |
| MedCPT, k = 1 | 55.3 | 51.3 | 65.8 | 57.5 |
| MedCPT + Rationale, k = 1 | 58.0 | 52.1 | 70.3 | 60.1 |
| MedRAG | 56.4 | 56.6 | 69.2 | 60.7 |
| RAG² | 64.6 | 59.4 | 74.8 | 66.3 |

Filter ablation, Llama-3-8B-Instruct, top-1 snippet, original question as the query [Paper Table 3]: MedCPT without filtering 55.3 (MedQA) and 51.3 (MedMCQA); Adaptive-RAG 57.3 and 53.1; GPT-4o as filter 58.5 and 55.8; the rationale-guided filter 58.6 and 55.8.

## 7. What the paper and the code do not state

These points are not specified by either source. They are listed so that nothing is assumed silently:

- the maximum length of the rationale, and the answer prompt;
- the window size and overlap used to cut documents into snippets;
- how many PubMed groups the authors used (the code default is one; the README of the code mentions four);
- whether the re-ranker read the question or the rationale (the paper and the code differ, see section 3);
- which snippets and how many per question were labelled, the pool over which τ is taken, and what "5%" in the sample's file name means;
- whether the perplexity in Eq. 4 is that of the rationale or of the question;
- how an unreadable answer counts when labelling or scoring;
- how a filter input longer than 512 tokens is judged;
- whether one filter was trained on MedQA and MedMCQA together or one per dataset.
