# Evaluation guide: what each metric measures and how to read it

| | |
|---|---|
| Document status | Current as of 2026-11-24. Implemented in `src/evaluation/` (run through `python -m src.main`). |
| Audience | The researcher, the supervisor, and readers of the thesis (plain English; scientific accuracy kept). |
| Related documents | `results/results.md` (the metrics applied), `data/datasheet.md` (the data) |

Scores are written as percentages here and in the screen output (25.4% = 61 of 240 right). The JSON and CSV files keep fractions (0.2542) so that programs can read them; p-values and the semantic-similarity cosine are not percentages.

What each number means, how it is calculated, and what it can and cannot tell you. Code: `src/evaluation/` (`statistics.py`, `diagnostic.py`, `retrieval.py`, `hallucination.py`, `evaluator.py`).
Marks: ✔ = checked in this repository (a test, or a recomputation by hand); ◻ = from the literature, check the paper before citing it.
"Research question" means the author's reading of the objectives in `README.md` ("Objectives"); the researcher is asked to confirm or correct it.

## 1. The experiment in one page

- **Data:** the Kaggle "Skin Cancer ISIC 9 classes" images, 8 classes (MEL, NV, BCC, AK, BKL, DF, VASC, SCC), 2,197 cases. Splits by lesion: train 1,797, validation 160 (20 per class), test 240 (30 per class). No lesion is in two splits ✔.
- **Three views of the same cases:** image only, text only (age, sex, body site), image + text. Every question is multiple choice with 8 options in a fixed shuffled order.
- **Four systems:** (1) LLM-only, (2) standard RAG, (3) the RAG² paper's method, (4) the proposed RAG. Only (1) is built so far.
- **Four models, used as they are:** claude-sonnet-5-5, gpt-4o, gemini-3.8-flash, llama-4-maverick.
- **Rule:** everything is scored by the computer. No clinician reads a case.

## 2. Four groups of metrics (kept apart on purpose)

| Group | Question | Used by |
|---|---|---|
| A. Diagnosis | Is the final answer right? | every system |
| B. Retrieval | Did the system find the right passages? | RAG systems only |
| C. Faithfulness and hallucination | Does the answer rest on the evidence? | RAG systems (and the text-only image-claim check for LLM-only) |
| D. Explanation quality and cost | Does the reasoning resemble medical text, and what did it cost? | every system |

### A. Diagnosis: is the final answer right?

| Metric | In words | Calculation | Read it as | Question it helps answer |
|---|---|---|---|---|
| Accuracy | Share of questions answered with the right class | right / all questions. A missing or invalid answer counts as wrong | 12.5% is chance (8 classes, balanced). Higher is better | Does retrieval improve the answer? |
| 95% interval | How much the number could move by chance | 1,000 resamples of whole lesions (photos of one lesion stay together); the 2.5th and 97.5th percentiles | If two intervals overlap a lot, the difference may be luck | How sure are we? |
| Macro-F1 | Accuracy that gives every class the same weight | F1 of each class = 2·TP / (2·TP + FP + FN); the mean over the 8 classes | Low macro-F1 with fair accuracy means some classes are ignored | Does the model fail on rare or hard classes? |
| Precision, recall, specificity, F1 per class | Per-class detail | precision = TP/(TP+FP); recall = TP/(TP+FN); specificity = TN/(TN+FP) | Recall of MEL is the share of melanomas found | Which diseases are confused? |
| Confusion matrix | Rows = true class, columns = answer | counts, plus a column for invalid answers | Off-diagonal cells show typical mistakes | Which pairs are confused (e.g. BCC and SCC)? |
| Invalid answers, refusals | Answers with no usable letter, and those where the model refused | counted, refusals apart | Many refusals mean the model or host blocks the task | Is the comparison fair? |
| Chosen letters | How often each letter A to H was picked | counts | One letter far above 12.5% means a position habit | Is there position bias? |
| McNemar exact test, Holm correction | Is system A different from system B on the same questions? | Only questions where exactly one system is right count; two-sided binomial test with p = 0.5; Holm multiplies the sorted p-values by (m − i + 1) to correct for many pairs | p below 0.05 after Holm: the difference is unlikely to be chance | Is RAG really better than LLM-only? |

Checked ✔: macro-F1, specificity and McNemar were recomputed from their definitions on 400 random answers and match; Holm matches a hand example.
Sources: McNemar (1947) Psychometrika 12:153-157; Holm (1979) Scandinavian Journal of Statistics 6:65-70; bootstrap: Efron & Tibshirani (1993) ◻.
Top-1 accuracy only: top-k accuracy, balanced accuracy and MCC were removed on the researcher's decision (on a balanced test set balanced accuracy equals accuracy).

### B. Retrieval: did the system find the right passages? (RAG systems only)

Weak label: a passage is "relevant" for a question when its tags contain the true class. The tags come from the corpus builder (the class a passage was searched for, plus class names found in its text). So these numbers measure whether the right **topic** was retrieved, not whether a passage proves the diagnosis.

| Metric | In words | Calculation | Question it helps answer |
|---|---|---|---|
| Hit@k | Was at least one relevant passage in the top k? | share of questions with a relevant passage in the top k | Does the retriever find anything useful? |
| Precision@k | How much of the top k is relevant? | relevant in the top k / k | Is the context clean or noisy? |
| Recall@k | How much of all relevant material was found? | relevant in the top k / all corpus passages tagged with the true class (257 to 1,123 per class) | Compare systems only: the number is small for every system because the corpus is large |
| MRR | How high is the first relevant passage? | mean of 1 / rank of the first relevant passage | Does the retriever put good passages first? |
| nDCG@k | Ranking quality with position discount | DCG / ideal DCG, DCG = sum of relevance / log2(position + 1) | Same as MRR over the whole top k |
| Context relevance (MedCPT) | A trained biomedical model scores how well each passage fits the query | mean cross-encoder score per question, then over questions (needs `--relevance-models`; not a probability) | Is the retrieved text on topic for this case? |

Sources: BEIR, Thakur et al. (2021) NeurIPS Datasets and Benchmarks ◻; nDCG: Järvelin & Kekäläinen (2002) ACM TOIS 20:422-446; MedCPT: Jin et al. (2023) Bioinformatics 39(11):btad651; "context relevance" as a name: RAGAS, Es et al. (EACL 2024) and ARES, Saad-Falcon et al. (NAACL 2024) ◻.
Checked ✔: nDCG@3, MRR, Precision@3, Hit@3, Recall@3 and context precision were recomputed by hand and match. The nDCG assumption (at least k relevant passages exist per class) holds ✔ because every class has 257 or more tagged passages.
"Context precision" was dropped on 2026-11-24 (it partly repeated Precision@k and MRR).

### C. Faithfulness and hallucination: does the answer rest on the evidence? (RAG systems)

The model must end with "Evidence: [passage number] "<sentence copied word for word>"" or "Evidence: none". (Built and tested with a fake model; not yet run on the real models.) A reply with no readable evidence line is counted apart (`evidence_unparsed`): it is neither evidence nor a rejection.

| Metric | In words | Calculation | What it does NOT prove |
|---|---|---|---|
| Coverage, selective risk | How often the system gives a verified answer (a valid diagnosis and a quote that is word for word in the cited passage: lower-cased, no punctuation, at least 5 words, "..." pieces in order), and how often those are wrong | coverage = verified answers / all; risk = wrong among verified (Geifman & El-Yaniv 2017) | A real quote is not proof that it supports the diagnosis |
| Negative rejection | Does it say "none" when nothing relevant was retrieved? | "Evidence: none" among questions where no retrieved passage is tagged with the true class (RGB, Chen et al. 2024) | Uses the true class to know when nothing relevant exists, so it is an evaluation-only number |
| Image claims in the text-only view | Does the reasoning describe a picture that was never given? | a word list (dermoscopy, "the image shows"…); a sentence with a denial or "if/would" is skipped | A lower bound: a paraphrase is missed. Not a published metric. Not used for image views |

An NLI model (a language model that decides whether a passage supports a statement) was tried to test "the passage supports the diagnosis". It failed its pre-set pass mark twice (83% pairwise; 50% top-1 over 8 classes, chance 12.5%), so it was removed ✔.
**Honest limit:** no published metric measures "this reasoning is false" without an expert. All hallucination numbers here are proxies.

### D. Explanation quality and cost

| Metric | In words | Calculation | Limit |
|---|---|---|---|
| Semantic similarity | Does the reasoning resemble the reference text of the true class? | cosine of sentence embeddings (model `pritamdeka/S-PubMedBert-MS-MARCO`) between the reasoning and each reference text of the true class; the best match is kept (40 texts quoted from Dermoscopedia, CC BY-NC-SA 4.0) | Measures resemblance, not truth. It follows accuracy in part (a right answer talks about the true class). An honest "cannot tell" scores low |
| Tokens and seconds | Cost and speed | sums and means from each call | Prices not included |

Sentence embeddings: Reimers & Gurevych (2019) EMNLP ◻.

**Validity check (approved 2026-11-24; run on the 12 LLM-only test runs).** The first results gave almost the same similarity for every model, which suggested the score might not discriminate. `python -m src.main similarity-check` tests this: the reasoning is compared with the reference texts of all 8 diseases, and it is "identified" when the disease the model answered has the single highest similarity (chance 12.5%). Pass mark, set before the first run (my proposal, not a published standard): identified in at least 50% of the answers overall and in at least 25% for every answered disease with 10 or more answers. **Result:** the answered disease is identified in 39.2% to 91.0% of answers in every run (far above chance), so the score carries information; the pass mark is met in 6 of 12 runs and fails where a class matches its reference texts poorly (BCC in text only; melanoma for Claude) or the overall rate is under 50%. Differences between models in the mean score are too small (0.006 or less in the image views) to rank models. Details: `results/results.md`.

## 3. How to read a result table

1. Start with **accuracy and its interval** for each system (same model, view and split).
2. Check **macro-F1** and the **confusion matrix**: a system that wins on accuracy by ignoring rare classes shows here.
3. Compare two systems with **McNemar + Holm**, never by eye.
4. For RAG, look at **retrieval** (B) first. If Hit@k is low, a poor answer is a retrieval problem; if Hit@k is high and the answer is still wrong, it is a reading problem.
5. Look at **evidence** (C) for the hallucination side: the share of answers whose quote is real, and what happens when nothing relevant was retrieved.
6. Report **refusals and invalid answers** next to accuracy.
7. State the limits: pinned host and snapshot per model; Claude has no fixed temperature; the corpus is not balanced across classes; a quote is not proof.

## 4. Where the numbers are collected

`python -m src.main collect` writes `report.json` next to the predictions of every finished run and one table, `results/comparisons/summary.csv`, with one row per system, model, view and split (accuracy and its interval, macro-F1, invalid answers, refusals, semantic similarity, the text-only image-claim rate, tokens, seconds, the host that answered, the question-file hash and the folder). Trials (`.partial` folders) are never collected.

## 5. Validation before test runs

Validation (160 questions) is for checking the format and the code, not for tuning to a good score. The test split (240 questions) is run once per system, model and view, after the validation checks pass and the settings are fixed. Nothing is changed after a test result is seen.
Small trials (`--limit 10 --answered-only`) check that the format parses; their accuracy is never reported.

## 6. Open points

- The exact **proposed system** and the **RAG² reproduction scope** (not yet decided).
- The researcher's confirmation of the research-question wording used in this guide.
- Whether to add reference texts for melanoma and BCC, where the validity check failed (section 2D).

## 7. References

✔ = confirmed in a search result seen during this project; ◻ = from memory, check the paper before citing.

- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions. *Psychometrika* 12:153-157 ◻.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6:65-70 ◻.
- Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall ◻.
- Järvelin, K., Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM TOIS* 20:422-446 ◻.
- Thakur, N. et al. (2021). BEIR: a heterogeneous benchmark for zero-shot evaluation of information retrieval models. *NeurIPS Datasets and Benchmarks* ◻.
- Jin, Q. et al. (2023). MedCPT: contrastive pre-trained transformers with large-scale PubMed search logs. *Bioinformatics* 39(11):btad651 ◻.
- Geifman, Y., El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS* ◻.
- Chen, J. et al. (2024). Benchmarking large language models in retrieval-augmented generation (RGB). *AAAI* 38:17754-17762 ✔.
- Gao, T. et al. (2023). Enabling large language models to generate text with citations (ALCE). *EMNLP* 6465-6488 ✔.
- Es, S. et al. (2024). RAGAs: automated evaluation of retrieval augmented generation. *EACL (demos)* ✔ (named here only to explain "context relevance"; RAGAS is not used).
- Saad-Falcon, J. et al. (2024). ARES: an automated evaluation framework for RAG systems. *NAACL* ✔ (same note).
- Xiong, G. et al. (2024). Benchmarking retrieval-augmented generation for medicine (MIRAGE, MedRAG). *ACL Findings* ✔.
- Sohn, J. et al. (2025). Rationale-guided retrieval augmented generation for medical question answering (RAG²). *NAACL* ✔ (abstract only).
- Reimers, N., Gurevych, I. (2019). Sentence-BERT. *EMNLP* ◻.
