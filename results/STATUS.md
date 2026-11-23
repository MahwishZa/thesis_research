# Status of the research

| | |
|---|---|
| Updated | 2026-11-24 |
| What this file is | The current state: what is done, what is running, what is decided, what is open. History is in `results/CHANGELOG.md`. |
| Marks | ✔ checked against a file or a run; ◻ assumed or not yet checked |
| Freeze | **Nothing is frozen.** The dataset, splits, metrics, prompt and models are frozen only after the research is final and the supervisor agrees (researcher's decision, 2026-11-24). |

## 1. Where things are

| What | Where |
|---|---|
| The research question, objectives | `README.md` (the researcher's text; a test keeps it to "Overview" and "Objectives") |
| The data and how they were built | `data/datasheet.md` |
| The code, commands and tests | `src/methodology.md` |
| The metrics: definitions, sources, how to read | `results/evaluation_guide.md` |
| The literature survey behind the metric choice | `results/metric_survey.md` |
| The audit of the repository (2026-11-24) | `results/repository_audit.md` |
| Results so far | `results/llm_only_text_only_results.md`, `results/summary.csv`, `results/README.md` (how the results folder is organised) |
| History of decisions and changes | `results/CHANGELOG.md` |

## 2. The experiment grid

Four systems × three views × four models, on the test split (240 questions, 30 per class). Chance accuracy is 12.5%.

| System | Text only | Image only | Image + text | Notes |
|---|---|---|---|---|
| **LLM-only** | **done** (4 models, test) ✔ | validated (10 questions per model) ✔; test not run | validated (10 questions per model) ✔; test not run | the prompt is `llm_only`; the same for every model and view |
| **Standard RAG** | not built | not built | not built | design approved: MedCPT encoders, top 3 passages, "Evidence: [n] "quote"" or "Evidence: none" |
| **RAG²** (Sohn et al., NAACL 2025) | not built | not built | not built | waits for the researcher's decision (full reproduction or a "RAG²-style" system without retriever training) |
| **Proposed RAG** | not built | not built | not built | the design is not finalized |

## 3. Results so far: LLM-only, text-only view, test split ✔

Source: `results/llm_only_text_only_results.md` (all numbers from the committed run files). 240 questions per model; no invalid answers and no refusals.

| Model | Accuracy | 95% interval | Macro-F1 |
|---|---|---|---|
| Claude Sonnet 5.5 | 25.4% | 19.3% to 31.9% | 19.4% |
| Llama 4 Maverick | 21.7% | 15.5% to 27.8% | 17.2% |
| Gemini 3.8 Flash | 20.8% | 15.1% to 26.5% | 15.1% |
| GPT-4o | 20.0% | 14.5% to 25.7% | 19.3% |

- All four are above chance (12.5%) but low: in this view the model sees only age, sex and body site.
- McNemar with Holm correction: only Claude against Gemini differs (Holm p = 0.044); the other five pairs do not.
- No model identifies melanoma (0 of 30 each).
- Image claims in the text-only view (a word-list proxy for one kind of hallucination): 0 of 240 for every model.
- These numbers are the baseline for the RAG systems. They say nothing about retrieval.

## 4. Validation trials (format and cost checks; no result reported)

- Text-only, image-only and multimodal trials of 10 questions per model passed: answers parsed, no unexpected failures. The prompt wording was changed after the first trials so that a model is never told to cite an image it was not given.
- GPT-4o **refuses** some image questions (2 of 10 multimodal and 1 of 10 image-only in the trials). Refusals are counted as invalid answers and wrong answers, and are reported next to accuracy.
- Gemini's output limit was raised (2,048 → 8,192 tokens) after one answer was cut off before its letter.

## 5. Decisions

**Made (the researcher's, with the date):**

| Date | Decision |
|---|---|
| 2026-11-22 | Only the Kaggle dataset; automation only (no manual reading, no clinician); balanced test split (30 per class) and validation split (20 per class); top-1 accuracy only |
| 2026-11-23 | Answer format "Answer: C (Melanoma)" after "Reasoning and evidence: …"; prompt named `llm_only`; the verified quote replaces the yes/no evidence flag |
| 2026-11-24 | Metrics: accuracy, macro-F1 (with per-class scores), McNemar with Holm; Hit@k, Recall@k, Precision@k, MRR, nDCG@k and MedCPT context relevance for RAG; Exact Match, Jaccard, RAGAS, context precision, the NLI support check and balanced accuracy removed |
| 2026-11-24 | Text-only image-claim check (approximate) and the semantic-similarity validity check approved; results shown as percentages |
| 2026-11-24 | Standard RAG design approved (MedCPT encoders, k = 3); the sampling files (oversampled and undersampled lists, class weights) stay, because model training is involved or may be |
| 2026-11-24 | The proposed system is still being designed; first report LLM-only and standard RAG |

**Open:**

| # | Open item | Who |
|---|---|---|
| 1 | The proposed RAG: its design | researcher and supervisor |
| 2 | RAG²: reproduce fully, or a RAG²-style system; the paper itself must be read first | researcher, supervisor, then me |
| 3 | The image-only and multimodal LLM-only test runs (about 1,920 paid calls; about 1.7 million tokens in and 0.38 million out, from the trials; no price was looked up ◻) | researcher (budget) |
| 4 | The semantic-similarity validity check: run `python src/evaluate.py similarity-check` after the test runs | researcher |
| 5 | Whether the ethics committee must register or exempt the study ◻ | supervisor |
| 6 | The supervisor's agreement to freeze the dataset, splits, metrics, prompt and models | supervisor, after the research is final |

## 6. Known issues and limits

1. **Missing patient facts hint at the class.** Missing age, sex or site occurs mostly in NV and MEL (28 test cases, 15 of them NV). Report text-only results on all cases and on the cases with all facts (`--by facts_recorded`).
2. **Source confound.** All MEL and NV come from the older ISIC archive; the source is detectable from the images (AUC 0.953). Report results per source.
3. **The semantic similarity looks flat** (cosine score 0.921 to 0.932 for the four models, intervals about ±0.002; not a percentage); its validity is not yet tested.
4. **Hallucination is only approximated.** No published metric measures a false statement about a skin photograph without an expert. The image-claim check is a lower bound; the quote metrics (RAG runs) show only that a quote exists in a passage.
5. **Claude's sampling is not fixed** (the API takes no temperature), so a rerun may differ; the other three models use temperature 0 and a seed, pinned to one host each.
6. **The corpus is not balanced across classes** (melanoma-tagged passages are about 4 times as many as dermatofibroma-tagged ones), and its tags partly come from class names.
7. **Small test set** (240 cases, 190 lesions): always report the 95% interval.

## 7. Next actions

1. Researcher: the image-only and multimodal LLM-only test runs (commands in `src/methodology.md`, section 4), then `collect`, `similarity-check` and a commit of the results.
2. Me: write the results of those runs in the same form as `results/llm_only_text_only_results.md`.
3. Me: build standard RAG (retrieval, the RAG prompt, a runner, tests; a dry run is free) and then ask for approval of a small validation run before any test run.
4. Researcher and supervisor: the proposed system and the RAG² scope.
