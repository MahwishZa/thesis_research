# Results: LLM-only baseline (test split)

Updated 2026-11-24. All numbers come from the committed run files (`results/llm_only/`, `results/comparisons/summary.csv`, `results/comparisons/tables/similarity_check.json`). Scores are percentages (25.4% = 61 of 240 right); p-values and cosine similarities are not. Metric definitions: the docstrings of `src/evaluation/metrics.py` until the final evaluation guide is written.

## 1. Setup and status

| Item | Value |
|---|---|
| Data | 240 test questions per view (30 per class, 190 lesions), the same cases in all three views; chance = 12.5% |
| Views | Text only (age, sex, site) · Image only · Image + text |
| Models (used as they are) | Claude Sonnet 5.5 · Gemini 3.8 Flash · GPT-4o (2024-08-06) · Llama 4 Maverick (Parasail, 8-bit); one host each |
| Settings | Temperature 0 and seed 0, except Claude (no temperature accepted); `max_tokens` 1024, Gemini 8192 |
| Prompt | `llm_only`, the same for all; "Reasoning and evidence: …" then "Answer: C (Melanoma)"; the letter decides |
| Repeats | One run per model and view; a reply without a usable letter, or a refusal, counts as wrong |

| System | Text only | Image only | Image + text |
|---|---|---|---|
| **LLM-only** | done (4 models) ✔ | done (4 models) ✔ | done (4 models) ✔ |
| Standard RAG | built, not run | built, not run | built, not run |
| RAG² | not decided | not decided | not decided |
| Proposed RAG | not designed | not designed | not designed |

## 2. Main results

**Accuracy (95% interval over lesions)** ✔

| Model | Text only | Image only | Image + text |
|---|---|---|---|
| Claude Sonnet 5.5 | **25.4%** (19.3% to 31.9%) | **47.1%** (39.2% to 54.6%) | **54.6%** (47.7% to 62.0%) |
| Gemini 3.8 Flash | **20.8%** (15.1% to 26.5%) | **50.8%** (43.6% to 58.3%) | **53.3%** (45.7% to 61.0%) |
| GPT-4o | **20.0%** (14.5% to 25.7%) | **36.2%** (29.4% to 42.9%) | **41.2%** (34.4% to 47.8%) |
| Llama 4 Maverick | **21.7%** (15.5% to 27.8%) | **36.7%** (29.9% to 43.4%) | **35.8%** (28.9% to 43.2%) |

**Macro-F1 (every class counts the same)** ✔

| Model | Text only | Image only | Image + text |
|---|---|---|---|
| Claude Sonnet 5.5 | 19.4% (14.8% to 24.2%) | 46.0% (38.6% to 52.6%) | 53.5% (46.7% to 59.6%) |
| Gemini 3.8 Flash | 15.1% (11.0% to 19.3%) | 48.8% (42.3% to 54.5%) | 51.4% (44.5% to 57.7%) |
| GPT-4o | 19.3% (13.9% to 24.0%) | 35.6% (29.7% to 41.0%) | 41.0% (34.5% to 46.5%) |
| Llama 4 Maverick | 17.2% (13.1% to 21.0%) | 32.0% (26.5% to 37.4%) | 31.0% (25.5% to 36.0%) |

## 3. Findings

| # | Finding | Evidence |
|---|---|---|
| 1 | The image carries the diagnosis: every model gains 15 to 30 points from text only to image only | Claude +21.7, Gemini +30.0, GPT-4o +16.2, Llama +15.0; Holm p ≤ 0.0004 (table 6.4) |
| 2 | Claude and Gemini are above GPT-4o and Llama in both image views; inside each pair the models cannot be separated | Holm p 0.0008 to 0.014 (image only), < 0.0001 to 0.0044 (image + text); Claude vs Gemini 0.60 and 0.78; GPT-4o vs Llama 1.0 and 0.26 |
| 3 | Adding patient facts to the image helps only Claude | +7.5 points, Holm p = 0.0014; GPT-4o +5.0, Gemini +2.5, Llama −0.8 not established |
| 4 | No model finds melanoma from text alone | recall 0 of 30 for all four; with the image 26.7% to 56.7% |
| 5 | Vascular lesion is found best, squamous cell carcinoma worst; Llama never finds AK in the image views | tables 4.1 to 4.3 |
| 6 | Models lean on one answer: Llama names BKL in 66.2% (text), 42.9% (image), 49.6% (image + text) of its answers | table 7.1 |
| 7 | GPT-4o refuses image questions: 28 (11.7%) image only, 12 (5.0%) image + text; none in text only | table 5 |
| 8 | No described image in the text-only reasonings | 0 of 240 for each model (word list; lower bound) |
| 9 | The semantic similarity identifies the answered disease far above chance (39.2% to 91.0% vs 12.5%) but meets the pass mark in 6 of 12 runs only | table 8.2 |

## 4. Per-class recall (30 cases per class: one value is about ±18 points)

**4.1 Text only** ✔

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 0.0% | 0.0% | 0.0% | 0.0% |
| NV | 90.0% | 90.0% | 23.3% | 60.0% |
| BCC | 16.7% | 16.7% | 16.7% | 3.3% |
| AK | 6.7% | 6.7% | 33.3% | 33.3% |
| BKL | 53.3% | 26.7% | 36.7% | 70.0% |
| DF | 33.3% | 26.7% | 36.7% | 6.7% |
| VASC | 0.0% | 0.0% | 6.7% | 0.0% |
| SCC | 3.3% | 0.0% | 6.7% | 0.0% |

**4.2 Image only** ✔

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 43.3% | 40.0% | 26.7% | 33.3% |
| NV | 40.0% | 36.7% | 53.3% | 50.0% |
| BCC | 33.3% | 63.3% | 33.3% | 10.0% |
| AK | 66.7% | 53.3% | 6.7% | 0.0% |
| BKL | 53.3% | 63.3% | 53.3% | 73.3% |
| DF | 33.3% | 46.7% | 50.0% | 50.0% |
| VASC | 96.7% | 96.7% | 66.7% | 76.7% |
| SCC | 10.0% | 6.7% | 0.0% | 0.0% |

**4.3 Image + text** ✔

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 56.7% | 50.0% | 36.7% | 33.3% |
| NV | 46.7% | 36.7% | 50.0% | 50.0% |
| BCC | 36.7% | 66.7% | 46.7% | 6.7% |
| AK | 73.3% | 63.3% | 30.0% | 0.0% |
| BKL | 60.0% | 56.7% | 53.3% | 83.3% |
| DF | 46.7% | 53.3% | 53.3% | 30.0% |
| VASC | 100.0% | 93.3% | 60.0% | 83.3% |
| SCC | 16.7% | 6.7% | 0.0% | 0.0% |

Precision, F1 and specificity per class: appendix B. Confusion matrices: appendix A.

## 5. Invalid answers and refusals

All invalid answers are refusals, and only GPT-4o refused. The last column is not a standard metric; it only sizes the effect ✔.

| Run | Invalid (all refusals) | Share of 240 | Accuracy as reported | Accuracy on the answered questions only |
|---|---|---|---|---|
| GPT-4o, Image only | 28 (28 refusals) | 11.7% | 36.2% (87 of 240) | 41.0% (87 of 212) |
| GPT-4o, Image + text | 12 (12 refusals) | 5.0% | 41.2% (99 of 240) | 43.4% (99 of 228) |

## 6. Comparisons (McNemar's exact test, Holm-corrected)

**6.1 Between models, text only**

| Pair | Only first right | Only second right | p | Holm-adjusted p |
|---|---|---|---|---|
| Claude (25.4%) vs GPT-4o (20.0%) | 27 | 14 | 0.0596 | **0.2979** |
| Claude (25.4%) vs Llama (21.7%) | 27 | 18 | 0.2327 | **0.9308** |
| Claude (25.4%) vs Gemini (20.8%) | 13 | 2 | 0.0074 | **0.0443** |
| GPT-4o (20.0%) vs Llama (21.7%) | 25 | 29 | 0.6835 | **1.0000** |
| GPT-4o (20.0%) vs Gemini (20.8%) | 20 | 22 | 0.8776 | **1.0000** |
| Llama (21.7%) vs Gemini (20.8%) | 24 | 22 | 0.8830 | **1.0000** |

**6.2 Between models, image only**

| Pair | Only first right | Only second right | p | Holm-adjusted p |
|---|---|---|---|---|
| Claude (47.1%) vs GPT-4o (36.2%) | 51 | 25 | 0.0038 | **0.0142** |
| Claude (47.1%) vs Llama (36.7%) | 47 | 22 | 0.0035 | **0.0142** |
| Claude (47.1%) vs Gemini (50.8%) | 25 | 34 | 0.2976 | **0.5952** |
| GPT-4o (36.2%) vs Llama (36.7%) | 22 | 23 | 1.0000 | **1.0000** |
| GPT-4o (36.2%) vs Gemini (50.8%) | 23 | 58 | 0.0001 | **0.0008** |
| Llama (36.7%) vs Gemini (50.8%) | 24 | 58 | 0.0002 | **0.0011** |

**6.3 Between models, image + text**

| Pair | Only first right | Only second right | p | Holm-adjusted p |
|---|---|---|---|---|
| Claude (54.6%) vs GPT-4o (41.2%) | 51 | 19 | 0.0002 | **0.0007** |
| Claude (54.6%) vs Llama (35.8%) | 62 | 17 | < 0.0001 | **< 0.0001** |
| Claude (54.6%) vs Gemini (53.3%) | 27 | 24 | 0.7798 | **0.7798** |
| GPT-4o (41.2%) vs Llama (35.8%) | 38 | 25 | 0.1299 | **0.2598** |
| GPT-4o (41.2%) vs Gemini (53.3%) | 25 | 54 | 0.0015 | **0.0044** |
| Llama (35.8%) vs Gemini (53.3%) | 24 | 66 | < 0.0001 | **< 0.0001** |

**6.4 Between views, per model** (change in accuracy, same 240 cases)

| Model | Text → image | Text → image + text | Image → image + text |
|---|---|---|---|
| Claude Sonnet 5.5 | +21.7 points (Holm p < 0.0001) | +29.2 points (Holm p < 0.0001) | +7.5 points (Holm p 0.0014) |
| Gemini 3.8 Flash | +30.0 points (Holm p < 0.0001) | +32.5 points (Holm p < 0.0001) | +2.5 points (Holm p 0.3915) |
| GPT-4o | +16.2 points (Holm p 0.0002) | +21.2 points (Holm p < 0.0001) | +5.0 points (Holm p 0.0961) |
| Llama 4 Maverick | +15.0 points (Holm p 0.0004) | +14.2 points (Holm p 0.0004) | -0.8 points (Holm p 0.8506) |

## 7. Answer habits

**7.1 Class named most often** (a balanced test set gives 12.5%)

| Run | Class answered most often | Share of the 240 answers | Expected (balanced) |
|---|---|---|---|
| Claude Sonnet 5.5, Text only | BKL (102) | 42.5% | 12.5% |
| Gemini 3.8 Flash, Text only | NV (90) | 37.5% | 12.5% |
| GPT-4o, Text only | BKL (90) | 37.5% | 12.5% |
| Llama 4 Maverick, Text only | BKL (159) | 66.2% | 12.5% |
| Claude Sonnet 5.5, Image only | AK (58) | 24.2% | 12.5% |
| Gemini 3.8 Flash, Image only | BKL (51) | 21.2% | 12.5% |
| GPT-4o, Image only | DF (58) | 24.2% | 12.5% |
| Llama 4 Maverick, Image only | BKL (103) | 42.9% | 12.5% |
| Claude Sonnet 5.5, Image + text | AK (52) | 21.7% | 12.5% |
| Gemini 3.8 Flash, Image + text | BKL (48) | 20.0% | 12.5% |
| GPT-4o, Image + text | BKL (50) | 20.8% | 12.5% |
| Llama 4 Maverick, Image + text | BKL (119) | 49.6% | 12.5% |

**7.2 Letter chosen most often** (options are shuffled, so a letter has no meaning)

| Run | Most chosen letter | Share of the 240 answers | Expected (1 in 8) |
|---|---|---|---|
| Claude Sonnet 5.5, Text only | B (54) | 22.5% | 12.5% |
| Gemini 3.8 Flash, Text only | B (52) | 21.7% | 12.5% |
| GPT-4o, Text only | B (46) | 19.2% | 12.5% |
| Llama 4 Maverick, Text only | C (40) | 16.7% | 12.5% |
| Claude Sonnet 5.5, Image only | B (35) | 14.6% | 12.5% |
| Gemini 3.8 Flash, Image only | B (37) | 15.4% | 12.5% |
| GPT-4o, Image only | B (39) | 18.4% | 12.5% |
| Llama 4 Maverick, Image only | B (40) | 16.7% | 12.5% |
| Claude Sonnet 5.5, Image + text | B (40) | 16.7% | 12.5% |
| Gemini 3.8 Flash, Image + text | B (40) | 16.7% | 12.5% |
| GPT-4o, Image + text | B (44) | 19.3% | 12.5% |
| Llama 4 Maverick, Image + text | B (38) | 15.8% | 12.5% |

## 8. Semantic similarity

**8.1 Mean similarity** (cosine of `S-PubMedBert-MS-MARCO` embeddings, reasoning against the reference texts of the true disease; answers without a usable letter are left out; n = answers scored). Differences between models are too small to rank them (0.006 or less in the image views).

| Model | Text only | Image only | Image + text |
|---|---|---|---|
| Claude Sonnet 5.5 | 0.9307 (n = 240) | 0.9495 (n = 240) | 0.9501 (n = 240) |
| Gemini 3.8 Flash | 0.9320 (n = 240) | 0.9481 (n = 240) | 0.9494 (n = 240) |
| GPT-4o | 0.9268 (n = 240) | 0.9439 (n = 212) | 0.9482 (n = 228) |
| Llama 4 Maverick | 0.9211 (n = 240) | 0.9498 (n = 240) | 0.9502 (n = 240) |

**8.2 Validity check** (pass mark set in advance: the answered disease has the single highest similarity of the 8 in ≥ 50% of the answers, and in ≥ 25% for every answered class with 10 or more answers; chance 12.5%) ✔

| Run | Answers read | Answered disease has the highest similarity | Floor 25% failed by (answered class, share) | Result |
|---|---|---|---|---|
| Claude Sonnet 5.5, Text only | 240 | 45.8% | BCC 0% | not passed |
| Gemini 3.8 Flash, Text only | 240 | 59.6% | BCC 0% | not passed |
| GPT-4o, Text only | 240 | 53.3% | BCC 15% | not passed |
| Llama 4 Maverick, Text only | 240 | 47.9% | BCC 0% | not passed |
| Claude Sonnet 5.5, Image only | 240 | 45.0% | MEL 0%, BKL 22% | not passed |
| Gemini 3.8 Flash, Image only | 240 | 56.2% | none | PASSED |
| GPT-4o, Image only | 212 | 91.0% | none | PASSED |
| Llama 4 Maverick, Image only | 240 | 69.6% | none | PASSED |
| Claude Sonnet 5.5, Image + text | 240 | 39.2% | MEL 3%, AK 23%, BKL 17% | not passed |
| Gemini 3.8 Flash, Image + text | 240 | 55.4% | none | PASSED |
| GPT-4o, Image + text | 228 | 83.8% | none | PASSED |
| Llama 4 Maverick, Image + text | 240 | 62.1% | none | PASSED |

## 9. Hallucination proxy

| Check | Result | Limit |
|---|---|---|
| Text-only reasonings that describe an image that was not given | 0 of 240 for Claude, Gemini, GPT-4o and Llama | word list: a lower bound; other kinds of hallucination are not measured |

## 10. Cost and speed

Minutes are the sum of the calls (some runs ran at the same time). Total over the 12 runs: 1,926,836 tokens in, 471,833 tokens out. Prices are not included.

| Run | Tokens in | Tokens out | Minutes (sum of the calls) | Seconds per question |
|---|---|---|---|---|
| Claude Sonnet 5.5, Text only | 75,044 | 44,032 | 11.2 | 2.8 |
| Claude Sonnet 5.5, Image only | 162,960 | 42,703 | 11.8 | 3.0 |
| Claude Sonnet 5.5, Image + text | 171,044 | 45,301 | 12.1 | 3.0 |
| Gemini 3.8 Flash, Text only | 48,431 | 93,578 | 23.7 | 5.9 |
| Gemini 3.8 Flash, Image only | 301,680 | 56,154 | 53.7 | 13.4 |
| Gemini 3.8 Flash, Image + text | 307,631 | 71,406 | 49.9 | 12.5 |
| GPT-4o, Text only | 50,534 | 18,355 | 6.3 | 1.6 |
| GPT-4o, Image only | 150,960 | 11,801 | 15.4 | 3.9 |
| GPT-4o, Image + text | 156,134 | 13,695 | 15.6 | 3.9 |
| Llama 4 Maverick, Text only | 50,449 | 24,474 | 8.3 | 2.1 |
| Llama 4 Maverick, Image only | 223,440 | 24,580 | 20.7 | 5.2 |
| Llama 4 Maverick, Image + text | 228,529 | 25,754 | 21.7 | 5.4 |

## 11. Limits

| Limit | Consequence |
|---|---|
| One prompt, one dataset (240 balanced cases), one run per model, one host per model | Results are not "the model's dermatology skill" |
| Claude's sampling is not fixed | A rerun may differ |
| Refusals counted as wrong | GPT-4o's score mixes lower skill and refusal; compare with the reported accuracy only |
| Semantic similarity measures resemblance to class-level texts | Not a measure of reasoning quality; fails where reference texts are thin (melanoma: 2, BCC) |
| Balanced splits | Precision is not a clinic's precision |
| Source confound (MEL and NV from the older archive) | Image results may partly reflect source |
| 28 of 240 test cases lack age, sex or site (NV 15, MEL 7, DF 2, VASC 2, BCC 1, SCC 1) | "Not recorded" hints at nevus, so text-only accuracy is partly inflated; accuracy by subgroup is not reported |
| No clinician reads any answer | Hallucination is only approximated |

## 12. Open items

| Item | Status |
|---|---|
| Standard RAG (BM25 candidates re-ordered by the MedCPT cross-encoder, image search for the image views, top 3) | built and tested; indexes built on the laptop; retrieval validated (free, `results/standard_rag/<view>/retrieval_validation.json`); no model run yet |
| RAG backbone | Qwen3.5-2B, local, chosen 2026-11-27; its LLM-only baseline and the validation pilot (`python -m src.main pilot-check`) come first; no run yet |
| Proposed RAG and RAG² scope | not decided |
| Reference texts for melanoma and BCC (semantic similarity) | not decided |
| GPT-4o refusals per class | not done |
| Freeze of dataset, splits, metrics, prompt and models | after the research is final and the supervisor agrees |

## 13. Files and reproduction

| What | Where |
|---|---|
| Predictions, settings, report of each run | `results/llm_only/<view>/<model>_test/{predictions.jsonl, run.json, report.json}` (before 2026-11-25: `<model>/<view>_test/`) |
| One row per run | `results/comparisons/summary.csv` |
| Every number of every run's report, one per row | `results/comparisons/metrics.csv` |
| Tests between models / between views | `results/comparisons/tables/llm_only_models_<view>.json` / `llm_only_views_<model>.json` |
| Validity check | `results/comparisons/tables/similarity_check.json` |

```powershell
python -m src.main llm-only --model gpt-4o --view image_only --split test   # one run (paid, resumable)
python -m src.main collect --force                                          # reports and results/comparisons/summary.csv
python -m src.main similarity-check                                         # validity check
python -m src.main compare --out <file> --run claude=<questions>,<predictions> --run gemini=...
```

## Appendix A: confusion matrices (rows: true class; columns: class answered; last column: invalid)

**Claude Sonnet 5.5, Text only**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 0 | 8 | 7 | 2 | 12 | 1 | 0 | 0 | 0 |
| **NV** | 0 | 27 | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| **BCC** | 1 | 7 | 5 | 0 | 16 | 1 | 0 | 0 | 0 |
| **AK** | 0 | 2 | 14 | 2 | 12 | 0 | 0 | 0 | 0 |
| **BKL** | 0 | 5 | 7 | 0 | 16 | 2 | 0 | 0 | 0 |
| **DF** | 0 | 3 | 0 | 2 | 14 | 10 | 0 | 1 | 0 |
| **VASC** | 2 | 12 | 2 | 1 | 11 | 1 | 0 | 1 | 0 |
| **SCC** | 3 | 0 | 5 | 1 | 19 | 1 | 0 | 1 | 0 |

**Gemini 3.8 Flash, Text only**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 0 | 12 | 9 | 1 | 8 | 0 | 0 | 0 | 0 |
| **NV** | 0 | 27 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
| **BCC** | 2 | 9 | 5 | 0 | 14 | 0 | 0 | 0 | 0 |
| **AK** | 1 | 1 | 19 | 2 | 7 | 0 | 0 | 0 | 0 |
| **BKL** | 0 | 9 | 11 | 0 | 8 | 2 | 0 | 0 | 0 |
| **DF** | 1 | 10 | 0 | 2 | 9 | 8 | 0 | 0 | 0 |
| **VASC** | 4 | 12 | 2 | 1 | 10 | 1 | 0 | 0 | 0 |
| **SCC** | 2 | 10 | 5 | 0 | 11 | 2 | 0 | 0 | 0 |

**GPT-4o, Text only**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 0 | 1 | 6 | 7 | 11 | 2 | 0 | 3 | 0 |
| **NV** | 0 | 7 | 0 | 1 | 18 | 4 | 0 | 0 | 0 |
| **BCC** | 0 | 0 | 5 | 2 | 20 | 2 | 0 | 1 | 0 |
| **AK** | 0 | 0 | 18 | 10 | 1 | 0 | 0 | 1 | 0 |
| **BKL** | 0 | 0 | 13 | 4 | 11 | 2 | 0 | 0 | 0 |
| **DF** | 2 | 0 | 0 | 11 | 5 | 11 | 0 | 1 | 0 |
| **VASC** | 2 | 3 | 1 | 5 | 14 | 1 | 2 | 2 | 0 |
| **SCC** | 3 | 0 | 4 | 10 | 10 | 1 | 0 | 2 | 0 |

**Llama 4 Maverick, Text only**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 0 | 2 | 2 | 5 | 20 | 0 | 1 | 0 | 0 |
| **NV** | 0 | 18 | 0 | 1 | 10 | 0 | 1 | 0 | 0 |
| **BCC** | 0 | 1 | 1 | 0 | 28 | 0 | 0 | 0 | 0 |
| **AK** | 0 | 0 | 5 | 10 | 14 | 0 | 0 | 1 | 0 |
| **BKL** | 0 | 0 | 6 | 3 | 21 | 0 | 0 | 0 | 0 |
| **DF** | 0 | 3 | 0 | 2 | 20 | 2 | 2 | 1 | 0 |
| **VASC** | 1 | 3 | 1 | 1 | 24 | 0 | 0 | 0 | 0 |
| **SCC** | 1 | 0 | 2 | 4 | 22 | 0 | 1 | 0 | 0 |

**Claude Sonnet 5.5, Image only**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 13 | 0 | 0 | 4 | 13 | 0 | 0 | 0 | 0 |
| **NV** | 4 | 12 | 1 | 1 | 10 | 2 | 0 | 0 | 0 |
| **BCC** | 7 | 0 | 10 | 6 | 2 | 1 | 3 | 1 | 0 |
| **AK** | 0 | 0 | 0 | 20 | 8 | 0 | 0 | 2 | 0 |
| **BKL** | 1 | 3 | 1 | 7 | 16 | 0 | 0 | 2 | 0 |
| **DF** | 1 | 1 | 5 | 6 | 3 | 10 | 2 | 2 | 0 |
| **VASC** | 1 | 0 | 0 | 0 | 0 | 0 | 29 | 0 | 0 |
| **SCC** | 3 | 0 | 2 | 14 | 6 | 2 | 0 | 3 | 0 |

**Gemini 3.8 Flash, Image only**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 12 | 2 | 1 | 7 | 8 | 0 | 0 | 0 | 0 |
| **NV** | 1 | 11 | 1 | 3 | 8 | 6 | 0 | 0 | 0 |
| **BCC** | 4 | 0 | 19 | 1 | 4 | 2 | 0 | 0 | 0 |
| **AK** | 1 | 0 | 3 | 16 | 7 | 2 | 0 | 1 | 0 |
| **BKL** | 0 | 3 | 1 | 6 | 19 | 1 | 0 | 0 | 0 |
| **DF** | 1 | 1 | 4 | 7 | 2 | 14 | 1 | 0 | 0 |
| **VASC** | 0 | 1 | 0 | 0 | 0 | 0 | 29 | 0 | 0 |
| **SCC** | 5 | 0 | 9 | 5 | 3 | 6 | 0 | 2 | 0 |

**GPT-4o, Image only**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 8 | 8 | 0 | 0 | 9 | 2 | 0 | 0 | 3 |
| **NV** | 1 | 16 | 0 | 0 | 3 | 8 | 0 | 0 | 2 |
| **BCC** | 4 | 1 | 10 | 1 | 2 | 5 | 1 | 0 | 6 |
| **AK** | 0 | 1 | 5 | 2 | 12 | 8 | 0 | 0 | 2 |
| **BKL** | 1 | 2 | 0 | 3 | 16 | 6 | 0 | 0 | 2 |
| **DF** | 2 | 0 | 3 | 2 | 2 | 15 | 1 | 0 | 5 |
| **VASC** | 1 | 1 | 0 | 0 | 1 | 2 | 20 | 0 | 5 |
| **SCC** | 1 | 0 | 7 | 0 | 7 | 12 | 0 | 0 | 3 |

**Llama 4 Maverick, Image only**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 10 | 5 | 0 | 0 | 14 | 1 | 0 | 0 | 0 |
| **NV** | 2 | 15 | 0 | 0 | 11 | 2 | 0 | 0 | 0 |
| **BCC** | 7 | 1 | 3 | 0 | 13 | 3 | 1 | 2 | 0 |
| **AK** | 0 | 0 | 0 | 0 | 26 | 3 | 1 | 0 | 0 |
| **BKL** | 1 | 3 | 0 | 1 | 22 | 3 | 0 | 0 | 0 |
| **DF** | 6 | 0 | 2 | 0 | 4 | 15 | 3 | 0 | 0 |
| **VASC** | 5 | 0 | 0 | 0 | 1 | 1 | 23 | 0 | 0 |
| **SCC** | 6 | 3 | 1 | 0 | 12 | 7 | 1 | 0 | 0 |

**Claude Sonnet 5.5, Image + text**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 17 | 1 | 0 | 4 | 8 | 0 | 0 | 0 | 0 |
| **NV** | 4 | 14 | 1 | 0 | 9 | 2 | 0 | 0 | 0 |
| **BCC** | 8 | 0 | 11 | 4 | 1 | 1 | 2 | 3 | 0 |
| **AK** | 0 | 0 | 0 | 22 | 6 | 0 | 0 | 2 | 0 |
| **BKL** | 3 | 2 | 1 | 5 | 18 | 0 | 0 | 1 | 0 |
| **DF** | 2 | 0 | 4 | 5 | 1 | 14 | 2 | 2 | 0 |
| **VASC** | 0 | 0 | 0 | 0 | 0 | 0 | 30 | 0 | 0 |
| **SCC** | 4 | 0 | 5 | 12 | 3 | 1 | 0 | 5 | 0 |

**Gemini 3.8 Flash, Image + text**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 15 | 0 | 1 | 5 | 7 | 2 | 0 | 0 | 0 |
| **NV** | 2 | 11 | 0 | 0 | 10 | 7 | 0 | 0 | 0 |
| **BCC** | 5 | 2 | 20 | 1 | 2 | 0 | 0 | 0 | 0 |
| **AK** | 3 | 0 | 1 | 19 | 6 | 0 | 0 | 1 | 0 |
| **BKL** | 3 | 2 | 3 | 4 | 17 | 1 | 0 | 0 | 0 |
| **DF** | 1 | 0 | 5 | 5 | 2 | 16 | 1 | 0 | 0 |
| **VASC** | 0 | 0 | 1 | 0 | 0 | 0 | 28 | 1 | 0 |
| **SCC** | 7 | 0 | 11 | 5 | 4 | 1 | 0 | 2 | 0 |

**GPT-4o, Image + text**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 11 | 2 | 1 | 2 | 12 | 0 | 0 | 0 | 2 |
| **NV** | 0 | 15 | 0 | 0 | 4 | 10 | 0 | 0 | 1 |
| **BCC** | 5 | 0 | 14 | 4 | 2 | 3 | 0 | 0 | 2 |
| **AK** | 0 | 0 | 9 | 9 | 6 | 3 | 0 | 2 | 1 |
| **BKL** | 1 | 1 | 2 | 7 | 16 | 2 | 0 | 0 | 1 |
| **DF** | 4 | 0 | 2 | 2 | 3 | 16 | 0 | 0 | 3 |
| **VASC** | 1 | 1 | 1 | 0 | 1 | 8 | 18 | 0 | 0 |
| **SCC** | 4 | 0 | 9 | 4 | 6 | 4 | 1 | 0 | 2 |

**Llama 4 Maverick, Image + text**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 10 | 3 | 0 | 0 | 17 | 0 | 0 | 0 | 0 |
| **NV** | 2 | 15 | 0 | 0 | 10 | 3 | 0 | 0 | 0 |
| **BCC** | 7 | 0 | 2 | 0 | 14 | 1 | 4 | 2 | 0 |
| **AK** | 2 | 0 | 3 | 0 | 23 | 0 | 1 | 1 | 0 |
| **BKL** | 2 | 2 | 0 | 0 | 25 | 1 | 0 | 0 | 0 |
| **DF** | 4 | 1 | 0 | 0 | 10 | 9 | 4 | 2 | 0 |
| **VASC** | 2 | 2 | 0 | 0 | 1 | 0 | 25 | 0 | 0 |
| **SCC** | 8 | 1 | 1 | 0 | 19 | 0 | 1 | 0 | 0 |

## Appendix B: precision, F1 and specificity per class

**Precision, Text only**

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 0.0% | 0.0% | 0.0% | 0.0% |
| NV | 42.2% | 30.0% | 63.6% | 66.7% |
| BCC | 12.5% | 9.8% | 10.6% | 5.9% |
| AK | 25.0% | 28.6% | 20.0% | 38.5% |
| BKL | 15.7% | 11.8% | 12.2% | 13.2% |
| DF | 58.8% | 57.1% | 47.8% | 100.0% |
| VASC | 0.0% | 0.0% | 100.0% | 0.0% |
| SCC | 33.3% | 0.0% | 20.0% | 0.0% |

**Precision, Image only**

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 43.3% | 50.0% | 44.4% | 27.0% |
| NV | 75.0% | 61.1% | 55.2% | 55.6% |
| BCC | 52.6% | 50.0% | 40.0% | 50.0% |
| AK | 34.5% | 35.6% | 25.0% | 0.0% |
| BKL | 27.6% | 37.3% | 30.8% | 21.4% |
| DF | 66.7% | 45.2% | 25.9% | 42.9% |
| VASC | 85.3% | 96.7% | 90.9% | 79.3% |
| SCC | 30.0% | 66.7% | 0.0% | 0.0% |

**Precision, Image + text**

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 44.7% | 41.7% | 42.3% | 27.0% |
| NV | 82.4% | 73.3% | 78.9% | 62.5% |
| BCC | 50.0% | 47.6% | 36.8% | 33.3% |
| AK | 42.3% | 48.7% | 32.1% | 0.0% |
| BKL | 39.1% | 35.4% | 32.0% | 21.0% |
| DF | 77.8% | 59.3% | 34.8% | 64.3% |
| VASC | 88.2% | 96.6% | 94.7% | 71.4% |
| SCC | 38.5% | 50.0% | 0.0% | 0.0% |

**F1, Text only**

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 0.0% | 0.0% | 0.0% | 0.0% |
| NV | 57.4% | 45.0% | 34.1% | 63.2% |
| BCC | 14.3% | 12.3% | 13.0% | 4.3% |
| AK | 10.5% | 10.8% | 25.0% | 35.7% |
| BKL | 24.2% | 16.3% | 18.3% | 22.2% |
| DF | 42.6% | 36.4% | 41.5% | 12.5% |
| VASC | 0.0% | 0.0% | 12.5% | 0.0% |
| SCC | 6.1% | 0.0% | 10.0% | 0.0% |

**F1, Image only**

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 43.3% | 44.4% | 33.3% | 29.9% |
| NV | 52.2% | 45.8% | 54.2% | 52.6% |
| BCC | 40.8% | 55.9% | 36.4% | 16.7% |
| AK | 45.5% | 42.7% | 10.5% | 0.0% |
| BKL | 36.4% | 46.9% | 39.0% | 33.1% |
| DF | 44.4% | 45.9% | 34.1% | 46.2% |
| VASC | 90.6% | 96.7% | 76.9% | 78.0% |
| SCC | 15.0% | 12.1% | 0.0% | 0.0% |

**F1, Image + text**

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 50.0% | 45.5% | 39.3% | 29.9% |
| NV | 59.6% | 48.9% | 61.2% | 55.6% |
| BCC | 42.3% | 55.6% | 41.2% | 11.1% |
| AK | 53.7% | 55.1% | 31.0% | 0.0% |
| BKL | 47.4% | 43.6% | 40.0% | 33.6% |
| DF | 58.3% | 56.1% | 42.1% | 40.9% |
| VASC | 93.8% | 94.9% | 73.5% | 76.9% |
| SCC | 23.3% | 11.8% | 0.0% | 0.0% |

**Specificity, Text only**

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 97.1% | 95.2% | 96.7% | 99.0% |
| NV | 82.4% | 70.0% | 98.1% | 95.7% |
| BCC | 83.3% | 78.1% | 80.0% | 92.4% |
| AK | 97.1% | 97.6% | 81.0% | 92.4% |
| BKL | 59.0% | 71.4% | 62.4% | 34.3% |
| DF | 96.7% | 97.1% | 94.3% | 100.0% |
| VASC | 100.0% | 100.0% | 100.0% | 97.6% |
| SCC | 99.0% | 100.0% | 96.2% | 99.0% |

**Specificity, Image only**

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 91.9% | 94.3% | 95.2% | 87.1% |
| NV | 98.1% | 96.7% | 93.8% | 94.3% |
| BCC | 95.7% | 91.0% | 92.9% | 98.6% |
| AK | 81.9% | 86.2% | 97.1% | 99.5% |
| BKL | 80.0% | 84.8% | 82.9% | 61.4% |
| DF | 97.6% | 91.9% | 79.5% | 90.5% |
| VASC | 97.6% | 99.5% | 99.0% | 97.1% |
| SCC | 96.7% | 99.5% | 100.0% | 99.0% |

**Specificity, Image + text**

| Class | Claude Sonnet 5.5 | Gemini 3.8 Flash | GPT-4o | Llama 4 Maverick |
|---|---|---|---|---|
| MEL | 90.0% | 90.0% | 92.9% | 87.1% |
| NV | 98.6% | 98.1% | 98.1% | 95.7% |
| BCC | 94.8% | 89.5% | 88.6% | 98.1% |
| AK | 85.7% | 90.5% | 91.0% | 100.0% |
| BKL | 86.7% | 85.2% | 83.8% | 55.2% |
| DF | 98.1% | 94.8% | 85.7% | 97.6% |
| VASC | 98.1% | 99.5% | 99.5% | 95.2% |
| SCC | 96.2% | 99.0% | 99.0% | 97.6% |
