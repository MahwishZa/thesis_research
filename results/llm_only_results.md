# LLM-only baseline: results in the three views (test split)

| | |
|---|---|
| Document status | Final for the LLM-only baseline, as of 2026-11-24. Every number is read from the committed run files (`results/llm_only/`, `results/summary.csv`, `results/similarity_check.json`) or recomputed from them. |
| What it covers | 4 models × 3 views = 12 runs of 240 questions (test split, 30 per class, 190 lesions). Chance accuracy is 12.5%. |
| Related documents | `results/evaluation_guide.md` (what each metric is and why it was chosen), `src/methodology.md` (how to run and reproduce), `data/datasheet.md` (the data), `results/STATUS.md` (where the work stands). |
| Marks | ✔ read from a committed file or recomputed from it; ◻ my reading, not tested. Scores are percentages (25.4% = 61 of 240 right). p-values and the semantic-similarity cosine are not percentages. |

## 1. What was run

| Item | Value |
|---|---|
| System | LLM-only: no retrieval; one question, one answer (prompt `llm_only`, the same for all models and views, except one phrase that names where findings may come from: "in the patient facts" for text only, "in the image" for image only, both for image + text) |
| Questions | 240 per view (test split), the same 240 cases in all three views, 8 options in a fixed shuffled order per case |
| Models (as used, no training) | Claude Sonnet 5.5 (Anthropic); Gemini 3.8 Flash (Google AI Studio); GPT-4o, snapshot 2024-08-06 (OpenAI); Llama 4 Maverick (Parasail host, 8-bit weights). Each model is pinned to one host |
| Settings | Temperature 0 and seed 0 for GPT-4o, Gemini and Llama; Claude takes no temperature (default sampling), so a rerun may differ. `max_tokens`: Claude 1024, GPT-4o 1024, Llama 1024, Gemini 8192 |
| Answer format | "Reasoning and evidence: …" then "Answer: C (Melanoma)"; the letter decides. A reply without a usable letter, or a refusal, counts as wrong |
| One run per model and view | No repeats: sampling noise between reruns is not measured |
| Status | All 12 runs complete (240 of 240 answered) |

## 2. The main result

**Accuracy (95% interval from 1,000 resamples of whole lesions)** ✔

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

Chance is 12.5%. All 12 accuracies are above chance (the lowest interval end is 14.5%).

## 3. What the results show

1. **The image carries the diagnosis.** From text only to image only, every model gains 15 to 30 points (Claude +21.7, Gemini +30.0, GPT-4o +16.2, Llama +15.0), and every gain is established (Holm p ≤ 0.0004, section 6.2). With age, sex and body site alone, the four models reach 20.0% to 25.4%.
2. **Two groups in the image views.** Claude and Gemini are clearly above GPT-4o and Llama (image only: Holm p from 0.0008 to 0.014; image + text: from < 0.0001 to 0.0044). Within a group the models cannot be separated: Claude and Gemini (Holm p 0.60 and 0.78), GPT-4o and Llama (1.0 and 0.26).
3. **Adding the patient facts to the image helps one model.** Claude gains 7.5 points (Holm p = 0.0014). GPT-4o (+5.0), Gemini (+2.5) and Llama (−0.8) show no established change.
4. **Melanoma is not found from text.** No model identifies a melanoma from age, sex and site (0 of 30 each). With the image, melanoma recall is 26.7% to 43.3% (image only) and 33.3% to 56.7% (image + text).
5. **Some diseases are easy and some are not.** Vascular lesion is found best (image only: 66.7% to 96.7%), squamous cell carcinoma worst (0% to 10.0% image only, 0% to 16.7% image + text). Llama never finds actinic keratosis in the image views (0%).
6. **Models lean on one answer.** Llama names benign keratosis-like lesion (BKL) in 66.2% of its text-only answers, 42.9% (image only) and 49.6% (image + text), against 12.5% if answers were spread evenly (section 4.6). Claude names actinic keratosis in 24.2% (image only) and 21.7% (image + text) of its answers.
7. **GPT-4o refuses some image questions:** 28 of 240 (11.7%) image only and 12 of 240 (5.0%) image + text, none in text only. A refusal counts as wrong, so its accuracy is lower than its accuracy on the questions it answered (section 4.4).
8. **No described image in the text-only view:** 0 of 240 reasonings for each model (a word-list check; a lower bound).
9. **The semantic similarity passes the validity check in 6 of 12 runs** (all of GPT-4o, Gemini and Llama in the image views); it fails in all four text-only runs and for Claude in both image views, because the answered disease is identified in fewer than 50% of Claude's answers or a class stays under 25% (section 4.8).

## 4. Metric by metric

### 4.1 Accuracy
The share of the 240 questions answered with the right class. It is the one metric every medical multiple-choice RAG paper reports, so it allows comparison with the literature. The interval shows how far chance alone could move it: about ±7 to 8 points here, so two models should be compared with the paired test of section 4.7, not by eye.

### 4.2 Macro-F1
The mean of the eight per-class F1 scores. It exposes a model that is right on common answers but ignores classes. Llama's macro-F1 is 4.7 points below its accuracy in the image views (32.0% against 36.7%; 31.0% against 35.8%), a larger gap than the other models' (0.2 to 2.0 points). Its recall is 0% to 10% for AK, BCC and SCC, so a few classes carry its score.

### 4.3 Per-class recall, precision, F1 and specificity
Recall is the share of the 30 cases of a disease that were found; precision is the share of answers naming a disease that were right. With 30 cases per class, one recall value has an interval of about ±18 points, so only large differences between classes or models are informative.

**Recall, text only** ✔

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

**Recall, image only** ✔

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

**Recall, image + text** ✔

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

Precision, F1 and specificity per class are in Appendix B; the confusion matrices are in Appendix A.

### 4.4 Invalid answers and refusals
An answer with no usable letter is wrong; one where the model declined is also counted as a refusal. Only GPT-4o refused. ✔

| Run | Invalid (all refusals) | Share of 240 | Accuracy as reported | Accuracy on the answered questions only |
|---|---|---|---|---|
| GPT-4o, Image only | 28 (28 refusals) | 11.7% | 36.2% (87 of 240) | 41.0% (87 of 212) |
| GPT-4o, Image + text | 12 (12 refusals) | 5.0% | 41.2% (99 of 240) | 43.4% (99 of 228) |

GPT-4o's text is always the same kind of sentence ("I'm sorry, I can't help with identifying or diagnosing images of skin lesions."). The last column is **not** a standard metric; it only shows how much of GPT-4o's lower score comes from refusing. The reported accuracy (refusals counted as wrong) is the one to use in comparisons. Even on the answered questions, GPT-4o's image-only accuracy (41.0%) stays below Claude's and Gemini's (47.1% and 50.8%).

### 4.5 Patient facts and source (secondary analysis)
Age, sex or site is missing for 28 of the 240 test cases (NV 15, MEL 7, DF 2, VASC 2, BCC 1, SCC 1), and "missing" hints at nevus. Accuracy by group, computed with `evaluate.py`'s `subgroup_accuracy` on the committed predictions (no intervals; n = 28 is small):

| Run | All three facts recorded (n = 212) | Some missing (n = 28) |
|---|---|---|
| Claude Sonnet 5.5, Text only | 21.7% | 53.6% |
| Gemini 3.8 Flash, Text only | 16.5% | 53.6% |
| GPT-4o, Text only | 22.2% | 3.6% |
| Llama 4 Maverick, Text only | 20.3% | 32.1% |
| Claude Sonnet 5.5, Image only | 47.2% | 46.4% |
| Gemini 3.8 Flash, Image only | 52.8% | 35.7% |
| GPT-4o, Image only | 34.4% | 50.0% |
| Llama 4 Maverick, Image only | 35.4% | 46.4% |
| Claude Sonnet 5.5, Image + text | 56.1% | 42.9% |
| Gemini 3.8 Flash, Image + text | 55.7% | 35.7% |
| GPT-4o, Image + text | 40.1% | 50.0% |
| Llama 4 Maverick, Image + text | 34.0% | 50.0% |

In the text-only view a model that answers "nevus" when facts are missing gains accuracy there (Claude and Gemini 53.6%) while the other cases stay low (Gemini 16.5%): with all three facts recorded, the four models lie between 16.5% and 22.2%. The same split by image source (a confound: every MEL and NV image comes from the older ISIC archive, so the groups differ in disease mix and are not comparable in size or difficulty):

| Run | HAM10000 (n = 175) | ISIC archive 2016-2017 era (n = 65) |
|---|---|---|
| Claude Sonnet 5.5, Text only | 19.4% | 41.5% |
| Gemini 3.8 Flash, Text only | 13.1% | 41.5% |
| GPT-4o, Text only | 23.4% | 10.8% |
| Llama 4 Maverick, Text only | 19.4% | 27.7% |
| Claude Sonnet 5.5, Image only | 50.3% | 38.5% |
| Gemini 3.8 Flash, Image only | 56.0% | 36.9% |
| GPT-4o, Image only | 36.0% | 36.9% |
| Llama 4 Maverick, Image only | 35.4% | 40.0% |
| Claude Sonnet 5.5, Image + text | 57.1% | 47.7% |
| Gemini 3.8 Flash, Image + text | 58.3% | 40.0% |
| GPT-4o, Image + text | 41.7% | 40.0% |
| Llama 4 Maverick, Image + text | 34.9% | 38.5% |

### 4.6 Which answers and letters are favoured
A balanced test set gives each class 12.5% of the right answers. The class a model names most often, and the most chosen letter (options are shuffled, so a letter has no meaning):

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

Letter B is the most chosen in 11 of 12 runs (14.6% to 22.5%; Llama text only chose C). It is a mild position habit and does not explain the low text-only scores.

### 4.7 McNemar's exact test with Holm correction
Only the questions where exactly one of two models is right count; Holm corrects for the six pairs in each view. The adjusted p is the one to read. ✔

**Text only**

| Pair | Only first right | Only second right | p | Holm-adjusted p |
|---|---|---|---|---|
| Claude (25.4%) vs GPT-4o (20.0%) | 27 | 14 | 0.0596 | **0.2979** |
| Claude (25.4%) vs Llama (21.7%) | 27 | 18 | 0.2327 | **0.9308** |
| Claude (25.4%) vs Gemini (20.8%) | 13 | 2 | 0.0074 | **0.0443** |
| GPT-4o (20.0%) vs Llama (21.7%) | 25 | 29 | 0.6835 | **1.0000** |
| GPT-4o (20.0%) vs Gemini (20.8%) | 20 | 22 | 0.8776 | **1.0000** |
| Llama (21.7%) vs Gemini (20.8%) | 24 | 22 | 0.8830 | **1.0000** |

**Image only**

| Pair | Only first right | Only second right | p | Holm-adjusted p |
|---|---|---|---|---|
| Claude (47.1%) vs GPT-4o (36.2%) | 51 | 25 | 0.0038 | **0.0142** |
| Claude (47.1%) vs Llama (36.7%) | 47 | 22 | 0.0035 | **0.0142** |
| Claude (47.1%) vs Gemini (50.8%) | 25 | 34 | 0.2976 | **0.5952** |
| GPT-4o (36.2%) vs Llama (36.7%) | 22 | 23 | 1.0000 | **1.0000** |
| GPT-4o (36.2%) vs Gemini (50.8%) | 23 | 58 | 0.0001 | **0.0008** |
| Llama (36.7%) vs Gemini (50.8%) | 24 | 58 | 0.0002 | **0.0011** |

**Image + text**

| Pair | Only first right | Only second right | p | Holm-adjusted p |
|---|---|---|---|---|
| Claude (54.6%) vs GPT-4o (41.2%) | 51 | 19 | 0.0002 | **0.0007** |
| Claude (54.6%) vs Llama (35.8%) | 62 | 17 | < 0.0001 | **< 0.0001** |
| Claude (54.6%) vs Gemini (53.3%) | 27 | 24 | 0.7798 | **0.7798** |
| GPT-4o (41.2%) vs Llama (35.8%) | 38 | 25 | 0.1299 | **0.2598** |
| GPT-4o (41.2%) vs Gemini (53.3%) | 25 | 54 | 0.0015 | **0.0044** |
| Llama (35.8%) vs Gemini (53.3%) | 24 | 66 | < 0.0001 | **< 0.0001** |

### 4.8 Semantic similarity and its validity check
The similarity is the cosine of sentence embeddings (`S-PubMedBert-MS-MARCO`) between the reasoning and the 40 reference texts of the true disease (best match). It measures resemblance to class-level medical text, not whether this case's reasoning is right. Answers without a usable letter are not scored (corrected 2026-11-24: GPT-4o's refusal sentences had been embedded).

| Model | Text only | Image only | Image + text |
|---|---|---|---|
| Claude Sonnet 5.5 | 0.9307 (n = 240) | 0.9495 (n = 240) | 0.9501 (n = 240) |
| Gemini 3.8 Flash | 0.9320 (n = 240) | 0.9481 (n = 240) | 0.9494 (n = 240) |
| GPT-4o | 0.9268 (n = 240) | 0.9439 (n = 212) | 0.9482 (n = 228) |
| Llama 4 Maverick | 0.9211 (n = 240) | 0.9498 (n = 240) | 0.9502 (n = 240) |

The scores are similar across models (differences of 0.006 or less in the image views), so **they must not be used to rank models**. They are higher in the image views (0.944 to 0.950) than in text only (0.921 to 0.932).

**Validity check** (pass mark set before the first run: the disease the model answered has the single highest similarity of the 8 in at least 50% of the answers, and in at least 25% for every answered class with 10 or more answers; chance is 12.5%) ✔

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

- In **every** run the answered disease is identified far above chance (39.2% to 91.0%). The score therefore carries information about which disease the reasoning argues for.
- The pass mark fails in two ways. (1) One answered class matches its reference texts poorly: BCC in all four text-only runs, and melanoma (only 2 reference texts) for Claude. (2) Claude's three runs (45.0%, 39.2%, 45.8%) and Llama's text-only run (47.9%) are also under the overall 50% bar. The first reason points to thin reference texts, not to bad reasoning.
- GPT-4o's high rates (91.0% and 83.8%) may reflect its habit of naming the disease in the textbook's own words ◻ (my reading, not tested): the metric partly measures vocabulary alignment.

### 4.9 Image claims in the text-only view
The share of text-only reasonings that describe an image or a dermoscopy although none was given (a word list; sentences with a denial or "if/would" are skipped). It is an approximate lower bound for one kind of hallucination. Result ✔: **0 of 240 for every model**. With the first prompt wording, one model had invented dermoscopic findings in a trial; the view-specific wording removed it.

### 4.10 Cost and speed
Tokens and seconds per run (the minutes are the sum of the calls; some runs ran at the same time, so the clock time was shorter). Prices are not included. ✔

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

The 12 runs used 1,926,836 tokens in and 471,833 tokens out. Gemini writes the most (hidden reasoning is part of its output) and is the slowest per question.

## 5. Comparing the views for each model (McNemar with Holm)
Change in accuracy and the adjusted p for the same 240 cases ✔ (`results/llm_only/compare_views_<model>.json`):

| Model | Text → image | Text → image + text | Image → image + text |
|---|---|---|---|
| Claude Sonnet 5.5 | +21.7 points (Holm p < 0.0001) | +29.2 points (Holm p < 0.0001) | +7.5 points (Holm p 0.0014) |
| Gemini 3.8 Flash | +30.0 points (Holm p < 0.0001) | +32.5 points (Holm p < 0.0001) | +2.5 points (Holm p 0.3915) |
| GPT-4o | +16.2 points (Holm p 0.0002) | +21.2 points (Holm p < 0.0001) | +5.0 points (Holm p 0.0961) |
| Llama 4 Maverick | +15.0 points (Holm p 0.0004) | +14.2 points (Holm p 0.0004) | -0.8 points (Holm p 0.8506) |

## 6. Interpretation

**What the results say.**
- In this benchmark the models use the image, not the patient facts, to diagnose: text alone gives 20% to 25%, the image 36% to 51%, and the image with the facts 36% to 55%.
- Claude and Gemini are the stronger pair in both image views; GPT-4o and Llama are weaker and indistinguishable from each other.
- The LLM-only baseline is therefore moderate with images (about half of the questions at best) and weak without them. These numbers are the baseline against which the RAG systems will be compared. They say nothing about retrieval.

**What the results do not say.**
- That any model is "better at dermatology" in general: one prompt, one dataset (balanced, 240 cases), one run per model, one host per model.
- That GPT-4o is worse than Llama or Gemini on the questions it answered: refusals mix two effects, and only the reported accuracy (refusals wrong) is a fair comparison.
- That 0 described images proves no hallucination: the other kinds are not measured, and the check is a word list.
- That the semantic similarity measures reasoning quality (section 4.8).
- Anything about a clinic: the splits are balanced, so precision is not a clinic's precision, and no clinician judged any answer.

**Limits.** Claude's sampling is not fixed; the test set has 190 lesions; images are standardised (600×450, colour constancy), not the originals; the source confound (section 4.5) remains; the intervals are bootstrap intervals over lesions.

## 7. Open points
- Standard RAG, RAG² and the proposed system are not run (`results/STATUS.md`); their results will be added in the same form.
- The semantic similarity needs reference texts for the classes that fail the check (melanoma, BCC) before it is used to compare systems.
- GPT-4o's refusals could be reported per class, to see which diseases it declines (not done).

## 8. Files and how to reproduce

| What | Where |
|---|---|
| Predictions, settings, report of each run | `results/llm_only/<model>/<view>_test/{predictions.jsonl, run.json, report.json}` |
| One row per run | `results/summary.csv` |
| Pairwise tests between models | `results/llm_only/compare_<view>_test.json` |
| Pairwise tests between views | `results/llm_only/compare_views_<model>.json` |
| The semantic-similarity validity check | `results/similarity_check.json` |

```powershell
python src/run.py llm-only --model gpt-4o --view image_only --split test      # one run (paid; resumable)
python src/evaluate.py collect                                                # report.json for every run and results/summary.csv
python src/evaluate.py similarity-check                                       # the validity check
python src/evaluate.py compare --out results/llm_only/compare_image_only_test.json --run claude=data/views/image_only/test.jsonl,results/llm_only/claude-sonnet-5-5/image_only_test/predictions.jsonl --run gemini=...
```

## Appendix A: confusion matrices (rows: true class; columns: class answered; the last column counts invalid answers)

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

## Sources
- McNemar (1947) Psychometrika 12:153-157 ◻; Holm (1979) Scandinavian Journal of Statistics 6:65-70 ◻; bootstrap: Efron and Tibshirani (1993) ◻.
- Sentence embeddings: Reimers and Gurevych (2019) EMNLP ◻; model `pritamdeka/S-PubMedBert-MS-MARCO` (Hugging Face).
- Reference texts: Dermoscopedia, CC BY-NC-SA 4.0 (`data/knowledge_corpus/reference_texts.json`).
- Metric definitions and further sources: `results/evaluation_guide.md`.
