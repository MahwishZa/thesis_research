# Closed-book results

Four LLMs answered the test split of three datasets without retrieval (closed-book). Each question has eight possible diagnoses and the model must name exactly one (forced choice). There is one run per model and dataset. Every number on this page comes from the saved answers in `results/`; no number is typed by hand.

- **Image:** 320 photos, 40 per diagnosis ([data sheet](../data/image/datasheet.md)).
- **Text:** 221 written cases, unequal across diagnoses ([data sheet](../data/text/datasheet.md)).
- **Multimodal:** the image dataset's 320 photos, each with the patient's age, sex and body site ([data sheet](../data/image_metadata/datasheet.md)).

The image and multimodal datasets use the same photos, so they are compared question by question. The text dataset describes different patients, so its scores are not compared with the other two (section 4).

## 1. Metric definitions

| Metric | Meaning |
|---|---|
| Accuracy | The share of questions answered with the right diagnosis. |
| Macro Recall (balanced accuracy) | The mean recall over the eight diagnoses, each counting the same. It is the main score and is also called balanced accuracy, the name used in the text below and in the reports. Chance is 12.5%. |
| Precision, recall, F1 | Per diagnosis: of the answers naming it, the share that are right (precision); of its questions, the share answered with it (recall); their harmonic mean (F1). "Macro" is the plain mean over the eight diagnoses. A diagnosis a model never names has no precision; it counts as 0 in the macro precision. |
| Specificity | Per diagnosis: of the questions that are not that diagnosis, the share not answered with it. With eight diagnoses, most questions are negatives of any one diagnosis, so specificity is high by construction. Read it beside recall, never alone. |
| Hallucination rate | The share of questions answered with a wrong diagnosis, averaged over the diagnoses. An answer that is not one diagnosis name ("no valid answer") counts as wrong but is not a hallucination. |
| Malignancy sensitivity | Of the cancers (melanoma, basal cell carcinoma, squamous cell carcinoma), the share answered as any cancer. |
| Missed malignancy | Of the cancers, the share answered as a benign diagnosis. It is not 100% minus the sensitivity: a cancer answered as actinic keratosis (precancerous) or without a valid answer is neither detected nor missed. |
| Benign specificity, over-call | Of the benign lesions (nevus, pigmented benign keratosis, dermatofibroma, vascular lesion), the share answered as any benign diagnosis, and the share answered as a cancer. |
| 95% interval | A bootstrap interval that resamples whole lesions (image, multimodal) or whole articles (text) within each diagnosis, 2,000 times with a fixed seed. |
| McNemar's exact test | For two runs on the same questions: whether the questions only one run answers right split more unevenly than chance would. |
| Holm correction | This page holds 22 significance tests. The Holm p-value corrects each for that number, so that a chance finding among many tests is not read as real. |

**Not available: ROC curve and AUC.** They need a score for every diagnosis on every question. Each model returned one diagnosis name and no scores, so there is nothing to rank. Scores would need a new experiment with a different prompt, which was not run. Malignancy sensitivity together with benign specificity is the one point of such a curve that these answers give.

**Not applicable yet:** abstention metrics (the runs that may answer "uncertain" are prepared in the `-abstain` configs and were not run) and retrieval metrics (no retrieval was used).

## 2. Results

The tables in this section are written by `python -m src.evaluate summary`.

<!-- tables: start -->

### Scores

Percentages. Macro recall (balanced accuracy) is the main score, with its 95% interval. Each model name links to the run's report: every score with its interval, the per-class table and the confusion matrix.

**Image**

| Model | Accuracy | Macro Recall (95% CI) | Macro Precision | Macro F1 | Macro Specificity | Hallucination Rate | No Valid Answer |
|---|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](../results/image/claude-sonnet-5-5/test_cf5351f1/report.md) | 51.2 | 51.2 (45.7 to 56.7) | 50.8 | 49.5 | 93.0 | 48.8 | 0 of 320 |
| [GPT-4o](../results/image/gpt-4o/test_14e3059f/report.md) | 34.1 | 34.1 (29.6 to 38.7) | 36.4 | 30.1 | 90.6 | 65.9 | 0 of 320 |
| [Gemini 3.8 Flash](../results/image/gemini-3.8-flash/test_b086a9a1/report.md) | 49.7 | 49.7 (44.3 to 55.0) | 46.6 | 46.2 | 92.8 | 50.3 | 0 of 320 |
| [Llama 4 Maverick](../results/image/llama-4-maverick/test_055451fe/report.md) | 28.7 | 28.7 (24.4 to 33.1) | 39.1 | 25.7 | 89.8 | 71.2 | 0 of 320 |

**Text**

| Model | Accuracy | Macro Recall (95% CI) | Macro Precision | Macro F1 | Macro Specificity | Hallucination Rate | No Valid Answer |
|---|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](../results/text/claude-sonnet-5-5/test_06ce1664/report.md) | 64.3 | 55.9 (49.6 to 61.8) | 63.9 | 57.9 | 95.3 | 34.1 | 10 of 221 |
| [GPT-4o](../results/text/gpt-4o/test_5fc7ce07/report.md) | 56.6 | 50.8 (45.1 to 56.6) | 54.3 | 49.4 | 93.6 | 49.2 | 0 of 221 |
| [Gemini 3.8 Flash](../results/text/gemini-3.8-flash/test_6ee1c09b/report.md) | 69.2 | 61.4 (55.8 to 67.0) | 65.1 | 61.3 | 95.4 | 38.6 | 0 of 221 |
| [Llama 4 Maverick](../results/text/llama-4-maverick/test_32905170/report.md) | 53.8 | 46.8 (40.6 to 52.8) | 53.8 | 47.0 | 93.9 | 48.5 | 11 of 221 |

**Multimodal**

| Model | Accuracy | Macro Recall (95% CI) | Macro Precision | Macro F1 | Macro Specificity | Hallucination Rate | No Valid Answer |
|---|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](../results/multimodal/claude-sonnet-5-5/test_5db7a077/report.md) | 56.2 | 56.2 (50.6 to 61.7) | 58.3 | 55.7 | 93.8 | 43.8 | 0 of 320 |
| [GPT-4o](../results/multimodal/gpt-4o/test_12259e56/report.md) | 41.6 | 41.6 (36.4 to 46.7) | 49.7 | 40.6 | 91.7 | 58.4 | 0 of 320 |
| [Gemini 3.8 Flash](../results/multimodal/gemini-3.8-flash/test_2ff4fbdd/report.md) | 54.4 | 54.4 (48.6 to 59.9) | 58.9 | 53.1 | 93.5 | 45.6 | 0 of 320 |
| [Llama 4 Maverick](../results/multimodal/llama-4-maverick/test_c3b17116/report.md) | 34.1 | 34.1 (29.0 to 39.0) | 47.2 | 33.9 | 90.6 | 65.9 | 0 of 320 |

### Clinical risk

Percentages with 95% intervals. Sensitivity and missed malignancy are shares of the cancers; benign specificity and over-call are shares of the benign lesions.

**Image**

| Model | Malignancy Sensitivity | Missed Malignancy | Benign Specificity | Over-call |
|---|---|---|---|---|
| Claude Sonnet 5.5 | 50.8 (41.4 to 60.2) | 25.0 (17.4 to 33.1) | 74.4 (66.7 to 82.2) | 21.9 (14.5 to 29.4) |
| GPT-4o | 77.5 (69.4 to 86.0) | 22.5 (14.0 to 30.6) | 55.6 (47.6 to 64.0) | 44.4 (36.0 to 52.4) |
| Gemini 3.8 Flash | 57.5 (48.8 to 66.7) | 21.7 (14.4 to 29.3) | 78.8 (71.4 to 85.4) | 16.2 (10.2 to 22.6) |
| Llama 4 Maverick | 78.3 (70.2 to 86.5) | 19.2 (11.5 to 27.1) | 45.6 (37.6 to 53.9) | 53.8 (45.2 to 61.9) |

**Text**

| Model | Malignancy Sensitivity | Missed Malignancy | Benign Specificity | Over-call |
|---|---|---|---|---|
| Claude Sonnet 5.5 | 90.8 (85.8 to 95.8) | 5.8 (1.7 to 10.0) | 66.7 (57.6 to 75.8) | 27.3 (18.8 to 36.4) |
| GPT-4o | 83.3 (76.7 to 90.0) | 14.2 (8.3 to 20.8) | 74.7 (66.0 to 83.2) | 25.3 (16.8 to 34.0) |
| Gemini 3.8 Flash | 93.3 (89.2 to 97.5) | 5.0 (1.7 to 9.2) | 68.7 (59.6 to 77.3) | 31.3 (22.7 to 40.4) |
| Llama 4 Maverick | 84.2 (77.5 to 90.8) | 10.0 (5.0 to 15.8) | 69.7 (59.8 to 78.2) | 24.2 (16.5 to 33.0) |

**Multimodal**

| Model | Malignancy Sensitivity | Missed Malignancy | Benign Specificity | Over-call |
|---|---|---|---|---|
| Claude Sonnet 5.5 | 62.5 (52.6 to 71.8) | 20.8 (13.6 to 28.3) | 74.4 (66.7 to 82.1) | 23.1 (15.7 to 30.6) |
| GPT-4o | 79.2 (71.1 to 86.4) | 19.2 (12.4 to 26.8) | 55.6 (47.4 to 64.3) | 43.1 (34.3 to 51.6) |
| Gemini 3.8 Flash | 65.8 (56.1 to 74.8) | 21.7 (14.1 to 29.9) | 79.4 (72.1 to 86.6) | 19.4 (12.5 to 26.8) |
| Llama 4 Maverick | 80.0 (71.0 to 87.9) | 10.8 (5.0 to 17.9) | 46.9 (38.1 to 54.9) | 52.5 (44.5 to 61.4) |

### Recall per class

Percentages: of the questions of a class, the share answered with it. Precision, F1 and specificity per class are in each report.

**Image**

| Class | Test questions | Claude Sonnet 5.5 | GPT-4o | Gemini 3.8 Flash | Llama 4 Maverick |
|---|---|---|---|---|---|
| actinic keratosis | 40 | 45.0 | 0.0 | 40.0 | 0.0 |
| basal cell carcinoma | 40 | 65.0 | 40.0 | 77.5 | 10.0 |
| dermatofibroma | 40 | 67.5 | 65.0 | 62.5 | 40.0 |
| melanoma | 40 | 42.5 | 77.5 | 30.0 | 85.0 |
| nevus | 40 | 60.0 | 25.0 | 30.0 | 22.5 |
| pigmented benign keratosis | 40 | 37.5 | 25.0 | 65.0 | 10.0 |
| squamous cell carcinoma | 40 | 5.0 | 0.0 | 0.0 | 5.0 |
| vascular lesion | 40 | 87.5 | 40.0 | 92.5 | 57.5 |

**Text**

| Class | Test questions | Claude Sonnet 5.5 | GPT-4o | Gemini 3.8 Flash | Llama 4 Maverick |
|---|---|---|---|---|---|
| actinic keratosis | 2 | 0.0 | 0.0 | 0.0 | 0.0 |
| basal cell carcinoma | 40 | 65.0 | 40.0 | 65.0 | 40.0 |
| dermatofibroma | 21 | 57.1 | 52.4 | 61.9 | 28.6 |
| melanoma | 40 | 70.0 | 60.0 | 90.0 | 57.5 |
| nevus | 40 | 62.5 | 55.0 | 47.5 | 52.5 |
| pigmented benign keratosis | 22 | 36.4 | 36.4 | 68.2 | 40.9 |
| squamous cell carcinoma | 40 | 75.0 | 75.0 | 77.5 | 80.0 |
| vascular lesion | 16 | 81.2 | 87.5 | 81.2 | 75.0 |

**Multimodal**

| Class | Test questions | Claude Sonnet 5.5 | GPT-4o | Gemini 3.8 Flash | Llama 4 Maverick |
|---|---|---|---|---|---|
| actinic keratosis | 40 | 52.5 | 20.0 | 40.0 | 20.0 |
| basal cell carcinoma | 40 | 65.0 | 52.5 | 80.0 | 15.0 |
| dermatofibroma | 40 | 75.0 | 72.5 | 72.5 | 47.5 |
| melanoma | 40 | 50.0 | 70.0 | 35.0 | 82.5 |
| nevus | 40 | 57.5 | 32.5 | 42.5 | 30.0 |
| pigmented benign keratosis | 40 | 45.0 | 37.5 | 62.5 | 17.5 |
| squamous cell carcinoma | 40 | 15.0 | 0.0 | 15.0 | 5.0 |
| vascular lesion | 40 | 90.0 | 47.5 | 87.5 | 55.0 |

### The same photos without and with the patient's facts

Balanced accuracy in percent; the difference is multimodal minus image in percentage points, with a paired 95% interval. p: McNemar's exact test; Holm p: the same, corrected for all the tests of this page.

| Model | Image | Multimodal | Difference (95% CI) | p | Holm p | Wrong answers corrected / introduced |
|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 51.2 | 56.2 | +5.0 (+0.8 to +9.1) | 0.017 | 0.119 | 28 / 12 |
| GPT-4o | 34.1 | 41.6 | +7.5 (+3.7 to +11.4) | <0.001 | 0.003 | 33 / 9 |
| Gemini 3.8 Flash | 49.7 | 54.4 | +4.7 (+0.0 to +9.2) | 0.053 | 0.278 | 34 / 19 |
| Llama 4 Maverick | 28.7 | 34.1 | +5.3 (+1.6 to +9.3) | 0.012 | 0.115 | 29 / 12 |

### Model against model

The difference in balanced accuracy, second model minus first, in percentage points with a paired 95% interval, on the same questions. p: McNemar's exact test on right against wrong answers; Holm p: corrected for all the tests of this page (22). A difference counts as significant when its Holm p is below 0.05.

**Image**

| Second minus first | Difference (95% CI) | p | Holm p |
|---|---|---|---|
| GPT-4o − Claude Sonnet 5.5 | -17.2 (-23.1 to -11.5) | <0.001 | <0.001 |
| Gemini 3.8 Flash − Claude Sonnet 5.5 | -1.6 (-6.8 to +3.8) | 0.644 | 1.000 |
| Llama 4 Maverick − Claude Sonnet 5.5 | -22.5 (-28.8 to -16.3) | <0.001 | <0.001 |
| Gemini 3.8 Flash − GPT-4o | +15.6 (+10.0 to +21.4) | <0.001 | <0.001 |
| Llama 4 Maverick − GPT-4o | -5.3 (-10.1 to -0.6) | 0.046 | 0.278 |
| Llama 4 Maverick − Gemini 3.8 Flash | -20.9 (-27.1 to -15.0) | <0.001 | <0.001 |

**Text**

| Second minus first | Difference (95% CI) | p | Holm p |
|---|---|---|---|
| GPT-4o − Claude Sonnet 5.5 | -5.1 (-10.3 to +0.1) | 0.012 | 0.115 |
| Gemini 3.8 Flash − Claude Sonnet 5.5 | +5.5 (+0.4 to +11.0) | 0.126 | 0.505 |
| Llama 4 Maverick − Claude Sonnet 5.5 | -9.1 (-15.0 to -3.2) | 0.003 | 0.029 |
| Gemini 3.8 Flash − GPT-4o | +10.6 (+5.4 to +16.2) | <0.001 | 0.002 |
| Llama 4 Maverick − GPT-4o | -4.0 (-9.7 to +1.3) | 0.418 | 1.000 |
| Llama 4 Maverick − Gemini 3.8 Flash | -14.6 (-21.2 to -8.4) | <0.001 | <0.001 |

**Multimodal**

| Second minus first | Difference (95% CI) | p | Holm p |
|---|---|---|---|
| GPT-4o − Claude Sonnet 5.5 | -14.7 (-20.7 to -8.3) | <0.001 | <0.001 |
| Gemini 3.8 Flash − Claude Sonnet 5.5 | -1.9 (-7.1 to +3.2) | 0.550 | 1.000 |
| Llama 4 Maverick − Claude Sonnet 5.5 | -22.2 (-28.5 to -15.9) | <0.001 | <0.001 |
| Gemini 3.8 Flash − GPT-4o | +12.8 (+5.8 to +19.6) | <0.001 | 0.003 |
| Llama 4 Maverick − GPT-4o | -7.5 (-13.6 to -1.6) | 0.015 | 0.119 |
| Llama 4 Maverick − Gemini 3.8 Flash | -20.3 (-26.6 to -14.0) | <0.001 | <0.001 |

<!-- tables: end -->

Reference points from the data sheets: chance is 12.5%; the patient's facts alone, with no photo, reach 35.9% (30.0 to 41.5) on the multimodal test set; a classifier that sees only the colour of the image border reaches 34.7% on the image test set.

## 3. Statistical findings

- **Claude Sonnet 5.5 and Gemini 3.8 Flash are not distinguishable** on any dataset: their differences are -1.6 (image), +5.5 (text) and -1.9 (multimodal) points, none significant.
- **Both are significantly better than GPT-4o and Llama 4 Maverick on the photos,** with and without the facts: differences of 12.8 to 22.5 points, all with a Holm p below 0.01.
- **On text the picture is weaker.** Gemini is significantly better than GPT-4o and Maverick, and Claude than Maverick. Claude against GPT-4o (5.1 points) is not significant after the correction.
- **GPT-4o against Llama 4 Maverick** is not significant on any dataset after the correction.
- **The patient's facts raise every model's score on the same photos,** by 4.7 to 7.5 points. Before the correction three of the four gains are significant (Gemini: p = 0.053). After the Holm correction only GPT-4o's gain is (p = 0.003). The four gains agree in direction and size, which a single test does not capture, but each on its own is modest.

McNemar's test counts right against wrong answers, so on the text dataset, whose diagnoses are unequal in number, it can disagree with the difference in balanced accuracy.

## 4. Interpretation

- **The level is low for every model.** The best balanced accuracy is 51% on photos and 61% on text. Without retrieval, no model is a reliable diagnostic aid on these questions.
- **GPT-4o and Llama 4 Maverick lean on one answer.** On the image dataset, Maverick answers melanoma for 62% of the photos and GPT-4o for 46% (Claude 16%, Gemini 10%). This gives them a high malignancy sensitivity (78% for both) and a high over-call (54% and 44%) together: they do not detect cancers better, they call most lesions cancer. Their balanced accuracy on photos (29% and 34%) is at or below what the border colour alone reaches, so their reading of the photos is not shown to go beyond cues that are not the lesion.
- **Claude and Gemini fail on cancers in a different way.** On photos they answer 24% and 21% of the cancers as actinic keratosis, which is precancerous: neither a detection nor a miss. Their malignancy sensitivity is therefore only 51% and 58%, although their missed malignancy (25% and 22%) is similar to the other models'. Missed malignancy alone hides this.
- **Squamous cell carcinoma is almost never recognised from a photo** (0% to 5% recall for all four models, 15% at most with the facts) but is recognised in 75% to 80% of the written cases.
- **Text scores are higher than photo scores, and this is not a like-for-like comparison.** A text question is a clinician's written case (25 to 443 words, median 87); a multimodal question is a photo with about ten words of facts and describes a different patient. No question holds both a written case and its photo. The supported multimodal comparison is the same photos without and with the facts (section 2). Claude's text score (55.9%) is close to its multimodal score (56.2%) only because of actinic keratosis, which has 2 text test cases, none recognised; without that class its text score is 63.9%.
- **The facts help, modestly.** Part of the gain may come from the facts alone, which reach 35.9% without any photo. Claude (56.2%) and Gemini (54.4%) are well above that level on the multimodal questions; GPT-4o (41.6%) is at the upper end of that interval and Maverick (34.1%) is not above it. Scoring only the 289 questions with all three facts changes no score by more than 1.5 points (each multimodal report).
- **In this forced-choice condition the hallucination rate is the wrong-answer rate.** A model cannot decline, so every wrong answer names a wrong diagnosis. Whether a method reduces hallucination, and not only errors, can be shown only with the abstention condition or with retrieval.
- **Answers that were not one diagnosis name** occurred only on text: Claude 10 of 221 and Maverick 11 of 221. Reading a diagnosis name alone on the last line of such an answer, as a sensitivity analysis, moves Claude from 55.9% to 59.1% and Maverick from 46.8% to 47.1% (each text report). The strict reading stays the main result: it was fixed before any result was seen and applies to every model alike.

## 5. Limits

- **One run per model and dataset.** Run-to-run variation was not measured. Claude cannot be seeded, and the other models' seeds are best effort.
- **The models were not set up identically** (section 6): the answer limit differs, Claude receives no temperature, and Gemini spends hidden reasoning tokens. The prompt, the answer reader and the data are the same for all four. The models are also of different generations.
- **Small groups.** A single diagnosis has 40 test photos, so its recall has an interval of roughly ±15 points. The text test set has 2 actinic keratosis cases, which count for one eighth of its balanced accuracy; each text report also gives the scores without that class.
- **Possible exposure.** The text cases are published reports, so a model may have seen some of them (limitation 4 of the text data sheet).
- **Labels.** 74% of the photo diagnoses are confirmed by biopsy; about 2.7% of the text case labels are estimated to be wrong (data sheets).
- **Not clinical performance.** The test sets come from public archives and published reports, and their mix of diagnoses is not a clinic's. Precision in particular depends on that mix.

## 6. How each model was run

| Model | Model ID and Route | Answer Limit | Other Settings |
|---|---|---|---|
| Claude Sonnet 5.5 | `claude-sonnet-5-5`, Anthropic API | 256 | no temperature (the model rejects it); thinking between tools; effort low |
| GPT-4o | `openai/gpt-4o-2024-08-06`, OpenRouter, pinned to OpenAI | 64 | temperature 0, seed 0, image detail high |
| Gemini 3.8 Flash | `google/gemini-3.8-flash`, OpenRouter, pinned to Google AI Studio | 2048 | temperature 0, seed 0, reasoning effort minimal (hidden reasoning still used a median of 138 to 192 output tokens per answer and at most 1,969 of the 2,048 allowed; no answer was cut off) |
| Llama 4 Maverick | `meta-llama/llama-4-maverick`, OpenRouter, pinned to Parasail fp8 | 256 | temperature 0, seed 0; 8-bit (fp8) precision |

The exact settings are in `configs/<dataset>/<model>.toml`. Each run folder's `run_card.json` records the config, the prompt, the dataset fingerprint and the Git commit the run used. The prompts are `prompts/<dataset>/diagnosis.txt`, the same for every model. Llama 3.2 11B Vision, the first choice, is no longer served by any host, so Llama 4 Maverick replaced it.

## 7. Reproducing this page

No model call is needed. Every report and every table is rebuilt from the saved answers:

```
python -m src.evaluate report results/<dataset>/<model>/<run folder>
python -m src.evaluate summary
```
