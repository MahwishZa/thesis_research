> **Version 1** (commit `a1c9209`), archived on 2026-11-19: a frozen copy kept for reference. It does not bind Version 2 (see `docs/v2_proposal.md` in the repository).

# Closed-book results: first phase

Four LLMs answered the test split of three datasets without retrieval (closed-book). Each question has eight possible diagnoses and the model had to name exactly one (forced choice, no way to say it was unsure). There is one run per model and dataset. Every number on this page comes from the saved answers in `results/phase1/`; no number is typed by hand.

- **Image:** 320 photos, 40 per diagnosis ([data sheet](../data/image/datasheet.md)).
- **Text:** 221 written cases, unequal across diagnoses: version 2 of the text dataset ([data sheet](../data/text/datasheet.md); version 3, used from now on, takes out 12 of these test cases).
- **Image + metadata:** the image dataset's 320 photos, each with the patient's age, sex and body site ([data sheet](../data/image_metadata/datasheet.md); called "multimodal" in the run cards of these runs).

The image and image + metadata datasets use the same photos, so they are compared question by question. The text dataset describes different patients, so its scores are not compared with the other two (section 4).

These runs used the first answer format: one diagnosis name alone. From now on every run answers in the JSON format of [method.md](method.md), with a differential of three and an evidence flag, and the first-phase models are run again in that format. So this page is the baseline of the old format, not the closed-book baseline of the thesis.

## 1. Scores

The definitions and their sources are in [method.md](method.md). In short:

- **Balanced accuracy** (main score): the mean recall over the eight diagnoses, each counting the same. Chance is 12.5%.
- **Macro-F1** (second score): the mean F1 over the eight diagnoses. A diagnosis a model never names counts as 0.
- **Recall per class**, and in each run's scores also precision, F1 and the confusion matrix.
- **Invalid answers**: answers that are not one diagnosis name. They count as wrong.
- **95% intervals**: a bootstrap that resamples whole lesions (image, image + metadata) or whole articles (text) within each diagnosis, 2,000 times with a fixed seed.
- **Comparisons**: the paired 95% interval of the difference in balanced accuracy is the main comparison; McNemar's exact test, with the Holm correction for the 22 tests of this page, supports it.

Top-3 balanced accuracy, misdiagnosis detection, coverage and selective risk need a differential and an evidence flag, which these answers do not have. Plain accuracy is not reported (method.md, section 3).

Each run's archived `report.md` and `metrics.json` are unchanged. The scores above are written beside them in `scores.md` and `scores.json`; the balanced accuracy and its interval are the same as in the archived files.

## 2. Results

The tables in this section are written by `python -m src.evaluate summary`.

<!-- tables: start -->

### Scores

Percentages with 95% intervals. Balanced accuracy is the main score. Each model name links to the run's scores: every score with its interval, the per-class table and the confusion matrix.

**Image**

| Model | Balanced Accuracy (95% CI) | Macro-F1 (95% CI) | Invalid Answers |
|---|---|---|---|
| [Claude Sonnet 5.5](../results/phase1/image/claude-sonnet-5-5/test_cf5351f1/scores.md) | 51.2 (45.7 to 56.7) | 49.5 (44.0 to 54.7) | 0 of 320 |
| [GPT-4o](../results/phase1/image/gpt-4o/test_14e3059f/scores.md) | 34.1 (29.6 to 38.7) | 30.1 (25.5 to 34.5) | 0 of 320 |
| [Gemini 3.8 Flash](../results/phase1/image/gemini-3.8-flash/test_b086a9a1/scores.md) | 49.7 (44.3 to 55.0) | 46.2 (41.1 to 51.2) | 0 of 320 |
| [Llama 4 Maverick](../results/phase1/image/llama-4-maverick/test_055451fe/scores.md) | 28.7 (24.4 to 33.1) | 25.7 (20.8 to 30.5) | 0 of 320 |

**Text**

| Model | Balanced Accuracy (95% CI) | Macro-F1 (95% CI) | Invalid Answers |
|---|---|---|---|
| [Claude Sonnet 5.5](../results/phase1/text/claude-sonnet-5-5/test_06ce1664/scores.md) | 55.9 (49.6 to 61.8) | 57.9 (51.4 to 63.5) | 10 of 221 |
| [GPT-4o](../results/phase1/text/gpt-4o/test_5fc7ce07/scores.md) | 50.8 (45.1 to 56.6) | 49.4 (43.4 to 55.2) | 0 of 221 |
| [Gemini 3.8 Flash](../results/phase1/text/gemini-3.8-flash/test_6ee1c09b/scores.md) | 61.4 (55.8 to 67.0) | 61.3 (55.8 to 66.7) | 0 of 221 |
| [Llama 4 Maverick](../results/phase1/text/llama-4-maverick/test_32905170/scores.md) | 46.8 (40.6 to 52.8) | 47.0 (40.0 to 53.0) | 11 of 221 |

**Image + metadata**

| Model | Balanced Accuracy (95% CI) | Macro-F1 (95% CI) | Invalid Answers |
|---|---|---|---|
| [Claude Sonnet 5.5](../results/phase1/image_metadata/claude-sonnet-5-5/test_5db7a077/scores.md) | 56.2 (50.6 to 61.7) | 55.7 (50.0 to 61.1) | 0 of 320 |
| [GPT-4o](../results/phase1/image_metadata/gpt-4o/test_12259e56/scores.md) | 41.6 (36.4 to 46.7) | 40.6 (35.1 to 45.6) | 0 of 320 |
| [Gemini 3.8 Flash](../results/phase1/image_metadata/gemini-3.8-flash/test_2ff4fbdd/scores.md) | 54.4 (48.6 to 59.9) | 53.1 (47.2 to 58.6) | 0 of 320 |
| [Llama 4 Maverick](../results/phase1/image_metadata/llama-4-maverick/test_c3b17116/scores.md) | 34.1 (29.0 to 39.0) | 33.9 (27.9 to 39.2) | 0 of 320 |

### Recall per class

Percentages: of the questions of a class, the share answered with it. Precision and F1 per class are in each run's scores.

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

**Image + metadata**

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

### Inside each photo collection

Balanced accuracy inside each group of classes that comes from one ISIC collection, in percent, and in brackets the share of the group's questions answered with a class of the same group. Chance inside the groups: 50.0 for melanoma and nevus, 16.7 for the six others.

**Image**

| Model | Melanoma and nevus | The six other classes |
|---|---|---|
| Claude Sonnet 5.5 | 51.2 (75.0) | 51.3 (83.8) |
| GPT-4o | 51.2 (87.5) | 28.3 (58.8) |
| Gemini 3.8 Flash | 30.0 (41.2) | 56.2 (92.5) |
| Llama 4 Maverick | 53.8 (92.5) | 20.4 (32.5) |

**Image + metadata**

| Model | Melanoma and nevus | The six other classes |
|---|---|---|
| Claude Sonnet 5.5 | 53.8 (77.5) | 57.1 (86.2) |
| GPT-4o | 51.2 (85.0) | 38.3 (62.1) |
| Gemini 3.8 Flash | 38.7 (48.8) | 59.6 (87.1) |
| Llama 4 Maverick | 56.2 (88.8) | 26.7 (41.7) |

### The same photos without and with the patient's facts

Balanced accuracy in percent; the difference is image + metadata minus image in percentage points, with a paired 95% interval (the main comparison). p: McNemar's exact test; Holm p: the same, corrected for all the tests of this page.

| Model | Image | Image + metadata | Difference (95% CI) | p | Holm p |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 51.2 | 56.2 | +5.0 (+0.8 to +9.1) | 0.017 | 0.119 |
| GPT-4o | 34.1 | 41.6 | +7.5 (+3.7 to +11.4) | < 0.001 | 0.003 |
| Gemini 3.8 Flash | 49.7 | 54.4 | +4.7 (+0.0 to +9.2) | 0.053 | 0.278 |
| Llama 4 Maverick | 28.7 | 34.1 | +5.3 (+1.6 to +9.3) | 0.012 | 0.115 |

### Model against model

The difference in balanced accuracy, second model minus first, in percentage points with a paired 95% interval, on the same questions: the main comparison. p: McNemar's exact test on right against wrong answers; Holm p: corrected for all the tests of this page (22).

**Image**

| Second minus first | Difference (95% CI) | p | Holm p |
|---|---|---|---|
| GPT-4o − Claude Sonnet 5.5 | -17.2 (-23.1 to -11.5) | < 0.001 | < 0.001 |
| Gemini 3.8 Flash − Claude Sonnet 5.5 | -1.6 (-6.8 to +3.8) | 0.644 | 1.000 |
| Llama 4 Maverick − Claude Sonnet 5.5 | -22.5 (-28.8 to -16.3) | < 0.001 | < 0.001 |
| Gemini 3.8 Flash − GPT-4o | +15.6 (+10.0 to +21.4) | < 0.001 | < 0.001 |
| Llama 4 Maverick − GPT-4o | -5.3 (-10.1 to -0.6) | 0.046 | 0.278 |
| Llama 4 Maverick − Gemini 3.8 Flash | -20.9 (-27.1 to -15.0) | < 0.001 | < 0.001 |

**Text**

| Second minus first | Difference (95% CI) | p | Holm p |
|---|---|---|---|
| GPT-4o − Claude Sonnet 5.5 | -5.1 (-10.3 to +0.1) | 0.012 | 0.115 |
| Gemini 3.8 Flash − Claude Sonnet 5.5 | +5.5 (+0.4 to +11.0) | 0.126 | 0.505 |
| Llama 4 Maverick − Claude Sonnet 5.5 | -9.1 (-15.0 to -3.2) | 0.003 | 0.029 |
| Gemini 3.8 Flash − GPT-4o | +10.6 (+5.4 to +16.2) | < 0.001 | 0.002 |
| Llama 4 Maverick − GPT-4o | -4.0 (-9.7 to +1.3) | 0.418 | 1.000 |
| Llama 4 Maverick − Gemini 3.8 Flash | -14.6 (-21.2 to -8.4) | < 0.001 | < 0.001 |

**Image + metadata**

| Second minus first | Difference (95% CI) | p | Holm p |
|---|---|---|---|
| GPT-4o − Claude Sonnet 5.5 | -14.7 (-20.7 to -8.3) | < 0.001 | < 0.001 |
| Gemini 3.8 Flash − Claude Sonnet 5.5 | -1.9 (-7.1 to +3.2) | 0.550 | 1.000 |
| Llama 4 Maverick − Claude Sonnet 5.5 | -22.2 (-28.5 to -15.9) | < 0.001 | < 0.001 |
| Gemini 3.8 Flash − GPT-4o | +12.8 (+5.8 to +19.6) | < 0.001 | 0.003 |
| Llama 4 Maverick − GPT-4o | -7.5 (-13.6 to -1.6) | 0.015 | 0.119 |
| Llama 4 Maverick − Gemini 3.8 Flash | -20.3 (-26.6 to -14.0) | < 0.001 | < 0.001 |

<!-- tables: end -->

Reference points from the data sheets: chance is 12.5%; the patient's facts alone, with no photo, reach 35.9% (30.0 to 41.5) on the image + metadata test set; a classifier that sees only the colour of the image border reaches 34.7% on the image test set.

## 3. Statistical findings

A difference is called **established** when its paired interval excludes 0 and its Holm-corrected p-value is below 0.05; **suggestive** when only the interval excludes 0; and **not shown** when the interval holds 0.

- **Claude Sonnet 5.5 and Gemini 3.8 Flash are not distinguishable on the photos:** -1.6 (image) and -1.9 (image + metadata) points, both intervals holding 0. On text Gemini is ahead by 5.5 points (+0.4 to +11.0): suggestive only (Holm p = 0.505).
- **Both are better than GPT-4o and Llama 4 Maverick on the photos,** with and without the facts: differences of 12.8 to 22.5 points, all established.
- **On text the picture is weaker.** Gemini is better than GPT-4o (+10.6) and Maverick (+14.6), and Claude than Maverick (+9.1), all established. Claude against GPT-4o (5.1 points, -10.3 to +0.1) is not shown.
- **GPT-4o is ahead of Llama 4 Maverick on the photos** by 5.3 (image) and 7.5 (image + metadata) points: suggestive only (Holm p 0.278 and 0.119). On text the difference is not shown.
- **The patient's facts raise every model's score on the same photos,** by 4.7 to 7.5 points. All four intervals exclude 0, Gemini's only just (its lower end is +0.01 points). Only GPT-4o's gain (+7.5, Holm p = 0.003) is established; the other three are suggestive. The four gains agree in direction and size, which a single test does not capture, but each on its own is modest.

McNemar's test counts right against wrong answers, so on the text dataset, whose diagnoses are unequal in number, it can disagree with the difference in balanced accuracy.

## 4. Interpretation

- **The level is low for every model.** The best balanced accuracy is 51% on photos and 61% on text. Without retrieval, no model is a reliable diagnostic aid on these questions.
- **No model is shown to tell melanoma from nevus on the photos.** Melanoma and nevus come from one ISIC collection and the six other diagnoses from another. Inside the melanoma-and-nevus group, every model's balanced accuracy is near or below the 50% of chance (30.0% to 53.8% on image, 38.7% to 56.2% with the facts). Part of the overall score therefore comes from telling the two collections apart, which the look of the photos may give away without any reading of the lesion. Inside the six other diagnoses, Claude (51.3%) and Gemini (56.2%) are well above the 16.7% of chance; GPT-4o (28.3%) and Maverick (20.4%) are close to it.
- **GPT-4o and Llama 4 Maverick lean on one answer.** On the image dataset, Maverick answers melanoma for 62% of the photos and GPT-4o for 46% (Claude 16%, Gemini 10%). Their melanoma recall is high (85% and 78%) only because of this; their balanced accuracy on photos (29% and 34%) is at or below what the border colour alone reaches, so their reading of the photos is not shown to go beyond cues that are not the lesion.
- **Squamous cell carcinoma is almost never recognised from a photo** (0% to 5% recall for all four models, 15% at most with the facts) but is recognised in 75% to 80% of the written cases.
- **Text scores are higher than photo scores, and this is not a like-for-like comparison.** A text question is a clinician's written case (25 to 443 words, median 87); an image + metadata question is a photo with about ten words of facts and describes a different patient. No question of these three datasets holds both a written case and its photo (the case dataset, built since, does). The supported comparison is the same photos without and with the facts (section 2). Claude's text score (55.9%) is close to its image + metadata score (56.2%) only because of actinic keratosis, which has 2 text test cases, none recognised; without that class its text score is 63.9% (each text run's scores).
- **The facts help, modestly.** Part of the gain may come from the facts alone, which reach 35.9% without any photo. Claude (56.2%) and Gemini (54.4%) are well above that level on the image + metadata questions; GPT-4o (41.6%) is at the upper end of that interval and Maverick (34.1%) is not above it. Scoring only the 289 questions with all three facts changes no score by more than 1.1 points (each image + metadata run's scores).
- **These answers cannot show hallucination apart from error.** A model could not say it was unsure, so every wrong answer was a confident one. The JSON format of the next runs adds the evidence flag, and with it the misdiagnosis detection and selective risk of method.md.
- **Answers that were not one diagnosis name** occurred only on text: Claude 10 of 221 and Maverick 11 of 221. Reading a diagnosis name alone on the last line of such an answer, as a sensitivity analysis, moves Claude from 55.9% to 59.1% and Maverick from 46.8% to 47.1% (each text run's scores). The strict reading stays the main result: it was fixed before any result was seen and applies to every model alike.

## 5. Limits

- **One run per model and dataset.** Run-to-run variation was not measured. Claude cannot be seeded, and the other models' seeds are best effort.
- **The models were not set up identically** (section 6): the answer limit differs, Claude receives no temperature, and Gemini spends hidden reasoning tokens. The prompt, the answer reader and the data are the same for all four. The models are also of different generations.
- **Small groups.** A single diagnosis has 40 test photos, so its recall has an interval of roughly ±15 points. The text test set has 2 actinic keratosis cases, which count for one eighth of its balanced accuracy; each text run's scores also give the scores without that class.
- **Many comparisons.** The paired intervals are not corrected for the 22 comparisons of this page; the Holm-corrected p-values are, which is why section 3 asks for both.
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

Each run folder's `run_card.json` records the settings, the prompt, the dataset fingerprint and the Git commit the run used. At the time of these runs the settings were one file per model and dataset (`configs/<dataset>/<model>.toml`) and the prompts were `prompts/<dataset>/diagnosis.txt`. The settings are now in `configs/models.toml`, with an answer limit of 512 for Claude, GPT-4o and Maverick (the JSON answer is longer), and the prompts are `prompts/<dataset>/closed_book.txt`, rewritten for the JSON answer. Llama 3.2 11B Vision, the first choice, is no longer served by any host, so Llama 4 Maverick replaced it.

## 7. Reproducing this page

No model call is needed. Every score and every table is rebuilt from the saved answers:

```
python -m src.evaluate report results/phase1/<dataset>/<model>/<run folder>
python -m src.evaluate summary
```

The first command writes `scores.md` and `scores.json` beside a first-phase run's archived files and changes nothing else.
