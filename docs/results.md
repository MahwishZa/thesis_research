# Closed-book results

Four LLMs answered the test split of three datasets without retrieval (closed-book). Each question has eight possible diagnoses and the model must name exactly one (forced choice). Every number below comes from the saved run folders in `results/` (each linked report holds the full per-class tables and confusion matrix). This page is the reading of those numbers; it adds no new experiment.

- **Datasets:** image (320 test photos), text (221 test cases), multimodal (the same 320 photos, each with the patient's age, sex and body site). Descriptions: [image](../data/image/datasheet.md), [text](../data/text/datasheet.md), [multimodal](../data/multimodal/datasheet.md).
- **Main score: balanced accuracy**, the mean recall over the eight classes, with a 95% interval that resamples whole lesions (image, multimodal) or whole articles (text). Chance is 12.5%.
- **Hallucination rate:** the share of questions where the model named a wrong diagnosis, averaged over the classes (class-balanced). An answer that is not one class name counts as wrong but is not a hallucination (column *No valid answer*).
- **Missed malignancy:** melanoma, basal cell carcinoma or squamous cell carcinoma answered as a benign class, as a share of all cancers. **Over-call:** a benign lesion answered as a cancer, as a share of all benign lesions. Actinic keratosis (precancerous) counts in neither group.

## 1. Results per dataset

### Image

| Model | Balanced accuracy (95% CI) | Accuracy | Hallucination rate | No valid answer | Missed malignancy | Over-call |
|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](../results/image/claude-sonnet-5-5/test_cf5351f1/report.md) | 51.2% (45.7 to 56.7) | 51.2% | 48.8% | 0 of 320 | 25.0% | 21.9% |
| [GPT-4o](../results/image/gpt-4o/test_14e3059f/report.md) | 34.1% (29.6 to 38.7) | 34.1% | 65.9% | 0 of 320 | 22.5% | 44.4% |
| [Gemini 3.8 Flash](../results/image/gemini-3.8-flash/test_b086a9a1/report.md) | 49.7% (44.3 to 55.0) | 49.7% | 50.3% | 0 of 320 | 21.7% | 16.2% |
| [Llama 4 Maverick](../results/image/llama-4-maverick/test_055451fe/report.md) | 28.7% (24.4 to 33.1) | 28.7% | 71.2% | 0 of 320 | 19.2% | 53.8% |

### Text

| Model | Balanced accuracy (95% CI) | Accuracy | Hallucination rate | No valid answer | Missed malignancy | Over-call |
|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](../results/text/claude-sonnet-5-5/test_06ce1664/report.md) | 55.9% (49.6 to 61.8) | 64.3% | 34.1% | 10 of 221 | 5.8% | 27.3% |
| [GPT-4o](../results/text/gpt-4o/test_5fc7ce07/report.md) | 50.8% (45.1 to 56.6) | 56.6% | 49.2% | 0 of 221 | 14.2% | 25.3% |
| [Gemini 3.8 Flash](../results/text/gemini-3.8-flash/test_6ee1c09b/report.md) | 61.4% (55.8 to 67.0) | 69.2% | 38.6% | 0 of 221 | 5.0% | 31.3% |
| [Llama 4 Maverick](../results/text/llama-4-maverick/test_32905170/report.md) | 46.8% (40.6 to 52.8) | 53.8% | 48.5% | 11 of 221 | 10.0% | 24.2% |

### Multimodal

| Model | Balanced accuracy (95% CI) | Accuracy | Hallucination rate | No valid answer | Missed malignancy | Over-call |
|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](../results/multimodal/claude-sonnet-5-5/test_5db7a077/report.md) | 56.2% (50.6 to 61.7) | 56.2% | 43.8% | 0 of 320 | 20.8% | 23.1% |
| [GPT-4o](../results/multimodal/gpt-4o/test_12259e56/report.md) | 41.6% (36.4 to 46.7) | 41.6% | 58.4% | 0 of 320 | 19.2% | 43.1% |
| [Gemini 3.8 Flash](../results/multimodal/gemini-3.8-flash/test_2ff4fbdd/report.md) | 54.4% (48.6 to 59.9) | 54.4% | 45.6% | 0 of 320 | 21.7% | 19.4% |
| [Llama 4 Maverick](../results/multimodal/llama-4-maverick/test_c3b17116/report.md) | 34.1% (29.0 to 39.0) | 34.1% | 65.9% | 0 of 320 | 10.8% | 52.5% |

Balanced accuracy by model and dataset:

| Model | Image | Text | Multimodal |
|---|---|---|---|
| Claude Sonnet 5.5 | 51.2% | 55.9% | 56.2% |
| GPT-4o | 34.1% | 50.8% | 41.6% |
| Gemini 3.8 Flash | 49.7% | 61.4% | 54.4% |
| Llama 4 Maverick | 28.7% | 46.8% | 34.1% |

Reference points from the data sheets: chance 12.5%; the patient's facts alone, with no photo, 35.9% (30.0 to 41.5) on the multimodal test set; a classifier that sees only the colour of the image border, 34.7% on the image test set.

## 2. Do the patient's facts help?

The same 320 photos were answered without facts (image) and with facts (multimodal). Differences are multimodal minus image, paired by photo; the interval is a paired bootstrap over lesions, and the p-value is McNemar's exact test on right-versus-wrong.

| Model | Image | Multimodal | Difference (95% CI) | McNemar p | Hallucinations removed / introduced |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 51.2% | 56.2% | +5.0 points (+0.8 to +9.1) | 0.0166 | 28 / 12 |
| GPT-4o | 34.1% | 41.6% | +7.5 points (+3.7 to +11.4) | 0.0003 | 33 / 9 |
| Gemini 3.8 Flash | 49.7% | 54.4% | +4.7 points (+0.0 to +9.2) | 0.0534 | 34 / 19 |
| Llama 4 Maverick | 28.7% | 34.1% | +5.3 points (+1.6 to +9.3) | 0.0115 | 29 / 12 |

Scoring only the 289 of 320 questions that have all three facts changes no score by more than 1.5 points (see each multimodal report).

## 3. Model against model

Balanced-accuracy difference, second model minus first, on the same questions; *p* is McNemar's exact test on right-versus-wrong, so on the text dataset (unequal classes) it can disagree with the balanced-accuracy difference. Differences with p below 0.05 are marked with an asterisk. There are 18 tests and none is corrected for that, so treat a lone asterisk near 0.05 with caution.

| Pair (second minus first) | Image | Text | Multimodal |
|---|---|---|---|
| GPT-4o − Claude Sonnet 5.5 | -17.2 (p <0.001)* | -5.1 (p 0.012)* | -14.7 (p <0.001)* |
| Gemini 3.8 Flash − Claude Sonnet 5.5 | -1.6 (p 0.644) | +5.5 (p 0.126) | -1.9 (p 0.550) |
| Llama 4 Maverick − Claude Sonnet 5.5 | -22.5 (p <0.001)* | -9.1 (p 0.003)* | -22.2 (p <0.001)* |
| Gemini 3.8 Flash − GPT-4o | +15.6 (p <0.001)* | +10.6 (p <0.001)* | +12.8 (p <0.001)* |
| Llama 4 Maverick − GPT-4o | -5.3 (p 0.046)* | -4.0 (p 0.418) | -7.5 (p 0.015)* |
| Llama 4 Maverick − Gemini 3.8 Flash | -20.9 (p <0.001)* | -14.6 (p <0.001)* | -20.3 (p <0.001)* |

## 4. Answers that were not one class name

The main score reads an answer only when it is exactly one class name, as the prompt asks. Only the text dataset produced such answers:

| Run | Not a valid answer | Of which finished with a class name on the last line | Balanced accuracy, strict (main) | Balanced accuracy, last line also accepted |
|---|---|---|---|---|
| Text, Claude Sonnet 5.5 | 10 of 221 (4.5%) | 9 | 55.9% | 59.1% |
| Text, Llama 4 Maverick | 11 of 221 (5.0%) | 1 | 46.8% | 47.1% |

The other ten runs had no such answer. The strict reading stays the main result, because it was fixed before any result was seen and applies to every model alike.

## 5. What the results show

- **Claude Sonnet 5.5 and Gemini 3.8 Flash lead.** On images (51.2% and 49.7%) and on multimodal questions (56.2% and 54.4%) they show no significant difference (section 3). On text Gemini is higher by 5.5 points, which the McNemar test does not call significant (p = 0.126).
- **GPT-4o and Llama 4 Maverick are far behind on photos.** On images they score 34.1% and 28.7%, about 16 to 22 points below the two leaders. Both also call many benign lesions cancer (over-call 44% and 54%, against 16% to 22% for Gemini and Claude). GPT-4o answered melanoma for 147 of the 320 images. On the image dataset these two models are at or below the 34.7% a border-colour classifier reaches, so their image reading is not shown to be better than the non-lesion cues in the photos.
- **All four do better on text than on photos alone**, but the text and image datasets are different cases, not paired, so this is not a like-for-like comparison. A case description often names the findings a model needs.
- **The patient's facts help every model, modestly:** +4.7 to +7.5 points, significant for three of four (Gemini p = 0.053). Part of that gain may come from the facts alone, which reach 35.9% on their own. Claude (56.2%) and Gemini (54.4%) are well above that level on multimodal questions; GPT-4o (41.6%) is at the upper end of that interval (30.0 to 41.5), and Maverick (34.1%) is not above it.
- **The hallucination rate adds little beyond accuracy in this forced-choice condition.** Every wrong answer is a hallucination, so the rate is about 100% minus accuracy. Whether a model avoids hallucinating by declining to answer can only be seen in the abstention condition (prepared in the `-abstain` configs, not run) and later with retrieval.
- **Missed malignancy is lowest on text** (Gemini 5.0%, Claude 5.8%). On photos alone every model misses 19% to 25% of cancers.

## 6. What the results do not show

- **One run per model and dataset.** Run-to-run noise was not measured. Claude cannot be seeded, and the other seeds are best effort, so a repeat can differ.
- **The models were not set up identically.** The answer limit differs (64, 256 or 2048 tokens), Claude receives no temperature and Gemini still spends hidden reasoning tokens (section 7). The prompt, the answer reader and the data are identical for all four.
- **Small subgroups.** The text test has 2 actinic keratosis cases (each text report also gives the scores without that class; for example, Gemini 70.2% without it).
- **Possible exposure to the text cases.** The case reports were published up to 2026, so a model may have seen some (limitation 4 in the [text data sheet](../data/text/datasheet.md)).
- **Not clinical performance.** The test sets come from public archives and published case reports, and their mix of classes is not a clinic's, so the shares above are not what a clinic would see.

## 7. How each model was run

| Model | Model id and route | Answer limit | Other settings |
|---|---|---|---|
| Claude Sonnet 5.5 | `claude-sonnet-5-5`, Anthropic API | 256 | no temperature (the model rejects it); thinking between tools; effort low |
| GPT-4o | `openai/gpt-4o-2024-08-06`, OpenRouter, pinned to OpenAI | 64 | temperature 0, seed 0, image detail high |
| Gemini 3.8 Flash | `google/gemini-3.8-flash`, OpenRouter, pinned to Google AI Studio | 2048 | temperature 0, seed 0, reasoning effort minimal (hidden reasoning still used 47 to 532 tokens per answer in the trial) |
| Llama 4 Maverick | `meta-llama/llama-4-maverick`, OpenRouter, pinned to Parasail fp8 | 256 | temperature 0, seed 0; 8-bit (fp8) precision |

The exact settings are in `configs/<dataset>/<model>.toml`, and each run folder's `run_card.json` records the config, prompt hash, dataset hash and Git commit it ran on. The prompts are `prompts/<dataset>/diagnosis.txt`, the same for every model. Llama 3.2 11B Vision, the first choice, is no longer served by any host, so Llama 4 Maverick replaced it.

## 8. Reproducing these numbers

No model call is needed. The reports and every table above can be rebuilt from the saved answers:

```
python -m src.evaluate report results/<dataset>/<model>/<run folder>
python -m src.evaluate compare results/image/<model>/<run folder> results/multimodal/<model>/<run folder>
```

The run folders are linked in the tables above. A rebuild on another operating system can differ in the last digit of a stored decimal, so a value that sits exactly on a rounding boundary (Llama 4 Maverick's 28.75% on images) can print as 28.7% or 28.8%.
