# gemini-3.8-flash: test split, run test_b086a9a1

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Started 2026-11-15T00:22:38+00:00, finished 2026-11-15T00:45:53+00:00 (UTC)
- Prompt: `prompts/image/diagnosis.txt` (SHA-256 `5e1ed90af51d`); images: 320 (hash `cfa2c9e9b7c6`), sent exactly as stored; dataset table `data/image/dataset.csv` (SHA-256 `394408d529cf`); Git commit `ba9f33c36c`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **49.7%** | 44.3% to 55.0% (bootstrap over lesions) |
| Accuracy | 49.7% (159/320) | 44.1% to 55.1% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **50.3%** | 45.0% to 55.7% (bootstrap over lesions) |
| Hallucination rate, all questions | 50.3% (161/320) | 44.9% to 55.9% (bootstrap over lesions) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 0 | |
| Missed malignancy: cancers answered as benign | 21.7% (26/120) | 14.4% to 29.3% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 16.2% (26/160) | 10.2% to 22.6% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 46.2% | 41.1% to 51.2% (bootstrap over lesions) |
| Macro-precision | 46.6% | |
| Weighted F1 (by number of true images) | 46.2% | |
| Matthews correlation (MCC, -1 to 1) | 0.434 | |
| Cohen's kappa (-1 to 1) | 0.425 | |

### Per class

| Class | True images | Answered as this class | Precision | Recall (95% interval: bootstrap over lesions) | F1 |
|---|---|---|---|---|---|
| actinic keratosis | 40 | 49 | 32.7% | 40.0% (22.7% to 60.5%) | 36.0% |
| basal cell carcinoma | 40 | 73 | 42.5% | 77.5% (60.5% to 92.5%) | 54.9% |
| dermatofibroma | 40 | 45 | 55.6% | 62.5% (41.5% to 81.6%) | 58.8% |
| melanoma | 40 | 33 | 36.4% | 30.0% (17.5% to 45.0%) | 32.9% |
| nevus | 40 | 18 | 66.7% | 30.0% (17.5% to 45.0%) | 41.4% |
| pigmented benign keratosis | 40 | 62 | 41.9% | 65.0% (47.2% to 81.1%) | 51.0% |
| squamous cell carcinoma | 40 | 2 | 0.0% | 0.0% (0.0% to 0.0%) | 0.0% |
| vascular lesion | 40 | 38 | 97.4% | 92.5% (81.6% to 100.0%) | 94.9% |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 16 | 9 | 6 | 3 | 1 | 4 | 1 | 0 | 0 |
| 2. basal cell carcinoma | 4 | 31 | 0 | 2 | 0 | 3 | 0 | 0 | 0 |
| 3. dermatofibroma | 2 | 9 | 25 | 2 | 1 | 0 | 0 | 1 | 0 |
| 4. melanoma | 5 | 7 | 1 | 12 | 3 | 12 | 0 | 0 | 0 |
| 5. nevus | 1 | 1 | 7 | 6 | 12 | 13 | 0 | 0 | 0 |
| 6. pigmented benign keratosis | 5 | 1 | 3 | 4 | 1 | 26 | 0 | 0 | 0 |
| 7. squamous cell carcinoma | 16 | 13 | 3 | 4 | 0 | 4 | 0 | 0 | 0 |
| 8. vascular lesion | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 37 | 0 |

Notes on these scores:
- Balanced accuracy is the main score: every class counts the same, whatever its number of test images (the measure of the ISIC 2018 lesion-diagnosis challenge).
- Every class has the same number of test images (40), so plain accuracy equals balanced accuracy.
- Precision and F1 depend on how common each class is in the test set, and the test set's mix of classes is not a clinic's: a rare disease is far more common here, so its precision is higher than it would be in a clinic, and it is not a clinical figure. Recall and balanced accuracy do not depend on it.
- Photos of one lesion are not independent: the intervals resample whole lesions (248 lesions for 320 images).
- A dash means undefined: precision when a class was never answered, recall when no image has that class. A class that was never answered counts as precision 0 in the macro-precision.
- With one answer per image, micro-averaged precision, recall and F1 equal accuracy, so they are not listed.
- Hallucination: an answer that names a wrong diagnosis. Right answers, hallucinations and answers with no valid answer (a refusal, a sentence, two classes, a cut-off) add up to every question; a model that refuses is not hallucinating, but it is not right either. The models must choose a class, so none can avoid hallucinating by declaring itself unsure. With the same number of images in every class, the two rates are the same.
- Missed malignancy: melanoma, basal cell carcinoma and squamous cell carcinoma answered as nevus, pigmented benign keratosis, dermatofibroma or vascular lesion, as a share of all cancers. Over-call: the reverse, as a share of all benign lesions. An unreadable answer, or actinic keratosis (precancerous) given for a cancer, is no miss but stays in the total; actinic keratosis as a true class is in neither group.
- No ROC curve or AUC: the models answer with one class name, not a score per class, so there is nothing to rank.

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: ba9f33c36c
- Interruptions before the run finished: 0
