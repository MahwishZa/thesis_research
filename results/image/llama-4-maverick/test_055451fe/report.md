# llama-4-maverick: test split, run test_055451fe

- Model requested: `meta-llama/llama-4-maverick`; versions returned by the API: `meta-llama/llama-4-maverick@Parasail`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 256, "image_detail": null, "extra_body": {"provider": {"order": ["parasail/fp8"], "allow_fallbacks": false}}}`
- Started 2026-11-15T02:11:07+00:00, finished 2026-11-15T02:27:05+00:00 (UTC)
- Prompt: `prompts/image/diagnosis.txt` (SHA-256 `5e1ed90af51d`); images: 320 (hash `cfa2c9e9b7c6`), sent exactly as stored; dataset table `data/image/dataset.csv` (SHA-256 `394408d529cf`); Git commit `387fcfe590`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **28.7%** | 24.4% to 33.1% (bootstrap over lesions) |
| Accuracy | 28.7% (92/320) | 24.4% to 33.1% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **71.2%** | 66.9% to 75.6% (bootstrap over lesions) |
| Hallucination rate, all questions | 71.2% (228/320) | 66.9% to 75.6% (bootstrap over lesions) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 0 | |
| Missed malignancy: cancers answered as benign | 19.2% (23/120) | 11.5% to 27.1% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 53.8% (86/160) | 45.2% to 61.9% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 25.7% | 20.8% to 30.5% (bootstrap over lesions) |
| Macro-precision | 39.1% | |
| Weighted F1 (by number of true images) | 25.7% | |
| Matthews correlation (MCC, -1 to 1) | 0.227 | |
| Cohen's kappa (-1 to 1) | 0.186 | |

### Per class

| Class | True images | Answered as this class | Precision | Recall (95% interval: bootstrap over lesions) | F1 |
|---|---|---|---|---|---|
| actinic keratosis | 40 | 4 | 0.0% | 0.0% (0.0% to 0.0%) | 0.0% |
| basal cell carcinoma | 40 | 9 | 44.4% | 10.0% (0.0% to 22.2%) | 16.3% |
| dermatofibroma | 40 | 27 | 59.3% | 40.0% (21.6% to 59.0%) | 47.8% |
| melanoma | 40 | 198 | 17.2% | 85.0% (72.5% to 95.0%) | 28.6% |
| nevus | 40 | 38 | 23.7% | 22.5% (10.0% to 35.0%) | 23.1% |
| pigmented benign keratosis | 40 | 9 | 44.4% | 10.0% (2.3% to 20.0%) | 16.3% |
| squamous cell carcinoma | 40 | 4 | 50.0% | 5.0% (0.0% to 15.4%) | 9.1% |
| vascular lesion | 40 | 31 | 74.2% | 57.5% (41.9% to 73.7%) | 64.8% |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 0 | 4 | 1 | 27 | 4 | 2 | 0 | 2 | 0 |
| 2. basal cell carcinoma | 2 | 4 | 5 | 23 | 5 | 0 | 0 | 1 | 0 |
| 3. dermatofibroma | 0 | 1 | 16 | 18 | 4 | 0 | 1 | 0 | 0 |
| 4. melanoma | 1 | 0 | 0 | 34 | 2 | 1 | 0 | 2 | 0 |
| 5. nevus | 0 | 0 | 1 | 29 | 9 | 1 | 0 | 0 | 0 |
| 6. pigmented benign keratosis | 1 | 0 | 1 | 21 | 12 | 4 | 1 | 0 | 0 |
| 7. squamous cell carcinoma | 0 | 0 | 1 | 31 | 2 | 1 | 2 | 3 | 0 |
| 8. vascular lesion | 0 | 0 | 2 | 15 | 0 | 0 | 0 | 23 | 0 |

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
- Sessions (starts and restarts of this run): 1; Git commits: 387fcfe590
- Interruptions before the run finished: 0
