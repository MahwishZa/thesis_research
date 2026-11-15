# gpt-4o: test split, run test_14e3059f

- Model requested: `openai/gpt-4o-2024-08-06`; versions returned by the API: `openai/gpt-4o-2024-08-06@OpenAI`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 64, "image_detail": "high", "extra_body": {"provider": {"order": ["openai"], "allow_fallbacks": false}}}`
- Started 2026-11-14T23:01:02+00:00, finished 2026-11-14T23:17:04+00:00 (UTC)
- Prompt: `prompts/image/diagnosis.txt` (SHA-256 `5e1ed90af51d`); images: 320 (hash `cfa2c9e9b7c6`), sent exactly as stored; dataset table `data/image/dataset.csv` (SHA-256 `394408d529cf`); Git commit `499bc650b3`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **34.1%** | 29.6% to 38.7% (bootstrap over lesions) |
| Accuracy | 34.1% (109/320) | 29.4% to 38.7% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **65.9%** | 61.3% to 70.4% (bootstrap over lesions) |
| Hallucination rate, all questions | 65.9% (211/320) | 61.3% to 70.6% (bootstrap over lesions) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 0 | |
| Missed malignancy: cancers answered as benign | 22.5% (27/120) | 14.0% to 30.6% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 44.4% (71/160) | 36.0% to 52.4% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 30.1% | 25.5% to 34.5% (bootstrap over lesions) |
| Macro-precision | 36.4% | |
| Weighted F1 (by number of true images) | 30.1% | |
| Matthews correlation (MCC, -1 to 1) | 0.274 | |
| Cohen's kappa (-1 to 1) | 0.246 | |

### Per class

| Class | True images | Answered as this class | Precision | Recall (95% interval: bootstrap over lesions) | F1 |
|---|---|---|---|---|---|
| actinic keratosis | 40 | 0 | – | 0.0% (0.0% to 0.0%) | 0.0% |
| basal cell carcinoma | 40 | 40 | 40.0% | 40.0% (24.3% to 56.4%) | 40.0% |
| dermatofibroma | 40 | 75 | 34.7% | 65.0% (47.7% to 81.1%) | 45.2% |
| melanoma | 40 | 147 | 21.1% | 77.5% (65.0% to 90.0%) | 33.2% |
| nevus | 40 | 22 | 45.5% | 25.0% (12.5% to 40.0%) | 32.3% |
| pigmented benign keratosis | 40 | 20 | 50.0% | 25.0% (9.5% to 41.7%) | 33.3% |
| squamous cell carcinoma | 40 | 0 | – | 0.0% (0.0% to 0.0%) | 0.0% |
| vascular lesion | 40 | 16 | 100.0% | 40.0% (25.5% to 57.9%) | 57.1% |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 0 | 8 | 12 | 15 | 1 | 4 | 0 | 0 | 0 |
| 2. basal cell carcinoma | 0 | 16 | 3 | 20 | 1 | 0 | 0 | 0 | 0 |
| 3. dermatofibroma | 0 | 5 | 26 | 7 | 2 | 0 | 0 | 0 | 0 |
| 4. melanoma | 0 | 1 | 2 | 31 | 2 | 4 | 0 | 0 | 0 |
| 5. nevus | 0 | 1 | 1 | 27 | 10 | 1 | 0 | 0 | 0 |
| 6. pigmented benign keratosis | 0 | 1 | 9 | 15 | 5 | 10 | 0 | 0 | 0 |
| 7. squamous cell carcinoma | 0 | 4 | 14 | 21 | 0 | 1 | 0 | 0 | 0 |
| 8. vascular lesion | 0 | 4 | 8 | 11 | 1 | 0 | 0 | 16 | 0 |

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
- Sessions (starts and restarts of this run): 1; Git commits: 499bc650b3
- Interruptions before the run finished: 0
