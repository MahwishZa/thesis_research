# claude-sonnet-5-5: test split, run test_cf5351f1

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 256, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Started 2026-11-14T16:50:46+00:00, finished 2026-11-14T17:12:47+00:00 (UTC)
- Prompt: `prompts/image/diagnosis.txt` (SHA-256 `5e1ed90af51d`); images: 320 (hash `cfa2c9e9b7c6`), sent exactly as stored; dataset table `data/image/dataset.csv` (SHA-256 `394408d529cf`); Git commit `b0f7fe7cd8`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **51.2%** | 45.7% to 56.7% (bootstrap over lesions) |
| Accuracy | 51.2% (164/320) | 45.6% to 56.8% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **48.8%** | 43.3% to 54.3% (bootstrap over lesions) |
| Hallucination rate, all questions | 48.8% (156/320) | 43.2% to 54.4% (bootstrap over lesions) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 0 | |
| Missed malignancy: cancers answered as benign | 25.0% (30/120) | 17.4% to 33.1% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 21.9% (35/160) | 14.5% to 29.4% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 49.5% | 44.0% to 54.7% (bootstrap over lesions) |
| Macro-precision | 50.8% | |
| Weighted F1 (by number of true images) | 49.5% | |
| Matthews correlation (MCC, -1 to 1) | 0.447 | |
| Cohen's kappa (-1 to 1) | 0.443 | |

### Per class

| Class | True images | Answered as this class | Precision | Recall (95% interval: bootstrap over lesions) | F1 |
|---|---|---|---|---|---|
| actinic keratosis | 40 | 53 | 34.0% | 45.0% (28.9% to 62.9%) | 38.7% |
| basal cell carcinoma | 40 | 52 | 50.0% | 65.0% (47.4% to 80.6%) | 56.5% |
| dermatofibroma | 40 | 33 | 81.8% | 67.5% (48.7% to 84.1%) | 74.0% |
| melanoma | 40 | 50 | 34.0% | 42.5% (27.5% to 57.5%) | 37.8% |
| nevus | 40 | 49 | 49.0% | 60.0% (45.0% to 75.0%) | 53.9% |
| pigmented benign keratosis | 40 | 36 | 41.7% | 37.5% (20.0% to 56.1%) | 39.5% |
| squamous cell carcinoma | 40 | 7 | 28.6% | 5.0% (0.0% to 15.4%) | 8.5% |
| vascular lesion | 40 | 40 | 87.5% | 87.5% (73.7% to 100.0%) | 87.5% |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 18 | 11 | 2 | 1 | 2 | 4 | 1 | 1 | 0 |
| 2. basal cell carcinoma | 5 | 26 | 1 | 2 | 2 | 0 | 2 | 2 | 0 |
| 3. dermatofibroma | 0 | 6 | 27 | 2 | 3 | 0 | 1 | 1 | 0 |
| 4. melanoma | 5 | 1 | 0 | 17 | 6 | 11 | 0 | 0 | 0 |
| 5. nevus | 0 | 1 | 1 | 13 | 24 | 1 | 0 | 0 | 0 |
| 6. pigmented benign keratosis | 6 | 0 | 1 | 8 | 9 | 15 | 1 | 0 | 0 |
| 7. squamous cell carcinoma | 19 | 7 | 0 | 4 | 2 | 5 | 2 | 1 | 0 |
| 8. vascular lesion | 0 | 0 | 1 | 3 | 1 | 0 | 0 | 35 | 0 |

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
- Sessions (starts and restarts of this run): 1; Git commits: b0f7fe7cd8
- Interruptions before the run finished: 0
