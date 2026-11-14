# claude-sonnet-5-5: test split, run test_5db7a077

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 256, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Started 2026-11-14T17:27:43+00:00, finished 2026-11-14T17:49:01+00:00 (UTC)
- Prompt: `prompts/multimodal/diagnosis.txt` (SHA-256 `d1a433195f62`); images: 320 (hash `c3de85ddfda0`), sent exactly as stored, each with its patient's facts in the prompt; dataset table `data/multimodal/questions.csv` (SHA-256 `150d699e3752`); Git commit `b0f7fe7cd8`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **56.2%** | 50.6% to 61.7% (bootstrap over lesions) |
| Accuracy | 56.2% (180/320) | 50.6% to 61.9% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **43.8%** | 38.3% to 49.4% (bootstrap over lesions) |
| Hallucination rate, all questions | 43.8% (140/320) | 38.1% to 49.4% (bootstrap over lesions) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 0 | |
| Missed malignancy: cancers answered as benign | 20.8% (25/120) | 13.6% to 28.3% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 23.1% (37/160) | 15.7% to 30.6% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 55.7% | 50.0% to 61.1% (bootstrap over lesions) |
| Macro-precision | 58.3% | |
| Weighted F1 (by number of true images) | 55.7% | |
| Matthews correlation (MCC, -1 to 1) | 0.504 | |
| Cohen's kappa (-1 to 1) | 0.500 | |

### Per class

| Class | True images | Answered as this class | Precision | Recall (95% interval: bootstrap over lesions) | F1 |
|---|---|---|---|---|---|
| actinic keratosis | 40 | 45 | 46.7% | 52.5% (34.8% to 69.7%) | 49.4% |
| basal cell carcinoma | 40 | 55 | 47.3% | 65.0% (47.6% to 81.1%) | 54.7% |
| dermatofibroma | 40 | 35 | 85.7% | 75.0% (56.8% to 91.7%) | 80.0% |
| melanoma | 40 | 57 | 35.1% | 50.0% (35.0% to 65.0%) | 41.2% |
| nevus | 40 | 38 | 60.5% | 57.5% (42.5% to 72.5%) | 59.0% |
| pigmented benign keratosis | 40 | 41 | 43.9% | 45.0% (27.8% to 63.6%) | 44.4% |
| squamous cell carcinoma | 40 | 12 | 50.0% | 15.0% (3.0% to 28.9%) | 23.1% |
| vascular lesion | 40 | 37 | 97.3% | 90.0% (76.2% to 100.0%) | 93.5% |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 21 | 9 | 0 | 1 | 2 | 5 | 2 | 0 | 0 |
| 2. basal cell carcinoma | 5 | 26 | 1 | 2 | 1 | 1 | 3 | 1 | 0 |
| 3. dermatofibroma | 0 | 6 | 30 | 2 | 1 | 0 | 1 | 0 | 0 |
| 4. melanoma | 4 | 1 | 0 | 20 | 6 | 9 | 0 | 0 | 0 |
| 5. nevus | 0 | 1 | 1 | 13 | 23 | 2 | 0 | 0 | 0 |
| 6. pigmented benign keratosis | 4 | 2 | 2 | 10 | 4 | 18 | 0 | 0 | 0 |
| 7. squamous cell carcinoma | 11 | 10 | 0 | 7 | 0 | 6 | 6 | 0 | 0 |
| 8. vascular lesion | 0 | 0 | 1 | 2 | 1 | 0 | 0 | 36 | 0 |

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

## Questions with all facts recorded

A missing fact ("not recorded") points to the photo's collection, and so mostly to melanoma or nevus (data/multimodal/datasheet.md, limitation 1). These scores count only the 289 of 320 questions with every fact recorded, beside the main score, never instead of it.

| Metric | Value | 95% interval |
|---|---|---|
| Balanced accuracy (8 classes) | 57.3% | 51.3% to 63.0% (bootstrap over lesions) |
| Accuracy | 56.7% (164/289) | 50.7% to 62.5% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 55.8% | 49.9% to 61.3% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **42.7%** | 37.0% to 48.7% (bootstrap over lesions) |
| Hallucination rate, all questions | 43.3% (125/289) | 37.5% to 49.3% (bootstrap over lesions) |
| Missed malignancy: cancers answered as benign | 18.8% (21/112) | 11.5% to 27.0% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 21.9% (30/137) | 14.2% to 30.8% (bootstrap over lesions) |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: b0f7fe7cd8
- Interruptions before the run finished: 0
