# gpt-4o: test split, run test_12259e56

- Model requested: `openai/gpt-4o-2024-08-06`; versions returned by the API: `openai/gpt-4o-2024-08-06@OpenAI`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 64, "image_detail": "high", "extra_body": {"provider": {"order": ["openai"], "allow_fallbacks": false}}}`
- Started 2026-11-14T23:28:17+00:00, finished 2026-11-14T23:44:18+00:00 (UTC)
- Prompt: `prompts/multimodal/diagnosis.txt` (SHA-256 `d1a433195f62`); images: 320 (hash `c3de85ddfda0`), sent exactly as stored, each with its patient's facts in the prompt; dataset table `data/multimodal/questions.csv` (SHA-256 `150d699e3752`); Git commit `499bc650b3`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **41.6%** | 36.4% to 46.7% (bootstrap over lesions) |
| Accuracy | 41.6% (133/320) | 36.3% to 46.9% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **58.4%** | 53.3% to 63.6% (bootstrap over lesions) |
| Hallucination rate, all questions | 58.4% (187/320) | 53.1% to 63.7% (bootstrap over lesions) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 0 | |
| Missed malignancy: cancers answered as benign | 19.2% (23/120) | 12.4% to 26.8% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 43.1% (69/160) | 34.3% to 51.6% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 40.6% | 35.1% to 45.6% (bootstrap over lesions) |
| Macro-precision | 49.7% | |
| Weighted F1 (by number of true images) | 40.6% | |
| Matthews correlation (MCC, -1 to 1) | 0.358 | |
| Cohen's kappa (-1 to 1) | 0.332 | |

### Per class

| Class | True images | Answered as this class | Precision | Recall (95% interval: bootstrap over lesions) | F1 |
|---|---|---|---|---|---|
| actinic keratosis | 40 | 12 | 66.7% | 20.0% (6.7% to 35.9%) | 30.8% |
| basal cell carcinoma | 40 | 51 | 41.2% | 52.5% (37.5% to 67.6%) | 46.2% |
| dermatofibroma | 40 | 41 | 70.7% | 72.5% (55.0% to 88.1%) | 71.6% |
| melanoma | 40 | 137 | 20.4% | 70.0% (55.0% to 85.0%) | 31.6% |
| nevus | 40 | 22 | 59.1% | 32.5% (17.5% to 47.5%) | 41.9% |
| pigmented benign keratosis | 40 | 38 | 39.5% | 37.5% (19.0% to 55.1%) | 38.5% |
| squamous cell carcinoma | 40 | 0 | – | 0.0% (0.0% to 0.0%) | 0.0% |
| vascular lesion | 40 | 19 | 100.0% | 47.5% (29.3% to 67.8%) | 64.4% |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 8 | 15 | 2 | 9 | 0 | 6 | 0 | 0 | 0 |
| 2. basal cell carcinoma | 1 | 21 | 3 | 14 | 0 | 1 | 0 | 0 | 0 |
| 3. dermatofibroma | 0 | 2 | 29 | 7 | 2 | 0 | 0 | 0 | 0 |
| 4. melanoma | 0 | 1 | 0 | 28 | 2 | 9 | 0 | 0 | 0 |
| 5. nevus | 0 | 0 | 1 | 25 | 13 | 1 | 0 | 0 | 0 |
| 6. pigmented benign keratosis | 2 | 2 | 2 | 16 | 3 | 15 | 0 | 0 | 0 |
| 7. squamous cell carcinoma | 1 | 8 | 2 | 23 | 0 | 6 | 0 | 0 | 0 |
| 8. vascular lesion | 0 | 2 | 2 | 15 | 2 | 0 | 0 | 19 | 0 |

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
| Balanced accuracy (8 classes) | 40.9% | 35.1% to 46.8% (bootstrap over lesions) |
| Accuracy | 40.8% (118/289) | 35.1% to 46.5% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 39.5% | 33.5% to 44.9% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **59.1%** | 53.2% to 64.9% (bootstrap over lesions) |
| Hallucination rate, all questions | 59.2% (171/289) | 53.5% to 64.9% (bootstrap over lesions) |
| Missed malignancy: cancers answered as benign | 18.8% (21/112) | 11.3% to 27.2% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 41.6% (57/137) | 31.8% to 51.7% (bootstrap over lesions) |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: 499bc650b3
- Interruptions before the run finished: 0
