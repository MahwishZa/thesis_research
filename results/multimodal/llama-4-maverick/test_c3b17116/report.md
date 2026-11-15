# llama-4-maverick: test split, run test_c3b17116

- Model requested: `meta-llama/llama-4-maverick`; versions returned by the API: `meta-llama/llama-4-maverick@Parasail`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 256, "image_detail": null, "extra_body": {"provider": {"order": ["parasail/fp8"], "allow_fallbacks": false}}}`
- Started 2026-11-15T02:38:22+00:00, finished 2026-11-15T02:54:21+00:00 (UTC)
- Prompt: `prompts/multimodal/diagnosis.txt` (SHA-256 `d1a433195f62`); images: 320 (hash `c3de85ddfda0`), sent exactly as stored, each with its patient's facts in the prompt; dataset table `data/multimodal/questions.csv` (SHA-256 `150d699e3752`); Git commit `387fcfe590`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **34.1%** | 29.0% to 39.0% (bootstrap over lesions) |
| Accuracy | 34.1% (109/320) | 28.9% to 39.0% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **65.9%** | 61.0% to 71.0% (bootstrap over lesions) |
| Hallucination rate, all questions | 65.9% (211/320) | 61.0% to 71.1% (bootstrap over lesions) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 0 | |
| Missed malignancy: cancers answered as benign | 10.8% (13/120) | 5.0% to 17.9% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 52.5% (84/160) | 44.5% to 61.4% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 33.9% | 27.9% to 39.2% (bootstrap over lesions) |
| Macro-precision | 47.2% | |
| Weighted F1 (by number of true images) | 33.9% | |
| Matthews correlation (MCC, -1 to 1) | 0.288 | |
| Cohen's kappa (-1 to 1) | 0.246 | |

### Per class

| Class | True images | Answered as this class | Precision | Recall (95% interval: bootstrap over lesions) | F1 |
|---|---|---|---|---|---|
| actinic keratosis | 40 | 20 | 40.0% | 20.0% (7.0% to 37.1%) | 26.7% |
| basal cell carcinoma | 40 | 13 | 46.2% | 15.0% (4.2% to 28.6%) | 22.6% |
| dermatofibroma | 40 | 26 | 73.1% | 47.5% (28.6% to 66.7%) | 57.6% |
| melanoma | 40 | 184 | 17.9% | 82.5% (70.0% to 92.5%) | 29.5% |
| nevus | 40 | 27 | 44.4% | 30.0% (15.0% to 45.0%) | 35.8% |
| pigmented benign keratosis | 40 | 17 | 41.2% | 17.5% (5.6% to 31.7%) | 24.6% |
| squamous cell carcinoma | 40 | 6 | 33.3% | 5.0% (0.0% to 15.4%) | 8.7% |
| vascular lesion | 40 | 27 | 81.5% | 55.0% (37.5% to 73.2%) | 65.7% |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 8 | 6 | 1 | 16 | 1 | 7 | 1 | 0 | 0 |
| 2. basal cell carcinoma | 2 | 6 | 3 | 23 | 3 | 1 | 1 | 1 | 0 |
| 3. dermatofibroma | 0 | 0 | 19 | 17 | 3 | 0 | 0 | 1 | 0 |
| 4. melanoma | 4 | 0 | 1 | 33 | 0 | 1 | 1 | 0 | 0 |
| 5. nevus | 0 | 0 | 1 | 26 | 12 | 1 | 0 | 0 | 0 |
| 6. pigmented benign keratosis | 1 | 0 | 1 | 24 | 6 | 7 | 1 | 0 | 0 |
| 7. squamous cell carcinoma | 5 | 1 | 0 | 29 | 0 | 0 | 2 | 3 | 0 |
| 8. vascular lesion | 0 | 0 | 0 | 16 | 2 | 0 | 0 | 22 | 0 |

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
| Balanced accuracy (8 classes) | 33.4% | 28.1% to 39.0% (bootstrap over lesions) |
| Accuracy | 32.2% (93/289) | 26.9% to 37.7% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 32.5% | 26.2% to 38.2% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **66.6%** | 61.0% to 71.9% (bootstrap over lesions) |
| Hallucination rate, all questions | 67.8% (196/289) | 62.3% to 73.1% (bootstrap over lesions) |
| Missed malignancy: cancers answered as benign | 11.6% (13/112) | 5.4% to 19.3% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 51.1% (70/137) | 41.7% to 60.6% (bootstrap over lesions) |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: 387fcfe590
- Interruptions before the run finished: 0
