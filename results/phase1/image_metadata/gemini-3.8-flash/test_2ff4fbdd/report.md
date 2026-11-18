# gemini-3.8-flash: test split, run test_2ff4fbdd

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Started 2026-11-15T01:00:02+00:00, finished 2026-11-15T01:24:48+00:00 (UTC)
- Prompt: `prompts/multimodal/diagnosis.txt` (SHA-256 `d1a433195f62`); images: 320 (hash `c3de85ddfda0`), sent exactly as stored, each with its patient's facts in the prompt; dataset table `data/multimodal/questions.csv` (SHA-256 `150d699e3752`); Git commit `27c7ec0f34`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **54.4%** | 48.6% to 59.9% (bootstrap over lesions) |
| Accuracy | 54.4% (174/320) | 48.6% to 59.9% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **45.6%** | 40.1% to 51.4% (bootstrap over lesions) |
| Hallucination rate, all questions | 45.6% (146/320) | 40.1% to 51.4% (bootstrap over lesions) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 0 | |
| Malignancy sensitivity: cancers answered as a cancer | 65.8% (79/120) | 56.1% to 74.8% (bootstrap over lesions) |
| Missed malignancy: cancers answered as benign | 21.7% (26/120) | 14.1% to 29.9% (bootstrap over lesions) |
| Benign specificity: benign lesions answered as benign | 79.4% (127/160) | 72.1% to 86.6% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 19.4% (31/160) | 12.5% to 26.8% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 53.1% | 47.2% to 58.6% (bootstrap over lesions) |
| Macro-precision | 58.9% | 51.4% to 65.6% (bootstrap over lesions) |
| Macro-specificity | 93.5% | 92.6% to 94.3% (bootstrap over lesions) |
| Weighted F1 (by number of true images) | 53.1% | |
| Matthews correlation (MCC, -1 to 1) | 0.486 | |
| Cohen's kappa (-1 to 1) | 0.479 | |

### Per class

| Class | True images | Answered as this class | Precision | Recall (95% interval: bootstrap over lesions) | F1 | Specificity (95% interval: bootstrap over lesions) |
|---|---|---|---|---|---|---|
| actinic keratosis | 40 | 33 | 48.5% | 40.0% (21.7% to 60.5%) | 43.8% | 93.9% (90.5% to 97.0%) |
| basal cell carcinoma | 40 | 68 | 47.1% | 80.0% (63.9% to 94.9%) | 59.3% | 87.1% (82.5% to 91.2%) |
| dermatofibroma | 40 | 46 | 63.0% | 72.5% (53.8% to 88.9%) | 67.4% | 93.9% (90.9% to 96.5%) |
| melanoma | 40 | 47 | 29.8% | 35.0% (22.5% to 50.0%) | 32.2% | 88.2% (83.8% to 92.6%) |
| nevus | 40 | 23 | 73.9% | 42.5% (27.5% to 57.5%) | 54.0% | 97.9% (95.7% to 99.6%) |
| pigmented benign keratosis | 40 | 59 | 42.4% | 62.5% (46.0% to 78.9%) | 50.5% | 87.9% (83.7% to 91.6%) |
| squamous cell carcinoma | 40 | 9 | 66.7% | 15.0% (2.5% to 29.3%) | 24.5% | 98.9% (97.6% to 100.0%) |
| vascular lesion | 40 | 35 | 100.0% | 87.5% (73.8% to 100.0%) | 93.3% | 100.0% (100.0% to 100.0%) |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 16 | 6 | 4 | 7 | 1 | 5 | 1 | 0 | 0 |
| 2. basal cell carcinoma | 0 | 32 | 1 | 4 | 0 | 3 | 0 | 0 | 0 |
| 3. dermatofibroma | 0 | 6 | 29 | 2 | 1 | 0 | 2 | 0 | 0 |
| 4. melanoma | 4 | 6 | 1 | 14 | 2 | 13 | 0 | 0 | 0 |
| 5. nevus | 0 | 2 | 6 | 6 | 17 | 9 | 0 | 0 | 0 |
| 6. pigmented benign keratosis | 2 | 2 | 2 | 9 | 0 | 25 | 0 | 0 | 0 |
| 7. squamous cell carcinoma | 11 | 12 | 2 | 5 | 0 | 4 | 6 | 0 | 0 |
| 8. vascular lesion | 0 | 2 | 1 | 0 | 2 | 0 | 0 | 35 | 0 |

Notes on these scores:
- Balanced accuracy is the main score: every class counts the same, whatever its number of test images (the measure of the ISIC 2018 lesion-diagnosis challenge).
- Every class has the same number of test images (40), so plain accuracy equals balanced accuracy.
- Precision and F1 depend on how common each class is in the test set, and the test set's mix of classes is not a clinic's: a rare disease is far more common here, so its precision is higher than it would be in a clinic, and it is not a clinical figure. Recall and balanced accuracy do not depend on it.
- Photos of one lesion are not independent: the intervals resample whole lesions (248 lesions for 320 images).
- A dash means undefined: precision when a class was never answered, recall when no image has that class. A class that was never answered counts as precision 0 in the macro-precision.
- With one answer per image, micro-averaged precision, recall and F1 equal accuracy, so they are not listed.
- Hallucination: an answer that names a wrong diagnosis. Right answers, hallucinations and answers with no valid answer (a refusal, a sentence, two classes, a cut-off) add up to every question; a model that refuses is not hallucinating, but it is not right either. The models must choose a class, so none can avoid hallucinating by declaring itself unsure. With the same number of images in every class, the two rates are the same.
- Missed malignancy: melanoma, basal cell carcinoma and squamous cell carcinoma answered as nevus, pigmented benign keratosis, dermatofibroma or vascular lesion, as a share of all cancers. Over-call: the reverse, as a share of all benign lesions. An unreadable answer, or actinic keratosis (precancerous) given for a cancer, is no miss but stays in the total; actinic keratosis as a true class is in neither group.
- Malignancy sensitivity: cancers answered as any cancer (the right one or another), as a share of all cancers. It is not 100% minus the missed malignancy: a cancer answered as actinic keratosis, or not readably, is neither detected nor missed. Benign specificity: benign lesions answered as any benign class, as a share of all benign lesions.
- Specificity of a class: of the images that are not that class, the share not answered as it. With 8 classes most are negatives of any one class, so specificity is high by construction and is read beside recall.
- No ROC curve or AUC: the models answer with one class name, not a score per class, so there is nothing to rank. Malignancy sensitivity with benign specificity is the one point of such a curve these answers give.

## Questions with all facts recorded

A missing fact ("not recorded") points to the photo's collection, and so mostly to melanoma or nevus (data/multimodal/datasheet.md, limitation 1). These scores count only the 289 of 320 questions with every fact recorded, beside the main score, never instead of it.

| Metric | Value | 95% interval |
|---|---|---|
| Balanced accuracy (8 classes) | 54.8% | 48.7% to 60.9% (bootstrap over lesions) |
| Accuracy | 55.7% (161/289) | 49.7% to 61.6% (bootstrap over lesions) |
| Macro-F1 (classes with images: 8) | 53.8% | 47.3% to 59.8% (bootstrap over lesions) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **45.2%** | 39.1% to 51.3% (bootstrap over lesions) |
| Hallucination rate, all questions | 44.3% (128/289) | 38.4% to 50.3% (bootstrap over lesions) |
| Malignancy sensitivity: cancers answered as a cancer | 67.0% (75/112) | 57.1% to 76.4% (bootstrap over lesions) |
| Missed malignancy: cancers answered as benign | 19.6% (22/112) | 11.9% to 28.8% (bootstrap over lesions) |
| Benign specificity: benign lesions answered as benign | 79.6% (109/137) | 71.6% to 87.4% (bootstrap over lesions) |
| Over-calls: benign lesions answered as cancer | 19.0% (26/137) | 11.1% to 27.0% (bootstrap over lesions) |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: 27c7ec0f34
- Interruptions before the run finished: 0
