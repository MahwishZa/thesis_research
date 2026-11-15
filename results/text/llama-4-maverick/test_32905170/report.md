# llama-4-maverick: test split, run test_32905170

- Model requested: `meta-llama/llama-4-maverick`; versions returned by the API: `meta-llama/llama-4-maverick@Parasail`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 256, "image_detail": null, "extra_body": {"provider": {"order": ["parasail/fp8"], "allow_fallbacks": false}}}`
- Started 2026-11-15T02:27:08+00:00, finished 2026-11-15T02:38:18+00:00 (UTC)
- Prompt: `prompts/text/diagnosis.txt` (SHA-256 `5b6fe4db8d52`); cases: 221 (hash `a75d8b54c810`), each put into the prompt word for word; dataset table `data/text/dataset.csv` (SHA-256 `f4d40e300b62`); Git commit `387fcfe590`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **46.8%** | 40.6% to 52.8% (bootstrap over articles) |
| Accuracy | 53.8% (119/221) | 47.5% to 60.2% (bootstrap over articles) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **48.5%** | 42.6% to 54.6% (bootstrap over articles) |
| Hallucination rate, all questions | 41.2% (91/221) | 35.0% to 47.1% (bootstrap over articles) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 11 | |
| Missed malignancy: cancers answered as benign | 10.0% (12/120) | 5.0% to 15.8% (bootstrap over articles) |
| Over-calls: benign lesions answered as cancer | 24.2% (24/99) | 16.5% to 33.0% (bootstrap over articles) |
| Macro-F1 (classes with cases: 8) | 47.0% | 40.0% to 53.0% (bootstrap over articles) |
| Macro-precision | 53.8% | |
| Weighted F1 (by number of true cases) | 54.4% | |
| Matthews correlation (MCC, -1 to 1) | 0.466 | |
| Cohen's kappa (-1 to 1) | 0.458 | |

### Per class

| Class | True cases | Answered as this class | Precision | Recall (95% interval: bootstrap over articles) | F1 |
|---|---|---|---|---|---|
| actinic keratosis | 2 | 2 | 0.0% | 0.0% (0.0% to 0.0%) | 0.0% |
| basal cell carcinoma | 40 | 19 | 84.2% | 40.0% (25.0% to 55.0%) | 54.2% |
| dermatofibroma | 21 | 8 | 75.0% | 28.6% (5.3% to 52.2%) | 41.4% |
| melanoma | 40 | 49 | 46.9% | 57.5% (42.5% to 72.5%) | 51.7% |
| nevus | 40 | 31 | 67.7% | 52.5% (37.5% to 67.5%) | 59.2% |
| pigmented benign keratosis | 22 | 16 | 56.2% | 40.9% (18.2% to 59.1%) | 47.4% |
| squamous cell carcinoma | 40 | 59 | 54.2% | 80.0% (67.5% to 92.5%) | 64.6% |
| vascular lesion | 16 | 26 | 46.2% | 75.0% (50.0% to 93.8%) | 57.1% |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| 2. basal cell carcinoma | 0 | 16 | 0 | 8 | 0 | 1 | 12 | 2 | 1 |
| 3. dermatofibroma | 0 | 2 | 6 | 2 | 4 | 1 | 0 | 4 | 2 |
| 4. melanoma | 0 | 0 | 1 | 23 | 1 | 0 | 9 | 2 | 4 |
| 5. nevus | 0 | 1 | 0 | 9 | 21 | 2 | 0 | 5 | 2 |
| 6. pigmented benign keratosis | 0 | 0 | 0 | 5 | 3 | 9 | 3 | 1 | 1 |
| 7. squamous cell carcinoma | 2 | 0 | 1 | 1 | 1 | 3 | 32 | 0 | 0 |
| 8. vascular lesion | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 12 | 1 |

Notes on these scores:
- Balanced accuracy is the main score: every class counts the same, whatever its number of test cases (the measure of the ISIC 2018 lesion-diagnosis challenge).
- Precision and F1 depend on how common each class is in the test set, and the test set's mix of classes is not a clinic's: a rare disease is far more common here, so its precision is higher than it would be in a clinic, and it is not a clinical figure. Recall and balanced accuracy do not depend on it.
- Patients described in one article are not independent: the intervals resample whole articles (219 articles for 221 cases).
- A dash means undefined: precision when a class was never answered, recall when no case has that class. A class that was never answered counts as precision 0 in the macro-precision.
- With one answer per case, micro-averaged precision, recall and F1 equal accuracy, so they are not listed.
- Hallucination: an answer that names a wrong diagnosis. Right answers, hallucinations and answers with no valid answer (a refusal, a sentence, two classes, a cut-off) add up to every question; a model that refuses is not hallucinating, but it is not right either. The models must choose a class, so none can avoid hallucinating by declaring itself unsure. With the same number of cases in every class, the two rates are the same.
- Missed malignancy: melanoma, basal cell carcinoma and squamous cell carcinoma answered as nevus, pigmented benign keratosis, dermatofibroma or vascular lesion, as a share of all cancers. Over-call: the reverse, as a share of all benign lesions. An unreadable answer, or actinic keratosis (precancerous) given for a cancer, is no miss but stays in the total; actinic keratosis as a true class is in neither group.
- No ROC curve or AUC: the models answer with one class name, not a score per class, so there is nothing to rank.

## Without actinic keratosis

The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Metric | Value | 95% interval |
|---|---|---|
| Balanced accuracy (7 classes) | 53.5% | 46.1% to 60.6% (bootstrap over articles) |
| Accuracy | 54.3% (119/219) | 47.9% to 60.8% (bootstrap over articles) |
| Macro-F1 (classes with cases: 7) | 53.8% | 45.7% to 61.1% (bootstrap over articles) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **41.1%** | 34.3% to 48.2% (bootstrap over articles) |
| Hallucination rate, all questions | 40.6% (89/219) | 34.5% to 47.0% (bootstrap over articles) |
| Missed malignancy: cancers answered as benign | 10.0% (12/120) | 5.0% to 15.8% (bootstrap over articles) |
| Over-calls: benign lesions answered as cancer | 24.2% (24/99) | 16.2% to 33.0% (bootstrap over articles) |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | squamous cell carcinoma |

## Sensitivity analysis: the last line of the answer

The main score reads an answer only when it is exactly one class name, as the prompt asks. 1 answer wrote its reasoning first and then a class name alone on the last line; 1 of them names the right class. Here it is read as that class. This is a check on the main score, never a replacement for it.

| Metric | Value | 95% interval |
|---|---|---|
| Balanced accuracy (8 classes) | 47.1% | 40.9% to 53.1% (bootstrap over articles) |
| Accuracy | 54.3% (120/221) | 47.9% to 60.6% (bootstrap over articles) |
| Macro-F1 (classes with cases: 8) | 47.2% | 40.2% to 53.2% (bootstrap over articles) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **48.5%** | 42.6% to 54.6% (bootstrap over articles) |
| Hallucination rate, all questions | 41.2% (91/221) | 35.0% to 47.1% (bootstrap over articles) |
| Missed malignancy: cancers answered as benign | 10.0% (12/120) | 5.0% to 15.8% (bootstrap over articles) |
| Over-calls: benign lesions answered as cancer | 24.2% (24/99) | 16.5% to 33.0% (bootstrap over articles) |

## Notes

- How answers ended (finish reason): MAX_TOKENS: 9, STOP: 212
- 9 answers reached the output limit but already differed from every class name before it (an essay, or a second class name), so the limit did not change their score; they count as wrong.
- Sessions (starts and restarts of this run): 1; Git commits: 387fcfe590
- Interruptions before the run finished: 0
