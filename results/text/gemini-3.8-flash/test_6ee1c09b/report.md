# gemini-3.8-flash: test split, run test_6ee1c09b

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Started 2026-11-15T00:45:56+00:00, finished 2026-11-15T01:00:00+00:00 (UTC)
- Prompt: `prompts/text/diagnosis.txt` (SHA-256 `5b6fe4db8d52`); cases: 221 (hash `a75d8b54c810`), each put into the prompt word for word; dataset table `data/text/dataset.csv` (SHA-256 `f4d40e300b62`); Git commit `27c7ec0f34`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **61.4%** | 55.8% to 67.0% (bootstrap over articles) |
| Accuracy | 69.2% (153/221) | 63.7% to 75.1% (bootstrap over articles) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **38.6%** | 33.0% to 44.2% (bootstrap over articles) |
| Hallucination rate, all questions | 30.8% (68/221) | 24.9% to 36.3% (bootstrap over articles) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 0 | |
| Missed malignancy: cancers answered as benign | 5.0% (6/120) | 1.7% to 9.2% (bootstrap over articles) |
| Over-calls: benign lesions answered as cancer | 31.3% (31/99) | 22.7% to 40.4% (bootstrap over articles) |
| Macro-F1 (classes with cases: 8) | 61.3% | 55.8% to 66.7% (bootstrap over articles) |
| Macro-precision | 65.1% | |
| Weighted F1 (by number of true cases) | 68.9% | |
| Matthews correlation (MCC, -1 to 1) | 0.643 | |
| Cohen's kappa (-1 to 1) | 0.635 | |

### Per class

| Class | True cases | Answered as this class | Precision | Recall (95% interval: bootstrap over articles) | F1 |
|---|---|---|---|---|---|
| actinic keratosis | 2 | 2 | 0.0% | 0.0% (0.0% to 0.0%) | 0.0% |
| basal cell carcinoma | 40 | 38 | 68.4% | 65.0% (50.0% to 80.0%) | 66.7% |
| dermatofibroma | 21 | 15 | 86.7% | 61.9% (36.8% to 82.6%) | 72.2% |
| melanoma | 40 | 60 | 60.0% | 90.0% (80.0% to 97.5%) | 72.0% |
| nevus | 40 | 19 | 100.0% | 47.5% (32.5% to 62.5%) | 64.4% |
| pigmented benign keratosis | 22 | 20 | 75.0% | 68.2% (50.0% to 86.4%) | 71.4% |
| squamous cell carcinoma | 40 | 47 | 66.0% | 77.5% (62.5% to 90.0%) | 71.3% |
| vascular lesion | 16 | 20 | 65.0% | 81.2% (62.5% to 100.0%) | 72.2% |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| 2. basal cell carcinoma | 0 | 26 | 0 | 6 | 0 | 2 | 6 | 0 | 0 |
| 3. dermatofibroma | 0 | 4 | 13 | 2 | 0 | 0 | 1 | 1 | 0 |
| 4. melanoma | 0 | 0 | 0 | 36 | 0 | 0 | 3 | 1 | 0 |
| 5. nevus | 0 | 2 | 1 | 12 | 19 | 2 | 1 | 3 | 0 |
| 6. pigmented benign keratosis | 0 | 3 | 1 | 1 | 0 | 15 | 2 | 0 | 0 |
| 7. squamous cell carcinoma | 2 | 2 | 0 | 2 | 0 | 1 | 31 | 2 | 0 |
| 8. vascular lesion | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 13 | 0 |

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
| Balanced accuracy (7 classes) | 70.2% | 63.7% to 76.3% (bootstrap over articles) |
| Accuracy | 69.9% (153/219) | 64.1% to 75.6% (bootstrap over articles) |
| Macro-F1 (classes with cases: 7) | 70.3% | 63.7% to 76.1% (bootstrap over articles) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **29.8%** | 23.7% to 36.3% (bootstrap over articles) |
| Hallucination rate, all questions | 30.1% (66/219) | 24.4% to 35.9% (bootstrap over articles) |
| Missed malignancy: cancers answered as benign | 5.0% (6/120) | 1.7% to 9.2% (bootstrap over articles) |
| Over-calls: benign lesions answered as cancer | 31.3% (31/99) | 22.7% to 40.6% (bootstrap over articles) |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 221
- Sessions (starts and restarts of this run): 1; Git commits: 27c7ec0f34
- Interruptions before the run finished: 0
