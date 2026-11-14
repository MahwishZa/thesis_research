# gpt-4o: test split, run test_5fc7ce07

- Model requested: `openai/gpt-4o-2024-08-06`; versions returned by the API: `openai/gpt-4o-2024-08-06@OpenAI`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 64, "image_detail": null, "extra_body": {"provider": {"order": ["openai"], "allow_fallbacks": false}}}`
- Started 2026-11-14T23:17:12+00:00, finished 2026-11-14T23:28:13+00:00 (UTC)
- Prompt: `prompts/text/diagnosis.txt` (SHA-256 `5b6fe4db8d52`); cases: 221 (hash `a75d8b54c810`), each put into the prompt word for word; dataset table `data/text/dataset.csv` (SHA-256 `f4d40e300b62`); Git commit `f29aab5b6f`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **50.8%** | 45.1% to 56.6% (bootstrap over articles) |
| Accuracy | 56.6% (125/221) | 50.7% to 63.0% (bootstrap over articles) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **49.2%** | 43.4% to 54.9% (bootstrap over articles) |
| Hallucination rate, all questions | 43.4% (96/221) | 37.0% to 49.3% (bootstrap over articles) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 0 | |
| Missed malignancy: cancers answered as benign | 14.2% (17/120) | 8.3% to 20.8% (bootstrap over articles) |
| Over-calls: benign lesions answered as cancer | 25.3% (25/99) | 16.8% to 34.0% (bootstrap over articles) |
| Macro-F1 (classes with cases: 8) | 49.4% | 43.4% to 55.2% (bootstrap over articles) |
| Macro-precision | 54.3% | |
| Weighted F1 (by number of true cases) | 56.4% | |
| Matthews correlation (MCC, -1 to 1) | 0.496 | |
| Cohen's kappa (-1 to 1) | 0.488 | |

### Per class

| Class | True cases | Answered as this class | Precision | Recall (95% interval: bootstrap over articles) | F1 |
|---|---|---|---|---|---|
| actinic keratosis | 2 | 3 | 0.0% | 0.0% (0.0% to 0.0%) | 0.0% |
| basal cell carcinoma | 40 | 20 | 80.0% | 40.0% (25.0% to 55.0%) | 53.3% |
| dermatofibroma | 21 | 17 | 64.7% | 52.4% (26.3% to 73.9%) | 57.9% |
| melanoma | 40 | 53 | 45.3% | 60.0% (45.0% to 75.0%) | 51.6% |
| nevus | 40 | 30 | 73.3% | 55.0% (40.0% to 70.0%) | 62.9% |
| pigmented benign keratosis | 22 | 11 | 72.7% | 36.4% (18.2% to 59.1%) | 48.5% |
| squamous cell carcinoma | 40 | 54 | 55.6% | 75.0% (60.0% to 87.5%) | 63.8% |
| vascular lesion | 16 | 33 | 42.4% | 87.5% (68.8% to 100.0%) | 57.1% |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| 2. basal cell carcinoma | 1 | 16 | 1 | 10 | 0 | 0 | 9 | 3 | 0 |
| 3. dermatofibroma | 0 | 1 | 11 | 2 | 1 | 0 | 2 | 4 | 0 |
| 4. melanoma | 0 | 2 | 0 | 24 | 2 | 0 | 7 | 5 | 0 |
| 5. nevus | 0 | 0 | 3 | 8 | 22 | 2 | 0 | 5 | 0 |
| 6. pigmented benign keratosis | 0 | 1 | 1 | 6 | 3 | 8 | 3 | 0 | 0 |
| 7. squamous cell carcinoma | 2 | 0 | 1 | 2 | 2 | 1 | 30 | 2 | 0 |
| 8. vascular lesion | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 14 | 0 |

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
| Balanced accuracy (7 classes) | 58.0% | 51.4% to 64.4% (bootstrap over articles) |
| Accuracy | 57.1% (125/219) | 50.7% to 63.3% (bootstrap over articles) |
| Macro-F1 (classes with cases: 7) | 56.6% | 49.5% to 63.2% (bootstrap over articles) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **42.0%** | 35.6% to 48.6% (bootstrap over articles) |
| Hallucination rate, all questions | 42.9% (94/219) | 36.7% to 49.3% (bootstrap over articles) |
| Missed malignancy: cancers answered as benign | 14.2% (17/120) | 8.3% to 20.8% (bootstrap over articles) |
| Over-calls: benign lesions answered as cancer | 25.3% (25/99) | 17.2% to 34.0% (bootstrap over articles) |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 221
- Sessions (starts and restarts of this run): 1; Git commits: f29aab5b6f
- Interruptions before the run finished: 0
