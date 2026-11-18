# claude-sonnet-5-5: test split, run test_06ce1664

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 256, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Started 2026-11-14T17:12:56+00:00, finished 2026-11-14T17:27:37+00:00 (UTC)
- Prompt: `prompts/text/diagnosis.txt` (SHA-256 `5b6fe4db8d52`); cases: 221 (hash `a75d8b54c810`), each put into the prompt word for word; dataset table `data/text/dataset.csv` (SHA-256 `f4d40e300b62`); Git commit `8dad1045d9`

## Results

| Metric | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score: the mean recall of the classes)** | **55.9%** | 49.6% to 61.8% (bootstrap over articles) |
| Accuracy | 64.3% (142/221) | 57.9% to 70.6% (bootstrap over articles) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **34.1%** | 24.4% to 43.8% (bootstrap over articles) |
| Hallucination rate, all questions | 31.2% (69/221) | 25.1% to 37.4% (bootstrap over articles) |
| No valid answer (a refusal, a sentence, two classes, a cut-off: counted as wrong, but no hallucination) | 10 | |
| Malignancy sensitivity: cancers answered as a cancer | 90.8% (109/120) | 85.8% to 95.8% (bootstrap over articles) |
| Missed malignancy: cancers answered as benign | 5.8% (7/120) | 1.7% to 10.0% (bootstrap over articles) |
| Benign specificity: benign lesions answered as benign | 66.7% (66/99) | 57.6% to 75.8% (bootstrap over articles) |
| Over-calls: benign lesions answered as cancer | 27.3% (27/99) | 18.8% to 36.4% (bootstrap over articles) |
| Macro-F1 (classes with cases: 8) | 57.9% | 51.4% to 63.5% (bootstrap over articles) |
| Macro-precision | 63.9% | 58.3% to 69.5% (bootstrap over articles) |
| Macro-specificity | 95.3% | 94.4% to 96.2% (bootstrap over articles) |
| Weighted F1 (by number of true cases) | 65.2% | |
| Matthews correlation (MCC, -1 to 1) | 0.580 | |
| Cohen's kappa (-1 to 1) | 0.577 | |

### Per class

| Class | True cases | Answered as this class | Precision | Recall (95% interval: bootstrap over articles) | F1 | Specificity (95% interval: bootstrap over articles) |
|---|---|---|---|---|---|---|
| actinic keratosis | 2 | 1 | 0.0% | 0.0% (0.0% to 0.0%) | 0.0% | 99.5% (98.6% to 100.0%) |
| basal cell carcinoma | 40 | 43 | 60.5% | 65.0% (50.0% to 80.0%) | 62.7% | 90.6% (86.6% to 94.5%) |
| dermatofibroma | 21 | 14 | 85.7% | 57.1% (31.6% to 78.3%) | 68.6% | 99.0% (97.5% to 100.0%) |
| melanoma | 40 | 44 | 63.6% | 70.0% (55.0% to 82.5%) | 66.7% | 91.2% (86.6% to 95.1%) |
| nevus | 40 | 33 | 75.8% | 62.5% (47.5% to 77.5%) | 68.5% | 95.6% (92.7% to 98.3%) |
| pigmented benign keratosis | 22 | 9 | 88.9% | 36.4% (18.2% to 54.5%) | 51.6% | 99.5% (98.5% to 100.0%) |
| squamous cell carcinoma | 40 | 50 | 60.0% | 75.0% (60.0% to 87.5%) | 66.7% | 89.0% (84.4% to 93.3%) |
| vascular lesion | 16 | 17 | 76.5% | 81.2% (62.5% to 100.0%) | 78.8% | 98.0% (96.1% to 99.5%) |

### Confusion matrix

Rows: the true class. Columns: the answer, numbered like the rows; the last column counts unreadable answers.

| True class | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | unreadable |
|---|---|---|---|---|---|---|---|---|---|
| 1. actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| 2. basal cell carcinoma | 0 | 26 | 0 | 3 | 0 | 1 | 9 | 0 | 1 |
| 3. dermatofibroma | 0 | 4 | 12 | 2 | 1 | 0 | 1 | 1 | 0 |
| 4. melanoma | 0 | 4 | 0 | 28 | 1 | 0 | 5 | 2 | 0 |
| 5. nevus | 0 | 2 | 1 | 8 | 25 | 0 | 0 | 1 | 3 |
| 6. pigmented benign keratosis | 0 | 4 | 0 | 1 | 4 | 8 | 3 | 0 | 2 |
| 7. squamous cell carcinoma | 1 | 3 | 1 | 1 | 2 | 0 | 30 | 0 | 2 |
| 8. vascular lesion | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 13 | 1 |

Notes on these scores:
- Balanced accuracy is the main score: every class counts the same, whatever its number of test cases (the measure of the ISIC 2018 lesion-diagnosis challenge).
- Precision and F1 depend on how common each class is in the test set, and the test set's mix of classes is not a clinic's: a rare disease is far more common here, so its precision is higher than it would be in a clinic, and it is not a clinical figure. Recall and balanced accuracy do not depend on it.
- Patients described in one article are not independent: the intervals resample whole articles (219 articles for 221 cases).
- A dash means undefined: precision when a class was never answered, recall when no case has that class. A class that was never answered counts as precision 0 in the macro-precision.
- With one answer per case, micro-averaged precision, recall and F1 equal accuracy, so they are not listed.
- Hallucination: an answer that names a wrong diagnosis. Right answers, hallucinations and answers with no valid answer (a refusal, a sentence, two classes, a cut-off) add up to every question; a model that refuses is not hallucinating, but it is not right either. The models must choose a class, so none can avoid hallucinating by declaring itself unsure. With the same number of cases in every class, the two rates are the same.
- Missed malignancy: melanoma, basal cell carcinoma and squamous cell carcinoma answered as nevus, pigmented benign keratosis, dermatofibroma or vascular lesion, as a share of all cancers. Over-call: the reverse, as a share of all benign lesions. An unreadable answer, or actinic keratosis (precancerous) given for a cancer, is no miss but stays in the total; actinic keratosis as a true class is in neither group.
- Malignancy sensitivity: cancers answered as any cancer (the right one or another), as a share of all cancers. It is not 100% minus the missed malignancy: a cancer answered as actinic keratosis, or not readably, is neither detected nor missed. Benign specificity: benign lesions answered as any benign class, as a share of all benign lesions.
- Specificity of a class: of the cases that are not that class, the share not answered as it. With 8 classes most are negatives of any one class, so specificity is high by construction and is read beside recall.
- No ROC curve or AUC: the models answer with one class name, not a score per class, so there is nothing to rank. Malignancy sensitivity with benign specificity is the one point of such a curve these answers give.

## Without actinic keratosis

The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Metric | Value | 95% interval |
|---|---|---|
| Balanced accuracy (7 classes) | 63.9% | 57.1% to 70.6% (bootstrap over articles) |
| Accuracy | 64.8% (142/219) | 58.5% to 71.0% (bootstrap over articles) |
| Macro-F1 (classes with cases: 7) | 66.3% | 59.1% to 72.7% (bootstrap over articles) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **31.8%** | 25.3% to 38.5% (bootstrap over articles) |
| Hallucination rate, all questions | 31.1% (68/219) | 24.9% to 37.0% (bootstrap over articles) |
| Malignancy sensitivity: cancers answered as a cancer | 90.8% (109/120) | 85.0% to 95.8% (bootstrap over articles) |
| Missed malignancy: cancers answered as benign | 5.8% (7/120) | 1.7% to 10.8% (bootstrap over articles) |
| Benign specificity: benign lesions answered as benign | 66.7% (66/99) | 57.6% to 75.8% (bootstrap over articles) |
| Over-calls: benign lesions answered as cancer | 27.3% (27/99) | 18.6% to 36.1% (bootstrap over articles) |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | (unreadable) |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | squamous cell carcinoma |

## Sensitivity analysis: the last line of the answer

The main score reads an answer only when it is exactly one class name, as the prompt asks. 9 answers wrote their reasoning first and then a class name alone on the last line; 7 of them name the right class. Here they are read as that class. This is a check on the main score, never a replacement for it.

| Metric | Value | 95% interval |
|---|---|---|
| Balanced accuracy (8 classes) | 59.1% | 52.9% to 64.7% (bootstrap over articles) |
| Accuracy | 67.4% (149/221) | 61.2% to 73.5% (bootstrap over articles) |
| Macro-F1 (classes with cases: 8) | 60.1% | 53.8% to 65.5% (bootstrap over articles) |
| **Hallucination rate (a wrong diagnosis named; mean over the classes)** | **40.6%** | 35.1% to 46.8% (bootstrap over articles) |
| Hallucination rate, all questions | 32.1% (71/221) | 26.0% to 38.4% (bootstrap over articles) |
| Malignancy sensitivity: cancers answered as a cancer | 93.3% (112/120) | 89.2% to 97.5% (bootstrap over articles) |
| Missed malignancy: cancers answered as benign | 5.8% (7/120) | 1.7% to 10.0% (bootstrap over articles) |
| Benign specificity: benign lesions answered as benign | 70.7% (70/99) | 60.8% to 79.6% (bootstrap over articles) |
| Over-calls: benign lesions answered as cancer | 28.3% (28/99) | 19.8% to 37.4% (bootstrap over articles) |

## Notes

- How answers ended (finish reason): MAX_TOKENS: 1, STOP: 220
- 1 answers reached the output limit but already differed from every class name before it (an essay, or a second class name), so the limit did not change their score; they count as wrong.
- Sessions (starts and restarts of this run): 1; Git commits: 8dad1045d9
- Interruptions before the run finished: 0
