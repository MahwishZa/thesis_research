# claude-sonnet-5-5: test split, run test_16d5a224

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 512, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Dataset `clinic_photo_record`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-19T19:37:08+00:00, finished 2026-11-20T01:38:19+00:00 (UTC)
- Prompt: `prompts/clinic_photo_record/closed_book.txt` (SHA-256 `921fcbf8e5c9`); photos: 678 (hash `af552c9257cc`), sent exactly as stored, each with its patient's facts in the prompt; dataset table `data/clinic_patients/dataset.csv` (SHA-256 `688206d50bcf`); Git commit `ecb2f8d407`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **64.5%** | 58.6% to 70.0% (bootstrap over patients) |
| Macro-F1 | 58.8% | 54.0% to 63.5% (bootstrap over patients) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 95.6% | 93.8% to 97.2% (bootstrap over patients) |
| Invalid answers (counted as wrong) | 0.1% (1/678) | |

### Selective prediction

Over the 677 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 0.4% (3/678) | 0.0% to 1.0% |
| Selective risk (confident misdiagnosis): wrong among them | 0.0% (0/3) | – |

### Per class

| Class | Photos | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 217 | 49.8% (42.6% to 57.3%) | 88.5% | 63.7% |
| basal cell carcinoma | 250 | 77.6% (72.0% to 83.3%) | 70.8% | 74.0% |
| melanoma | 15 | 80.0% (53.3% to 100.0%) | 42.9% | 55.8% |
| nevus | 73 | 78.1% (67.6% to 87.0%) | 70.4% | 74.0% |
| seborrheic keratosis | 70 | 35.7% (23.3% to 47.9%) | 78.1% | 49.0% |
| squamous cell carcinoma | 53 | 66.0% (52.8% to 78.8%) | 25.0% | 36.3% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | melanoma | nevus | seborrheic keratosis | squamous cell carcinoma | (invalid) |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 108 | 47 | 2 | 2 | 2 | 56 | 0 |
| basal cell carcinoma | 7 | 194 | 2 | 1 | 0 | 46 | 0 |
| melanoma | 0 | 0 | 12 | 3 | 0 | 0 | 0 |
| nevus | 1 | 9 | 0 | 57 | 5 | 0 | 1 |
| seborrheic keratosis | 4 | 9 | 11 | 18 | 25 | 3 | 0 |
| squamous cell carcinoma | 2 | 15 | 1 | 0 | 0 | 35 | 0 |

- Photos of one patient are not independent: the intervals resample whole patients (498 patients for 678 photos).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Questions with all facts recorded

A missing fact ("unknown") can itself point to the class: which facts are missing is not the same for every class (see the data sheet of the dataset). These scores count only the 670 of 678 questions with every fact recorded, beside the main score, never instead of it.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **64.7%** | 58.8% to 69.9% (bootstrap over patients) |
| Macro-F1 | 59.2% | 54.2% to 64.0% (bootstrap over patients) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 95.8% | 94.1% to 97.4% (bootstrap over patients) |
| Invalid answers (counted as wrong) | 0.1% (1/670) | |

## Notes

- How answers ended (finish reason): STOP: 678
- Why answers were invalid: a claim must have the key statement, may have the key evidence, and no other: 1
- Sessions (starts and restarts of this run): 2; Git commits: ca60c35b1f, ecb2f8d407; **the run spanned more than one commit, so its answers do not all come from the same code**
- Interruptions before the run finished: 1; 2026-11-19T19:38:58+00:00 after 25 answers (FatalError)
