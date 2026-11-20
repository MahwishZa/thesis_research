# claude-sonnet-5-5: test split, run test_7ae640ea

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 512, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Dataset `clinic_record`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-19T17:50:24+00:00, finished 2026-11-19T18:36:28+00:00 (UTC)
- Prompt: `prompts/clinic_record/closed_book.txt` (SHA-256 `419f79cdeab8`); records: 678 (hash `56c8a830ad2e`), each put into the prompt word for word; dataset table `data/clinic_patients/record.csv` (SHA-256 `854f1ebef050`); Git commit `ecb2f8d407`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **42.0%** | 38.6% to 45.9% (bootstrap over patients) |
| Macro-F1 | 40.3% | 36.9% to 44.0% (bootstrap over patients) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 67.4% | 63.5% to 71.9% (bootstrap over patients) |
| Invalid answers (counted as wrong) | 0.0% (0/678) | |

### Selective prediction

Over the 678 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 0.0% (0/678) | 0.0% to 0.0% |
| Selective risk (confident misdiagnosis): wrong among them | – (0/0) | – |

### Per class

| Class | Records | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 217 | 58.5% (51.4% to 66.0%) | 74.7% | 65.6% |
| basal cell carcinoma | 250 | 70.8% (64.0% to 77.1%) | 66.3% | 68.5% |
| melanoma | 15 | 0.0% (0.0% to 0.0%) | – | 0.0% |
| nevus | 73 | 74.0% (62.7% to 84.3%) | 52.4% | 61.4% |
| seborrheic keratosis | 70 | 30.0% (19.4% to 42.5%) | 21.9% | 25.3% |
| squamous cell carcinoma | 53 | 18.9% (7.7% to 32.7%) | 23.8% | 21.1% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | melanoma | nevus | seborrheic keratosis | squamous cell carcinoma | (invalid) |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 127 | 38 | 0 | 18 | 26 | 8 | 0 |
| basal cell carcinoma | 16 | 177 | 0 | 7 | 28 | 22 | 0 |
| melanoma | 1 | 0 | 0 | 10 | 4 | 0 | 0 |
| nevus | 3 | 9 | 0 | 54 | 6 | 1 | 0 |
| seborrheic keratosis | 13 | 23 | 0 | 12 | 21 | 1 | 0 |
| squamous cell carcinoma | 10 | 20 | 0 | 2 | 11 | 10 | 0 |

- Photos and records of one patient are not independent: the intervals resample whole patients (498 patients for 678 records).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Questions with all facts recorded

A missing fact ("unknown") can itself point to the class: which facts are missing is not the same for every class (see the data sheet of the dataset). These scores count only the 670 of 678 questions with every fact recorded, beside the main score, never instead of it.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **42.1%** | 38.3% to 45.9% (bootstrap over patients) |
| Macro-F1 | 40.2% | 36.7% to 43.8% (bootstrap over patients) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 67.7% | 63.9% to 72.2% (bootstrap over patients) |
| Invalid answers (counted as wrong) | 0.0% (0/670) | |

## Notes

- How answers ended (finish reason): STOP: 678
- Sessions (starts and restarts of this run): 1; Git commits: ecb2f8d407
- Interruptions before the run finished: 0
