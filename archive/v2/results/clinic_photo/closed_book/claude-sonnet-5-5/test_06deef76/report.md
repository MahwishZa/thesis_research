# claude-sonnet-5-5: test split, run test_06deef76

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 512, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Dataset `clinic_photo`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-19T18:51:11+00:00, finished 2026-11-19T19:36:58+00:00 (UTC)
- Prompt: `prompts/clinic_photo/closed_book.txt` (SHA-256 `68b9b55d82fd`); photos: 678 (hash `e33647324386`), sent exactly as stored; dataset table `data/clinic_patients/dataset.csv` (SHA-256 `688206d50bcf`); Git commit `ecb2f8d407`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **58.0%** | 52.7% to 62.8% (bootstrap over patients) |
| Macro-F1 | 52.8% | 48.1% to 57.5% (bootstrap over patients) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 94.3% | 92.2% to 96.1% (bootstrap over patients) |
| Invalid answers (counted as wrong) | 1.2% (8/678) | |

### Selective prediction

Over the 670 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 0.1% (1/678) | 0.0% to 0.5% |
| Selective risk (confident misdiagnosis): wrong among them | 0.0% (0/1) | – |

### Per class

| Class | Photos | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 217 | 37.8% (31.0% to 45.1%) | 85.4% | 52.4% |
| basal cell carcinoma | 250 | 65.2% (58.6% to 72.0%) | 64.2% | 64.7% |
| melanoma | 15 | 80.0% (61.5% to 100.0%) | 46.2% | 58.5% |
| nevus | 73 | 63.0% (50.0% to 75.0%) | 67.6% | 65.2% |
| seborrheic keratosis | 70 | 35.7% (23.0% to 47.9%) | 69.4% | 47.2% |
| squamous cell carcinoma | 53 | 66.0% (51.9% to 78.8%) | 18.4% | 28.8% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | melanoma | nevus | seborrheic keratosis | squamous cell carcinoma | (invalid) |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 82 | 55 | 0 | 0 | 1 | 75 | 4 |
| basal cell carcinoma | 5 | 163 | 1 | 1 | 2 | 74 | 4 |
| melanoma | 0 | 0 | 12 | 2 | 1 | 0 | 0 |
| nevus | 1 | 12 | 5 | 46 | 7 | 2 | 0 |
| seborrheic keratosis | 6 | 8 | 8 | 19 | 25 | 4 | 0 |
| squamous cell carcinoma | 2 | 16 | 0 | 0 | 0 | 35 | 0 |

- Photos of one patient are not independent: the intervals resample whole patients (498 patients for 678 photos).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Notes

- How answers ended (finish reason): STOP: 678
- Why answers were invalid: a claim must have the key statement, may have the key evidence, and no other: 7; missing key evidence: 1
- Sessions (starts and restarts of this run): 1; Git commits: ecb2f8d407
- Interruptions before the run finished: 0
