# gemini-3.8-flash: test split, run test_702914d8

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Dataset `clinic_record`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-20T02:22:38+00:00, finished 2026-11-20T03:03:19+00:00 (UTC)
- Prompt: `prompts/clinic_record/closed_book.txt` (SHA-256 `419f79cdeab8`); records: 678 (hash `56c8a830ad2e`), each put into the prompt word for word; dataset table `data/clinic_patients/record.csv` (SHA-256 `854f1ebef050`); Git commit `129cfce169`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **45.0%** | 39.3% to 51.4% (bootstrap over patients) |
| Macro-F1 | 41.2% | 37.2% to 45.5% (bootstrap over patients) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 81.3% | 76.9% to 84.8% (bootstrap over patients) |
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
| actinic keratosis | 217 | 47.5% (40.2% to 55.0%) | 73.0% | 57.5% |
| basal cell carcinoma | 250 | 49.6% (42.5% to 57.1%) | 72.9% | 59.0% |
| melanoma | 15 | 26.7% (0.0% to 57.1%) | 8.7% | 13.1% |
| nevus | 73 | 74.0% (62.5% to 85.1%) | 60.7% | 66.7% |
| seborrheic keratosis | 70 | 38.6% (26.1% to 51.4%) | 18.5% | 25.0% |
| squamous cell carcinoma | 53 | 34.0% (20.0% to 49.1%) | 20.9% | 25.9% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | melanoma | nevus | seborrheic keratosis | squamous cell carcinoma | (invalid) |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 103 | 18 | 15 | 17 | 43 | 21 | 0 |
| basal cell carcinoma | 16 | 124 | 19 | 7 | 41 | 43 | 0 |
| melanoma | 1 | 0 | 4 | 4 | 6 | 0 | 0 |
| nevus | 3 | 3 | 0 | 54 | 12 | 1 | 0 |
| seborrheic keratosis | 14 | 17 | 3 | 6 | 27 | 3 | 0 |
| squamous cell carcinoma | 4 | 8 | 5 | 1 | 17 | 18 | 0 |

- Photos and records of one patient are not independent: the intervals resample whole patients (498 patients for 678 records).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Questions with all facts recorded

A missing fact ("unknown") can itself point to the class: which facts are missing is not the same for every class (see the data sheet of the dataset). These scores count only the 670 of 678 questions with every fact recorded, beside the main score, never instead of it.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **45.3%** | 39.4% to 51.8% (bootstrap over patients) |
| Macro-F1 | 41.4% | 37.1% to 45.6% (bootstrap over patients) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 81.7% | 77.4% to 85.4% (bootstrap over patients) |
| Invalid answers (counted as wrong) | 0.0% (0/670) | |

## Notes

- How answers ended (finish reason): STOP: 678
- Sessions (starts and restarts of this run): 1; Git commits: 129cfce169
- Interruptions before the run finished: 0
