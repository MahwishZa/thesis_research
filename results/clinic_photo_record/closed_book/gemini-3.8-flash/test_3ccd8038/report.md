# gemini-3.8-flash: test split, run test_3ccd8038

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Dataset `clinic_photo_record`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-20T05:00:35+00:00, finished 2026-11-20T07:13:02+00:00 (UTC)
- Prompt: `prompts/clinic_photo_record/closed_book.txt` (SHA-256 `921fcbf8e5c9`); photos: 678 (hash `af552c9257cc`), sent exactly as stored, each with its patient's facts in the prompt; dataset table `data/clinic_patients/dataset.csv` (SHA-256 `688206d50bcf`); Git commit `d933159c5b`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **64.4%** | 58.5% to 69.7% (bootstrap over patients) |
| Macro-F1 | 63.4% | 58.0% to 68.4% (bootstrap over patients) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 95.4% | 93.6% to 97.1% (bootstrap over patients) |
| Invalid answers (counted as wrong) | 0.7% (5/678) | |

### Selective prediction

Over the 673 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 20.8% (141/678) | 17.6% to 24.1% |
| Selective risk (confident misdiagnosis): wrong among them | 9.2% (13/141) | 4.9% to 14.2% |

### Per class

| Class | Photos | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 217 | 52.1% (44.9% to 59.6%) | 90.4% | 66.1% |
| basal cell carcinoma | 250 | 86.8% (82.4% to 91.3%) | 65.6% | 74.7% |
| melanoma | 15 | 73.3% (46.2% to 93.8%) | 55.0% | 62.9% |
| nevus | 73 | 65.8% (53.4% to 77.0%) | 88.9% | 75.6% |
| seborrheic keratosis | 70 | 72.9% (62.3% to 82.9%) | 65.4% | 68.9% |
| squamous cell carcinoma | 53 | 35.8% (21.6% to 50.9%) | 29.2% | 32.2% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | melanoma | nevus | seborrheic keratosis | squamous cell carcinoma | (invalid) |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 113 | 64 | 1 | 2 | 11 | 26 | 0 |
| basal cell carcinoma | 8 | 217 | 2 | 0 | 0 | 19 | 4 |
| melanoma | 0 | 0 | 11 | 1 | 3 | 0 | 0 |
| nevus | 1 | 10 | 1 | 48 | 13 | 0 | 0 |
| seborrheic keratosis | 1 | 9 | 5 | 3 | 51 | 1 | 0 |
| squamous cell carcinoma | 2 | 31 | 0 | 0 | 0 | 19 | 1 |

- Photos of one patient are not independent: the intervals resample whole patients (498 patients for 678 photos).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Questions with all facts recorded

A missing fact ("unknown") can itself point to the class: which facts are missing is not the same for every class (see the data sheet of the dataset). These scores count only the 670 of 678 questions with every fact recorded, beside the main score, never instead of it.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **65.1%** | 59.4% to 70.2% (bootstrap over patients) |
| Macro-F1 | 64.0% | 58.8% to 68.8% (bootstrap over patients) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 95.7% | 93.7% to 97.2% (bootstrap over patients) |
| Invalid answers (counted as wrong) | 0.7% (5/670) | |

## Notes

- How answers ended (finish reason): STOP: 678
- Why answers were invalid: not a class of the list: 'amelanotic melanoma': 4; not a class of the list: 'keratoacanthoma': 1
- Sessions (starts and restarts of this run): 1; Git commits: d933159c5b
- Interruptions before the run finished: 0
