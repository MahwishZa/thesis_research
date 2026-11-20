# gemini-3.8-flash: test split, run test_d9a96c51

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Dataset `clinic_photo`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-20T03:03:29+00:00, finished 2026-11-20T05:00:31+00:00 (UTC)
- Prompt: `prompts/clinic_photo/closed_book.txt` (SHA-256 `68b9b55d82fd`); photos: 678 (hash `e33647324386`), sent exactly as stored; dataset table `data/clinic_patients/dataset.csv` (SHA-256 `688206d50bcf`); Git commit `129cfce169`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **53.7%** | 48.3% to 59.1% (bootstrap over patients) |
| Macro-F1 | 53.2% | 47.9% to 58.0% (bootstrap over patients) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 96.0% | 94.3% to 97.7% (bootstrap over patients) |
| Invalid answers (counted as wrong) | 0.4% (3/678) | |

### Selective prediction

Over the 675 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 22.7% (154/678) | 19.5% to 26.0% |
| Selective risk (confident misdiagnosis): wrong among them | 17.5% (27/154) | 11.8% to 23.6% |

### Per class

| Class | Photos | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 217 | 41.5% (34.5% to 48.6%) | 84.9% | 55.7% |
| basal cell carcinoma | 250 | 86.4% (81.6% to 91.0%) | 60.5% | 71.2% |
| melanoma | 15 | 46.7% (25.0% to 69.2%) | 38.9% | 42.4% |
| nevus | 73 | 43.8% (31.4% to 54.9%) | 91.4% | 59.3% |
| seborrheic keratosis | 70 | 70.0% (58.6% to 80.0%) | 54.4% | 61.3% |
| squamous cell carcinoma | 53 | 34.0% (19.2% to 49.1%) | 26.1% | 29.5% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | melanoma | nevus | seborrheic keratosis | squamous cell carcinoma | (invalid) |
|---|---|---|---|---|---|---|---|
| actinic keratosis | 90 | 87 | 1 | 0 | 9 | 29 | 1 |
| basal cell carcinoma | 11 | 216 | 0 | 0 | 1 | 21 | 1 |
| melanoma | 0 | 0 | 7 | 0 | 8 | 0 | 0 |
| nevus | 0 | 15 | 3 | 32 | 23 | 0 | 0 |
| seborrheic keratosis | 3 | 7 | 7 | 3 | 49 | 1 | 0 |
| squamous cell carcinoma | 2 | 32 | 0 | 0 | 0 | 18 | 1 |

- Photos of one patient are not independent: the intervals resample whole patients (498 patients for 678 photos).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Notes

- How answers ended (finish reason): STOP: 678
- Why answers were invalid: not a class of the list: 'amelanotic melanoma': 2; not a class of the list: 'keratoacanthoma': 1
- Sessions (starts and restarts of this run): 3; Git commits: 129cfce169, d933159c5b; **the run spanned more than one commit, so its answers do not all come from the same code**
- Interruptions before the run finished: 1; 2026-11-20T03:13:30+00:00 after 102 answers (FatalError)
