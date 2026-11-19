# gemini-3.8-flash: test split, run test_b086a9a1

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Dataset `image`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-15T00:22:38+00:00, finished 2026-11-15T00:45:53+00:00 (UTC)
- Prompt: `prompts/image/diagnosis.txt` (SHA-256 `5e1ed90af51d`); images: 320 (hash `cfa2c9e9b7c6`), sent exactly as stored; dataset table `data/image/dataset.csv` (SHA-256 `394408d529cf`); Git commit `27c7ec0f34`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **49.7%** | 44.3% to 55.0% (bootstrap over lesions) |
| Macro-F1 | 46.2% | 41.1% to 51.2% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/320) | |

### Per class

| Class | Images | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 40 | 40.0% (22.7% to 60.5%) | 32.7% | 36.0% |
| basal cell carcinoma | 40 | 77.5% (60.5% to 92.5%) | 42.5% | 54.9% |
| dermatofibroma | 40 | 62.5% (41.5% to 81.6%) | 55.6% | 58.8% |
| melanoma | 40 | 30.0% (17.5% to 45.0%) | 36.4% | 32.9% |
| nevus | 40 | 30.0% (17.5% to 45.0%) | 66.7% | 41.4% |
| pigmented benign keratosis | 40 | 65.0% (47.2% to 81.1%) | 41.9% | 51.0% |
| squamous cell carcinoma | 40 | 0.0% (0.0% to 0.0%) | 0.0% | 0.0% |
| vascular lesion | 40 | 92.5% (81.6% to 100.0%) | 97.4% | 94.9% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 16 | 9 | 6 | 3 | 1 | 4 | 1 | 0 | 0 |
| basal cell carcinoma | 4 | 31 | 0 | 2 | 0 | 3 | 0 | 0 | 0 |
| dermatofibroma | 2 | 9 | 25 | 2 | 1 | 0 | 0 | 1 | 0 |
| melanoma | 5 | 7 | 1 | 12 | 3 | 12 | 0 | 0 | 0 |
| nevus | 1 | 1 | 7 | 6 | 12 | 13 | 0 | 0 | 0 |
| pigmented benign keratosis | 5 | 1 | 3 | 4 | 1 | 26 | 0 | 0 | 0 |
| squamous cell carcinoma | 16 | 13 | 3 | 4 | 0 | 4 | 0 | 0 | 0 |
| vascular lesion | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 37 | 0 |

- Photos of one lesion are not independent: the intervals resample whole lesions (248 lesions for 320 images).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Inside each photo collection

Melanoma and nevus come from one ISIC collection, the other six classes from another, so the collection alone could separate the two groups. Inside each group the collection gives nothing away: a score near chance there means the model told the groups apart, not the diseases.

| Classes | Questions | Balanced accuracy inside the group | Chance | Answers that stay in the group |
|---|---|---|---|---|
| melanoma, nevus | 80 | 30.0% | 50.0% | 41.2% |
| actinic keratosis, basal cell carcinoma, dermatofibroma, pigmented benign keratosis, squamous cell carcinoma, vascular lesion | 240 | 56.2% | 16.7% | 92.5% |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: 27c7ec0f34
- Interruptions before the run finished: 0
