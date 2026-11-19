# gpt-4o: test split, run test_14e3059f

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `openai/gpt-4o-2024-08-06`; versions returned by the API: `openai/gpt-4o-2024-08-06@OpenAI`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 64, "image_detail": "high", "extra_body": {"provider": {"order": ["openai"], "allow_fallbacks": false}}}`
- Dataset `image`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-14T23:01:02+00:00, finished 2026-11-14T23:17:04+00:00 (UTC)
- Prompt: `prompts/image/diagnosis.txt` (SHA-256 `5e1ed90af51d`); images: 320 (hash `cfa2c9e9b7c6`), sent exactly as stored; dataset table `data/image/dataset.csv` (SHA-256 `394408d529cf`); Git commit `499bc650b3`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **34.1%** | 29.6% to 38.7% (bootstrap over lesions) |
| Macro-F1 | 30.1% | 25.5% to 34.5% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/320) | |

### Per class

| Class | Images | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 40 | 0.0% (0.0% to 0.0%) | – | 0.0% |
| basal cell carcinoma | 40 | 40.0% (24.3% to 56.4%) | 40.0% | 40.0% |
| dermatofibroma | 40 | 65.0% (47.7% to 81.1%) | 34.7% | 45.2% |
| melanoma | 40 | 77.5% (65.0% to 90.0%) | 21.1% | 33.2% |
| nevus | 40 | 25.0% (12.5% to 40.0%) | 45.5% | 32.3% |
| pigmented benign keratosis | 40 | 25.0% (9.5% to 41.7%) | 50.0% | 33.3% |
| squamous cell carcinoma | 40 | 0.0% (0.0% to 0.0%) | – | 0.0% |
| vascular lesion | 40 | 40.0% (25.5% to 57.9%) | 100.0% | 57.1% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 8 | 12 | 15 | 1 | 4 | 0 | 0 | 0 |
| basal cell carcinoma | 0 | 16 | 3 | 20 | 1 | 0 | 0 | 0 | 0 |
| dermatofibroma | 0 | 5 | 26 | 7 | 2 | 0 | 0 | 0 | 0 |
| melanoma | 0 | 1 | 2 | 31 | 2 | 4 | 0 | 0 | 0 |
| nevus | 0 | 1 | 1 | 27 | 10 | 1 | 0 | 0 | 0 |
| pigmented benign keratosis | 0 | 1 | 9 | 15 | 5 | 10 | 0 | 0 | 0 |
| squamous cell carcinoma | 0 | 4 | 14 | 21 | 0 | 1 | 0 | 0 | 0 |
| vascular lesion | 0 | 4 | 8 | 11 | 1 | 0 | 0 | 16 | 0 |

- Photos of one lesion are not independent: the intervals resample whole lesions (248 lesions for 320 images).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Inside each photo collection

Melanoma and nevus come from one ISIC collection, the other six classes from another, so the collection alone could separate the two groups. Inside each group the collection gives nothing away: a score near chance there means the model told the groups apart, not the diseases.

| Classes | Questions | Balanced accuracy inside the group | Chance | Answers that stay in the group |
|---|---|---|---|---|
| melanoma, nevus | 80 | 51.2% | 50.0% | 87.5% |
| actinic keratosis, basal cell carcinoma, dermatofibroma, pigmented benign keratosis, squamous cell carcinoma, vascular lesion | 240 | 28.3% | 16.7% | 58.8% |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: 499bc650b3
- Interruptions before the run finished: 0
