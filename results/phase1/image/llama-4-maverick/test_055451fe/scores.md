# llama-4-maverick: test split, run test_055451fe

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `meta-llama/llama-4-maverick`; versions returned by the API: `meta-llama/llama-4-maverick@Parasail`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 256, "image_detail": null, "extra_body": {"provider": {"order": ["parasail/fp8"], "allow_fallbacks": false}}}`
- Dataset `image`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-15T02:11:07+00:00, finished 2026-11-15T02:27:05+00:00 (UTC)
- Prompt: `prompts/image/diagnosis.txt` (SHA-256 `5e1ed90af51d`); images: 320 (hash `cfa2c9e9b7c6`), sent exactly as stored; dataset table `data/image/dataset.csv` (SHA-256 `394408d529cf`); Git commit `387fcfe590`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **28.7%** | 24.4% to 33.1% (bootstrap over lesions) |
| Macro-F1 | 25.7% | 20.8% to 30.5% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/320) | |

### Per class

| Class | Images | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 40 | 0.0% (0.0% to 0.0%) | 0.0% | 0.0% |
| basal cell carcinoma | 40 | 10.0% (0.0% to 22.2%) | 44.4% | 16.3% |
| dermatofibroma | 40 | 40.0% (21.6% to 59.0%) | 59.3% | 47.8% |
| melanoma | 40 | 85.0% (72.5% to 95.0%) | 17.2% | 28.6% |
| nevus | 40 | 22.5% (10.0% to 35.0%) | 23.7% | 23.1% |
| pigmented benign keratosis | 40 | 10.0% (2.3% to 20.0%) | 44.4% | 16.3% |
| squamous cell carcinoma | 40 | 5.0% (0.0% to 15.4%) | 50.0% | 9.1% |
| vascular lesion | 40 | 57.5% (41.9% to 73.7%) | 74.2% | 64.8% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 4 | 1 | 27 | 4 | 2 | 0 | 2 | 0 |
| basal cell carcinoma | 2 | 4 | 5 | 23 | 5 | 0 | 0 | 1 | 0 |
| dermatofibroma | 0 | 1 | 16 | 18 | 4 | 0 | 1 | 0 | 0 |
| melanoma | 1 | 0 | 0 | 34 | 2 | 1 | 0 | 2 | 0 |
| nevus | 0 | 0 | 1 | 29 | 9 | 1 | 0 | 0 | 0 |
| pigmented benign keratosis | 1 | 0 | 1 | 21 | 12 | 4 | 1 | 0 | 0 |
| squamous cell carcinoma | 0 | 0 | 1 | 31 | 2 | 1 | 2 | 3 | 0 |
| vascular lesion | 0 | 0 | 2 | 15 | 0 | 0 | 0 | 23 | 0 |

- Photos of one lesion are not independent: the intervals resample whole lesions (248 lesions for 320 images).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Inside each photo collection

Melanoma and nevus come from one ISIC collection, the other six classes from another, so the collection alone could separate the two groups. Inside each group the collection gives nothing away: a score near chance there means the model told the groups apart, not the diseases.

| Classes | Questions | Balanced accuracy inside the group | Chance | Answers that stay in the group |
|---|---|---|---|---|
| melanoma, nevus | 80 | 53.8% | 50.0% | 92.5% |
| actinic keratosis, basal cell carcinoma, dermatofibroma, pigmented benign keratosis, squamous cell carcinoma, vascular lesion | 240 | 20.4% | 16.7% | 32.5% |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: 387fcfe590
- Interruptions before the run finished: 0
