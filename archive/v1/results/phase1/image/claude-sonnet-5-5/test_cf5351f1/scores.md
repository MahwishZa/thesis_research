# claude-sonnet-5-5: test split, run test_cf5351f1

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 256, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Dataset `image`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-14T16:50:46+00:00, finished 2026-11-14T17:12:47+00:00 (UTC)
- Prompt: `prompts/image/diagnosis.txt` (SHA-256 `5e1ed90af51d`); images: 320 (hash `cfa2c9e9b7c6`), sent exactly as stored; dataset table `data/image/dataset.csv` (SHA-256 `394408d529cf`); Git commit `8dad1045d9`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **51.2%** | 45.7% to 56.7% (bootstrap over lesions) |
| Macro-F1 | 49.5% | 44.0% to 54.7% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/320) | |

### Per class

| Class | Images | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 40 | 45.0% (28.9% to 62.9%) | 34.0% | 38.7% |
| basal cell carcinoma | 40 | 65.0% (47.4% to 80.6%) | 50.0% | 56.5% |
| dermatofibroma | 40 | 67.5% (48.7% to 84.1%) | 81.8% | 74.0% |
| melanoma | 40 | 42.5% (27.5% to 57.5%) | 34.0% | 37.8% |
| nevus | 40 | 60.0% (45.0% to 75.0%) | 49.0% | 53.9% |
| pigmented benign keratosis | 40 | 37.5% (20.0% to 56.1%) | 41.7% | 39.5% |
| squamous cell carcinoma | 40 | 5.0% (0.0% to 15.4%) | 28.6% | 8.5% |
| vascular lesion | 40 | 87.5% (73.7% to 100.0%) | 87.5% | 87.5% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 18 | 11 | 2 | 1 | 2 | 4 | 1 | 1 | 0 |
| basal cell carcinoma | 5 | 26 | 1 | 2 | 2 | 0 | 2 | 2 | 0 |
| dermatofibroma | 0 | 6 | 27 | 2 | 3 | 0 | 1 | 1 | 0 |
| melanoma | 5 | 1 | 0 | 17 | 6 | 11 | 0 | 0 | 0 |
| nevus | 0 | 1 | 1 | 13 | 24 | 1 | 0 | 0 | 0 |
| pigmented benign keratosis | 6 | 0 | 1 | 8 | 9 | 15 | 1 | 0 | 0 |
| squamous cell carcinoma | 19 | 7 | 0 | 4 | 2 | 5 | 2 | 1 | 0 |
| vascular lesion | 0 | 0 | 1 | 3 | 1 | 0 | 0 | 35 | 0 |

- Photos of one lesion are not independent: the intervals resample whole lesions (248 lesions for 320 images).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Inside each photo collection

Melanoma and nevus come from one ISIC collection, the other six classes from another, so the collection alone could separate the two groups. Inside each group the collection gives nothing away: a score near chance there means the model told the groups apart, not the diseases.

| Classes | Questions | Balanced accuracy inside the group | Chance | Answers that stay in the group |
|---|---|---|---|---|
| melanoma, nevus | 80 | 51.2% | 50.0% | 75.0% |
| actinic keratosis, basal cell carcinoma, dermatofibroma, pigmented benign keratosis, squamous cell carcinoma, vascular lesion | 240 | 51.3% | 16.7% | 83.8% |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: 8dad1045d9
- Interruptions before the run finished: 0
