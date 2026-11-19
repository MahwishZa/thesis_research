# claude-sonnet-5-5: test split, run test_5db7a077

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 256, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Dataset `image_metadata`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-14T17:27:43+00:00, finished 2026-11-14T17:49:01+00:00 (UTC)
- Prompt: `prompts/multimodal/diagnosis.txt` (SHA-256 `d1a433195f62`); images: 320 (hash `c3de85ddfda0`), sent exactly as stored, each with its patient's facts in the prompt; dataset table `data/multimodal/questions.csv` (SHA-256 `150d699e3752`); Git commit `8dad1045d9`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **56.2%** | 50.6% to 61.7% (bootstrap over lesions) |
| Macro-F1 | 55.7% | 50.0% to 61.1% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/320) | |

### Per class

| Class | Images | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 40 | 52.5% (34.8% to 69.7%) | 46.7% | 49.4% |
| basal cell carcinoma | 40 | 65.0% (47.6% to 81.1%) | 47.3% | 54.7% |
| dermatofibroma | 40 | 75.0% (56.8% to 91.7%) | 85.7% | 80.0% |
| melanoma | 40 | 50.0% (35.0% to 65.0%) | 35.1% | 41.2% |
| nevus | 40 | 57.5% (42.5% to 72.5%) | 60.5% | 59.0% |
| pigmented benign keratosis | 40 | 45.0% (27.8% to 63.6%) | 43.9% | 44.4% |
| squamous cell carcinoma | 40 | 15.0% (3.0% to 28.9%) | 50.0% | 23.1% |
| vascular lesion | 40 | 90.0% (76.2% to 100.0%) | 97.3% | 93.5% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 21 | 9 | 0 | 1 | 2 | 5 | 2 | 0 | 0 |
| basal cell carcinoma | 5 | 26 | 1 | 2 | 1 | 1 | 3 | 1 | 0 |
| dermatofibroma | 0 | 6 | 30 | 2 | 1 | 0 | 1 | 0 | 0 |
| melanoma | 4 | 1 | 0 | 20 | 6 | 9 | 0 | 0 | 0 |
| nevus | 0 | 1 | 1 | 13 | 23 | 2 | 0 | 0 | 0 |
| pigmented benign keratosis | 4 | 2 | 2 | 10 | 4 | 18 | 0 | 0 | 0 |
| squamous cell carcinoma | 11 | 10 | 0 | 7 | 0 | 6 | 6 | 0 | 0 |
| vascular lesion | 0 | 0 | 1 | 2 | 1 | 0 | 0 | 36 | 0 |

- Photos of one lesion are not independent: the intervals resample whole lesions (248 lesions for 320 images).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Questions with all facts recorded

A missing fact ("not recorded") points to the photo's collection, and so mostly to melanoma or nevus (data/image_metadata/datasheet.md, limitation 1). These scores count only the 289 of 320 questions with every fact recorded, beside the main score, never instead of it.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **57.3%** | 51.3% to 63.0% (bootstrap over lesions) |
| Macro-F1 | 55.8% | 49.9% to 61.3% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/289) | |

## Inside each photo collection

Melanoma and nevus come from one ISIC collection, the other six classes from another, so the collection alone could separate the two groups. Inside each group the collection gives nothing away: a score near chance there means the model told the groups apart, not the diseases.

| Classes | Questions | Balanced accuracy inside the group | Chance | Answers that stay in the group |
|---|---|---|---|---|
| melanoma, nevus | 80 | 53.8% | 50.0% | 77.5% |
| actinic keratosis, basal cell carcinoma, dermatofibroma, pigmented benign keratosis, squamous cell carcinoma, vascular lesion | 240 | 57.1% | 16.7% | 86.2% |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: 8dad1045d9
- Interruptions before the run finished: 0
