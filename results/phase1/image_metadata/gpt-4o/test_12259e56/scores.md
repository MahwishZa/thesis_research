# gpt-4o: test split, run test_12259e56

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `openai/gpt-4o-2024-08-06`; versions returned by the API: `openai/gpt-4o-2024-08-06@OpenAI`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 64, "image_detail": "high", "extra_body": {"provider": {"order": ["openai"], "allow_fallbacks": false}}}`
- Dataset `image_metadata`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-14T23:28:17+00:00, finished 2026-11-14T23:44:18+00:00 (UTC)
- Prompt: `prompts/multimodal/diagnosis.txt` (SHA-256 `d1a433195f62`); images: 320 (hash `c3de85ddfda0`), sent exactly as stored, each with its patient's facts in the prompt; dataset table `data/multimodal/questions.csv` (SHA-256 `150d699e3752`); Git commit `499bc650b3`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **41.6%** | 36.4% to 46.7% (bootstrap over lesions) |
| Macro-F1 | 40.6% | 35.1% to 45.6% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/320) | |

### Per class

| Class | Images | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 40 | 20.0% (6.7% to 35.9%) | 66.7% | 30.8% |
| basal cell carcinoma | 40 | 52.5% (37.5% to 67.6%) | 41.2% | 46.2% |
| dermatofibroma | 40 | 72.5% (55.0% to 88.1%) | 70.7% | 71.6% |
| melanoma | 40 | 70.0% (55.0% to 85.0%) | 20.4% | 31.6% |
| nevus | 40 | 32.5% (17.5% to 47.5%) | 59.1% | 41.9% |
| pigmented benign keratosis | 40 | 37.5% (19.0% to 55.1%) | 39.5% | 38.5% |
| squamous cell carcinoma | 40 | 0.0% (0.0% to 0.0%) | – | 0.0% |
| vascular lesion | 40 | 47.5% (29.3% to 67.8%) | 100.0% | 64.4% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 8 | 15 | 2 | 9 | 0 | 6 | 0 | 0 | 0 |
| basal cell carcinoma | 1 | 21 | 3 | 14 | 0 | 1 | 0 | 0 | 0 |
| dermatofibroma | 0 | 2 | 29 | 7 | 2 | 0 | 0 | 0 | 0 |
| melanoma | 0 | 1 | 0 | 28 | 2 | 9 | 0 | 0 | 0 |
| nevus | 0 | 0 | 1 | 25 | 13 | 1 | 0 | 0 | 0 |
| pigmented benign keratosis | 2 | 2 | 2 | 16 | 3 | 15 | 0 | 0 | 0 |
| squamous cell carcinoma | 1 | 8 | 2 | 23 | 0 | 6 | 0 | 0 | 0 |
| vascular lesion | 0 | 2 | 2 | 15 | 2 | 0 | 0 | 19 | 0 |

- Photos of one lesion are not independent: the intervals resample whole lesions (248 lesions for 320 images).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Questions with all facts recorded

A missing fact ("not recorded") points to the photo's collection, and so mostly to melanoma or nevus (data/image_metadata/datasheet.md, limitation 1). These scores count only the 289 of 320 questions with every fact recorded, beside the main score, never instead of it.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **40.9%** | 35.1% to 46.8% (bootstrap over lesions) |
| Macro-F1 | 39.5% | 33.5% to 44.9% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/289) | |

## Inside each photo collection

Melanoma and nevus come from one ISIC collection, the other six classes from another, so the collection alone could separate the two groups. Inside each group the collection gives nothing away: a score near chance there means the model told the groups apart, not the diseases.

| Classes | Questions | Balanced accuracy inside the group | Chance | Answers that stay in the group |
|---|---|---|---|---|
| melanoma, nevus | 80 | 51.2% | 50.0% | 85.0% |
| actinic keratosis, basal cell carcinoma, dermatofibroma, pigmented benign keratosis, squamous cell carcinoma, vascular lesion | 240 | 38.3% | 16.7% | 62.1% |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: 499bc650b3
- Interruptions before the run finished: 0
