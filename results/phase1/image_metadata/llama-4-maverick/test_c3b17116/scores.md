# llama-4-maverick: test split, run test_c3b17116

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `meta-llama/llama-4-maverick`; versions returned by the API: `meta-llama/llama-4-maverick@Parasail`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 256, "image_detail": null, "extra_body": {"provider": {"order": ["parasail/fp8"], "allow_fallbacks": false}}}`
- Dataset `image_metadata`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-15T02:38:22+00:00, finished 2026-11-15T02:54:21+00:00 (UTC)
- Prompt: `prompts/multimodal/diagnosis.txt` (SHA-256 `d1a433195f62`); images: 320 (hash `c3de85ddfda0`), sent exactly as stored, each with its patient's facts in the prompt; dataset table `data/multimodal/questions.csv` (SHA-256 `150d699e3752`); Git commit `387fcfe590`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **34.1%** | 29.0% to 39.0% (bootstrap over lesions) |
| Macro-F1 | 33.9% | 27.9% to 39.2% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/320) | |

### Per class

| Class | Images | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 40 | 20.0% (7.0% to 37.1%) | 40.0% | 26.7% |
| basal cell carcinoma | 40 | 15.0% (4.2% to 28.6%) | 46.2% | 22.6% |
| dermatofibroma | 40 | 47.5% (28.6% to 66.7%) | 73.1% | 57.6% |
| melanoma | 40 | 82.5% (70.0% to 92.5%) | 17.9% | 29.5% |
| nevus | 40 | 30.0% (15.0% to 45.0%) | 44.4% | 35.8% |
| pigmented benign keratosis | 40 | 17.5% (5.6% to 31.7%) | 41.2% | 24.6% |
| squamous cell carcinoma | 40 | 5.0% (0.0% to 15.4%) | 33.3% | 8.7% |
| vascular lesion | 40 | 55.0% (37.5% to 73.2%) | 81.5% | 65.7% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 8 | 6 | 1 | 16 | 1 | 7 | 1 | 0 | 0 |
| basal cell carcinoma | 2 | 6 | 3 | 23 | 3 | 1 | 1 | 1 | 0 |
| dermatofibroma | 0 | 0 | 19 | 17 | 3 | 0 | 0 | 1 | 0 |
| melanoma | 4 | 0 | 1 | 33 | 0 | 1 | 1 | 0 | 0 |
| nevus | 0 | 0 | 1 | 26 | 12 | 1 | 0 | 0 | 0 |
| pigmented benign keratosis | 1 | 0 | 1 | 24 | 6 | 7 | 1 | 0 | 0 |
| squamous cell carcinoma | 5 | 1 | 0 | 29 | 0 | 0 | 2 | 3 | 0 |
| vascular lesion | 0 | 0 | 0 | 16 | 2 | 0 | 0 | 22 | 0 |

- Photos of one lesion are not independent: the intervals resample whole lesions (248 lesions for 320 images).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Questions with all facts recorded

A missing fact ("not recorded") points to the photo's collection, and so mostly to melanoma or nevus (data/image_metadata/datasheet.md, limitation 1). These scores count only the 289 of 320 questions with every fact recorded, beside the main score, never instead of it.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **33.4%** | 28.1% to 39.0% (bootstrap over lesions) |
| Macro-F1 | 32.5% | 26.2% to 38.2% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/289) | |

## Inside each photo collection

Melanoma and nevus come from one ISIC collection, the other six classes from another, so the collection alone could separate the two groups. Inside each group the collection gives nothing away: a score near chance there means the model told the groups apart, not the diseases.

| Classes | Questions | Balanced accuracy inside the group | Chance | Answers that stay in the group |
|---|---|---|---|---|
| melanoma, nevus | 80 | 56.2% | 50.0% | 88.8% |
| actinic keratosis, basal cell carcinoma, dermatofibroma, pigmented benign keratosis, squamous cell carcinoma, vascular lesion | 240 | 26.7% | 16.7% | 41.7% |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: 387fcfe590
- Interruptions before the run finished: 0
