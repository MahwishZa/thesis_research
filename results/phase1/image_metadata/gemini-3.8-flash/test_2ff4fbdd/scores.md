# gemini-3.8-flash: test split, run test_2ff4fbdd

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Dataset `image_metadata`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-15T01:00:02+00:00, finished 2026-11-15T01:24:48+00:00 (UTC)
- Prompt: `prompts/multimodal/diagnosis.txt` (SHA-256 `d1a433195f62`); images: 320 (hash `c3de85ddfda0`), sent exactly as stored, each with its patient's facts in the prompt; dataset table `data/multimodal/questions.csv` (SHA-256 `150d699e3752`); Git commit `27c7ec0f34`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **54.4%** | 48.6% to 59.9% (bootstrap over lesions) |
| Macro-F1 | 53.1% | 47.2% to 58.6% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/320) | |

### Per class

| Class | Images | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 40 | 40.0% (21.7% to 60.5%) | 48.5% | 43.8% |
| basal cell carcinoma | 40 | 80.0% (63.9% to 94.9%) | 47.1% | 59.3% |
| dermatofibroma | 40 | 72.5% (53.8% to 88.9%) | 63.0% | 67.4% |
| melanoma | 40 | 35.0% (22.5% to 50.0%) | 29.8% | 32.2% |
| nevus | 40 | 42.5% (27.5% to 57.5%) | 73.9% | 54.0% |
| pigmented benign keratosis | 40 | 62.5% (46.0% to 78.9%) | 42.4% | 50.5% |
| squamous cell carcinoma | 40 | 15.0% (2.5% to 29.3%) | 66.7% | 24.5% |
| vascular lesion | 40 | 87.5% (73.8% to 100.0%) | 100.0% | 93.3% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 16 | 6 | 4 | 7 | 1 | 5 | 1 | 0 | 0 |
| basal cell carcinoma | 0 | 32 | 1 | 4 | 0 | 3 | 0 | 0 | 0 |
| dermatofibroma | 0 | 6 | 29 | 2 | 1 | 0 | 2 | 0 | 0 |
| melanoma | 4 | 6 | 1 | 14 | 2 | 13 | 0 | 0 | 0 |
| nevus | 0 | 2 | 6 | 6 | 17 | 9 | 0 | 0 | 0 |
| pigmented benign keratosis | 2 | 2 | 2 | 9 | 0 | 25 | 0 | 0 | 0 |
| squamous cell carcinoma | 11 | 12 | 2 | 5 | 0 | 4 | 6 | 0 | 0 |
| vascular lesion | 0 | 2 | 1 | 0 | 2 | 0 | 0 | 35 | 0 |

- Photos of one lesion are not independent: the intervals resample whole lesions (248 lesions for 320 images).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Questions with all facts recorded

A missing fact ("not recorded") points to the photo's collection, and so mostly to melanoma or nevus (data/image_metadata/datasheet.md, limitation 1). These scores count only the 289 of 320 questions with every fact recorded, beside the main score, never instead of it.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **54.8%** | 48.7% to 60.9% (bootstrap over lesions) |
| Macro-F1 | 53.8% | 47.3% to 59.8% (bootstrap over lesions) |
| Invalid answers (counted as wrong) | 0.0% (0/289) | |

## Inside each photo collection

Melanoma and nevus come from one ISIC collection, the other six classes from another, so the collection alone could separate the two groups. Inside each group the collection gives nothing away: a score near chance there means the model told the groups apart, not the diseases.

| Classes | Questions | Balanced accuracy inside the group | Chance | Answers that stay in the group |
|---|---|---|---|---|
| melanoma, nevus | 80 | 38.7% | 50.0% | 48.8% |
| actinic keratosis, basal cell carcinoma, dermatofibroma, pigmented benign keratosis, squamous cell carcinoma, vascular lesion | 240 | 59.6% | 16.7% | 87.1% |

## Notes

- How answers ended (finish reason): STOP: 320
- Sessions (starts and restarts of this run): 1; Git commits: 27c7ec0f34
- Interruptions before the run finished: 0
