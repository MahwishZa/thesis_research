# gemini-3.8-flash: test split, run test_c1de9829

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Dataset `published_photos`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-20T02:11:32+00:00, finished 2026-11-20T02:22:30+00:00 (UTC)
- Prompt: `prompts/published_photos/closed_book.txt` (SHA-256 `eed273f166a5`); cases: 100 (hash `a2387a58e7e4`), sent exactly as stored; dataset table `data/published_cases/photos.csv` (SHA-256 `cee3aa4b2699`); Git commit `129cfce169`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **45.1%** | 37.3% to 52.8% (bootstrap) |
| Macro-F1 | 43.5% | 35.3% to 51.3% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 79.1% | 71.9% to 86.6% (bootstrap) |
| Invalid answers (counted as wrong) | 0.0% (0/100) | |

### Selective prediction

Over the 100 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 57.0% (57/100) | 47.2% to 66.3% |
| Selective risk (confident misdiagnosis): wrong among them | 40.4% (23/57) | 28.6% to 53.3% |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 1 | 0.0% (0.0% to 79.3%) | 0.0% | 0.0% |
| basal cell carcinoma | 19 | 84.2% (62.4% to 94.5%) | 55.2% | 66.7% |
| dermatofibroma | 10 | 10.0% (1.8% to 40.4%) | 50.0% | 16.7% |
| melanoma | 17 | 29.4% (13.3% to 53.1%) | 41.7% | 34.5% |
| nevus | 15 | 40.0% (19.8% to 64.3%) | 42.9% | 41.4% |
| pigmented benign keratosis | 10 | 60.0% (31.3% to 83.2%) | 66.7% | 63.2% |
| squamous cell carcinoma | 17 | 82.4% (59.0% to 93.8%) | 73.7% | 77.8% |
| vascular lesion | 11 | 54.5% (28.0% to 78.7%) | 42.9% | 48.0% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| basal cell carcinoma | 0 | 16 | 0 | 2 | 0 | 0 | 1 | 0 | 0 |
| dermatofibroma | 0 | 3 | 1 | 1 | 2 | 0 | 1 | 2 | 0 |
| melanoma | 1 | 4 | 1 | 5 | 1 | 2 | 1 | 2 | 0 |
| nevus | 0 | 4 | 0 | 2 | 6 | 0 | 0 | 3 | 0 |
| pigmented benign keratosis | 0 | 1 | 0 | 0 | 3 | 6 | 0 | 0 | 0 |
| squamous cell carcinoma | 0 | 1 | 0 | 1 | 0 | 0 | 14 | 1 | 0 |
| vascular lesion | 0 | 0 | 0 | 1 | 2 | 1 | 1 | 6 | 0 |

- No two cases share one article, so every case counts as independent.
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 1 case in this split, so its recall can only be 0% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **51.5%** | 42.5% to 60.5% (bootstrap) |
| Macro-F1 | 50.1% | 40.8% to 59.2% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 76.1% | 67.4% to 84.2% (bootstrap) |
| Invalid answers (counted as wrong) | 0.0% (0/99) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 100
- Sessions (starts and restarts of this run): 1; Git commits: 129cfce169
- Interruptions before the run finished: 0
