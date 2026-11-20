# llama-4-maverick: test split, run test_69dbecaf

- Model requested: `meta-llama/llama-4-maverick`; versions returned by the API: `meta-llama/llama-4-maverick@Parasail`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 512, "image_detail": null, "extra_body": {"provider": {"order": ["parasail/fp8"], "allow_fallbacks": false}}}`
- Dataset `published_photos`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-20T05:42:16+00:00, finished 2026-11-20T06:08:44+00:00 (UTC)
- Prompt: `prompts/published_photos/closed_book.txt` (SHA-256 `eed273f166a5`); cases: 100 (hash `a2387a58e7e4`), sent exactly as stored; dataset table `data/published_cases/photos.csv` (SHA-256 `cee3aa4b2699`); Git commit `761308709c`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **30.0%** | 23.7% to 36.2% (bootstrap) |
| Macro-F1 | 24.8% | 19.0% to 30.8% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 72.5% | 64.9% to 80.1% (bootstrap) |
| Invalid answers (counted as wrong) | 0.0% (0/100) | |

### Selective prediction

Over the 100 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 3.0% (3/100) | 1.0% to 8.5% |
| Selective risk (confident misdiagnosis): wrong among them | 100.0% (3/3) | 43.8% to 100.0% |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 1 | 0.0% (0.0% to 79.3%) | 0.0% | 0.0% |
| basal cell carcinoma | 19 | 0.0% (0.0% to 16.8%) | – | 0.0% |
| dermatofibroma | 10 | 10.0% (1.8% to 40.4%) | 33.3% | 15.4% |
| melanoma | 17 | 52.9% (31.0% to 73.8%) | 28.1% | 36.7% |
| nevus | 15 | 40.0% (19.8% to 64.3%) | 54.5% | 46.2% |
| pigmented benign keratosis | 10 | 0.0% (0.0% to 27.8%) | 0.0% | 0.0% |
| squamous cell carcinoma | 17 | 82.4% (59.0% to 93.8%) | 42.4% | 56.0% |
| vascular lesion | 11 | 54.5% (28.0% to 78.7%) | 37.5% | 44.4% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| basal cell carcinoma | 0 | 0 | 0 | 8 | 0 | 0 | 9 | 2 | 0 |
| dermatofibroma | 0 | 0 | 1 | 4 | 1 | 0 | 1 | 3 | 0 |
| melanoma | 1 | 0 | 0 | 9 | 1 | 2 | 2 | 2 | 0 |
| nevus | 0 | 0 | 0 | 5 | 6 | 1 | 1 | 2 | 0 |
| pigmented benign keratosis | 0 | 0 | 0 | 4 | 2 | 0 | 4 | 0 | 0 |
| squamous cell carcinoma | 0 | 0 | 0 | 2 | 0 | 0 | 14 | 1 | 0 |
| vascular lesion | 0 | 0 | 2 | 0 | 1 | 1 | 1 | 6 | 0 |

- No two cases share one article, so every case counts as independent.
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 1 case in this split, so its recall can only be 0% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **34.3%** | 26.9% to 41.9% (bootstrap) |
| Macro-F1 | 28.6% | 21.4% to 35.5% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 68.5% | 59.7% to 77.4% (bootstrap) |
| Invalid answers (counted as wrong) | 0.0% (0/99) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 100
- Sessions (starts and restarts of this run): 1; Git commits: 761308709c
- Interruptions before the run finished: 0
