# llama-4-maverick: test split, run test_53efcf3d

- Model requested: `meta-llama/llama-4-maverick`; versions returned by the API: `meta-llama/llama-4-maverick@Parasail`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 512, "image_detail": null, "extra_body": {"provider": {"order": ["parasail/fp8"], "allow_fallbacks": false}}}`
- Dataset `published_text_photos`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-20T05:30:53+00:00, finished 2026-11-20T05:42:13+00:00 (UTC)
- Prompt: `prompts/published_text_photos/closed_book.txt` (SHA-256 `b1a51ff6f089`); cases: 100 (hash `a2387a58e7e4`), each put into the prompt word for word; dataset table `data/published_cases/photos.csv` (SHA-256 `cee3aa4b2699`); Git commit `d933159c5b`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **46.1%** | 39.3% to 53.4% (bootstrap) |
| Macro-F1 | 44.9% | 36.1% to 53.1% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 85.1% | 78.3% to 91.3% (bootstrap) |
| Invalid answers (counted as wrong) | 2.0% (2/100) | |

### Selective prediction

Over the 98 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 10.0% (10/100) | 5.5% to 17.4% |
| Selective risk (confident misdiagnosis): wrong among them | 30.0% (3/10) | 10.8% to 60.3% |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 1 | 0.0% (0.0% to 79.3%) | – | 0.0% |
| basal cell carcinoma | 19 | 42.1% (23.1% to 63.7%) | 100.0% | 59.3% |
| dermatofibroma | 10 | 20.0% (5.7% to 51.0%) | 100.0% | 33.3% |
| melanoma | 17 | 58.8% (36.0% to 78.4%) | 41.7% | 48.8% |
| nevus | 15 | 40.0% (19.8% to 64.3%) | 66.7% | 50.0% |
| pigmented benign keratosis | 10 | 20.0% (5.7% to 51.0%) | 50.0% | 28.6% |
| squamous cell carcinoma | 17 | 88.2% (65.7% to 96.7%) | 42.9% | 57.7% |
| vascular lesion | 11 | 100.0% (74.1% to 100.0%) | 68.8% | 81.5% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| basal cell carcinoma | 0 | 8 | 0 | 3 | 0 | 0 | 8 | 0 | 0 |
| dermatofibroma | 0 | 0 | 2 | 2 | 2 | 0 | 2 | 2 | 0 |
| melanoma | 0 | 0 | 0 | 10 | 0 | 1 | 6 | 0 | 0 |
| nevus | 0 | 0 | 0 | 5 | 6 | 1 | 0 | 3 | 0 |
| pigmented benign keratosis | 0 | 0 | 0 | 4 | 1 | 2 | 3 | 0 | 0 |
| squamous cell carcinoma | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 2 |
| vascular lesion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 |

- No two cases share one article, so every case counts as independent.
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 1 case in this split, so its recall can only be 0% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **52.7%** | 44.7% to 60.7% (bootstrap) |
| Macro-F1 | 51.5% | 40.8% to 60.8% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 82.9% | 75.1% to 90.2% (bootstrap) |
| Invalid answers (counted as wrong) | 2.0% (2/99) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 100
- Why answers were invalid: not a class of the list: 'glomus tumor': 1; not a class of the list: 'seborrheic keratosis': 1
- Sessions (starts and restarts of this run): 1; Git commits: d933159c5b
- Interruptions before the run finished: 0
