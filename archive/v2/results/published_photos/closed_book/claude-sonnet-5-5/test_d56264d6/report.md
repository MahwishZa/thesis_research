# claude-sonnet-5-5: test split, run test_d56264d6

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 512, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Dataset `published_photos`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-19T18:43:58+00:00, finished 2026-11-19T18:51:01+00:00 (UTC)
- Prompt: `prompts/published_photos/closed_book.txt` (SHA-256 `eed273f166a5`); cases: 100 (hash `a2387a58e7e4`), sent exactly as stored; dataset table `data/published_cases/photos.csv` (SHA-256 `cee3aa4b2699`); Git commit `ecb2f8d407`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **39.8%** | 32.7% to 46.6% (bootstrap) |
| Macro-F1 | 37.6% | 30.2% to 44.4% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 75.2% | 68.2% to 82.1% (bootstrap) |
| Invalid answers (counted as wrong) | 2.0% (2/100) | |

### Selective prediction

Over the 98 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 3.0% (3/100) | 1.0% to 8.5% |
| Selective risk (confident misdiagnosis): wrong among them | 0.0% (0/3) | 0.0% to 56.2% |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 1 | 0.0% (0.0% to 79.3%) | 0.0% | 0.0% |
| basal cell carcinoma | 19 | 63.2% (41.0% to 80.9%) | 66.7% | 64.9% |
| dermatofibroma | 10 | 0.0% (0.0% to 27.8%) | 0.0% | 0.0% |
| melanoma | 17 | 52.9% (31.0% to 73.8%) | 50.0% | 51.4% |
| nevus | 15 | 46.7% (24.8% to 69.9%) | 77.8% | 58.3% |
| pigmented benign keratosis | 10 | 10.0% (1.8% to 40.4%) | 50.0% | 16.7% |
| squamous cell carcinoma | 17 | 82.4% (59.0% to 93.8%) | 40.0% | 53.8% |
| vascular lesion | 11 | 63.6% (35.4% to 84.8%) | 50.0% | 56.0% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| basal cell carcinoma | 0 | 12 | 0 | 2 | 0 | 0 | 5 | 0 | 0 |
| dermatofibroma | 0 | 2 | 0 | 3 | 0 | 0 | 1 | 3 | 1 |
| melanoma | 1 | 1 | 0 | 9 | 1 | 1 | 4 | 0 | 0 |
| nevus | 0 | 1 | 1 | 3 | 7 | 0 | 1 | 2 | 0 |
| pigmented benign keratosis | 0 | 1 | 0 | 0 | 1 | 1 | 6 | 1 | 0 |
| squamous cell carcinoma | 0 | 1 | 0 | 1 | 0 | 0 | 14 | 1 | 0 |
| vascular lesion | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 7 | 1 |

- No two cases share one article, so every case counts as independent.
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 1 case in this split, so its recall can only be 0% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **45.5%** | 37.6% to 53.8% (bootstrap) |
| Macro-F1 | 43.2% | 35.2% to 51.1% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 71.7% | 64.0% to 79.9% (bootstrap) |
| Invalid answers (counted as wrong) | 2.0% (2/99) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 100
- Why answers were invalid: a claim must have the key statement, may have the key evidence, and no other: 2
- Sessions (starts and restarts of this run): 1; Git commits: ecb2f8d407
- Interruptions before the run finished: 0
