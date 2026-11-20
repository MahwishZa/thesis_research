# claude-sonnet-5-5: test split, run test_5ef83f27

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 512, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Dataset `published_text_photos`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-19T18:36:40+00:00, finished 2026-11-19T18:43:50+00:00 (UTC)
- Prompt: `prompts/published_text_photos/closed_book.txt` (SHA-256 `b1a51ff6f089`); cases: 100 (hash `a2387a58e7e4`), each put into the prompt word for word; dataset table `data/published_cases/photos.csv` (SHA-256 `cee3aa4b2699`); Git commit `ecb2f8d407`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **53.2%** | 46.0% to 60.5% (bootstrap) |
| Macro-F1 | 51.4% | 43.4% to 58.8% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 85.7% | 79.9% to 91.6% (bootstrap) |
| Invalid answers (counted as wrong) | 1.0% (1/100) | |

### Selective prediction

Over the 99 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 10.0% (10/100) | 5.5% to 17.4% |
| Selective risk (confident misdiagnosis): wrong among them | 0.0% (0/10) | 0.0% to 27.8% |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 1 | 0.0% (0.0% to 79.3%) | – | 0.0% |
| basal cell carcinoma | 19 | 78.9% (56.7% to 91.5%) | 65.2% | 71.4% |
| dermatofibroma | 10 | 10.0% (1.8% to 40.4%) | 50.0% | 16.7% |
| melanoma | 17 | 47.1% (26.2% to 69.0%) | 57.1% | 51.6% |
| nevus | 15 | 46.7% (24.8% to 69.9%) | 63.6% | 53.8% |
| pigmented benign keratosis | 10 | 70.0% (39.7% to 89.2%) | 77.8% | 73.7% |
| squamous cell carcinoma | 17 | 82.4% (59.0% to 93.8%) | 56.0% | 66.7% |
| vascular lesion | 11 | 90.9% (62.3% to 98.4%) | 66.7% | 76.9% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| basal cell carcinoma | 0 | 15 | 0 | 0 | 0 | 1 | 3 | 0 | 0 |
| dermatofibroma | 0 | 4 | 1 | 0 | 2 | 0 | 0 | 3 | 0 |
| melanoma | 0 | 1 | 0 | 8 | 2 | 0 | 5 | 0 | 1 |
| nevus | 0 | 1 | 0 | 4 | 7 | 1 | 1 | 1 | 0 |
| pigmented benign keratosis | 0 | 1 | 0 | 1 | 0 | 7 | 1 | 0 | 0 |
| squamous cell carcinoma | 0 | 1 | 0 | 1 | 0 | 0 | 14 | 1 | 0 |
| vascular lesion | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 10 | 0 |

- No two cases share one article, so every case counts as independent.
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 1 case in this split, so its recall can only be 0% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **60.8%** | 52.6% to 69.1% (bootstrap) |
| Macro-F1 | 58.9% | 50.0% to 67.8% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 83.7% | 76.2% to 90.6% (bootstrap) |
| Invalid answers (counted as wrong) | 1.0% (1/99) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 100
- Why answers were invalid: a claim must have the key statement, may have the key evidence, and no other: 1
- Sessions (starts and restarts of this run): 1; Git commits: ecb2f8d407
- Interruptions before the run finished: 0
