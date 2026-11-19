# llama-4-maverick: test split, run test_5719a8d3

- Model requested: `meta-llama/llama-4-maverick`; versions returned by the API: `meta-llama/llama-4-maverick@Parasail`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 512, "image_detail": null, "extra_body": {"provider": {"order": ["parasail/fp8"], "allow_fallbacks": false}}}`
- Dataset `published_text`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-19T17:15:22+00:00, finished 2026-11-19T17:28:38+00:00 (UTC)
- Prompt: `prompts/published_text/closed_book.txt` (SHA-256 `f1e9dc53d9cf`); cases: 209 (hash `80be06e00be2`), each put into the prompt word for word; dataset table `data/published_cases/dataset.csv` (SHA-256 `23bcfb6fcb02`); Git commit `e3d5510c6a`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **41.9%** | 35.9% to 47.9% (bootstrap over articles) |
| Macro-F1 | 41.6% | 34.8% to 47.7% (bootstrap over articles) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 83.7% | 78.4% to 88.5% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 1.9% (4/209) | |

### Selective prediction

Over the 205 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 6.7% (14/209) | 3.4% to 10.1% |
| Selective risk (confident misdiagnosis): wrong among them | 28.6% (4/14) | 7.1% to 54.5% |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 2 | 0.0% (0.0% to 0.0%) | 0.0% | 0.0% |
| basal cell carcinoma | 40 | 30.0% (17.5% to 45.0%) | 80.0% | 43.6% |
| dermatofibroma | 20 | 20.0% (0.0% to 38.5%) | 66.7% | 30.8% |
| melanoma | 37 | 51.4% (35.1% to 67.6%) | 40.4% | 45.2% |
| nevus | 39 | 53.8% (38.5% to 69.2%) | 70.0% | 60.9% |
| pigmented benign keratosis | 18 | 33.3% (11.1% to 55.6%) | 60.0% | 42.9% |
| squamous cell carcinoma | 37 | 83.8% (70.3% to 94.6%) | 43.1% | 56.9% |
| vascular lesion | 16 | 62.5% (37.5% to 87.5%) | 45.5% | 52.6% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| basal cell carcinoma | 2 | 12 | 0 | 8 | 0 | 0 | 17 | 1 | 0 |
| dermatofibroma | 0 | 1 | 4 | 4 | 4 | 0 | 3 | 4 | 0 |
| melanoma | 0 | 0 | 1 | 19 | 2 | 1 | 13 | 1 | 0 |
| nevus | 0 | 1 | 0 | 9 | 21 | 2 | 0 | 6 | 0 |
| pigmented benign keratosis | 0 | 0 | 0 | 6 | 2 | 6 | 3 | 0 | 1 |
| squamous cell carcinoma | 1 | 1 | 1 | 0 | 0 | 1 | 31 | 0 | 2 |
| vascular lesion | 0 | 0 | 0 | 1 | 1 | 0 | 3 | 10 | 1 |

- Patients described in one article are not independent: the intervals resample whole articles (207 articles for 209 cases).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **47.8%** | 40.6% to 54.4% (bootstrap over articles) |
| Macro-F1 | 47.7% | 39.6% to 54.6% (bootstrap over articles) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 81.4% | 75.4% to 86.8% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 1.9% (4/207) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 209
- Why answers were invalid: not a class of the list: 'glomus tumor': 1; not a class of the list: 'giant condyloma acuminata': 1; not a class of the list: 'seborrheic keratosis': 1; not a class of the list: 'amelanotic melanoma': 1
- Sessions (starts and restarts of this run): 1; Git commits: e3d5510c6a
- Interruptions before the run finished: 0
