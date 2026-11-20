# gemini-3.8-flash: test split, run test_6e2bd64c

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Dataset `published_text`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-19T13:12:27+00:00, finished 2026-11-19T13:31:28+00:00 (UTC)
- Prompt: `prompts/published_text/closed_book.txt` (SHA-256 `f1e9dc53d9cf`); cases: 209 (hash `80be06e00be2`), each put into the prompt word for word; dataset table `data/published_cases/dataset.csv` (SHA-256 `23bcfb6fcb02`); Git commit `5d411e5af9`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **59.6%** | 53.5% to 65.1% (bootstrap over articles) |
| Macro-F1 | 58.6% | 52.6% to 64.0% (bootstrap over articles) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 91.6% | 87.4% to 95.3% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 0.0% (0/209) | |

### Selective prediction

Over the 209 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 33.0% (69/209) | 27.1% to 39.3% |
| Selective risk (confident misdiagnosis): wrong among them | 10.1% (7/69) | 3.6% to 17.9% |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 2 | 0.0% (0.0% to 0.0%) | 0.0% | 0.0% |
| basal cell carcinoma | 40 | 62.5% (47.5% to 77.5%) | 75.8% | 68.5% |
| dermatofibroma | 20 | 55.0% (27.8% to 76.9%) | 84.6% | 66.7% |
| melanoma | 37 | 70.3% (54.1% to 83.8%) | 59.1% | 64.2% |
| nevus | 39 | 48.7% (33.3% to 64.1%) | 90.5% | 63.3% |
| pigmented benign keratosis | 18 | 72.2% (50.0% to 88.9%) | 65.0% | 68.4% |
| squamous cell carcinoma | 37 | 86.5% (73.0% to 97.3%) | 60.4% | 71.1% |
| vascular lesion | 16 | 81.2% (62.5% to 100.0%) | 56.5% | 66.7% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| basal cell carcinoma | 1 | 25 | 0 | 5 | 0 | 2 | 7 | 0 | 0 |
| dermatofibroma | 0 | 3 | 11 | 1 | 0 | 0 | 3 | 2 | 0 |
| melanoma | 0 | 0 | 0 | 26 | 2 | 1 | 5 | 3 | 0 |
| nevus | 0 | 2 | 2 | 10 | 19 | 2 | 1 | 3 | 0 |
| pigmented benign keratosis | 0 | 1 | 0 | 2 | 0 | 13 | 2 | 0 | 0 |
| squamous cell carcinoma | 1 | 0 | 0 | 0 | 0 | 2 | 32 | 2 | 0 |
| vascular lesion | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 13 | 0 |

- Patients described in one article are not independent: the intervals resample whole articles (207 articles for 209 cases).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **68.1%** | 61.1% to 74.5% (bootstrap over articles) |
| Macro-F1 | 67.2% | 60.0% to 73.4% (bootstrap over articles) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 90.5% | 85.9% to 94.7% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 0.0% (0/207) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 209
- Sessions (starts and restarts of this run): 2; Git commits: 5d411e5af9
- Interruptions before the run finished: 1; 2026-11-19T13:15:21+00:00 after 48 answers (FatalError)
