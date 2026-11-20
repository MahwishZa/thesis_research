# claude-sonnet-5-5: test split, run test_3a656648

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 512, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Dataset `published_text`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-19T12:46:57+00:00, finished 2026-11-19T13:01:15+00:00 (UTC)
- Prompt: `prompts/published_text/closed_book.txt` (SHA-256 `f1e9dc53d9cf`); cases: 209 (hash `80be06e00be2`), each put into the prompt word for word; dataset table `data/published_cases/dataset.csv` (SHA-256 `23bcfb6fcb02`); Git commit `5d411e5af9`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **61.0%** | 55.1% to 66.3% (bootstrap over articles) |
| Macro-F1 | 60.9% | 55.0% to 66.0% (bootstrap over articles) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 91.4% | 87.2% to 95.2% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 0.0% (0/209) | |

### Selective prediction

Over the 209 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 5.3% (11/209) | 2.4% to 8.2% |
| Selective risk (confident misdiagnosis): wrong among them | 0.0% (0/11) | 0.0% to 0.0% |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 2 | 0.0% (0.0% to 0.0%) | – | 0.0% |
| basal cell carcinoma | 40 | 57.5% (42.5% to 72.5%) | 65.7% | 61.3% |
| dermatofibroma | 20 | 65.0% (44.4% to 85.0%) | 81.2% | 72.2% |
| melanoma | 37 | 64.9% (48.6% to 78.4%) | 60.0% | 62.3% |
| nevus | 39 | 59.0% (43.6% to 74.4%) | 79.3% | 67.6% |
| pigmented benign keratosis | 18 | 61.1% (38.9% to 83.3%) | 73.3% | 66.7% |
| squamous cell carcinoma | 37 | 86.5% (75.7% to 97.3%) | 57.1% | 68.8% |
| vascular lesion | 16 | 93.8% (81.2% to 100.0%) | 83.3% | 88.2% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| basal cell carcinoma | 0 | 23 | 1 | 5 | 0 | 2 | 9 | 0 | 0 |
| dermatofibroma | 0 | 2 | 13 | 1 | 2 | 0 | 1 | 1 | 0 |
| melanoma | 0 | 2 | 1 | 24 | 2 | 0 | 7 | 1 | 0 |
| nevus | 0 | 2 | 1 | 8 | 23 | 2 | 2 | 1 | 0 |
| pigmented benign keratosis | 0 | 4 | 0 | 1 | 0 | 11 | 2 | 0 | 0 |
| squamous cell carcinoma | 0 | 2 | 0 | 1 | 2 | 0 | 32 | 0 | 0 |
| vascular lesion | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 15 | 0 |

- Patients described in one article are not independent: the intervals resample whole articles (207 articles for 209 cases).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **69.7%** | 63.0% to 75.9% (bootstrap over articles) |
| Macro-F1 | 69.8% | 63.0% to 75.9% (bootstrap over articles) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 90.1% | 85.4% to 94.5% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 0.0% (0/207) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 209
- Sessions (starts and restarts of this run): 1; Git commits: 5d411e5af9
- Interruptions before the run finished: 0
