# gpt-4o: test split, run test_2fab9eb6

- Model requested: `openai/gpt-4o-2024-08-06`; versions returned by the API: `openai/gpt-4o-2024-08-06@OpenAI`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 512, "image_detail": "high", "extra_body": {"provider": {"order": ["openai"], "allow_fallbacks": false}}}`
- Dataset `published_text`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-19T13:01:22+00:00, finished 2026-11-19T13:12:20+00:00 (UTC)
- Prompt: `prompts/published_text/closed_book.txt` (SHA-256 `f1e9dc53d9cf`); cases: 209 (hash `80be06e00be2`), each put into the prompt word for word; dataset table `data/published_cases/dataset.csv` (SHA-256 `23bcfb6fcb02`); Git commit `5d411e5af9`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **56.1%** | 45.5% to 66.3% (bootstrap over articles) |
| Macro-F1 | 52.8% | 45.0% to 60.7% (bootstrap over articles) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 87.5% | 82.3% to 92.1% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 0.5% (1/209) | |

### Selective prediction

Over the 208 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 1.9% (4/209) | 0.5% to 3.8% |
| Selective risk (confident misdiagnosis): wrong among them | 25.0% (1/4) | – |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 2 | 50.0% (0.0% to 100.0%) | 14.3% | 22.2% |
| basal cell carcinoma | 40 | 45.0% (30.0% to 60.0%) | 72.0% | 55.4% |
| dermatofibroma | 20 | 50.0% (27.8% to 72.2%) | 76.9% | 60.6% |
| melanoma | 37 | 56.8% (40.5% to 73.0%) | 51.2% | 53.8% |
| nevus | 39 | 56.4% (41.0% to 71.8%) | 64.7% | 60.3% |
| pigmented benign keratosis | 18 | 38.9% (16.7% to 61.1%) | 63.6% | 48.3% |
| squamous cell carcinoma | 37 | 70.3% (54.1% to 83.8%) | 50.0% | 58.4% |
| vascular lesion | 16 | 81.2% (62.5% to 100.0%) | 52.0% | 63.4% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| basal cell carcinoma | 3 | 18 | 0 | 5 | 1 | 0 | 12 | 1 | 0 |
| dermatofibroma | 0 | 1 | 10 | 2 | 2 | 0 | 2 | 3 | 0 |
| melanoma | 0 | 2 | 0 | 21 | 3 | 0 | 7 | 4 | 0 |
| nevus | 0 | 1 | 2 | 8 | 22 | 2 | 0 | 4 | 0 |
| pigmented benign keratosis | 0 | 1 | 0 | 5 | 2 | 7 | 3 | 0 | 0 |
| squamous cell carcinoma | 3 | 1 | 1 | 0 | 3 | 2 | 26 | 0 | 1 |
| vascular lesion | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 13 | 0 |

- Patients described in one article are not independent: the intervals resample whole articles (207 articles for 209 cases).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **56.9%** | 50.0% to 63.9% (bootstrap over articles) |
| Macro-F1 | 57.3% | 49.8% to 64.3% (bootstrap over articles) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 85.7% | 79.9% to 90.9% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 0.5% (1/207) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | actinic keratosis |

## Notes

- How answers ended (finish reason): STOP: 209
- Why answers were invalid: not a class of the list: 'glomus tumor': 1
- Sessions (starts and restarts of this run): 1; Git commits: 5d411e5af9
- Interruptions before the run finished: 0
