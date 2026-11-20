# gpt-4o: test split, run test_366e2923

- Model requested: `openai/gpt-4o-2024-08-06`; versions returned by the API: `openai/gpt-4o-2024-08-06@OpenAI`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 512, "image_detail": "high", "extra_body": {"provider": {"order": ["openai"], "allow_fallbacks": false}}}`
- Dataset `published_photos`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-20T05:36:31+00:00, finished 2026-11-20T06:18:22+00:00 (UTC)
- Prompt: `prompts/published_photos/closed_book.txt` (SHA-256 `eed273f166a5`); cases: 100 (hash `a2387a58e7e4`), sent exactly as stored; dataset table `data/published_cases/photos.csv` (SHA-256 `cee3aa4b2699`); Git commit `761308709c`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **30.3%** | 23.0% to 37.2% (bootstrap) |
| Macro-F1 | 30.0% | 22.2% to 36.8% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 68.9% | 61.6% to 76.4% (bootstrap) |
| Invalid answers (counted as wrong) | 8.0% (8/100) | |

### Selective prediction

Over the 92 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 0.0% (0/100) | 0.0% to 3.7% |
| Selective risk (confident misdiagnosis): wrong among them | – (0/0) | – |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 1 | 0.0% (0.0% to 79.3%) | – | 0.0% |
| basal cell carcinoma | 19 | 36.8% (19.1% to 59.0%) | 53.8% | 43.8% |
| dermatofibroma | 10 | 0.0% (0.0% to 27.8%) | 0.0% | 0.0% |
| melanoma | 17 | 41.2% (21.6% to 64.0%) | 38.9% | 40.0% |
| nevus | 15 | 46.7% (24.8% to 69.9%) | 46.7% | 46.7% |
| pigmented benign keratosis | 10 | 20.0% (5.7% to 51.0%) | 40.0% | 26.7% |
| squamous cell carcinoma | 17 | 70.6% (46.9% to 86.7%) | 40.0% | 51.1% |
| vascular lesion | 11 | 27.3% (9.7% to 56.6%) | 37.5% | 31.6% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| basal cell carcinoma | 0 | 7 | 0 | 3 | 1 | 0 | 7 | 0 | 1 |
| dermatofibroma | 0 | 1 | 0 | 2 | 3 | 0 | 3 | 1 | 0 |
| melanoma | 0 | 2 | 1 | 7 | 2 | 0 | 1 | 1 | 3 |
| nevus | 0 | 1 | 0 | 3 | 7 | 2 | 0 | 2 | 0 |
| pigmented benign keratosis | 0 | 1 | 1 | 1 | 1 | 2 | 4 | 0 | 0 |
| squamous cell carcinoma | 0 | 0 | 0 | 1 | 0 | 0 | 12 | 1 | 3 |
| vascular lesion | 0 | 1 | 1 | 1 | 1 | 1 | 2 | 3 | 1 |

- No two cases share one article, so every case counts as independent.
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 1 case in this split, so its recall can only be 0% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **34.6%** | 26.4% to 43.0% (bootstrap) |
| Macro-F1 | 34.4% | 25.6% to 42.9% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 64.4% | 55.7% to 72.7% (bootstrap) |
| Invalid answers (counted as wrong) | 8.1% (8/99) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): REFUSED: 7, STOP: 93
- Why answers were invalid: refused by the host's content moderation (HTTP 403: Error code: 403 - openai/gpt-4o-2024-08-06 requires moderation on OpenAI. Your input was flagged for "violence/graphic". No credits were charged.): 7; not one JSON object: 1
- **7 questions were refused by the host's content moderation: the model never saw them. Each is an invalid answer and counts as wrong, like any other.**
- Sessions (starts and restarts of this run): 2; Git commits: 761308709c, b7f9e0b988; **the run spanned more than one commit, so its answers do not all come from the same code**
- Interruptions before the run finished: 1; 2026-11-20T05:36:52+00:00 after 3 answers (FatalError)
