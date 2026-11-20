# gpt-4o: test split, run test_51aca59c

- Model requested: `openai/gpt-4o-2024-08-06`; versions returned by the API: `openai/gpt-4o-2024-08-06@OpenAI`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 512, "image_detail": "high", "extra_body": {"provider": {"order": ["openai"], "allow_fallbacks": false}}}`
- Dataset `published_text_photos`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-20T05:36:13+00:00, finished 2026-11-20T06:06:49+00:00 (UTC)
- Prompt: `prompts/published_text_photos/closed_book.txt` (SHA-256 `b1a51ff6f089`); cases: 100 (hash `a2387a58e7e4`), each put into the prompt word for word; dataset table `data/published_cases/photos.csv` (SHA-256 `cee3aa4b2699`); Git commit `761308709c`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **48.7%** | 41.2% to 56.2% (bootstrap) |
| Macro-F1 | 47.9% | 39.1% to 56.0% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 89.6% | 83.7% to 94.8% (bootstrap) |
| Invalid answers (counted as wrong) | 4.0% (4/100) | |

### Selective prediction

Over the 96 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 5.0% (5/100) | 2.2% to 11.2% |
| Selective risk (confident misdiagnosis): wrong among them | 20.0% (1/5) | 3.6% to 62.4% |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 1 | 0.0% (0.0% to 79.3%) | – | 0.0% |
| basal cell carcinoma | 19 | 36.8% (19.1% to 59.0%) | 63.6% | 46.7% |
| dermatofibroma | 10 | 20.0% (5.7% to 51.0%) | 66.7% | 30.8% |
| melanoma | 17 | 52.9% (31.0% to 73.8%) | 60.0% | 56.2% |
| nevus | 15 | 66.7% (41.7% to 84.8%) | 71.4% | 69.0% |
| pigmented benign keratosis | 10 | 40.0% (16.8% to 68.7%) | 57.1% | 47.1% |
| squamous cell carcinoma | 17 | 82.4% (59.0% to 93.8%) | 46.7% | 59.6% |
| vascular lesion | 11 | 90.9% (62.3% to 98.4%) | 62.5% | 74.1% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| basal cell carcinoma | 0 | 7 | 0 | 2 | 0 | 0 | 7 | 1 | 2 |
| dermatofibroma | 0 | 0 | 2 | 2 | 2 | 0 | 1 | 2 | 1 |
| melanoma | 0 | 3 | 0 | 9 | 1 | 0 | 4 | 0 | 0 |
| nevus | 0 | 0 | 0 | 1 | 10 | 2 | 0 | 2 | 0 |
| pigmented benign keratosis | 0 | 1 | 0 | 1 | 1 | 4 | 3 | 0 | 0 |
| squamous cell carcinoma | 0 | 0 | 0 | 0 | 0 | 1 | 14 | 1 | 1 |
| vascular lesion | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 10 | 0 |

- No two cases share one article, so every case counts as independent.
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 1 case in this split, so its recall can only be 0% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **55.7%** | 46.6% to 64.2% (bootstrap) |
| Macro-F1 | 55.0% | 44.1% to 64.1% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 88.1% | 81.0% to 94.5% (bootstrap) |
| Invalid answers (counted as wrong) | 4.0% (4/99) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): REFUSED: 4, STOP: 96
- Why answers were invalid: refused by the host's content moderation (HTTP 403: Error code: 403 - openai/gpt-4o-2024-08-06 requires moderation on OpenAI. Your input was flagged for "violence/graphic". No credits were charged.): 4
- **4 questions were refused by the host's content moderation: the model never saw them. Each is an invalid answer and counts as wrong, like any other.**
- Sessions (starts and restarts of this run): 2; Git commits: 761308709c, b7f9e0b988; **the run spanned more than one commit, so its answers do not all come from the same code**
- Interruptions before the run finished: 1; 2026-11-20T05:36:29+00:00 after 3 answers (FatalError)
