# gemini-3.8-flash: test split, run test_17449ed7

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Dataset `published_text_photos`, approach `closed_book`, answer format `json-v2`
- Started 2026-11-20T02:00:08+00:00, finished 2026-11-20T02:11:25+00:00 (UTC)
- Prompt: `prompts/published_text_photos/closed_book.txt` (SHA-256 `b1a51ff6f089`); cases: 100 (hash `a2387a58e7e4`), each put into the prompt word for word; dataset table `data/published_cases/photos.csv` (SHA-256 `cee3aa4b2699`); Git commit `129cfce169`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **66.3%** | 59.0% to 73.3% (bootstrap) |
| Macro-F1 | 67.1% | 59.0% to 73.9% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 91.2% | 85.8% to 96.7% (bootstrap) |
| Invalid answers (counted as wrong) | 2.0% (2/100) | |

### Selective prediction

Over the 98 valid answers. The model answers a question by marking its evidence "sufficient" and abstains by marking it "not sufficient" (selective classification, Geifman & El-Yaniv 2017).

| Score | Value | 95% interval |
|---|---|---|
| Coverage: questions answered with "sufficient" | 66.0% (66/100) | 56.3% to 74.5% |
| Selective risk (confident misdiagnosis): wrong among them | 16.7% (11/66) | 9.6% to 27.4% |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 1 | 0.0% (0.0% to 79.3%) | – | 0.0% |
| basal cell carcinoma | 19 | 84.2% (62.4% to 94.5%) | 69.6% | 76.2% |
| dermatofibroma | 10 | 50.0% (23.7% to 76.3%) | 100.0% | 66.7% |
| melanoma | 17 | 70.6% (46.9% to 86.7%) | 75.0% | 72.7% |
| nevus | 15 | 66.7% (41.7% to 84.8%) | 90.9% | 76.9% |
| pigmented benign keratosis | 10 | 80.0% (49.0% to 94.3%) | 80.0% | 80.0% |
| squamous cell carcinoma | 17 | 88.2% (65.7% to 96.7%) | 75.0% | 81.1% |
| vascular lesion | 11 | 90.9% (62.3% to 98.4%) | 76.9% | 83.3% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| basal cell carcinoma | 0 | 16 | 0 | 1 | 0 | 0 | 2 | 0 | 0 |
| dermatofibroma | 0 | 2 | 5 | 0 | 0 | 0 | 0 | 2 | 1 |
| melanoma | 0 | 1 | 0 | 12 | 1 | 1 | 2 | 0 | 0 |
| nevus | 0 | 2 | 0 | 1 | 10 | 1 | 0 | 0 | 1 |
| pigmented benign keratosis | 0 | 1 | 0 | 1 | 0 | 8 | 0 | 0 | 0 |
| squamous cell carcinoma | 0 | 1 | 0 | 0 | 0 | 0 | 15 | 1 | 0 |
| vascular lesion | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 10 | 0 |

- No two cases share one article, so every case counts as independent.
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 1 case in this split, so its recall can only be 0% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **75.8%** | 67.2% to 83.9% (bootstrap) |
| Macro-F1 | 77.0% | 67.8% to 84.8% (bootstrap) |
| Macro-averaged top-3 accuracy (the true class in the differential) | 90.0% | 83.3% to 95.7% (bootstrap) |
| Invalid answers (counted as wrong) | 2.0% (2/99) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): MAX_TOKENS: 2, STOP: 98
- Why answers were invalid: not one JSON object: 2
- **2 answers were cut off at the output limit (finish reason MAX_TOKENS) and count as wrong: the output limit in the model list (max_tokens) is too low for this model (hidden reasoning can count toward it). Results are not valid until it is raised, under a new model entry.**
- Sessions (starts and restarts of this run): 1; Git commits: 129cfce169
- Interruptions before the run finished: 0
