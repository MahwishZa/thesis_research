# gemini-3.8-flash: test split, run test_6ee1c09b

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `google/gemini-3.8-flash`; versions returned by the API: `google/gemini-3.8-flash@Google AI Studio`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 2048, "image_detail": null, "extra_body": {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": false}, "reasoning": {"effort": "minimal"}}}`
- Dataset `text`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-15T00:45:56+00:00, finished 2026-11-15T01:00:00+00:00 (UTC)
- Prompt: `prompts/text/diagnosis.txt` (SHA-256 `5b6fe4db8d52`); cases: 221 (hash `a75d8b54c810`), each put into the prompt word for word; dataset table `data/text/dataset.csv` (SHA-256 `f4d40e300b62`); Git commit `27c7ec0f34`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **61.4%** | 55.8% to 67.0% (bootstrap over articles) |
| Macro-F1 | 61.3% | 55.8% to 66.7% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 0.0% (0/221) | |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 2 | 0.0% (0.0% to 0.0%) | 0.0% | 0.0% |
| basal cell carcinoma | 40 | 65.0% (50.0% to 80.0%) | 68.4% | 66.7% |
| dermatofibroma | 21 | 61.9% (36.8% to 82.6%) | 86.7% | 72.2% |
| melanoma | 40 | 90.0% (80.0% to 97.5%) | 60.0% | 72.0% |
| nevus | 40 | 47.5% (32.5% to 62.5%) | 100.0% | 64.4% |
| pigmented benign keratosis | 22 | 68.2% (50.0% to 86.4%) | 75.0% | 71.4% |
| squamous cell carcinoma | 40 | 77.5% (62.5% to 90.0%) | 66.0% | 71.3% |
| vascular lesion | 16 | 81.2% (62.5% to 100.0%) | 65.0% | 72.2% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| basal cell carcinoma | 0 | 26 | 0 | 6 | 0 | 2 | 6 | 0 | 0 |
| dermatofibroma | 0 | 4 | 13 | 2 | 0 | 0 | 1 | 1 | 0 |
| melanoma | 0 | 0 | 0 | 36 | 0 | 0 | 3 | 1 | 0 |
| nevus | 0 | 2 | 1 | 12 | 19 | 2 | 1 | 3 | 0 |
| pigmented benign keratosis | 0 | 3 | 1 | 1 | 0 | 15 | 2 | 0 | 0 |
| squamous cell carcinoma | 2 | 2 | 0 | 2 | 0 | 1 | 31 | 2 | 0 |
| vascular lesion | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 13 | 0 |

- Patients described in one article are not independent: the intervals resample whole articles (219 articles for 221 cases).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **70.2%** | 63.7% to 76.3% (bootstrap over articles) |
| Macro-F1 | 70.3% | 63.7% to 76.1% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 0.0% (0/219) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 221
- Sessions (starts and restarts of this run): 1; Git commits: 27c7ec0f34
- Interruptions before the run finished: 0
