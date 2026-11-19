# gpt-4o: test split, run test_5fc7ce07

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `openai/gpt-4o-2024-08-06`; versions returned by the API: `openai/gpt-4o-2024-08-06@OpenAI`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 64, "image_detail": null, "extra_body": {"provider": {"order": ["openai"], "allow_fallbacks": false}}}`
- Dataset `text`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-14T23:17:12+00:00, finished 2026-11-14T23:28:13+00:00 (UTC)
- Prompt: `prompts/text/diagnosis.txt` (SHA-256 `5b6fe4db8d52`); cases: 221 (hash `a75d8b54c810`), each put into the prompt word for word; dataset table `data/text/dataset.csv` (SHA-256 `f4d40e300b62`); Git commit `499bc650b3`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **50.8%** | 45.1% to 56.6% (bootstrap over articles) |
| Macro-F1 | 49.4% | 43.4% to 55.2% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 0.0% (0/221) | |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 2 | 0.0% (0.0% to 0.0%) | 0.0% | 0.0% |
| basal cell carcinoma | 40 | 40.0% (25.0% to 55.0%) | 80.0% | 53.3% |
| dermatofibroma | 21 | 52.4% (26.3% to 73.9%) | 64.7% | 57.9% |
| melanoma | 40 | 60.0% (45.0% to 75.0%) | 45.3% | 51.6% |
| nevus | 40 | 55.0% (40.0% to 70.0%) | 73.3% | 62.9% |
| pigmented benign keratosis | 22 | 36.4% (18.2% to 59.1%) | 72.7% | 48.5% |
| squamous cell carcinoma | 40 | 75.0% (60.0% to 87.5%) | 55.6% | 63.8% |
| vascular lesion | 16 | 87.5% (68.8% to 100.0%) | 42.4% | 57.1% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| basal cell carcinoma | 1 | 16 | 1 | 10 | 0 | 0 | 9 | 3 | 0 |
| dermatofibroma | 0 | 1 | 11 | 2 | 1 | 0 | 2 | 4 | 0 |
| melanoma | 0 | 2 | 0 | 24 | 2 | 0 | 7 | 5 | 0 |
| nevus | 0 | 0 | 3 | 8 | 22 | 2 | 0 | 5 | 0 |
| pigmented benign keratosis | 0 | 1 | 1 | 6 | 3 | 8 | 3 | 0 | 0 |
| squamous cell carcinoma | 2 | 0 | 1 | 2 | 2 | 1 | 30 | 2 | 0 |
| vascular lesion | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 14 | 0 |

- Patients described in one article are not independent: the intervals resample whole articles (219 articles for 221 cases).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **58.0%** | 51.4% to 64.4% (bootstrap over articles) |
| Macro-F1 | 56.6% | 49.5% to 63.2% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 0.0% (0/219) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | squamous cell carcinoma |

## Notes

- How answers ended (finish reason): STOP: 221
- Sessions (starts and restarts of this run): 1; Git commits: 499bc650b3
- Interruptions before the run finished: 0
