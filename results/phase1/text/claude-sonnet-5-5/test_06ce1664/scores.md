# claude-sonnet-5-5: test split, run test_06ce1664

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `claude-sonnet-5-5`; versions returned by the API: `claude-sonnet-5-5`
- Generation settings: `{"max_tokens": 256, "thinking": "between_tools", "effort": "low", "sampling": "provider default (temperature, top_p and top_k are not sent)"}`
- Dataset `text`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-14T17:12:56+00:00, finished 2026-11-14T17:27:37+00:00 (UTC)
- Prompt: `prompts/text/diagnosis.txt` (SHA-256 `5b6fe4db8d52`); cases: 221 (hash `a75d8b54c810`), each put into the prompt word for word; dataset table `data/text/dataset.csv` (SHA-256 `f4d40e300b62`); Git commit `8dad1045d9`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **55.9%** | 49.6% to 61.8% (bootstrap over articles) |
| Macro-F1 | 57.9% | 51.4% to 63.5% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 4.5% (10/221) | |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 2 | 0.0% (0.0% to 0.0%) | 0.0% | 0.0% |
| basal cell carcinoma | 40 | 65.0% (50.0% to 80.0%) | 60.5% | 62.7% |
| dermatofibroma | 21 | 57.1% (31.6% to 78.3%) | 85.7% | 68.6% |
| melanoma | 40 | 70.0% (55.0% to 82.5%) | 63.6% | 66.7% |
| nevus | 40 | 62.5% (47.5% to 77.5%) | 75.8% | 68.5% |
| pigmented benign keratosis | 22 | 36.4% (18.2% to 54.5%) | 88.9% | 51.6% |
| squamous cell carcinoma | 40 | 75.0% (60.0% to 87.5%) | 60.0% | 66.7% |
| vascular lesion | 16 | 81.2% (62.5% to 100.0%) | 76.5% | 78.8% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| basal cell carcinoma | 0 | 26 | 0 | 3 | 0 | 1 | 9 | 0 | 1 |
| dermatofibroma | 0 | 4 | 12 | 2 | 1 | 0 | 1 | 1 | 0 |
| melanoma | 0 | 4 | 0 | 28 | 1 | 0 | 5 | 2 | 0 |
| nevus | 0 | 2 | 1 | 8 | 25 | 0 | 0 | 1 | 3 |
| pigmented benign keratosis | 0 | 4 | 0 | 1 | 4 | 8 | 3 | 0 | 2 |
| squamous cell carcinoma | 1 | 3 | 1 | 1 | 2 | 0 | 30 | 0 | 2 |
| vascular lesion | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 13 | 1 |

- Patients described in one article are not independent: the intervals resample whole articles (219 articles for 221 cases).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **63.9%** | 57.1% to 70.6% (bootstrap over articles) |
| Macro-F1 | 66.3% | 59.1% to 72.7% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 4.1% (9/219) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | (invalid) |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | squamous cell carcinoma |

## Sensitivity analysis: the last line of the answer

The main score reads an answer only when it is exactly one class name, as the prompt of the first phase asked. 9 answers wrote their reasoning first and then a class name alone on the last line; 7 of them name the right class. Here they are read as that class: a check on the main score, never a replacement for it.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **59.1%** | 52.9% to 64.7% (bootstrap over articles) |
| Macro-F1 | 60.1% | 53.8% to 65.5% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 0.5% (1/221) | |

## Notes

- How answers ended (finish reason): MAX_TOKENS: 1, STOP: 220
- 1 answers reached the output limit but already differed from every class name, so the limit did not change their score; they count as wrong.
- Sessions (starts and restarts of this run): 1; Git commits: 8dad1045d9
- Interruptions before the run finished: 0
