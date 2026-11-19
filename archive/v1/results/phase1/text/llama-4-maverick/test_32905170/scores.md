# llama-4-maverick: test split, run test_32905170

**A run of the first phase: its answers are single class names, so the scores that need a differential or an evidence flag are not available. Written beside the archived report.md and metrics.json, which are unchanged.**

- Model requested: `meta-llama/llama-4-maverick`; versions returned by the API: `meta-llama/llama-4-maverick@Parasail`
- Generation settings: `{"temperature": 0.0, "seed": 0, "max_tokens": 256, "image_detail": null, "extra_body": {"provider": {"order": ["parasail/fp8"], "allow_fallbacks": false}}}`
- Dataset `text`, approach `closed_book`, answer format `class name (first phase)`
- Started 2026-11-15T02:27:08+00:00, finished 2026-11-15T02:38:18+00:00 (UTC)
- Prompt: `prompts/text/diagnosis.txt` (SHA-256 `5b6fe4db8d52`); cases: 221 (hash `a75d8b54c810`), each put into the prompt word for word; dataset table `data/text/dataset.csv` (SHA-256 `f4d40e300b62`); Git commit `387fcfe590`

## Results

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **46.8%** | 40.6% to 52.8% (bootstrap over articles) |
| Macro-F1 | 47.0% | 40.0% to 53.0% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 5.0% (11/221) | |

### Per class

| Class | Cases | Recall (95% interval) | Precision | F1 |
|---|---|---|---|---|
| actinic keratosis | 2 | 0.0% (0.0% to 0.0%) | 0.0% | 0.0% |
| basal cell carcinoma | 40 | 40.0% (25.0% to 55.0%) | 84.2% | 54.2% |
| dermatofibroma | 21 | 28.6% (5.3% to 52.2%) | 75.0% | 41.4% |
| melanoma | 40 | 57.5% (42.5% to 72.5%) | 46.9% | 51.7% |
| nevus | 40 | 52.5% (37.5% to 67.5%) | 67.7% | 59.2% |
| pigmented benign keratosis | 22 | 40.9% (18.2% to 59.1%) | 56.2% | 47.4% |
| squamous cell carcinoma | 40 | 80.0% (67.5% to 92.5%) | 54.2% | 64.6% |
| vascular lesion | 16 | 75.0% (50.0% to 93.8%) | 46.2% | 57.1% |

### Confusion matrix

Rows: the true class; columns: the diagnosis given.

| True \ Answer | actinic keratosis | basal cell carcinoma | dermatofibroma | melanoma | nevus | pigmented benign keratosis | squamous cell carcinoma | vascular lesion | (invalid) |
|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| basal cell carcinoma | 0 | 16 | 0 | 8 | 0 | 1 | 12 | 2 | 1 |
| dermatofibroma | 0 | 2 | 6 | 2 | 4 | 1 | 0 | 4 | 2 |
| melanoma | 0 | 0 | 1 | 23 | 1 | 0 | 9 | 2 | 4 |
| nevus | 0 | 1 | 0 | 9 | 21 | 2 | 0 | 5 | 2 |
| pigmented benign keratosis | 0 | 0 | 0 | 5 | 3 | 9 | 3 | 1 | 1 |
| squamous cell carcinoma | 2 | 0 | 1 | 1 | 1 | 3 | 32 | 0 | 0 |
| vascular lesion | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 12 | 1 |

- Patients described in one article are not independent: the intervals resample whole articles (219 articles for 221 cases).
- A dash means undefined: precision when a class was never given. A class never given counts as 0 in the macro-F1.
- The definitions and their sources are in docs/method.md.

## Without actinic keratosis

The class actinic keratosis has only 2 cases in this split, so its recall can only be 0%, 50% or 100%, yet it weighs as much as any other class in the balanced accuracy above (one class in 8).

The scores below leave out the cases of actinic keratosis, to show how much it moves the result. They are shown beside the main score, never instead of it. The answer options stay the same 8 classes, so answering actinic keratosis for another case still counts as wrong.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **53.5%** | 46.1% to 60.6% (bootstrap over articles) |
| Macro-F1 | 53.8% | 45.7% to 61.1% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 5.0% (11/219) | |

The cases left out, and the answers:

| Case | True class | Answer |
|---|---|---|
| `test/actinic keratosis/3842183-1.txt` | actinic keratosis | squamous cell carcinoma |
| `test/actinic keratosis/pubmed-41181771.txt` | actinic keratosis | squamous cell carcinoma |

## Sensitivity analysis: the last line of the answer

The main score reads an answer only when it is exactly one class name, as the prompt of the first phase asked. 1 answers wrote their reasoning first and then a class name alone on the last line; 1 of them name the right class. Here they are read as that class: a check on the main score, never a replacement for it.

| Score | Value | 95% interval |
|---|---|---|
| **Balanced accuracy (main score)** | **47.1%** | 40.9% to 53.1% (bootstrap over articles) |
| Macro-F1 | 47.2% | 40.2% to 53.2% (bootstrap over articles) |
| Invalid answers (counted as wrong) | 4.5% (10/221) | |

## Notes

- How answers ended (finish reason): MAX_TOKENS: 9, STOP: 212
- 9 answers reached the output limit but already differed from every class name, so the limit did not change their score; they count as wrong.
- Sessions (starts and restarts of this run): 1; Git commits: 387fcfe590
- Interruptions before the run finished: 0
