# Results: closed-book runs on the test split

Temporary page, removed with `docs/` when the research is final. Every number comes from a finished run in `results/` and nothing is
estimated. **An empty cell means the run has not been done yet**; `n/a` means the score cannot be computed. The datasets are not frozen,
so nothing here is final.

| Section | What it holds |
|---|---|
| [1. How to read](#1-how-to-read-this-page) | the scores in plain words, and how to add a new result |
| [2. Overview](#2-overview-balanced-accuracy-of-every-model-on-every-dataset) | one number per model and dataset |
| [3. Results by dataset](#3-results-by-dataset) | all scores of every model, one table per dataset |
| [4. Recall of each class](#4-recall-of-each-class) | which diagnoses each model finds |
| [5. Beside the main score](#5-beside-the-main-score) | two checks that are always reported with it |
| [6. Paired comparisons](#6-paired-comparisons) | is a difference more than chance? |
| [7. Open-book results](#7-open-book-results-not-started) | empty tables for the later approaches |
| [8. What the results show](#8-what-the-results-show-so-far) and [9. Limits](#9-limits) | short summary |

## 1. How to read this page

**The models.** Claude Sonnet 5.5, GPT-4o, Gemini 3.8 Flash and Llama 4 Maverick, each asked the same questions with the same prompt (the
exact versions are in `configs/models.toml`). One run per model and dataset.

**The datasets** (the test split only; the diagnosis options are listed in each prompt):

| Dataset | What the model sees | Test questions | Classes | Chance |
|---|---|---|---|---|
| `published_text` | the case text | 209 cases | 8 | 12.5% |
| `published_text_photos` | the case text and the article's photos | 100 cases | 8 | 12.5% |
| `published_photos` | the article's photos only | 100 cases | 8 | 12.5% |
| `clinic_record` | the clinic record only | 678 patients' records | 6 | 16.7% |
| `clinic_photo` | the phone photo only | 678 photos | 6 | 16.7% |
| `clinic_photo_record` | the phone photo and the record | 678 photos with records | 6 | 16.7% |

The two published-photo datasets hold only the 100 test cases that have photos; the 209 cases of `published_text` are a larger set.
Compare the three published datasets only on the same 100 cases (section 6).

**The scores** (full definitions and sources: `docs/method.md` section 3):

| Score | In plain words |
|---|---|
| **Balanced accuracy** (main score) | The share of right diagnoses for each class, averaged over the classes, so a rare class counts as much as a common one. |
| **95% interval** | The range the true score probably lies in. A wide range means few cases. |
| Macro-F1 | A second score that balances missed cases and false alarms, averaged over the classes. |
| Top-3 | How often the right class is among the model's three listed diagnoses, averaged over the classes. |
| Invalid answers | Answers that could not be read, or questions the host refused (below). They count as wrong. Shown as number (share). |
| Refused by the host | Questions the host's content moderation refused (for example a photo flagged as graphic), so the model never saw them. Each is saved as an invalid answer; this column says how many. |
| Coverage | The share of answers where the model says its evidence is sufficient ("yes"). |
| Selective risk | Of those "yes" answers, the share with a wrong diagnosis (a confident misdiagnosis). `n/a` when there are none. |

**Adding a new result.** After a run is committed:

1. Print the tables: `python -m src.evaluate tables --output tables.md` (about a minute; use a file outside the repository, for example
   `$env:TEMP\tables.md` in PowerShell). Every score is computed from the saved answers.
2. In this page, replace the tables of sections 2 to 6 by the ones in that file (same headings). A test checks that every table line
   the command prints is in this page.
3. Write the run in the status board of `docs/progress.md`.

The numbers of one run are also in its folder `results/<dataset>/closed_book/<model>/<run>/` (`report.md`), and a comparison of any two
runs is `python -m src.evaluate compare <folder of run A> <folder of run B>`.

## 2. Overview: balanced accuracy of every model on every dataset

Balanced accuracy in % (95% interval), test split.

| Dataset | Claude Sonnet 5.5 | GPT-4o | Gemini 3.8 Flash | Llama 4 Maverick |
|---|---|---|---|---|
| `published_text` | 61.0% (55.1 to 66.3) | 56.1% (45.5 to 66.3) | 59.6% (53.5 to 65.1) | 41.9% (35.9 to 47.9) |
| `published_text_photos` | 53.2% (46.0 to 60.5) |  |  | 46.1% (39.3 to 53.4) |
| `published_photos` | 39.8% (32.7 to 46.6) |  |  | 30.0% (23.7 to 36.2) |
| `clinic_record` | 42.0% (38.6 to 45.9) |  |  |  |
| `clinic_photo` | 58.0% (52.7 to 62.8) |  |  |  |
| `clinic_photo_record` | 64.5% (58.6 to 70.0) |  |  |  |

## 3. Results by dataset

Every table has the same columns. Run = the results folder of that run.

### 3.1 `published_text`: the case text

209 cases, 8 classes, chance 12.5%.

| Model | Balanced accuracy | 95% interval | Macro-F1 | Top-3 | Invalid answers | Refused by the host | Coverage | Selective risk | Run |
|---|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 61.0% | 55.1 to 66.3 | 60.9% | 91.4% | 0 (0.0%) | 0 | 5.3% | 0.0% (0 of 11) | `test_3a656648` |
| GPT-4o | 56.1% | 45.5 to 66.3 | 52.8% | 87.5% | 1 (0.5%) | 0 | 1.9% | 25.0% (1 of 4) | `test_2fab9eb6` |
| Gemini 3.8 Flash | 59.6% | 53.5 to 65.1 | 58.6% | 91.6% | 0 (0.0%) | 0 | 33.0% | 10.1% (7 of 69) | `test_6e2bd64c` |
| Llama 4 Maverick | 41.9% | 35.9 to 47.9 | 41.6% | 83.7% | 4 (1.9%) | 0 | 6.7% | 28.6% (4 of 14) | `test_5719a8d3` |

### 3.2 `published_text_photos`: the case text and the article's photos

100 cases, 8 classes, chance 12.5%.

| Model | Balanced accuracy | 95% interval | Macro-F1 | Top-3 | Invalid answers | Refused by the host | Coverage | Selective risk | Run |
|---|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 53.2% | 46.0 to 60.5 | 51.4% | 85.7% | 1 (1.0%) | 0 | 10.0% | 0.0% (0 of 10) | `test_5ef83f27` |
| GPT-4o |  |  |  |  |  |  |  |  |  |
| Gemini 3.8 Flash |  |  |  |  |  |  |  |  |  |
| Llama 4 Maverick | 46.1% | 39.3 to 53.4 | 44.9% | 85.1% | 2 (2.0%) | 0 | 10.0% | 30.0% (3 of 10) | `test_53efcf3d` |

### 3.3 `published_photos`: the article's photos only

100 cases, 8 classes, chance 12.5%.

| Model | Balanced accuracy | 95% interval | Macro-F1 | Top-3 | Invalid answers | Refused by the host | Coverage | Selective risk | Run |
|---|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 39.8% | 32.7 to 46.6 | 37.6% | 75.2% | 2 (2.0%) | 0 | 3.0% | 0.0% (0 of 3) | `test_d56264d6` |
| GPT-4o |  |  |  |  |  |  |  |  |  |
| Gemini 3.8 Flash |  |  |  |  |  |  |  |  |  |
| Llama 4 Maverick | 30.0% | 23.7 to 36.2 | 24.8% | 72.5% | 0 (0.0%) | 0 | 3.0% | 100.0% (3 of 3) | `test_69dbecaf` |

### 3.4 `clinic_record`: the clinic record only

678 patients' records, 6 classes, chance 16.7%.

| Model | Balanced accuracy | 95% interval | Macro-F1 | Top-3 | Invalid answers | Refused by the host | Coverage | Selective risk | Run |
|---|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 42.0% | 38.6 to 45.9 | 40.3% | 67.4% | 0 (0.0%) | 0 | 0.0% | n/a | `test_7ae640ea` |
| GPT-4o |  |  |  |  |  |  |  |  |  |
| Gemini 3.8 Flash |  |  |  |  |  |  |  |  |  |
| Llama 4 Maverick |  |  |  |  |  |  |  |  |  |

### 3.5 `clinic_photo`: the phone photo only

678 photos, 6 classes, chance 16.7%.

| Model | Balanced accuracy | 95% interval | Macro-F1 | Top-3 | Invalid answers | Refused by the host | Coverage | Selective risk | Run |
|---|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 58.0% | 52.7 to 62.8 | 52.8% | 94.3% | 8 (1.2%) | 0 | 0.1% | 0.0% (0 of 1) | `test_06deef76` |
| GPT-4o |  |  |  |  |  |  |  |  |  |
| Gemini 3.8 Flash |  |  |  |  |  |  |  |  |  |
| Llama 4 Maverick |  |  |  |  |  |  |  |  |  |

### 3.6 `clinic_photo_record`: the phone photo and the record

678 photos with records, 6 classes, chance 16.7%.

| Model | Balanced accuracy | 95% interval | Macro-F1 | Top-3 | Invalid answers | Refused by the host | Coverage | Selective risk | Run |
|---|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 64.5% | 58.6 to 70.0 | 58.8% | 95.6% | 1 (0.1%) | 0 | 0.4% | 0.0% (0 of 3) | `test_16d5a224` |
| GPT-4o |  |  |  |  |  |  |  |  |  |
| Gemini 3.8 Flash |  |  |  |  |  |  |  |  |  |
| Llama 4 Maverick |  |  |  |  |  |  |  |  |  |

## 4. Recall of each class

The recall of a class is the share of its test questions the model diagnosed right (%). The balanced accuracy is the mean of these
numbers. Intervals are in each run's `report.md`. Cases = test questions of the class.

### 4.1 `published_text`

| Class | Cases | Claude Sonnet 5.5 | GPT-4o | Gemini 3.8 Flash | Llama 4 Maverick |
|---|---|---|---|---|---|
| actinic keratosis | 2 | 0.0 | 50.0 | 0.0 | 0.0 |
| basal cell carcinoma | 40 | 57.5 | 45.0 | 62.5 | 30.0 |
| dermatofibroma | 20 | 65.0 | 50.0 | 55.0 | 20.0 |
| melanoma | 37 | 64.9 | 56.8 | 70.3 | 51.4 |
| nevus | 39 | 59.0 | 56.4 | 48.7 | 53.8 |
| pigmented benign keratosis | 18 | 61.1 | 38.9 | 72.2 | 33.3 |
| squamous cell carcinoma | 37 | 86.5 | 70.3 | 86.5 | 83.8 |
| vascular lesion | 16 | 93.8 | 81.2 | 81.2 | 62.5 |

### 4.2 `published_text_photos`

| Class | Cases | Claude Sonnet 5.5 | GPT-4o | Gemini 3.8 Flash | Llama 4 Maverick |
|---|---|---|---|---|---|
| actinic keratosis | 1 | 0.0 |  |  | 0.0 |
| basal cell carcinoma | 19 | 78.9 |  |  | 42.1 |
| dermatofibroma | 10 | 10.0 |  |  | 20.0 |
| melanoma | 17 | 47.1 |  |  | 58.8 |
| nevus | 15 | 46.7 |  |  | 40.0 |
| pigmented benign keratosis | 10 | 70.0 |  |  | 20.0 |
| squamous cell carcinoma | 17 | 82.4 |  |  | 88.2 |
| vascular lesion | 11 | 90.9 |  |  | 100.0 |

### 4.3 `published_photos`

| Class | Cases | Claude Sonnet 5.5 | GPT-4o | Gemini 3.8 Flash | Llama 4 Maverick |
|---|---|---|---|---|---|
| actinic keratosis | 1 | 0.0 |  |  | 0.0 |
| basal cell carcinoma | 19 | 63.2 |  |  | 0.0 |
| dermatofibroma | 10 | 0.0 |  |  | 10.0 |
| melanoma | 17 | 52.9 |  |  | 52.9 |
| nevus | 15 | 46.7 |  |  | 40.0 |
| pigmented benign keratosis | 10 | 10.0 |  |  | 0.0 |
| squamous cell carcinoma | 17 | 82.4 |  |  | 82.4 |
| vascular lesion | 11 | 63.6 |  |  | 54.5 |

### 4.4 `clinic_record`

| Class | Cases | Claude Sonnet 5.5 | GPT-4o | Gemini 3.8 Flash | Llama 4 Maverick |
|---|---|---|---|---|---|
| actinic keratosis | 217 | 58.5 |  |  |  |
| basal cell carcinoma | 250 | 70.8 |  |  |  |
| melanoma | 15 | 0.0 |  |  |  |
| nevus | 73 | 74.0 |  |  |  |
| seborrheic keratosis | 70 | 30.0 |  |  |  |
| squamous cell carcinoma | 53 | 18.9 |  |  |  |

### 4.5 `clinic_photo`

| Class | Cases | Claude Sonnet 5.5 | GPT-4o | Gemini 3.8 Flash | Llama 4 Maverick |
|---|---|---|---|---|---|
| actinic keratosis | 217 | 37.8 |  |  |  |
| basal cell carcinoma | 250 | 65.2 |  |  |  |
| melanoma | 15 | 80.0 |  |  |  |
| nevus | 73 | 63.0 |  |  |  |
| seborrheic keratosis | 70 | 35.7 |  |  |  |
| squamous cell carcinoma | 53 | 66.0 |  |  |  |

### 4.6 `clinic_photo_record`

| Class | Cases | Claude Sonnet 5.5 | GPT-4o | Gemini 3.8 Flash | Llama 4 Maverick |
|---|---|---|---|---|---|
| actinic keratosis | 217 | 49.8 |  |  |  |
| basal cell carcinoma | 250 | 77.6 |  |  |  |
| melanoma | 15 | 80.0 |  |  |  |
| nevus | 73 | 78.1 |  |  |  |
| seborrheic keratosis | 70 | 35.7 |  |  |  |
| squamous cell carcinoma | 53 | 66.0 |  |  |  |

## 5. Beside the main score

Each check is reported next to the main score, never instead of it (`docs/method.md` section 5). Columns as in section 3.

### 5.1 Without actinic keratosis (published cases)

The test split has only 2 actinic keratosis cases (1 in each photo dataset). Each weighs one eighth of the balanced accuracy, so the
score is also given without them. The answer options do not change.

| Dataset | Model | Questions | Balanced accuracy | 95% interval | Macro-F1 |
|---|---|---|---|---|---|
| `published_text` | Claude Sonnet 5.5 | 207 | 69.7% | 63.0 to 75.9 | 69.8% |
| `published_text` | GPT-4o | 207 | 56.9% | 50.0 to 63.9 | 57.3% |
| `published_text` | Gemini 3.8 Flash | 207 | 68.1% | 61.1 to 74.5 | 67.2% |
| `published_text` | Llama 4 Maverick | 207 | 47.8% | 40.6 to 54.4 | 47.7% |
| `published_text_photos` | Claude Sonnet 5.5 | 99 | 60.8% | 52.6 to 69.1 | 58.9% |
| `published_text_photos` | GPT-4o |  |  |  |  |
| `published_text_photos` | Gemini 3.8 Flash |  |  |  |  |
| `published_text_photos` | Llama 4 Maverick | 99 | 52.7% | 44.7 to 60.7 | 51.5% |
| `published_photos` | Claude Sonnet 5.5 | 99 | 45.5% | 37.6 to 53.8 | 43.2% |
| `published_photos` | GPT-4o |  |  |  |  |
| `published_photos` | Gemini 3.8 Flash |  |  |  |  |
| `published_photos` | Llama 4 Maverick | 99 | 34.3% | 26.9 to 41.9 | 28.6% |

### 5.2 Questions with a complete record (clinic data)

A record line the patient could not answer reads "unknown", and a gap can point to the class. These scores count only the questions
with no "unknown".

| Dataset | Model | Questions | Balanced accuracy | 95% interval | Macro-F1 |
|---|---|---|---|---|---|
| `clinic_record` | Claude Sonnet 5.5 | 670 | 42.1% | 38.3 to 45.9 | 40.2% |
| `clinic_record` | GPT-4o |  |  |  |  |
| `clinic_record` | Gemini 3.8 Flash |  |  |  |  |
| `clinic_record` | Llama 4 Maverick |  |  |  |  |
| `clinic_photo_record` | Claude Sonnet 5.5 | 670 | 64.7% | 58.8 to 69.9 | 59.2% |
| `clinic_photo_record` | GPT-4o |  |  |  |  |
| `clinic_photo_record` | Gemini 3.8 Flash |  |  |  |  |
| `clinic_photo_record` | Llama 4 Maverick |  |  |  |  |

## 6. Paired comparisons

Two runs are compared on the same questions. **A is the first run, B the second; "B minus A" is B's score minus A's** (positive = B is
higher), in percentage points. The interval is a paired bootstrap: if it does not hold 0, the difference is more than chance.
"Right only in B / only in A" counts questions that exactly one of the two runs answered right; McNemar's exact test (p) is a
supporting test on those counts. Balanced accuracy is the main result; the two can disagree because McNemar's test does not weigh the
classes equally.

### 6.1 The same model with different inputs

| Model | A | B | Questions | Balanced accuracy, B minus A | 95% interval | Macro-F1, B minus A | Right only in B / only in A | McNemar p |
|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | `published_text` | `published_text_photos` | 100 | -10.7 | -18.1 to -3.4 | -13.3 | 5 / 15 | 0.0414 |
| Claude Sonnet 5.5 | `published_text_photos` | `published_photos` | 100 | -13.4 | -20.8 to -6.5 | -13.7 | 5 / 17 | 0.0169 |
| Claude Sonnet 5.5 | `published_text` | `published_photos` | 100 | -24.1 | -32.6 to -15.7 | -27.0 | 5 / 27 | 0.0001 |
| Claude Sonnet 5.5 | `clinic_record` | `clinic_photo` | 678 | +15.9 | +9.7 to +22.0 | +12.5 | 138 / 164 | 0.1501 |
| Claude Sonnet 5.5 | `clinic_photo` | `clinic_photo_record` | 678 | +6.6 | +2.0 to +11.0 | +6.0 | 96 / 28 | <0.0001 |
| Claude Sonnet 5.5 | `clinic_record` | `clinic_photo_record` | 678 | +22.5 | +16.3 to +28.3 | +18.5 | 128 / 86 | 0.0049 |
| GPT-4o | `published_text` | `published_text_photos` | 100 |  |  |  |  |  |
| GPT-4o | `published_text_photos` | `published_photos` | 100 |  |  |  |  |  |
| GPT-4o | `published_text` | `published_photos` | 100 |  |  |  |  |  |
| GPT-4o | `clinic_record` | `clinic_photo` | 678 |  |  |  |  |  |
| GPT-4o | `clinic_photo` | `clinic_photo_record` | 678 |  |  |  |  |  |
| GPT-4o | `clinic_record` | `clinic_photo_record` | 678 |  |  |  |  |  |
| Gemini 3.8 Flash | `published_text` | `published_text_photos` | 100 |  |  |  |  |  |
| Gemini 3.8 Flash | `published_text_photos` | `published_photos` | 100 |  |  |  |  |  |
| Gemini 3.8 Flash | `published_text` | `published_photos` | 100 |  |  |  |  |  |
| Gemini 3.8 Flash | `clinic_record` | `clinic_photo` | 678 |  |  |  |  |  |
| Gemini 3.8 Flash | `clinic_photo` | `clinic_photo_record` | 678 |  |  |  |  |  |
| Gemini 3.8 Flash | `clinic_record` | `clinic_photo_record` | 678 |  |  |  |  |  |
| Llama 4 Maverick | `published_text` | `published_text_photos` | 100 | +8.4 | +1.3 to +15.8 | +9.2 | 13 / 3 | 0.0213 |
| Llama 4 Maverick | `published_text_photos` | `published_photos` | 100 | -16.2 | -23.4 to -9.3 | -20.1 | 3 / 21 | 0.0003 |
| Llama 4 Maverick | `published_text` | `published_photos` | 100 | -7.7 | -16.5 to +0.9 | -10.8 | 9 / 17 | 0.1686 |
| Llama 4 Maverick | `clinic_record` | `clinic_photo` | 678 |  |  |  |  |  |
| Llama 4 Maverick | `clinic_photo` | `clinic_photo_record` | 678 |  |  |  |  |  |
| Llama 4 Maverick | `clinic_record` | `clinic_photo_record` | 678 |  |  |  |  |  |

### 6.2 Models against each other

A is Claude Sonnet 5.5 (the first model run), B the other model, on the same questions.

| Dataset | B | Questions | Balanced accuracy, B minus A | 95% interval | Macro-F1, B minus A | Right only in B / only in A | McNemar p |
|---|---|---|---|---|---|---|---|
| `published_text` | GPT-4o | 209 | -4.9 | -14.6 to +5.2 | -8.1 | 11 / 34 | 0.0008 |
| `published_text` | Gemini 3.8 Flash | 209 | -1.4 | -6.1 to +2.8 | -2.3 | 15 / 17 | 0.8601 |
| `published_text` | Llama 4 Maverick | 209 | -19.1 | -25.4 to -13.0 | -19.3 | 10 / 48 | <0.0001 |
| `published_text_photos` | GPT-4o | 100 |  |  |  |  |  |
| `published_text_photos` | Gemini 3.8 Flash | 100 |  |  |  |  |  |
| `published_text_photos` | Llama 4 Maverick | 100 | -7.1 | -14.5 to +0.9 | -6.5 | 9 / 17 | 0.1686 |
| `published_photos` | GPT-4o | 100 |  |  |  |  |  |
| `published_photos` | Gemini 3.8 Flash | 100 |  |  |  |  |  |
| `published_photos` | Llama 4 Maverick | 100 | -9.9 | -16.2 to -3.6 | -12.8 | 4 / 18 | 0.0043 |
| `clinic_record` | GPT-4o | 678 |  |  |  |  |  |
| `clinic_record` | Gemini 3.8 Flash | 678 |  |  |  |  |  |
| `clinic_record` | Llama 4 Maverick | 678 |  |  |  |  |  |
| `clinic_photo` | GPT-4o | 678 |  |  |  |  |  |
| `clinic_photo` | Gemini 3.8 Flash | 678 |  |  |  |  |  |
| `clinic_photo` | Llama 4 Maverick | 678 |  |  |  |  |  |
| `clinic_photo_record` | GPT-4o | 678 |  |  |  |  |  |
| `clinic_photo_record` | Gemini 3.8 Flash | 678 |  |  |  |  |  |
| `clinic_photo_record` | Llama 4 Maverick | 678 |  |  |  |  |  |

## 7. Open-book results (not started)

The open-book approaches (standard RAG, RAG² and the proposed method) are not built yet. They will use the same answer format and the
same scores, plus the two retrieval scores below, with a local open-weight model (to be decided). The evidence-grounding scores are
not defined yet (`docs/method.md` section 7), so they have no column. Precision@k and nDCG@k are given beside the share expected from
random retrieval.

| Dataset | Approach | Model | Balanced accuracy | 95% interval | Macro-F1 | Top-3 | Invalid answers | Refused by the host | Coverage | Selective risk | Precision@k | nDCG@k | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `published_text` | Standard RAG | | | | | | | | | | | | |
| `published_text` | RAG² | | | | | | | | | | | | |
| `published_text` | Proposed method | | | | | | | | | | | | |
| `published_text_photos` | Standard RAG | | | | | | | | | | | | |
| `published_text_photos` | RAG² | | | | | | | | | | | | |
| `published_text_photos` | Proposed method | | | | | | | | | | | | |
| `published_photos` | Standard RAG | | | | | | | | | | | | |
| `published_photos` | RAG² | | | | | | | | | | | | |
| `published_photos` | Proposed method | | | | | | | | | | | | |
| `clinic_record` | Standard RAG | | | | | | | | | | | | |
| `clinic_record` | RAG² | | | | | | | | | | | | |
| `clinic_record` | Proposed method | | | | | | | | | | | | |
| `clinic_photo` | Standard RAG | | | | | | | | | | | | |
| `clinic_photo` | RAG² | | | | | | | | | | | | |
| `clinic_photo` | Proposed method | | | | | | | | | | | | |
| `clinic_photo_record` | Standard RAG | | | | | | | | | | | | |
| `clinic_photo_record` | RAG² | | | | | | | | | | | | |
| `clinic_photo_record` | Proposed method | | | | | | | | | | | | |

## 8. What the results show so far

Claude has been run on all six datasets and Llama on the two published photo datasets (by decision). Gemini and GPT-4o results are not in
the repository yet. Every statement below is about one run per model.

1. **Clinic data: photo and record together are best.** Claude scores 64.5% with both, 58.0% with the photo alone and 42.0% with the
   record alone. All three paired intervals stay above 0. The record alone finds none of the 15 melanomas and 19% of the squamous cell
   carcinomas; the photo finds most of them.
2. **Record against photo:** the balanced-accuracy gap (+15.9) is clear, but McNemar's test is not (p = 0.15): the two runs are right about
   equally often overall (164 questions only the record got right, 138 only the photo). The gap probably comes from the rare classes,
   which balanced accuracy weighs fully; this was not tested.
3. **Published cases: the text alone is best.** On the same 100 cases the text alone scores 63.9% (53.2 + 10.7), the text with photos
   53.2% and the photos alone 39.8%. A published photo is chosen to illustrate a diagnosis the authors already know, which may explain it;
   this was not tested.
4. **Actinic keratosis (published cases):** there are only 2 test cases (1 in the photo datasets), and Claude, Gemini and Llama got none right (GPT-4o got one of the 2 on `published_text`).
   Without them the scores rise (section 5.1; for example Claude on `published_text`, 61.0% to 69.7%).
5. **Models on `published_text`:** Gemini (-1.4 points, interval -6.1 to +2.8) is not clearly different from Claude. GPT-4o is 4.9 points
   lower, but its interval holds 0 (-14.6 to +5.2) while McNemar's test is clear (p = 0.0008). Its interval is wide probably because it is
   the only model that got one of the 2 actinic keratosis cases right: a resample that drops or doubles them moves its score by up to
   6.25 points (a class is 1/8 of the score). Llama is clearly lower (-19.1 points).
6. **The article's photos help Llama but hurt Claude.** On the same 100 cases, Llama scores 37.7% on the text alone (46.1 - 8.4), 46.1% with the
   photos added (+8.4 points, interval +1.3 to +15.8, p = 0.021) and 30.0% on the photos alone (-16.2 points against text with photos,
   interval -23.4 to -9.3). For Claude the first change had the opposite sign (-10.7). The effect of the photos therefore depends on the
   model; why was not tested.
7. **Llama against Claude on the photo datasets:** with text and photos Llama is 7.1 points lower, and the interval holds 0 (-14.5 to +0.9);
   on the photos alone it is 9.9 points lower, and the interval does not hold 0 (-16.2 to -3.6).
8. **The "evidence sufficient" flag:** Claude almost never uses it (coverage 0% to 10%), so its selective risk says little. Gemini uses it
   in 33% of its `published_text` answers and 10.1% of those are wrong (7 of 69). GPT-4o and Llama use it in 1.9% and 6.7% there; Llama on
   the two photo datasets uses it in 10.0% and 3.0%, too few answers (10 and 3) to say anything.

## 9. Limits

- One run per model and dataset, one prompt, the test split only. Llama is run on `published_text` and the two published photo datasets only,
  and GPT-4o will be run on the two published photo datasets only (decided 2026-11-20); the clinic datasets stay empty for both. Gemini
  and GPT-4o results are not in the repository yet.
- 12 of the 1,556 Claude answers (0.8%) on the four photo datasets were invalid: 11 because a claim had a key that is not allowed (an
  extra key such as `similar_lesion`, or a wrong spelling of `statement`) and 1 because the top-level key `evidence` was missing. They
  count as wrong; the reader rule was not changed again. Llama had 2 invalid answers on `published_text_photos`: in both it named a
  diagnosis that is not in the class list ("glomus tumor", "seborrheic keratosis").
- **A host's content moderation can refuse a photo.** OpenAI's moderation (through OpenRouter) refused a photo flagged "violence/graphic" in the GPT-4o runs on the published photo datasets (decided 2026-11-20, option A). A refused question is saved as an invalid answer, counts as wrong, and is counted in the column "Refused by the host". The model never saw those cases, so its score on them says nothing about its diagnosis; no other model has been refused so far. A refusal is not a fixed property of a case: the same photo (case 4 of both published photo datasets) was refused in the first attempt and answered when the run was resumed, so a rerun could refuse other cases.
- The labels of the published cases were not checked by a clinician (a stated limitation of the dataset).
- The Version 1 numbers (`archive/v1/`) are obsolete and are not compared with these.
