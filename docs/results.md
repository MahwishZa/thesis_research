# Closed-book results so far

Temporary page, removed with `docs/` when the research is final. Every number comes from the saved `metrics.json` of a finished test
run in `results/` (one run per model and dataset). Nothing here is final: the datasets are not frozen yet (see `docs/progress.md`).
"Closed-book" means the model gets the question alone and no reference text.

## How to read the numbers

- **Balanced accuracy** is the main score: the recall of each class, averaged. Every class counts the same, however rare it is. A model
  that always guesses has 1 divided by the number of classes (12.5% on the published cases with 8 classes, 16.7% on the clinic data with 6).
- **95% interval:** the range the true score probably lies in. It is computed by resampling whole patients (or articles), not single
  photos, because one patient can have several photos. A wide interval means the test split is small for that class or dataset.
- **Macro-F1** is the average over classes of a score that balances precision and recall. **Top-3** is the share of cases where the
  right class is among the model's three listed diagnoses, averaged over classes.
- **Invalid answers** could not be read (for example a wrong key name in the answer). They count as wrong and are shown apart.

## Claude Sonnet 5.5, all six datasets (test split)

| Dataset | What the model sees | Questions | Balanced accuracy (95% interval) | Macro-F1 | Top-3 | Invalid | Chance |
|---|---|---|---|---|---|---|---|
| `published_text` | the case text | 209 | 61.0% (55.1 to 66.3) | 60.9% | 91.4% | 0 | 12.5% |
| `published_text_photos` | the case text and the article's photos | 100 | 53.2% (46.0 to 60.5) | 51.4% | 85.7% | 1 | 12.5% |
| `published_photos` | the article's photos only | 100 | 39.8% (32.7 to 46.6) | 37.6% | 75.2% | 2 | 12.5% |
| `clinic_record` | the clinic record only | 678 | 42.0% (38.6 to 45.9) | 40.3% | 67.4% | 0 | 16.7% |
| `clinic_photo` | the phone photo only | 678 | 58.0% (52.7 to 62.8) | 52.8% | 94.3% | 8 | 16.7% |
| `clinic_photo_record` | the phone photo and the record | 678 | 64.5% (58.6 to 70.0) | 58.8% | 95.6% | 1 | 16.7% |

The two photo datasets of the published cases hold only the 100 test cases that have photos. The 209 cases of `published_text` are a
different, larger set, so the first row must not be compared with the next two rows directly. The paired comparison below uses the
same 100 cases for both.

### Recall of each class (%)

Published cases (8 classes):

| Class | `published_text` | `published_text_photos` | `published_photos` |
|---|---|---|---|
| actinic keratosis | 0 | 0 | 0 |
| basal cell carcinoma | 57 | 79 | 63 |
| dermatofibroma | 65 | 10 | 0 |
| melanoma | 65 | 47 | 53 |
| nevus | 59 | 47 | 47 |
| pigmented benign keratosis | 61 | 70 | 10 |
| squamous cell carcinoma | 86 | 82 | 82 |
| vascular lesion | 94 | 91 | 64 |

Clinic data (6 classes):

| Class | Test cases | `clinic_record` | `clinic_photo` | `clinic_photo_record` |
|---|---|---|---|---|
| actinic keratosis | 217 | 59 | 38 | 50 |
| basal cell carcinoma | 250 | 71 | 65 | 78 |
| melanoma | 15 | 0 | 80 | 80 |
| nevus | 73 | 74 | 63 | 78 |
| seborrheic keratosis | 70 | 30 | 36 | 36 |
| squamous cell carcinoma | 53 | 19 | 66 | 66 |

Melanoma has only 15 test cases in the clinic data, so each of its recalls rests on very few cases (80% is 12 of 15).

## Paired comparisons (same questions, Claude Sonnet 5.5)

Each row compares two runs on the same questions. **B minus A** is the score of run B minus the score of run A: a positive number means
B scored higher. The interval is a paired bootstrap, so it accounts for both runs answering the same patients.
McNemar's test counts only right against wrong answers, without weighting by class; it is a supporting test.

| A (first) | B (second) | Questions | Balanced accuracy, B minus A (95% interval) | Macro-F1, B minus A | Answers right only in B / only in A | McNemar p |
|---|---|---|---|---|---|---|
| `clinic_record` | `clinic_photo` | 678 | +15.9 (+9.7 to +22.0) | +12.5 | 138 / 164 | 0.15 |
| `clinic_photo` | `clinic_photo_record` | 678 | +6.6 (+2.0 to +11.0) | +6.0 | 96 / 28 | below 0.0001 |
| `clinic_record` | `clinic_photo_record` | 678 | +22.5 (+16.3 to +28.3) | +18.5 | 128 / 86 | 0.0049 |
| `published_text` | `published_text_photos` | 100 | -10.7 (-18.1 to -3.4) | -13.3 | 5 / 15 | 0.041 |
| `published_text_photos` | `published_photos` | 100 | -13.4 (-20.8 to -6.5) | -13.7 | 5 / 17 | 0.017 |
| `published_text` | `published_photos` | 100 | -24.1 (-32.6 to -15.7) | -27.0 | 5 / 27 | 0.0001 |

On these 100 cases the text alone scores 53.2 + 10.7 = 63.9%, which is why it differs from the 61.0% on all 209 cases.

## What the results show

1. **Clinic data: photo and record together are best** (64.5%), then the photo (58.0%), then the record (42.0%). The paired intervals of
   all three comparisons stay above 0. The record alone finds none of the 15 melanomas and few squamous cell carcinomas (19%); the
   photo finds most of them.
2. **The record against the photo:** the balanced-accuracy gap (+15.9) is clear, but McNemar's test is not (p = 0.15), because the two
   runs are right about equally often overall (138 questions only the photo got right, 164 only the record). The gap probably comes
   from the rare classes (melanoma, squamous cell carcinoma), which balanced accuracy weights fully; this was not tested. Report the balanced accuracy as the main result and say that McNemar's test does not agree.
3. **Published cases: the text alone is best.** Adding the article's photos lowers the score by 10.7 points, and the photos alone score
   24.1 points below the text. A published photo is chosen to illustrate a diagnosis already known to the authors, which may explain it;
   this was not tested.
4. **Actinic keratosis is never found in the published cases (0% in all three views).** The cause was not checked.
5. **Model comparison on `published_text` only** (the one dataset run for several models):

| Model | Balanced accuracy (95% interval) | Macro-F1 | Top-3 | Invalid |
|---|---|---|---|---|
| Claude Sonnet 5.5 | 61.0% (55.1 to 66.3) | 60.9% | 91.4% | 0 |
| Gemini 3.8 Flash | 59.6% (53.5 to 65.1) | 58.6% | 91.6% | 0 |
| GPT-4o | 56.1% (45.5 to 66.3) | 52.8% | 87.5% | 1 |
| Llama 4 Maverick | 41.9% (35.9 to 47.9) | 41.6% | 83.7% | 4 |

   The intervals of Claude, Gemini and GPT-4o overlap, so these three are not clearly different; Llama is lower than Claude and Gemini.

## Limits to keep in mind

- One run per model and dataset, one prompt, and the test split only. The numbers say nothing yet about other models on the other five
  datasets (Gemini is next; GPT-4o and Llama wait for funding).
- **The "evidence sufficient" flag is almost unused by Claude** (it says yes in 0% to 10% of answers, never wrongly), so coverage and
  selective risk say little for it. Gemini uses the flag in 33% of its `published_text` answers.
- 12 of the 1,556 Claude answers (0.8%) on the four datasets that show photos were invalid: 11 because a claim had a key that is
  not allowed (an extra key such as `similar_lesion`, or a wrong spelling of `statement`) and 1 because the top-level key `evidence`
  was missing. They count as wrong; the reader rule was not changed again.
- The labels of the published cases were not checked by a clinician (a stated limitation of the dataset).
- The Version 1 numbers (`archive/v1/`) are obsolete and are not compared with these.
