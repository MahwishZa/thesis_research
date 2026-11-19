# Method: questions, answers and scores

This page fixes how every run from now on asks its questions, reads the answers and scores them. It was written before any run of this answer format, and it applies to every model and dataset alike. The first phase of Version 1 used an earlier answer format; section 6 says where it is.

## 1. The question

Each question is one item of a dataset's test split: a clinic photo (`clinic_photo`), a clinic patient's record without the photo (`clinic_record`), the photo with the same record (`clinic_photo_record`), a published case report's text, which ends before the diagnosis (`published_text`), the same text with the photos of the same report (`published_text_photos`), or those photos alone (`published_photos`). The three clinic views ask the same questions, and so do the three views of a published case (the two with photos ask the cases that have photos, some of the text's), so their answers can be paired. The prompt (`prompts/<dataset>/<approach>.txt`) is the same for every model on a dataset. It lists the possible diagnoses (six for the clinic patients, eight for the published cases), and the model must choose among them.

Approaches:

- **Closed-book:** the question alone. No reference text is given.
- **Open-book (retrieval):** the question with passages taken from the knowledge index (`data/knowledge/index.csv`). Not built yet; it will use the same answer format and the same scores.

## 2. The answer

The model answers with one JSON object and nothing else:

```
{"differential": ["<most likely>", "<second>", "<third>"], "evidence_sufficient": "yes" or "no", "reason": "<at most three sentences>", "claims": [{"statement": "<a short claim>", "evidence": [<passage ids>]}], "evidence": [<passage ids>]}
```

- **differential:** three different diagnoses from the list, the most likely first. The first one is the model's diagnosis.
- **evidence_sufficient:** "yes" if what the model was given is enough to be confident of its diagnosis, otherwise "no". This is how a model can say it does not know, without leaving the question unanswered.
- **reason:** a short justification. It is saved but not scored.
- **claims** (answer format `json-v2`, approved 2026-11-19): one to three short statements that support the first diagnosis, each with the ids of the passages it relies on. Closed-book, every claim's evidence is empty. The claims are what a later evidence-grounding score checks (is each claim supported by the passage it cites); the score is not defined yet, and the field is only read, saved and checked for valid ids.
- **evidence:** the ids of the passages the reason relies on. Closed-book, it must be empty.

**Reading (strict).** The reader (`parse_json_answer` in `src/prompts.py`) forgives only spaces around the object, a Markdown code block around it, and the case and spacing of a diagnosis name or of yes/no. Everything else makes the answer **invalid**: text before or after the object, a missing or added key, a differential that is not three different diagnoses of the list, a flag that is not yes or no, an empty reason, and evidence that names a passage the run did not give (closed-book: any evidence). An answer cut off at the output limit is invalid; the report says how many were cut off, and if any were, the results are not valid until the limit is raised under a new model entry.

**An invalid answer counts as wrong** in every score, and the share of invalid answers is reported on its own. The reason each answer was invalid is saved in `predictions.jsonl` and counted in the report.

## 3. The scores

Every score is computed by `src/metrics.py` from the saved answers. The classes are weighted equally: the published cases have unequal numbers of cases per class, and a model can reach a high plain accuracy by naming the most common class.

| Score | Definition | Source |
|---|---|---|
| **Balanced accuracy (main score)** | The mean recall over the classes present in the test split. Recall of a class: of its questions, the share whose diagnosis is that class. Chance is 1 over the number of classes (1 / 6 = 16.7% for the clinic patients, 1 / 8 = 12.5% for the published cases). | Brodersen et al. 2010 |
| **Macro-F1** (second score) | The mean F1 over the classes present. F1 of a class: the harmonic mean of its precision and recall. A class never given has no precision and counts as 0. | Sokolova & Lapalme 2009 |
| Per-class recall, precision and F1; confusion matrix | In each run's report. The confusion matrix has one more column for invalid answers. | Sokolova & Lapalme 2009 |
| **Top-3 balanced accuracy** | The mean over the classes of the share of questions whose true class is anywhere in the differential. It shows whether the right diagnosis was considered at all. | Liu et al. 2020 (top-3 accuracy of a skin-disease differential) |
| **Invalid-answer rate** | The share of questions with an invalid answer. | – |
| **Misdiagnosis detection: sensitivity** | Over the valid answers. The positive case is a wrong diagnosis. Of the wrong diagnoses, the share the model marked with "evidence_sufficient": "no". | Hendrycks & Gimpel 2017 (detecting misclassified examples) |
| **Misdiagnosis detection: specificity** | Of the right diagnoses, the share the model did not mark. | Hendrycks & Gimpel 2017 |
| **Coverage** | Of all questions, the share answered with "yes". An invalid answer is not covered. | Geifman & El-Yaniv 2017 |
| **Confident misdiagnosis (selective risk)** | Of the covered questions, the share with a wrong diagnosis: the error rate when the model's own "yes" is trusted. | Geifman & El-Yaniv 2017 |

Plain accuracy is not reported: it adds nothing to balanced accuracy where the classes are equal in size, and it misleads where they are not. A ROC curve and its AUC are not reported: they need a score for every class on every question, and the answers give a ranked list of three, not scores. The evidence flag gives one operating point (the sensitivity and specificity above), not a curve.

**Confident misdiagnosis** (named so on 2026-11-19; earlier "hallucination"). A confident misdiagnosis is a wrong diagnosis marked "yes". Its share among the covered questions is the selective risk. A method reduces it if it lowers the selective risk without lowering the balanced accuracy, or if it raises the misdiagnosis detection sensitivity at a similar specificity. It is one part of the hallucination the research studies: it does not measure unsupported claims in the reason, which need an evidence-grounding score (not defined yet).

## 4. Intervals and comparisons

**95% intervals.** A percentile bootstrap (Efron & Tibshirani 1993) with 2,000 resamples and a fixed seed (0). Each resample draws whole groups with replacement inside each class, so every class keeps its number of groups. The group is the patient (clinic views: several photos of one patient are not independent), the article (text: several patients of one article are not independent) or the case (cases); resampling whole groups follows Field & Welsh 2007. When every question is its own group, a share (one class's recall, coverage) gets a Wilson interval instead (Wilson 1927).

**Comparing two runs on the same questions** (two models, or one model with and without a method):

- **Main comparison:** the difference in balanced accuracy, with a paired 95% interval: both runs are resampled together, with the same groups, 2,000 times. A difference whose interval excludes 0 is called beyond chance. The difference in macro-F1 is given the same way.
- **Supporting test:** McNemar's exact test on the questions exactly one of the two runs answers right (McNemar 1947). Where a page holds several tests, each p-value is also given after the Holm correction for their number (Holm 1979).

The intervals are not corrected for the number of comparisons, so on a page of many comparisons one interval can exclude 0 by chance. Where the interval excludes 0 but the Holm-corrected p-value is above 0.05, the results page says so and does not call the difference established.

## 5. Analyses beside the main score

Each is reported beside the main score, never instead of it.

- **Without actinic keratosis** (`published_text`, `published_text_photos`, `published_photos`). The published cases' test split has only 2 actinic keratosis cases, which weigh one eighth of the balanced accuracy. The scores are also given without them. The answer options stay the same.
- **Questions with a complete record** (`clinic_photo_record`). A record line the patient could not answer reads `unknown`, and the pattern of gaps can point to the class. The scores are also given for the questions with no `unknown` in the record.
- **Hints that are not the lesion** (clinic patients). The `shortcuts` step of the preparation (`data/clinic_patients/dataset_report.md`) measures how much the empty record fields and the photo's properties alone give away about the class; those numbers are reported beside the results.

## 6. The first phase of Version 1

The first phase of Version 1 asked for one diagnosis name alone, read with a strict class-name reader. Its 12 runs, their results and
their interpretation are in `archive/v1/` (`archive/v1/docs/results.md`). The code of this repository reads the answer format of
section 2 only and refuses a run of the first phase, with a pointer to the archive.

## References

- Brodersen KH, Ong CS, Stephan KE, Buhmann JM. The balanced accuracy and its posterior distribution. *20th International Conference on Pattern Recognition (ICPR)*, 2010.
- Efron B, Tibshirani RJ. *An Introduction to the Bootstrap.* Chapman & Hall, 1993.
- Field CA, Welsh AH. Bootstrapping clustered data. *Journal of the Royal Statistical Society, Series B* 69(3), 2007.
- Geifman Y, El-Yaniv R. Selective classification for deep neural networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
- Hendrycks D, Gimpel K. A baseline for detecting misclassified and out-of-distribution examples in neural networks. *International Conference on Learning Representations (ICLR)*, 2017.
- Holm S. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2), 1979.
- Liu Y, Jain A, Eng C, et al. A deep learning system for differential diagnosis of skin diseases. *Nature Medicine* 26, 2020.
- McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* 12(2), 1947.
- Sokolova M, Lapalme G. A systematic analysis of performance measures for classification tasks. *Information Processing & Management* 45(4), 2009.
- Wilson EB. Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association* 22(158), 1927.
