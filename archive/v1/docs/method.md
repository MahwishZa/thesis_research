> **Version 1** (commit `a1c9209`), archived on 2026-11-19: a frozen copy kept for reference. It does not bind Version 2 (see `docs/v2_proposal.md` in the repository).

# Method: questions, answers and scores

This page fixes how every run from now on asks its questions, reads the answers and scores them. It was written before any run of this answer format, and it applies to every model and dataset alike. The first-phase runs (`results/phase1/`) used an earlier answer format; section 6 says what changes for them.

## 1. The question

Each question is one item of a dataset's test split: a photo (image), a photo with the patient's age, sex and body site (image + metadata), a written case that ends before the diagnosis (text), or a case with its photos (cases, case photos). The prompt (`prompts/<dataset>/<approach>.txt`) is the same for every model on a dataset. It lists the eight possible diagnoses, and the model must choose among them.

Approaches:

- **Closed-book:** the question alone. No reference text is given.
- **Open-book (retrieval):** the question with passages taken from the knowledge index (`data/knowledge/index.csv`). Not built yet; it will use the same answer format and the same scores.

## 2. The answer

The model answers with one JSON object and nothing else:

```
{"differential": ["<most likely>", "<second>", "<third>"], "evidence_sufficient": "yes" or "no", "reason": "<at most three sentences>", "evidence": [<passage ids>]}
```

- **differential:** three different diagnoses from the list, the most likely first. The first one is the model's diagnosis.
- **evidence_sufficient:** "yes" if what the model was given is enough to be confident of its diagnosis, otherwise "no". This is how a model can say it does not know, without leaving the question unanswered.
- **reason:** a short justification. It is saved but not scored.
- **evidence:** the ids of the passages the reason relies on. Closed-book, it must be empty.

**Reading (strict).** The reader (`parse_json_answer` in `src/prompts.py`) forgives only spaces around the object, a Markdown code block around it, and the case and spacing of a diagnosis name or of yes/no. Everything else makes the answer **invalid**: text before or after the object, a missing or added key, a differential that is not three different diagnoses of the list, a flag that is not yes or no, an empty reason, and evidence that names a passage the run did not give (closed-book: any evidence). An answer cut off at the output limit is invalid; the report says how many were cut off, and if any were, the results are not valid until the limit is raised under a new model entry.

**An invalid answer counts as wrong** in every score, and the share of invalid answers is reported on its own. The reason each answer was invalid is saved in `predictions.jsonl` and counted in the report.

## 3. The scores

Every score is computed by `src/metrics.py` from the saved answers. The classes are weighted equally: the text dataset has unequal numbers of cases per class, and a model can reach a high plain accuracy by naming the most common class.

| Score | Definition | Source |
|---|---|---|
| **Balanced accuracy (main score)** | The mean recall over the classes present in the test split. Recall of a class: of its questions, the share whose diagnosis is that class. Chance is 1 / 8 = 12.5%. | Brodersen et al. 2010 |
| **Macro-F1** (second score) | The mean F1 over the classes present. F1 of a class: the harmonic mean of its precision and recall. A class never given has no precision and counts as 0. | Sokolova & Lapalme 2009 |
| Per-class recall, precision and F1; confusion matrix | In each run's report. The confusion matrix has one more column for invalid answers. | Sokolova & Lapalme 2009 |
| **Top-3 balanced accuracy** | The mean over the classes of the share of questions whose true class is anywhere in the differential. It shows whether the right diagnosis was considered at all. | Liu et al. 2020 (top-3 accuracy of a skin-disease differential) |
| **Invalid-answer rate** | The share of questions with an invalid answer. | – |
| **Misdiagnosis detection: sensitivity** | Over the valid answers. The positive case is a wrong diagnosis. Of the wrong diagnoses, the share the model marked with "evidence_sufficient": "no". | Hendrycks & Gimpel 2017 (detecting misclassified examples) |
| **Misdiagnosis detection: specificity** | Of the right diagnoses, the share the model did not mark. | Hendrycks & Gimpel 2017 |
| **Coverage** | Of all questions, the share answered with "yes". An invalid answer is not covered. | Geifman & El-Yaniv 2017 |
| **Selective risk** | Of the covered questions, the share with a wrong diagnosis: the error rate when the model's own "yes" is trusted. | Geifman & El-Yaniv 2017 |

Plain accuracy is not reported: it adds nothing to balanced accuracy where the classes are equal in size, and it misleads where they are not. A ROC curve and its AUC are not reported: they need a score for every class on every question, and the answers give a ranked list of three, not scores. The evidence flag gives one operating point (the sensitivity and specificity above), not a curve.

**Hallucination.** In this thesis a hallucination is a confident wrong diagnosis: a wrong diagnosis marked "yes". Its share among the covered questions is the selective risk. A method reduces hallucination if it lowers the selective risk without lowering the balanced accuracy, or if it raises the misdiagnosis detection sensitivity at a similar specificity.

## 4. Intervals and comparisons

**95% intervals.** A percentile bootstrap (Efron & Tibshirani 1993) with 2,000 resamples and a fixed seed (0). Each resample draws whole groups with replacement inside each class, so every class keeps its number of groups. The group is the lesion (image, image + metadata: several photos of one lesion are not independent), the article (text: several patients of one article are not independent) or the case (cases); resampling whole groups follows Field & Welsh 2007. When every question is its own group, a share (one class's recall, coverage) gets a Wilson interval instead (Wilson 1927).

**Comparing two runs on the same questions** (two models, or one model with and without a method):

- **Main comparison:** the difference in balanced accuracy, with a paired 95% interval: both runs are resampled together, with the same groups, 2,000 times. A difference whose interval excludes 0 is called beyond chance. The difference in macro-F1 is given the same way.
- **Supporting test:** McNemar's exact test on the questions exactly one of the two runs answers right (McNemar 1947). Where a page holds several tests, each p-value is also given after the Holm correction for their number (Holm 1979).

The intervals are not corrected for the number of comparisons, so on a page of many comparisons one interval can exclude 0 by chance. Where the interval excludes 0 but the Holm-corrected p-value is above 0.05, the results page says so and does not call the difference established.

## 5. Analyses beside the main score

Each is reported beside the main score, never instead of it.

- **Inside each photo collection** (image, image + metadata). Melanoma and nevus photos come from one ISIC collection and the six other classes from another, so the look of the collection alone could separate the two groups. Inside each group, balanced accuracy is computed with an answer outside the group counted as wrong, next to the share of answers that stay in the group and the chance level of the group (1/2 and 1/6). A score near chance inside a group means the model told the collections apart, not the diseases.
- **Without actinic keratosis** (text). The text test split has only 2 actinic keratosis cases, which weigh one eighth of the balanced accuracy. The scores are also given without them. The answer options stay the same.
- **Questions with all facts recorded** (image + metadata). A missing fact points to the photo's collection, so the scores are also given for the questions with every fact recorded.

## 6. The first-phase runs

The first-phase runs (`results/phase1/`) asked for one diagnosis name alone, read with a strict class-name reader. Their archived `report.md` and `metrics.json` are kept unchanged. The current scores of those answers are written beside them in `scores.md` and `scores.json` (`python -m src.evaluate report <run folder>`). Balanced accuracy, macro-F1, the per-class scores, the confusion matrix and the collection analysis are computed as above. Top-3 balanced accuracy and the misdiagnosis-detection scores need a differential and a flag, so they are not available for those runs. Their reports also give a sensitivity analysis that reads a diagnosis name alone on the last line of an otherwise invalid answer.

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
