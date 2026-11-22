# Status of the research

Updated 2026-11-22. Done = present and checked; in progress = started, not finished; pending = not started. Every item names its files.

## Where things are

| Folder | Holds | In Git |
|---|---|---|
| `data/raw/` | the Kaggle download "Skin Cancer ISIC" (2,357 files), untouched | no (laptop only) |
| `data/images/` | the 2,197 cleaned images (600 × 450) | no (laptop only) |
| `data/meta/` | the official label files the cleaning used (ISIC 2019 ground truth and metadata, HAM10000 metadata, ISIC Archive records) | yes |
| `data/qa/` | the multiple-choice question files: 3 views × 3 splits | yes |
| `data/` (tables) | `master_clean.csv` (every case), the three view tables, the audit and near-duplicate tables, label map, normalisation, class weights, balanced lists, `DATA_CARD.md` | yes |
| `src/` | the scripts: `preprocess.py`, `build_qa.py`, `feature_check.py`, `build_knowledge_corpus.py`, `evaluate.py` | yes |
| `results/` | run outputs; now the feature check (`results/feature_check/`) and this page | yes |
| `tests/` | `test_data.py` (the tables and questions agree), `test_scripts.py` (parts of the scripts), `test_evaluate.py` (the evaluation code), `test_workspace.py` and `test_archive.py` (structure and archive) | yes |

## Done (checked on 2026-11-22 against the tables, by `tests/test_data.py`)
- Cleaning of the Kaggle dataset: duplicates removed, labels taken from official records, 8 classes, images standardised (`src/preprocess.py`).
- Split by lesion group: train 1,568, validation 314, test 315, plus 5 folds on train and validation; no lesion group in two splits.
- Three views on the same cases (image only, text only, image + text) and their question files.
- Class weights and balanced training lists (training cases only).
- Feature and shortcut check (`src/feature_check.py`, `results/feature_check/`).

## Corrected (approved 2026-11-22, rerun on the laptop the same day, checked)
1. Corrected labels: 158 among the 2,202 ISIC numbers, 157 among the 2,197 kept images; the one between is ISIC_0010990, corrected and
   then removed as the same picture as ISIC_0000521 (checked against the Version 1 records). The data card says both.
2. `original_files_audit.csv` lists all 2,357 original files; the 6 files of the 5 removed copies say "removed: same picture as …".
3. The patient text no longer says "A dermoscopic image of the lesion is available"; the questions' text is unchanged.
4. The 8 options of each question are shuffled with a fixed seed (the same order for a case in all three views).
5. The cleaning reads the Kaggle download as unpacked in `data/raw/`.
6. Image paths are written `images/<id>.jpg` on every system (the rerun on Windows had written `images\<id>.jpg`; the committed tables
   were corrected to the value the fixed code writes).
The rerun changed nothing else: the same 2,197 cases, splits, folds, groups, labels, balanced lists, class weights and normalisation.

## Evaluation code corrected (approved 2026-11-22, step 1; `src/evaluate.py`, `tests/test_evaluate.py`)
1. An answer letter is read with the options of its own question (the options are shuffled per case). Before, the per-class results,
   macro-F1, confusion matrix, MCC and kappa treated a letter as a class, so they were wrong; plain accuracy was right.
2. A question without a prediction, or with an answer that is not an option letter, counts as wrong (before, missing ones were left out).
3. Accuracy is the main score, with a 95% interval; balanced accuracy, MCC, Cohen's kappa and macro AUC are no longer reported
   (supervisor: accuracy is enough). Kept: macro-F1, per-class precision, sensitivity, specificity and F1, melanoma sensitivity,
   confusion matrix, McNemar's exact test.
4. The 95% intervals draw whole lesions, so photos of one lesion stay together (44 test lesions have more than one photo).
5. `hit_recall@k` is renamed `hit@k` (it is the share of questions with at least one relevant passage in the top k).
6. nDCG@k no longer rewards a system that returns fewer than k passages.
7. Relevance of the passages to the query the system searched with (its saved `query`, else the question): Jaccard similarity of the
   content words (no model), and with `--relevance-models` the MedCPT cosine and cross-encoder scores. This replaces the cosine of the
   question with the passages.
8. Diagnosis and retrieval metrics need only Python (no numpy, scikit-learn or scipy); the file stops on a broken question file or on
   predictions for questions that are not in it.
Checked: 19 tests, and 12 deliberate breaks each caught by a test. Not checked: the MedCPT scorers (they download 3 models; not run yet).

## Known issues
1. Missing age, sex or site is common for melanoma and nevus only, so "not recorded" hints at the class. Decision: report the text
   results on all 315 test cases and again on the 277 with all three facts recorded.
2. Source confound: all melanoma and nevus come from the older ISIC archive, the other classes almost all from HAM10000.
3. The cleaning used outside label files (ISIC 2019, HAM10000), and the documents propose adding HAM10000 images; the supervisor has
   not yet confirmed either is allowed under "only this Kaggle dataset".
4. No clinician is available: reference texts must come from cited sources only, and no answer is rated by a dermatologist.

## Pending
- The supervisor's answers: outside label files; adding HAM10000 images; a text corpus as evidence.
- Pinning the package versions in `requirements.txt` (`pip freeze` on the laptop). Note: installing `ragas` replaced `openai` 3.49 with 1.109 in the laptop's environment; only the archived Version 2 code uses `openai`.
- The knowledge corpus (`src/build_knowledge_corpus.py`, never run).
- A cited reference text per class (needed for semantic similarity).
- The four baselines (LLM only, standard RAG, RAG², proposed) on the three views; then `src/evaluate.py` (tested on made-up files and the committed questions; never run on real predictions).
- Tests for `src/feature_check.py` and `src/build_knowledge_corpus.py` (the patient text, the options, the raw layout and the audit are tested in `tests/test_scripts.py`; the evaluation code in `tests/test_evaluate.py`).
- A first run of the MedCPT relevance scorers on the laptop (`--relevance-models`).
