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
| `tests/` | `test_data.py` (the tables and questions agree), `test_workspace.py` and `test_archive.py` (structure and archive) | yes |

## Done (checked on 2026-11-22 against the tables, by `tests/test_data.py`)
- Cleaning of the Kaggle dataset: duplicates removed, labels taken from official records, 8 classes, images standardised (`src/preprocess.py`).
- Split by lesion group: train 1,568, validation 314, test 315, plus 5 folds on train and validation; no lesion group in two splits.
- Three views on the same cases (image only, text only, image + text) and their question files.
- Class weights and balanced training lists (training cases only).
- Feature and shortcut check (`src/feature_check.py`, `results/feature_check/`).

## In progress
- Moving the files into this layout and committing them (2026-11-22).

## Known issues (corrections proposed, waiting for approval)
1. The documents say 158 corrected labels; `master_clean.csv` marks 157.
2. `original_files_audit.csv` lists 2,351 of the 2,357 original files; the 6 files of the 5 removed same-picture copies are missing.
3. The text-only table ends every text with "A dermoscopic image of the lesion is available." (the text-only question files do not). A test marks this as a known failure.
4. Missing age, sex or site is common for melanoma and nevus only, so "not recorded" hints at the class.
5. Source confound: all melanoma and nevus come from the older ISIC archive, the other classes almost all from HAM10000.
6. The answer options are always in the same order (A is always melanoma).
7. The cleaning used outside label files (ISIC 2019, HAM10000); the supervisor has not yet confirmed this is allowed.

## Pending
- The supervisor's answers: outside label files; adding HAM10000 images; a text corpus as evidence.
- Pinning the package versions in `requirements.txt`.
- The knowledge corpus (`src/build_knowledge_corpus.py`, never run).
- A cited reference text per class (needed for semantic similarity).
- The four baselines (LLM only, standard RAG, RAG², proposed) on the three views; then `src/evaluate.py` (never run).
- Tests for the scripts in `src/`.
