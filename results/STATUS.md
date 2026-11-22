# Status of the research

Updated 2026-11-22. Done = present and checked; in progress = started, not finished; pending = not started. Every item names its files.

## Where things are

| Folder | Holds | In Git |
|---|---|---|
| `data/raw/` | the Kaggle download "Skin Cancer ISIC" (2,357 files), untouched | no (laptop only) |
| `data/images/` | the 2,197 cleaned images (600 × 450) | no (laptop only) |
| `data/meta/` | the official label files the cleaning used (ISIC 2019 ground truth and metadata, HAM10000 metadata, ISIC Archive records) | yes |
| `data/processed/` | step 1: `master_clean.csv` (every kept case with its split), the audit and near-duplicate tables, label map, normalisation, class weights, balanced lists | yes |
| `data/views/` | step 2: `image_only/`, `text_only/`, `multimodal/`, each with `cases.csv` and `qa_{train,val,test}.jsonl` | yes |
| `data/corpus/` | the knowledge corpus (not built yet); only its manifest and card go to Git | partly |
| `data/DATA_CARD.md` | the data card: pipeline, files, views, limitations | yes |
| `src/` | `common.py` (shared), `preprocess.py` (step 1), `build_views.py` (step 2), `build_knowledge_corpus.py`, `feature_check.py`, `evaluate.py`; guide in `src/README.md` | yes |
| `results/` | run outputs; now the feature check (`results/feature_check/`) and this page | yes |
| `tests/` | `test_data.py` (the tables and questions agree), `test_scripts.py` (parts of the scripts), `test_evaluate.py` (the evaluation code), `test_corpus.py` (the corpus builder), `test_workspace.py` and `test_archive.py` (structure and archive) | yes |

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

## Knowledge corpus builder rewritten (approved 2026-11-22, step 4; `src/build_knowledge_corpus.py`, `tests/test_corpus.py`)
Four libraries as in RAG² (approved list): PubMed abstracts; open-access PMC papers (licence from the PMC OA service, text from
BioC); guidelines (open-access guideline and consensus papers in PMC, and the public Meditron guideline collection without WikiDoc
and NICE); Dermoscopedia (CC BY-NC-SA 4.0). Rules: no AI or image-dataset papers, no retracted papers, only CC licences for PMC,
passages of whole sentences (at most 200 words, one sentence overlap), no duplicates, at most 150 passages per library and
diagnosis, disease tags from names and synonyms, a keyword check of four topics per diagnosis. No e-mail address is sent; the
optional NCBI API key comes from the environment variable `NCBI_API_KEY` and is never stored. Output in `data/corpus/`:
`passages.jsonl` and the download cache stay on the laptop; `manifest.csv` and `corpus_card.json` (no text) are committed.
Checked: 21 tests on made-up answers in the services' formats, and 15 deliberate breaks each caught by a test.
Not checked: the real services (no internet here) — the PubMed, PMC OA, BioC and Dermoscopedia answers and the Meditron
collection's columns are written from their documentation; the first run on the laptop shows the counts per library.

## Repository audit and reorganisation (2026-11-22)
1. `data/` follows the pipeline: `raw/` and `meta/` (inputs) → `images/` and `processed/` (step 1) → `views/<view>/` (step 2);
   the files were moved with `git mv`, so their history follows them. The question files are unchanged byte for byte.
2. `src/build_qa.py` became `src/build_views.py` (Python only); it writes the view tables too. The view tables now hold only the
   inputs of their view (the 11 model features stay in `processed/master_clean.csv`, where the feature check reads them).
3. `src/common.py` holds what the scripts shared or repeated: the classes and their names, the folder layout, safe writing (a file
   is replaced in one step, so an interrupted run never leaves half a file), reading, logging.
4. `src/preprocess.py`: writes to `processed/`, checks its inputs first, logs to `processed/preprocess.log`; the unused
   `--ham-images` option is removed (only the Kaggle dataset is used). Its cleaning logic is unchanged.
5. `src/evaluate.py`: each report records its input files with their SHA-256, how often each letter was chosen, cost and time when
   the predictions give them, and accuracy per subgroup (`--cases ... --by source facts_recorded`).
6. Removed: the empty `.gitkeep` files. Ignored: `.ruff_cache/`, logs, half-written files.
Checked: 76 tests; 39 deliberate breaks (12 new, 15 corpus, 12 evaluation), each caught. Not checked here: `preprocess.py` and
`feature_check.py` end to end (they need the images and packages of the laptop). A rerun of step 1 on the laptop should leave
`git status` without changes in `data/processed/`.

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
- The first run of the corpus builder on the laptop, then a check of its card (counts, removed, empty topics) before freezing.
- A cited reference text per class (needed for semantic similarity).
- The four baselines (LLM only, standard RAG, RAG², proposed) on the three views; then `src/evaluate.py` (tested on made-up files and the committed questions; never run on real predictions).
- Tests for `src/feature_check.py` (the patient text, the options, the raw layout and the audit are tested in `tests/test_scripts.py`; the evaluation code in `tests/test_evaluate.py`; the corpus builder in `tests/test_corpus.py`).
- A first run of the MedCPT relevance scorers on the laptop (`--relevance-models`).
