# Status of the research

Updated 2026-11-22. Done = present and checked; in progress = started, not finished; pending = not started. Every item names its files.

## Where things are
- Data: `data/datasheet.md` (the pipeline, every folder and file, the three views, the limitations).
- Code: `src/README.md` (every script, its order, inputs, outputs and command).
- Results: `results/feature_check/` (the data analysis); `results/llm_only/<run>/<view>_<split>/` (LLM-only runs, none yet).
- Checks: `tests/` (`python -m unittest discover -s tests`).
The sections below are dated; older ones use the names of their day (renamed on 2026-11-22, see the last dated section).

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

## Clear names, the LLM-only run script, fewer metrics (2026-11-22)
1. Renamed with `git mv` (history kept): `raw/` → `kaggle_original/` (laptop), `meta/` → `official_labels/`, `images/` →
   `clean_images/` (laptop), `processed/` → `clean_cases/` (`master_clean.csv` → `all_cases.csv`, `original_files_audit.csv` →
   `kaggle_files_audit.csv`, `near_duplicate_pairs.csv` → `near_duplicates.csv`, `label_map.json` → `classes.json`,
   `class_balance.json` → `class_weights.json`, `train_*_index.csv` → `train_*.csv`), `views/<view>/qa_<split>.jsonl` →
   `questions_<split>.jsonl`, `corpus/` → `knowledge_corpus/`. Image paths now read `clean_images/<id>.jpg`; all 6,591 questions
   are otherwise unchanged (checked one by one).
2. `src/run_llm_only.py`: the LLM-only baseline for OpenAI, OpenAI-compatible servers (e.g. Llama hosts), Anthropic and Gemini;
   one prompt for every view and model; the answer letter read with one fixed rule; each answer saved at once; a rerun continues
   where it stopped; a run folder ends in `.partial` until complete; keys only from environment variables. Not run yet.
3. `src/evaluate.py`: options renamed `--questions`, `--predictions`, `--predictions-b`; per class "sensitivity" is now called
   recall (the same number); the separate melanoma sensitivity is removed (it is the melanoma row of the per-class table);
   BERTScore is removed (it repeated semantic similarity).
4. `src/preprocess.py`: reads only the unpacked Kaggle layout (`Train/`, `Test/`); the zip-part layout is removed.
Checked: 94 tests; 13 new deliberate breaks of the run script, each caught (and the 39 earlier ones still caught).
Not checked: the real APIs (no keys or internet here) — the request formats follow the providers' documentation.

## Preferred names, the four research models (2026-11-22)
1. Renamed (with `git mv`): `kaggle_original/` → `raw/` (laptop), `clean_images/` → `processed_images/` (laptop), `clean_cases/` →
   `processed_cases/`, `kaggle_files_audit.csv` → `raw_files_audit.csv`, `data/DATA_CARD.md` → `data/datasheet.md`. Image paths now
   read `processed_images/<id>.jpg`; the questions are otherwise unchanged (checked one by one); the text-only files did not change.
2. `src/run_llm_only.py` runs the four research models with fixed settings (taken from the Version 2 model settings in the archive):
   Claude Sonnet 5.5 through Anthropic's official package (no temperature: the model rejects one; hidden reasoning off), GPT-4o,
   Gemini 3.8 Flash and Llama 4 Maverick through OpenRouter, each pinned to one host with no fallback, temperature 0, seed 0.
   Keys are read from `.env` (CLAUDE_API_KEY, OPENROUTER_API_KEY). A question refused by a host's content moderation is saved as
   invalid and the run goes on. The models are used as they are (no fine-tuning), as in RAG².
3. The training-only files (class weights, balanced lists, normalisation, folds, model features) stay: model training is still part
   of the research.
Checked: 98 tests; 17 deliberate breaks of the run script, each caught. Not checked: the real APIs (no keys or internet here).

## Top-3 accuracy, selective classification, Holm (approved 2026-11-22)
1. The answer format of the LLM-only runs (prompt `llm_only_v2`): a short explanation, then "Answer: X, Y, Z" (the three most
   likely options, most likely first) and "Evidence sufficient: yes/no". The first letter is the diagnosis.
2. `src/evaluate.py` adds top-3 accuracy (Liu et al. 2020), coverage and selective risk (Geifman & El-Yaniv 2017): coverage is
   the share of questions answered with a valid diagnosis marked "yes"; selective risk is the share of those that are wrong
   (confident misdiagnoses). Each has a 95% interval over lesions except selective risk (a share of a subset).
3. `src/compare.py` compares several runs on the same cases: McNemar for every pair, with Holm's correction (Holm 1979).
Checked: 107 tests; 12 deliberate breaks of the new metrics and the answer rule, each caught. Sources as cited in the archived
Version 2 method (`archive/v2/docs/method.md`), not re-checked now.

## Question files renamed; semantic similarity completed (2026-11-22)
1. `data/views/<view>/questions_<split>.jsonl` → `data/views/<view>/<split>.jsonl` (`train`, `val`, `test`), with `git mv`.
2. Semantic similarity (`src/evaluate.py`): the cosine between the embedding of each model explanation and the embedding of a cited
   reference text of the true class (sentence encoder `pritamdeka/S-PubMedBert-MS-MARCO`), with its mean, a 95% interval over
   lesions and the mean per class; a class without a reference stops the evaluation; a reference may carry its source. It applies
   to the LLM-only runs too (they save an explanation). Context relevance (Jaccard, MedCPT cosine, MedCPT cross-encoder) and
   retrieval quality (Hit@k, Precision@k, nDCG@k, MRR, context precision) were already in place and tested.
Checked: 111 tests; 5 deliberate breaks of semantic similarity, each caught. Still missing: the reference texts themselves.

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
- The model(s) for the LLM-only baseline (approval needed: it costs credit), then a small trial on validation, then the runs.
- The other baselines (standard RAG, RAG², proposed) on the three views.
- Tests for `src/feature_check.py` (the patient text, the options, the raw layout and the audit are tested in `tests/test_scripts.py`; the evaluation code in `tests/test_evaluate.py`; the corpus builder in `tests/test_corpus.py`).
- A first run of the MedCPT relevance scorers on the laptop (`--relevance-models`).
