# Status of the research

Updated 2026-11-22. Done = present and checked; in progress = started, not finished; pending = not started. Every item names its files.

## Where things are
- Data: `data/datasheet.md` (the pipeline, every folder and file, the three views, the limitations).
- Code: `src/methodology.md` (the six files and their commands, the four test files, the old names).
- Results: `results/feature_check/` (the data analysis); `results/llm_only/<run>/<view>_<split>/` (LLM-only runs, none yet).
- Checks: `tests/` (`python -m unittest discover -s tests`).
The sections below are dated; older ones use the names of their day (files and tests renamed on 2026-11-22: see the table in `src/methodology.md`).

## Done (checked on 2026-11-22 against the tables, by `tests/test_system.py`)
- Cleaning of the Kaggle dataset: duplicates removed, labels taken from official records, 8 classes, images standardised (`src/preprocess.py`).
- Split by lesion group: train 1,568, validation 314, test 315, plus 5 folds on train and validation; no lesion group in two splits.
  (Replaced on 2026-11-22 by the balanced test split: train 1,643, validation 314, test 240; see the dated section.)
- Three views on the same cases (image only, text only, image + text) and their question files.
- Class weights and balanced training lists (training cases only).
- Feature and shortcut check (`src/check.py features`, `results/feature_check/`; made on the old split, rerun if reported).

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

## Reference texts for semantic similarity (approved 2026-11-22, option A)
`src/build_reference_texts.py` quotes one paragraph per disease from Dermoscopedia (CC BY-NC-SA 4.0) by a fixed rule: the page whose
title equals a search term (else the first result), its first section whose heading mentions dermoscopy (else the introduction),
whole sentences up to 200 words. Output `data/knowledge_corpus/reference_texts.json` with source, URL, section, rule, licence and
date. Checked: 6 tests, 6 deliberate breaks caught. Not run yet (needs internet): the 8 texts must be checked once after the run.

## Pipeline checker; reference-text rule corrected in part (2026-11-22)
1. `src/check_pipeline.py` reads every step's output and reports it as done, not run or a problem (incomplete, out of date, an
   unfinished run, a run on changed questions, a run without its report), with the reason. It changes nothing.
2. The first run of `build_reference_texts.py` on the laptop gave introductions (chapter menus and wiki template lines) for 6 of 8
   diseases and the lichen planus-like keratosis page for BKL: not usable, not committed. Fixed: headings cleaned (entities,
   footnote marks), template lines dropped. Added `--inspect` to see each disease's pages and sections before the rule is set.
3. `data/datasheet.md`: licences and research ethics of every part of `data/`.
Checked: 6 new tests of the checker.
4. Reference-text rule set from the `--inspect` output (laptop, 2026-11-22): the first search term with results decides the page
   (an equal title wins within it); a chapter page without sections is replaced by its first subchapter (melanoma, vascular
   lesions); the text starts at the dermoscopy heading, else at the first section (pages whose sections are the criteria: BCC,
   AK, SCC), else the introduction; NV uses the site's general nevus-pattern page (the page "Nevus" is about nail nevi).
   Checked: 9 tests on a made-up wiki with the site's layouts; 7 deliberate breaks caught. The checker now flags only texts under
   50 words. Test output is quiet (the screen log follows a redirected output).

## Reproducible tables on Windows; left-over images named (2026-11-22)
- The laptop rerun of `preprocess.py` showed the 5 tables written by pandas as changed (`all_cases.csv`, `raw_files_audit.csv`,
  `near_duplicates.csv`, `train_oversampled.csv`, `train_undersampled.csv`), while the JSON files and the views were unchanged. Likely
  cause ◻: pandas on Windows ends lines with CRLF. `save_csv` now always writes LF (test in `tests/test_scripts.py`; it runs where
  pandas is installed). To confirm on the laptop: rerun, then `git status` shows no change.
- `check_pipeline.py` now names the missing images and the images in `data/processed_images/` that are not in the case table;
  `preprocess.py` logs such left-over images at the end (it does not delete them). The shared rule is `images_not_in_table` in
  `src/common.py` (tests in `tests/test_check_pipeline.py`; 5 deliberate breaks, all caught).

## Reference texts: one text per dermoscopy section (approved 2026-11-22, option B)
- Review of the first file (8 texts, the first 200 words of each page): DF, SCC, VASC good; AK acceptable; BKL partial (seborrheic
  keratosis only); BCC, MEL and NV weak (BCC: pigmented nests only, no arborizing vessels; MEL: half a general statement, its list
  of criteria missing ◻; NV: congenital globular nevi only, no reticular pattern).
- `src/build_reference_texts.py` now keeps every section of the chosen part of the page, each split into texts of whole sentences
  of at most 200 words; texts under 20 words are left out and listed under "skipped"; sections on references, literature, see
  also, external links, further reading and differential diagnosis are left out; a section is read without its subsections (no
  text twice); "(Figure )" and ". ." leftovers are removed. The page choice (steps 1-2) is unchanged.
- `src/evaluate.py`: an explanation is compared with every reference text of its true class and the highest cosine is kept; the
  report gives the number of texts per class and counts texts longer than the model reads.
- Word limit: kept at 200. Raising it would only make each text longer, not add content (every section is now kept whole, in
  pieces), and the model ignores text past its token limit. That limit could not be read here (Hugging Face is blocked) ◻:
  `--token-check` measures it on the laptop.
- Not yet: BKL still lacks solar lentigo and lichen planus-like keratosis (separate pages; a limitation, not hand-picked). The
  committed `reference_texts.json` has the old form until it is rebuilt (the checker says so).
- Tests: `tests/test_reference_texts.py`, `tests/test_evaluate.py`, `tests/test_check_pipeline.py`; 25 deliberate breaks on the final code, all caught (2 missed at first; the tests were fixed).

## Reference texts rebuilt; reference lists removed (2026-11-22)
- Laptop rebuild: 41 texts (MEL 2, NV 10, BCC 8, AK 5, BKL 8, DF 5, VASC 1, SCC 2). BCC now has leaf-like, spoke-wheel, arborizing
  vessels, ulceration and shiny white structures; NV has globular, reticular, starburst, blue and site-related nevi; AK has
  pigmented AK.
- Token check ✔ (laptop output): the similarity model reads at most 350 tokens. 200 words gave up to 294 tokens of plain text, so
  the word limit stays at 200 (250 words would exceed 350 tokens ◻, about 1.5 tokens per word). The two texts over the limit (AK
  413, SCC 393 tokens) were reference lists written as plain text at the end of a section.
- Fix: `without_bibliography` cuts a section at the first sentence that starts with an author ("Kittler, H., ...") or names a book
  edition; the file lists what was cut ("bibliography_removed"). On the committed texts it cuts only those two lists (AK, SCC); the
  other 39 texts are unchanged; a citation inside a sentence is kept. Tests; 8 deliberate breaks, all caught.
- Open: MEL's two lists of criteria are still missing ("... are the following:" with nothing after), likely in a table, which
  the parser skips ◻. Some texts are not dermoscopy (NV: histopathology, "unclassifiable lesions"; BKL: the wobble test).
- Note for the thesis: with the best-matching reference kept, a class with more texts tends to get higher scores (NV 10 texts,
  VASC 1), so per-class similarity is not comparable between classes; comparisons between systems are fair (same references).

## Decisions on the reference texts (approved 2026-11-22)
1. MEL's missing lists: `--show-tables` prints the text of the tables in the sections the rule reads (from the laptop's cache; it
   writes nothing). If the criteria are there, table text will be allowed for every class by a fixed rule (not for MEL alone).
   Tests; 7 deliberate breaks, 6 caught; the 7th changes no output (the buffer is cleared at each table's start).
2. Texts that are not dermoscopy (NV histopathology and "unclassifiable lesions", BKL wobble test) stay: removing them would be
   hand-picking, and with the best match kept they rarely decide a score.

## MEL's lists: not in tables; a look at the other skipped text (approved 2026-11-22, option A)
- `--show-tables` on the laptop: the only tables in the chosen sections are the page information boxes of MEL and VASC (author,
  status, date), so no table rule is added.
- `--show-skipped` prints the text outside paragraphs, list items and tables (captions, definition lists, divs, image alt text)
  in the chosen sections; it writes nothing. If melanoma's criteria are there, one fixed rule for every class will be proposed.
  Tests; 11 deliberate breaks, all caught (2 missed at first; tests added).
- Laptop output, ✔ checked here: the committed file has 40 texts (MEL 2, NV 10, BCC 8, AK 4, BKL 8, DF 5, VASC 1, SCC 2); the
  longest has 294 tokens, the model reads 350.

## MEL's lists are images; no new rule (2026-11-22)
- `--show-skipped` on the laptop ✔: MEL's list of criteria is an image ("Dermoscopic features associated with melanoma.jpeg"),
  not text, so it cannot be quoted. The other skipped text is site furniture (app advert, ratings, author lines, "References"
  headings), image file names, and a few short image captions (BCC, AK) that repeat what the texts already say.
- Decision (recommended, no rule change): no caption or image text is added; the reference texts stay as built (40 texts).
- Limitation for the thesis: MEL has 2 texts from the superficial spreading melanoma introduction; they name asymmetry of shape
  and structures, more than 2 colours, atypical network and regression, but not the full list of local criteria.

## LLM-only: prompt accepted, first paid trial, fixed asking order (2026-11-22)
- Dry run on the laptop ✔: the three views give the right inputs; the same option order for a case in every view; images found.
  The prompt (`llm_only_v2`) was accepted.
- Paid trial 1 (approved): 5 validation questions x 4 models x 3 views = 60 calls. All 60 answered and read (valid), no refusal,
  no cut-off; each served by the pinned host; 2-8 s per question. Image questions used more input tokens than text-only ones
  (GPT-4o 640 vs 222, Claude 700 vs 332), as expected for a 600x450 image ◻.
- Flaw: the question files are sorted by class, so all 5 trial questions were AK. 5 cases of one class say nothing about
  accuracy. GPT-4o (image only) answered BKL / NV / DF for all five images, Llama BKL / NV ◻ (to watch in the full runs).
- Fix (approved, option A): `run_llm_only.py` asks the questions in a fixed shuffled order (sha256 of "42:<case id>"), the same
  for every model, view and split; the files and the scores are unchanged; `run.json` records the order. The 60 answers stay
  valid. The next 5 validation questions are NV, NV, BCC, BCC, MEL. Tests; 6 deliberate breaks, all caught (removing the
  order note from only one of its two writes changes nothing: the end-of-run write replaces the first).

## Refusals reported apart from other invalid answers (approved 2026-11-22, option A)
- Trial 2 (5 more validation questions x 4 models x 3 views, mixed classes): 60 answers, 59 valid. The invalid one is a refusal:
  GPT-4o, image only, ISIC_0010866 (MEL): "I'm sorry, I can't help with identifying or diagnosing images of skin lesions."
  It ended normally ("stop"), so the host did not flag it.
- `evaluate.py` now splits the invalid answers into refusals and other invalid answers (`invalid_answers_by_cause`, with the
  ids). A refusal: the host's refusal flag, or a fixed phrase (REFUSAL: "I'm sorry", "I can't / cannot / won't / am unable to /
  am not able to" + help, assist, provide, identify, diagnose, answer, comply, interpret, analyse, evaluate). Only invalid answers
  are checked; the scores do not change (a refusal is wrong, as before); refusals are not retried. Tests; 11 deliberate breaks,
  all caught (1 missed at first; test added; the test also caught a fault in the first pattern: "I’m unable to").

## Balanced test split (decided 2026-11-22)
- Decision: the test split holds 30 images of every class (240 images, 190 lesions). Rule (`src/balance_test.py`, used by
  `preprocess.py` step 8): whole lesion groups only, in a fixed order (sha256 of "42:<group_id>"); a class keeps its test groups
  that fit, the rest move to train; a class with too few takes train groups. Validation is not changed.
- Effect ✔ (the rule on the committed table): test 240 (30 each); train 1,568 -> 1,643; val 314 (the same cases). Moved: 111
  test -> train (MEL 35, BKL 38, BCC 26, NV 12), 36 train -> test (DF 14, AK 12, VASC 9, SCC 1). No lesion in two splits.
- Consequences: the training tables (class weights, balanced lists, normalisation, CV folds) change with the rerun; the
  validation questions stay the same, so the 120 trial answers stay valid ◻ (to confirm with `check_pipeline.py` after the rerun);
  `results/feature_check/` was made on the old split (an analysis, not an experiment; rerun if it is reported).
- `check_pipeline.py` reports a test split that is not balanced. Tests (`tests/test_balance_test.py`, `tests/test_check_pipeline.py`);
  12 deliberate breaks, 11 caught; the 12th removed an early exit that changed nothing, so those lines were removed.
- Not yet: the laptop rerun of `preprocess.py` and `build_views.py`, and the commit of the new tables.
- Other decisions the same day: nothing is frozen yet; the repository stays private until publication.

## Code and tests consolidated (researcher's request, 2026-11-22)
- `src/` is six files: `common.py`, `preprocess.py`, `build.py` (views, corpus, references), `check.py` (pipeline status, features),
  `run.py` (llm-only), `evaluate.py` (report, compare). The balanced-test rule moved into `common.py`. The old names and where each
  went are in `src/methodology.md`. Nothing was changed in the rules; only moved, renamed and reconnected (the commands are now
  `python src/build.py views`, `python src/check.py`, `python src/run.py llm-only ...`, `python src/evaluate.py compare ...`).
- `tests/` is four files: `test_unit.py`, `test_integration.py`, `test_system.py`, `test_acceptance.py` (what each level checks is in
  `src/methodology.md`). All 150 earlier tests are kept (none dropped), plus 22 new ones: 3 for the commands (`build.py`, `check.py`,
  `evaluate.py`, `run.py` hand the right arguments to the right command), 2 for the command line (every command answers `--help`;
  only the six files exist), 2 for the committed folder (no data step has a problem; a free dry run of `run.py` needs no key) and
  15 acceptance tests of the researcher's requirements.
- Deliberate breaks: 26 on the moved code and the new commands; 4 were missed at first (the arguments handed to a command were hidden
  by `--help`; the default seed of the balanced split was not covered), tests added, all caught. 7 breaks of the acceptance tests (a
  moved test case, a changed licence, an image in the text view, a renamed question file on a copy of the data; a key-like string,
  an e-mail address, a Git tag): all caught.
- Also: the one-line summary of `evaluate.py` no longer prints "melanoma recall" (the separate melanoma figure was removed on
  2026-11-22, per-class recall stays).

## Corpus build failed on the laptop; top-k accuracy removed (2026-11-22)
- The first `python src/build.py corpus` run stopped with HTTP 404 on the PMC licence lookup (the PMC "OA web service", `oa.fcgi`).
  NLM announced in its 2026 bulletin that this service will no longer be available ✔ (jf26 "Changes to PMC article dataset
  distribution services"). The licence of a PMC paper is now read from the paper's own full-text XML (E-utilities efetch db=pmc:
  the `<license>` link to creativecommons.org); a retraction notice or a paper with a related "retract..." article is flagged
  retracted. A lookup that fails is counted ("pmc: licence lookup failed") and the build goes on, so one paper cannot stop it.
  Untested against the live service (no network here) ◻: tested with made-up JATS XML. The efetch answers are cached (about
  100-500 KB per paper ◻, in the ignored cache folder). 10 deliberate breaks, all caught.
- Top-k accuracy removed from `evaluate.py` (the researcher's decision): accuracy is top-1 accuracy. The prompt still asks for three
  letters and the answer is still parsed that way; whether to ask for one letter (a new prompt version, `llm_only_v3`) is open.
- The code guide `src/README.md` was renamed `src/methodology.md` by the researcher; the references were updated.

## Corpus: keys from .env, the Meditron download made once (2026-11-22)
- Laptop run 2 of the trial: the licence fix worked (the run got to the Meditron guideline download), then stopped on read timeouts
  of the 1 file download from Hugging Face (`open_guidelines.jsonl`; the researcher interrupted it). The warning "unauthenticated
  requests to the HF Hub" showed that the HF_TOKEN in the researcher's `.env` was not used: only `run.py` read `.env`; `build.py`
  did not. A token is optional for this public collection (it gives faster downloads and higher limits).
- Changes: `build.py corpus` now reads `.env` (`--env-file`, default `.env`; a variable already set in the terminal wins) for
  `NCBI_API_KEY` and `HF_TOKEN`, and logs only "found" or "not set", never the value (the path of the key file is not recorded in
  the corpus card either). `load_env` moved to `common.py` (shared with `run.py`). The Meditron pass keeps the skin-cancer articles
  only, saves them once after a complete pass (`<cache>/meditron_skin_cancer.json`; an interrupted pass saves nothing) and reads
  them from there afterwards. The Hugging Face read timeout is 60 s (default 10 s) and the Windows symlink warning is off.
  `--no-meditron` skips the collection (PMC guideline papers only), recorded in the card's settings.
- Tests: 5 new; 14 deliberate breaks of this code, all caught (3 missed at first; tests strengthened).
- The names in the researcher's `.env` (`CLAUDE_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`) are the ones the code reads ✔ (names
  only were shown to me; no value was seen). Hardening: an empty variable in the terminal no longer hides the `.env` value; an
  unusable `.env` line (no "=", no name, no value) is reported by its line number only, never its content, by `build.py corpus`
  and `run.py llm-only`. 6 more deliberate breaks, all caught; 178 tests.

## Balanced validation split, 20 per class (decided 2026-11-22)
- Decision: validation holds 20 images of every class (160 images, 126 lesions); train keeps the natural class mix. The rule is the
  one of the test split (`balanced_splits(..., split="val")` in `src/common.py`, constant `VAL_PER_CLASS = 20`), applied after the
  test split is balanced; the test split is not touched by it. `preprocess.py` step 8 runs both passes (`--val-per-class`, default 20).
- Effect ✔ (the rule on the committed table): train 1,643 -> 1,797, validation 314 -> 160, test 240 (unchanged); 160 validation images
  go to train and 6 train images come to validation, so 154 of today's 314 validation questions stay.
- The 12 validation trial folders (10 answers each) no longer match the new question files and are to be deleted.
- Code and tests are committed (unit test of both passes, a test that the committed cases can reach exactly 20 per class and keep the
  test split, 6 deliberate breaks, all caught). Not yet: the laptop rerun of `preprocess.py` and `build.py views`, and the commit of
  the new tables. Not tested here: the wiring of the two passes inside `preprocess.py` (pandas is not installed here); the rerun
  prints "balanced test split ..." and "balanced val split ..." lines. After the tables are committed: assertions on the committed
  validation split (acceptance and system tests), the pipeline check for an unbalanced validation split, and the data sheet table.
- Open: whether the prompt asks for one letter (`llm_only_v3`) instead of three.
- Done 2026-11-23 ✔: the researcher committed the new tables (validation 160 = 20 per class, train 1,797, test 240); the pipeline check, the system and acceptance tests and the data sheet now assert the balanced validation split.

## Knowledge corpus built on the laptop (2026-11-23) ✔ from `data/knowledge_corpus/corpus_card.json`
- 3,294 passages from 868 documents (the full run, with the Meditron guidelines: 83 passages from cdc 42, pubmed 33, cco 8). Every
  class has all four topics (appearance, patients, differential, basics): `empty_topic_boxes` is empty. Licences: CC BY and CC BY-NC
  variants (PMC), PubMed abstracts (research use; cite the PMID), Meditron sources (terms of the source). The text file
  `passages.jsonl` stays on the laptop; the manifest and the card are committed.
- Removed by the rules: WikiDoc 33,058 and NICE 1,656 Meditron articles (excluded sources), 363 + 304 + 257 + 18 passages under
  30 words, 40 + 3 AI or image-dataset papers, 1 retracted, 5 PMC papers with no open licence, 6 with no full text. The PMC licence
  lookup through the E-utilities worked at scale.
- Gaps: PubMed BCC has 0 passages (all other classes 150, VASC 74): the removal counts do not explain it (the search found no
  documents to remove), so the search result itself is empty ◻; a warning that prints NCBI's reading of the query was added
  (`esearch`). Guidelines: DF 0 and BKL 12 (few or no open guidelines for these two ◻). Textbook (Dermoscopedia): AK 11, DF 10, VASC 4,
  SCC 4 (pages are mostly captions: 363 passages under 30 words).
- PubMed BCC = 0 explained ✔ (the new warning, laptop output): NCBI reported `quotedphrasesnotfound: ["carcinoma, basal cell"[MeSH Terms]]`
  and "No items found". NLM renamed that MeSH heading for 2026 ("Basal Cell Carcinoma"; one source: NLM Classification 2026 page, not
  checked on the MeSH Browser ◻). Fix: the BCC query lists both names of the heading and the plain words as alternatives; the search
  warning now also appears when a quoted heading is not found but other results exist. 3 deliberate breaks, all caught. To redo: the
  corpus build on the laptop (cached except that search), then commit the new card and manifest.
- The repository's `src/methodology.md` rename was pushed by the researcher.

## Prompt v3: one letter, reasoning and evidence (researcher's request, 2026-11-23)
- Balanced validation split rerun on the laptop ✔ (output): validation 160 (20 per class), train 1,797, test 240 (unchanged; `test.jsonl`
  did not change). The new tables are to be committed by the researcher.
- Top-k accuracy was removed on 2026-11-22, but the prompt still asked for three letters: corrected now. The prompt (`llm_only`)
  asks for two lines: "Reasoning and evidence: <at most three sentences that name the findings of this case, in the image and/or in the
  patient facts, that support your choice>" and "Answer: <the letter of the most likely option>". The yes/no "Evidence sufficient" flag is
  gone. The prediction file no longer has `top3` or `evidence_sufficient`; the reasoning is the `rationale` field that
  `evaluate.py` already scores (semantic similarity, and the faithfulness metrics proposed).
- Consequence (open): coverage and selective risk (approved on 2026-11-21) need a yes/no flag, so LLM-only runs no longer produce them.
  The functions stay in `evaluate.py` and report only when a flag is present. Decision needed: drop both metrics, or take the support
  signal from somewhere else for the RAG systems (e.g. whether the quoted evidence is found in the retrieved passages).
- A reasoning that quotes "source text" is only possible for the RAG systems (they have retrieved passages); the LLM-only systems have none, so
  their evidence is the findings they name. The RAG prompts will add "Evidence: [passage number] "<sentence copied word for word>"" so a
  quote can be checked automatically against the passage.
- Tests: the answer rule, the instruction, the run and the dry run were updated; 9 deliberate breaks, 8 caught at first, the 9th (the
  wording of "findings ... that support your choice") covered by a new assertion.

## Answer format with the option name; prompt named `llm_only` (researcher's request, 2026-11-23)
- The answer line is now "Answer: <the letter of the most likely option> (<the name of that option>)", e.g. "Answer: C (Melanoma)" for
  the options of that question (no concrete example is put in the prompt: it would favour that class). The letter decides (a name that
  does not fit the letter is ignored by the scoring and stays in `raw_output`).
- The prompt is named `llm_only` (`PROMPT_NAME`; the field is `prompt` in `run.json` and in every answer). No older prompt text is kept
  in `src/`: `llm_only_v2` existed only as the previous text (Git history); every run saves its exact `system` and `instruction` text in
  `run.json`, so a changed prompt cannot be mistaken for this one.
- Coverage and selective risk (option B, the evidence quote as the selection signal): approved by the researcher on the condition that it
  is verified to be suitable. Verdict: it cannot be fully verified before the RAG systems exist (see the answer given on 2026-11-23);
  specified as a RAG-only design, to be checked on a free pilot; until then nothing is built or removed.

## Metric survey and option B built (researcher's request, 2026-11-23)
- Survey of RAG papers and benchmarks and what they measure: `results/metric_survey.md` (what the field measures, paper by paper,
  fit for this thesis, limits). It comes from search summaries only (the pages could not be opened here): ✔ and ◻ mark what was and was not seen.
- Option B built in `src/evaluate.py` (RAG answers only; nothing for LLM-only): a prediction may carry `"evidence": {"chunk_id", "quote"}`
  or `null` ("Evidence: none") and its `contexts`. The report gets an `evidence` section: quote accuracy (the quote is word for word in
  the cited passage: lower case, no punctuation, whole words, "..." allowed, at least 5 words), support (the cited passage entails
  "The diagnosis is <answered class>.", an NLI model, `--support-model`; default `cross-encoder/nli-deberta-v3-base`, threshold 0.5), coverage and
  selective risk with verified evidence (an answer is covered when its quote is found and supported), supported accuracy (right and
  covered, after KILT-AC), negative rejection (after RGB: "none" when no retrieved passage is tagged with the true class). The selection never
  uses the true class (a test). The old yes/no flag metrics stay in the code but need a flag, which no prompt produces now.
- `evaluate.py support-check` tests the support scorer on the corpus tags alone (no case, no label): does the statement of a passage's own
  diagnosis score higher than the statement of another one? First version (pairwise) is described below; it failed.
- **First support-check failed its pass mark (2026-11-23, laptop run, pairwise version).** 800 passages: the right statement scored higher
  in 0.83 (mean 0.0571 against 0.0011). Per class: MEL .92, NV .88, BCC .80, AK .91, BKL .51, DF .93, VASC .91, SCC .78. The pass mark
  (0.90 overall, no class under 0.80) was not met. Reading: the ranking works, but the raw entailment probabilities are tiny, so a fixed
  0.5 threshold cannot be used; the BKL statement was too narrow; BCC and SCC are confused. Change (not yet re-run): (1) the default rule
  is now "top": the passage is scored against the statements of all 8 diagnoses, and the answered diagnosis must have the single highest
  score (a tie is not support); the "threshold" rule stays as an option; (2) statements in the wording of the literature (BKL names
  seborrheic keratosis, solar lentigo and lichen planus-like keratosis; VASC names hemangioma or angioma); (3) `support-check` is now an
  8-way zero-shot classification of single-tag passages. New pass mark, set before the run: top-1 accuracy at least 0.70 overall and no
  class under 0.50 (chance 0.125). What "top" means: the passage is about the answered diagnosis, not that it entails this case's
  findings. The passage tags come from class names, so the check is partly lexical. Image claims are never verified.
  Tests: 201 pass (7 skipped, no pandas or sklearn here); 9 deliberate breaks of the new code, 7 caught at first; the two missed ones
  (the per-class floor and the 0.70 overall bar of the pass mark) now have a test each.
- Expected weakness ◻: a general NLI model may rate "Melanomas show asymmetry ..." as neutral to "The diagnosis is melanoma"; the
  support-check shows it before any RAG run. Fallbacks: AlignScore (ACL 2023), a different statement, or the quote check alone.
- Not built: the RAG prompts and runner (they will ask for `Evidence: [passage number] "<sentence copied word for word>"` or `Evidence: none`).
- Tests: 14 for the evidence code, 22 deliberate breaks of it, 21 caught at first, the 22nd (a score exactly at the threshold) by a new test; 199 tests.

## 2026-11-24: decisions on the evaluation metrics (researcher's approvals)
- **Support check failed twice, NLI support removed.** Second run (8-way top rule, 400 passages): top-1 accuracy 0.50 (pass mark 0.70, no class
  under 0.50): MEL .62, NV .16, BCC .38, AK .38, BKL .56, DF .76, VASC .60, SCC .54. Removed from the code: the NLI scorer, the statements, the
  `support-check` command and the `--support-*` options. Kept, no model needed: quote accuracy (word for word in the cited passage), coverage and
  selective risk with a verified quote, supported accuracy (right and quote verified), negative rejection. Limit to state: a real quote is not proof
  that it supports the diagnosis.
- **RAGAS removed** (and `--ragas`, and `ragas` from requirements.txt): an LLM judge costs credit and is not exactly reproducible.
- **Approved main metrics:** accuracy (top-1 only; one version), macro-F1 with per-class F1, McNemar exact with Holm; retrieval: Hit@k,
  Recall@k, Precision@k, MRR, nDCG@k. Exact Match is not reported (it equals accuracy for a letter answer).
- **Recall@k added** (`--passages data/knowledge_corpus/passages.jsonl`): the share of all the corpus passages tagged with the true class that are
  found in the top k. Low in absolute terms because the corpus is large; read it as a comparison between systems.
- **Context relevance:** option A approved (MedCPT cross-encoder, `--relevance-models`; the same quantity RAGAS and ARES call context relevance).
  Option B (similarity of the rationale to the cited passage) was not approved yet: it repeats the existing semantic similarity method and is
  high by construction when the quote is copied from the passage; recommendation is to drop it.
- **Decided 2026-11-24:** option B (similarity of the rationale to the cited passage) is dropped (it was never built); Jaccard is dropped
  (code, tests and source line removed). Context relevance is now MedCPT only (`--relevance-models`), the same quantity RAGAS and ARES call
  context relevance. Semantic similarity stays (rationale against the class reference texts). A test checks that Jaccard stays gone.
- Supervisor remark (user's report): a low LLM-only score helps to justify the proposed method. We report what we get and tune nothing to it.
- Tests: 199 pass (7 skipped); 5 deliberate breaks of the new code, all caught.

## Known issues
1. Missing age, sex or site is common for melanoma and nevus only, so "not recorded" hints at the class. Decision: report the text
   results on all test cases and again on those with all three facts recorded (counts change with the balanced test split).
2. Source confound: all melanoma and nevus come from the older ISIC archive, the other classes almost all from HAM10000.
3. The cleaning used outside label files (ISIC 2019, HAM10000), and the documents propose adding HAM10000 images; the supervisor has
   not yet confirmed either is allowed under "only this Kaggle dataset".
4. No clinician is available: reference texts must come from cited sources only, and no answer is rated by a dermatologist.

## Pending
- The supervisor's answers: outside label files; adding HAM10000 images; a text corpus as evidence.
- Pinning the package versions in `requirements.txt` (`pip freeze` on the laptop). Note: installing `ragas` replaced `openai` 3.49 with 1.109 in the laptop's environment; only the archived Version 2 code uses `openai`.
- The first run of the corpus builder on the laptop, then a check of its card (counts, removed, empty topics) before freezing.
- Rebuild the reference texts in the new form on the laptop, with `--token-check`, and review them before freezing.
- The model(s) for the LLM-only baseline (approval needed: it costs credit), then a small trial on validation, then the runs.
- The other baselines (standard RAG, RAG², proposed) on the three views.
- Tests for `src/feature_check.py` (the patient text, the options, the raw layout and the audit are tested in `tests/test_scripts.py`; the evaluation code in `tests/test_evaluate.py`; the corpus builder in `tests/test_corpus.py`).
- A first run of the MedCPT relevance scorers on the laptop (`--relevance-models`).

## 2026-11-24: feature check
- `check.py features` now prints and saves `accuracy` instead of balanced accuracy (the same number on the balanced validation and test splits). The laptop run on the new splits (✔ researcher's output): test accuracy shortcut features (original size) 0.317, age/sex/site 0.304, image features 0.525, image + metadata 0.554 (chance 0.125); source detectable from the standardised image features, AUC 0.953. A test keeps "balanced accuracy" out of `check.py`.
