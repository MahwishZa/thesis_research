# Improving RAG for Medical Question Answering

## Overview

Existing methods do X. This is insufficient under condition Y because of limitation Z. My research will investigate whether approach W addresses that limitation, using experiments A and B.

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Results

No results yet: every model will be evaluated on the same prepared test split (40 images per class). The main score is balanced accuracy (the mean recall of the classes, so every class counts the same) with its 95% interval, which resamples whole lesions because photos of one lesion are not independent. Each run's report also gives plain accuracy, macro-F1, recall per class, MCC, Cohen's kappa and the confusion matrix. Each row below will link to the run's full report.

## Repository Structure

```
src/
  common.py                 shared by the dataset folders: polite downloads that try again when a service is busy, and the
                            reports with one section per step
  evaluate.py               shared by every dataset: runs the AI models and scores them (check, run, report): the metrics,
                            one adapter per AI service (Claude, Gemini, GPT-4o; Llama through an OpenAI-style service), and
                            the pipeline that checks every file against the dataset's table, sends it with the prompt, reads
                            the answer and writes the report
  image/prepare.py          makes the image dataset, one command per step, in this order: inventory (every original file,
                            opened and fingerprinted; identical files are one photo), isic (the ISIC Archive's record of every
                            photo), identity (each photo against ISIC's own picture), ham10000 (a second source for the
                            HAM10000 photos), labels (each photo's label from its ISIC record, by the naming rule), duplicates
                            (photos that look alike), lesions (photos that share a skin pattern: second photos of one lesion),
                            splits (train, validation and test: every lesion in one split, 40 test and 15 validation photos
                            per class, balanced within each class), images (every photo in a split fitted whole within
                            600 x 450 and saved the same way), shortcuts (how much simple image properties such as size, dark
                            corners or sharpness reveal the class; changes nothing)
  text/prepare.py           makes the text dataset, one command per step: survey (checks the names in class_terms.csv against
                            MeSH and counts what PubMed holds per class), cases (downloads PMC-Patients once and lists the
                            patients whose article title names a class, with marks), abstracts (the same for the case
                            reports PubMed holds only as an abstract, or an open full text's case section where
                            PMC has one), mesh (asks PubMed whether each
                            candidate article is indexed and which classes its subject headings name, on the skin), build
                            (makes the dataset with no person in the loop: a label where the sources agree, the case text
                            up to its diagnosis as the question, checks, and splits by article)
  text/audit.py             checks the built text dataset with code of its own: every question against its file, its source
                            and the rules (no class name, short form or test result; long enough; a lesion described)
  text/library.py           builds the reference library for retrieval, four corpora as in RAG²: PubMed abstracts of reviews,
                            PMC open full texts of reviews, clinical practice guidelines, StatPearls chapters; cut into
                            passages, with nothing that could hold a test answer
tests/                      one test file per code file (no API call, no real data)
  test_evaluate.py
  image/test_prepare.py
  text/test_prepare.py
  text/test_audit.py
  text/test_library.py
configs/image/              one settings file per AI model for the image dataset (model and generation settings, prompt)
prompts/image/              the image prompt, diagnosis.txt; a changed prompt is a new file, never an edit
data/image/                 the image dataset
  raw/                      the original Kaggle dataset, never changed (not in Git)
  downloads/                ISIC records and pictures, HAM10000's metadata (not in Git)
  dataset/                  the prepared images: train/, val/, test/ (not in Git)
  raw_photos.csv            one row per original photo: everything the preparation finds out about it
  similar_photo_pairs.csv   pairs of photos found alike: by comparing whole pictures, or by skin pattern
  isic_to_class.csv         written by people: which ISIC wording belongs to which class
  eye_check_verdicts.csv    written by people: the eye check's verdict on flagged pairs, and which copy of a photo stays
  dataset_report.md         the preparation report: one section per step, in the order the steps run
  datasheet.md              the data sheet: what the dataset is, how it was made, how to use it, its limitations
  dataset.csv               the prepared images the experiments read: split, label, file, fingerprint, lesion group
data/text/                  the text dataset
  class_terms.csv           written by people: the names doctors use for each class, and whether each name counts for it
  sources/                  downloads kept as received: PMC-Patients' patient descriptions, PubMed abstracts, the StatPearls
                            archive (not in Git)
  term_check.csv            every name in class_terms.csv, checked against MeSH and counted in PubMed
  case_candidates.csv       the patients whose article title names a class: candidates for the cases, with marks
  abstract_candidates.csv   the same for PubMed's case reports not in PMC-Patients (abstracts, or open full texts)
  article_mesh.csv          for each candidate article: whether PubMed has indexed it, and the classes its headings name
  dataset/                  the question texts: test/, val/, train/ (not in Git: research use only)
  dataset.csv               the text dataset the experiments read: split, label, file, fingerprint, source numbers
  dataset_report.md         the preparation report: one section per step, in the order the steps run
  datasheet.md              the data sheet: what the dataset is, how it was made, how to use it, its limitations
  library/                  the library's passages, passages.jsonl (not in Git: research use only)
  library.csv               the reference library: one row per passage (corpus, source, licence, fingerprint; not its text)
results/image/              one folder per AI model, holding its finished test run (validation runs stay on your computer)
```

The text dataset is in `src/text/` and `data/text/`; it gets its `configs/text/`, `prompts/text/` and `results/text/` when its evaluation starts. The multimodal dataset gets the same folders when its phase starts.

## How to Run

Set up a Python 3.11+ virtual environment in the project folder and install the packages into it, never into the computer's main Python (OpenCV needs numpy 2, which other projects may not accept). On Windows: `python -m venv .venv` once; then, in every new PowerShell window, `.\.venv\Scripts\Activate.ps1` (the prompt then starts with `(.venv)`); then `python -m pip install -r requirements.txt`. Put the original dataset in `data/image/raw` (the `Train` and `Test` folders from the Kaggle ZIP), and add the key of the model you run to `.env` (`CLAUDE_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY`). The tests run with `python -m unittest discover -s tests`.

The image dataset is prepared step by step: `python -m src.image.prepare inventory`, then `python -m src.image.prepare isic`, `python -m src.image.prepare identity`, `python -m src.image.prepare ham10000`, `python -m src.image.prepare labels`, `python -m src.image.prepare duplicates`, `python -m src.image.prepare lesions`, `python -m src.image.prepare splits`, `python -m src.image.prepare images` and `python -m src.image.prepare shortcuts`. Each step rewrites only its own part of `data/image/raw_photos.csv`, `data/image/similar_photo_pairs.csv` and `data/image/dataset_report.md` (the images step also writes `data/image/dataset.csv` and the images in `data/image/dataset`), so running a step again gives identical files.

The text dataset is prepared the same way, step by step: `python -m src.text.prepare survey`, then `python -m src.text.prepare cases`, `python -m src.text.prepare abstracts`, `python -m src.text.prepare mesh` and `python -m src.text.prepare build`; then `python -m src.text.audit` checks the dataset and `python -m src.text.library` builds the reference library. Each step rewrites only its own part of `data/text/dataset_report.md`; the case texts, abstracts and library passages stay in `data/text/sources`, `data/text/dataset` and `data/text/library` and never go into Git.

To evaluate a model on the image dataset, run `python -m src.evaluate check --config configs/image/claude-sonnet-5-5.toml`, then `python -m src.evaluate run --config configs/image/claude-sonnet-5-5.toml --split test` (`configs/image/gpt-4o.toml` and `configs/image/gemini-3.1-flash-lite.toml` for the others; a stopped run continues when repeated).
