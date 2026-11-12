# Improving RAG for Medical Question Answering

## Overview

Existing methods do X. This is insufficient under condition Y because of limitation Z. My research will investigate whether approach W addresses that limitation, using experiments A and B.

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Results

No results yet: the dataset is being prepared, and every model will then be evaluated on the same prepared test split (40 images per class). The main score is balanced accuracy (the mean recall of the classes, so every class counts the same) with its 95% interval, which resamples whole lesions because photos of one lesion are not independent. Each run's report also gives plain accuracy, macro-F1, recall per class, MCC, Cohen's kappa and the confusion matrix. Each row below will link to the run's full report.

## Repository Structure

```
configs/            one settings file per AI model (model and generation settings, prompt); everything else is shared
prompts/            the prompt texts; a changed prompt is a new file, never an edit
src/
  prepare.py        makes the dataset, one command per step, in this order: inventory (every original file, opened and
                    fingerprinted; identical files are one photo), isic (the ISIC Archive's record of every photo), identity
                    (each photo against ISIC's own picture), ham10000 (a second source for the HAM10000 photos), labels (each
                    photo's label from its ISIC record, by the naming rule), duplicates (photos that look alike), lesions
                    (photos that share a skin pattern: second photos of one lesion), splits (train, validation and test:
                    every lesion in one split, 40 test and 15 validation photos per class, balanced within each class),
                    images (every photo in a split fitted whole within 600 x 450 and saved the same way), shortcuts (how
                    much simple image properties such as size, dark corners or sharpness reveal the class; changes nothing)
  evaluate.py       runs the AI models and scores them (check, run, report): the metrics, one adapter per AI service (Claude,
                    Gemini, GPT-4o; Llama through an OpenAI-style service), and the pipeline that checks every image against
                    data/dataset.csv, sends it with the prompt, reads the answer and writes the report
tests/              test_prepare.py and test_evaluate.py, one per code file (no API call, no real data)
data/
  raw/              the original Kaggle dataset, never changed (not in Git)
  downloads/        ISIC records and pictures, HAM10000's metadata (not in Git)
  dataset/          the prepared images: train/, val/, test/ (not in Git)
  raw_photos.csv    one row per original photo: everything the preparation finds out about it
  similar_photo_pairs.csv  pairs of photos found alike: by comparing whole pictures, or by skin pattern
  isic_to_class.csv written by people: which ISIC wording belongs to which class
  eye_check_verdicts.csv  written by people: the eye check's verdict on flagged pairs, and which copy of a photo stays
  dataset_report.md the preparation report: one section per step, in the order the steps run
  datasheet.md      the data sheet: what the dataset is, how it was made, how to use it, its limitations
  dataset.csv       the prepared images the experiments read: split, label, file, fingerprint, lesion group
results/            one folder per finished test run (validation runs stay on your computer)
```

## How to Run

Set up a Python 3.11+ virtual environment in the project folder and install the packages into it, never into the computer's main Python (OpenCV needs numpy 2, which other projects may not accept). On Windows: `python -m venv .venv` once; then, in every new PowerShell window, `.\.venv\Scripts\Activate.ps1` (the prompt then starts with `(.venv)`); then `python -m pip install -r requirements.txt`. Put the original dataset in `data/raw` (the `Train` and `Test` folders from the Kaggle ZIP), and add the key of the model you run to `.env` (`CLAUDE_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY`). The tests run with `python -m unittest discover -s tests`.

The dataset is prepared step by step: `python -m src.prepare inventory`, then `python -m src.prepare isic`, `python -m src.prepare identity`, `python -m src.prepare ham10000`, `python -m src.prepare labels`, `python -m src.prepare duplicates`, `python -m src.prepare lesions`, `python -m src.prepare splits`, `python -m src.prepare images` and `python -m src.prepare shortcuts`. Each step rewrites only its own part of `data/raw_photos.csv`, `data/similar_photo_pairs.csv` and `data/dataset_report.md` (the images step also writes `data/dataset.csv` and the images in `data/dataset`), so running a step again gives identical files.

Once the dataset is prepared, run `python -m src.evaluate check --config configs/claude-sonnet-5-5.toml`, then `python -m src.evaluate run --config configs/claude-sonnet-5-5.toml --split test` (`configs/gpt-4o.toml` and `configs/gemini-3.1-flash-lite.toml` for the others; a stopped run continues when repeated).
