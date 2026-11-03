# Improving RAG for Medical Question Answering

## Overview

Existing methods do X. This is insufficient under condition Y because of limitation Z. My research will investigate whether approach W addresses that limitation, using experiments A and B.

## Research Objectives

## Results

No model has been evaluated yet. Each finished run is listed here with a link to its report in `results/`.

## Repository Structure

```
configs/          one settings file per model run (model, generation settings, prompt, dataset split folders)
prompts/          the frozen prompt; a changed prompt is a new file (diagnosis_v2.txt), never an edit
src/
  dataset.py      reads the dataset folders and writes the dataset profile
  prompt.py       builds the prompt and reads the model's answer (the same rule for every model)
  metrics.py      accuracy, balanced accuracy, macro-F1, per-class figures, confusion matrix, intervals
  evaluate.py     the command line: profile, run, report
  models/         one adapter per model provider (Gemini only, for now)
tests/            unit tests (no API calls, no real data)
docs/
  experiment_log.md   dated decisions, settings and supervisor approvals
results/          created by the commands: the dataset profile and one folder per model run
data/             put the extracted dataset here; only data/README.md is committed, never the images
```

## How to Run

Windows PowerShell, from the repository folder, with Python 3.11 or newer:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

1. Extract the dataset ZIP into `data\` (see `data/README.md`), so that each split is a folder with one subfolder per class.
2. Profile it (no model, no API): `python -m src.evaluate profile --root data --train <training folder> --test <testing folder>`.
   This writes `results/dataset_profile/` (committed) and thumbnail sheets in `data/contact_sheets/` (not committed).
3. Write the two folder names into `[data]` in `configs/gemini-2.5-flash.toml`.
4. Set the API key for this PowerShell session: `$env:GEMINI_API_KEY = "<your key>"`. Never put the key in a file in the repository.
5. Development check on a few training images: `python -m src.evaluate run --config configs/gemini-2.5-flash.toml --split dev --limit 5`.
6. Test run, once, after the supervisor approves the settings: `python -m src.evaluate run --config configs/gemini-2.5-flash.toml --split test`.

A stopped run continues where it ended: run the same command again with `--run-id <run folder name>`.
