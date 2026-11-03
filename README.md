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
  dataset.py      reads the dataset folders, writes the dataset profile, prepares images for sending
  isic.py         checks the dataset's labels against the ISIC Archive; lists the corrected testing labels
  prompt.py       builds the prompt and reads the model's answer (the same rule for every model)
  metrics.py      accuracy, balanced accuracy, macro-F1, per-class figures, confusion matrix, intervals
  evaluate.py     the command line: profile, isic, run, report
  models/         one adapter per model provider (Gemini only, for now)
tests/            unit tests (no API calls, no real data)
results/          the dataset profile, the ISIC label check and corrections, and one folder per model run
EXPERIMENT_LOG.md dated decisions, findings, settings and supervisor approvals
data/             the extracted dataset (Train/ and Test/, one subfolder per class); git-ignored, never committed
```

## How to Run

Windows PowerShell, from the repository folder, with Python 3.11 or newer:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

1. Extract the dataset ZIP into `data\`, giving `data\Train\<class>\` and `data\Test\<class>\`.
2. Profile it (no model, no API): `python -m src.evaluate profile --root data --train Train --test Test`.
   This writes `results/dataset_profile/` (committed) and thumbnail sheets in `data/contact_sheets/` (not committed).
3. Check the labels against the ISIC Archive (needs internet; a few minutes): `python -m src.evaluate isic`.
   This writes `results/dataset_profile/isic_check.md`, `isic_check.csv` and `label_corrections.csv` (the testing images
   whose folder label ISIC contradicts, with the label ISIC supports); raw ISIC records stay in `data/isic_cache/`.
   Running it again re-checks only the records that could not be fetched.
4. Set the API key for this PowerShell session: `$env:GEMINI_API_KEY = "<your key>"`. Never put the key in a file in the repository.
5. Development check on a few training images: `python -m src.evaluate run --config configs/gemini-2.5-flash.toml --split dev --limit 5`.
6. Test run, once, after the supervisor approves the settings: `python -m src.evaluate run --config configs/gemini-2.5-flash.toml --split test`.
   Its report scores the same answers twice: against the dataset's folder labels and against the ISIC-corrected labels.

A stopped run continues where it ended: run the same command again with `--run-id <run folder name>`.
