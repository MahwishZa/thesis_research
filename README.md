# Improving RAG for Medical Question Answering

## Overview

Existing methods do X. This is insufficient under condition Y because of limitation Z. My research will investigate whether approach W addresses that limitation, using experiments A and B.

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Results

Accuracy (%) on the 118 test images, zero-shot, temperature 0, one run, scored against the ISIC-corrected labels (folder labels in the last column). The final Gemini run is pending; each finished run is listed with a link to its report.

| Model | ISIC-corrected labels (main) | 95% interval | Folder labels | Report |
|---|---|---|---|---|
| Gemini 3.5 Flash-Lite (0-shot) | pending | pending | pending | pending |

## Repository Structure

```
configs/          one settings file per model run (model, generation settings, prompt, dataset split folders)
prompts/          the frozen prompt; a changed prompt is a new file (diagnosis_v2.txt), never an edit
src/
  dataset.py      reads the dataset folders, writes the dataset profile, prepares images for sending
  isic.py         checks the dataset's labels against the ISIC Archive; lists the corrected testing labels
  prompt.py       builds the prompt and reads the model's answer (the same rule for every model)
  metrics.py      accuracy and its 95% interval (the only metric until the supervisor approves others)
  preflight.py    the setup check that runs before every model run (no dataset image is sent)
  evaluate.py     the command line: profile, isic, check, run, report
  models/         one adapter per model provider (Gemini)
tests/            unit tests (no API calls, no real data); they also check that README, configs and results agree
results/          the dataset profile, the ISIC label check and corrections, and one folder per finished model run
docs/             EXPERIMENT_LOG.md (dated decisions and findings) and meeting_brief.md (result and open decisions)
data/             the extracted dataset (Train/ and Test/, one subfolder per class); git-ignored, never committed
```

## How to Run

With Python 3.11+, run `python -m venv .venv`, `.venv\Scripts\Activate.ps1` and `pip install -r requirements.txt`, put the dataset in `data/Train` and `data/Test`, and add `GEMINI_API_KEY=<your key>` to a `.env` file.

Then run `python -m src.evaluate check --config configs/gemini-3.5-flash-lite.toml` followed by `python -m src.evaluate run --config configs/gemini-3.5-flash-lite.toml --split test` (a stopped run continues when repeated).
