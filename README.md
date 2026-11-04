# Improving RAG for Medical Question Answering

## Overview

Existing methods do X. This is insufficient under condition Y because of limitation Z. My research will investigate whether approach W addresses that limitation, using experiments A and B.

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Results

Accuracy (%) on the 118 test images, zero-shot, temperature 0, one run. The main figure is scored against the ISIC-corrected
labels; each row links to the run's full report.

| Model | ISIC-corrected labels (main) | 95% interval | Folder labels | Report |
|---|---|---|---|---|
| Gemini 3.5 Flash-Lite (0-shot) | 37.3 | 29.1 to 46.3 | 36.4 | [report](results/gemini-3.5-flash-lite/test_88d88f32_r1/report.md) |

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
results/          the dataset profile, the ISIC label check and corrections, one folder per finished model run, and
                  EXPERIMENT_LOG.md (dated decisions, findings, settings and supervisor approvals)
docs/             temporary supervisor-meeting pack (open decisions, RAG data audit); delete later; not used by runs
data/             the extracted dataset (Train/ and Test/, one subfolder per class); git-ignored, never committed
```

## How to Run

Requires Python 3.11 or newer. Commands are for PowerShell, run in the repository folder.

1. **Set up (once):**

   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Add the data and the key (once):** put the dataset images in `data/Train/<class>/` and `data/Test/<class>/`, and create a
   file `.env` containing `GEMINI_API_KEY=<your key>`. Git ignores both.

3. **Check the setup** (sends one blank image, no dataset image):

   ```
   python -m src.evaluate check --config configs/gemini-3.5-flash-lite.toml
   ```

4. **Run the test split** (118 requests). It runs once per config and is never overwritten; if it stops, run the same command
   again to continue.

   ```
   python -m src.evaluate run --config configs/gemini-3.5-flash-lite.toml --split test
   ```

5. **Read the result** in `results/<model>/<run>/report.md`.
