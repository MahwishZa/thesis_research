# Improving RAG for Medical Question Answering

## Overview

Existing methods do X. This is insufficient under condition Y because of limitation Z. My research will investigate whether approach W addresses that limitation, using experiments A and B.

## Research Objectives

## Results

Each finished run is listed here with a link to its report in `results/`. Accuracy on the 118 test images, scored against the
ISIC-corrected labels (the dataset's folder labels in brackets); temperature 0, one run.

| Model | Accuracy (%) | Report |
|---|---|---|
| Gemini 3.5 Flash-Lite (0-shot) | 37.3 (36.4) | [report](results/gemini-3.5-flash-lite/test_88d88f32_r1/report.md) |

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

Commands are for PowerShell, run in the repository folder. Python 3.11 or newer.

**1. Set up (once).**

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Activate `.venv` in every new terminal before the commands below.

**2. Add the dataset and the API key (once).**
- Extract the dataset so that the images are in `data/Train/<class>/` and `data/Test/<class>/`. Git ignores `data/`.
- Create a file named `.env` in the repository folder with the line `GEMINI_API_KEY=<your key>`. Git ignores `.env`; never commit it.

**3. Check the setup (before every run).** It tests the config, prompt, dataset and key, and sends one blank image
(1 request); no dataset image is sent.

```
python -m src.evaluate check --config configs/gemini-3.5-flash-lite.toml
```

**4. Run.** Each answer is saved as it arrives. If a run stops (quota, network), run the same command again: it continues.

```
python -m src.evaluate run --config configs/gemini-3.5-flash-lite.toml --split dev --limit 5
python -m src.evaluate run --config configs/gemini-3.5-flash-lite.toml --split test
```

- `dev` sends a few real training images to check the pipeline; it is never a reported result.
- `test` sends the 118 test images (118 requests). The test split is run once per config and a finished run is never
  overwritten; a changed setting needs a new config file.

**5. Read the result.** A finished run is a folder `results/<model>/<run>/` with `report.md` (accuracy), `metrics.json`,
`predictions.jsonl` (every answer) and `run_card.json` (settings, versions, commit). To rewrite the report from the saved
answers: `python -m src.evaluate report results/<model>/<run>`.

**Tests** (no API calls): `python -m unittest discover -s tests`
