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
  preflight.py    the setup check that runs before every model run (no dataset image is sent)
  evaluate.py     the command line: profile, isic, check, run, report
  models/         one adapter per model provider (Gemini only, for now)
tests/            unit tests (no API calls, no real data); they also check that README, configs and results agree
results/          the dataset profile, the ISIC label check and corrections, one folder per finished model run, and
                  EXPERIMENT_LOG.md (dated decisions, findings, settings and supervisor approvals)
docs/             temporary supervisor-meeting pack (open decisions, RAG data audit); delete later; not used by runs
data/             the extracted dataset (Train/ and Test/, one subfolder per class); git-ignored, never committed
```

## How to Run

Windows PowerShell, from the repository folder, with Python 3.11 or newer. **Run one command at a time and read its
output before the next one.**

Setup, once:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

The tests must end with `OK`.

Dataset, once:

1. Extract the dataset ZIP into `data\`, giving `data\Train\<class>\` and `data\Test\<class>\`.
2. Profile it (no model, no API): `python -m src.evaluate profile --root data --train Train --test Test`.
   This writes `results/dataset_profile/` (committed) and thumbnail sheets in `data/contact_sheets/` (not committed).
3. Check the labels against the ISIC Archive (needs internet; a few minutes): `python -m src.evaluate isic`.
   This writes `results/dataset_profile/isic_check.md`, `isic_check.csv` and `label_corrections.csv` (the testing images
   whose folder label ISIC contradicts, with the label ISIC supports); raw ISIC records stay in `data/isic_cache/`.
4. Put the API key in a file named `.env` in the repository folder (next to `README.md`). Create it with `notepad .env`,
   answer "Yes" to create the file, type one line and save: `GEMINI_API_KEY=<your key>` (no quotes needed). Further keys
   go on further lines. Git ignores `.env`, so it never leaves your computer; never paste a key into a chat or any
   other file. A key set in the terminal (`$env:GEMINI_API_KEY = "<your key>"`) takes priority over the file.

Every model run, in this order:

5. Check the setup: `python -m src.evaluate check --config configs/gemini-3.8-flash.toml`.
   Every line must say OK (a WARN explains itself). A FAIL line says what to do; fix it and run the check again. If
   Google is temporarily busy the check says so: wait a few minutes and run it again. If the model refuses a thinking
   setting, `python -m src.evaluate check --config configs/gemini-3.8-flash.toml --thinking-levels` shows which settings it
   accepts. The check sends only a blank synthetic image, never a dataset image.
6. Development check on a few training images: `python -m src.evaluate run --config configs/gemini-3.8-flash.toml --split dev --limit 5`.
7. Test run, once, after the supervisor approves the settings: `python -m src.evaluate run --config configs/gemini-3.8-flash.toml --split test`.
   Its report scores the same answers twice: against the dataset's folder labels and against the ISIC-corrected labels.
8. Only when a run prints `FINISHED` is its folder complete. Then commit it:
   `git add results`, `git commit -m "<what it is>"`, `git push origin main` (one command at a time).
   A finished run's report can be rewritten from its saved answers: `python -m src.evaluate report results/<model>/<run-folder>`.

A run that prints `NOT FINISHED` or `STOPPED` has nothing to commit. Its answers so far are in a folder ending in
`.partial`, which Git ignores; a run that has answered nothing leaves no folder. Run the same command again to
continue where it stopped (the folder name follows from the settings, so this also works on another day).

## If Something Goes Wrong

| Message | What it means | What to do |
|---|---|---|
| `no Gemini API key found` | no key in `.env` or the terminal | step 4 |
| `HTTP 404 ... no longer available` | the model is not available to your key | the check lists the models your key can use; set `id` in a new config file |
| `HTTP 400 ... Thinking level ... not supported` | the model refuses the thinking setting | `check --thinking-levels`, then change the setting in a new config file |
| `HTTP 400 ... API key not valid` | wrong key | correct the `GEMINI_API_KEY` line in `.env` |
| `HTTP 503 ... high demand` | Google is overloaded; temporary | the run retries for about 8 minutes, then stops with `NOT FINISHED`; run the same command again later |
| `HTTP 429` | too many requests or quota | wait, or lower `requests_per_minute` in a new config file |
| `network problem` | no connection, or no answer within 120 seconds | check the internet connection, then run the same command again |
| `answers were cut off at the output limit (MAX_TOKENS)` | the model's thinking used up `max_output_tokens` | raise `max_output_tokens` in a new config file; results are not valid until then |
| `images on disk differ from the committed profile` | dataset files changed or are missing | extract the ZIP again, or run steps 2 and 3 again and review the differences |
| `was started with a different config, prompt or image set` | an explicit `--run-id` was reused after a change | delete that `.partial` folder, or choose another `--run-id` |
| `a finished test run already exists` | the test split runs once per config | nothing; results are never overwritten |
| `an unfinished test run exists` | a test run was stopped | run its command again with the `--run-id` named in the message |
| `ERROR: ...` | a problem with the config, the data or a path, explained in the message | fix what it names |
