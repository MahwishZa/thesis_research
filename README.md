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
  models/         one adapter per model provider (Gemini)
tests/            unit tests (no API calls, no real data); they also check that README, configs and results agree
results/          the dataset profile, the ISIC label check and corrections, one folder per finished model run, and
                  EXPERIMENT_LOG.md (dated decisions, findings, settings and supervisor approvals)
docs/             temporary supervisor-meeting pack (open decisions, RAG data audit); delete later; not used by runs
data/             the extracted dataset (Train/ and Test/, one subfolder per class); git-ignored, never committed
```

## How to Run

## If Something Goes Wrong

| Message | What it means | What to do |
|---|---|---|
| `unknown model provider ... no provider adapter is installed yet` | the config names a provider that has no adapter in `src/models/` | add the adapter, or use a provider that has one |
| `no ... API key found` | no key in `.env` or the terminal | put the key line in the git-ignored file `.env` in the repository folder |
| `HTTP 404` | the model is not available to your key | the check lists the models your key can use (if the adapter can); set `id` in a new config file |
| `HTTP 400` or `401`/`403` about the key | wrong or refused key | correct the key line in `.env` |
| `HTTP 503` ("high demand") | the provider is overloaded; temporary | the run retries for about 8 minutes, then stops with `NOT FINISHED`; run the same command again later |
| `Daily quota used up ...` (HTTP 429, "per day") | the per-day request limit of your tier for that model is reached; a test run needs one request per test image | the run stops at once and keeps its answers; run the same command after the reset the provider names, or use a paid key, or another model |
| `HTTP 429` (per minute) | too many requests in a minute | retried automatically; lower `requests_per_minute` in a new config file if it keeps happening |
| `HTTP 504`, `network problem (... timeout)` | the provider answered too slowly, or the connection was lost | retried automatically |
| `answers were cut off at the output limit (MAX_TOKENS)` | the answer hit `max_output_tokens` (hidden reasoning can count toward it) | raise `max_output_tokens` in a new config file; results are not valid until then |
| `images on disk differ from the committed profile` | dataset files changed or are missing | extract the ZIP again, or run the `profile` and `isic` commands again and review the differences |
| `was started with a different config, prompt or image set` | an explicit `--run-id` was reused after a change | delete that `.partial` folder, or choose another `--run-id` |
| `a finished test run already exists` | the test split runs once per config | nothing; results are never overwritten |
| `an unfinished test run exists` | a test run was stopped | run its command again with the `--run-id` named in the message |
| `ERROR: ...` | a problem with the config, the data or a path, explained in the message | fix what it names |
