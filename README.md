# Improving RAG for Medical Question Answering

## Overview

This thesis investigates retrieval-augmented generation (RAG) for medical question answering. Its base paper, and the template
for its method and reporting, is *Rationale-Guided Retrieval Augmented Generation for Medical Question Answering* (RAG²; Sohn
et al., NAACL 2025). The experiments use a skin-lesion image dataset supplied by the supervisor and proceed in three phases;
each model is reviewed with the supervisor before the next one is started.

| Phase | Description | Status |
|---|---|---|
| 1 | Evaluate large language models (LLMs) individually, without retrieval | In progress: Gemini 3.5 Flash-Lite is complete; further LLMs await the supervisor's approval |
| 2 | Standard RAG | Not started |
| 3 | RAG² baseline | Not started |

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant
   models and existing works.

## Dataset

- **Content:** 2,357 dermoscopic skin-lesion images in 9 classes, split by the dataset into 2,239 training and 118 test images
  (per-class counts: `results/dataset_profile/profile.md`).
- **Source:** ISIC Archive images; the same set is published on Kaggle as `nodoubttome/skin-cancer9-classesisic`. The images
  are not part of this repository (`data/` is ignored by Git).
- **Labels:** the class is the folder name. For 3 test images labelled seborrheic keratosis, the ISIC record (histopathology)
  gives melanoma; every report scores both the ISIC-corrected and the folder labels (`results/dataset_profile/`).

## Evaluation Protocol (Phase 1)

- **Task:** zero-shot classification. The LLM receives one image and a fixed prompt (`prompts/diagnosis_v1.txt`) listing the
  9 class names, and must reply with one class name. No LLM is trained or fine-tuned, and none sees a label.
- **Comparability:** the same prompt, answer reader and image rule for every LLM. An unreadable answer counts as wrong.
- **Settings:** temperature 0 and one run, as in the base paper; each model's other settings are in its config file in `configs/`.
- **Metric:** accuracy on the 118 test images with its 95% Wilson interval, main result against the ISIC-corrected labels.
  Retrieval metrics and any further metrics are added only with the supervisor's approval.
- **Reproducibility:** the test split is run once per config and never overwritten. Each run saves every answer, its settings,
  prompt and image-set hashes, package versions and the Git commit (`run_card.json`). Dated decisions are in
  `results/EXPERIMENT_LOG.md`.

## Results

Accuracy (%) on the 118 test images, zero-shot, temperature 0, one run. Each row links to the run's full report.

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

## Reference

Sohn, J., Park, Y., Yoon, C., Park, S., Hwang, H., Sung, M., Kim, H., and Kang, J. (2025). Rationale-Guided Retrieval
Augmented Generation for Medical Question Answering. *Proceedings of NAACL 2025*, pp. 12739-12753.
[ACL Anthology](https://aclanthology.org/2025.naacl-long.635) · [Code](https://github.com/dmis-lab/RAG2)
