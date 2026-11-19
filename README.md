# Improving RAG for Medical Question Answering

## Overview

Existing medical RAG methods retrieve text passages by their similarity to a text question and are judged by answer accuracy alone. This is insufficient for diagnostic question answering on skin lesions, where the question is a photo, a case description or both, because the retrieval ignores the image and no answer is checked against its evidence, so models still give confident wrong diagnoses and unsupported claims. My research will investigate whether multimodal, rationale-guided retrieval with evidence-grounded answers reduces these hallucinations, using closed-book and open-book experiments on image, text and multimodal datasets.

## Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Results

Version 2 is in progress: the datasets are being prepared (the clinic patients cohort is prepared; no model has been run on it yet), so there are no Version 2 results yet.

Version 1 (commit `a1c9209`) is kept in [archive/v1/](archive/v1/README.md): four LLMs answered three datasets of dermoscopic photos and published case reports without retrieval (closed-book), one run each. Its results and their interpretation are in [archive/v1/docs/results.md](archive/v1/docs/results.md). That archive is temporary and is removed when the research is final.

## Repository Structure

```
thesis_research/
├── src/                 the code
│   ├── build/           how each dataset and the knowledge base are made and checked
│   ├── datasets.py      reads a prepared dataset and checks every file
│   ├── prompts.py       the prompt and the answer reader
│   ├── models.py        one adapter per model API
│   ├── run.py           runs a model on a dataset (python -m src.run)
│   ├── metrics.py       the scores
│   ├── evaluate.py      reports and comparisons from saved answers (python -m src.evaluate)
│   └── common.py        helpers they share
├── configs/             the models and their settings (models.toml)
├── prompts/             one prompt per dataset and approach
├── data/                one folder per dataset and the knowledge base: table, report, data sheet, build records
│   ├── clinic_patients/     PAD-UFES-20: phone photos of skin lesions with each patient's clinic record (Version 2)
│   ├── text/                written cases from published case reports
│   ├── cases/               written cases with the same patient's published photos
│   ├── image/               Version 1: dermoscopic photos (to be removed from here; kept in archive/v1)
│   ├── image_metadata/      Version 1: photos with age, sex and body site (to be removed from here; kept in archive/v1)
│   └── knowledge/           what open-book runs may retrieve
├── docs/                temporary: the Version 2 plan and the method; removed when the research is final
├── archive/             temporary: Version 1 (commit a1c9209), frozen for reference; removed when the research is final
├── tests/               unit tests
└── requirements.txt
```

## How to Run

```
pip install -r requirements.txt
```

Put the API keys in a file named `.env` next to this README (`CLAUDE_API_KEY=...`, `OPENROUTER_API_KEY=...`). A run is one model of `configs/models.toml`, one dataset and one approach; its answers and report are written to `results/<dataset>/<approach>/<model>/<run>/` (the folder appears with the first Version 2 run):

- Check everything a run needs (one tiny request, nothing from the dataset): `python -m src.run check --model claude-sonnet-5-5 --dataset text`
- Run the test split: `python -m src.run run --model claude-sonnet-5-5 --dataset text --split test`
- Compare two finished runs on the same questions (nothing is sent or changed): python -m src.evaluate compare, then the two run folders
- Prepare the clinic patients dataset (needs the PAD-UFES-20 download in `data/clinic_patients/raw/`; the photos stay on this computer): `python -m src.build.clinic_patients.prepare all`
