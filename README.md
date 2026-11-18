# Improving RAG for Medical Question Answering

## Overview

Existing medical RAG methods retrieve text passages by their similarity to a text question and are judged by answer accuracy alone. This is insufficient for diagnostic question answering on skin lesions, where the question is a photo, a case description or both, because the retrieval ignores the image and no answer is checked against its evidence, so models still give confident wrong diagnoses and unsupported claims. My research will investigate whether multimodal, rationale-guided retrieval with evidence-grounded answers reduces these hallucinations, using closed-book and open-book experiments on image, text and multimodal datasets.

## Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Results

Phase 1, closed-book (no retrieval): four LLMs answered the test split of three datasets, one run each. Each question has eight possible diagnoses and the model names one. The values are balanced accuracy in percent (the mean recall of the eight diagnoses; chance is 12.5), with the 95% interval. Every other score, the runs' reports and the comparisons: [docs/results.md](docs/results.md).

| Model | Image | Text | Image + metadata |
|---|---|---|---|
| Claude Sonnet 5.5 | 51.2 (45.7 to 56.7) | 55.9 (49.6 to 61.8) | 56.2 (50.6 to 61.7) |
| GPT-4o | 34.1 (29.6 to 38.7) | 50.8 (45.1 to 56.6) | 41.6 (36.4 to 46.7) |
| Gemini 3.8 Flash | 49.7 (44.3 to 55.0) | 61.4 (55.8 to 67.0) | 54.4 (48.6 to 59.9) |
| Llama 4 Maverick | 28.7 (24.4 to 33.1) | 46.8 (40.6 to 52.8) | 34.1 (29.0 to 39.0) |

- **Image:** 320 dermoscopic photos, 40 per diagnosis. **Text:** 221 written cases (version 2 of the text dataset; version 3, used from now on, has 209). **Image + metadata:** the same 320 photos, each with the patient's age, sex and body site.
- The image and image + metadata runs answer the same photos and can be compared directly; the text cases are other patients.

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
│   ├── image/           dermoscopic photos
│   ├── text/            written cases from published case reports
│   ├── cases/           written cases with the same patient's published photos
│   ├── image_metadata/  photos with the patient's age, sex and body site (ablation)
│   └── knowledge/       what open-book runs may retrieve
├── results/             one folder per test run: <dataset>/<approach>/<model>/ (phase 1 in phase1/)
├── docs/                results and interpretation
├── tests/               unit tests
└── requirements.txt
```

## How to Run

```
pip install -r requirements.txt
```

Put the API keys in a file named `.env` next to this README (`CLAUDE_API_KEY=...`, `OPENROUTER_API_KEY=...`). A run is one model of `configs/models.toml`, one dataset and one approach:

- Check everything a run needs (one tiny request, nothing from the dataset): `python -m src.run check --model claude-sonnet-5-5 --dataset text`
- Run the test split: `python -m src.run run --model claude-sonnet-5-5 --dataset text --split test`
- Compare two finished runs on the same questions (nothing is sent or changed): `python -m src.evaluate compare results/phase1/image/claude-sonnet-5-5/test_cf5351f1 results/phase1/image_metadata/claude-sonnet-5-5/test_5db7a077`
