# Improving RAG for Medical Question Answering

## Overview

Existing medical RAG methods retrieve text passages by their similarity to a text question and are judged by answer accuracy alone. This is insufficient for diagnostic question answering on skin lesions, where the question is a photo, a case description or both, because the retrieval ignores the image and no answer is checked against its evidence, so models still give confident wrong diagnoses and unsupported claims. My research will investigate whether multimodal, rationale-guided retrieval with evidence-grounded answers reduces these hallucinations, using closed-book and open-book experiments on image, text and multimodal datasets.

## Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Results

Closed-book results (no retrieval) of four LLMs on the test split of the three datasets. Each question has eight possible diagnoses and the model names one. All values are percentages from one run per model and dataset. Each model name links to the run's report, which holds the confusion matrix and the per-class scores. Comparisons, interpretation and limits: [docs/results.md](docs/results.md).

### Image

Test split: 320 photos.

| Model | Accuracy | Precision | Recall | F1 | Hallucination rate | Missed malignancy |
|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](results/image/claude-sonnet-5-5/test_cf5351f1/report.md) | 51.2 | 50.8 | 51.2 | 49.5 | 48.8 | 25.0 |
| [GPT-4o](results/image/gpt-4o/test_14e3059f/report.md) | 34.1 | 36.4 | 34.1 | 30.1 | 65.9 | 22.5 |
| [Gemini 3.8 Flash](results/image/gemini-3.8-flash/test_b086a9a1/report.md) | 49.7 | 46.6 | 49.7 | 46.2 | 50.3 | 21.7 |
| [Llama 4 Maverick](results/image/llama-4-maverick/test_055451fe/report.md) | 28.7 | 39.1 | 28.7 | 25.7 | 71.2 | 19.2 |

### Text

Test split: 221 written cases (version 2 of the text dataset; version 3, used from now on, has 209).

| Model | Accuracy | Precision | Recall | F1 | Hallucination rate | Missed malignancy |
|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](results/text/claude-sonnet-5-5/test_06ce1664/report.md) | 64.3 | 63.9 | 55.9 | 57.9 | 34.1 | 5.8 |
| [GPT-4o](results/text/gpt-4o/test_5fc7ce07/report.md) | 56.6 | 54.3 | 50.8 | 49.4 | 49.2 | 14.2 |
| [Gemini 3.8 Flash](results/text/gemini-3.8-flash/test_6ee1c09b/report.md) | 69.2 | 65.1 | 61.4 | 61.3 | 38.6 | 5.0 |
| [Llama 4 Maverick](results/text/llama-4-maverick/test_32905170/report.md) | 53.8 | 53.8 | 46.8 | 47.0 | 48.5 | 10.0 |

### Image + metadata

Test split: the image dataset's 320 photos, each with the patient's age, sex and body site.

| Model | Accuracy | Precision | Recall | F1 | Hallucination rate | Missed malignancy |
|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](results/multimodal/claude-sonnet-5-5/test_5db7a077/report.md) | 56.2 | 58.3 | 56.2 | 55.7 | 43.8 | 20.8 |
| [GPT-4o](results/multimodal/gpt-4o/test_12259e56/report.md) | 41.6 | 49.7 | 41.6 | 40.6 | 58.4 | 19.2 |
| [Gemini 3.8 Flash](results/multimodal/gemini-3.8-flash/test_2ff4fbdd/report.md) | 54.4 | 58.9 | 54.4 | 53.1 | 45.6 | 21.7 |
| [Llama 4 Maverick](results/multimodal/llama-4-maverick/test_c3b17116/report.md) | 34.1 | 47.2 | 34.1 | 33.9 | 65.9 | 10.8 |

- **Precision, Recall, F1:** averaged over the eight diagnoses, each counting the same (macro average). Recall is the main score (balanced accuracy); chance is 12.5.
- **Hallucination rate:** the share of questions answered with a wrong diagnosis, averaged over the diagnoses.
- **Missed malignancy:** melanoma, basal cell carcinoma or squamous cell carcinoma answered as a benign diagnosis, as a share of all cancers.
- **Comparing datasets:** the image and image + metadata datasets use the same photos, so they can be compared directly. The text dataset describes different patients, so its scores are not directly comparable with the other two.

## Repository Structure

```
thesis_research/
├── src/                 evaluation and dataset preparation code
│   ├── evaluate.py
│   └── build/           how each dataset and the knowledge base are made and checked
├── configs/             one configuration per model and dataset
├── prompts/             one prompt per dataset
├── data/                one folder per dataset and the knowledge base: table, report, data sheet, build records
│   ├── image/           dermoscopic photos
│   ├── text/            written cases from published case reports
│   ├── cases/           written cases with the same patient's published photos
│   ├── image_metadata/  photos with the patient's age, sex and body site (ablation)
│   └── knowledge/       what open-book runs may retrieve
├── results/             one folder per test run
├── docs/                results and interpretation
├── tests/               unit tests
└── requirements.txt
```

## How to Run

```
pip install -r requirements.txt
```
