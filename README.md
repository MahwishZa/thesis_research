# Improving RAG for Medical Question Answering

## Overview

Existing medical RAG methods retrieve text passages by their similarity to a text question. This is insufficient for diagnosing skin lesions from photos and patient facts, because the retrieval ignores the image and returns noisy evidence, so models still hallucinate diagnoses. My research will investigate whether multimodal, rationale-guided retrieval reduces these errors, using closed-book and open-book experiments on image, text and multimodal datasets.

## Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Results

Closed-book results (no retrieval) of four LLMs on the test split of the three datasets. Each question has eight possible diagnoses and the model names one. All values are percentages from one run per model and dataset. Each model name links to the run's report, which holds the confusion matrix and the per-class scores. Definitions, every interval, the significance tests and the interpretation: [docs/results.md](docs/results.md).

### Image

Test split: 320 photos.

| Model | Accuracy | Precision | Recall | F1 | Specificity | Hallucination rate | Malignancy sensitivity | Missed malignancy |
|---|---|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](results/image/claude-sonnet-5-5/test_cf5351f1/report.md) | 51.2 | 50.8 | 51.2 (45.7 to 56.7) | 49.5 | 93.0 | 48.8 | 50.8 | 25.0 |
| [GPT-4o](results/image/gpt-4o/test_14e3059f/report.md) | 34.1 | 36.4 | 34.1 (29.6 to 38.7) | 30.1 | 90.6 | 65.9 | 77.5 | 22.5 |
| [Gemini 3.8 Flash](results/image/gemini-3.8-flash/test_b086a9a1/report.md) | 49.7 | 46.6 | 49.7 (44.3 to 55.0) | 46.2 | 92.8 | 50.3 | 57.5 | 21.7 |
| [Llama 4 Maverick](results/image/llama-4-maverick/test_055451fe/report.md) | 28.7 | 39.1 | 28.7 (24.4 to 33.1) | 25.7 | 89.8 | 71.2 | 78.3 | 19.2 |

### Text

Test split: 221 written cases.

| Model | Accuracy | Precision | Recall | F1 | Specificity | Hallucination rate | Malignancy sensitivity | Missed malignancy |
|---|---|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](results/text/claude-sonnet-5-5/test_06ce1664/report.md) | 64.3 | 63.9 | 55.9 (49.6 to 61.8) | 57.9 | 95.3 | 34.1 | 90.8 | 5.8 |
| [GPT-4o](results/text/gpt-4o/test_5fc7ce07/report.md) | 56.6 | 54.3 | 50.8 (45.1 to 56.6) | 49.4 | 93.6 | 49.2 | 83.3 | 14.2 |
| [Gemini 3.8 Flash](results/text/gemini-3.8-flash/test_6ee1c09b/report.md) | 69.2 | 65.1 | 61.4 (55.8 to 67.0) | 61.3 | 95.4 | 38.6 | 93.3 | 5.0 |
| [Llama 4 Maverick](results/text/llama-4-maverick/test_32905170/report.md) | 53.8 | 53.8 | 46.8 (40.6 to 52.8) | 47.0 | 93.9 | 48.5 | 84.2 | 10.0 |

### Multimodal

Test split: the image dataset's 320 photos, each with the patient's age, sex and body site.

| Model | Accuracy | Precision | Recall | F1 | Specificity | Hallucination rate | Malignancy sensitivity | Missed malignancy |
|---|---|---|---|---|---|---|---|---|
| [Claude Sonnet 5.5](results/multimodal/claude-sonnet-5-5/test_5db7a077/report.md) | 56.2 | 58.3 | 56.2 (50.6 to 61.7) | 55.7 | 93.8 | 43.8 | 62.5 | 20.8 |
| [GPT-4o](results/multimodal/gpt-4o/test_12259e56/report.md) | 41.6 | 49.7 | 41.6 (36.4 to 46.7) | 40.6 | 91.7 | 58.4 | 79.2 | 19.2 |
| [Gemini 3.8 Flash](results/multimodal/gemini-3.8-flash/test_2ff4fbdd/report.md) | 54.4 | 58.9 | 54.4 (48.6 to 59.9) | 53.1 | 93.5 | 45.6 | 65.8 | 21.7 |
| [Llama 4 Maverick](results/multimodal/llama-4-maverick/test_c3b17116/report.md) | 34.1 | 47.2 | 34.1 (29.0 to 39.0) | 33.9 | 90.6 | 65.9 | 80.0 | 10.8 |

- **Precision, Recall, F1, Specificity:** averaged over the eight diagnoses, each counting the same (macro average). Recall is the main score (balanced accuracy), shown with its 95% interval; chance is 12.5. With eight diagnoses specificity is high by construction.
- **Hallucination rate:** the share of questions answered with a wrong diagnosis, averaged over the diagnoses.
- **Malignancy sensitivity:** of the cancers (melanoma, basal cell carcinoma, squamous cell carcinoma), the share answered as any cancer. **Missed malignancy:** the share answered as a benign diagnosis. The rest were answered as actinic keratosis or without a valid answer.
- **AUC-ROC** is not available: the models return one diagnosis name and no scores.
- **Comparing datasets:** the image and multimodal datasets use the same photos, so they can be compared directly. The text dataset describes different patients, so its scores are not directly comparable with the other two.

## Repository Structure

```
thesis_research/
├── src/                 evaluation and dataset preparation code
│   ├── evaluate.py
│   ├── image/
│   ├── text/
│   └── multimodal/
├── configs/             one configuration per model and dataset
├── prompts/             one prompt per dataset
├── data/                dataset tables and data sheets
├── results/             one folder per test run
├── docs/                results and interpretation
├── tests/               unit tests
└── requirements.txt
```

## How to Run

```
pip install -r requirements.txt
```
