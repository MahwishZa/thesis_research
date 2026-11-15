# Improving RAG for Medical Question Answering

## Overview

Existing medical RAG methods retrieve text passages by their similarity to a text question. This is insufficient for diagnosing skin lesions from photos and patient facts, because the retrieval ignores the image and returns noisy evidence, so models still hallucinate diagnoses. My research will investigate whether multimodal, rationale-guided retrieval reduces these errors, using closed-book and open-book experiments on image, text and multimodal datasets.

## Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Results

Closed-book results (no retrieval) of four LLMs on the test split of the three datasets: image (320 photos), text (221 cases) and multimodal (the same 320 photos with the patient's age, sex and body site). Each question has eight possible diagnoses. One run per model and dataset. Comparisons, interpretation and limits: [docs/results.md](docs/results.md).

### Balanced accuracy (%)

The main score: the mean recall over the eight diagnoses, with its 95% interval. Higher is better; chance is 12.5. Each value links to the run's full report.

| Model | Image | Text | Multimodal |
|---|---|---|---|
| Claude Sonnet 5.5 | [51.2 (45.7 to 56.7)](results/image/claude-sonnet-5-5/test_cf5351f1/report.md) | [55.9 (49.6 to 61.8)](results/text/claude-sonnet-5-5/test_06ce1664/report.md) | [56.2 (50.6 to 61.7)](results/multimodal/claude-sonnet-5-5/test_5db7a077/report.md) |
| GPT-4o | [34.1 (29.6 to 38.7)](results/image/gpt-4o/test_14e3059f/report.md) | [50.8 (45.1 to 56.6)](results/text/gpt-4o/test_5fc7ce07/report.md) | [41.6 (36.4 to 46.7)](results/multimodal/gpt-4o/test_12259e56/report.md) |
| Gemini 3.8 Flash | [49.7 (44.3 to 55.0)](results/image/gemini-3.8-flash/test_b086a9a1/report.md) | [61.4 (55.8 to 67.0)](results/text/gemini-3.8-flash/test_6ee1c09b/report.md) | [54.4 (48.6 to 59.9)](results/multimodal/gemini-3.8-flash/test_2ff4fbdd/report.md) |
| Llama 4 Maverick | [28.7 (24.4 to 33.1)](results/image/llama-4-maverick/test_055451fe/report.md) | [46.8 (40.6 to 52.8)](results/text/llama-4-maverick/test_32905170/report.md) | [34.1 (29.0 to 39.0)](results/multimodal/llama-4-maverick/test_c3b17116/report.md) |

### Hallucination rate (%)

The share of questions answered with a wrong diagnosis, averaged over the diagnoses. Lower is better.

| Model | Image | Text | Multimodal |
|---|---|---|---|
| Claude Sonnet 5.5 | 48.8 | 34.1 | 43.8 |
| GPT-4o | 65.9 | 49.2 | 58.4 |
| Gemini 3.8 Flash | 50.3 | 38.6 | 45.6 |
| Llama 4 Maverick | 71.2 | 48.5 | 65.9 |

### Missed malignancy (%)

Melanoma, basal cell carcinoma or squamous cell carcinoma answered as a benign diagnosis, as a share of all cancers. Lower is better.

| Model | Image | Text | Multimodal |
|---|---|---|---|
| Claude Sonnet 5.5 | 25.0 | 5.8 | 20.8 |
| GPT-4o | 22.5 | 14.2 | 19.2 |
| Gemini 3.8 Flash | 21.7 | 5.0 | 21.7 |
| Llama 4 Maverick | 19.2 | 10.0 | 10.8 |

## Repository Structure

| Path | Contents |
|---|---|
| `src/` | Evaluation code (`evaluate.py`) and dataset preparation code |
| `configs/` | One configuration per model and dataset |
| `prompts/` | The prompt of each dataset, the same for every model |
| `data/` | Dataset tables, data sheets and preparation reports (images are not stored) |
| `results/` | One folder per test run: answers, metrics and report |
| `docs/` | Results, comparisons and interpretation |
| `tests/` | Unit tests |
| `requirements.txt` | Pinned dependencies (Python 3.11 or newer) |

## How to Run

```
pip install -r requirements.txt
```
