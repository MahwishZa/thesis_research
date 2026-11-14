# Improving RAG for Medical Question Answering

## Overview

Existing medical RAG methods retrieve text passages by their similarity to a text question. This is insufficient for diagnosing skin lesions from photos and patient facts, because the retrieval ignores the image and returns noisy evidence, so models still hallucinate diagnoses. My research will investigate whether multimodal, rationale-guided retrieval reduces these errors, using closed-book and open-book experiments on image, text and multimodal datasets.

## Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Results

### Image

| Model | Setting | Balanced accuracy (95% CI) | Hallucination rate | Missed malignancy |
|---|---|---|---|---|
| [Claude Sonnet 5.5](results/image/claude-sonnet-5-5/test_cf5351f1/report.md) | Closed-book | 51.2% (45.7 to 56.7) | 48.8% | 25.0% |
| [GPT-4o](results/image/gpt-4o/test_14e3059f/report.md) | Closed-book | 34.1% (29.6 to 38.7) | 65.9% | 22.5% |

### Text

| Model | Setting | Balanced accuracy (95% CI) | Hallucination rate | Missed malignancy |
|---|---|---|---|---|
| [Claude Sonnet 5.5](results/text/claude-sonnet-5-5/test_06ce1664/report.md) | Closed-book | 55.9% (49.6 to 61.8) | 34.1% | 5.8% |
| [GPT-4o](results/text/gpt-4o/test_5fc7ce07/report.md) | Closed-book | 50.8% (45.1 to 56.6) | 49.2% | 14.2% |

### Multimodal

| Model | Setting | Balanced accuracy (95% CI) | Hallucination rate | Missed malignancy |
|---|---|---|---|---|
| [Claude Sonnet 5.5](results/multimodal/claude-sonnet-5-5/test_5db7a077/report.md) | Closed-book | 56.2% (50.6 to 61.7) | 43.8% | 20.8% |
| [GPT-4o](results/multimodal/gpt-4o/test_12259e56/report.md) | Closed-book | 41.6% (36.4 to 46.7) | 58.4% | 19.2% |

## Repository Structure

```
src/
    evaluate.py
    common.py
    image/
    text/
    multimodal/
tests/
configs/
prompts/
data/
    image/
        datasheet.md
    text/
        datasheet.md
    multimodal/
        datasheet.md
results/
requirements.txt
```

## How to Run
