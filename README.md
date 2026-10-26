# Improving RAG for Medical Question Answering

## Overview

Existing medical RAG filters, such as RAG² (Sohn et al., 2025), learn which retrieved passages to keep from labels that compare the model's answer and confidence with and without each passage. This is insufficient when adding any text changes the model's answer, because the labels then credit a passage for changes that an irrelevant passage would also cause. My research will investigate whether labels adjusted with irrelevant control passages address this, using two experiments on MedQA: (A) checking how often the labels call irrelevant passages helpful, and (B) comparing accuracy and harm across no retrieval, unfiltered retrieval, the original filter, the adjusted filter and an equal-budget baseline.

## Research Objectives

## Results

## Repository Structure

## How to Run

Install the code once, from the repository folder:

```
pip install -e .
```

Run the tests from the same folder:

```
python -m unittest discover -s tests -t .
```

Tests that need PyTorch and Transformers are skipped unless those packages are installed (`pip install -e ".[rationale]"`).
