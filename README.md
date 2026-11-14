# Improving RAG for Medical Question Answering

## Overview

Existing medical RAG methods retrieve text passages by their similarity to a text question. This is insufficient for diagnosing skin lesions from photos and patient facts, because the retrieval ignores the image and returns noisy evidence, so models still hallucinate diagnoses. My research will investigate whether multimodal, rationale-guided retrieval reduces these errors, using closed-book and open-book experiments on image, text and multimodal datasets.

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Repository Structure

```
src/evaluate.py      runs the AI models on a dataset and scores them (check, run, report, compare); shared by all datasets
src/common.py        helpers the dataset folders share
src/image/           prepare.py: makes the image dataset
src/text/            prepare.py, audit.py: make and check the text dataset; library.py: builds the reference library
src/multimodal/      prepare.py, audit.py: make and check the multimodal dataset
tests/               one test file per code file: test_evaluate.py; image/, text/ and multimodal/ hold test_prepare.py,
                     test_audit.py and test_library.py
configs/<dataset>/   one settings file per AI model
prompts/<dataset>/   the prompt, diagnosis.txt
data/<dataset>/      the dataset's tables and report; datasheet.md says how it was made and how to rebuild it
results/<dataset>/   one folder per AI model, holding its finished test run
```

## How to Run

With Python 3.11 or newer, `pip install -r requirements.txt`, and the model's API key in `.env`: `python -m src.evaluate check --config configs/text/claude-sonnet-5-5.toml`, then `python -m src.evaluate run --config configs/text/claude-sonnet-5-5.toml --split test`. The other datasets and models have their own files in `configs/`.

## Results

| Dataset | Model | Setting | Balanced accuracy (test) | 95% interval | Hallucination rate (wrong diagnoses) | Missed malignancy (cancers called benign) | Report |
|---|---|---|---|---|---|---|---|
