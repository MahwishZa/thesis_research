# Thesis Title (Temporary Placeholder)

## Overview

This thesis studies retrieval-augmented generation (RAG), in which a language model looks up scientific studies before answering, for medical question answering in Alzheimer's disease and related dementias, building on RAG² (Sohn et al., NAACL 2025). The research is being rebuilt step by step; the previous implementation and its experiments are kept for reference in `_archive/`.

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant baseline models and existing works.

## Results

## Repository Structure

```
.
├── README.md
├── pyproject.toml   packaging; dependencies are added when a step needs them
├── docs/            research documentation; docs/log.md is the dated research log
├── src/medrag/      the pipeline parts, added one step at a time
├── experiments/     one numbered folder per experiment (see experiments/README.md)
├── tests/           automatic checks for src/medrag/
└── _archive/        the previous implementation and its experiments (reference only)
```

## How to Run

Run from the repository root:

```bash
pip install -e .
python -m unittest discover -s tests -t .
```

The pipeline itself is being rebuilt step by step; there is nothing else to run yet.
