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
├── pyproject.toml   packaging; the packages for the model and the encoder are optional extras
├── docs/            research documentation (see docs/README.md); docs/log.md is the dated research log
├── src/             the pipeline parts, one folder each: rationale_query/ is Part 1 of RAG²; the others follow, one step at a time
├── experiments/     one numbered folder per experiment (see experiments/README.md)
├── tests/           automatic checks, in folders by purpose (see tests/README.md)
└── _archive/        the previous implementation and its experiments (reference only)
```

## How to Run

Run from the repository root:

```bash
pip install -e .
python -m unittest discover -s tests -t .
```

The tests that run the real llama.cpp on a tiny stand-in model need one more package (`pip install -e ".[test]"`); without
it those tests are skipped and say so.

Part 1 of RAG² is built: a language model writes a step-by-step rationale for a question, and the rationale alone becomes the
search query. It needs the model files in `models/` and the packages `llama-cpp-python`, `numpy`, `torch` and `transformers`
(`pip install -e ".[rationale]"`). The commands, the settings and what the comparison of writers shows are in
[`experiments/01_rationale_writers/NOTES.md`](experiments/01_rationale_writers/NOTES.md); `python -m rationale_query --help` lists
the tool's commands. Parts 2 and 3 are not built yet.
