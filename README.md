# Improving RAG for Medical Question Answering

## Overview

This thesis studies retrieval-augmented generation (RAG), in which a language model looks up scientific studies before answering, for medical question answering, building on RAG² (Sohn et al., NAACL 2025). The first goal is a faithful, smaller-scale reproduction of the RAG² baseline on MedQA, MedMCQA and MMLU-Med; the scope of the retrieval corpora is still to be decided with the supervisor. The research is being rebuilt step by step; the previous implementation and its experiments are kept for reference in `_archive/`.

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant baseline models and existing works.

## Results

No results yet. The baseline (a faithful, smaller-scale reproduction of RAG²) is built but has not been run on the real models and corpora.

## Repository Structure

```
thesis_research/
├── README.md
├── pyproject.toml   packaging; the packages for the model and the encoder are optional extras
├── docs/            research documentation (see docs/README.md)
├── src/             the pipeline parts, one folder each: rationale_query/ is Part 1 of RAG² and balanced_retrieval/ is Part 2; the others follow, one step at a time (see src/README.md)
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
the tool's commands.

Part 2 of RAG² is built: the rationale is searched in four corpora, the best passages of each are pooled, and one cross-encoder orders the
pool. The corpora themselves (PubMed and PMC for Alzheimer's disease and its neighbours; the guidelines and textbooks of the paper) are
built with `python -m balanced_retrieval fetch`, `chunk` and `embed` (about 10 million passages; built and checked on 2026-10-20) and are not in the repository; the commands are in [`src/README.md`](src/README.md)
(`pip install -e ".[retrieval]"`).

Part 3 of RAG² is built: a small Flan-T5 filter judges each retrieved snippet as helpful or not helpful. `python -m rationale_filter` makes the
training labels (`label`, `labels`), prepares and trains the filter (`prepare-model`, `train`), tests whether the training fits in memory
(`memory-test`) and scores a trained filter (`evaluate`); it needs `pip install -e ".[filter]"` for the training. Part 4 is built: `python -m answer_evaluation`
answers the questions with each system (no retrieval, retrieval without the filter, retrieval with the filter) and reports accuracy and the
paired comparison. `python -m data_roles` reads the three question sets (MedQA, MedMCQA, MMLU-Med) and fixes which question is used for
training, validation or test. All of it is tested on stand-in models; it has not yet been run end to end on the real Llama-3 file and the
real corpora, so the baseline has no result yet. The commands are in [`src/README.md`](src/README.md), the decisions and what is still open
in [`docs/methodology.md`](docs/methodology.md).
