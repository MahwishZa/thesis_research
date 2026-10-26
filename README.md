# Improving RAG for Medical Question Answering

## Overview

Existing passage filters for medical retrieval-augmented generation, such as RAG² (Sohn et al., NAACL 2025), learn which retrieved passages to keep from automatic labels: a passage counts as helpful if adding it to the prompt turns the language model's wrong answer into a correct one, or keeps the answer correct while making the model more confident (a large enough drop in perplexity). These labels are insufficient when the model's answer or confidence shifts whenever any text is added to its prompt, because each passage is compared only with answering without a passage, so a change that an irrelevant passage would also cause is credited to the passage's content. My research will investigate whether control-adjusted labels, which also measure the effect of an irrelevant control passage, address that limitation, using two experiments on MedQA: (A) comparing how often the labelling marks retrieved and irrelevant passages as helpful, and (B) comparing answer accuracy and harmful retrieval effects (correct answers made wrong by retrieval) across five systems: no retrieval, retrieval without filtering, the original RAG²-style filter, the control-adjusted filter, and an equal-budget baseline that keeps the same number of top-ranked passages as the control-adjusted filter.

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
