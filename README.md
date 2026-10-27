# Improving RAG for Medical Question Answering

## Overview

Existing methods do X. This is insufficient under condition Y because of limitation Z. My research will investigate whether approach W addresses that limitation, using experiments A and B.

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
