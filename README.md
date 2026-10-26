# Improving RAG for Medical Question Answering

## Overview

Existing medical RAG filters, such as RAG² (Sohn et al., 2025), learn which retrieved passages to keep from labels that compare the model's answer and confidence with and without each passage. This is insufficient when adding any text changes the model's answer, because the labels then credit a passage for changes that an irrelevant passage would also cause. My research will investigate whether labels adjusted with irrelevant control passages address this, using two experiments on MedQA: (A) checking how often the labels call irrelevant passages helpful, and (B) comparing accuracy and harm across no retrieval, unfiltered retrieval, the original filter, the adjusted filter and an equal-budget baseline.

## Research Objectives

## Results

### Earlier study (temporary)

*Temporary. These are the results of an earlier study, copied unchanged from Git history so that they can be compared with this thesis. They answer a different question (Cochrane-review verdicts, an Alzheimer's-focused evidence check) and do not use MedQA or test the RAG² labels. Source: `README.md` at commit `63388ec` (13 October 2026); the paragraph on the pre-registered test is from commit `3eb5803` (12 October 2026). The files named in the text (`docs/protocol.md`, `docs/evaluation.md`) are no longer in the repository; they are in Git history.*

**What was done.** The proposed system was built and run end to end, with the baseline and four control systems, under one frozen design, on Cochrane-review questions asked as of each review's date (only earlier studies can be used). The **primary set** is the research domain: 208 questions on Alzheimer's disease and related dementias, of which 48 name Alzheimer's disease (23%), 156 name dementia (15 name both) and 19 name neither; it is not an Alzheimer's-only evaluation. The **secondary set** is general medicine: 528 held-out questions (226 more were used while the study was built) and, for a pre-registered test, 850 new questions. Presenting the Alzheimer's disease set as primary was decided after all results were known: the protocol had named the held-out general-medicine questions as the primary test (`docs/protocol.md` §8), and one rule reads a difference on every set. Accuracy is the share of answers that match the verdict of the Cochrane review (*supported*, *refuted*, *not enough information*); differences are in percentage points ("points"). The baseline, adapted RAG², is this study's adaptation of a published medical RAG method (RAG²), not a full reproduction of it.

| System (same language model, same studies) | Primary set: Alzheimer's disease (n = 208) | Secondary set: general medicine (n = 528) |
|---|---|---|
| Always the most frequent verdict (no model) | 42.3% | 44.3% |
| No retrieval | 33.2% | 46.6% |
| Standard retrieval | 30.8% | 48.1% |
| Adapted RAG² (baseline) | 34.6% | 48.7% |
| Proposed system | 38.0% | 50.0% |

| Proposed system minus baseline | Difference (points) | 95% range | p-value |
|---|---|---|---|
| Primary set: Alzheimer's disease | +3.4 | −1.9 to +8.7 | 0.28 |
| Secondary set: general medicine | +1.3 | −1.9 to +4.7 | 0.51 |

**The pre-registered test.** The last test asked the question again on 850 new general-medicine questions, with a measure that gives each of the three verdicts the
same weight (macro-F1). The extra check did not change it: the proposed check minus the baseline was −0.002 (95% range −0.027 to +0.023), which counts as **no evidence**
of an improvement. Accuracy was 45.4% with the extra check and 47.2% for the baseline (−1.8 points, range −4.2 to +0.8, which cannot be told apart from chance). The gains
seen earlier on the other two sets, which led to this test, did not repeat on new questions. Further tables, controls and the earlier approaches that did not work are in
`docs/evaluation.md`.

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
