# Improving RAG for Medical Question Answering

## Overview

Existing medical RAG methods retrieve text passages by their similarity to a text question and are judged by answer accuracy alone. This is insufficient for diagnostic question answering on skin lesions, where the question is a photo, a case description or both, because the retrieval ignores the image and no answer is checked against its evidence, so models still give confident wrong diagnoses and unsupported claims. My research will investigate whether multimodal, rationale-guided retrieval with evidence-grounded answers reduces these hallucinations, using closed-book and open-book experiments on image, text and multimodal datasets.

### Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.

## Research Update

**Status (2026-11-19).** Version 2 is in progress. The datasets are prepared but not frozen: they are being finalised with my supervisor.
The closed-book pipeline (a model answers from the question alone) is built, tested and checked without running any model. **No Version 2
experiment has been run, so there are no results yet.** Nothing below is a final result.

- **Data.** Two datasets, each asked in three ways (image only, text only, image and text): the clinic patients (PAD-UFES-20: 2,269 phone photos
  with each visit's record, 6 diagnoses) and the published cases (716 case reports in 8 diagnoses, 324 with the patient's photos from the
  article), and a knowledge base of 5,982 items for the open-book runs.
- **Pipeline.** Built: the closed-book approach, the answer reader, the scores, and the checks that need no model (`python -m src.validate`).
  Not built yet: standard RAG, RAG² and the proposed method, and the metrics that check an answer against its evidence.
- **Earlier work.** Version 1 (closed-book runs on other datasets) is kept in `archive/v1/` for reference only and is not reported as a result.
- **Details.** Progress: `docs/progress.md`. Audit and open decisions: `docs/audit.md`. Code, structure and how to run: `docs/code_guide.md`.
  Method: `docs/method.md`.
