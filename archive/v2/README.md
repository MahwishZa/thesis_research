# Improving RAG for Medical Question Answering

## Overview

Existing medical RAG methods retrieve text passages by their similarity to a text question and are judged by answer accuracy alone. This is insufficient for diagnostic question answering on skin lesions, where the question is a photo, a case description or both, because the retrieval ignores the image and no answer is checked against its evidence, so models still give confident wrong diagnoses and unsupported claims. My research will investigate whether multimodal, rationale-guided retrieval with evidence-grounded answers reduces these hallucinations, using closed-book and open-book experiments on image, text and multimodal datasets.

### Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant models and existing works.
