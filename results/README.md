# Results folder: guide

| | |
|---|---|
| Document status | Current as of 2026-11-24. |
| Purpose | Tells a reader what is in `results/`, which document to read first, how a run is stored, and what the terms mean. |
| Rule | A number appears in a document only if it can be read from a committed file in this folder or recomputed from one. Marks: ✔ checked, ◻ assumed. |

## 1. Which document to read

| Question | Document |
|---|---|
| Where does the research stand now? | `STATUS.md` |
| What are the results so far, and how should they be read? | `llm_only_text_only_results.md` (LLM-only, text-only view, test split) |
| What does each metric measure, and why was it chosen? | `evaluation_guide.md` |
| What does the literature say about RAG evaluation? | `metric_survey.md` |
| Was the repository checked, and what was found? | `repository_audit.md` (a dated snapshot) |
| What was decided or changed, and when? | `CHANGELOG.md` |
| How is the data built and described? | `../data/datasheet.md` |
| How is the code run and tested? | `../src/methodology.md` |

Suggested reading order for the supervisor: `STATUS.md`, then `llm_only_text_only_results.md`, then `evaluation_guide.md`.

## 2. Folder layout

```
results/
  README.md                       this guide
  STATUS.md                       the current state
  CHANGELOG.md                    the history of decisions and changes
  evaluation_guide.md             the metrics, in plain English
  metric_survey.md                the literature survey (snapshot 2026-11-23)
  repository_audit.md             the audit (snapshot 2026-11-24)
  llm_only_text_only_results.md   results and their interpretation
  summary.csv                     one row per finished run (written by `evaluate.py collect`)
  similarity_check.json           the semantic-similarity validity check (when it has been run)
  feature_check/                  the shortcut analysis of the data (`check.py features`)
  llm_only/
    <model>/<view>_<split>/       one finished run: predictions.jsonl, run.json, report.json
    compare_<view>_<split>.json   McNemar tests between models (`evaluate.py compare`)
```

Each system gets a folder of its own beside `llm_only/` when it is built (for example `standard_rag/`), with the same layout, so `evaluate.py collect` finds every finished run.

## 3. What is kept in Git, and what is not

| Kept | Not kept (ignored on purpose) |
|---|---|
| finished test runs (`predictions.jsonl`, `run.json`, `report.json`), `summary.csv`, the comparison files, the documents above | unfinished and trial runs (`*.partial/`), validation runs (`*_val/`), trial reports (`trial_*.json`), run logs (`*.log`) |

Validation and trial runs check the format and the cost; they tune nothing and are never reported as results.

## 4. How one run is stored

`llm_only/<model>/<view>_<split>/` holds, for a finished run:

- `predictions.jsonl`: one line per question (the answer letter, the reasoning, the raw reply, tokens, seconds, which host answered);
- `run.json`: the model settings, the exact prompt, the SHA-256 of the question file and the status;
- `report.json`: every metric of the run, written by `python src/evaluate.py collect`.

Two runs can be compared only if their `questions_sha256` in `run.json` is the same.

## 5. Glossary

| Term | Meaning here |
|---|---|
| Case | One skin-lesion image with its label and the patient's age, sex and body site |
| View | The inputs a model gets for a case: image only, text only (age, sex, site) or both |
| Split | train (not used by the LLM runs), validation (160 cases, for trials) or test (240 cases, the final runs) |
| System | LLM-only (no retrieval), standard RAG, RAG² (the paper's method) or the proposed RAG |
| RAG | Retrieval-augmented generation: passages are retrieved from a text library and given to the model with the question |
| Run | One model, one system, one view, one split |
| Passage | A short piece of medical text (30 to 200 words) in the knowledge corpus |
| Invalid answer | An answer with no usable option letter; it counts as wrong |
| Refusal | An invalid answer where the model declined to answer; reported apart |
| Macro-F1 | The mean of the eight per-class F1 scores, so every class counts the same |
| McNemar / Holm | The paired test between two systems on the same questions, and the correction for many pairs |
| 95% interval | The range the true score probably lies in, from 1,000 resamples of whole lesions |
| Chance | 12.5% accuracy (1 of 8 options) on the balanced splits |
