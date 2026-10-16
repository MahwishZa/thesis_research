# src/

The code of the pipeline. Each part of RAG² (Sohn et al., NAACL 2025) gets its own package here, built one step at a time.

| Package | Part of RAG² | State |
|---|---|---|
| `rationale_query/` | Part 1: rationale-based query formulation (section 3.3) | built |
| (not created yet) | Part 2: retrieval and filtering | not built |
| (not created yet) | Part 3: answer generation | not built |

## What is in `rationale_query/`

A language model writes a step-by-step rationale for a question. The rationale alone, without the question, is the text that the
MedCPT query encoder turns into the search query for Part 2.

| File | What it does |
|---|---|
| `prompt.py` | the paper's prompt text and how a question is put into it |
| `search_text.py` | which text is searched with (the rationale) and which text the re-ranker reads |
| `generator.py` | the language model behind one interface (llama.cpp), with every setting recorded |
| `store.py` | reading the questions file; one resumable record per question on disk |
| `batch.py` | the loop that turns questions into rationale records, and can be stopped and started again |
| `encoder.py` | the MedCPT query encoder (first-token vector, 512 tokens) |
| `diagnostics.py` | what the rationales look like: length, cut-off, final option; no labels needed |
| `cli.py` | the commands of `python -m rationale_query` |
| `__main__.py` | lets `python -m rationale_query` start `cli.py` |
| `__init__.py` | the package description; it must list every module (a test checks this) |

Importing the package loads nothing heavy: `llama_cpp`, `torch`, `transformers` and `numpy` are imported only in the places that use them.

## How to use it

From the repository root, after `pip install -e .`:

```powershell
python -m rationale_query --help
python -m rationale_query show-prompt --question "Which drug is a cholinesterase inhibitor? A) Donepezil B) Aspirin C) Insulin D) Warfarin"
```

The full run, with its settings and the comparison of writers, is `experiments/01_rationale_writers/` (see its `NOTES.md`).
The requirements the code follows are in `docs/methodology.md`; the tests are described in `tests/README.md`.

## Adding a new part

Make a new folder `src/<name>/` with a docstring in every module and one line per module in the package docstring, add a test
file for every module under `tests/`, and add a row to the table above.
