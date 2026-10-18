# src/

The code of the pipeline. Each part of RAG² (Sohn et al., NAACL 2025) gets its own package here, built one step at a time.

| Package | Part of RAG² | State |
|---|---|---|
| `rationale_query/` | Part 1: rationale-based query formulation (section 3.3) | built |
| `balanced_retrieval/` | Part 2: balanced retrieval over four corpora (section 3.4) | built |
| (not created yet) | Part 3: rationale-guided filtering and answer generation | not built |

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

## What is in `balanced_retrieval/`

The rationale of Part 1 is searched in four corpora (PubMed, PMC, clinical guidelines, textbooks) by exact inner product; the same
number of best passages is taken from each, the lists are joined in a fixed order, and one cross-encoder orders the pool.

| File | What it does |
|---|---|
| `scope.py` | the search string that defines the Alzheimer's-scoped PubMed corpus (and so the PMC corpus) |
| `ncbi.py` | a polite E-utilities client and the date-window search that gets past the 10,000-hit limit |
| `documents.py` | a resumable batch download of documents into one JSON-lines file |
| `pubmed.py` | the PubMed corpus: titles and abstracts of the records in the scope |
| `pmc.py` | the PMC corpus: open-access full text of those records |
| `guidelines.py` | the guidelines corpus: the open guidelines of the Meditron collection |
| `textbooks.py` | the textbooks corpus: the 18 MedQA textbooks, joined back into books |
| `pool.py` | joining the four candidate lists, and ordering a pool by score |
| `retrieve.py` | one question through the four searches, the pool and the re-ranking |
| `diagnostics.py` | each corpus's share of the final lists and the duplicate rate across corpora |
| `chunking.py` | cutting a document into passages of 128 words that overlap by 32 |
| `corpus.py` | one corpus on disk: passages, a row-aligned embedding file, and a manifest of how they were made |
| `embedding.py` | the MedCPT article encoder and the resumable writer of a corpus's embeddings |
| `search.py` | exact inner-product search, block by block, so a corpus need not fit in memory |
| `scorer.py` | the MedCPT cross-encoder that scores a query against each pooled passage |
| `records.py` | one resumable record per question, bound to the settings of the run |
| `cli.py` | the commands of `python -m balanced_retrieval`: `fetch`, `chunk`, `embed`, `precision-check`, `check`, `retrieve`, `report` |

It needs numpy, torch and transformers for the commands that read or write vectors (`pip install -e ".[retrieval]"`); the list
operations, the diagnostics and the chunking need nothing. Keep the corpus folders in `retrieval_corpora\` (git ignores it); the
folder `corpus\` of earlier work is never read by this code.

```powershell
python -m balanced_retrieval fetch pubmed --as-of 2026-10-18 --count-only
python -m balanced_retrieval fetch pubmed --as-of 2026-10-18 --out retrieval_corpora\documents\pubmed.jsonl
python -m balanced_retrieval fetch pmc --pubmed-file retrieval_corpora\documents\pubmed.jsonl --out retrieval_corpora\documents\pmc.jsonl
python -m balanced_retrieval fetch guidelines --out retrieval_corpora\documents\guidelines.jsonl
python -m balanced_retrieval fetch textbooks --out retrieval_corpora\documents\textbooks.jsonl
python -m balanced_retrieval chunk --corpus textbooks --documents retrieval_corpora\documents\textbooks.jsonl --root retrieval_corpora
python -m balanced_retrieval embed --root retrieval_corpora --corpus textbooks
python -m balanced_retrieval check --root retrieval_corpora
python -m balanced_retrieval retrieve --questions questions.jsonl --rationales outputs\my_run\llama3-8b-instruct-q4km --root retrieval_corpora --out retrieved --top-k 32
python -m balanced_retrieval report retrieved
```

`--documents` is a JSON-lines file with one `{"doc_key": ..., "text": ...}` per document; `doc_key` is the canonical identity (PMID,
else PMCID, DOI or normalised title) and may be missing. Every command can be stopped and started again with the same arguments.

## How to use it

From the repository root, after `pip install -e .`:

```powershell
python -m rationale_query --help
python -m rationale_query show-prompt --question "Which drug is a cholinesterase inhibitor? A) Donepezil B) Aspirin C) Insulin D) Warfarin"
```

The full run of Part 1, with its settings and the comparison of writers, is `experiments/01_rationale_writers/` (see its `NOTES.md`).
The requirements the code follows are in `docs/methodology.md`; the tests are described in `tests/README.md`.

## Adding a new part

Make a new folder `src/<name>/` with a docstring in every module and one line per module in the package docstring, add a test
file for every module under `tests/unit/<name>/`, add the package to `LIGHT_MODULES` in `tests/repository/test_layout.py`, and add a row to
the table above.
