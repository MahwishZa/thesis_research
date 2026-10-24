# src/

The code of the pipeline. Each part of RAG² (Sohn et al., NAACL 2025) gets its own package here, built one step at a time.

| Package | Part of RAG² | State |
|---|---|---|
| `rationale_query/` | Part 1: rationale-based query formulation (section 3.3) | built |
| `balanced_retrieval/` | Part 2: balanced retrieval over four corpora (section 3.4) | built |
| `rationale_filter/` | Part 3: rationale-guided filtering (section 3.2) | the label rule, the labelling loop, the training file, the filter decision, the Flan-T5 training, the memory test and the evaluation are built (tested on a tiny random T5 only) |
| `answer_evaluation/` | Part 4: answering and judging the answers | the prompt, the strict reader, the paired statistics, the test-once ledger, the runner, the evidence step and the command line are built |
| `data_roles/` | which question belongs to which role, and the three benchmarks | built |

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

**Building the four corpora (as done on 2026-10-18 to 2026-10-20 on an RTX 2050 laptop).** Run the steps one at a time and read
the last line of each before going on; `chunk` and `fetch pmc` refuse a download that was not finished (`--allow-partial` is for
trials). Repeat `chunk` and `embed` for each corpus; `embed` for PubMed and PMC used `--compute-dtype float16` (decision D11).

| Corpus | Passages | Time to fetch | Time to embed | Vector file |
|---|---|---|---|---|
| textbooks | 133,903 | minutes | about 77 min (32-bit) | 0.4 GB |
| guidelines | 721,364 | minutes | about 6 h (32-bit) | 2.2 GB |
| pubmed | 1,048,208 | about 1 h | about 85 min (16-bit arithmetic) | 3.2 GB |
| pmc | 8,205,229 | a few hours | about 11 h (16-bit arithmetic) | 25.2 GB |

`fetch pubmed` names its snapshot by publication date (`--as-of`). PubMed adds records for old dates later, so searching again
with the same date can find a few more; the identifiers of a download are kept in `<out>.ids.txt` and are what the corpus is. Set
`NCBI_API_KEY` (free) to search at ten requests a second instead of three; NCBI asks for large downloads at night or at weekends
(US time). The service's warnings about the search terms are printed after `fetch pubmed`, because a term it does not know
silently changes what is found. Do not delete `retrieval_corpora\documents\`: fetching it again takes hours.

## What is in `data_roles/`

The three benchmarks (MedQA, MedMCQA, MMLU-Med) as one kind of question record, and which question belongs to which role.

| File | What it does |
|---|---|
| `splits.py` | one role per question id from SHA-256 of the seed and the id, so a question never moves when others are added; a check that no id has two roles |
| `questions.py` | the readers of the three datasets (`cop` of MedMCQA counted from 0, its withheld test split refused by name), a question with its options inline, a manifest of hashes |
| `overlap.py` | questions of one split that repeat questions of another, exact or near (five-word shingles) |
| `cli.py` | `python -m data_roles check`, `export` (questions, answer key and manifest) and `overlap` |

```powershell
python -m data_roles check --dataset medmcqa --hub openlifescienceai/medmcqa --split validation
python -m data_roles check --dataset medmcqa --hub openlifescienceai/medmcqa --split test      # exit status 3: the answers are withheld
python -m data_roles overlap --dataset medmcqa --reference-hub openlifescienceai/medmcqa --reference-split train --candidate-hub openlifescienceai/medmcqa --candidate-split validation
python -m data_roles export --dataset medqa --hub awinml/medqa --config questions --split validation --questions-out q.jsonl --answers-out a.jsonl --manifest-out m.json
```

`--hub` needs the `datasets` package; a local JSON, JSON-lines or Parquet file works with `--file` (Parquet needs pyarrow). The
commands only read; `export` refuses to overwrite.

## What is in `rationale_filter/`

For each question and each of its first N re-ranked snippets, the answer without and with the snippet is read and the rationale
is scored twice; the labels follow from all the records together (Figure 2 of the paper).

| File | What it does |
|---|---|
| `labels.py` | the label rule with both discard branches, the training tokens, and τ as the top 25% of the perplexity differences |
| `inference.py` | the filter input, the decision between the two label tokens, and keeping the helpful snippets |
| `perplexity.py` | perplexity from token log-probabilities; one rationale scored without and with a snippet |
| `scorer.py` | the log-probability of each token of a text under a Llama-3 prompt, with llama.cpp (a second, smaller context) |
| `label_records.py` | one resumable record per pair, bound to its settings; τ and the labels from the records |
| `labelling.py` | the loop over pairs (strict answers first, then the two perplexities) and the reading of the earlier steps' files |
| `training_data.py` | the training file in the format of the released sample, refusing to overwrite |
| `filter_eval.py` | accuracy per class, the majority rate and the AUC on held-out labels |
| `filter_model.py` | Flan-T5 with the two label tokens: first-step logits, windows of 512 tokens with stride 128, batched judging (needs torch) |
| `training.py` | the authors' fine-tuning (AdamW 3e-5, batch 16, 40 epochs), resumable; the memory test; evaluation on labelled files (needs torch) |
| `cli.py` | `python -m rationale_filter label`, `labels`, `prepare-model`, `train`, `memory-test` and `evaluate` |

```powershell
python -m rationale_filter label --questions q.jsonl --answers a.jsonl --rationales outputs\rationales --retrieved retrieved --out pairs --dataset medqa --split train --model models\llama3.gguf --limit 20
python -m rationale_filter labels --questions q.jsonl --answers a.jsonl --rationales outputs\rationales --retrieved retrieved --out pairs --training-file labels\medqa_train.json
```

```powershell
python -m rationale_filter prepare-model --base google/flan-t5-base --out filters\base0
python -m rationale_filter memory-test --model filters\base0 --train-file labels\medqa_train.json
python -m rationale_filter train --model filters\base0 --train-file labels\medqa_train.json --validation-file labels\medqa_val.json --out filters\base
python -m rationale_filter evaluate --model filters\base\best --file labels\medqa_val.json
```

Run `memory-test` first: it does one training step on the longest inputs and says whether batch 16 fits (exit status 3 if not;
then use `--grad-accum`). Flan-T5-large will not fit on a 4 GB card. `train` can be stopped and started again.

`label --limit 20` is the timing pilot: the last line gives the seconds per pair. Run it on training or validation questions;
the command refuses a test split. Every command can be stopped and started again with the same arguments.

## What is in `answer_evaluation/`

| File | What it does |
|---|---|
| `prompt.py` | the answer prompt: the paper's wording with an Evidence section when snippets are kept |
| `reader.py` | the strict reading of "the answer is (X)"; an unreadable answer is wrong for scoring and skipped for labelling |
| `paired_tests.py` | exact McNemar, the paired bootstrap interval and Holm's adjustment |
| `ledger.py` | an append-only record of the test sets that were scored; refuses a second scoring of the same system |
| `records.py` | one resumable record per answer: question, system and k |
| `runner.py` | every question answered by every system, resumable; a test run needs the ledger |
| `results.py` | accuracy per system, the paired comparison, and k chosen on validation data |
| `evidence.py` | the snippets each system reads: the re-ranked list, or only the snippets the filter calls helpful (each judged once) |
| `cli.py` | `python -m answer_evaluation run` and `report` |

```powershell
python -m answer_evaluation run --questions q.jsonl --answers a.jsonl --rationales outputs\rationales --retrieved 4=retrieved4 --retrieved 8=retrieved8 --filter-model filters\base\best --out answers_val --dataset medqa --split validation --model models\llama3.gguf
python -m answer_evaluation report --out answers_val --choose-k rag2 --baseline rag2_no_filter 4 --proposed rag2 4
```

Each k has its own retrieval folder, because the authors' retriever uses one top_k for both the candidates and the final list.
A test split needs `--ledger` and `--system`.

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
