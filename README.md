# thesis_research — Alzheimer's evidence corpus

Data-preparation repository for an MS thesis on **recency bias in confidence-derived
evidence-utility signals for retrieval-augmented Alzheimer's clinical reasoning**,
extending the RAG² framework (Sohn et al., NAACL 2025).

It holds two things, kept deliberately apart: the **retrieval corpus** (`pmc/`,
`pubmed/`) and the **reproduced original RAG² baseline** (`rag2/`). The thesis
experiments themselves (`experiments/`) are not written yet.

## Pipeline stages

| Stage | Directory | What it produces |
| --- | --- | --- |
| 1. PubMed acquisition | `pubmed/` | 43,409 records from the approved query bank |
| 2. PMC open-access inventory | `pmc/` | 27,508 candidates, licence + availability |
| 3. PMC full-text download | `pmc/fulltext/` | 25,742 MD5-verified JATS XML + manifest |
| 4. XML parsing | `pmc/parsed/` | One structured JSON record per article |
| 5. Quality control | `pmc/` (reports) | Full-corpus QC reports |
| 6. Corpus policy metadata (M1–M4) | `pmc/metadata/`, `pmc/currency_pack/` | Dates, eligibility, CPG layer, currency pack |
| 7. Retrieval-ready chunks | `pmc/chunks/` | Deterministic chunks with full provenance |
| 8. MedCPT index + retrieval | `pmc/index/` | Exact inner-product search, candidate replay |
| 9. Original RAG² baseline | `rag2/` | Rationale → balanced retrieval → filter → answer |

## Layout

```
pubmed/                    PubMed acquisition
  fetch_pubmed.py            E-utilities pipeline
  search_queries.txt         approved query bank (read-only input)
  pubmed_results.csv/.json   retrieved records (Git LFS)
  search_log.csv             per-query audit log

pmc/                       PMC acquisition, parsing, QC, corpus policy
  inventory_pmc_oa.py        builds the OA inventory
  download_pmc_xml.py        MD5-verified XML downloader
  parse_pmc_xml.py           JATS -> structured records
  qc_investigate.py          independent QC detector
  build_corpus_metadata.py   M1-M4 corpus policy metadata
  test_*.py                  test suites (see below)
  pmc_oa_inventory*.csv      inventory + reconciliation snapshot
  pmc_qc_report_*.md         generated QC reports
  fulltext/                  manifest.csv, failures.csv  (xml/ is gitignored)
  parsed/                    parsed records (gitignored, regenerable)
  metadata/                  M1-M4 overlays + registries  (see its README)
  currency_pack/             externally ingested currency-pack documents
  build_chunks.py            retrieval-ready chunk layer
  validate_chunks.py         chunk/provenance integrity gate
  chunks/                    chunk_stats.json committed; chunks.jsonl gitignored
  embed_chunks.py            MedCPT index builder  (pmc/index/ is gitignored)
  retrieve.py                exact search + candidate replay

rag2/                      reproduced ORIGINAL RAG2 baseline  (see below)
  rag2/                      the reproduction package
  configs/                   experiment configs, incl. thesis_corpus.yaml
  scripts/                   one CLI per stage + smoke test
  tests/                     baseline test suite
  docs/                      the reproduction's own specification + results
  retriever/, classifier/    the RAG2 authors' released code, UNMODIFIED

experiments/               the three-layer separation  (see experiments/README.md)
  baseline/                  original RAG2 runs
  recency_bias/              thesis probe        (not started)
  scaf/                      SCAF extension      (not started)

docs/
  rag2_reproduction_audit.md  what was reproduced, verified, and left unverified
```

## Chunking (stage 7)

Strategy is taken from the thesis proposal (§5.1), not invented: **256-token
sliding windows with 32-token overlap**, sized against the article encoder's
512-token limit with headroom for a prepended title and section header, plus
exact content-hash deduplication.

Two implementation decisions follow from that text:

- **Windows never cross a section boundary.** The proposal requires that a
  recommendation is never separated from its qualifying conditions; windowing
  inside sections also keeps section provenance exact for every chunk.
- **Windows are measured in whitespace words** — deterministic and dependency
  free (this repository is standard-library only). A 256-word window is always
  fewer than 512 sub-word tokens, so it stays inside the encoder limit with
  headroom. `--window/--overlap` can later be set in sub-word tokens without
  changing any other logic.

Title and section heading are stored as separate fields, not baked into the
text; `build_chunks.compose_embed_text()` is the single shared rule for
composing what the encoder sees.

Frozen policy is enforced, never re-decided: records whose M4
`eligibility_status` is `excluded` are not chunked; everything else carries its
frozen status through so retrieval can filter. Exact-duplicate text is
**flagged** via `duplicate_of`, never deleted — distinct versions and source
types must survive, because recency is an experimental variable.

```bash
python3 pmc/build_chunks.py        # writes pmc/chunks/{chunks.jsonl,chunk_stats.json}
python3 pmc/validate_chunks.py     # integrity gate; exits non-zero on failure
```

Both are deterministic: the same frozen inputs produce byte-identical output.
`validate_chunks.py` prints a content digest for cross-run comparison.

## Retrieval infrastructure (stage 8)

Models are fixed by the base paper and the proposal (§5.3), not chosen for
convenience — the retriever is deliberately frozen, and that is the thesis's
central internal-validity guarantee:

| Role | Model |
| --- | --- |
| Document encoder | `ncbi/MedCPT-Article-Encoder` |
| Query encoder | `ncbi/MedCPT-Query-Encoder` |
| Reranker | `ncbi/MedCPT-Cross-Encoder` |

**Exact flat search, not ANN.** Validity control V3 requires the candidate set
to be replayed byte-identically to every experimental arm; an approximate index
introduces run-to-run variation. Search is an exact inner-product scan over
unit-normalized vectors, and ties break on `chunk_id` so ordering is total.
FAISS/numpy are used when present purely for speed and give identical results.

**Balanced retrieval** draws an equal quota per `source_category` before
merging (base paper §3.4). Without it a PubMed-trained dense retriever drowns
the small but decisive CPG and currency-pack corpora.

**Candidate-set replay (V3).** `save_candidates()` serialises the candidate list
with a digest over identity *and order*; `replay_candidates()` reloads and
verifies it; `verify_replay()` proves a later arm scored the same population.

### Installing torch with CUDA

**`pip install torch` gives you the CPU-only build on Windows.** That build makes
`torch.cuda.is_available()` return `False`, and the embedding run then executes
on the CPU — for ~780k chunks that is the difference between hours and days. The
GPU build must be installed from the PyTorch CUDA index explicitly:

```bash
pip uninstall -y torch
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers                  # unchanged; numpy optional, for speed

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The CUDA runtime ships inside the wheel — no system CUDA toolkit is needed. Pick
the `cuXXX` suffix your driver supports; a driver new enough for CUDA 12.1 or
later runs the `cu121` build, and newer drivers stay backward compatible.

### Building the index

```bash
python3 pmc/embed_chunks.py --device cuda --batch-size 8   # -> pmc/index/
python3 pmc/verify_index.py                                # integrity gate
python3 pmc/retrieve.py --query "..." --query-id q1
python3 pmc/retrieve.py --replay pmc/candidates/q1.json
```

Three properties matter for a run this long:

- **`--device cuda` is a requirement, not a preference.** It fails with an
  actionable message rather than silently falling back to the CPU. `--device
  auto` keeps the old fallback but says which device it chose.
- **The run resumes by default.** Output is flushed every batch, and a restart
  picks up at the last complete row — a partial vector or half-written manifest
  line is trimmed first. A resumed index is byte-identical to an uninterrupted
  one, `content_digest` included. `--restart` starts over.
- **CUDA OOM halves the batch and continues**, and stays reduced. It never falls
  back to the CPU mid-run, which would make the index internally inconsistent.

`--batch-size 8` is sized for a 4 GB card at 512 tokens; raise it on a larger
GPU. `--limit N` embeds only the first N chunks for a smoke test and stamps the
result `partial_index_limit` so it cannot be mistaken for the production index.

`verify_index.py` is the gate to run before anything retrieves: it checks row
count against the chunk layer, dimension, row-for-row alignment, absence of
NaN/Inf, L2 normalisation, and that `content_digest` recomputes to the recorded
value.

A deterministic stub encoder exists for offline testing only. It refuses to
write an index without `--allow-stub` and stamps `production=false`, so a stub
index can never be mistaken for a real one.

## Original RAG² baseline (stage 9)

`rag2/` holds a reproduction of the **original** RAG² system — the thesis
baseline, not a thesis contribution. Read
**[`docs/rag2_reproduction_audit.md`](docs/rag2_reproduction_audit.md)** first:
it records, component by component, what was reproduced, how it was verified,
and what remains unverified. The reproduction's own specification (every
assumption, every place the paper and the authors' code disagree) is
[`rag2/docs/rag2_reproduction.md`](rag2/docs/rag2_reproduction.md).

Inside `rag2/`, two things are kept apart on purpose:

| Path | What it is |
| --- | --- |
| `rag2/retriever/`, `rag2/classifier/` | the RAG² authors' released code, **unmodified** |
| `rag2/rag2/`, `configs/`, `scripts/`, `tests/` | the reproduction |

**Models required** (none are downloaded by the tests or the smoke test):

| Role | Model | Needed for |
| --- | --- | --- |
| Query encoder | `ncbi/MedCPT-Query-Encoder` | retrieval |
| Article encoder | `ncbi/MedCPT-Article-Encoder` | building `pmc/index/` |
| Reranker | `ncbi/MedCPT-Cross-Encoder` | reranking |
| Filter | `google/flan-t5-large` + a trained checkpoint | filtering |
| Backbone LLM | `meta-llama/Meta-Llama-3-8B-Instruct` | rationales, answers |

The paper's trained filter checkpoint **is not distributed by its authors** and
must be retrained (`rag2/scripts/03_build_filter_labels.py`, then `04_train_filter.py`).

### Running it

```bash
cd rag2
pip install -r requirements.txt
python3 scripts/smoke_test.py     # offline wiring check: no GPU, no downloads, ~2s
python3 -m pytest                 # baseline test suite
```

Configure it against this repository's corpus with
`rag2/configs/thesis_corpus.yaml`, which points the baseline at `pmc/index/` and
`pmc/chunks/chunks.jsonl` through the `thesis_chunks` corpus loader. Run the
stage scripts from the repository root so those relative paths resolve. Stage
commands are in [`experiments/baseline/README.md`](experiments/baseline/README.md).

### What is verified, and what is not

**Verified here:** all stages wired end to end (the smoke test executes
question → rationale → balanced retrieval → rerank → cache → ΔPPL labeling →
filter → answer → evaluation); the filter prompt and option format round-trip
byte-identically against the authors' released training data; Figure 2's
labeling tree matches the paper path by path; provenance never reaches a model
input.

**Not verified:** anything requiring model weights. This container has no torch,
transformers, faiss or GPU, so MedCPT encoding, Flan-T5 filtering, ΔPPL under a
real LLM and answer generation were checked by code reading and stub-driven
execution, not by running a real model. **No accuracy has been measured** —
`rag2/docs/reproduction_results.md` is deliberately blank. See audit §10.

## Where thesis experiments belong

`experiments/` establishes a one-directional boundary: a layer may call the layer
below and may not modify it.

1. `experiments/baseline/` — original RAG² runs. Runnable now.
2. `experiments/recency_bias/` — the thesis probe. **Not started.**
3. `experiments/scaf/` — the SCAF extension. **Not started.**

Nothing in this repository implements SCAF, recency weighting, authority
weighting, currency scoring, supersession or abstention. The baseline measures
the *untouched* original, so `rag2/tests/test_metadata_isolation.py` fails the
build if any baseline module starts reading publication dates.

## Running the tests

Tests import their module by name, so run them from inside the package directory:

```bash
cd pmc    && python3 -m unittest test_parse_pmc_xml test_qc_investigate \
                                 test_download_pmc_xml test_inventory_pmc_oa \
                                 test_build_corpus_metadata test_build_chunks \
                                 test_retrieval test_verify_index
cd pubmed && python3 -m unittest test_fetch_pubmed test_pipeline_integration
cd rag2   && python3 -m pytest
```

All suites are offline — no network, no corpus files, no model weights required.
Last measured: **355 passed** (`pmc/`), **51 passed** (`pubmed/`),
**194 passed, 2 skipped** (`rag2/`; both skips are torch-gated modules).

## Regenerating derived data

Raw XML (`pmc/fulltext/xml/`) and parsed records (`pmc/parsed/`) are gitignored: they are
research data, reconstructible from the inventory plus the manifest's MD5s. The corpus
policy overlays are committed as research evidence and are regenerated with:

```bash
python3 pmc/build_corpus_metadata.py --no-fetch      # Linux/macOS
python  pmc\build_corpus_metadata.py --no-fetch      # Windows
```

Run this on the machine holding the **complete** parsed corpus: it upgrades canonical dates
from PubMed fallback to JATS-primary for all PMC records. See `pmc/metadata/README.md`.

## Current status

Acquisition, parsing, QC and corpus-policy metadata (M1–M4) are complete. The chunk layer
and the retrieval stack are built. The original RAG² baseline is integrated and audited.

Outstanding, in order:

1. Full-corpus canonical-date regeneration (above), on the machine with the complete
   parsed corpus. `pmc/chunks/chunk_stats.json` should be refreshed from that run — the
   committed copy records a partial container run, not the production one.
2. Build the MedCPT index on the GPU machine:
   `python pmc\embed_chunks.py --device cuda --batch-size 8`, then
   `python pmc\verify_index.py`. Resumable — re-run the same command after any
   interruption. Install the CUDA torch build first (see stage 8); a `+cpu`
   build silently runs this on the CPU.
3. Train the RAG² filter, then run the baseline and record results in
   `rag2/docs/reproduction_results.md`. **No accuracy has been measured yet.**
4. Only then: the recency-bias probe.

## Provenance notes

- `pmc/pmc_oa_inventory.csv` is immutable — the authoritative acquisition record.
- `pmc/currency_pack/xml/PMC13082890.xml` is an exact pinned snapshot (MD5
  `dcb1ac4eaa24b75ab3202f2315c6b2e4`). The PMC object is revised in place, so re-fetching
  yields a different hash; this snapshot must not be replaced. Licensed CC BY-NC —
  redistribution is non-commercial, with attribution to the Cochrane review it contains.
- Document identity is **PMID-primary**, PMCID for the full-text join; DOI is a consistency
  check only, never an identity key.
