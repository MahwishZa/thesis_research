# thesis_research — Alzheimer's evidence corpus

Data-preparation repository for an MS thesis on **recency bias in confidence-derived
evidence-utility signals for retrieval-augmented Alzheimer's clinical reasoning**,
extending the RAG² framework (Sohn et al., NAACL 2025).

This repository builds and documents the **retrieval corpus**. It does not contain the
retrieval system, embeddings, or thesis experiments.

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

**Not yet built (next phase):** embeddings, FAISS/vector index, retrieval and
reranking. Chunk and provenance validation comes first, and the retrieval stack
must be frozen and replayed byte-identically across experimental arms — that
belongs to the retrieval phase, not corpus preparation.

## Running the tests

Tests import their module by name, so run them from inside the package directory:

```bash
cd pmc    && python3 -m unittest test_parse_pmc_xml test_qc_investigate \
                                 test_download_pmc_xml test_inventory_pmc_oa \
                                 test_build_corpus_metadata
cd pubmed && python3 -m unittest test_fetch_pubmed test_pipeline_integration
```

All suites are offline — no network, no corpus files required.

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

Acquisition, parsing, QC and corpus-policy metadata (M1–M4) are complete. One step remains
before the corpus is frozen: the full-corpus canonical-date regeneration above.

## Provenance notes

- `pmc/pmc_oa_inventory.csv` is immutable — the authoritative acquisition record.
- `pmc/currency_pack/xml/PMC13082890.xml` is an exact pinned snapshot (MD5
  `dcb1ac4eaa24b75ab3202f2315c6b2e4`). The PMC object is revised in place, so re-fetching
  yields a different hash; this snapshot must not be replaced. Licensed CC BY-NC —
  redistribution is non-commercial, with attribution to the Cochrane review it contains.
- Document identity is **PMID-primary**, PMCID for the full-text join; DOI is a consistency
  check only, never an identity key.
