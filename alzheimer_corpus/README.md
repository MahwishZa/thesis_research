# alzheimer_corpus

An **Alzheimer's disease** evidence corpus for the MS thesis on recency bias in
retrieval-augmented clinical reasoning. Config, queries, data, metadata, logs, scripts and
reports live together so the corpus can be inspected, reproduced and moved as one component.

> **Status: pipeline implemented and tested offline. No live retrieval performed.**
> NCBI egress is blocked by organisation policy in this environment (`403` on
> `eutils.ncbi.nlm.nih.gov:443`). Nothing here reports retrieval statistics.

## It is an Alzheimer's corpus, not a dementia corpus

**Alzheimer Disease is the single anchor concept.** Dementia, cognitive dysfunction, memory
impairment, neurodegeneration, amyloid/tau pathology and comparator diseases are **supporting**
concepts: each can qualify a record only when an Alzheimer's anchor is also present, never alone.

Enforced twice — every query family AND-s the anchor block, and the relevance gate re-checks it
per record with a full decision trace. Verified on the fixture run:

| Case | Verdict |
|---|---|
| Atopic dermatitis using the token "AD" | **exclude** — `ambiguous_abbreviation_only` |
| Amyloid/tau with no AD anchor | **exclude** — `no_ad_evidence` |
| Vascular dementia with no AD anchor | **exclude** — `comparator_disease_no_ad_anchor` |
| Generic dementia prevalence study | **exclude** — `generic_dementia_no_ad_anchor` |
| Alzheimer's vs Lewy body disease | **include** — differential diagnosis is in scope |

## Pipeline

```
01_pubmed_download → 02_pmc_download → 03_guidelines
        ↓
raw → 04_normalize → normalized → 05_deduplicate → deduplicated → 06_chunk → chunks
                                                                        ↓
                                                          07_claim_classification
```

Deduplication is document-level and runs **before** chunking, so expensive tokenization is never
spent on duplicates. The fixture run demonstrates it catching a preprint/journal pair by
`title_year` — the exact case that would otherwise let an old claim survive under a new date.

## Key design rules

- **Tokenizer, not whitespace.** 256 tokens / 32 overlap / 224 stride against the MedCPT article
  encoder's 512-token limit. `--tokenizer whitespace` exists only as an explicit fallback and
  stamps `tokenizer_used` on every chunk so it can never be mistaken for the specification.
- **Licensing fails closed.** Unknown licence ⇒ `redistribution_allowed=False` ⇒ text not
  committed. Textbooks are metadata only.
- **Surface forms preserved verbatim**: `Aβ42`, `Aβ40`, `p-tau181`, `p-tau217`, `ARIA-E`,
  `ARIA-H`, `APOE ε4`. Case-folding is used for matching only and never written back.
- **Nothing deleted silently.** Excluded records keep their reason; duplicates are recorded, not
  removed; retracted documents stay retrievable and flagged.
- **Source tiers carry no ordering** — authority is a tested variable in the thesis (ablation A12).
- **Classification is not a delete filter** — it drives ranking, down-weighting and review.

## Run it

Offline, works anywhere:
```bash
python3 alzheimer_corpus/scripts/01_pubmed_download.py --print-queries   # no network
python3 alzheimer_corpus/scripts/04_normalize.py
python3 alzheimer_corpus/scripts/05_deduplicate.py
python3 alzheimer_corpus/scripts/06_chunk.py --tokenizer whitespace
python3 alzheimer_corpus/scripts/07_claim_classification.py --sample 300
```

Where NCBI is reachable — always count-only first:
```bash
python3 alzheimer_corpus/scripts/01_pubmed_download.py --dry-run
python3 alzheimer_corpus/scripts/01_pubmed_download.py --query Q01_alzheimer_core --limit 100
```

## Open decisions

- **Publication window.** `config/search_queries.yaml` inherits a five-year window (2021–2026)
  from the executed strategy. The proposal states no date restriction. Measured consequence: over
  that span at a five-year half-life the currency term never falls below 0.536 — minimal dynamic
  range for the very mechanism this thesis studies. **Highest-impact open decision.**
- **Taxonomy keywords are empty.** 50 classes are defined; populating keywords is gated on the
  granularity decision (group vs leaf), which determines whether the contested state can fire.
- **Supersession** is not yet populated, so no supersession or contested state can be exercised.
