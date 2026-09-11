# alzheimers_corpus

A corpus of **Alzheimer's disease** evidence for the MS thesis on recency bias in
retrieval-augmented clinical reasoning. Self-contained: every corpus-specific config, query,
script, registry, report and document lives in this one folder.

> **Status: scaffold. No corpus data has been retrieved.** Network egress to NCBI is blocked by
> organisation policy in the build environment (`403` on `eutils.ncbi.nlm.nih.gov:443`). See
> [`MANUAL_STEPS.md`](MANUAL_STEPS.md) **M0**. Nothing here reports retrieval statistics.

## It is an Alzheimer's corpus, not a dementia corpus

**Alzheimer Disease is the single anchor concept.** Dementia, cognitive dysfunction, memory
impairment, neurodegeneration, amyloid/tau pathology and comparator diseases are **supporting**
concepts: each can qualify a record only when an Alzheimer's anchor is also present, and none
can qualify one alone.

This is enforced twice — every Q01–Q07 query AND-s the anchor block, and the relevance gate
re-checks it per record with a full decision trace. Four behaviours are pinned by tests:

| Case | Verdict |
|---|---|
| Atopic dermatitis using the token "AD", no long form | **exclude** |
| Amyloid/tau with no Alzheimer's anchor | **exclude** |
| Vascular dementia with no Alzheimer's anchor | **exclude** |
| Alzheimer's vs Lewy body disease | **include** — differential diagnosis is in scope |

## Structure

```
config/     sources · mesh_terms · claim_taxonomy · controlled_vocabulary · corpus_config
queries/    pubmed (Q01-Q07, generated) · pmc · guidelines
data/       raw/{pubmed,pmc,guidelines,textbooks,currency_pack} → parsed → normalized
            → deduplicated → chunks          (gitignored; .gitkeep marks the architecture)
metadata/   document · duplicate · guideline · textbook · currency_pack registries
annotations/ claim_validation_300.csv        (300-passage human validation)
logs/       retrieval, parsing, dedup, classification logs (gitignored)
reports/    corpus statistics and dry-run output
src/        retrieval · parsing · normalization · deduplication · chunking · metadata
            · classification
tests/      19 tests + offline fixtures
```

### Why five config files, not three

`sources.yaml`, `mesh_terms.yaml` and `claim_taxonomy.yaml` are the baseline. Two more exist
because they carry distinct, necessary content and merging them would create one large file
mixing unrelated concerns:

- **`controlled_vocabulary.yaml`** — 89 Alzheimer's surface forms across 10 groups, plus the
  `preserve_verbatim` list. Consumed by both the query builder and the relevance gate.
- **`corpus_config.yaml`** — the **processing** pipeline: relevance rules, chunking, dates,
  retraction handling. `sources.yaml` owns the **sources**: pools, tiers and licensing. Neither
  duplicates the other.

## Processing pipeline

```
raw → parsed → normalized → deduplicated → chunks
```

Deduplication is document-level and happens **before** chunking, so expensive chunking is never
spent on duplicates. Normalization preserves biomedical surface forms verbatim — `Aβ42`, `Aβ40`,
`p-tau181`, `p-tau217`, `ARIA-E`, `ARIA-H`, `APOE ε4`. Case-folding is used for **matching only**
and is never written back.

Chunking: 256 tokens / 32 overlap / 224 stride, measured with the **MedCPT article-encoder
tokenizer** (512-token limit), never whitespace. Guidelines are structure-aware
(document → section → subsection → paragraph → sentence); a recommendation is never split from
its qualifying conditions, and exceptions are recorded per chunk.

## Licensing

Fails closed: unknown licence ⇒ not redistributable ⇒ text not committed. Full policy in
[`LICENSE_NOTES.md`](LICENSE_NOTES.md).

## Claim classification

50 classes in 9 groups, derived from the 11 topics the thesis proposal names. Multi-label.
**Never a deletion filter** — it drives ranking, down-weighting and review only. `keywords` in
`claim_taxonomy.yaml` are intentionally empty pending decisions E1/E3/E5
(`../docs/STAGE_1_CLAIM_CLASS_TAXONOMY.md`, `MANUAL_STEPS.md` M7).

## Run the offline test (works anywhere, no network)

```bash
python3 alzheimers_corpus/src/retrieval/build_queries.py --check      # config/query drift
cd alzheimers_corpus/tests && python3 -m unittest test_ad_relevance -v # 19 tests
```

Offline relevance run on **synthetic fixtures** (ids are `FIXTURE-*`, never real PMIDs):

```bash
python3 alzheimers_corpus/src/classification/ad_relevance.py \
  --input  alzheimers_corpus/tests/fixtures/dry_run_records.jsonl \
  --output alzheimers_corpus/reports/dry_run_ad_relevance.jsonl
```

Current offline result: **5 included / 5 excluded**. This is a fixture test, **not** corpus
retrieval statistics.

## Run retrieval (only where NCBI is reachable)

Check first, then use the existing tested E-utilities client in `../pubmed/fetch_pubmed.py` —
this folder does not duplicate it. Exact commands, escalation steps and the API-key procedure
are in [`MANUAL_STEPS.md`](MANUAL_STEPS.md) M0–M2.

## Reproducibility

corpus 0.1.0 · sources 0.1.0 · mesh_terms 0.1.0 · controlled_vocabulary 0.1.0 ·
claim_taxonomy 0.1.0 · corpus_config 0.1.0. Every chunk traces to
document → source → identifier → query id → retrieval date.
