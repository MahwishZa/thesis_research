# AD-CORPUS — Corpus Card

**Version 0.1.0 (scaffold)** · configuration and gate implemented; **no corpus data retrieved**.

## Purpose
Retrieval haystack for an MS thesis on recency bias in confidence-derived evidence-utility
signals for retrieval-augmented Alzheimer's clinical reasoning. The corpus is not the
measurement instrument: the primary probe material is externally authored (MedChangeQA), and this
corpus supports the counterfactual index, the contested case study, domain replication and
expert vignettes.

## Scope — Alzheimer's disease
**Alzheimer Disease is the central retrieval concept.** Dementia, cognitive dysfunction,
amyloid/tau pathology and comparator diseases are **supporting** concepts: they can qualify a
record only alongside an Alzheimer's anchor, never alone. Every Q01–Q07 query AND-s the anchor
block, and the relevance gate enforces the same rule per record.

## Inclusion / exclusion
| Rule | Effect |
|---|---|
| R1 AD MeSH descriptor | include (decisive) |
| R2 AD term in title | include (decisive) |
| R3 AD term in abstract | include (decisive) |
| R4 amyloid/tau **with** anchor | include |
| R5 comparator disease **with** anchor | include (differential diagnosis) |
| X1 generic dementia, no anchor | exclude |
| X2 comparator disease, no anchor | exclude |
| X3 bare "AD" with no long form | exclude |

Excluded records keep their metadata and reason — nothing is dropped silently.

## Sources
PubMed (E-utilities), PMC Open Access, clinical guidelines, consensus statements, systematic
reviews, currency pack. Textbooks: **metadata only**.

## Licensing policy — fails closed
`default_redistribution_allowed: false`. A record with an unknown licence is treated as **not**
redistributable and its text is never committed. Restricted material is represented by metadata
plus a manual procedure (`MANUAL_STEPS.md`).

## Preprocessing
`download → parse → normalize → validate → deduplicate → chunk`. Deduplication precedes chunking.
Normalization preserves biomedical surface forms verbatim: `Aβ42`, `Aβ40`, `p-tau181`,
`p-tau217`, `ARIA-E`, `ARIA-H`, `APOE ε4`. Case-folding is used for matching only and never
written back.

## Chunking
256 tokens / 32 overlap / 224 stride, measured with the **MedCPT article-encoder tokenizer**
(512-token limit), not whitespace. Guidelines are structure-aware
(document → section → subsection → paragraph → sentence); a recommendation is never split from
its qualifying conditions, and any exception is recorded per chunk.

## Claim classification
50 classes in 9 groups, derived from the 11 topics the proposal names. Multi-label.
**Never a deletion filter** — it supports ranking, down-weighting and review, because γ
down-weights rather than deletes when tagging precision is unverified.

## Retractions
`retracted`, `withdrawn`, `expression_of_concern`, `corrected` are flags. Retracted documents stay
retrievable so that rejection at admission can be measured; `usable_for_evidence` is derived.

## Source tiers
Nine tier values, **deliberately unordered**. Authority ordering is a tested thesis variable
(ablation A12); encoding an order here would prejudge it.

## Known limitations
- **No data retrieved.** Egress to NCBI is blocked in the build environment (M0).
- Taxonomy `keywords` are empty pending decisions E1/E3/E5 (M7).
- Chunk truncation against the 512-token limit is **unmeasured** — the tokenizer is unreachable.
- Date policy (online-first vs print) is **provisional**; the proposal says only "publication date".
- Topical scope vs the proposal's MeSH-only rule remains open (**C1′**).

## Provenance and versioning
corpus 0.1.0 · mesh_terms 0.1.0 · controlled_vocabulary 0.1.0 · claim_taxonomy 0.1.0 ·
corpus_config 0.1.0. Every chunk traces to document → source → identifier → query id → retrieval
date.

## Reproduction
```bash
python3 alzheimer_corpus/scripts/retrieval/build_queries.py --check
cd alzheimer_corpus/tests && python3 -m unittest test_ad_relevance -v
python3 alzheimer_corpus/scripts/classification/ad_relevance.py \
  --input alzheimer_corpus/tests/fixtures/dry_run_records.jsonl \
  --output alzheimer_corpus/reports/dry_run_ad_relevance.jsonl
```
Retrieval steps require a machine with NCBI access (M0).
