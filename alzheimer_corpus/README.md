# alzheimer_corpus

Alzheimer's-disease-scoped corpus pipeline. **Scaffold — no corpus data retrieved.**

Read [`CORPUS_CARD.md`](CORPUS_CARD.md) for scope and policy, and
[`MANUAL_STEPS.md`](MANUAL_STEPS.md) for everything that cannot be done automatically —
including **M0**, the blocked network egress that prevents retrieval in this environment.

## Relationship to the existing repository
This directory does **not** replace `pubmed/` or `pmc/`, which already implement a tested
E-utilities client, PMC download and parsing, corpus-policy metadata and chunking. It adds the
Alzheimer's-specific layer those lack: a versioned MeSH scope, a controlled vocabulary, the
claim taxonomy, and the **AD-relevance gate** that keeps the corpus from drifting into generic
dementia. Retrieval should reuse `pubmed/fetch_pubmed.py` rather than duplicating it.

## Layout
```
config/     mesh_terms · controlled_vocabulary · claim_taxonomy · corpus_config   (versioned)
queries/    Q01-Q07, GENERATED from config - do not hand-edit
scripts/    retrieval/build_queries.py · classification/ad_relevance.py
metadata/   five registries (headers only)
tests/      19 tests, all passing
reports/    dry-run output
data/       gitignored - raw and processed corpus never enters git
```

## Quick start
```bash
python3 alzheimer_corpus/scripts/retrieval/build_queries.py          # regenerate queries
python3 alzheimer_corpus/scripts/retrieval/build_queries.py --check  # detect config drift
cd alzheimer_corpus/tests && python3 -m unittest test_ad_relevance -v
```
