# Licensing and reuse notes

This corpus is intended for public research use and public GitHub distribution.
Everything below exists to keep that true.

## The governing rule — fail closed

```
unknown licence  →  not assumed reusable  →  text NOT committed
```

Set in `config/sources.yaml`: `licensing.default_redistribution_allowed: false`.
A record whose licence cannot be **positively determined** is treated as not redistributable.
Metadata may still be kept; the text may not.

## What may be redistributed

Only content carrying a licence on the allow-list in `config/sources.yaml`:

`CC0` · `CC-BY` · `CC-BY-SA` · `CC-BY-NC` · `CC-BY-NC-SA` · `CC-BY-ND` ·
`public-domain` · `us-government-work`

Note `CC-BY-NC` and `CC-BY-NC-SA`: local text mining is permitted, redistribution is
**non-commercial only**. Record the exact terms; do not flatten them to "open".

## What may only be processed locally

| Source | Status | Where it lives |
|---|---|---|
| Medical textbooks | Copyrighted. **Never** committed | metadata only, `metadata/textbook_registry.csv` |
| Restricted guidelines | Free to read ≠ free to redistribute | metadata only, `metadata/guideline_registry.csv`; file under `data/raw/guidelines/` (gitignored) |
| Paywalled article text | Institutional access is not redistribution rights | not committed |
| Any unknown-licence item | Fails closed | metadata only |

Free to read does **not** imply free to redistribute. Reuse rights are verified **per document,
never per organisation** — one NICE document being open says nothing about the next.

## How licensing is recorded

Every record carries: `source`, `source_id`, `title`, `source_url`, `license`, `license_url`,
`reuse_status`, `redistribution_allowed`, `retrieval_date`.

`redistribution_allowed` is the single gate consulted before any text is written into the
distributable corpus.

## PubMed and PMC specifics

- **PubMed** — bibliographic metadata is broadly reusable; **abstract text rights vary by
  publisher**. Abstract text is recorded per record with its own licence field.
- **PMC** — only the Open Access subset is used. The per-article `ali:license_ref` is
  authoritative and is recorded verbatim, not inferred from the journal.
- Retrieval uses **official NCBI E-utilities and the PMC OA service**. HTML scraping is never used.

## Why raw data is intentionally absent from Git

`data/**` is gitignored except `.gitkeep` markers. Three reasons:

1. **Licensing** — some raw material may not be redistributable, and a blanket commit would
   distribute it by accident.
2. **Size** — a full corpus is far too large for a repository that must stay practical to clone.
3. **Reproducibility** — the corpus is rebuilt from versioned queries, configs and manifests,
   which are committed. The data is an output, not a source.

Registries, manifests, reports and `.gitkeep` markers **are** tracked: they are the provenance
record and the architecture.

## Before adding any new source

1. Identify the licence and its canonical URL — not a summary, the actual terms.
2. Decide `redistribution_allowed` explicitly. If you cannot determine it, it is `false`.
3. Add the pool to `config/sources.yaml` with its retrieval method and redistribution policy.
4. Record every document in the appropriate registry under `metadata/`.
5. If restricted: metadata only, file outside the repository, and a procedure in `MANUAL_STEPS.md`.

## Never committed

Pirated or restricted textbook content · restricted guidelines · paywalled text without
redistribution permission · private or patient-identifiable data · credentials, API keys,
`.env` files, authentication tokens · any copyrighted material whose redistribution is not
permitted.
