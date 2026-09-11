# Manual steps

Everything this pipeline cannot legally or technically complete automatically.

---

## M0 — Network egress is blocked in the Claude Code session ⚠️ blocking

**Why manual.** This session's outbound HTTPS goes through a policy-enforcing proxy.
`eutils.ncbi.nlm.nih.gov:443` is **denied by organisation egress policy**:

```
{"kind":"connect_rejected",
 "detail":"gateway answered 403 to CONNECT (policy denial or upstream failure)",
 "host":"eutils.ncbi.nlm.nih.gov:443"}
```

`huggingface.co` is blocked the same way. The proxy documentation states such denials must be
reported, never retried or routed around. **No PubMed or PMC record has been retrieved, and no
retrieval statistic in this repository comes from a live run.**

**What you must do.** Run every retrieval step on your own machine.

**Expected output.** Raw responses under `alzheimer_corpus/data/raw/`, manifests under
`alzheimer_corpus/data/manifests/`.

**Do NOT commit.** Raw bulk downloads. Commit manifests and metadata instead.

---

## M1 — NCBI API key

**Why manual.** A credential. It must never enter the repository.

```bash
export NCBI_API_KEY="your-key"       # never commit, never echo into a file in the repo
export NCBI_EMAIL="you@example.org"  # NCBI asks for a contact address
```

Without a key E-utilities allows ~3 requests/second; with one, ~10.

**Do NOT commit.** The key, any `.env` file, or a shell history containing it.

---

## M2 — Count-only dry run before any bulk retrieval

**Why manual.** Depends on M0.

```bash
python3 alzheimer_corpus/scripts/retrieval/build_queries.py --check   # queries match config
# then, per family, a count-only esearch (retmax=0) and record the hit count
```

**Expected output.** A hit count written into the `# Hits: ____  Date: ____` line of each
`alzheimer_corpus/queries/pubmed/Q0*.txt`. Commit those counts — they are provenance.

**Escalate progressively:** 100 → 1,000 → 10,000 → full. Never jump straight to bulk.

---

## M3 — Guidelines requiring manual acquisition

**Why manual.** Many authoritative guidelines (NICE, some society guidance) are free to read but
**not** free to redistribute. Reuse rights must be verified per document, not per organisation.

**What you must do.** For each guideline: obtain it through its official route, record the licence
and its URL, and set `redistribution_allowed` in
`alzheimer_corpus/metadata/guideline_registry.csv`.

- `redistribution_allowed = true` → text may enter the distributable corpus
- `redistribution_allowed = false` → **metadata only**; keep the file outside the repository and
  point `local_file` at a path under `data/raw/guidelines/` (gitignored)

**Do NOT.** Scrape sites against their terms, or commit a non-redistributable document.

---

## M4 — Medical textbooks

**Why manual.** Textbooks are copyrighted. There is no lawful automated route.

**What you must do.** Populate `alzheimer_corpus/metadata/textbook_registry.csv` with metadata
only, for textbooks you legitimately own or licence. Process locally if your licence permits;
keep derived text outside the repository.

**Do NOT.** Download textbooks from unauthorised sources, or commit textbook text. Bibliographic
metadata is fine; content is not.

---

## M5 — Currency-pack document identification

**Why manual.** The specification names four items but deliberately does not name the drugs.
Identifying the authoritative document for each — correct title, organisation, date, **version**
— is a research judgement, and guessing would corrupt the temporal axis.

**What you must do.** Identify and verify each, then record it in
`alzheimer_corpus/metadata/currency_pack_registry.csv` with its verified licence.

**Note.** The repository already holds a verified currency-pack layer at
`pmc/metadata/currency_pack.csv` (7 documents). Reconcile rather than duplicate.

---

## M6 — Annotate the 300 validation passages

**Why manual.** Human ground truth by definition.

**What you must do.** Fill `human_labels` and `annotator_notes` in
`alzheimer_corpus/annotations/claim_validation_300.csv`. The sample must be **stratified** across
source, tier and claim class — not the first 300 chunks.

**Note.** Sampling method is still an open research decision (**C10′**): the proposal specifies a
**random** 300-passage sample, while γ's per-class error argues for stratification. Settle this
with your supervisor before sampling.

---

## M7 — Taxonomy decisions before any matcher is built ⚠️ blocking

**Why manual.** Three decisions materially change results and must not be made by a tool.

- **E1 granularity** — demonstrated to flip whether the contested state can fire at all
- **E3 multi-label vs single-label** — changes the contest false-positive rate
- **E5 ψ(q) time-invariance per class** — directly changes γ

See `docs/STAGE_1_CLAIM_CLASS_TAXONOMY.md`. `keywords` in `claim_taxonomy.yaml` are
intentionally empty until these are settled.

---

## M8 — Git history conflict ⚠️ blocking for pushes to `main`

**Why manual.** This repository contains **two unrelated histories**:

| Ref | Commits | Content |
|---|---|---|
| local `main` | 53 | The real implementation |
| `origin/main` | 4 | An empty re-created repository |

`git merge-base main origin/main` returns **no common ancestor**. Pushing local `main` to
`origin/main` would require `--force` and would **destroy** the remote history. This pipeline is
therefore pushed to a feature branch instead.

**What you must do — choose one, then tell Claude:**

1. **Open a PR** from `claude/stage1-phase1-spec` into `main` and merge it through GitHub (no
   history is destroyed; the histories stay unrelated but the content lands).
2. **Adopt the real history as `main`** — an explicit, authorised, destructive operation you run
   yourself after confirming the 4 remote commits are expendable.
3. **Rename**: keep the real history under its own long-lived branch and change the repository's
   default branch on GitHub.

**Do NOT.** Force-push without deciding this deliberately.
