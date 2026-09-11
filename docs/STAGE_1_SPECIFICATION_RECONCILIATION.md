# Stage 1 — Specification Reconciliation Report

**Date:** 2026-09-11 · **Phase:** 1 (Freeze Corpus Specification) · **Status:** NOT RECONCILED

Compares what the **thesis proposal** requires, what **RAG²** supports, and what the
**repository currently implements**. Produced before any further corpus acquisition.

---

## 0. Source documents and how they were verified

| Document | Location | Verification |
|---|---|---|
| MS Thesis Research Proposal (27 pp., Aug 2026) | `MS_Thesis_Proposal.pdf` (repo root) | MD5 `f8de7826ce5073bb707a4c5a947b772c` — **byte-identical** to the copy supplied in session uploads. One authoritative source, no ambiguity. |
| Sohn et al. (2025), *Rationale-Guided RAG for Medical QA*, NAACL 2025, pp. 12739–12753 | session uploads | Text extracted and read in full (15 pp. incl. appendix). |

Both were read in full. Every requirement below carries a line-level citation.

**Authority rule applied here:** the proposal is primary. RAG² is authoritative only for
what the *frozen baseline* does, because the proposal freezes the upstream pipeline to it.

---

## 1. Verified repository state

Facts established by direct inspection, not by reading prior claims:

- Branch `main`, HEAD `2566d2a`, **53 commits**, working tree clean, **183 files on disk**.
- `docs/` contains exactly two files: `architecture.md`, `rag2_reproduction_audit.md`.
- Implementation trees present: `pubmed/`, `pmc/`, `rag2/`, `thesis/`, `configs/`, `experiments/`.
- Corpus policy metadata present and non-empty:
  `canonical_dates.csv` (43,409 rows), `corpus_policy.csv` (43,409 rows),
  `cpg_registry.csv` (19 rows), `currency_pack.csv` (7 rows), `cpg_external_targets.csv`.
- `pmc/chunks/chunk_stats.json` records **60,874 chunks** over **76 parsed records**,
  `window_words: 256`, `overlap_words: 32`.

**Not verifiable in this checkout** (absence of evidence, recorded as such, not as failure):

- No raw XML (`pmc/fulltext/xml/` empty — gitignored), no parsed records (`pmc/parsed/` empty),
  no chunk payload (`chunks.jsonl` gitignored), no MedCPT index (`pmc/index/` empty).
- `pubmed/pubmed_results.csv` is an unresolved **Git-LFS pointer** (4 KB standing for 102 MB).
- The production build referenced in `docs/architecture.md` §9
  (42,964 documents / 781,563 chunks / digest `da1886b0…`) **cannot be verified here**.
  `architecture.md` itself states runs stay `reportable: false` until digests match.

**Therefore:** the corpus-acquisition *code* is present and tested; the corpus *data* is not
present in this environment and none of its counts are independently confirmed here.

---

## 2. Requirements explicitly stated in the proposal

Line numbers refer to the extracted proposal text.

| # | Requirement | Cite |
|---|---|---|
| P1 | Five corpora: PubMed-AD, PMC-AD, CPG-AD, Textbooks, Currency pack | §5.1 table |
| P2 | PubMed-AD scoping rule: **MeSH — Alzheimer Disease, Dementia, Cognitive Dysfunction, Amyloid beta-Peptides, tau Proteins, Lewy Body Disease, Frontotemporal Dementia** | §5.1 |
| P3 | PMC-AD: PMC Open Access subset, **same filter**, open-licence only | §5.1 |
| P4 | CPG-AD: dementia-relevant guidance, **~10³ documents** | §5.1 |
| P5 | Textbooks: neurology, psychiatry, internal medicine, **18 volumes** | §5.1 |
| P6 | Currency pack: **20–40 documents**, manual ingestion; named anchors = 2024 revised criteria, lecanemab AUR, donanemab AUR, ARIA guidance, Cochrane CD016297 | §5.1 |
| P7 | Scale estimates are tagged **[OPEN]** — estimates, not frozen targets | §5.1 table header |
| P8 | Per-passage metadata: chunk + document ids; **source tier**; publication date; journal + persistent id; **guideline family, version, supersession pointer**; retraction/withdrawal flags; claim-classes; section header; **character span** | §5.1 |
| P9 | Chunking: sliding window at **256 tokens, 32-token overlap** `[DES]`, sized against the article encoder's **512-token limit** with headroom for prepended title + section header | §5.1 L726–727 |
| P10 | Exact deduplication by content hash | §5.1 L728 |
| P11 | Near-duplicate detection by **MinHash over 5-gram shingles at Jaccard 0.85, retaining the newest and highest-tier cluster member** — specifically so preprint-to-journal duplicates cannot let an old claim survive under a new date | §5.1 L728–731 |
| P12 | Guideline passages **respect section boundaries** | §5.1 L731–733 |
| P13 | Every passage **inherits its parent document's publication date** | §5.1 L734 |
| P14 | Date imputation from issue metadata applied; **per-corpus null rate is a reported figure** | §5.1 L736–737 |
| P15 | Claim-class taxonomy of **40–80 classes**; assignment by keyword-and-embedding matching; **manual verification of a 300-passage RANDOM sample, reporting precision and recall** | §5.1 L740–743 |
| P16 | Counterfactual corpus-update: one index of documents dated **before the June 2024 criteria revision**, one containing all | §6.4 |
| P17 | Retrieval frozen: **MedCPT query + article encoders, MedCPT cross-encoder reranker** | §5.3 |
| P18 | Scope: English-language public corpora; all resources public and openly licensed; no patient-identifiable data; credentialed resources optional and never load-bearing | §1.7, §10 |
| P19 | The **identical scoped corpus is used by every experimental arm** — scoping is a controlled constant | §5.1 |
| P20 | Retracted articles must remain **retrievable**; the currency gate γ rejects them at admission (γ = 0 only for retracted/withdrawn) | §4.3 |

**Notably absent from the proposal** (verified by exhaustive grep, not assumed):

- **No publication-date window / temporal cutoff of any kind.** The scoping rules are MeSH terms only.
- **No clinical-reasoning or diagnosis requirement** in the scoping rule.
- **No definition of "source tier"** — the field is required (P8) but its values are never enumerated.
- **No pre-training cutoff date, and no knowledge-boundary audit procedure.** The mechanism depends on "pre-training-era priors" (§1.2 L139, L188) but no cutoff is stated.
- **No Stage-1 retrieval-reachability requirement.** Recall@k / nDCG@10 appear only in §7.1 as *Stage-5 evaluation* metrics against pooled human judgements.
- **No online-first vs print vs version date policy.** The proposal says only "publication date".

---

## 3. Requirements supported by RAG² (the frozen baseline)

| # | What RAG² establishes | Cite |
|---|---|---|
| R1 | Balanced retrieval draws **equal quotas from four corpora** (PubMed, PMC, CPG, textbooks) | §3.4 |
| R2 | MedCPT retriever + MedCPT cross-encoder reranker; reranker encodes the **original query**, retrieval uses the **rationale** | §3.3, §3.4 |
| R3 | Baseline corpus: 37.6M docs / 116.7M passages / 564.2 GB | Table A1 |
| R4 | Chunking: "sliding window mechanism with overlap" — **no token size is stated anywhere** | §A.3 |
| R5 | Filter is Flan-T5-large (770M) on correctness-flip labels with ΔPPL tie-break at τ = top 25% | §3.2, Eq. 3 |
| R6 | **No temporal representation** — no date, version or supersession field appears anywhere | verified by absence across 15 pp. |

**R4 matters for this phase:** the 256/32 figures are *not* inherited from RAG². The proposal
tags them `[DES]` — a design decision of this thesis. They are therefore the thesis's own
choice, and changing them costs no baseline fidelity, but they must not be changed silently.

**R1 matters:** balanced retrieval assumes **four** source corpora. A corpus missing textbooks
changes the quota structure of a stage the thesis declares frozen.

---

## 4. What the repository currently documents

| # | Repo position | Where |
|---|---|---|
| D1 | Scope operationalised as **Alzheimer's ∩ clinical-reasoning/diagnosis** | `pubmed/search_queries.txt` |
| D2 | **Publication window 2021-08-30 → 2026-08-30** ("last 5 years"), AND-ed onto every search | `search_queries.txt` L18–22 |
| D3 | CPG-AD = **19 curated documents**; the proposal's "~10³" figure is recorded as **"rejected by the CPG Source Policy"** | `pmc/metadata/README.md` |
| D4 | Currency pack = **7 documents** (README text says 4 — internally inconsistent) | `currency_pack.csv` vs its README |
| D5 | Canonical date = **earliest real publication event (electronic first), excluding preprint dates**; JATS primary, PubMed fallback | `pmc/metadata/README.md` |
| D6 | `canonical_date_precision` ∈ {day, month, year}, **never upgraded** | same |
| D7 | Chunking = **256 whitespace words / 32-word overlap**, never crossing a section boundary | `README.md`, `chunk_stats.json` |
| D8 | Exact-duplicate chunks are **flagged via `duplicate_of`, never deleted** | `README.md` |
| D9 | Retracted → **eligible, flagged**; errata/retraction notices → excluded; no-licence → manual-review | `pmc/metadata/README.md` |
| D10 | `authority_tier_label` is a **label only, no ordering encoded** (authority is tested variable A12) | same |
| D11 | Exact flat search, not ANN, for byte-identical candidate replay (V3) | `README.md` |
| D12 | Canonical dates are **not final** pending a re-run where the full parsed corpus lives | `pmc/metadata/README.md` |

---

## 5. Critical contradictions

| # | Proposal | Repository | Severity |
|---|---|---|---|
| **C1** | Scoping rule is **MeSH disease terms only** (P2) | Requires AD **AND** clinical-reasoning/diagnosis language (D1) | **High** — changes what evidence exists to retrieve |
| **C2** | **No temporal window** (verified absent) | **Five-year window**, 2021-08-30 → 2026-08-30 (D2) | **High** — bounds what "old evidence" can mean |
| **C3** | CPG-AD **~10³ documents** (P4) | **19 documents**; proposal figure explicitly rejected (D3) | **High** — a prior decision overrode the proposal |
| **C4** | Currency pack **20–40 documents** (P6) | **7 documents** (D4) | **High** — new evidence must be present to be admitted |
| **C5** | Textbooks, **18 volumes** (P5); RAG² balanced retrieval assumes 4 corpora (R1) | **No textbook corpus exists** | **High** — alters a stage declared frozen |
| **C6** | Chunk window = **256 tokens** against a 512-**token** encoder limit (P9) | **256 whitespace words** (D7) — a different and larger unit | **High** — changes what the retriever sees |
| **C7** | Near-duplicates via **MinHash 5-gram Jaccard 0.85, retain newest + highest tier** (P11) | **Exact hash only**, and duplicates are **flagged, not removed** (D8) | **Medium-High** — P11 exists precisely to stop an old claim surviving under a new date |
| **C8** | Per-passage **supersession pointer** and **character span** (P8) | Free-text `supersession_note` on 19 CPG rows only; **word** offsets, not character spans | **Medium** |
| **C9** | Per-passage **source tier** (P8) | `corpus_policy.csv` has **no `source_tier` column**; `authority_tier_label` exists for 19 CPG rows only | **Medium** |
| **C10** | 300-passage **RANDOM** sample (P15) | Not yet implemented | **Medium** — note the proposal says random; stratification would be a *change*, not a refinement |

**Internal inconsistencies inside the repository** (independent of the proposal):

- `pmc/metadata/README.md` says the currency pack has **4** documents; the CSV has **7**.
- The same README reports `pre 10,965 / post 13,521 / unknown 1,257`; the CSV yields
  `pre 18,707 / post 22,789 / unknown 1,913`.

Neither is treated as authoritative here.

---

## 6. Unresolved research decisions

Carried into `STAGE_1_CORPUS_SPECIFICATION.md` under their labels. Summary:

**REQUIRES HUMAN DECISION** — C1, C2, C3, C4, C5, C6, C7, and the audit sampling method (C10).

**NOT YET DETERMINED** (missing from the source material, cannot be resolved by inspection):

- Pre-training cutoff of the label-generating model, and the knowledge-boundary audit procedure.
- The definition and permitted values of **source tier**.
- The supersession-pointer data format.
- Whether a Stage-1 retrieval-reachability audit exists at all (none is specified).
- Filter replication backbone and entailment teacher (both `[OPEN]` in the proposal).

---

## 7. Assumptions that must NOT be silently converted into methodology

Recorded explicitly so that convenience does not become method:

1. **"256 words ≈ 256 tokens."** It is not. 256 English words is typically ~340–400 MedCPT
   sub-word tokens. The word implementation may still fit under 512, but it is a *different
   passage size* than the proposal specifies, chosen — per the repo's own README — because it
   is "deterministic and dependency free". That is an engineering rationale for a research
   parameter.
2. **"The five-year window is harmless because the counterfactual split is June 2024."**
   The split is indeed unaffected (18,707 pre / 22,789 post). But the window still determines
   whether genuinely superseded documents exist in the corpus at all.
3. **"19 curated guidelines are enough because the authoritative AD-guidance universe is small."**
   This may well be correct, but it contradicts the proposal and needs supervisor sign-off,
   not a repository note.
4. **"Retrieval reachability will be fine."** Untested. The proposal's own cited expert
   evaluation reports 22% top-16 relevance and 31% of queries returning nothing relevant.
5. **"Publication date is a single unambiguous thing."** It is not — online-first, print,
   version and revision dates differ, sometimes by more than a year, in a thesis whose
   dependent variable is evidence age.
6. **"Flagging duplicates is equivalent to resolving them."** Flagging preserves information,
   which is right; but P11 requires an actual retention rule, and none is implemented.

---

## 8. Reconciliation verdict

**NOT RECONCILED.** Seven high-severity contradictions between the proposal and the
implemented corpus remain open. The specification cannot be frozen, and large-scale
acquisition must not resume, until the authority question in §5 is settled.
