# Stage 1 · Phase 1A — Feasibility Measurements

**Executed:** 2026-09-11 · **Commit at execution:** `36635a8` · **Branch:** `claude/stage1-phase1-spec`

Four measurements identified by the architecture-to-corpus analysis. Purpose: replace guesswork
with evidence so that the outstanding research decisions can be made scientifically.

**Nothing was redesigned to make a measurement pass. No research decision was resolved silently.
Where a measurement could not run, it reports `NOT_RUN` with a reason rather than an estimate.**

## Reproducibility envelope

| Item | Value |
|---|---|
| Repository commit | `36635a8776731340419addd34a0dae1d93e33f70` |
| Scripts | `pmc/measurements/m{1,2,3,4}_*.py` (committed with these results) |
| Machine-readable outputs | `docs/stage1_measurements/m{1,2,3,4}_*.json` |
| Input: `canonical_dates.csv` | 43,409 rows · sha256[:16] `1e2af87eb6b95866` |
| Input: `corpus_policy.csv` | 43,409 rows · sha256[:16] `e0f081f607a5f4c6` |
| Input: `cpg_registry.csv` | 19 rows · sha256[:16] `9e4bd44342ed8f69` |
| Input: `currency_pack.csv` | 7 rows · sha256[:16] `26a8bc8af6e62479` |
| Retrieval config | frozen MedCPT stack (§5.3) — **not exercised**, no index present |
| Tokenizer | `ncbi/MedCPT-Article-Encoder` — **not exercised**, unreachable here |
| Corpus snapshot | **no corpus snapshot exists in this environment** (see below) |

### Environment limits, stated before the results

Verified by direct inspection at execution time:

| Artifact | State |
|---|---|
| `pmc/chunks/chunks.jsonl` | **absent** (gitignored derived artifact) |
| `pmc/parsed/` | **absent** |
| `pmc/fulltext/xml/` | **absent** |
| `pmc/index/` | **absent** |
| `pubmed/pubmed_results.csv` | **unresolved Git-LFS pointer** (4 KB standing for 102 MB) |
| `transformers` | not installed |
| HuggingFace (`ncbi/MedCPT-Article-Encoder`) | **unreachable** — `403 Forbidden` through the proxy |

So M1 ran at **metadata level only**, M2 and M3 could not run at all, and M4 ran on the
structural preconditions that frozen metadata can answer. The scripts are written to run fully
on the machine holding the corpus and refuse to run otherwise.

---

## Summary

| Measurement | Result | Research implication | Decision affected |
|---|---|---|---|
| **M1** Claim-class coverage | **BULK TAGGING ABSENT.** 21 distinct classes hand-assigned to 20 documents. `corpus_policy.csv` has **no `claim_class` column** — the 43,409-record corpus is untagged | γ and the contested test operate *per claim class*. For 43,383 of 43,409 records there is no class to operate on. This is not a coverage shortfall; the taxonomy layer does not exist yet | **C1′, C4′** |
| **M2** Retrieval reachability | **NOT RUN — two blockers, one of them a research decision.** No index (environment) **and no decisive-evidence set exists** (research) | R-M is untestable until someone defines what evidence is decisive. The probe set determines the answer, so it cannot be an engineering default | **A1** |
| **M3** Token/truncation | **NOT RUN.** No chunks, no tokenizer, no `transformers` | C6 stays open on evidence, not opinion. Script ready | **C6** |
| **M4** Contested pairs | **STRUCTURAL CANDIDATE EXISTS; DECISION BLOCKED.** Anchor + 2 same-class counterparties at **11** and **36 ± 12** months | The contested state is not impossible, but whether it can fire depends entirely on the contest-window length, which is `[OPEN]` | **A3** |

---

## M1 — Claim-class coverage

**Ran:** metadata level. **Did not run:** passage level (`chunks.jsonl` absent).

### The finding

The proposal (§5.1) requires "a hand-built taxonomy of **forty to eighty classes** … assignment
by keyword-and-embedding matching with manual verification of a 300-passage random sample."

Measured:

| Property | Required | Observed |
|---|---|---|
| Distinct classes | 40–80 | **21** |
| Documents tagged | corpus-wide | **20** |
| Bulk `claim_class` column in `corpus_policy.csv` | required for γ | **absent** |
| Assignment mechanism | keyword-and-embedding matcher | **none — values are hardcoded literals** in `pmc/build_corpus_metadata.py` (lines 66–190) |

**Interpretation.** This is not "some classes are thinly covered". There is no taxonomy layer
over the corpus at all. Claim classes exist as hand-written strings attached to the 19 curated
guidelines and 7 currency-pack documents. For the other ~43,383 records there is nothing for γ's
supersession branch or the contested test to key on.

The vocabulary is also not normalised between the two curated layers: `criteria` vs
`diagnostic-criteria`, `plasma` vs `plasma-biomarkers` denote the same classes under different
strings. Any matcher built later must fix this first or it will split classes silently.

### Observed classes (20 documents)

`pharmacotherapy` 9 · `differential-diagnosis` 9 · `aria` 8 · `amyloid-imaging` 7 ·
`diagnosis` 7 · `biomarkers` 4 · `eligibility` 4 · `criteria` 2 · `tau` 2 · `plasma` 2 ·
`apoe` 2 · `care` 2 · `amyloid` 1 · `staging` 1 · `vascular` 1 · `management` 1 ·
`prevention` 1 · `comorbidity` 1 · `diagnostic-criteria` 1 · `contested` 1 ·
`plasma-biomarkers` 1

Against the classes the proposal names explicitly: **deprescribing 0** and
**behavioural management 0** (`management`/`care` are not the same class and were not assigned
as such). Diagnosis, pharmacotherapy, ARIA, eligibility and amyloid imaging are represented —
but only within those 20 curated documents.

### Corpus date distribution (real, all 43,409 records)

| | Count |
|---|---|
| Pre-June-2024 | **18,707** |
| Post-June-2024 | **22,789** |
| Unknown (year-only 2024) | **1,913** |

By year: 2021 · 2,198 | 2022 · 7,003 | 2023 · 7,030 | 2024 · 8,499 | 2025 · 10,671 | 2026 · 7,999

Precision: day 17,055 · month 17,405 · **year-only 8,949**
Date source: **PubMed fallback 43,333** · JATS 76

**Two consequences.** R-I (counterfactual index) has ample volume on both sides of June 2024.
But 8,949 year-only records cannot be placed in a contest window finer than a year, and
**43,333 of 43,409 dates rest on the PubMed fallback**, so the canonical dates are provisional
until the pipeline is re-run where the full parsed corpus lives (JATS-primary).

---

## M2 — Retrieval reachability

**NOT RUN.** Two blockers, and they are not the same kind of problem.

| Blocker | Kind | Detail |
|---|---|---|
| `NO_INDEX` | **Environment** | `pmc/index/` absent. Fixed by `embed_chunks.py` + `verify_index.py` on the corpus machine |
| `NO_DECISIVE_EVIDENCE_SET` | **REQUIRES HUMAN DECISION** | No (query, expected-evidence) probe set exists anywhere in the repository |

**The second blocker is the important one, and it would still block on the corpus machine.**
Reachability is measured against a set of things we assert *ought* to be retrieved. Choosing
which evidence is "decisive", and the queries used to reach it, determines the number that comes
out. An easy probe set produces high reachability; a demanding one produces low reachability.
That makes it a research decision, so the script refuses to invent one and exits rather than
manufacturing a defensible-looking result.

This is consistent with `docs/architecture.md` §9, which already records "no evaluation query set
exists yet — `queries.path` is empty by necessity."

The script accepts an approved probe file at
`docs/stage1_measurements/decisive_evidence_probes.json` and then reports reachability@{5,10,20,50},
zero-retrieval rate, rank distribution, and failures broken down by claim class, source type and
temporal category — separating exact-passage from document-level hits.

---

## M3 — MedCPT token-length / truncation audit

**NOT RUN.** `chunks.jsonl` absent; `transformers` not installed; HuggingFace returns
`403 Forbidden` through this environment's proxy, so the MedCPT tokenizer cannot be loaded.

**No estimate is offered in place of the measurement.** A words→tokens ratio guessed from
general knowledge would be exactly the kind of convenience-substituted-for-evidence this phase
exists to avoid, and C6 turns on the truncation percentage, which a ratio cannot give.

What the script will report, once run on the corpus machine: token-length min/max/mean/median and
p50/p75/p90/p95/p99; percentage of chunks exceeding the **512-token** encoder limit; total and
mean tokens lost to truncation; and truncation broken down by source category and document type.
It tokenizes `compose_embed_text()` — the repository's single shared rule for what the encoder
actually sees — rather than the raw chunk text, so it measures the real input.

**Prerequisite note:** only the *tokenizer* is needed, not the model weights. On a machine with
network access this is a small download.

---

## M4 — Contested-pair existence

**Ran** on frozen metadata. Verdict: **`STRUCTURAL_CANDIDATE_EXISTS_DECISION_BLOCKED`.**

**Anchor:** `PMC13082890` — Cochrane CD016297, *Amyloid-beta-targeting monoclonal antibodies…*
dated **2026-04-16** (day precision), classes `contested; pharmacotherapy`.

**Same-claim-class counterparties found:**

| PMCID | Document | Date | Precision | Shared class | Gap from anchor |
|---|---|---|---|---|---|
| PMC12180672 | Donanemab: Appropriate use recommendations | 2025-05 | month | `pharmacotherapy` | **11 months** |
| PMC10313141 | Lecanemab: Appropriate Use Recommendations | 2023 | year | `pharmacotherapy` | **36 months ± 12** |

**Precondition status:**

| Precondition (§4.4) | Status |
|---|---|
| Same-claim-class counterparty exists | **YES** |
| Opposing conclusions | **NOT MACHINE-CHECKABLE** — a research judgement; not asserted here |
| Both within the contest window | **UNDECIDABLE** — the window length is `[OPEN]` in §4.4 |
| Both above minimum source tier | **UNDECIDABLE** — tier values are undefined, and authority ordering is a *tested* variable (ablation A12), so assigning tiers here would prejudge a hypothesis |
| Retrievable | **NOT RUN** — no index |

**Interpretation.** The earlier concern that the contested state might have no counterparty at
all is **not confirmed** — a same-class candidate exists at 11 months. But the state's firability
now rests entirely on one undecided parameter. A contest window matching §1.3's narrative
("contested within days") excludes both counterparties and leaves nothing. A window of ~12 months
admits the donanemab AUR. A window wide enough for the lecanemab AUR spans three years, which
weakens the claim that the dispute is contemporaneous.

**No rebuttal documents were introduced**, per instruction. Whether to ingest the Alzheimer's
Association / UK DRI / *Lancet* responses named in §1.3 remains an open decision, not an action
taken here.

---

## Decision reassessment

| ID | Classification | Why |
|---|---|---|
| **C1′** scope widening | **REQUIRES HUMAN DECISION** — but the question changed | M1 shows the blocker is not scope breadth. Bulk claim-class tagging does not exist, so per-class coverage is currently unmeasurable at any scope. **Build the taxonomy layer first, then re-measure, then decide scope.** Deciding now would be deciding blind |
| **C4′** currency-pack composition | **REQUIRES HUMAN DECISION** | M1 gives partial evidence: `deprescribing` and `behavioural management` are unrepresented among 20 curated documents. Whether that matters depends on which classes the evaluation sets exercise — not yet defined |
| **C5** time-invariant content | **REQUIRES HUMAN DECISION** — unchanged | No measurement bore on it. Still open |
| **C6** chunk unit | **NOT YET DETERMINED** (was REQUIRES HUMAN DECISION) | Correctly downgraded: this is answerable by measurement, and the measurement did not run. It becomes a decision only if truncation proves material |
| **C7** near-duplicate policy | **REQUIRES HUMAN DECISION** — unchanged | No measurement bore on it. Still architecturally critical (§5.1 names the preprint-to-journal failure explicitly) |
| **C10′** audit sampling | **REQUIRES HUMAN DECISION**, now clearly premature | The 300-passage audit validates claim-class tagging precision. With no bulk tagging, there is nothing to audit. Revisit after the taxonomy layer exists |
| **A3** contested state | **REQUIRES HUMAN DECISION**, now sharply narrowed | Reduced from "does any counterparty exist?" to a single parameter: **the contest-window length**, plus whether rebuttals are ingested |

**Resolved by evidence: none of the seven.** That is the honest outcome — two of four
measurements could not run, and the one that ran fully (M1) revealed a missing prerequisite
rather than a coverage number.

---

## What the measurements changed

1. **M1 found a missing layer, not a missing quantity.** C1′, C4′ and C10′ all depend on
   per-claim-class counts that cannot exist until the taxonomy and matcher are built. The
   ordering of Stage 1 work changes accordingly: taxonomy before scope decisions.
2. **M2 surfaced a research blocker hiding behind an environment blocker.** Building the index
   would not have unblocked it.
3. **M4 narrowed A3 from an existential question to a parameter choice.**
4. **A new finding, incidental to M1:** 43,333 of 43,409 canonical dates use the PubMed
   fallback, and 8,949 are year-only. Any contest window shorter than a year is unenforceable
   for a fifth of the corpus.
