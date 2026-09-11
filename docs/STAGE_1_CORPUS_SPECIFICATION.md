# Stage 1 — Alzheimer's Corpus Specification (AD-CORPUS-v1)

**Status: DRAFT — NOT FROZEN.** Stage 1, Phases 1 and 1A. Companions:
[`STAGE_1_ARCHITECTURE_TO_CORPUS_REQUIREMENTS.md`](STAGE_1_ARCHITECTURE_TO_CORPUS_REQUIREMENTS.md)
(Phase 1A — requirements derived backwards from the architecture; **this is the governing
document**) and
[`STAGE_1_SPECIFICATION_RECONCILIATION.md`](STAGE_1_SPECIFICATION_RECONCILIATION.md)
(Phase 1 — proposal-vs-implementation reconciliation, now superseded in framing).

> **Revision after Phase 1A.** The Phase 1 reconciliation compared the proposal's prose against
> the implementation and found ten "contradictions". Re-deriving requirements from the
> architecture dissolved several of them (C2, C3, and the framing of C1, C4, C10) and hardened
> others (C6, C7). It also surfaced requirements no prose comparison could have found — chiefly
> that **SCAF is a filter over an already-retrieved candidate set and therefore cannot correct
> evidence retrieval never returns.** The decision register in §13 reflects the revised view.

Primary authority: `MS_Thesis_Proposal.pdf` (MD5 `f8de7826…`), with Sohn et al. (NAACL 2025)
authoritative for the frozen RAG² baseline only.

## Label meanings

| Label | Meaning |
|---|---|
| **FROZEN** | Directly required by the proposal (or by RAG², where the proposal freezes it). Not re-decidable without amending the proposal. |
| **PROVISIONAL** | A reasonable engineering choice, not yet approved. Changeable without contradicting the proposal. |
| **REQUIRES HUMAN DECISION** | Could materially affect experimental validity. Must be approved by student/supervisor. |
| **NOT YET DETERMINED** | Missing from the source material. Cannot be resolved by inspection or by reasoning. |

---

## 1. Purpose of the corpus

**FROZEN.** The corpus is the *retrieval haystack* for a study of whether the evidence-admission
stage of a medical RAG pipeline preferentially admits older evidence. It is not itself the
measurement instrument.

Consequences that follow, and constrain everything below:

- **It must contain both sides of real Alzheimer's knowledge changes.** If only current
  consensus is present, a preference for older evidence cannot be observed.
- **The identical scoped corpus is used by every experimental arm** (proposal §5.1), so
  scoping is a controlled constant, not a confound.
- **Absolute performance figures will not be comparable to published baselines**, because the
  corpus is domain-scoped. Only within-study arm contrasts are interpretable (§10).
- The primary probe material (FRB-PAIRS) comes from **MedChangeQA**, which is external.
  The corpus supports domain replication, the contested-state case study, and the
  counterfactual corpus-update procedure — not the primary claim.

---

## 2. What counts as Alzheimer's evidence

### 2.1 Proposal scoping rule — FROZEN as written

MeSH terms (§5.1): **Alzheimer Disease, Dementia, Cognitive Dysfunction, Amyloid beta-Peptides,
tau Proteins, Lewy Body Disease, Frontotemporal Dementia.**

*MeSH in plain language:* PubMed librarians hand-tag every article with subject labels from a
fixed vocabulary. Tagging is more reliable than word-searching because it catches papers that
use different wording and skips papers that only mention a topic in passing. The labels nest,
so asking for "Dementia" automatically includes narrower kinds beneath it ("explosion").

The proposal's rule is **disease-scoped only**. It imposes no topic, study-type or date
restriction beyond these terms plus English-language and open-licence (§1.7, §10).

### 2.2 Implemented scoping rule — REQUIRES HUMAN DECISION *(contradiction C1)*

`pubmed/search_queries.txt` requires **Alzheimer's AND clinical-reasoning-or-diagnosis
language** (decision-making, differential diagnosis, diagnostic criteria, biomarkers, imaging,
cognitive tests). This is materially narrower than §2.1 and is not stated in the proposal.

*Why it matters:* the narrower scope excludes pharmacotherapy, ARIA management, deprescribing
and behavioural-management literature — several of which are named claim-classes in the
proposal's own taxonomy (§5.1) and are exactly where the 2023–2026 knowledge changes occurred.

### 2.3 Temporal coverage — REQUIRES HUMAN DECISION *(contradiction C2)*

The proposal states **no publication-date window**. The implementation AND-s
`("2021/08/30"[Date - Publication] : "2026/08/30"[Date - Publication])` onto every search.

Observed year distribution in `canonical_dates.csv` (43,409 rows):

```
2021: 2,198   2022: 7,003   2023: 7,030
2024: 8,499   2025: 10,671  2026: 7,999      (+ 9 stray pre-2021 records)
```

*Why it matters:* the oldest evidence is from 2021, so the 2011 NIA-AA criteria that the 2024
criteria replaced are absent. The "old" side of that supersession is represented only
indirectly, by 2021–2024 papers still applying older thinking. The June-2024 counterfactual
split (§6.4) is **unaffected** — 18,707 pre / 22,789 post — but the supersession story is.

---

## 3. Source databases and document types

| Corpus | Proposal requirement | Implemented | Label |
|---|---|---|---|
| PubMed-AD | E-utilities, MeSH rule §2.1, 2–4×10⁵ abstracts `[OPEN]` | 43,409 records under narrowed scope | **REQUIRES HUMAN DECISION** (C1, C2) |
| PMC-AD | PMC OA subset, same filter, open-licence only, 2–4×10⁴ `[OPEN]` | 25,742 XML claimed; **not verifiable here** | **FROZEN** in rule; scale `[OPEN]` |
| CPG-AD | Dementia-relevant guidance, **~10³ documents** | **19 documents**; proposal figure recorded as rejected | **REQUIRES HUMAN DECISION** (C3) |
| Textbooks | Neurology, psychiatry, internal medicine, **18 volumes** | **absent** | **REQUIRES HUMAN DECISION** (C5) |
| Currency pack | Manual ingestion, **20–40 documents** | **7 documents** | **REQUIRES HUMAN DECISION** (C4) |

**FROZEN constraints on all sources** (§1.7, §10): English-language; public and openly
licensed; no patient-identifiable data; credentialed resources optional and never load-bearing.

**Note on RAG² dependency.** Balanced retrieval draws **equal quotas from four corpora**
(RAG² §3.4). Dropping textbooks changes the quota structure of a stage the proposal declares
frozen — so C5 is not merely a missing corpus, it is a modification to the frozen baseline.

### 3.1 Document types — PROVISIONAL

The proposal does not enumerate admissible document types. The repository's rules:
errata and retraction notices **excluded** (metadata, not standalone evidence); preprints,
editorials, letters and comments **eligible but flagged**; stubs and body-less records
**abstract-level only**. These are consistent with the proposal and are adopted provisionally.

**Retracted articles remain eligible and retrievable, flagged — FROZEN.** Required by §4.3:
the currency gate γ returns 0 for retracted/withdrawn documents *at admission*, so they must
survive into the candidate set for that rejection to be measurable.

---

## 4. Date policy

### 4.1 The distinctions that matter

| Date | What it is | Risk if used wrongly |
|---|---|---|
| **Online / e-published** | First public appearance, ahead of print | Usually the true "when the world learned it" |
| **Print** | Issue date | Can lag online by **months to over a year** |
| **Version date** | This revision of a guideline | Confusing it with first publication makes old guidance look new |
| **Updated date** | Last touched, possibly trivially | Can make a stale document look current |
| **Retrieval date** | When *we* downloaded it | Never evidence age; provenance only |
| **Canonical date** | The single date the experiment uses | This is the dependent variable's foundation |

*Why this matters more here than in most projects:* the thesis measures preference **by
evidence age**. Choosing print over online date can shift a document across the June-2024
counterfactual boundary, silently changing which index it belongs to.

### 4.2 Requirements — FROZEN

- **Every passage inherits its parent document's publication date** (§5.1).
- Date imputation from issue metadata is applied, and the **per-corpus null rate is reported**.
- Counterfactual split anchored at the **June 2024 criteria revision** (§6.4).

### 4.3 Canonical-date rule — PROVISIONAL

The proposal says only "publication date" and never distinguishes online-first from print.
The repository's rule — *earliest real publication event (electronic first), excluding preprint
dates; JATS primary, PubMed fallback* — is a defensible reading and is adopted **provisionally**.

- `canonical_date_precision` ∈ {day, month, year}, **never upgraded** — **FROZEN**
  (no false precision: a year-only record stays year-only).
- Every canonical date traceable to its source via `canonical_date_type` + `date_source`.
- Current state is **not final**: dates rest on a PubMed fallback in this checkout and require
  re-running where the full parsed corpus exists.

---

## 5. Duplicates and versions

**FROZEN (proposal §5.1):**
- Exact deduplication by content hash.
- Near-duplicate detection by **MinHash over 5-gram shingles at Jaccard 0.85, retaining the
  newest and highest-tier cluster member** — stated precisely so that preprint-to-journal
  duplicates cannot let an old claim survive under a new date.
- Guideline passages **respect section boundaries** (a recommendation must never be separated
  from its qualifying conditions).

**REQUIRES HUMAN DECISION *(contradiction C7)*.** The implementation does exact content-hash
detection only, and **flags** duplicates rather than removing them. Flagging preserves
information and protects genuine versions — which is right — but the proposal's retention rule
is not implemented, and the specific failure mode P11 guards against is currently unguarded.

**Versions are not duplicates — FROZEN in principle.** A 2021 guideline and its 2024 revision
must remain distinct documents with an explicit relationship. The proposal requires a
per-passage **supersession pointer** (§5.1); only a free-text `supersession_note` on 19 CPG
rows exists. Format **NOT YET DETERMINED**.

---

## 6. Currency pack

**In plain language:** a small, hand-picked set of the documents that *define the current
answer* — the 2024 criteria, the lecanemab and donanemab recommendations, ARIA guidance, and
the contested April-2026 Cochrane review. If these are absent or unreachable, the system cannot
give an up-to-date answer no matter how good the admission policy is, and the thesis measures
nothing.

| | Proposal | Implemented |
|---|---|---|
| Size | **20–40 documents** (§5.1) | **7** |
| Ingestion | Manual | Manual + OA fetch, provenance recorded, MD5-pinned |
| Named anchors | 2024 criteria, lecanemab AUR, donanemab AUR, ARIA guidance, Cochrane CD016297 | all 5 present, plus blood-biomarker CPG and amyloid/tau PET AUC |

**REQUIRES HUMAN DECISION *(contradiction C4)*** — size is 7 against a specified 20–40, and the
repository's own README says 4 while its CSV says 7.

A **quota** decision is also outstanding and distinct from size: under balanced retrieval, does
the currency pack receive its own retrieval quota, or compete inside CPG-AD? Giving it a quota
modifies the frozen upstream stage; not giving it one may leave it unreachable. **NOT YET
DETERMINED** — the proposal does not address it.

---

## 7. Passage / chunk specification

**Proposal (§5.1, tagged `[DES]`):** sliding window at **256 tokens with 32-token overlap**,
sized against the article encoder's **512-token limit** with headroom for a prepended title and
section header.

**RAG² states no chunk size at all** (§A.3 says only "sliding window mechanism with overlap"),
so these numbers are the thesis's own decision, not inherited baseline fidelity.

**Implemented:** 256 **whitespace words** / 32-word overlap, never crossing a section boundary.

### REQUIRES HUMAN DECISION *(contradiction C6)* — the unit

*Why tokenizer choice matters, in plain language:* the retrieval model does not read words. It
splits text into sub-word pieces ("tokens") — `hippocampal` may become three. MedCPT's article
encoder accepts at most **512 tokens** and silently discards anything beyond. So the unit
decides how much text actually reaches the model:

| Unit | 256 units ≈ | Fits in 512 tokens? |
|---|---|---|
| MedCPT sub-word tokens (proposal) | 256 tokens | Yes, with large headroom |
| Whitespace words (implemented) | ~340–400 tokens | Usually — but headroom is much smaller |

The repository's stated reason for words is that it is "deterministic and dependency free".
That is an **engineering** rationale for a **research** parameter, and passage size affects
every retrieval score in the thesis.

**Recommendation:** measure before deciding. Tokenize a sample of existing chunks with the real
MedCPT tokenizer and report the distribution and truncation rate. If truncation is ~0%, keeping
words is defensible and should be *documented as a deliberate deviation*; if not, move to real
tokens. Either way the choice becomes evidence-based rather than convenient.

**FROZEN regardless of unit:** windows never cross a section boundary; title and section
heading composed separately from body text; every passage traceable to its document; exact
duplicate text flagged, never deleted.

---

## 8. Per-passage metadata schema

**FROZEN (§5.1).** Each passage carries: chunk and document identifiers; **source tier**;
publication date; journal and persistent identifier; **guideline family, version and
supersession pointer**; retraction and withdrawal flags; claim-classes; section header;
**character span**.

Implementation status against that list:

| Required | Status |
|---|---|
| chunk + document ids, journal, persistent ids, section header | present |
| publication date (+ precision, + source) | present |
| retraction flag, claim-class, guideline family | present |
| **source tier** | **absent** — `authority_tier_label` exists for 19 CPG rows only |
| **supersession pointer** | **absent** — free-text note on 19 rows |
| **character span** | **absent** — word offsets only |

**Source tier is NOT YET DETERMINED.** The proposal requires the field but never enumerates its
values. It cannot be invented here: authority ordering is a *tested variable* (ablation A12), so
choosing tiers silently would prejudge a thesis hypothesis. The field and its permitted values
require a decision; **no ordering or weight may be encoded** in it.

---

## 9. Knowledge-boundary requirements — NOT YET DETERMINED

**In plain language:** we must be able to show that evidence we call "new" was genuinely
unavailable to the AI model when it was trained. A 2025 publication date does **not** prove the
model did not already know the fact.

What the proposal provides: the *mechanism* depends on "pre-training-era priors" (§1.2), and it
fixes the generators as **Llama-3-8B-Instruct** (primary) and **Meerkat-7B** (secondary) (§5.3).

What the proposal does **not** provide, verified by exhaustive search:

- Any pre-training cutoff date for any model.
- Any procedure for the knowledge-boundary audit.
- Any mapping from corpus dates to model knowledge.

**This cannot be resolved by inspection and must not be guessed.** A cutoff asserted from
memory would be fabricated methodology sitting under a central thesis claim.

---

## 10. Retrieval configuration relevant to the corpus audit

**FROZEN (§5.3, RAG² §3.4):** MedCPT Query Encoder, MedCPT Article Encoder, MedCPT
Cross-Encoder reranker; balanced retrieval with equal quota per source corpus; retrieval on the
rationale, reranking on the original query.

**PROVISIONAL:** exact flat inner-product search rather than approximate search. Well justified
— validity control V3 requires byte-identical candidate replay, which an approximate index
cannot guarantee — and adopted provisionally.

**Retrieval-reachability audit — NOT YET DETERMINED / recommended addition.** The proposal
specifies **no Stage-1 retrieval audit**; Recall@k and nDCG@10 appear only as Stage-5
evaluation metrics against pooled human judgements (§7.1). This specification **recommends**
adding a currency-pack reachability check as a Stage-1 gate, but records it as a
*recommendation by the assistant*, not a proposal requirement.

---

## 11. Manual quality audit

**FROZEN (§5.1):** claim-class taxonomy of **40–80 classes**; assignment by keyword-and-embedding
matching; **manual verification of a 300-passage RANDOM sample, reporting precision and recall.**

**The proposal says random, not stratified.** A stratified sample across years, source types or
tiers would give better coverage of rare strata — but it is a **change to a specified method**,
not a refinement of it, and it changes what the reported precision and recall mean (per-stratum
rather than corpus-wide). Flagged as **REQUIRES HUMAN DECISION** *(contradiction C10)* so it is
not silently substituted.

Distinct and not to be confused: the **judge validation** sample (§7.1) *is* stratified, with
Cohen's κ reported and size `[OPEN]`. That belongs to Stage 5, not Stage 1.

---

## 12. Reproducibility and frozen snapshot

**FROZEN in principle (§6.4, §7.3):** the corpus is a dated snapshot of live databases; the
counterfactual corpus-update procedure is undefined without a fixed snapshot boundary.

**PROVISIONAL / recommended,** as the proposal does not enumerate these: record E-utilities
query strings verbatim; MeSH explosion settings; pull timestamps; per-collection record counts;
tool and model versions; content digests. Raw documents preserved separately from processed
ones. Nothing silently discarded; every exclusion carries a recorded reason.

**AD-CORPUS-v1 is immutable once frozen.** Any later material change produces **AD-CORPUS-v2**.

---

## 13. Decision register (revised after Phase 1A)

Derived from architecture requirements R-A…R-N, not from prose comparison. Full reasoning in
`STAGE_1_ARCHITECTURE_TO_CORPUS_REQUIREMENTS.md`.

### Resolved by the architecture analysis — no longer open

| ID | Prior framing | Revised status |
|---|---|---|
| C2 | Temporal window: none vs 2021–2026 | **Largely dissolved.** R-I (counterfactual index) is satisfied: 18,707 pre / 22,789 post June 2024. Residual issue folded into C4′ |
| C3 | CPG-AD size ~10³ vs 19 | **Largely dissolved.** The proposal's figure is tagged `[OPEN]`; the architecture needs version relations and claim-class coverage (R-C, R-L), not count. Conditional on the coverage measurement in A2 |
| — | Matched old/new pairs inside this corpus | **NOT REQUIRED.** Matched pairs are external by design (§5.2 provenance firewall); building them here would breach it |
| — | Corpus volume for statistical power | **NOT REQUIRED of this corpus.** The primary claim (H1–H3) uses MedChangeQA, not this corpus |

### Genuinely open — REQUIRES HUMAN DECISION

| ID | Decision | Why it is material |
|---|---|---|
| **C1′** | Whether corpus scope must be widened to populate under-covered claim classes (pharmacotherapy, ARIA, eligibility, deprescribing) | R-L: an unpopulated class is a dead γ branch. **Decide after measuring A2, not before** |
| **C4′** | Currency-pack and historical-anchor composition — judged by **claim-class coverage and dispute completeness**, not document count | R-E, R-L. Supersedes the old "7 vs 20–40" framing |
| **C5** | Whether time-invariant content (R-D) is satisfied by textbooks, by time-invariant PubMed content, or is unnecessary | H7 non-inferiority depends on R-D |
| **C6** | Chunk unit: MedCPT subword tokens (specified) vs whitespace words (implemented) | Changes what the frozen retriever sees, systematically and for every arm |
| **C7** | Near-duplicate policy: implement the specified MinHash rule, or keep exact-hash-and-flag | §5.1 states the exact failure mode: a preprint-to-journal duplicate lets an old claim survive under a new date — direct corruption of the dependent variable |
| **C10′** | Audit sampling: random (as specified) vs stratified by claim class | γ's error is **per-class**, which is an architectural argument for stratification the prose comparison missed. Still a change to a specified method |
| **A3** | Contest-window length, and whether rebuttal documents are ingested | Determines whether the contested state can fire at all. RQ5 and CONTESTED-AD rest on it |

### NOT YET DETERMINED — missing from the source material

| Item | Note |
|---|---|
| Pre-training cutoff and knowledge-boundary procedure | No cutoff or procedure appears anywhere in the proposal. **Must not be invented** |
| Source-tier values | Field required by §5.1; values never enumerated. Ordering must not be encoded (A12) |
| Supersession-pointer format | Required by R-C; only free-text notes exist |
| Currency-pack retrieval quota | Not addressed by the proposal; a quota modifies the frozen upstream stage |
| **How H7 non-inferiority retrieves** over MedQA/MedMCQA/MMLU-Med given an AD-only corpus | Bears directly on whether the corpus needs non-AD content at all (A7) |

### Measurements that convert opinion into evidence

These are **not** decisions. Each replaces a guess with a number, and all need the machine
holding the corpus:

1. **Per-claim-class passage counts** → resolves C1′ and informs C4′
2. **Retrieval reachability** of currency-pack and decisive evidence → resolves A1
3. **MedCPT token-length distribution and truncation rate** over built chunks → resolves C6
4. **Contested-pair check**: do two opposing, same-class, in-window, above-tier passages exist? → resolves A3

**Gate:** this specification is **NOT FROZEN**. The four measurements should run before the
seven REQUIRES HUMAN DECISION items are settled, because each measurement removes guesswork
from the decision it feeds.
