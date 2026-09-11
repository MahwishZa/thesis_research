# Stage 1 · Phase 1A — Architecture-to-Corpus Requirements Analysis

**Date:** 2026-09-11 · **Method:** requirements derived *backwards* from the thesis research
design, not forwards from any existing implementation or from RAG²'s corpus.

**Authority hierarchy applied:** (1) thesis research questions, hypotheses and architecture →
(2) RAG² as methodological precedent → (3) public biomedical sources as candidates →
(4) existing implementation, evaluated against the derived requirements and *not* authoritative.

Line references are to the extracted text of `MS_Thesis_Proposal.pdf`
(MD5 `f8de7826ce5073bb707a4c5a947b772c`).

---

## STEP 0 — The finding that reframes Stage 1

**Only half the evaluation sets depend on this corpus at all.** Mapping every set in §5.2 to its
provenance and to the hypothesis it serves:

| Evaluation set | Serves | Needs the AD corpus? |
|---|---|---|
| FRB-PAIRS (150–300 pairs) | **H1, H2 — the primary claim** | **No** — drawn from MedChangeQA, externally authored |
| FRB-PERMUTED | H4 validity gate | **No** — same material |
| NEG-CONTROL (~80) | spurious-recency detector | **Yes** — AD claims stable across the window |
| CONTESTED-AD (~40) | RQ5 contested state | **Yes** |
| ADCUR-QA (~120) | domain replication | **Yes** |
| AD-VIGNETTE (60–80) | blinded expert evaluation | **Yes** |
| MedQA / MedMCQA / MMLU-Med | H7 non-inferiority | **Undetermined — see A7 below** |
| MedXpertQA | unsaturated accuracy | **Undetermined — see A7** |

The proposal states this directly: the provenance firewall (§5.2) exists so that the primary
claim rests on external material, and the thesis-curated corpus supports "replication and case
study only."

**Consequence for corpus design.** The corpus does not have to be big enough to establish the
thesis's headline result. It has to be *structurally correct* for four specific jobs: the
counterfactual index, the contested case study, domain replication, and expert vignettes.
This is the strongest possible argument against optimising for size (Step 10), and it means
corpus scope decisions are lower-risk to the primary contribution than the previous
reconciliation implied.

---

## STEP 1 — Research requirements → corpus capabilities

| Research requirement | Corpus capability required | Why the architecture/experiment needs it | Evidence |
|---|---|---|---|
| **R-A.** γ currency function computes `2^(-(t_q - date(s))/H)` | **Every passage carries a usable date** | γ is arithmetic on the date. A passage with no date cannot be scored at all | §4.3 L559–562 |
| **R-B.** γ returns 0 for retracted/withdrawn | Retraction/withdrawal flags; retracted items **retained and retrievable** | The gate is measured *at admission*; a deleted document can never demonstrate rejection | §4.3, §4.5 L613 |
| **R-C.** γ applies a supersession discount δ | **Superseded documents remain in the corpus**, with a supersession relation | Down-weighting requires the superseded item to still be present. Deletion makes the branch unreachable | §4.3 L561 |
| **R-D.** ψ(q)=0 ⇒ γ=1 for time-invariant questions | **Time-invariant AD content** (APOE biology, neuropathological staging, cholinesterase pharmacology) | Without it the non-inferiority hypothesis H7 cannot be tested in-domain | §4.3 L570–573 |
| **R-E.** Contested state fires when two passages of one claim-class carry **opposing conclusions**, both **within a contest window**, both **above a minimum source tier** | **Both sides of a live dispute**, dated, tiered, same claim-class | One-sided coverage means the state never fires and RQ5 is untestable | §4.4 L577–580 |
| **R-F.** Contested tested **before** supersession | Dates precise enough to order the dispute | Reversing presents a live controversy as resolved | §4.4 L581–584 |
| **R-G.** A(s) includes τ(s) source authority, **tested not assumed** | **Source-type/authority metadata as a label with no ordering encoded** | Ablation A12 varies and removes the ordering; a baked-in ordering prejudges the hypothesis | §4.5 L597–601, §7.4 A12 |
| **R-H.** σ(s) entailment over ≤8 atomic claims or verbalised options | Passages that are **self-contained enough to entail a claim** | A fragment severed from its qualifying condition entails the wrong thing | §4.2, §5.1 L731–733 |
| **R-I.** Counterfactual corpus-update: one index of documents **dated before the June 2024 criteria revision**, one containing all; report update-flip rate | **Sufficient pre- and post-June-2024 AD evidence to answer the evaluation questions on both sides** | If the pre-index cannot answer, the flip rate measures corpus emptiness, not pipeline responsiveness | §6.4 L956–962 |
| **R-J.** Upstream frozen; candidate set cached and replayed byte-identically (V3) | **Corpus immutable once frozen**; deterministic passage identity | Arms must be distinct functions over one candidate list | §6.3, §4.6 V3 |
| **R-K.** V5 negative control: claims **stable** across the window | **AD claims that did not change**, distinguishable from those that did | Separates real claim-recency from a general preference for newer, better-formatted prose | §4.6 V5 |
| **R-L.** Claim-class taxonomy of 40–80 classes: biomarker criteria, staging, plasma biomarkers, amyloid imaging, **eligibility, ARIA monitoring, APOE, pharmacotherapy, behavioural management, deprescribing**, differential diagnosis | **Content actually covering those classes** | γ and the contested test operate per claim-class. An empty class is a dead branch | §5.1 L739–743 |
| **R-M.** SCAF admits from the reranked candidate set only (Algorithm 1 line 1 input `S_rr`) | **Evidence must be retrievable by the frozen retriever** | **SCAF cannot correct evidence retrieval never returns.** Reachability is architectural, not cosmetic | §4.5 Algorithm 1 |
| **R-N.** Expert evaluation on open-ended AD vignettes | Corpus able to support realistic clinical vignette answering | Otherwise the expert rates retrieval failure, not admission policy | §7.5 |

---

## STEP 2 — Derived corpus properties

| Property | Class | Reasoning |
|---|---|---|
| Alzheimer's-specific content | **REQUIRED** | Every corpus-dependent evaluation set is AD-scoped (§5.2) |
| Publication dates on every passage | **REQUIRED** | R-A: γ is undefined without a date |
| Document-level provenance | **REQUIRED** | R-J reproducibility; every passage traceable to its document |
| Source-type / authority metadata | **REQUIRED** as an unordered label | R-G; ordering must **not** be encoded (A12) |
| Recent evidence (post-June-2024) | **REQUIRED** | R-I full index; the "new" side of every comparison |
| Historical evidence (pre-June-2024) | **REQUIRED** | R-I pre-index must be able to answer |
| Evidence spanning meaningful knowledge changes | **REQUIRED** | R-I update-flip rate is defined on changed answers |
| Superseded evidence retained | **REQUIRED** | R-C: the discount branch is unreachable otherwise |
| Retracted evidence retained and flagged | **REQUIRED** | R-B |
| Both sides of at least one live dispute | **REQUIRED** | R-E: RQ5 is otherwise untestable |
| Updated/revised guidelines with version relations | **REQUIRED** | R-C, R-F; also §5.1 metadata schema |
| Document/version relationships (supersession pointer) | **REQUIRED** | R-C |
| Retrievable passages (reachability demonstrated) | **REQUIRED** | R-M — architectural, and currently unmeasured |
| Time-invariant AD content | **REQUIRED** | R-D; H7 depends on it |
| Stable-claim (negative control) content | **REQUIRED** | R-K / V5 |
| Coverage of the claim-class taxonomy | **REQUIRED** | R-L |
| Clinical reasoning / diagnostic content | **DESIRABLE** | Serves AD-VIGNETTE (R-N) but is not required by γ, the contested test, or R-I |
| Disease/mechanism (basic-science) evidence | **NOT REQUIRED** | No architectural consumer. Adds retrieval noise without serving any hypothesis |
| Matched old/new evidence **inside this corpus** | **NOT REQUIRED** | Matched pairs are FRB-PAIRS, external by design (§5.2 firewall). Building them here would breach the firewall |
| Lexical/semantic diversity | **DESIRABLE** | Helps vignette realism; no hypothesis depends on it |
| Volume for statistical power | **NOT REQUIRED of this corpus** | Power lives in FRB-PAIRS (external). Corpus needs sufficiency per §5.2 set sizes, not scale |
| Textbooks as a source class | **REQUIRES HUMAN DECISION** | Serves R-D (stable, time-invariant, high-authority) but its necessity is a research judgement |
| Total corpus size | **REQUIRES HUMAN DECISION** | Proposal scale figures are tagged `[OPEN]` (§5.1 table header) — estimates, not targets |

---

## STEP 3 — Architecture data-flow trace

`corpus → processing → passages → index → retrieval → context → model → γ/contested → SCAF → answer`

| Stage | Assumption the corpus must satisfy | Status |
|---|---|---|
| **Document** | Date, persistent id, source type, licence, retraction status, guideline family + version | dates/ids/licence present; **source tier and supersession pointer absent** |
| **Processing** | Section structure preserved; a recommendation never severed from its qualifying condition | implemented (windows never cross a section boundary) |
| **Passage** | **Date must survive to passage level** — γ scores passages, not documents | implemented (`canonical_date` on every chunk) |
| **Passage** | Claim-class must survive | implemented (`claim_class` field) |
| **Passage** | Self-containment sufficient for entailment (R-H) | **unverified** — no readability/self-containment audit exists |
| **Index** | Deterministic identity and order for byte-identical replay (V3) | implemented (exact flat search, ties broken on `chunk_id`) |
| **Index** | Passage text must fit the MedCPT article encoder's **512-token** limit | **unverified — see Step 8** |
| **Retrieval** | **Target evidence must actually be returned** | **unmeasured — the single largest architectural gap** |
| **Context** | Balanced quota across source classes so small decisive corpora are not drowned | implemented; **currency-pack quota undefined** |
| **γ / contested** | Both dispute sides co-present, in-window, same claim-class, above minimum tier | **at risk — see A3** |
| **SCAF** | Can only act on what retrieval returned | **R-M unsatisfiable until reachability is measured** |

**Three architecture-level answers to the questions posed:**

- *Does SCAF require temporal metadata?* **Yes, unconditionally.** γ is arithmetic on dates.
- *Can SCAF correct evidence retrieval never returns?* **No.** Algorithm 1 takes `S_rr` as input.
  Everything SCAF does is a filter over an already-retrieved set.
- *Can the architecture distinguish genuine updates from duplicates?* **Only if a supersession
  relation exists.** None is implemented beyond free-text notes on 19 rows.

---

## STEP 4 — Using RAG² correctly

### A. Principles that should be adapted

| RAG² approach | Why relevant | What must change for AD | Resulting thesis design |
|---|---|---|---|
| Balanced retrieval across corpora (§3.4) | A PubMed-trained dense retriever drowns small corpora; the currency pack is small and decisive | Quota classes should follow the **thesis's** evidence classes, not RAG²'s four fixed corpora | Balanced quota over thesis source classes, with the currency pack's quota an explicit decision |
| Frozen retriever + reranker (§3.3–3.4) | Internal validity: the thesis isolates admission | None — adopt as-is | MedCPT encoders + cross-encoder, frozen |
| Cached candidate replay | Enables V3 byte-identical arm comparison | Extend to carry temporal metadata | Implemented |

### B. Implementation details that do NOT necessarily transfer

| RAG² detail | Why it should not simply transfer |
|---|---|
| **37.6M docs / 116.7M passages / 564.2 GB** (Table A1) | Sized for general medical QA. This thesis is domain-scoped by design (§5.1) and its primary claim does not use this corpus at all (Step 0) |
| **Exactly four corpora incl. 18 textbooks** | A structural choice for RAG²'s coverage problem. The thesis's source classes should follow its own claim-class taxonomy and tier requirements |
| **No temporal representation anywhere** | Directly inverted: this thesis requires dates, versions and supersession at every level |
| **Passage labelled individually; one snippet at a time** (RAG² Limitation) | The contested state is intrinsically **pairwise**. SCAF handles it outside the filter, so this limit must not be inherited |

### C. Components this thesis genuinely should reproduce

Only where there is a methodological reason: the **frozen MedCPT retrieval and reranking
stack** (§5.3 — baseline fidelity is what licenses attributing differences to admission), and
the **rationale-as-query / original-query-as-rerank** asymmetry (RAG² §3.3–3.4), because the
thesis freezes the upstream pipeline to RAG².

**Chunking is explicitly *not* in this category.** RAG² §A.3 states only "a sliding window
mechanism with overlap" and **gives no token size anywhere**. The 256/32 figures are the
thesis's own `[DES]` decision (§5.1 L726), not inherited fidelity.

---

## STEP 5 — Candidate source evaluation

Judged solely by: *does this source satisfy a demonstrated requirement from Step 1?*

| Source class | Serves | Temporal metadata | Provenance | Verdict |
|---|---|---|---|---|
| **PubMed abstracts** | R-I (both sides of June 2024), R-K, R-L | Structured, reliable | PMID/DOI, E-utilities reproducible | **Include** — the volume backbone for the counterfactual index |
| **PMC OA full text** | R-H (self-contained passages), R-N | JATS `pub-date` set, high quality | PMCID + licence + MD5 | **Include** — needed where abstracts cannot entail a claim |
| **Clinical practice guidelines** | R-C, R-E, R-F, R-G, R-L | Version/issue dates, sometimes poor in PDFs | Organisation + version | **Include** — the only source carrying genuine supersession relations |
| **Consensus / appropriate-use statements** | R-E (one side of the dispute), R-L | Good | Society attribution | **Include** |
| **Systematic reviews (incl. Cochrane)** | R-E (the other side) | Precise, versioned | DOI, review-group | **Include — but never classified as CPG** |
| **Direct rebuttals/commentaries to a contested review** | **R-E specifically** | Precise (days) | Journal | **REQUIRES HUMAN DECISION — see A3** |
| **Medical textbooks** | R-D (time-invariant, stable, high-authority) | Edition dates only; coarse | Publisher | **REQUIRES HUMAN DECISION** — serves a real requirement but licensing and date precision are weak |
| **Preprints** | none uniquely | Poor (versioned, revised in place) | weak | **Exclude from the curated layers**; if present in PubMed, flag as low tier |
| **Basic-science / animal-model literature** | none | — | — | **Exclude** — no architectural consumer; pure retrieval noise |

---

## STEP 6 — Existing implementation assessed against derived requirements

Re-examined against Step 1, **not** against the proposal's prose. Several previously recorded
"contradictions" dissolve under this test; others harden.

| ID | Requirement | Current implementation | Meets? | Required change / research impact |
|---|---|---|---|---|
| **C1** scope | R-L: cover the claim-class taxonomy (incl. pharmacotherapy, ARIA, eligibility, deprescribing, behavioural management) | Scope restricted to AD ∩ clinical-reasoning/diagnosis | **PARTIAL** | **Reframed.** The issue is not "MeSH vs reasoning" — it is whether the taxonomy classes are *populated*. Pharmacotherapy/ARIA/deprescribing sit outside a diagnosis-restricted query. **Test empirically: per-claim-class passage counts.** A class with near-zero passages is a dead γ branch |
| **C2** date window | R-I: enough evidence either side of June 2024 | 2021-08-30 → 2026-08-30 | **YES for R-I** (18,707 pre / 22,789 post) | **Largely dissolves.** R-I is satisfied. The residual issue is R-C: pre-2021 *superseded* documents (e.g. 2011 criteria) are absent, so the supersession discount has fewer real targets. Narrow, addressable by targeted ingestion |
| **C3** CPG size | R-C, R-E, R-G, R-L — relations and coverage, not count | 19 curated, all with provenance, licence, hash | **LIKELY YES** | **Largely dissolves.** The proposal's "~10³" is tagged `[OPEN]` — an estimate. 19 fully-provenanced guidelines with version relations serve R-C better than 1,000 unmanaged ones. **Conditional on demonstrating claim-class coverage** |
| **C4** currency pack | R-E, R-L: cover every changed claim-class **and both dispute sides** | 7 documents | **PARTIAL** | **Reframed from count to coverage.** "20–40" is an `[OPEN]` estimate; 7 is not automatically wrong. The real test is per-claim-class coverage and dispute completeness → **A3** |
| **C5** textbooks | R-D time-invariant content | absent | **NO** | Genuine gap against R-D, but textbooks are one way to satisfy it; time-invariant PubMed content is another. **REQUIRES HUMAN DECISION** |
| **C6** chunk unit | Index stage: passages must fit MedCPT's 512-token limit | 256 whitespace words (~340–400 subword tokens) | **UNVERIFIED** | **Hardens.** Not a doc mismatch but an unmeasured truncation risk → Step 8 |
| **C7** duplicates | R-C: distinguish genuine updates from duplicates | Exact hash only; duplicates flagged, not resolved | **PARTIAL** | **Hardens — architecturally critical.** §5.1 L729–731 states the precise failure: preprint-to-journal duplicates let an old claim survive under a new date. That directly corrupts the dependent variable |
| **C10** audit sampling | R-L: γ error is proportional to **per-class** tagging error | not implemented | **N/A** | **Reframed.** Since γ's damage is per-class, per-class precision is what the architecture needs — an argument *for* stratification that the earlier framing missed. Still a spec change → **REQUIRES HUMAN DECISION** |
| **new** | R-M: reachability | no reachability measurement exists | **NO** | **Now architecturally required**, not a nice-to-have: SCAF cannot admit what retrieval never returns |
| **new** | R-G: source tier | no `source_tier` column; `authority_tier_label` on 19 CPG rows only | **NO** | Field required by §5.1; values undefined in the proposal → **NOT YET DETERMINED** |

---

## STEP 7 — Temporal validity

| Question | Answer from the source material |
|---|---|
| How are dates represented? | Canonical date + explicit precision {day, month, year}, never upgraded; type and source recorded. **FROZEN** in principle (§5.1) |
| Which date should be used? | Proposal says only "publication date" (§5.1 L734). Online-first vs print vs version is **NOT specified**. Repo uses earliest electronic, excluding preprints — **PROVISIONAL** |
| How are old/new defined? | **Two distinct definitions coexist.** For R-I it is the **June 2024 criteria revision** (§6.4). For γ it is a **continuous decay** relative to `t_q` with half-life H (§4.3). These are different mechanisms and must not be conflated |
| Historical evidence sufficient? | For R-I, **yes** (18,707 pre-June-2024). For R-C supersession targets, **weaker** — nothing before 2021 |
| Current evidence sufficient? | Volume yes (22,789 post). **Coverage per claim-class unverified** |
| Can meaningful knowledge changes be identified? | Only via a supersession relation, which is **not implemented** beyond free-text notes |
| Might the model already know the "new" evidence? | **NOT YET DETERMINED.** The proposal fixes generators (Llama-3-8B-Instruct, Meerkat-7B, §5.3) but states **no pre-training cutoff and no knowledge-boundary procedure anywhere.** Must not be invented |
| Can retrieval surface the relevant evidence? | **Unmeasured** (R-M) |
| Must superseded evidence remain? | **Yes — FROZEN.** γ's supersession branch down-weights rather than deletes (§4.3) |
| Can old/new be matched fairly? | Not this corpus's job — matched pairs are external by design (§5.2) |

---

## STEP 8 — Chunking

**Specification (§5.1 L726–727, tagged `[DES]`):** sliding window at **256 tokens with 32-token
overlap**, sized against the article encoder's **512-token limit** with headroom for a prepended
title and section header.

**RAG² imposes no chunk size** (§A.3: "a sliding window mechanism with overlap" — no number).

**Implementation mismatch — reported, not fixed:** `pmc/build_chunks.py` and
`chunk_stats.json` use `window_words: 256` / `overlap_words: 32` — **whitespace words, not
model tokens.** The repository's own README states the reason as "deterministic and dependency
free." That is an engineering rationale applied to a research parameter.

| Unit | 256 units ≈ | Against the 512-token limit |
|---|---|---|
| MedCPT subword tokens (specified) | 256 tokens | Large headroom, as the proposal intends |
| Whitespace words (implemented) | ~340–400 tokens | Usually fits; headroom much reduced; **truncation rate unmeasured** |

**Does the architecture require a particular tokenizer?** Yes — the **MedCPT Article Encoder's**
tokenizer, because that is the model whose 512-token limit the specification is sized against
(§5.3 freezes MedCPT).

**Not fixed, per instruction.** The resolving evidence is a measurement, not an opinion:
tokenize a sample of built chunks with the real MedCPT tokenizer and report the token-length
distribution and truncation rate. **This could not be run here** — `pmc/chunks/chunks.jsonl` is
gitignored and absent from this container, and the MedCPT weights are not present. It must run
on the machine holding the corpus.

---

## STEP 10 — Sizing posture

No size target is adopted. The proposal's scale figures (2–4×10⁵ abstracts, ~10³ CPG,
20–40 currency documents) are **explicitly tagged `[OPEN]`** in the §5.1 table header —
estimates, not requirements. Combined with Step 0 (the primary claim does not use this corpus),
the governing criterion is **structural sufficiency**, measured as:

1. Per-claim-class passage coverage (no dead γ branches)
2. Answerability of the evaluation sets from the **pre**-June-2024 index (R-I)
3. Demonstrated retrieval reachability of decisive evidence (R-M)
4. Completeness of at least one contested claim, both sides (R-E)

None of these is a document count.

---

## STEP 11 — Decision separation

**Engineering decisions taken freely:** directory layout, logging, retries, pagination, hashing,
config organisation, determinism mechanics.

**Research decisions NOT taken here** — recorded as REQUIRES HUMAN DECISION or NOT YET
DETERMINED in `STAGE_1_CORPUS_SPECIFICATION.md` §13: corpus scope, source inclusion, temporal
coverage, evidence composition, currency-pack definition, old/new definitions, duplicate/version
policy, knowledge-boundary assumptions, chunking specification, retrieval configuration, audit
methodology.

---

## Architecture-derived issues, ranked by effect on validity

**A1 — Retrieval reachability is unmeasured (R-M).** SCAF is a filter over `S_rr`; it cannot
admit what retrieval never returns. If decisive evidence is unreachable, every SCAF result is
bounded near zero and a null is uninterpretable. Cheap to measure, and the proposal's own cited
expert evaluation (22% top-16 relevance; 31% of queries returning nothing relevant) makes a poor
result plausible.

**A2 — Claim-class coverage is unmeasured (R-L).** γ and the contested test operate per class.
An unpopulated class is a dead branch. A diagnosis-restricted scope plausibly under-covers
pharmacotherapy, ARIA, eligibility and deprescribing — classes where the 2023–2026 changes
actually happened.

**A3 — The contested state may be unfirable (R-E).** The curated layers hold the Cochrane
review and the appropriate-use recommendations, but **no deliberately ingested rebuttals**
(Alzheimer's Association, UK DRI, *Lancet* commentary — named in §1.3 as contesting "within
days"). Whether the AURs (2023, 2025) count as the opposing side depends entirely on the
**contest window, which the proposal leaves `[OPEN]`** (§4.4 L578). A short window excludes them
and leaves no counterparty; a window wide enough to include them spans three years, which
weakens the claim that the dispute is contemporaneous. **RQ5 and CONTESTED-AD (~40 items) rest
on this.**

**A4 — Near-duplicate resolution is unimplemented (R-C, C7).** The specified MinHash rule exists
precisely to stop a preprint-to-journal duplicate letting an old claim survive under a new date.
That failure corrupts the dependent variable directly.

**A5 — Supersession relation absent (R-C).** Without it the architecture cannot distinguish a
genuine update from a duplicate, and γ's supersession branch has no input.

**A6 — Chunk truncation unmeasured (Step 8).** Unresolved silently, it changes what the frozen
retriever sees for every arm equally — a systematic, not random, effect.

**A7 — Non-inferiority retrieval undefined.** H7 tests accuracy on MedQA/MedMCQA/MMLU-Med, but
the proposal specifies **only** a domain-scoped AD corpus (§5.1) used identically by every arm
(§6.3). Retrieving cardiology questions from an AD-only index is not defined behaviour. The
proposal never states whether these run closed-book, over a general corpus, or otherwise.
**NOT YET DETERMINED**, and it bears directly on whether the corpus needs non-AD content at all.

**A8 — Knowledge boundary has no cutoff and no procedure.** Not inventable. **NOT YET
DETERMINED.**
