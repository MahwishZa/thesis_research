# Stage 1 · Phase 1A — Claim-Class Taxonomy Specification

**Status: PROPOSED — NOT APPROVED.** No assignment mechanism may be built against it yet.
**Derived from:** `MS_Thesis_Proposal.pdf` (MD5 `f8de7826…`) §4.3, §4.4, §5.1 — **not** from the
21 literals currently in the repository, which are evidence about the implementation only.

Machine-readable: `configs/taxonomy/claim_classes_v0.draft.json`,
`configs/taxonomy/existing_label_mapping_v0.draft.json`

---

## A. Research basis

The taxonomy is not organisational tidiness. Three architecture mechanisms consume claim
classes, and each breaks without them. Every mention of claim-class in the proposal was
enumerated (7 occurrences); three are functional consumers:

| Consumer | What it does with the class | Cite |
|---|---|---|
| **γ supersession** | Supersession is evaluated per claim class; tagging error propagates directly into the currency score, which is why §4.3 confines it to down-weighting rather than deletion | §4.3 L564–568 |
| **Contested state** | "A claim-class **is contested** when two passages **of that class** carry opposing conclusions" | §4.4 L577 |
| **ψ(q) time-sensitivity** | "`[OPEN]` Whether ψ(q) is implemented **as a rule over the claim-class taxonomy** or as a trained classifier" | §4.3 L573–574 |

Plus two structural uses: `K(s)` — "**Claim-classes** assigned to s", plural (§4.1 L495) — and the
per-passage metadata schema (§5.1 L725).

**The granularity rule follows from the contested test, not from taste.** Because the contest
fires on *same-class opposition*:

- **Too coarse** → false contests. If `pharmacotherapy` is one class, "donepezil is effective"
  and "anti-amyloid mAbs show little benefit" are same-class opposing conclusions about
  different decisions. This is precisely the scope-difference failure §4.4 already warns of.
- **Too fine** → missed contests. If the class is `lecanemab-18-month-CDR-SB-outcome`, an update
  and the thing it updates may share no class, and the state never fires.

> **Granularity rule.** A class must be **fine enough that two passages sharing it make claims
> about the same clinical decision**, and **coarse enough that an update and the thing it
> updates share one class.**

This rule, not a target count, generated the taxonomy below.

---

## B. Proposed taxonomy — 9 groups, 50 classes

Built by subdividing the **11 topics the proposal names** in §5.1. `REQUIRED` = traceable to a
named topic; `DESIRABLE` = reasoned extension within a named topic's scope.

| Group | Covers named topic(s) | Classes |
|---|---|---|
| **AD-CRIT** Diagnostic criteria and staging | biomarker criteria, staging | 5 |
| **AD-BIOM** Biomarkers and imaging | plasma biomarkers, amyloid imaging | 10 |
| **AD-APOE** APOE | APOE | 3 |
| **AD-PHARM** Pharmacotherapy | pharmacotherapy | 7 |
| **AD-ELIG** Treatment eligibility | eligibility | 4 |
| **AD-ARIA** ARIA | ARIA monitoring | 4 |
| **AD-BEHAV** Behavioural management | behavioural management | 5 |
| **AD-DEPR** Deprescribing | deprescribing | 4 |
| **AD-DDX** Differential diagnosis | differential diagnosis | 8 |

**Totals: 50 classes — 40 REQUIRED, 10 DESIRABLE.** All 11 named topics are covered; no group
exists without a named parent.

Representative leaves (full definitions in the JSON):

- **AD-PHARM** — `lecanemab-efficacy`, `donanemab-efficacy`, `anti-amyloid-class-efficacy`,
  `anti-amyloid-benefit-harm-balance`, `cholinesterase-inhibitors`, `memantine`,
  `other-symptomatic-pharmacotherapy`
- **AD-ARIA** — `aria-definition-classification`, `aria-mri-monitoring-schedule`,
  `aria-risk-by-apoe`, `aria-management-discontinuation`
- **AD-DEPR** — `deprescribing-cholinesterase-memantine`, `deprescribing-antipsychotics`,
  `anticholinergic-burden`, `end-of-life-medication-review`
- **AD-BEHAV** — `bpsd-nonpharmacological`, `bpsd-antipsychotics`,
  `agitation-specific-management`, `sleep-disturbance-management`,
  `caregiver-support-interventions`

Each class carries: stable id, group, canonical name, definition, status, the named topic it
derives from, and empty `synonyms` / `inclusion_rules` / `exclusion_rules` fields — deliberately
empty, because populating them is assignment-mechanism work and must follow approval.

---

## Reconciling the 40–80 figure — verified wording

**Exact text (§5.1 L739–741), quoted:**

> "Claim-class tagging. A hand-built taxonomy of forty to eighty classes covering biomarker
> criteria, staging, plasma biomarkers, amyloid imaging, eligibility, ARIA monitoring, APOE,
> pharmacotherapy, behavioural management, deprescribing and differential diagnosis."

**Tag status — verified, and it matters:** this sentence carries **no tag**. The `[ASM]` that
follows attaches to the *next* sentence ("This is the weakest link in the pipeline"). Under the
proposal's own convention (§Evidential Labelling), the figure is therefore **not** `[DES]`
(justified design decision), **not** `[ASM]` (flagged assumption), and **not** `[OPEN]`
(flagged as undetermined). It is an **untagged methodological assertion**.

Note the contrast: the §5.1 corpus table explicitly tags its scale column `[OPEN]`. The taxonomy
size is not tagged at all.

**Classification: an approximate expected range, not a verified hard requirement.** Reasons:

1. The same sentence names **11** topics. Reaching 40–80 requires a 4–7× subdivision, and the
   proposal supplies **no subdivision principle**.
2. "forty to eighty" is a 2× span — the shape of an estimate, not a specification.
3. Nothing downstream references the count. No hypothesis, metric, test or ablation depends on it.

**How the proposal is satisfied without artificial categories.** The taxonomy was generated by
applying the granularity rule to the 11 named topics, then counted. It landed at **50**, and the
`REQUIRED` subset alone is **exactly 40** — the stated lower bound, arrived at independently
rather than targeted. No class was created to reach a number; had the rule produced 30, this
document would report 30 and flag the discrepancy.

---

## C. Mapping — existing 21 labels → proposed classes

| Resolution | Count | Meaning |
|---|---|---|
| **GROUP** | 9 | Existing label is **too coarse** — maps to a group, not a class. Needs re-tagging |
| **LEAF** | 6 | Maps cleanly to one class |
| **OUT** | 5 | **No parent among the 11 named topics** — a scope decision |
| **ERROR** | 1 | **Category error** |

| Existing | → | Resolution | Note |
|---|---|---|---|
| `pharmacotherapy` (9) | AD-PHARM | GROUP | Contest fires per decision, not per drug class |
| `differential-diagnosis` (9) | AD-DDX | GROUP | Must resolve to the specific alternative |
| `aria` (8) | AD-ARIA | GROUP | Definition / monitoring / risk / management are distinct decisions |
| `amyloid-imaging` (7) | AD-BIOM-05 | LEAF | amyloid-pet |
| `biomarkers` (4) | AD-BIOM | GROUP | |
| `eligibility` (4) | AD-ELIG | GROUP | |
| `criteria` (2) | AD-CRIT-01 | LEAF | |
| `tau` (2) | AD-BIOM-06 | LEAF | Ambiguous: tau PET vs CSF/plasma tau |
| `plasma` (2) | AD-BIOM | GROUP | Ambiguous across three assays |
| `apoe` (2) | AD-APOE | GROUP | Indication / risk / biology differ **in time-sensitivity** |
| `amyloid` (1) | AD-BIOM | GROUP | Ambiguous: imaging vs fluid vs therapy target |
| `staging` (1) | AD-CRIT-04 | LEAF | |
| `vascular` (1) | AD-DDX-02 | LEAF | |
| `diagnostic-criteria` (1) | AD-CRIT-01 | LEAF | **Duplicate of `criteria`** |
| `plasma-biomarkers` (1) | AD-BIOM | GROUP | **Duplicate of `plasma`** |
| `diagnosis` (7) | — | OUT | Not a named topic; too broad to be a claim class |
| `care` (2) | — | OUT | No named parent |
| `management` (1) | — | OUT | Ambiguous: behavioural management vs general care |
| `prevention` (1) | — | OUT | No named parent |
| `comorbidity` (1) | — | OUT | No named parent |
| **`contested` (1)** | — | **ERROR** | **Not a claim class.** "Contested" is an output *state* that §4.4 **computes over** a class. Storing it as a class means the anchor document is partly tagged with its own expected result |

**Normalisation preserved conceptual distinctions rather than merging strings.** `criteria` and
`diagnostic-criteria` merge because they denote one concept. `apoe` does **not** collapse to one
class, because `apoe-biology` is plausibly time-invariant while `apoe-genotyping-indication`
changed in 2023–2025 — collapsing them would force one ψ(q) value onto two different temporal
behaviours.

---

## D. Coverage requirements

Essential to the architecture, in dependency order:

1. **AD-PHARM, AD-ELIG, AD-ARIA** — carry the 2023–2026 knowledge changes the thesis is built
   on, and host the only identified contested candidate.
2. **AD-CRIT, AD-BIOM** — the 2024 criteria revision and plasma-biomarker sufficiency; the
   supersession backbone.
3. **AD-APOE** — spans time-sensitive and time-invariant, so it is the natural test case for ψ(q).
4. **AD-DDX** — largely time-invariant; candidate NEG-CONTROL (V5) and H7 material.
5. **AD-BEHAV, AD-DEPR** — **currently zero coverage** (M1). Named in §5.1, so REQUIRED, but no
   evidence in the corpus is tagged to them.

Against the six classes the previous analysis flagged: **diagnosis** — deliberately *not* a class
(too broad; decomposed across AD-CRIT/AD-BIOM/AD-DDX); **pharmacotherapy, ARIA, eligibility** —
present as groups, subdivided; **deprescribing, behavioural management** — present as groups,
**empty in the corpus**.

---

## E. Unresolved decisions

### E1 — Granularity level · REQUIRES HUMAN DECISION · *highest impact*

**This is demonstrated, not asserted.** Re-expressing the M4 contested candidates under the
proposed taxonomy:

| Granularity | Cochrane vs Donanemab AUR | Cochrane vs Lecanemab AUR |
|---|---|---|
| **Leaf** (AD-PHARM-03/04 vs AD-PHARM-02) | shared: **none** → contest **cannot fire** | shared: **none** → **cannot fire** |
| **Group** (AD-PHARM) | shared: `AD-PHARM` → contest **can fire** | **can fire** |

The granularity choice alone decides whether **RQ5 is testable at all**. Options: (a) contest at
group level, admit at leaf level; (b) contest at leaf level and accept that RQ5 may have no
material; (c) add an explicit `contest_group` field per class. **I have not chosen.**

### E2 — Is 40–80 binding? · REQUIRES HUMAN DECISION
The taxonomy lands at 50 by derivation. If the supervisor treats 40–80 as binding, that is
satisfied; if the proposal is amended to drop the figure, the taxonomy is unaffected. Either way
the count must not drive the design.

### E3 — Multi-label vs single-label · REQUIRES HUMAN DECISION
`K(s)` is plural (§4.1 L495), implying multi-label. But §4.4 says "*a* claim-class is contested",
implying single. Multi-label multiplies contest opportunities and changes the false-positive
rate; single-label forces an arbitrary primary class. Not resolvable from the source.

### E4 — Disposition of `contested` · REQUIRES HUMAN DECISION
Removing it is defensible (it is a computed state, not evidence) but it currently marks the
anchor document, and removing it changes what M4 finds.

### E5 — ψ(q) time-invariance flag · REQUIRES HUMAN DECISION
Every class carries `time_sensitivity_psi: REQUIRES_HUMAN_DECISION`, unset. Setting it
**directly changes γ** (ψ(q)=0 ⇒ γ=1, disabling decay). Guessing per-class values would silently
determine currency behaviour.

### E6 — Scope of the 5 OUT labels · REQUIRES HUMAN DECISION
`diagnosis`, `care`, `management`, `prevention`, `comorbidity` have no parent among the 11 named
topics. Admitting them widens the taxonomy beyond the proposal; dropping them discards existing
tags.

### Deliberately NOT decided

**Source tier / authority ordering.** Not defined anywhere in this document, though M4 is blocked
on it. Authority ordering is a *tested variable* (ablation A12); defining tiers to make the
contested test executable would prejudge that ablation. This remains **NOT YET DETERMINED**.

---

## F. Validation plan

Taxonomy and assignment mechanism are validated separately, because they can fail separately.

**F1 — Taxonomy validation** (before any matcher):
1. *Completeness* — every claim in the 20 curated documents assignable to ≥1 class; unassignable
   claims are gaps.
2. *Discrimination* — two annotators independently assign a sample; low agreement on a class
   means its definition is inadequate, not that annotators erred.
3. *Contest adequacy* — at the approved granularity, does at least one genuine opposing pair
   share a class? Re-runs M4.
4. *ψ(q) separability* — can each class be labelled time-invariant or time-sensitive without
   forcing the choice?

**F2 — Assignment-mechanism validation** (after approval): the proposal's **300-passage random
sample** with precision and recall per §5.1. Note the earlier finding stands — γ's error is
*per-class*, which argues for stratification, but that is a change to a specified method
(**C10′**, still open).

**F3 — Ordering.** F1 gates the matcher; F2 gates corpus freeze. A matcher built against an
unvalidated taxonomy would produce precise measurements of the wrong thing.
