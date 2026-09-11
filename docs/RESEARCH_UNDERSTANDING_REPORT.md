# Research Understanding Report

**Thesis:** *Does the Filter Prefer the Past? Measuring and Correcting Recency Bias in
Confidence-Derived Evidence Utility Signals for Retrieval-Augmented Alzheimer's Clinical Reasoning*

**Documents analysed**

| # | Document | Role |
|---|---|---|
| D1 | MS Thesis Research Proposal (27 pp., August 2026) | Primary design document. Self-declares: no experimental results; all quantitative statements about the proposed system are hypotheses or design targets. |
| D2 | Sohn, Park, Yoon, Park, Hwang, Sung, Kim & Kang (2025), *Rationale-Guided Retrieval Augmented Generation for Medical Question Answering*, NAACL 2025, pp. 12739–12753 | The foundational system (RAG²). The artefact whose admission stage this thesis measures and replaces. |

**Repository state at time of writing:** empty (`README.md`, 2 bytes). No corpus, code, data, pre-registration or results exist. Every stage of the pipeline is unstarted.

---

## Evidence labelling convention used in this report

| Tag | Meaning |
|---|---|
| `[DOC]` | Directly demonstrated by D1 or D2 (quoted or verified against the text) |
| `[PUB]` | Reported by published research cited in D1/D2, not independently re-verified here |
| `[INF]` | Reasonable methodological inference drawn by this report |
| `[STU]` | Student assumption stated in D1 (including D1's own `[ASM]` tags), not established by evidence |
| `[REC]` | Recommendation by this report — not a finding |

D1's own tags (`[PR] [PP] [AU] [DES] [ASM] [HYP] [OPEN]`) are preserved when quoting it.

---

## 1. Research Problem

**The problem as stated.** `[DOC]` RAG² admits or rejects each retrieved passage using a Flan-T5-large classifier trained on labels derived from a perplexity differential. D1 §1.2 argues this criterion "measures belief shift, not evidential support", and that belief shift is not neutral with respect to passage content: passages consonant with the base model's pre-training-era beliefs raise confidence and are therefore preferentially admitted, while newer, dissonant guidance lowers confidence and is preferentially rejected. The consequence would be a selection stage that is **directionally biased toward older evidence** — invisible to accuracy benchmarks, and clinically consequential in a domain whose guidance has recently changed.

**Verification of the problem's premises against D2.**

| Premise in D1 | Status against D2 | Note |
|---|---|---|
| RAG² filter is Flan-T5-large, 770M | `[DOC]` Confirmed (D2 §4.2) | |
| Label rule uses ΔPPL = PPL(x) − PPL(x,d) ≥ τ, τ = top 25% | `[DOC]` Confirmed (D2 Eq. 3, §3.2) | τ "fixed across all our experiments" |
| Corpus 37.6M docs / 116.7M passages / 564.2 GB | `[DOC]` Confirmed exactly (D2 Table A1) | |
| Reported gains +6.1 / +3.8 / +0.9 avg. accuracy points | `[DOC]` Confirmed exactly (D2 Table 2) | 60.2→66.3; 68.6→72.4; 86.0→86.9 |
| Rationale replaces the query for retrieval because concatenation exceeds retriever max length | `[DOC]` Confirmed (D2 §3.3) | |
| Only open-ended evaluation = 25 queries, ROUGE-L + BERTScore | `[DOC]` Confirmed (D2 §A.4, ClinicalQA25) | |
| RAG² contains no temporal representation of any kind | `[INF]` Confirmed by absence across all 15 pages | Argument from absence; resolvable definitively from the released code |
| Paper motivates itself with hallucination and outdated knowledge but measures neither | `[DOC]` Confirmed (D2 Abstract & §1 vs. Tables 2–5, all accuracy) | This is a fair and load-bearing characterisation |

**A material discrepancy — D1 mischaracterises its own target in §1.2.**

`[DOC]` D2 §3.2 specifies a **decision tree**, not a perplexity threshold. The primary criterion is a **correctness flip**: "If the model successfully answers a question with the aid of retrieved documents but fails to do so independently, the documents are labeled as 'helpful'." Perplexity enters only as a **tie-breaker** for the residual cases: "it fails to fully capture the document utility, particularly in cases where the model's accuracy remains unchanged regardless of whether the document is included or not. **To address this**, we calculate the difference in perplexity."

D1 §4.2 states this correctly ("a passage is helpful if it flips the model from incorrect to correct, ambiguous cases resolved by the perplexity differential"). D1 §1.2 does not ("A passage is labelled helpful when conditioning on it reduces the base model's perplexity").

`[INF]` This is not a wording slip. The thesis's causal story is *confidence gain → recency bias*. If the majority of training labels come from the correctness-flip branch, the dominant bias channel is **agreement with a gold answer key**, not confidence — a different mechanism with different predictions, and one that is entangled with the exam-era provenance of MedQA/MedMCQA gold keys. The proportion of labels arriving via each branch is currently unknown and is a measurable quantity.

**A second, sharper point in the thesis's favour.** `[DOC]` D2's Introduction and §5.1 both state the perplexity is computed over **the model's own generated rationale** ("labels derived from the differences in perplexity between rationales with and without retrieved documents"; "Our filtering method uses the perplexity of the LLM's rationale to annotate question-document pairs"). D2's Eq. 4 notation (`PPL(x)`, `PPL(x,d)`) is ambiguous and appears to score the query. `[INF]` If the rationale reading is correct, the mechanism D1 hypothesises is **substantially stronger than D1 argues**: the signal measures how well a passage agrees with a reasoning chain the model has already produced from parametric memory. That is confirmation bias operationalised almost directly. D1 does not currently exploit this. It also imposes an implementation requirement D1 §4.2 omits: the probe must replicate rationale generation, because ΔPPL is undefined without a rationale.

**Who is affected.** `[STU]` Clinicians and patients, via a system that reproduces pre-2024 Alzheimer's consensus: recommending CSF or PET confirmation where a validated plasma assay now suffices, omitting APOE-stratified ARIA counselling, or missing anticoagulation as a contraindication (D1 §1.3). These are plausible failure modes, not demonstrated ones.

**What remains unresolved.** Whether the asymmetry exists at all. D1 §1.4 states plainly: "This has never been measured."

---

## 2. Research Motivation

`[DOC]` Three motivations, in D1's own ascending order of importance:

1. **The evidence base moved recently and substantially.** `[PUB]` Jack et al. (2024) revised criteria redefined AD biologically, introduced Core 1/Core 2 biomarker categories, and elevated accurate plasma biomarkers to diagnostic sufficiency. Lecanemab (Cummings et al., 2023) and donanemab (Rabinovici et al., 2025) appropriate-use recommendations specify eligibility, mandatory APOE genotyping and ARIA monitoring schedules that did not previously exist.
2. **Staleness has direct clinical consequence.** `[STU]` As above.
3. **Knowledge does not move monotonically forward.** `[PUB]` Cochrane CD016297 (April 2026; 17 trials, 20,342 participants) concluded amyloid-beta-targeting monoclonal antibodies probably produce little to no clinically meaningful cognitive difference at 18 months against increased ARIA risk; the Alzheimer's Association, UK DRI and a *Lancet* commentary contested it within days.

`[INF]` The third motivation is the strongest and the most defensible, and D1 correctly identifies it as such. It converts an architectural assumption ("newer supersedes older") into a *testable design choice*, and it supplies a real, dated, externally verifiable controversy rather than a synthetic one. It is also the motivation least likely to be pre-empted by another group.

**Why existing approaches are insufficient.** `[PUB]` The largest expert evaluation of medical RAG to date (arXiv:2511.06738 — 18 clinicians, 80,502 annotations, 800 outputs) reports that standard retrieval augmentation *degrades* medical performance under several conditions: only 22% of top-16 retrieved passages judged relevant, 31% of queries returning no relevant passage in the top 16, and must-have statement coverage of 33%. `[INF]` D1 uses this correctly — as evidence that retrieval quality, evidence selection and answer faithfulness are distinct problems — and then, in §10, uses it *against itself* as the acknowledged design tension (see §14 below). That intellectual honesty is a genuine strength.

**Motivation that is overstated.** `[INF]` D1's abstract claims the signal family "now underlies reward signals in learned retrieval systems, so any bias would extend well beyond the specific filter examined here." The supporting citations (arXiv:2510.11358, arXiv:2605.13277) are unverified preprints. The generalisation claim is plausible but currently rests on `[PP]` material whose authorship D1 itself declines to vouch for.

---

## 3. Research Gap

D1 §1.4 tabulates six established claims and two open ones. Assessed independently:

| Claimed gap | This report's assessment |
|---|---|
| **Filter-stage admission asymmetry w.r.t. evidence age** | `[INF]` **Genuine, and correctly scoped.** ClashEval and Xie et al. establish confirmation bias and confidence-mediated adoption *at the generator*; Vladika et al. establish outdated-knowledge memorisation *parametrically*; Javadi et al. stratify by publication year *at evaluation*. None instruments the admission decision. The gap survives scrutiny. |
| **Explicit representation of contested evidence in medical RAG** | `[INF]` **Genuine but narrower than stated.** D1 supports it with Javadi et al. (ECIR 2026) calling for contradiction-aware filtering without implementing it. `[REC]` The claim should be "no *medical* RAG system implements a contested output state", not "no system represents disagreement" — contradiction detection and stance/conflict modelling are established in general NLP and would be a reasonable examiner challenge. |

**Gaps D1 identifies but does not list as gaps.** `[INF]`

- **Evaluation gap.** D2's own hallucination and staleness claims are unmeasured (D2 reports accuracy only, plus 25 open-ended queries on lexical/embedding overlap metrics). Establishing this is part of D1's contribution but is framed as background rather than as a gap.
- **Reproducibility gap.** `[DOC]` D1 §1.5 Objective 1 notes "the trained checkpoint is not distributed." `[INF]` D1 nowhere records that **D2's code is publicly released** (D2 §1: `https://github.com/dmis-lab/RAG2`). This materially changes the reproduction risk profile — see Finding F6.

**Novelty threat D1 raises against itself.** `[DOC]` D1 §2.5 identifies CoRM-RAG (arXiv:2605.01302) as "substantially closer to this work than a surface reading suggests" — sharing premise, diagnosis and architecture — and states the differentiator is the perturbation axis (query perturbation vs. evidence-age perturbation). `[INF]` The differentiation is real and well argued. But D1's own reference list marks CoRM-RAG's "[Authorship and venue not independently verified]". **The principal novelty threat is currently characterised from an unverified preprint.** This must be resolved before the novelty position can be defended.

---

## 4. Research Objectives

`[DOC]` **Aim.** To determine whether confidence-derived evidence utility signals exhibit a measurable recency bias in medical RAG, and to specify and evaluate a minimal corrective admission policy for Alzheimer's clinical reasoning.

Seven objectives (D1 §1.5): reproduce RAG² including retraining the filter; build a date-annotated AD corpus; build FRB-PAIRS plus permutation control; execute the probe across two label functions and two backbones; specify and implement SCAF; evaluate SCAF against RAG² and the current SOTA filtering pipeline; conduct blinded clinical evaluation and structured error analysis.

**Alignment check — objectives ↔ questions ↔ methodology ↔ evaluation.** `[INF]`

| Link | Verdict |
|---|---|
| Aim → RQ1–RQ3 (measurement) | **Aligned.** The probe design (matched pairs, identical backbone/capacity/training data, only the label function differs) is the correct instrument for a directional claim about the signal. |
| Aim → RQ5 (contested state) | **Partially aligned.** RQ5 asks whether an explicit contested state improves clinical appropriateness; it is assessed only by blinded expert rubric on ~40 CONTESTED-AD items with 2–3 raters. This can demonstrate mechanism; it cannot establish the claim RQ5 is phrased to ask. D1 §4.4 concedes exactly this. |
| Objective 2 (AD corpus) → RQ1–RQ3 | **Misaligned, and this is useful.** If FRB-PAIRS is drawn from MedChangeQA (general medical), the primary claim does not require the AD corpus. The pipeline's stated order (Corpus → Pairs → Probe) may not be the true dependency order. See Finding F4 and §20. |
| Evaluation → non-inferiority (RQ6/H7) | **Aligned but with an unset parameter.** δ is `[OPEN]`; "two accuracy points is offered for discussion, not adopted." A non-inferiority test without a pre-registered margin is not a test. |

---

## 5. Research Questions and Hypotheses

`[DOC]`

| ID | Question | Hypothesis |
|---|---|---|
| RQ1 | Do confidence-derived utility signals admit passages asymmetrically w.r.t. age, holding content, tier and length constant? | H1: Δ(perplexity) > 0. H2: Δ(perplexity) > Δ(support). |
| RQ2 | Does any asymmetry replicate across filter backbones? | H3: asymmetry replicates with the same sign. |
| RQ3 | Does an entailment-derived label reduce unsupported and superseded claims? | H5: unsupported-claim rate falls. |
| RQ4 | Is any reduction attributable to admission policy rather than mediated by retrieval quality? | H6: a direct arm effect persists after conditioning on recall. |
| RQ5 | Does an explicit contested state improve clinical appropriateness on unresolved questions? | Blinded expert rubric. |
| RQ6 | Is accuracy preserved on time-invariant medical QA? | H7: non-inferiority within pre-registered margin δ. |
| — | Validity gate | H4: on the permutation control, Δ = 0. |

Δ is defined (D1 §4.2) as **E over pairs of P(admit | older) − P(admit | newer)**; Δ > 0 indicates preferential admission of older evidence.

**Critical assessments.** `[INF]`

- **H1 is the load-bearing hypothesis and the only one that is both novel and cleanly testable.** Everything else either depends on it (H2, H3), is a downstream engineering claim (H5–H7), or is demonstrative (RQ5).
- **H4 as specified cannot fail for the primary comparison.** See Finding F1 — this is the most serious specification defect in the document.
- **H2 is not capacity-matched in the dimension that matters.** D1 §6.3 guarantees matched backbone, size, training-data volume and hyperparameters. It does not match **label informativeness**. The proposed label is supervised by entailment against the verbalised gold answer; the baseline label is a correctness flip with a perplexity tie-break. An examiner will observe that a richer supervision signal producing a better filter is unsurprising. This does not damage H1 (which concerns the baseline alone), but it weakens H2 as stated.
- **H1's mechanism has an untested precondition.** The mechanism requires that "newer" evidence lie *outside* the label-generating model's parametric priors. See Finding F2.

---

## 6. Existing Approaches

**The foundational system (RAG², D2).** `[DOC]` Three components: (1) rationale-based query formulation — a chain-of-thought rationale replaces the question as the retrieval query, the original query excluded for length; (2) balanced retrieval — equal quotas from PubMed, PMC full text, clinical practice guidelines and 18 textbooks, MedCPT dense retrieval, MedCPT cross-encoder reranking **using the original query**; (3) rationale-guided filtering — Flan-T5-large trained on the correctness-flip/perplexity decision tree. Trained 40 epochs, lr 3e-5, batch 16, single H100; inference via vLLM, greedy decoding, T = 0.

`[DOC]` D2's own stated limitations: tested only in biomedicine; only one filter size tried; an incorrect rationale can steer the retriever toward distractors; snippets are labelled **individually**, ignoring the combined effect of multiple passages, and Flan-T5's context limit means it filters **one snippet at a time**.

`[INF]` That last limitation is directly relevant to SCAF and is not discussed in D1: SCAF's contested state requires reasoning over **pairs** of passages ("two passages of that class carry opposing conclusions"). A one-snippet-at-a-time filter architecture cannot express this. SCAF's Algorithm 1 handles it by moving the contested test outside the filter into a post-scoring gate — which is correct, but it means SCAF is not a drop-in label swap; it is a structural change. The fairness guarantee "only the label function differs" therefore holds for the **probe** (§4.2, Filter A vs Filter B) but *not* for the full SCAF arm in Stage 5. D1 does not currently distinguish these two comparisons clearly.

**An unresolved question about D2 that affects reproduction.** `[INF]` Perplexity labels are base-model-specific, implying a filter per backbone. Yet D2 applies RAG² to GPT-4o (Table 2, +0.9), and ΔPPL over a generated rationale is not obtainable from the GPT-4o API. D2 never states how the GPT-4o filter was obtained. Either a transferred filter was used — which would contradict D1's "must be retrained per backbone" and halve the reproduction cost — or an undocumented procedure was used. D2 *does* demonstrate cross-**dataset** transfer explicitly (MedMCQA-trained filter → MMLU-Med, +5.3 / +4.9), but that is not cross-**backbone** transfer. Resolvable from the released code.

**Prior art positioned by D1.** `[PUB]` FILCO (context filtering by lexical and information-theoretic measures, ancestor of the perplexity differential); Sufficient Context, Joren et al. ICLR 2025 (entailment-based sufficiency); information-gain utility (arXiv:2510.11358); CoRM-RAG (lightweight distilled critic, risk-aware abstention, "relevance–robustness gap"); ClashEval (confidence-mediated adoption); Xie et al. ICLR 2024 (confirmation bias toward partially consonant evidence); Vladika et al. (MedRevQA 16,501 pairs from systematic-review abstracts, PubMed 2000–Jan 2024; MedChangeQA 512 changed-verdict pairs); Javadi et al. ECIR 2026 (publication-year stratification, calls for contradiction-aware filtering); CARE (EMNLP 2025).

**D1's directional caution on CARE is correct and worth preserving.** `[DOC]` CARE's stated purpose is resolving "the conflict between incorrect external context and correct parametric knowledge" — steering toward the more reliable source, frequently the parameters. A temporal system requires the inverse. `[INF]` Importing CARE's assessor unmodified would reinforce precisely the staleness the system exists to correct. D1 §2.4 states this explicitly rather than leaving it implicit; this is good practice and should survive into the thesis.

---

## 7. Proposed SCAF Contribution

`[DOC]` SCAF replaces the perplexity filter with an admission score

```
A(s) = w1·σ(s) + w2·γ(s,q) + w3·ρ(s) + w4·τ(s)      admit if A(s) ≥ θ_admit
```

- **σ(s) — entailment-derived support.** Training label: `σ* = P_entail(premise = s, hypothesis = verbalise(Q, gold answer))`, thresholded at θ_hi / θ_lo with a discard band mirroring D2. At inference the gold is unavailable, so σ(s) = max entailment over a hypothesis set (verbalised options for MCQ; ≤8 atomic claims decomposed from the rationale for open-ended). A discriminativeness margin diagnostic is recorded; whether it enters A(s) is `[OPEN]`.
- **γ(s,q,t_q) — three-state currency.** 0 if retracted/withdrawn; 1 if ψ(q)=0 (time-invariant); `δ·2^(−(t_q−date(s))/H)` if superseded; `2^(−(t_q−date(s))/H)` otherwise.
- **τ(s) — source authority**, included but treated as a *tested variable* (ablation A12), not an assumed ordering.
- **ρ(s)** — rank-normalised reranker score (rank-based, not min-max, so a global θ_admit is well defined).
- **Output policy** — four states: GROUNDED, FLAGGED, CONTESTED, ABSTAIN.
- **Verification** — a post-hoc claim verifier from a *different model family* (V4), to avoid correlated error.

**Complexity.** `[DOC]` ≤32 candidates × 8 hypotheses = 256 short forward passes on a sub-billion-parameter model per query. Negligible relative to generation. `[INF]` This figure is credible.

**Design decisions that are well justified.** `[INF]`

- **Soft supersession.** Hard rejection confined to retraction/withdrawal because claim-class tags come from an automatic matcher with only a sampled audit; a hard gate would silently delete correct evidence at a rate proportional to tagging error. Retraction and withdrawal are objective and externally verifiable. This is the right call and D1 argues it properly.
- **ψ(q)-conditioned decay.** An unconditional recency prior would penalise correct older sources on time-invariant questions (APOE biology, neuropathological staging, cholinesterase pharmacology). D1 is right that this conditioning is what makes non-inferiority (H7) plausible at all.
- **Contested tested before superseded.** Load-bearing, not incidental: the reverse order would classify the April 2026 Cochrane review as merely superseding the AURs, presenting a live controversy as resolved.

**What is genuinely novel vs. adapted.** `[DOC]` D1 §2.5 disclaims novelty for entailment-based selection, information-theoretic utility filtering, distilled critics with abstention, selective prediction, temporal medical benchmarks and AD-specific RAG. It claims novelty for exactly two things: **filter-stage temporal asymmetry as a measured property**, and **the contested state as an implemented output condition in a medical retrieval pipeline**.

`[INF]` This is an unusually disciplined novelty position and it is approximately right. Two qualifications: (a) SCAF as a *system* is an assembly of established parts — its contribution is the measurement it enables and the currency/contested semantics, not the architecture; D1 already says this, and the thesis must keep saying it; (b) the novelty of the contested state depends on the CoRM-RAG comparison, which rests on an unverified preprint.

**What must be demonstrated experimentally.**

| Claim | Required evidence | Currently |
|---|---|---|
| The asymmetry exists | H1 with CI on FRB-PAIRS, validity gate passed | Not established |
| It is a property of the signal, not one model | H3 across two backbones | Not established |
| Entailment supervision reduces it | H2, with label-informativeness confounder addressed | Not established |
| SCAF reduces unsupported/superseded claims | H5, mixed-effects logistic, dual-reported | Not established |
| The effect is admission, not retrieval | H6 mediation + unchanged-recall subset | Not established |
| SCAF costs nothing on time-invariant QA | H7 non-inferiority vs. pre-registered δ | δ not set |
| The contested state helps | Blinded rubric; mechanism only | Not established |
| SCAF introduces no new harm | Over-rejection, evidence-insufficiency, over-abstention rates | Design exists; not run |

---

## 8. Corpus Requirements (Stage 1)

`[DOC]` D1 §5.1 specifies five collections: PubMed-AD (MeSH-scoped, 2–4×10⁵ abstracts `[OPEN]`), PMC-AD (open-licence full texts, 2–4×10⁴), CPG-AD (~10³ dementia-relevant guidance documents), Textbooks (18 volumes), and a manually ingested **Currency pack** (20–40 documents: 2024 revised criteria, lecanemab/donanemab AURs, ARIA guidance, Cochrane CD016297).

**Metadata schema.** `[DOC]` Chunk and document IDs; source tier; publication date; journal and persistent identifier; guideline family, version, supersession pointer; retraction/withdrawal flags; claim-classes; section header; character span.

**Preprocessing.** `[DOC]` 256-token sliding window with 32-token overlap, sized against MedCPT's 512-token article-encoder limit with headroom for prepended title and section header. Exact dedup by content hash; near-duplicate detection by MinHash over 5-gram shingles at Jaccard 0.85, retaining the newest, highest-tier cluster member. Guideline passages respect section boundaries — "a recommendation must never be separated from its qualifying conditions."

`[INF]` The near-duplicate rule is specifically correct for this thesis: preprint-to-journal duplicates would otherwise let an old claim survive under a new date, directly contaminating the dependent variable.

**Inclusion/exclusion.** `[DOC]` English-language public corpora only; open-licence only for PMC; no private clinical data; credentialed resources optional and never load-bearing. Domain scoping is declared a *controlled constant* (identical scoped corpus for every arm) rather than a confound.

**Acknowledged data-quality risks.** `[DOC]` `[ASM]` Guideline PDFs expected to yield poorer date coverage than structured XML; uncorrected this would systematically penalise the guideline corpus — the opposite of intent. Date imputation from issue metadata applied; per-corpus null rate is a reported figure. `[ASM]` Claim-class tagging (40–80 hand-built classes, keyword-and-embedding matching, 300-passage manual audit reporting precision and recall) is "the weakest link in the pipeline."

**Requirements this report judges missing.** `[REC]`

1. **No retrieval quota is defined for the currency pack.** Balanced retrieval draws equal quotas from *four* corpora (D2 §3.4). D1's table lists *five* collections. If the currency pack is folded into CPG-AD it is 20–40 documents competing inside ~10³; if it is given its own quota, that is a change to the frozen upstream stage, breaking the internal-validity guarantee that the whole design rests on. This decision is unmade and unmarked.
2. **No retrievability acceptance criterion.** The corpus is currently specified by construction rules only. Nothing in D1 checks whether the frozen retriever ever *surfaces* the currency pack. If it does not, SCAF's ceiling is near zero regardless of admission policy — and D1's own cited evidence (22% top-16 relevance; 31% of queries with no relevant passage) makes this a live possibility rather than a pedantic one.
3. **No representativeness statement.** D1 does not state what population the corpus is meant to represent, nor how the MeSH scoping rule relates to the AD questions actually asked in AD-VIGNETTE and ADCUR-QA.

**Reproducibility requirements.** `[REC]` The corpus is a dated snapshot of live databases. A PubMed/PMC pull in October 2026 and one in March 2027 are different corpora. Snapshot date, E-utilities query strings, MeSH explosion settings, retrieval timestamps and per-collection record counts must be frozen and recorded, or no arm comparison is reproducible and the counterfactual corpus-update procedure (§6.4) is not well defined.

---

## 9. Test-Pair Design (Stage 2)

`[DOC]` A temporal-counterfactual pair is two passages addressing the same claim on opposite sides of a known change point, matched on **source tier**, on **token length** within a tolerance band `[OPEN]`, and on **topical similarity to the query** `[OPEN]`.

`[DOC]` D1's rationale for matching, tagged `[ASM]`: "older and newer statements of the same claim plausibly differ in hedging convention, citation density and prose style, any of which a filter may respond to instead of age. Matching reduces but cannot eliminate this."

**Provenance architecture.** `[DOC]` D1 §5.2 declares provenance separation "the single most important correction in this revision": an earlier formulation authored the evaluation items, supersession table and currency corpus from the same ~20 documents, "rendering a positive result unfalsifiable by construction." The firewall now puts the primary claim on externally-sourced MedChangeQA and confines thesis-curated material to replication and case study.

`[INF]` **This is the strongest single design decision in the proposal.** It should be preserved under pressure — and it *will* come under pressure, because the pair-count arithmetic (below) creates a direct incentive to breach it.

**Evaluation sets.** `[DOC]`

| Set | Size | Provenance | Purpose |
|---|---|---|---|
| FRB-PAIRS | 150–300 pairs | From MedChangeQA `[PR]` | Primary: admission asymmetry |
| FRB-PERMUTED | same pairs | Dates randomly reassigned | Validity gate (H4) |
| NEG-CONTROL | ~80 items | Claims stable across window | Detects spurious recency preference |
| CONTESTED-AD | ~40 items | Anti-amyloid efficacy pre/post CD016297 | Tests contested state |
| ADCUR-QA | ~120 items | Authored from documents excluded from supersession table | Domain replication |
| AD-VIGNETTE | 60–80 items | Open-ended clinical vignettes | Blinded expert evaluation |
| MedQA / MedMCQA / MMLU-Med | as published | Existing `[PR]` | Non-inferiority only |
| MedXpertQA | as published | Existing | Unsaturated accuracy comparison |

**Gaps this report judges critical.** `[REC]`

1. **The passage source for FRB-PAIRS is never specified.** MedChangeQA supplies 512 question–answer pairs whose verdict changed. It does not obviously supply *matched passage pairs*. Either the passages come from the source systematic-review abstracts underlying MedRevQA, or they are retrieved from a corpus, or they are authored. Each choice has different bias properties, and the third breaches the provenance firewall. **The primary instrument for the primary claim has no construction procedure.**
2. **Attrition from 512 to 150–300 is unmodelled.** After requiring a locatable old passage, a locatable new passage, tier match, length match within an unset band, and topical-similarity match by an unset criterion, the surviving count could be far below 150. D1 §8 already names the failure mode — "the most likely adverse outcome is not a clean negative but an inconclusive one, which is materially worse for a thesis" — but the count is asserted as a target, not derived from the data.
3. **No stratification by pre-training cutoff.** See Finding F2 — the single largest threat to the thesis.
4. **Matching thresholds are `[OPEN]` yet the power analysis is scheduled before pair construction** (Phase 0, weeks 1–2; Phase 3, weeks 10–16). A power analysis over an unspecified sampling frame is not informative. `[REC]` Sequence: pilot pair extraction (n≈40) → attrition and effect-size estimate → power analysis → pre-registration → full construction.

---

## 10. Bias-Probe Methodology (Stage 3)

`[DOC]` **Operational definition.** For a filter f, admission asymmetry Δ = E over pairs of `P(admit | older) − P(admit | newer)`. Δ > 0 = preferential admission of older evidence.

`[DOC]` **What is held constant.** Filter A (perplexity label, baseline) and Filter B (entailment label, proposed) share backbone, parameter count, training data and hyperparameters; only the label function differs. D1 §4.2: "This identity is what licenses attributing the measured asymmetry to the signal rather than to capacity, architecture or retrieval."

`[DOC]` **Statistics.** H1–H3 by paired bootstrap, 10,000 resamples, mean asymmetry with 95% interval, unit = matched pair. H4 by equivalence test against a pre-specified band. Holm–Bonferroni across the primary family at family-wise 0.05; secondary hypotheses uncorrected and labelled.

`[DOC]` **Validity controls V1–V5.** V1 permutation (blocking); V2 backbone replication; V3 cached candidate replay (byte-identical); V4 decoupled verifier (different model family); V5 negative control set (claims stable across the window).

**Does the probe measure the intended phenomenon? — the central question, and the answer is currently "not demonstrably".** `[INF]`

The probe's construct validity rests almost entirely on V1, which D1 designates blocking. V1 does not do the work D1 assigns it (Finding F1). V5 does — and is listed fifth, is not tied to a hypothesis, and is not among the load-bearing ablations scheduled first (A1–A3 are label comparison, permutation control, backbone replication).

**Confounders the design does not yet separate.** `[INF]`

| Confounder | Currently addressed by | Adequate? |
|---|---|---|
| Prose-era surface features (hedging, citation density, formatting) | Tier + length matching; V1 | No — V1 is vacuous; matching is acknowledged as incomplete |
| Claim-change vs. era-style | V5 NEG-CONTROL | Yes in principle — but not designated blocking |
| Label-generating LLM's priors vs. filter student's priors | H3 backbone replication | Partially — A10 (teacher vs. distilled student) is the correct instrument but is scheduled late |
| Label-branch mixture (correctness flip vs. perplexity) | Nothing | No — not currently measured at all |
| Change point inside vs. outside pre-training window | Nothing | No — Finding F2 |

**Interpretation criteria.** `[DOC]` D1 pre-declares the pivot: "A1 shows no label effect → headline shifts to the currency and contested contributions; H1 and H2 reported as null results, which remain publishable given the strength of the prior expectation." `[INF]` Designing for an informative null is correct practice and is one of this proposal's real strengths. It is only meaningful, however, if the null is *interpretable* — which requires the construct-validity controls to actually discriminate. A null obtained with a vacuous validity gate is not an informative null; it is an uninterpretable one.

---

## 11. Evaluation Strategy (Stage 5)

`[DOC]` **Arms.** B0 closed-book; B1 retrieve+rerank, no filter; B2 RAG² reproduction; B3 current SOTA (support-supervised filtering + query reformulation, arXiv:2511.06738); B4 agentic reference point (context only, not a competitor); P = SCAF.

`[DOC]` B3 is declared required rather than optional: "Comparing only against the base paper would benchmark the thesis against a system the field has already improved upon — and improved upon by two of that paper's own authors." `[INF]` Correct and necessary.

`[DOC]` **Fairness guarantees (§6.3).** (1) Upstream identity — reranked candidate set computed once, serialised, replayed byte-identically; arms are distinct functions over one cached list. (2) Matched context budget — all arms ≤5 passages, plus a matched-length variant padding SCAF with the next-best rejected passage. (3) Matched filter capacity. (4) Prompt parity — main comparison uses one template *without* date annotations, so gains attribute to admission rather than date cues; date-annotated prompting evaluated separately as a co-primary condition. (5) No test-set tuning — thresholds and weights selected on validation and frozen before any test run. (6) Contamination control — filters trained only on general medical splits; evaluation sets finalised after training data frozen.

`[INF]` Guarantees (1) and (4) are the strongest parts of the experimental design. (4) in particular pre-empts the obvious objection that SCAF wins only because dates appear in the prompt.

`[DOC]` **Counterfactual corpus-update procedure (§6.4).** Two indices — pre-June-2024 only, and full. Identical models and prompts. Report update-flip rate: proportion of items whose answer correctly transitions from superseded to current gold. Because generator weights are frozen and only the corpus changes, any difference is a property of the pipeline rather than the model. `[INF]` This is a clean, well-constructed instrument and is one of the more publishable secondary results in the design.

`[DOC]` **Two guards declared in advance.** *Judge validation guard*: automatic claim labels validated against human labels on a stratified sample with Cohen's κ; if κ < 0.6, automatic rates demote to secondary and expert ratings become primary. *Mandatory dual reporting*: every rate metric reported twice — conditional on answering, and with abstentions counted as failures, because "a system that abstains more computes its rates over a smaller, easier denominator."

`[INF]` Both are excellent and unusual for an MS proposal. Dual reporting in particular closes the most common way an abstaining system manufactures an apparent win.

`[DOC]` **Decoupling analysis (§7.2)** — per-item rank correlation between Δrecall and Δunsupported-claim rate; mixed-effects mediation reporting the arm coefficient with and without recall as mediator, with bootstrap proportion-mediated; and the subset of items where the admitted set differs but recall is unchanged, where "any difference there cannot be a retrieval effect."

`[INF]` The third instrument is the cleanest of the three and should be reported prominently — it isolates the admission effect by construction rather than by modelling assumption.

`[DOC]` **Both directions are evaluated.** Beyond the target metric, D1 measures over-rejection, evidence insufficiency, over-abstention, risk–coverage area, and coverage at fixed accuracy. `[INF]` This satisfies the requirement that an intervention be assessed for unintended consequences and not only for improvement on its target.

**Weaknesses.** `[REC]` δ is unset. The ablation set (A1–A14) is large for a single-GPU, 44-week budget and will require triage — D1 designates A1–A3 load-bearing, which is right, but A5/A7 (currency-only, support-only) are arguably also load-bearing for C2 and should be named as such. A10 should move into Stage 3.

---

## 12. Role of Doctor / Expert Review

`[DOC]` Three evaluators targeted: one specialist with dementia experience plus two of {senior trainee, clinical pharmacist, final-year medical student}. Outputs from four arms presented with arm identity removed and order randomised per item under a sealed key generated before packet assembly. Five ordinal dimensions with written anchors — factual correctness, evidence support, currency, clinical appropriateness and safety, completeness — plus two binary items. Krippendorff's α reported **before** arm differences; dimensions below 0.4 reported descriptively only. Analysis by mixed-effects cumulative-link regression with Cliff's delta.

`[DOC]` `[ASM]` Recruitment is named the largest schedule risk. Pre-declared fallback: "Clinician recruitment fails → structured checklist evaluation against clinician-authored answer keys, reported explicitly as a weaker instrument."

`[DOC]` Scope discipline: "Clinician evaluators participate as expert raters rather than as research subjects"; "the expert evaluation will not be described as clinical validation"; `[OPEN]` institutional review requirements must be confirmed before recruitment.

**Assessment.** `[INF]` The blinding, sealed-key and report-agreement-first protocol are correct. Two residual risks: (a) with 2–3 raters and ~40 CONTESTED-AD items, α will be unstable and the RQ5 result will likely be descriptive — this should be stated as an expectation now, not discovered later; (b) heterogeneous rater expertise (specialist vs. student) makes disagreement partly a competence effect rather than an item effect. `[REC]` Pre-specify whether ratings are pooled or the specialist is treated as a separate stratum, before packet assembly.

---

## 13. Expected Contribution

`[DOC]` **C1 (primary).** The first controlled measurement of admission asymmetry w.r.t. evidence age in a medical retrieval pipeline, with architecture, capacity, training data and retrieval held constant, replicated across two backbones and validated by a permutation control.

`[DOC]` **C2.** SCAF — entailment-derived support replacing confidence-derived utility, three-state currency with an explicit contested condition, source authority as a tested variable.

`[DOC]` **C3.** Released artefacts — FRB-PAIRS with permutation control, domain replication set, supersession table, timestamped pre-registered analysis plan.

**Assessment.** `[INF]` C1 is the real contribution and is correctly identified as primary. Its defensibility depends entirely on the construct validity of the probe — and C1's own wording, "validated by a permutation control", currently names the control that does not validate it. C3 is undervalued in D1: a released, externally-provenanced temporal-counterfactual pair set with a working validity control would be reusable by other groups and is the artefact most likely to be cited independently of whether H1 holds.

---

## 14. Known Limitations

`[DOC]` D1 §10 states, without prompting:

- The supersession table is small, manual, domain-scoped and non-exhaustive.
- The contested detector cannot reliably distinguish genuine contradiction from scope difference ("effective in early-stage disease" vs. "not effective in moderate disease" is a scope qualifier, not a conflict); it demonstrates mechanism, not validated capability.
- Absolute performance figures are not comparable to published baseline results because the corpus is domain-scoped; only within-study arm contrasts are interpretable.
- Findings are for one disease area; generalisation is a hypothesis, not a result.
- **The acknowledged design tension:** the thesis studies the admission stage while the field's own expert data indicate retrieval is the dominant bottleneck. "Freezing retrieval buys causal attribution at the cost of a lower achievable ceiling and a reduced effective sample. This trade-off is deliberate and is stated explicitly rather than left for an examiner to raise."
- Expected to disappoint, pre-declared: saturated benchmarks will show little movement; automatic hallucination metrics will be fragile (zero-shot filtering F1 reported at 0.44–0.52, rising to 0.62 fine-tuned; "a judge operating in that range cannot support fine-grained claims").

`[INF]` This is a strong limitations section — most of what an examiner would raise is already conceded and argued. Limitations this report adds: the currency-pack retrievability ceiling (§8); the label-branch mixture (§1); the pre-training-window precondition (§5); and the fact that the "only the label function differs" guarantee holds for the probe but not for the full SCAF arm (§6).

---

## 15. Potential Threats to Validity

**Construct validity**
- `[REC]` **V1/H4 does not test what it claims** — the filters never see dates, so permuting dates cannot change filter behaviour; Δ collapses to zero by construction. *(Finding F1 — most serious specification defect.)*
- `[REC]` **Δ conflates claim recency with prose-era style.** V5 NEG-CONTROL is the correct discriminator and is not designated blocking.
- `[REC]` **"Confidence" is not cleanly operationalised** while the baseline label is a mixture of correctness-flip and perplexity branches in unknown proportion.

**Internal validity**
- `[REC]` **Mechanism precondition untested** — if change points precede the label-generating model's pre-training cutoff, H1 cannot fire and a null is uninterpretable. *(Finding F2 — largest single risk.)*
- `[REC]` **Prior conflation** — the probe mixes the teacher LLM's priors with the Flan-T5 student's priors. A10 is the instrument; it is scheduled late.
- `[REC]` **Label-informativeness asymmetry in H2** — a richer supervision signal is confounded with the label *type*.
- `[DOC]` V3 cached candidate replay is a genuine and well-specified protection against arm contamination.

**External validity**
- `[DOC]` One disease area; domain-scoped corpus; English only; ≤1B filters, ≤8B generators.
- `[REC]` Corpus is a live-database snapshot — not reproducible without frozen query strings and snapshot dates.

**Statistical conclusion validity**
- `[DOC]` D1 correctly specifies mixed-effects models for clustered claims ("a naive per-claim chi-squared test would be anti-conservative").
- `[REC]` Power analysis is scheduled before the sampling frame exists; pair-count attrition from 512 is unmodelled; δ is unset. D1's own most-likely-adverse-outcome — an inconclusive result — follows directly from these three.

**Circularity and leakage**
- `[DOC]` The provenance firewall (§5.2) is a deliberate and effective response to a circularity the author already caught once. `[REC]` Its main future threat is pair-count pressure creating an incentive to top up FRB-PAIRS with thesis-curated material. Pre-register that the lanes do not cross.
- `[DOC]` Contamination control (filters trained only on general medical splits; evaluation sets finalised after training data frozen) is correctly specified.

**Expert-review bias**
- `[DOC]` Blinding, sealed key, randomised order, agreement-before-differences. `[REC]` Rater heterogeneity pooling rule unspecified.

---

## 16. Unresolved Questions

Carried from D1's own `[OPEN]` tags, plus questions this report adds (`[REC]`):

1. `[OPEN]` Token-length tolerance band for pair matching.
2. `[OPEN]` Topical-similarity criterion for pair matching.
3. `[OPEN]` Whether the discriminativeness margin diagnostic enters A(s).
4. `[OPEN]` Whether ψ(q) is a rule over the claim-class taxonomy or a trained classifier.
5. `[OPEN]` Contest-window length for the contested state.
6. `[OPEN]` Non-inferiority margin δ.
7. `[OPEN]` Weight-selection parameterisation for w1..w4 (full simplex grid infeasible).
8. `[OPEN]` Replication filter backbone ("modern sub-billion encoder").
9. `[OPEN]` Entailment teacher NLI corpus.
10. `[OPEN]` Corpus scale estimates.
11. `[OPEN]` Judge-validation sample size (must be sized by a precision target).
12. `[OPEN]` Power analysis determining evaluation-set sizes.
13. `[OPEN]` Institutional review requirements for the expert rating study.
14. `[REC]` **Where do FRB-PAIRS passages come from?** *(blocking)*
15. `[REC]` **Does the currency pack get its own retrieval quota?** *(blocking for Stage 1)*
16. `[REC]` **What fraction of RAG² training labels come from the correctness-flip branch vs. the perplexity branch?**
17. `[REC]` **Is ΔPPL computed over the rationale or the query?** (D2's prose and Eq. 4 disagree.)
18. `[REC]` **Did D2 train a separate filter per backbone, and how was the GPT-4o filter obtained?** (Determines 2 vs. 4 filter trainings.)
19. `[REC]` **What is the true achievable FRB-PAIRS count after all matching constraints?**
20. `[REC]` **How many MedChangeQA change points fall after the label model's pre-training cutoff?**

---

## 17. Assumptions Requiring Verification

| # | Assumption | Source | Verification route | Priority |
|---|---|---|---|---|
| A1 | Newer evidence in FRB-PAIRS lies outside the label model's parametric priors | `[INF]` implicit in D1 §1.2 mechanism | Date-histogram MedChangeQA change points vs. model cutoff | **Blocking** |
| A2 | The permutation control can detect prose-era artefacts | D1 §4.6 V1 | Analytic — it cannot; redesign | **Blocking** |
| A3 | 150–300 matched pairs are obtainable from 512 MedChangeQA items | D1 §5.2 | Pilot extraction, n≈40, measure attrition | **Blocking** |
| A4 | The currency pack is retrievable under frozen balanced retrieval | not stated in D1 | Measure currency-pack recall@k on AD questions | **Blocking (Stage 1)** |
| A5 | Perplexity labels require per-backbone retraining | D1 §2.1, §5.3 | Read `github.com/dmis-lab/RAG2` | High (cost) |
| A6 | Baseline labels are predominantly confidence-derived | D1 §1.2 | Instrument the label pipeline; report branch proportions | High |
| A7 | 20–30k subsampled label items suffice to reproduce B2 | D1 §5.3 `[DES]` | Cost-verify on 100 items first (D1 already requires this) | High |
| A8 | Claim-class tagging precision is adequate | D1 §5.1 `[ASM]` | 300-passage audit, report P/R | Medium |
| A9 | Guideline PDFs yield poorer date coverage | D1 §5.1 `[ASM]` | Report per-corpus null-date rate | Medium |
| A10 | Genuine vs. apparent contradiction is separable enough to demonstrate | D1 §4.4 `[ASM]` | Manual audit of CONTESTED-AD | Medium |
| A11 | Three clinician evaluators are recruitable | D1 §7.5 `[ASM]` | Begin recruitment now — 32-week lead time | Medium |
| A12 | CoRM-RAG is as described (principal novelty threat) | D1 §2.5 `[ASM]` | Verify arXiv:2605.01302 authorship/venue/content | High |
| A13 | 24 GB GPU, 64 GB RAM, 1–2 TB SSD suffice | D1 §5.3 `[ASM]` | Index-build dry run on PubMed-AD | Medium |

---

## 18. Mapping of Documents to the Six Pipeline Stages

| Stage | D1 (Proposal) | D2 (RAG², NAACL 2025) |
|---|---|---|
| **0. Pre-registration** *(implicit, D1 §9 Phase 0)* | §7.3 pre-registration; §1.6 hypotheses; §9 pre-declared pivots | — |
| **1. Build AD Corpus** | §5.1 corpus construction, metadata schema, chunking, dedup, date propagation, claim-class tagging | §3.4 balanced retrieval; Table A1 corpus statistics; §A.3 chunking (sliding window with overlap); MedCPT retriever/reranker |
| **2. Build Test Pairs** | §4.2 pair construction; §5.2 provenance firewall and all eight evaluation sets | §4.1 MedQA/MedMCQA/MMLU-Med; Table 1 splits; §A.2 dataset descriptions |
| **3. Run the Bias Probe** | §4.2 probe design and label functions; §4.6 V1–V5; §7.3 statistical plan; §7.4 A1–A3 | §3.2 the label function under test; Fig. 2 annotation decision tree; Eq. 3–4; Table 3 filtering-method ablation |
| **4. Build SCAF** | §4.3 currency; §4.4 contested; §4.5 admission score + Algorithm 1; §3 architecture | §3.2 the component SCAF replaces; §A.3 training hyperparameters; Limitations (one-snippet-at-a-time context constraint) |
| **5. Compare All Variants** | §6.1 arms; §6.2 variables; §6.3 fairness guarantees; §6.4 counterfactual update; §7.1–7.5 metrics, decoupling, ablations, clinical evaluation | Table 2 main results and baseline set; Fig. 3 top-k sweep; Tables 4–5 balanced-retrieval and rationale-quality ablations; §A.4 open-ended evaluation |
| **6. Analyze & Write Up** | §7.5 error taxonomy; §8 contributions; §10 limitations and ethics | §5.3 case study; §6 conclusion; Limitation section |

`[INF]` D2 supplies the frozen substrate for Stages 1, 4 and 5 and the object under test in Stage 3. It supplies **nothing** for Stage 2 as this thesis needs it — no temporal data, no dates, no matched pairs. Stage 2 is where the thesis is genuinely on its own, and it is also the stage with the most `[OPEN]` items and the largest unpriced risks. That coincidence is not an accident and should drive sequencing.

---

## 19. Current Readiness of Each Pipeline Stage

| Stage | Status | Evidence established | Blocking gaps |
|---|---|---|---|
| **0. Pre-registration** | **Not Ready** | Hypotheses, tests, units of analysis, multiplicity correction, pivot triggers all specified in D1 §1.6, §7.3, §9 | δ unset; power analysis not run; sampling frame undefined so power analysis is not yet possible |
| **1. Build AD Corpus** | **Not Ready** | Sources, scoping rules, metadata schema, chunking, dedup, date-propagation risk, tagging plan and audit all specified | Nothing built. Currency-pack retrieval quota undecided; no retrievability acceptance criterion; no snapshot-freezing protocol |
| **2. Build Test Pairs** | **Not Ready** | Provenance firewall designed; eight evaluation sets defined with sizes and purposes; matching dimensions named | Passage source unspecified; two matching criteria `[OPEN]`; attrition unmodelled; no pre-training-cutoff stratification |
| **3. Run the Bias Probe** | **Not Ready** | Δ operationally defined; filter-identity argument sound; statistics appropriate; V3/V4 well specified; informative-null design in place | V1 vacuous for the primary claim; V5 not designated blocking; prior conflation unresolved; label-branch mixture unmeasured |
| **4. Build SCAF** | **Not Ready** | Algorithm 1 complete and implementable; γ, σ, τ, ρ defined; ordering of contested-before-superseded justified; complexity bounded | Depends on Stage 3. Six `[OPEN]` parameters. Structural change vs. label swap not distinguished in the fairness argument |
| **5. Compare All Variants** | **Not Ready** | Six arms defined; six fairness guarantees; counterfactual update procedure; metrics with stated weaknesses; judge guard; dual reporting; both directions of effect measured | Depends on Stage 4. δ unset. Clinician recruitment not begun (32-week lead). Ablation triage not done |
| **6. Analyze & Write Up** | **Not Ready** | Error taxonomy with ten categories over logged intermediate state; decoupling analysis; limitations and ethics drafted to a high standard | Depends on Stage 5 |

---

## 20. Recommended Immediate Next Step

**A two-week Stage-1 feasibility audit, executed before any corpus-construction code is written.** Four checks, each deterministic, cheap, and capable of redirecting or killing the thesis at the point where redirection is inexpensive. This is consistent with D1's own principle — "If the primary claim fails, that must be discovered in month five, when a pivot is inexpensive, rather than in month nine" — applied one stage earlier than D1 applies it.

**Check A — Temporal reach of MedChangeQA vs. pre-training cutoff.** *(Answers A1; potential thesis-breaker.)*
Obtain MedChangeQA (Vladika et al., Findings of EMNLP 2025). For each of the 512 changed-verdict pairs, extract the date of the superseded verdict and of the current verdict. Histogram the change points. Count how many fall **after** the pre-training cutoff of the intended label-generating model (Llama-3-8B ≈ Dec 2023; Meerkat-7B via Mistral-7B ≈ 2023). Note that MedRevQA is sourced from PubMed 2000–January 2024, so the post-cutoff stratum may be small or empty.
*Decision rule:* if the post-cutoff stratum is too small to power H1, the thesis must choose explicitly between (i) an earlier-cutoff label model, which enlarges the post-cutoff stratum without breaching the provenance firewall — this is the cheapest fix and should be costed first; (ii) reframing H1's mechanism as **frequency dominance** (the old verdict is over-represented in pre-training even when both appear) rather than **cutoff exclusion**, which is weaker but still testable and must be pre-registered as the mechanism; or (iii) supplementing FRB-PAIRS with thesis-curated material — which breaches the firewall and should be the last resort.

**Check B — Redesign the validity gate.** *(Answers A2; blocking for Stage 3.)*
Establish analytically that V1 cannot discriminate for the primary comparison: neither Filter A nor Filter B receives a publication date as input, so permuting dates re-labels which member of each pair is "older" without changing any filter output, driving Δ→0 by construction. *(V1 does remain meaningful for the date-annotated prompting condition of ablation A6, where dates are genuinely in the input — restrict it to that scope rather than deleting it.)*
Promote **V5/NEG-CONTROL to the blocking validity gate**: on claims that did not change across the window, old and new passages differ in prose era but not in claim content, so Δ ≈ 0 on NEG-CONTROL alongside Δ > 0 on FRB-PAIRS is what isolates claim recency from era style. Add a content-preserving **age-cue ablation** (normalise in-text citation years, era-specific drug nomenclature and hedging markers) as a second discriminator. Re-word H4 and contribution C1 accordingly — C1 currently says "validated by a permutation control".

**Check C — Mine the released RAG² code.** *(Answers A5, A6, A7 and questions 16–18; de-risks Objectives 1 and 4.)*
`github.com/dmis-lab/RAG2` is public (D2 §1) and is not referenced anywhere in D1. Recover: the exact label decision tree and the proportion of labels arriving via the correctness-flip branch vs. the perplexity branch; whether ΔPPL is computed over the rationale or the query; whether separate filters were trained per backbone and how the GPT-4o filter was obtained; the label-generation scripts, prompts and thresholds. This determines whether B2 reproduction costs two filter trainings or four, and — more importantly — whether the thesis's central mechanism claim is aimed at the dominant label channel or a minority one. Fix D1 §1.2's characterisation to match §4.2 either way.

**Check D — Pilot FRB-PAIRS, then power, then pre-register.** *(Answers A3 and question 14; blocking for Stage 2 and Stage 0.)*
Specify where the old and new *passages* come from. Extract a pilot of ~40 pairs end to end under the full matching constraint set. Report attrition at each constraint and the realised pair yield. Use the pilot to estimate the effect-size variance, then run the power analysis on the real sampling frame, then set δ, then pre-register and timestamp. D1 currently schedules the power analysis (Phase 0, weeks 1–2) before pair construction (Phase 3, weeks 10–16); the dependency runs the other way.

**One addition to Stage 1 itself, to be adopted when corpus work begins.** `[REC]` Add a **currency-pack retrievability acceptance criterion**: for a held-out set of AD questions, measure how often the frozen retriever places at least one currency-pack passage in the reranked candidate set. This is a precondition for SCAF having any measurable effect — an admission policy cannot admit what retrieval never surfaces — and D1's own cited expert evaluation (22% top-16 relevance; 31% of queries with no relevant passage) makes a low value plausible. Resolve the currency pack's balanced-retrieval quota at the same time, noting that giving it its own quota modifies the frozen upstream stage and must be disclosed as a deviation from B2 fidelity.

**Explicitly not recommended now:** building the supersession table; building the claim-class taxonomy at scale; parsing PMC full text; implementing SCAF's contested detector; any weight tuning. All are Stage-1/4 work whose scope depends on the four answers above.

---

## Stage Gate

**Current Stage:** Stage 1 — Build Alzheimer's Corpus

**Status:** **Not Ready**

**Established Evidence:**
- A complete, internally consistent and unusually rigorous research design exists (D1), with pre-registration, an informative-null design, a provenance firewall, six fairness guarantees, a judge-validation guard, mandatory dual reporting and pre-declared pivots.
- The foundational system is fully characterised and every load-bearing quantitative claim D1 makes about it has been verified against D2 (corpus statistics, reported gains, label rule, rationale-query rationale, absence of temporal representation).
- The research gap survives independent scrutiny: no located work instruments the admission stage for temporal asymmetry.
- Corpus sources, scoping rules, metadata schema, chunking, deduplication and tagging plan are specified to an implementable level of detail.

**Missing Requirements:**
- Nothing is built. The repository is empty; no corpus, index, pilot, pre-registration or code exists.
- Four assumptions with thesis-redirecting consequences are unverified: A1 (pre-training-cutoff reach), A2 (validity-gate discriminability), A3 (achievable pair count), A4 (currency-pack retrievability).
- The currency pack has no balanced-retrieval quota decision and the corpus has no retrievability acceptance criterion or snapshot-freezing protocol.
- D1 §1.2 mischaracterises the baseline label function relative to D1 §4.2 and D2 §3.2; the thesis's central mechanism claim is currently aimed at a label channel of unmeasured share.

**Critical Risks:**
- **F2 (highest).** If MedChangeQA change points precede the label model's pre-training cutoff, H1's mechanism cannot fire and a null result is uninterpretable rather than informative — converting D1's designed-for null into the "inconclusive outcome" D1 itself names as materially worse for a thesis.
- **F1.** The designated blocking validity gate (V1/H4) cannot fail for the primary comparison, because no filter under test receives a date as input. Construct validity for C1 currently rests on a control that does not discriminate.
- **F4.** The primary instrument for the primary claim (FRB-PAIRS) has no specified passage-construction procedure.
- **F5.** SCAF's achievable effect is bounded by whether frozen retrieval ever surfaces the 20–40 currency-pack documents; this is unmeasured and unquotaed.
- **F3.** Baseline label-branch mixture unmeasured, so "confidence-derived" is not yet a verified description of what is being probed.

**Required Action:**
1. Execute Checks A–D (two weeks, deterministic, no corpus code).
2. Reconcile D1 §1.2 with §4.2 and D2 §3.2; re-word H4 and contribution C1 away from the permutation control.
3. Begin clinician recruitment in parallel — a 32-week lead time on a 44-week schedule means this cannot wait for Stage 5.
4. Freeze corpus snapshot protocol (query strings, MeSH explosion settings, pull dates, per-collection counts) before the first PubMed pull.

**Completion Criteria for Stage 1:**
- Checks A–D answered in writing, with the sequencing consequences of each recorded in the ledger.
- Corpus built to the D1 §5.1 specification with per-corpus null-date rates and claim-class tagging precision/recall reported from the 300-passage audit.
- Currency-pack quota decision made and disclosed; currency-pack recall@k measured and above a pre-declared floor.
- Snapshot protocol frozen and the corpus reproducible from recorded parameters.
- FRB-PAIRS passage provenance specified and a ~40-pair pilot extracted with measured attrition.

**Do Not Proceed To:** Stage 3 (Run the Bias Probe), Stage 4 (Build SCAF), Stage 5 (Compare All Variants), Stage 6 (Analyze & Write Up). Stage 2 (Build Test Pairs) may proceed **only** to pilot scale, because the Stage-1 corpus is not on the critical path for FRB-PAIRS if its passages come from external MedChangeQA provenance — and Check D is what determines whether that is so.

---

## Parking Lot / Future Work

Recorded so they do not distract from the current stage. None is necessary to answer RQ1–RQ6.

| Idea | Why parked | Re-entry condition |
|---|---|---|
| Improving the retriever | Freezing it is the central internal-validity guarantee (D1 §6.3.1) | Never within this thesis |
| Agentic / RL retrieval | Out of scope on a single-GPU budget (D1 §1.7); B4 retained as a non-competing reference point | Never within this thesis |
| Multi-passage joint labelling (D2's own stated limitation) | Would change the filter architecture and break capacity matching | Only if the probe's single-passage framing is shown to be the binding constraint |
| Generalising beyond Alzheimer's | D1 §10 already calls this a hypothesis, not a result | Post-thesis |
| Multimodal / imaging evidence | Not in the admission-stage scope | Post-thesis |
| Full 564.2 GB corpus reproduction | Domain scoping is what makes the hardware sufficient | Never within this thesis |
| Fine-tuning the hallucination judge (F1 0.52 → 0.62) | Judge validation guard already handles judge weakness by demoting automatic rates | Only if κ < 0.6 *and* expert ratings are also unavailable |
| Larger generators (>8B) | Out of declared scope | Post-thesis |
| Treating SCAF as a deployable clinical tool | D1 correctly forbids this framing | Never |
