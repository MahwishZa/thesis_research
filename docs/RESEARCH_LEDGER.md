# Research Ledger

Living record for the thesis *Does the Filter Prefer the Past?* — updated whenever new
information changes a decision. Companion to `RESEARCH_UNDERSTANDING_REPORT.md`.

**Last updated:** 2026-09-11 · **Current pipeline stage:** Stage 1 (Build Alzheimer's Corpus) · **Status:** Not Ready

Evidence tags: `[DOC]` demonstrated by the provided documents · `[PUB]` reported by published research ·
`[INF]` methodological inference · `[STU]` student assumption · `[REC]` recommendation by the assistant

---

## 1. Established Facts

| # | Fact | Tag | Source |
|---|---|---|---|
| E1 | RAG²'s filter is Flan-T5-large (770M), trained on labels from a correctness-flip decision tree with a perplexity differential as tie-breaker at τ = top 25% | `[DOC]` | D2 §3.2, Fig. 2, Eq. 3 |
| E2 | The primary label criterion is the correctness flip; perplexity was introduced explicitly "to address" cases where accuracy is unchanged | `[DOC]` | D2 §3.2 |
| E3 | Perplexity is computed over the model's generated **rationale**, per D2's Intro and §5.1; D2's Eq. 4 notation is ambiguous and appears to score the query | `[DOC]` | D2 §1, §5.1 vs Eq. 4 |
| E4 | RAG² corpus: 37.6M docs / 116.7M passages / 564.2 GB across PubMed, PMC, CPG, 18 textbooks | `[DOC]` | D2 Table A1 |
| E5 | Reported gains +6.1 / +3.8 / +0.9 avg accuracy points (Llama-3-8B 60.2→66.3; Meerkat-7B 68.6→72.4; GPT-4o 86.0→86.9) | `[DOC]` | D2 Table 2 |
| E6 | Retrieval uses the rationale as query (original query excluded for length); reranking uses the **original query** | `[DOC]` | D2 §3.3, §3.4 |
| E7 | RAG² contains no temporal representation anywhere in corpus, index, retriever, reranker or filter | `[INF]` | Absence across D2 (15 pp.) |
| E8 | RAG²'s only open-ended evaluation is ClinicalQA25 — 25 queries, ROUGE-L and BERTScore | `[DOC]` | D2 §A.4 |
| E9 | RAG² code is publicly released at `github.com/dmis-lab/RAG2`; the trained checkpoint is not distributed | `[DOC]` | D2 §1; D1 §1.5 |
| E10 | D2 demonstrates cross-**dataset** filter transfer (MedMCQA-trained → MMLU-Med, +5.3/+4.9). It does not demonstrate cross-**backbone** transfer, and never states how the GPT-4o filter was obtained | `[DOC]` | D2 §4.3, §4.2 |
| E11 | D2's stated limitations include: snippets labelled individually, ignoring combined effects; Flan-T5 context limit means one snippet filtered at a time | `[DOC]` | D2 Limitation |
| E12 | Largest medical-RAG expert evaluation to date: 18 clinicians, 80,502 annotations, 800 outputs; 22% top-16 relevance, 31% of queries with no relevant passage, 33% must-have coverage | `[PUB]` | arXiv:2511.06738 via D1 §1.1 |
| E13 | MedChangeQA: 512 changed-verdict pairs, derived from MedRevQA (16,501 pairs, PubMed 2000–Jan 2024) | `[PUB]` | Vladika et al. 2025 via D1 §2.3 |
| E14 | Cochrane CD016297 (April 2026, 17 trials, 20,342 participants) concluded anti-amyloid mAbs probably produce little/no clinically meaningful cognitive difference at 18 months; contested within days by the Alzheimer's Association, UK DRI and a *Lancet* commentary | `[AU]` | D1 §1.3 |
| E15 | Repository is empty. No corpus, index, code, pre-registration or results exist | `[DOC]` | Repo inspection 2026-09-11 |

---

## 2. Decisions

### Taken by the student (in D1)

| # | Decision | Rationale | Status |
|---|---|---|---|
| D-1 | Freeze everything upstream of admission; arms are distinct functions over one cached candidate list | Central internal-validity guarantee | **Endorsed** — strongest part of the design |
| D-2 | Provenance firewall: primary claim rests on externally-authored MedChangeQA; thesis-curated material supports replication and case study only | An earlier formulation authored items, supersession table and corpus from the same ~20 documents, making a positive result unfalsifiable | **Endorsed** — protect under pair-count pressure |
| D-3 | Soft supersession (down-weight), hard rejection only for retraction/withdrawal | Claim-class tagging precision is unverified; a hard gate would silently delete correct evidence in proportion to tagging error | **Endorsed** |
| D-4 | Test contested **before** superseded | Reverse order would present the live Cochrane controversy as resolved | **Endorsed** — load-bearing |
| D-5 | Condition currency decay on ψ(q) | An unconditional recency prior would penalise correct older sources on time-invariant questions | **Endorsed** — makes H7 plausible at all |
| D-6 | Include B3 (current SOTA) as a required arm | Comparing only to the base paper benchmarks against a system the field has already improved on | **Endorsed** |
| D-7 | Treat source authority τ as a tested variable (A12), not an assumed ordering | The Cochrane episode shows the ordering is contestable | **Endorsed** |
| D-8 | Main comparison uses a prompt template **without** date annotations; date-annotated prompting is a separate co-primary condition | Gains must attribute to which passages were admitted, not to date cues | **Endorsed** — pre-empts the obvious objection |
| D-9 | Judge validation guard: κ < 0.6 demotes automatic rates, expert ratings become primary | Automatic claim judges are weak (F1 0.44–0.52 zero-shot) | **Endorsed** |
| D-10 | Mandatory dual reporting of every rate metric (conditional on answering; and with abstentions as failures) | An abstaining system computes rates over a smaller, easier denominator | **Endorsed** |
| D-11 | Phase 4 (probe) precedes Phase 5 (SCAF) | Discover a failed primary claim in month five, not month nine | **Endorsed** — extend the same logic one stage earlier (see R-1) |
| D-12 | Rank-based rather than min-max normalisation of ρ(s) | Makes a global θ_admit well defined across queries | **Endorsed** |
| D-13 | Both filters trained on general medical QA, applied zero-shot to AD items | Removes suspicion that domain gains come from domain-specific fine-tuning | **Endorsed** |

### Recommended by this assistant (pending student/supervisor acceptance)

| # | Recommendation | Reason |
|---|---|---|
| R-1 | Run a two-week Stage-1 feasibility audit (Checks A–D) before writing corpus code | Four assumptions can redirect or kill the thesis; all four are cheap to test now |
| R-2 | Demote V1/H4 to the date-annotated condition only; promote **V5/NEG-CONTROL to the blocking validity gate**; add a content-preserving age-cue ablation | V1 cannot discriminate for the primary comparison (see Risk K1) |
| R-3 | Re-word contribution C1 and H4 away from "validated by a permutation control" | Follows from R-2 |
| R-4 | Reconcile D1 §1.2 with D1 §4.2 and D2 §3.2; measure and report the label-branch mixture | The thesis's central mechanism claim currently names a label channel of unmeasured share |
| R-5 | Add a **currency-pack retrievability acceptance criterion** to Stage 1, and decide the currency pack's balanced-retrieval quota | SCAF cannot admit what retrieval never surfaces |
| R-6 | Promote ablation A10 (teacher at inference vs. distilled student) into Stage 3 | Disentangles the label-generating LLM's priors from the Flan-T5 student's priors |
| R-7 | Re-sequence: pilot pair extraction → attrition/effect-size estimate → power analysis → set δ → pre-register | D1 currently schedules the power analysis before the sampling frame exists |
| R-8 | Freeze a corpus snapshot protocol (query strings, MeSH explosion settings, pull dates, per-collection counts) before the first PubMed pull | Live-database snapshots are otherwise irreproducible, and §6.4's counterfactual update is undefined without them |
| R-9 | Mine `github.com/dmis-lab/RAG2` before budgeting Phase 2 | Determines 2 vs. 4 filter trainings and resolves four open questions at once |
| R-10 | Begin clinician recruitment now | 32-week lead time on a 44-week schedule |

---

## 3. Assumptions (not yet verified)

| # | Assumption | Priority | Verification route |
|---|---|---|---|
| A1 | Newer evidence in FRB-PAIRS lies **outside** the label model's parametric priors | **Blocking** | Histogram MedChangeQA change points vs. Llama-3-8B / Mistral-7B cutoffs |
| A2 | The permutation control can detect prose-era artefacts | **Blocking** | Analytic — it cannot; redesign per R-2 |
| A3 | 150–300 matched pairs are obtainable from 512 MedChangeQA items | **Blocking** | Pilot extraction n≈40; measure attrition per constraint |
| A4 | The currency pack is retrievable under frozen balanced retrieval | **Blocking** | Measure currency-pack recall@k on held-out AD questions |
| A5 | Perplexity labels require per-backbone filter retraining | High (cost) | Read released RAG² code |
| A6 | Baseline labels are predominantly confidence-derived | High | Instrument the label pipeline; report branch proportions |
| A7 | 20–30k subsampled label items suffice to reproduce B2 | High | Cost-verify on 100 items (D1 already requires this) |
| A8 | Claim-class tagging precision is adequate for down-weighting | Medium | 300-passage audit; report P/R |
| A9 | Guideline PDFs yield poorer date coverage than XML | Medium | Report per-corpus null-date rates |
| A10 | Genuine vs. apparent contradiction is separable enough to demonstrate mechanism | Medium | Manual audit of CONTESTED-AD |
| A11 | Three clinician evaluators are recruitable | Medium | Begin now |
| A12 | CoRM-RAG (arXiv:2605.01302) is as characterised — the principal novelty threat | High | Verify authorship, venue, content |
| A13 | 24 GB GPU / 64 GB RAM / 1–2 TB SSD suffice | Medium | Index-build dry run on PubMed-AD |

---

## 4. Open Questions

**Carried from D1's own `[OPEN]` tags:** pair-matching length tolerance band · pair-matching topical-similarity
criterion · whether the discriminativeness margin enters A(s) · ψ(q) as rule vs. trained classifier ·
contest-window length · non-inferiority margin δ · weight-selection parameterisation for w1..w4 ·
replication filter backbone · entailment teacher NLI corpus · corpus scale estimates ·
judge-validation sample size · power analysis · institutional review requirements.

**Added by this report:**
1. Where do FRB-PAIRS **passages** come from? *(blocking)*
2. Does the currency pack get its own balanced-retrieval quota? *(blocking for Stage 1)*
3. What fraction of RAG² training labels come from the correctness-flip branch vs. the perplexity branch?
4. Is ΔPPL computed over the rationale or the query? (D2's prose and Eq. 4 disagree.)
5. Did D2 train a separate filter per backbone, and how was the GPT-4o filter obtained?
6. What is the true achievable FRB-PAIRS count after all matching constraints?
7. How many MedChangeQA change points fall after the label model's pre-training cutoff?

---

## 5. Evidence Gaps

- No empirical evidence of any kind yet exists for this thesis. Every quantitative statement about the proposed system is a hypothesis or design target, as D1 itself declares on its cover page.
- The claim that RAG² has no temporal representation rests on absence across D2's 15 pages; the released code would settle it definitively.
- The claim that the signal family is expanding ("now underlies reward signals in learned retrieval systems") rests on `[PP]` preprints whose authorship D1 declines to vouch for.
- The gap claim "no located work measures filter-stage admission asymmetry" is a negative result from a literature search that is not itself documented. A recorded, reproducible search protocol would strengthen it materially.

---

## 6. Risks

| # | Risk | Stage | Severity | Mitigation |
|---|---|---|---|---|
| K1 | **V1/H4 is vacuous for the primary comparison.** Neither filter under test receives a publication date, so permuting dates cannot change filter output; Δ→0 by construction. The designated blocking gate cannot fail, so passing it certifies nothing | 3 (fix at 2) | **High** | R-2, R-3 |
| K2 | **Mechanism precondition untested.** If change points precede the label model's pre-training cutoff, H1 cannot fire and a null is uninterpretable | 2 | **Critical** | Check A; stratify FRB-PAIRS by cutoff; pre-register which mechanism (cutoff exclusion vs. frequency dominance) is claimed |
| K3 | **FRB-PAIRS passage provenance unspecified.** The primary instrument for the primary claim has no construction procedure | 2 | **Critical** | Check D |
| K4 | **Currency-pack retrievability ceiling.** 20–40 documents with no quota inside a ~10³ CPG pool may never be retrieved; SCAF's effect is then bounded near zero | 1 | **High** | R-5 |
| K5 | **Label-branch mixture unmeasured**, so "confidence-derived" is not yet a verified description of the probe's target | 3 | **High** | R-4, R-9 |
| K6 | **Prior conflation** — teacher LLM's priors vs. filter student's priors are not separated | 3 | Medium | R-6 |
| K7 | **H2 is not label-informativeness-matched.** Entailment-against-gold is a richer supervision signal than correctness-flip; an examiner will say the comparison is unsurprising | 3/5 | Medium | Argue that the label *is* the contribution; keep H1 (baseline-only) as the primary claim |
| K8 | **Inconclusive rather than negative outcome.** D1 names this itself as materially worse for a thesis | 3 | **High** | R-7 (power before construction), Check A |
| K9 | **Pair-count pressure breaches the provenance firewall** — the cheapest way to reach 150–300 pairs is to author them | 2 | **High** | Pre-register that the lanes do not cross |
| K10 | **Corpus irreproducibility** — live-database snapshots without frozen parameters; §6.4's counterfactual update is undefined without them | 1 | Medium | R-8 |
| K11 | **Clinician recruitment fails** (D1's own largest schedule risk) | 5 | Medium | R-10; D1's pre-declared fallback to checklist evaluation |
| K12 | **Reproduction cost overrun** — label regeneration approaches millions of forward passes, ×4 if per-backbone retraining is required | 1/2 | **High** | R-9; D1's own 100-item cost verification before fixing the subsample |
| K13 | **Novelty threat unverified** — CoRM-RAG characterised from a preprint D1 does not vouch for | 0 | Medium | A12 |
| K14 | **"Only the label function differs" holds for the probe but not for the full SCAF arm**, which is a structural change (contested state needs pairwise reasoning; RAG²'s filter processes one snippet at a time) | 4/5 | Medium | Distinguish the two comparisons explicitly in the fairness argument |
| K15 | **Ablation set (A1–A14) is large for a single-GPU, 44-week budget** | 5 | Medium | Triage; name A5/A7 load-bearing for C2 alongside A1–A3 |

---

## 7. Supervisor Input Required

These should not be settled by an AI assistant alone.

1. **The non-inferiority margin δ.** D1 offers two accuracy points "for discussion, not adopted." This is a scientific judgement about clinically tolerable regression and must be pre-registered with justification.
2. **Response to Check A if the post-cutoff stratum is too small.** The three options (earlier-cutoff label model / reframe the mechanism as frequency dominance / supplement FRB-PAIRS and breach the firewall) have very different consequences for the thesis's defensibility. Option three should require explicit supervisory sign-off.
3. **Whether to re-word contribution C1 and hypothesis H4** away from the permutation control. This changes a stated contribution in an approved proposal.
4. **Scope triage if the schedule slips.** D1 §5.4 defines a Minimum Viable Implementation; confirming now which later phases are expendable is cheaper than deciding under pressure in month eight.
5. **Institutional review requirements** for the expert rating study (D1 leaves this `[OPEN]`).
6. **Whether giving the currency pack its own retrieval quota is an acceptable deviation** from B2 fidelity, given that it modifies the frozen upstream stage.
7. **Rater pooling rule** — whether the dementia specialist is pooled with trainees or treated as a separate stratum. Must be fixed before packet assembly.

---

## 8. Parking Lot / Future Work

None of these is necessary to answer RQ1–RQ6. Recorded so they do not distract.

| Idea | Why parked | Re-entry condition |
|---|---|---|
| Improving the retriever | Freezing it is the central internal-validity guarantee | Never within this thesis |
| Agentic / RL retrieval | Out of scope on a single-GPU budget; B4 is a non-competing reference point | Never within this thesis |
| Multi-passage joint labelling (D2's own limitation) | Changes filter architecture; breaks capacity matching | Only if single-passage framing is shown to be the binding constraint |
| Generalising beyond Alzheimer's | D1 already calls this a hypothesis, not a result | Post-thesis |
| Multimodal / imaging evidence | Outside the admission stage | Post-thesis |
| Full 564.2 GB corpus reproduction | Domain scoping is what makes the hardware sufficient | Never within this thesis |
| Fine-tuning the hallucination judge (F1 0.52 → 0.62) | The judge validation guard already handles judge weakness | Only if κ < 0.6 **and** expert ratings are unavailable |
| Larger generators (>8B) | Outside declared scope | Post-thesis |
| Treating SCAF as a deployable clinical tool | D1 correctly forbids this framing | Never |

---

## 9. Change Log

| Date | Change |
|---|---|
| 2026-09-11 | Ledger created. Initial research understanding built from D1 (proposal) and D2 (RAG², NAACL 2025). Fifteen established facts recorded; thirteen student decisions endorsed; ten recommendations raised; thirteen assumptions and fifteen risks registered. Stage 1 assessed **Not Ready** pending the Checks A–D feasibility audit. |
