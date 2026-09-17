# Research Ledger

Living record for the thesis — updated whenever new information changes a
decision. Companion to `research_understanding.md`.

**Current research question** (D-36; `docs/frozen_scope.md` governs): does the
proposed solution/system reduce the rate of hallucinated answers in
Alzheimer's disease question answering, relative to the baseline system, under
identical question and evidence conditions, while maintaining comparable QA
accuracy? Primary outcome: hallucination rate. Secondary: QA accuracy.

Decisions recorded before D-36 were taken under the superseded
admission-asymmetry framing. They are kept for provenance; where one concerns
what is *measured* rather than what is *built*, D-36 supersedes it.

**Last updated:** 2026-09-17 · **Current pipeline stage:** Step 1 (corpus) and Step 2 (questions) in progress; Steps 3–12 infrastructure built, unexecuted · **Status:** Not executable yet

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
| E14 | The experimental specification names **no** evaluation dataset: §10.3's change-point source was `[TO BE SPECIFIED]`. MedChangeQA's identity as the primary instrument comes from D1 via the ledger and understanding report, not from the specification. Recorded in §10.3 on 2026-09-15 so the primary instrument is named where the method is defined | `[DOC]` | Specification §10.3; D1 §5.2 |
| E15 | **Premise of D-20 — VERIFIED 2026-09-16, see D-34.** Superseded text follows. That each MedChangeQA item ships both review records is an inference from MedRevQA's construction, not a checked fact — the dataset is unreachable from the development environment, and the understanding report notes it "does not obviously supply *matched passage pairs*". If it does not, the fallback is a deterministic PMID lookup, which changes no design decision; the pipeline reports 100% `missing_evidence_text` rather than guessing | `[INF]` | Understanding report §Findings 1; egress probe |
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
| D-14 | Contested evidence is capped by the common context budget, never exempted from it. When contested evidence exceeds the budget it is retained first, by A(s) descending with `evidence_id` as tie-break, up to the budget; preserved and dropped contested positions are recorded in the run metadata | §16 makes the context budget invariant across arms; §29 requires both positions of a contested claim to survive. The two conflict only when contested evidence exceeds the budget. Exempting contested evidence would give SCAF more context than the baseline, so any Stage-5 difference would be confounded by context volume rather than by admission policy. Capping keeps the arms comparable and makes the cost visible instead of silent | **Fixed implementation policy** — not a supervisor-dependent scientific parameter |
| D-15 | Claim equivalence in the primary pool is inherited from the external item, not judged by the thesis: both passages are evidence for the same externally-authored question | Specification 10.2 leaves the "same claim" criterion `[TO BE SPECIFIED]`. Any thesis-authored equivalence judgement would have to be validated, at a cost the schedule cannot carry, and would weaken the provenance firewall. Inheriting it costs nothing and is auditable | **Adopted for Stage 2** |
| D-16 | Temporal eligibility is decided by date *intervals*, not point estimates: the older passage qualifies only when its date interval ends before the newer passage's interval begins | A date of `2023` means some day in 2023. Two passages both dated `2023` cannot be ordered, and a point estimate would silently order them anyway. The rule needs no invented tolerance parameter and excludes exactly the pairs that cannot support a recency claim | **Adopted for Stage 2** |
| D-17 | ~~`question_date` falls back to the newer side's publication date~~ | ~~A single global cutoff would make the older stratum stale by construction~~ | **SUPERSEDED by D-21 (2026-09-15).** The rationale was wrong: because γ decays in `t_q − date(s)`, the *difference* in age between the two passages is the publication separation whatever t_q is, so a global date does not differentially disadvantage older evidence. What the fallback did do was set the newer passage's γ to exactly 1 in every pair, maximising the currency contrast by construction |
| D-18 | Dev/validation/test assignment is a pure function of `question_id` and a recorded seed, with the question — not the pair — as the unit | Pairs sharing a question share its answer and usually a passage, so splitting on pairs leaks test items into the tuning partition (specification 33.1). Hash assignment is stable when questions are added, so the test partition cannot quietly change composition between pilot and final set | **Adopted for Stage 2**; proportions (0.2/0.2/0.6) remain configurable |
| D-19 | Automatic claim-class labels are recorded with `claim_class_source` and are never used as a matching criterion in the primary pool | The corpus's current labels come from a keyword-from-classname heuristic at confidence 0.0–0.4 (ledger A8). Treating them as ground truth would put unvalidated labels inside the inclusion rule for the primary evaluation set | **Adopted for Stage 2** |
| D-20 | **The primary pool's evidence travels with the external item; there is no retrospective mapping onto corpus chunks.** The Alzheimer's corpus is the retrieval population that supplies the rest of the candidate set | MedChangeQA-style items are built from two systematic-review records, so both passages are already determined, each with an identifier and a date. Specification 15.2 defines the admission effect as what happens "when the same candidate set is available", so Stage 3 injects both passages into one cached candidate set and replays it across arms; whether retrieval would have found them is a retrieval effect (15.1) and a secondary measurement. This resolves finding F4 and removes an unsolved dependency without any lexical or semantic matching | **Adopted** |
| D-21 | **`question_date` is the dataset's own date when supplied, otherwise one experiment-wide `evaluation_as_of_date`. It is never derived from a passage in the pair**, and the schema rejects such a value | Setting t_q to the newer passage's publication date gives that passage γ = 1 in every pair, inflating the currency contrast SCAF is measured on — and only SCAF, since the RAG² filter never sees dates. A fixed as-of date later than all evidence compresses γ differences instead, understating SCAF's advantage. The conservative direction is the defensible one | **Adopted; replaces D-17** |
| D-22 | **Claim classes are optional diagnostic metadata, not an inclusion criterion.** Contested-evidence detection and the supersession table move to secondary analysis | With claim equivalence inherited from the external item (D-15), nothing in the primary probe needs a claim class. The only components that did — contested detection and supersession — are not the central hypothesis, and both depend on metadata that does not exist. `claim_class_source` is still recorded | **Adopted** |
| D-23 | **No target pair count.** Use the whole eligible pool, report effect size with a confidence interval, and state detectable-effect sensitivity across plausible discordant rates | The pool is fixed by the external dataset and costs nothing per item, so "how many do we need?" is the wrong question; "what can this many detect?" is the right one. A pre-registered target computed from an assumed effect size would be fabricated power | **Adopted** |
| D-24 | **Stage 2 emits a frozen, hashed evaluation specification; Stage 3 performs all retrieval and asserts the hash before running.** Stage 2 performs none | Prevents duplicated retrieval logic, arms receiving different evidence, and post-hoc pair modification (specification 33.5, 33.6). `verify_frozen()` makes an edit after outcomes are seen visible rather than silent | **Adopted** |
| D-25 | **Scope reduction.** Primary SCAF is currency + rank-normalised relevance under one tunable weight and a threshold; support (σ) and authority (τ) become ablations. Arms reduce to no-filter control / RAG² / SCAF. Contested detection, supersession, the verifier and the clinician rating study become secondary or qualitative | σ needs an entailment model and τ needs a tier ordering the thesis itself calls contestable (D-7); tuning four weights plus θ on a small validation split is not defensible at this sample size. The central contribution — measuring admission recency bias and testing whether temporally aware admission reduces it without degrading evidence quality — is untouched | **Adopted** |
| D-26 | **Terminology fixed: "currency" becomes "recency", "SCAF" becomes "recency-aware admission" (no replacement acronym), and the method is an "admission policy", not a "framework"** | Two of the three currency states are out of scope and "currency" reads as money; SCAF named components that no longer exist; a deterministic scoring rule plus a threshold is not a framework. One term per concept, used everywhere | **Adopted** |
| D-27 | **The admission score is `A(s) = (1 - lambda)*rho(s) + lambda*R(s,q,t_q)`, admit if `A(s) >= theta`** | A single weight removes the redundant degree of freedom that two free weights give (scaling both changes nothing the threshold does not absorb) and makes `lambda = 0` a built-in pure-relevance ablation. Both signals are already in [0,1], so no extra normalisation is added. Tunables reduce to `lambda`, `theta`, `H`, all fitted on validation | **Adopted** |
| D-28 | **The primary recency score is plain age decay only.** Retraction exclusion, supersession discounting and time-invariance are secondary, opt-in, and recorded in run metadata when enabled | The hypothesis is about recency asymmetry. Folding validity rules into the same scalar would mean an observed effect could not be attributed to age. The corpus also has no populated retraction or supersession metadata, so the branches would be inert but misleading | **Adopted** |
| D-29 | **The no-filter control arm is implemented.** It was named in the reduced three-arm design but did not exist in `systems/` | Without it, RAG² and the proposed policy can only be compared to each other, and neither comparison shows whether filtering helps at all. It also rules out "the proposed policy only looks better because it admits more evidence" | **Adopted** |
| D-30 | **`λ = 0` is an internal ablation of the proposed method, not a fourth arm.** The primary experiment stays at three arms | `λ = 0` is not the no-filter control: it still admits at `θ`, so it is relevance-thresholded admission, not unfiltered. As an ablation it isolates what recency adds over thresholding relevance alone; as a fourth arm it would add a second tuning target and a fourth multiple comparison without answering the primary question | **Adopted** |
| D-31 | **The negative control is the unchanged-claim set**, not a date permutation. A free rank-balance diagnostic accompanies it | No filter under test receives a date, so permuting dates changes no filter output and drives Δ to zero by construction rather than by evidence (R-2/R-3). Unchanged-claim pairs share prose era but not claim change, so Δ ≈ 0 there alongside Δ > 0 on changed pairs is what separates recency from era style | **Adopted** |
| D-32 | **One filter backbone is primary; a second is optional robustness.** RQ2/H3 becomes conditional | The central question is whether this confidence-derived signal shows recency asymmetry, which one faithful reproduction answers. A second backbone upgrades the claim to the signal family — a generalisation, not the thesis — and costs a second full filter training, since the RAG² checkpoint is not distributed. If only one is run, the claim is narrowed and the limitation stated | **Adopted** |
| D-33 | **Primary candidate sets contain only dated passages**, applied identically to all three arms at construction | Stage-2 eligibility guarantees the evaluation pair is dated, but corpus-retrieved distractors need not be. Excluding undated passages at construction removes the undated branch of the recency score from the primary experiment entirely, keeping the tunable count at three (λ, θ, H) rather than adding `undated_score` as a fourth. Each run reports its `UNDATED` count; in a valid primary run it is zero | **Adopted** |
| D-34 | **MedChangeQA's released structure is verified, and the two review records are recovered by a deterministic index join over `AllStudyGroups.csv`** | `MedChangeQA.csv` carries only `Question`, `Newest Label`, `Outdated Label` — no PMIDs, no dates, no evidence text — so it cannot on its own satisfy the external input contract. `AllStudyGroups.csv` supplies the linkage: `Group_ID` is sparse and must be forward-filled; `Study_ID` is a 0-based row index into `MedRevQA.csv` (label agreement 4,379/4,379). Groups holding more than one distinct `Label` number exactly 512 and align 1:1 in file order with `MedChangeQA.csv` (`Newest Label` agreement 512/512). Evidence text is each version's `conclusions`, the id is `PMID`, the date comes from `DOI_Date`. No lexical or semantic matching is involved. Question-text matching was tested and fails: only 6 of 512 recover both labels | **Adopted — discharges E15 and the D-20 caveat** |
| D-35 | **The primary pool stays general-medical MedChangeQA in full. An Alzheimer's-restricted primary pool is rejected on measured grounds; the AD items become a descriptive case study** | `MedRevQA` is a complete Cochrane census (16,501/16,501 rows, 2000–2024). Within it only 342 reviews are AD-domain, forming 24 multi-version groups, of which 12 changed verdict; only 9 of the 512 MedChangeQA items are AD-domain under a deliberately generous criterion, and only 2 name Alzheimer's disease. The repository's own `power` module needs 164 total pairs (49 discordant) for 80% power at older-share 0.70; at n = 9 exact power is 0.000 and no effect size is detectable. Because Cochrane is the densest source of externally-documented dated verdict changes in medicine, this is a census ceiling rather than a sampling shortfall, so no AD-specific construction can clear it. Domain mismatch is therefore carried as a stated limitation, which D-2 and D-20 already anticipated | **Adopted** |
| D-36 | **The research question is hallucination rate (primary) and QA accuracy (secondary). Admission asymmetry is superseded and is no longer an outcome** | The thesis now asks whether the proposed solution/system reduces hallucinated answers relative to the baseline under identical question and evidence conditions, while maintaining comparable QA accuracy. The proposed method is unchanged — the same `A(s)` rule serves both framings — so this narrows what is measured, not what is built. D-35 is the proximate cause: the asymmetry design needed matched temporal-counterfactual pairs, and the AD census ceiling (9 usable items, exact power 0.000) put a powered AD-domain asymmetry measurement out of reach, while a hallucination comparison needs only paired questions over one frozen candidate set. `experiments/test_pairs/` and `docs/research_experimental_specification.md` are retained for provenance and produce no outcome. D-32 (backbone replication) lapses with RQ2 | **Adopted — supersedes the RQ1–RQ6 structure** |
| D-37 | **Retrieval and reranking are implemented in-repository as a MedCPT dense retriever plus MedCPT cross-encoder reranker over a deterministic flat exact-search index**, with corpus indexing deferred until Step 1 completes | RAG² fidelity requires MedCPT at both stages (E6). An approximate index (HNSW/IVF) introduces build-order-dependent recall, so a flat exact inner-product search is preferred: the AD corpus is a domain slice, not RAG²'s 116.7M passages, so exact search is affordable and removes a reproducibility hazard for free. Both arms consume the frozen output, so retrieval is upstream and shared by construction | **Adopted** |
| D-38 | **The generator stays Llama-3-8B-Instruct — RAG²'s own — at 4-bit NF4, executed on a free-tier remote T4 (16 GB), under a pinned execution contract** | Local execution is ruled out at every precision, not merely inconvenient: 4-bit weights are ≈4.5–5 GB and ≈6–8 GB with KV cache at RAG-length contexts, against 4 GB VRAM, and CPU-only inference of an 8B model is unusable for a run that must be repeatable. That rules out the *venue*, not the *model*. Since a free T4 fits the model at 4-bit with headroom, changing the model would trade away RAG² fidelity to solve a problem the venue already solves, so it is not changed. Quantisation is the one deviation, and it is a decoding-precision change applied identically to both arms, not a different system. Hugging Face gating is a licence click, recorded as a student action rather than a technical blocker. **Fallback, declared not silent:** if gated access cannot be obtained, Qwen2.5-7B-Instruct (ungated, same parameter scale) substitutes and the substitution is reported as a limitation | **Adopted** |
| D-39 | **The RAG² filter is trained by us: Flan-T5-large, full fine-tune, free-tier remote T4, on labels regenerated by the paper's own decision-tree procedure over a MedQA subsample** | The checkpoint is not distributed (E9). **Verified, not assumed:** the released `classifier/data/medqa/llama3_cot/5%-train.json` was downloaded and inspected — it contains **5 examples**, a format sample, not the ~23.6k-example 5% split its name suggests (ids run to `llama3_5%_23600`). Label regeneration is therefore unavoidable. Since the generator venue (D-38) is a free 16 GB T4, Flan-T5-large — the paper's own size (E1) — is trainable there, so the base model is **not** reduced; batch 16 does not fit at seq 512, so the effective batch is preserved by accumulation (4 × 4) rather than quietly shrunk. Deviations from the recipe: per-device batch, and an epoch count below the paper's 40, both recorded by `FilterTrainingConfig.deviations()`. The remaining cost is label generation itself, which is GPU time on a subsample and is stated, not hidden | **Adopted** |
| D-40 | **The labelled training set is a subsample, and its size is an honest budget, not a target** | The paper's label function needs two rationale generations per (question, passage) pair, so cost scales linearly with set size. A subsample keeps the **label function** faithful — which is what defines the baseline — while making the cost fit a student's free-tier budget. The consequence is a weaker filter than the paper's, which is a limitation on the **baseline's strength** and is reported as one. It is not a threat to the comparison's validity: the same filter defines the baseline for both outcomes, and the proposed system is compared against that baseline, never against the paper's published numbers. The no-filter control is the floor that makes a weak filter visible rather than flattering | **Adopted** |

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
| A1 | Newer evidence in FRB-PAIRS lies **outside** the label model's parametric priors | **Resolved — false, and retired as blocking** | Measured 2026-09-16: change points span 2004–2024, peak 2013–2015; **1 of 512** falls after the Llama-3-8B cutoff. Under the reduced design the primary claim is about the filter's admission behaviour on content, not parametric recall of post-cutoff facts, so A1 is no longer a precondition. Re-scope it to the *generator* analysis, where it remains a real constraint |
| A2 | The permutation control can detect prose-era artefacts | **Blocking** | Analytic — it cannot; redesign per R-2 |
| A3 | 150–300 matched pairs are obtainable from 512 MedChangeQA items | **Resolved — supported** | All 512 items reconstruct with both PMIDs, both dates and both `conclusions` texts via D-34, so attrition before eligibility is 0. Separation is min 1 y, median 12 y, max 23 y, so a `min_separation_days` threshold can be set from evidence (specification 10.3) |
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

**Stage-2 readiness audit (2026-09-15), measured on the repository as it stands.**

| # | Finding | Evidence |
|---|---|---|
| G-S2-1 | **No corpus exists yet.** The corpus tree contains a 10-document fixture set producing 18 chunks. Every metadata CSV (`pubmed`, `pmc`, `guidelines`, `textbooks`) is header-only | `alzheimer_corpus/data/chunks/chunks.jsonl`; `alzheimer_corpus/metadata/*.csv` |
| G-S2-2 | **The external evaluation dataset is absent and unobtainable from this environment.** `huggingface.co` and `eutils.ncbi.nlm.nih.gov` are blocked by organisational egress; `api.github.com` responds, so the block is host-specific | Egress probe, 2026-09-15 |
| G-S2-3 | **No supersession relationships exist.** Zero records, in any file. Chronological difference is not supersession and none were inferred | `alzheimer_corpus/metadata/*.csv` |
| G-S2-4 | **Claim classes are an unvalidated heuristic.** 15 of 18 fixture chunks carry labels from `keyword-from-classname` at confidence 0.0–0.4 | `alzheimer_corpus/data/chunks/chunks.jsonl` |
| G-S2-5 | **Chunks carry no persistent identifier.** PMID/PMCID/DOI are present in the document-level metadata schema but do not reach the chunk level, so pair-level provenance cannot currently cite one | `alzheimer_corpus/data/chunks/chunks.jsonl` |
| G-S2-6 | **Corpus-only material yields zero valid test pairs**, as expected. A 40-candidate pilot over the fixture chunks lost 100% to five simultaneous causes: no question date, no reference answer, unverified contradiction, unverified claim equivalence, insufficient provenance. No single fix recovers any pair (`sole_reason` is empty throughout) | `experiments/outputs/stage2_pilot/secondary_curated_attrition.csv` |
| G-S2-7 | **Check A cannot be evaluated.** It needs change points from the external dataset. The pilot reports `check_a_interpretable: false` rather than a number, because corpus publication dates are not evidence-change dates | `experiments/outputs/stage2_pilot/secondary_curated_manifest.json` |
| G-S2-8 | **Currency-pack retrievability (A4) cannot be evaluated.** It requires a built index and a held-out AD question set; neither exists. `alzheimer_corpus/data/raw/currency_pack/` is empty | Repository inspection |
| G-S2-9 | **Power analysis has no inputs.** It needs the discordant rate and the older-favouring share, both of which are Stage-3 measurements. The sizing function refuses to run without them rather than assuming an effect size | `experiments/test_pairs/scripts/power.py` |

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
| 2026-09-16 | Final design freeze. One genuine fairness defect found and fixed: the no-filter control emitted context in rank-sorted order while the other two arms emitted candidate-list order, so the arms would have diverged on any candidate list not already rank-ordered — which is exactly what Stage-3 pair injection produces. Decisions D-30 to D-33 recorded; `docs/frozen_scope.md` is now the canonical reference. 140 tests pass. Stage 3 may begin. |
| 2026-09-16 | Scope-alignment pass against the base paper and proposal. Verified that the proposal's PRIMARY contribution (C1, the recency-bias probe) is untouched by the reduction and that only the secondary contribution (C2, the admission policy) shrinks. Terminology unified (D-26), score reduced to two components under one weight (D-27), recency score narrowed to plain age decay (D-28), missing no-filter control implemented (D-29). `systems/` renamed and reduced accordingly; 139 tests pass, including the first committed tests for `systems/`. Proposal edits listed in docs/proposal_scope_amendment.md. |
| 2026-09-15 | Consistency audit of d203e28. Four defects corrected: pilot truncation was silently applicable to the primary pool (would have computed Check A on a sample rather than the full set, contradicting D-23); the freeze hash was not reproducible across days because the extraction date was stamped at run time; sample sizing used a normal approximation while the analysis is exact, delivering ~77% power where 80% was claimed, now computed by exact binomial enumeration; one exclusion reason was dead vocabulary. Added an acquisition validator and invariant tests. 103 tests pass. No change to `systems/`. |
| 2026-09-15 | Stage-2 methodological review. D-17 superseded by D-21 after the question-date fallback was found to inflate the currency contrast by construction. Evidence-to-corpus mapping resolved by D-20 (no mapping; finding F4 closed). Claim classes demoted (D-22), sample-size strategy replaced (D-23), Stage-2/3 boundary frozen and hashed (D-24), and a scope reduction adopted (D-25). 83 tests pass. |
| 2026-09-15 | Stage-2 readiness audit run against the actual repository. Verdict: **not ready for pair generation**; findings G-S2-1 to G-S2-9 recorded. Minimal Stage-2 infrastructure added under `experiments/test_pairs/` (schema, eligibility, attrition, split, Check-A stratification, sizing) with 72 tests. Decisions D-15 to D-19 adopted. No test pairs were generated: the external evaluation dataset is the binding dependency. |
| 2026-09-15 | D-14 recorded: `contested_budget_policy = "cap"` fixed as implementation policy. The `"exempt"` alternative removed from `systems/proposed/admission.py`, so no supported path can exceed the context budget. The numerical budget itself (`max_admitted_passages`) remains an unset experimental parameter and is recorded per run as `context_budget.budget_configured`. |
| 2026-09-11 | Ledger created. Initial research understanding built from D1 (proposal) and D2 (RAG², NAACL 2025). Fifteen established facts recorded; thirteen student decisions endorsed; ten recommendations raised; thirteen assumptions and fifteen risks registered. Stage 1 assessed **Not Ready** pending the Checks A–D feasibility audit. |
