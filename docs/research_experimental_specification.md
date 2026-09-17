# Research and Experimental Specification

## MS Thesis Research Project

**Thesis Title:** `[TITLE TO BE SPECIFIED]`

**Research Domain:** Alzheimer's disease and related dementias

**Research Pipeline:**

1. Build Alzheimer's corpus
2. Build test pairs
3. Run the bias probe
4. Build SCAF
5. Compare all variants
6. Analyze and write up

**Scope of this specification:** Steps 2–5

**Specification Version:** `[VERSION TO BE SPECIFIED]`

**Date:** `[DATE TO BE SPECIFIED]`

---

# 0. Scope Amendment (2026-09-16) — read this first

> The canonical, frozen statement of scope is `docs/frozen_scope.md`.
> This section summarises it; where the two differ, that file governs.

This specification was written for a larger design than the thesis now
executes. The reduced scope below **governs wherever the body of this document
disagrees with it**; the body is retained because its definitions, controls and
reproducibility requirements remain correct for the parts still in scope.

## 0.1 The primary experiment

Three arms over one frozen candidate set: a **no-filter control**, the **RAG²
baseline**, and the proposed **recency-aware admission policy**.

The proposed policy is:

    A(s) = (1 - lambda) * rho(s) + lambda * R(s, q, t_q),  admit if A(s) >= theta

where `rho` is the rank-normalised reranker score (section 28) and `R` is the
recency score — plain exponential decay in the age of the passage relative to
`t_q`. `lambda`, `theta` and the half-life `H` are the only tunable
quantities and are fitted on the validation split.

## 0.2 Removed from the primary scope

The following are no longer primary components. Sections 20-24, 27, 29, 30 and
46 describe them and are retained as secondary or future work:

* entailment-derived support, sigma (sections 20-22, 24);
* source authority, tau (section 27);
* contested-evidence handling (sections 29, 30);
* supersession discounting (section 25, secondary branch);
* answer verification (section 31 verifier);
* the clinician rating study (section 46);
* comparison against an additional state-of-the-art filtering system.

Code for contested detection and verification remains in `systems/proposed/`,
marked SECONDARY and disabled by default.

## 0.3 Terminology

| Old | Current | Why |
|---|---|---|
| currency score, gamma | **recency score, R** | Two of the three "currency" states are out of scope, and "currency" reads as money |
| SCAF | **recency-aware admission** | The acronym named components that no longer exist; no replacement acronym is introduced |
| framework | **admission policy** | It is a deterministic scoring rule plus a threshold |

Output states reduce to **GROUNDED** and **ABSTAIN**; CONTESTED is produced
only when the secondary contested detector is explicitly enabled.

## 0.4 What is unchanged

The research question, the provenance firewall, the frozen-candidate-set
control (section 16), the temporal definitions (sections 9, 10), the
leakage rules (section 33), dual reporting (section 42) and the
reproducibility requirements (section 50) all stand as written.

---

# 1. Purpose

This document defines the research and experimental specification governing Steps 2–5 of the thesis research pipeline.

Its purpose is to establish:

* the research objective and questions;
* the operational definition of the phenomenon under investigation;
* the construction and representation of evaluation data;
* the experimental controls and comparison conditions;
* the separation of retrieval, admission, and generation effects;
* the development, validation, and test methodology;
* the proposed SCAF admission policy;
* the evaluation framework;
* the statistical analysis framework;
* the reproducibility and provenance requirements.

The specification is intended to serve as the methodological contract for subsequent implementation.

Where a methodological value, threshold, dataset source, sampling rule, or implementation choice has not yet been established, it is represented explicitly as:

> `[TO BE SPECIFIED]`

Such placeholders must be resolved and recorded before the corresponding implementation stage is frozen.

---

# 2. Research Problem

Retrieval-augmented generation can provide external evidence to language models whose parametric knowledge may be incomplete or outdated. However, retrieved evidence is not necessarily used symmetrically. A filtering or admission mechanism may preferentially retain evidence that is compatible with the model's existing knowledge while rejecting evidence that conflicts with those priors.

This thesis investigates whether such an asymmetry can occur with respect to **evidence age** at the evidence-admission stage of a medical RAG pipeline.

The research focuses specifically on the distinction between:

1. **retrieval**, which determines which evidence is available;
2. **admission/filtering**, which determines which retrieved evidence is retained;
3. **generation**, which produces the final answer from the retained evidence.

The primary research problem is therefore:

> Whether a confidence-derived evidence-utility signal used for passage admission systematically favours older evidence over newer evidence when the underlying clinical claim has changed, after controlling the upstream retrieval process and relevant characteristics of the paired evidence.

The thesis subsequently investigates whether an admission policy based on evidential support and temporal information can reduce such undesirable asymmetry while preserving performance on claims for which recency is not relevant.

---

# 3. Research Objective

## 3.1 Aim

The research aim is:

> To determine whether confidence-derived evidence utility signals exhibit measurable recency-related admission asymmetry in medical retrieval-augmented generation, and to develop and evaluate a corrective admission policy for Alzheimer's clinical reasoning.

## 3.2 Objectives

The research objectives are:

1. Reproduce the baseline RAG² system sufficiently to establish a controlled experimental baseline.
2. Construct a date-annotated Alzheimer's retrieval corpus.
3. Construct controlled temporal-counterfactual test pairs representing changes in clinical evidence.
4. Measure admission asymmetry between older and newer evidence.
5. Determine whether any observed asymmetry is replicated across filter backbones.
6. Compare the baseline confidence-derived labelling mechanism with an entailment-derived alternative.
7. Develop SCAF as an admission policy incorporating evidential support, temporal currency, source authority, disagreement handling, and abstention.
8. Compare SCAF with the baseline and specified experimental controls under a frozen retrieval environment.
9. Evaluate whether SCAF reduces unsupported or superseded claims without unacceptable loss of accuracy or coverage.
10. Characterise remaining failure modes through structured error analysis and, where applicable, blinded expert evaluation.

The proposal identifies filter-stage admission asymmetry as the primary research gap and positions SCAF as the corrective component following measurement of the phenomenon.

---

# 4. Research Questions and Hypotheses

## 4.1 Research Questions

| ID  | Research Question                                                                                                                                                       |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RQ1 | Do confidence-derived utility signals admit passages asymmetrically with respect to evidence age when relevant content, source tier, and passage length are controlled? |
| RQ2 | Does any observed admission asymmetry replicate across filter backbones?                                                                                                |
| RQ3 | Does an entailment-derived evidence label reduce unsupported and superseded claims relative to the confidence-derived baseline?                                         |
| RQ4 | Are observed improvements attributable to the admission policy rather than differences in retrieval quality?                                                            |
| RQ5 | Does explicit representation of contested evidence improve clinical appropriateness on unresolved questions?                                                            |
| RQ6 | Can improved evidence selection be achieved without materially degrading performance on time-invariant medical questions?                                               |

These questions are derived from the established research proposal.

## 4.2 Hypotheses

| ID | Hypothesis                                                                                                                                              |
| -- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H1 | The baseline confidence-derived filter exhibits positive admission asymmetry, such that older evidence is admitted more frequently than newer evidence. |
| H2 | Admission asymmetry under the confidence-derived filter is greater than admission asymmetry under the entailment-derived filter.                        |
| H3 | The direction of the observed asymmetry replicates across filter backbones.                                                                             |
| H4 | `[VALIDITY-CONTROL HYPOTHESIS TO BE SPECIFIED AFTER FINAL CONTROL DESIGN]`                                                                              |
| H5 | The entailment-derived admission policy reduces the rate of unsupported claims and/or superseded claims relative to the baseline.                       |
| H6 | The effect of the admission policy persists after accounting for retrieval recall.                                                                      |
| H7 | Performance on time-invariant medical questions remains non-inferior to the baseline under a pre-specified margin.                                      |

The non-inferiority margin for H7 is:

**δ = `[TO BE SPECIFIED]`**

The statistical criterion and justification for δ must be established before final test evaluation.

---

# 5. Scope and Delimitations

## 5.1 Included

The research includes:

* Alzheimer's disease and related dementias;
* English-language public medical evidence;
* evidence admission within a single-pass RAG pipeline;
* filter models within the specified computational limit;
* open-weight generator models within the specified computational limit;
* the baseline RAG² admission mechanism;
* an entailment-derived admission alternative;
* SCAF;
* controlled temporal evaluation;
* structured clinical reasoning evaluation;
* evidence-support, currency, and safety-related outcomes.

The proposal specifies a maximum filter size of one billion parameters and open-weight generators of at most eight billion parameters, with an optional commercial comparison.

## 5.2 Excluded

The following are outside the primary research scope:

* retriever optimisation;
* replacement of the frozen retrieval system;
* agentic retrieval as a primary competitor;
* reinforcement-learned retrieval;
* private clinical data;
* claims of clinical validation.

The expert evaluation is intended to assess mechanism and clinical appropriateness at limited scale, not to constitute clinical validation.

---

# 6. Conceptual Experimental Model

The experimental pipeline is:

```text
Clinical Question + As-of Date
              │
              ▼
      Rationale Generation
              │
              ▼
       Frozen Retrieval
              │
              ▼
        Reranking
              │
              ▼
    Cached Candidate Set
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
 Baseline Filter   Proposed Filter
       │             │
       ▼             ▼
  Admitted Evidence
       │             │
       └──────┬──────┘
              ▼
       Frozen Generation
              │
              ▼
        Verification
              │
              ▼
        Final Output
```

All components upstream of the admission policy must remain identical across controlled experimental arms.

This is the principal internal-validity requirement of the study. The proposal explicitly defines isolation of one variable and byte-identical candidate replay as central design principles.

---

# 7. Role of the Alzheimer's Corpus

The Alzheimer's corpus serves as the common evidence environment for the experimental pipeline.

Its functions are:

1. provide the retrieval population;
2. provide temporal metadata;
3. provide source-tier information;
4. provide guideline/version information;
5. provide supersession and retraction metadata;
6. support controlled evidence selection;
7. support evaluation of evidence currency;
8. provide a reproducible retrieval substrate shared across experimental arms.

The corpus is not itself the experimental treatment.

The corpus must therefore remain identical across all primary experimental conditions.

---

# 8. Corpus and Evidence Representation

Each passage must have a persistent provenance record.

At minimum, passage metadata shall include:

| Field                 | Requirement               |
| --------------------- | ------------------------- |
| Passage ID            | Required                  |
| Document ID           | Required                  |
| Persistent identifier | Required where available  |
| Source tier           | Required                  |
| Publication date      | Required where available  |
| Date precision        | Required                  |
| Journal/source        | Required where applicable |
| Guideline family      | Required where applicable |
| Guideline version     | Required where applicable |
| Supersession pointer  | Required where applicable |
| Retraction status     | Required                  |
| Withdrawal status     | Required                  |
| Claim class           | Required                  |
| Section               | Required                  |
| Character span        | Required                  |
| Corpus snapshot       | Required                  |

The proposal establishes this metadata structure as part of the corpus specification.

---

# 9. Temporal Representation

Temporal information shall be represented independently of whether it is exposed to a particular experimental model.

Each relevant passage shall preserve:

* publication date;
* date precision;
* date provenance;
* guideline/version date where applicable;
* supersession relationship;
* retraction/withdrawal status;
* temporal position within an evaluation pair.

For a query, the system may additionally have an **as-of date**:

$$
t_q
$$

representing the temporal point at which the question is evaluated.

The exact representation of incomplete publication dates is:

**`[TO BE SPECIFIED]`**

The treatment of uncertain or interval-valued dates is:

**`[TO BE SPECIFIED]`**

---

# 10. Test-Pair Definition

## 10.1 Temporal-Counterfactual Pair

A temporal-counterfactual pair consists of two passages:

* an **older passage** representing the pre-change evidence state; and
* a **newer passage** representing the post-change evidence state,

where both passages address the same underlying clinical claim and occur on opposite sides of an externally established change point.

The pair is intended to vary evidence age while controlling other relevant factors.

The proposal specifies matching on:

1. underlying claim;
2. source tier;
3. passage length within a tolerance band;
4. topical similarity to the query.

## 10.2 Pair Relationship

The two passages must not merely discuss the same general topic.

They must represent the **same evaluative claim or clinical proposition**, such that the temporal difference can be meaningfully interpreted as an evidence update.

The operational criterion for "same claim" is:

**`[TO BE SPECIFIED]`**

The annotation or verification procedure for establishing claim equivalence is:

**`[TO BE SPECIFIED]`**

## 10.3 Change Point

Each pair must have an externally defensible change point:

**Change-point source:** MedChangeQA (Vladika et al., Findings of EMNLP 2025) — 512 changed-verdict items derived from MedRevQA, itself built from systematic-review abstracts indexed in PubMed 2000–January 2024. This records the dataset the proposal already names; it is stated here because this section previously left the primary instrument's source unspecified.

**Change-point definition:** the transition between the superseded verdict and the current verdict on the same review question, as established by the dataset. The thesis does not judge which verdict is correct, and a change point the dataset does not supply is left absent rather than substituted with a publication date.

**Required temporal separation:** `[TO BE SPECIFIED AFTER PILOT]` — to be read off the observed separation distribution rather than assumed. Until it is set, the rule is recorded as not enforced.

---

# 11. Pair Categories

The evaluation data shall distinguish the following evidence relationships.

## 11.1 Changed / Superseded

The newer evidence materially changes the applicable clinical conclusion or recommendation.

These pairs constitute the principal temporal-counterfactual condition.

## 11.2 Unchanged

The underlying claim remains materially stable across the relevant temporal interval.

These items constitute the principal negative-control condition.

## 11.3 Contested

Credible evidence supports materially opposing positions within a relevant temporal window, without a justified basis for treating one position as an unambiguous supersession of the other.

The contested state must be evaluated before supersession classification.

This ordering is explicitly established in the proposal.

## 11.4 Non-comparable

Two passages must be classified as non-comparable when their apparent disagreement results from differences in:

* population;
* disease stage;
* intervention;
* outcome;
* eligibility;
* clinical setting;
* scope;
* or another substantive contextual condition.

For example, differing conclusions for early-stage and moderate disease are not automatically contradictory.

The exact operational criteria are:

**`[TO BE SPECIFIED]`**

---

# 12. Meaningful Evidence Change

A meaningful change is a change that materially alters the answer or clinical recommendation associated with the underlying claim.

Candidate categories include:

* changed recommendation;
* changed diagnostic criterion;
* changed eligibility;
* changed contraindication;
* changed treatment conclusion;
* changed monitoring requirement;
* materially changed risk interpretation;
* materially changed evidence-supported conclusion.

The final claim-change taxonomy shall be aligned with:

**`[STAGE-1 CLAIM TAXONOMY / VERSION TO BE SPECIFIED]`**

Changes that are purely stylistic, terminological, or bibliographic shall not be treated as substantive evidence changes unless they alter the clinical proposition.

---

# 13. Definition of Admission Bias

For a filter \(f\), define admission asymmetry as:

$$
\Delta_f =
E[
P(\mathrm{admit}\mid\mathrm{older})
-
P(\mathrm{admit}\mid\mathrm{newer})
]
$$

where the expectation is taken over valid matched test pairs.

Interpretation:

| Value                  | Interpretation                              |
| ---------------------- | ------------------------------------------- |
| \(\Delta_f > 0\)       | Older evidence is preferentially admitted   |
| \(\Delta_f \approx 0\) | No directional admission asymmetry detected |
| \(\Delta_f < 0\)       | Newer evidence is preferentially admitted   |

The thesis does **not** define legitimate preference for newer evidence as bias.

A newer passage may legitimately be more useful when it reflects a genuine evidence update.

The research question is whether evidence age is associated with admission differences **beyond the substantive evidence change and controlled characteristics of the pair**.

---

# 14. Distinguishing Recency Bias from Legitimate Evidence Updating

The experimental design shall distinguish:

1. legitimate preference caused by changed evidence;
2. source-authority differences;
3. retrieval differences;
4. passage-level linguistic differences;
5. temporal/prose-era effects;
6. genuine admission asymmetry associated with evidence age.

Matching alone cannot establish that age is the sole remaining difference.

Therefore the study shall use negative controls and additional validity controls.

The primary negative control shall consist of claims that did not materially change across the study window.

If a comparable age preference is observed for unchanged claims, the interpretation of the changed-claim effect as recency-specific bias must be reconsidered.

The exact formal criterion for comparing changed and unchanged effects is:

**`[TO BE SPECIFIED]`**

---

# 15. Retrieval, Admission, and Generation Effects

The experiment shall treat the RAG pipeline as three analytically distinct stages.

## 15.1 Retrieval Effect

A retrieval effect occurs when relevant evidence is or is not included in the candidate set.

Primary measures include:

* Recall@k;
* precision;
* MRR;
* nDCG;
* evidence coverage.

## 15.2 Admission Effect

An admission effect occurs when the same candidate set is available but different filtering policies select different evidence.

This is the primary target of the bias probe.

## 15.3 Generation Effect

A generation effect occurs when the same or equivalent admitted evidence produces different final outputs.

Generation is therefore downstream of admission.

---

# 16. Candidate-Set Control

Retrieval and reranking must be performed once for each evaluation item.

The resulting candidate set must be serialised and replayed byte-identically across experimental arms.

The following must therefore be invariant across controlled arms:

* query;
* rationale;
* retrieval index;
* retrieval model;
* retrieved candidates;
* candidate order;
* reranker;
* reranker scores;
* candidate-set size;
* context budget.

This control is essential because otherwise differences in final output cannot be attributed specifically to admission policy. The proposal identifies cached candidate replay as a load-bearing validity control.

---

# 17. Baseline System

The primary baseline is the reproduced **RAG²** system.

The original RAG² framework contains:

1. rationale-based query formulation;
2. balanced retrieval;
3. MedCPT reranking;
4. rationale-guided passage filtering;
5. final answer generation.

The original paper reports rationale-based queries rather than concatenating the original question with the rationale, and uses balanced retrieval across multiple biomedical corpora followed by MedCPT reranking.

The thesis baseline shall reproduce the relevant admission mechanism rather than introducing a modified baseline.

The exact reproduced RAG² implementation shall be:

**`[IMPLEMENTATION VERSION / COMMIT TO BE SPECIFIED]`**

---

# 18. Baseline Admission Label

The baseline filter shall reproduce the RAG² label-generation procedure.

The proposal describes the baseline as:

1. determine whether evidence changes the model from an incorrect to a correct answer;
2. resolve ambiguous cases using the perplexity differential;
3. retain the relevant upper portion according to the original threshold;
4. discard unresolved cases.

The exact operational implementation shall follow the verified RAG² source.

Because the original paper and the thesis proposal do not fully establish every implementation detail required for exact reproduction, the following must be recorded before final Stage-3 execution:

* exact correctness-flip procedure;
* perplexity calculation target;
* perplexity equation;
* threshold;
* tie-handling;
* discard rule;
* label distribution;
* training-data construction.

**RAG² reproduction specification:** `[TO BE SPECIFIED AFTER SOURCE VERIFICATION]`

---

# 19. Experimental Filter Comparison

The primary filter comparison consists of two controlled conditions.

| Condition   | Label Function                           | Backbone | Training Data | Hyperparameters |
| ----------- | ---------------------------------------- | -------- | ------------- | --------------- |
| Baseline    | RAG² confidence/perplexity-derived label | Same     | Same          | Same            |
| Alternative | Entailment-derived support label         | Same     | Same          | Same            |

The purpose of this comparison is to isolate the effect of the **label/signal formulation**.

The two filters must therefore have:

* the same backbone;
* the same parameter count;
* the same training data;
* the same training procedure;
* the same hyperparameter policy;
* the same inference conditions.

Only the label function should differ.

This controlled comparison is explicitly established in the proposal.

---

# 20. Entailment-Derived Label

During training, the gold answer may be verbalised as a hypothesis:

$$
h^* = \mathrm{verbalise}(Q,\mathrm{gold})
$$

The support score is:

$$
\sigma^* =
P_{\mathrm{entail}}(
premise=s,
hypothesis=h^*
)
$$

The proposed binary label is:

$$
y(s)=
\begin{cases}
1 & \sigma^* \geq \theta_{hi}\\
0 & \sigma^* \leq \theta_{lo}\\
\mathrm{discard} & \text{otherwise}
\end{cases}
$$

The exact values of:

$$
\theta_{hi},\theta_{lo}
$$

are:

**`[TO BE SPECIFIED]`**

Both filters are intended to be trained on general medical QA and applied zero-shot to Alzheimer's evaluation data.

---

# 21. Inference-Time Support

At inference time, the gold answer is unavailable.

The support score shall therefore be derived from hypotheses constructed from the available query/rationale.

For multiple-choice questions:

* hypotheses may correspond to verbalised answer options.

For open-ended questions:

* hypotheses may correspond to atomic claims derived from the rationale.

The maximum entailment probability is used as the support score:

$$
\sigma(s)=
\max_h P_{\mathrm{entail}}(s,h)
$$

The maximum number of atomic hypotheses is currently specified as eight.

The exact decomposition procedure is:

**`[TO BE SPECIFIED]`**

---

# 22. Discriminativeness

The difference between the highest and second-highest entailment scores may provide information about whether a passage supports one answer specifically or several alternatives equally.

The discriminativeness margin shall be recorded as a diagnostic:

$$
D(s)=\sigma_{(1)}(s)-\sigma_{(2)}(s)
$$

Whether \(D(s)\) contributes directly to admission is:

**`[TO BE SPECIFIED]`**

This value must not be incorporated into SCAF without an explicit experimental decision.

---

# 23. SCAF

SCAF is the proposed corrective admission policy.

Its purpose is to replace confidence-derived utility with an admission score incorporating:

1. evidential support;
2. temporal currency;
3. retrieval/reranking position;
4. source authority.

The proposed admission score is:

$$
A(s)=
w_1\sigma(s)
+w_2\gamma(s,q,t_q)
+w_3\rho(s)
+w_4\tau(s)
$$

A passage is admitted when:

$$
A(s)\geq\theta_{\mathrm{admit}}
$$

The weights and threshold are:

$$
w_1,w_2,w_3,w_4,\theta_{\mathrm{admit}}
=
[\text{TO BE SPECIFIED}]
$$

---

# 24. SCAF Support Component

The support component is:

$$
\sigma(s)
$$

and is derived from entailment rather than model confidence.

The support model is:

**`[MODEL TO BE SPECIFIED]`**

The support threshold is:

$$
\theta_{\mathrm{support}}
=
[\text{TO BE SPECIFIED}]
$$

---

# 25. SCAF Currency Component

The currency component is:

$$
\gamma(s,q,t_q)
$$

The proposed formulation distinguishes:

1. retracted/withdrawn evidence;
2. time-invariant questions;
3. superseded evidence;
4. current evidence.

The proposal gives the following conceptual form:

$$
\gamma(s,q,t_q)=
\begin{cases}
0 & \text{if retracted or withdrawn}\\
1 & \text{if }\psi(q)=0\\
\delta 2^{-(t_q-date(s))/H}
& \text{if superseded}\\
2^{-(t_q-date(s))/H}
& \text{otherwise}
\end{cases}
$$

The following parameters remain:

* half-life \(H\);
* supersession discount \(\delta\);
* temporal-sensitivity function \(\psi(q)\).

Their final definitions are:

**`[TO BE SPECIFIED]`**

Superseded evidence shall be down-weighted rather than automatically deleted.

Hard exclusion shall be restricted to objectively established retraction or withdrawal states unless a different rule is explicitly approved.

---

# 26. Temporal Sensitivity

The currency mechanism shall be conditioned on whether the question is temporally sensitive.

This prevents a general recency preference from unnecessarily penalising older evidence for time-invariant claims.

The implementation of:

$$
\psi(q)
$$

is:

**`[TO BE SPECIFIED]`**

Possible approaches include:

* rule-based classification using claim classes;
* a trained classifier;
* another explicitly justified method.

The selected method must be frozen before final evaluation.

---

# 27. Source Authority

Source authority is represented as:

$$
\tau(s)
$$

The thesis does not assume a universal fixed ordering of source types.

Instead, source authority shall be evaluated as an experimental variable.

The authority representation is:

**`[TO BE SPECIFIED]`**

The source-authority ablation shall compare:

* authority included;
* authority excluded;
* alternative authority formulation where justified.

---

# 28. Rank-Normalised Relevance

The reranker contribution is:

$$
\rho(s)
$$

Rank-based normalisation shall be used rather than min-max normalisation.

This ensures comparability across queries and supports the use of a global admission threshold.

The exact rank-normalisation function is:

**`[TO BE SPECIFIED]`**

---

# 29. Contested Evidence

SCAF shall explicitly represent unresolved evidential disagreement.

A claim is considered contested when:

1. relevant passages belong to the same claim class;
2. the passages support opposing conclusions;
3. the passages fall within the defined contest window;
4. both satisfy the minimum source-quality requirement.

The contest window is:

**`[TO BE SPECIFIED]`**

The minimum source tier is:

**`[TO BE SPECIFIED]`**

The contested state is evaluated before supersession.

When a contested condition is triggered, the system shall preserve representative evidence for both positions and identify the disagreement in the output.

Preservation operates within the common context budget of section 16, which is never exceeded. Where contested evidence exceeds that budget, contested evidence is retained ahead of uncontested evidence, in descending A(s) with `evidence_id` as tie-break, up to the budget; the contested positions preserved and those dropped are recorded in the run metadata. This clarification resolves the conflict between this section and section 16 and does not alter either requirement in the non-exceeding case.

---

# 30. Non-Comparability and Contradiction

The contested detector must distinguish substantive contradiction from differences caused by scope.

The following are not automatically contradictions:

* different populations;
* different disease stages;
* different interventions;
* different outcomes;
* different clinical settings;
* conditional recommendations.

The formal contradiction criterion is:

**`[TO BE SPECIFIED]`**

Because the proposal identifies contradiction-versus-scope detection as a technical limitation, contested-state results shall be interpreted as evidence about the mechanism unless the validation procedure establishes sufficient reliability.

---

# 31. SCAF Output States

The proposed output policy contains four states:

| State     | Meaning                                                              |
| --------- | -------------------------------------------------------------------- |
| GROUNDED  | Adequately supported and sufficiently current evidence is available  |
| FLAGGED   | Evidence is usable but contains relevant currency/staleness concerns |
| CONTESTED | Credible evidence supports opposing positions                        |
| ABSTAIN   | Available evidence is insufficient for a supported answer            |

The exact state-transition thresholds are:

**`[TO BE SPECIFIED]`**

---

# 32. Development, Validation, and Test Data

The research shall use independent development, validation, and final-test partitions.

## Development

Used for:

* implementation;
* debugging;
* preliminary analysis;
* engineering decisions.

## Validation

Used for:

* parameter selection;
* threshold selection;
* model selection;
* SCAF tuning;
* classifier tuning;
* sensitivity analysis.

## Final Test

Used only after all relevant decisions have been frozen.

The final test set must not be used for:

* parameter tuning;
* threshold selection;
* model selection;
* pair-selection decisions;
* hypothesis modification;
* post-hoc control selection.

The exact partitioning scheme is:

**`[TO BE SPECIFIED]`**

---

# 33. Data Leakage Prevention

The following rules are mandatory.

### 33.1 Evaluation isolation

Final test questions and pairs must not be used for training or tuning.

### 33.2 Provenance isolation

The source of evaluation passages must be documented independently of the experimental result.

### 33.3 Answer isolation

Gold answers may be used where explicitly required for training-label construction but must not be available to inference-time admission.

### 33.4 Metadata isolation

Temporal metadata must not enter date-blind model conditions.

### 33.5 Retrieval isolation

All primary experimental arms must receive the same cached candidate set.

### 33.6 Configuration isolation

Once final test evaluation begins, model, prompt, retrieval, filter, SCAF, and evaluation configurations must be frozen.

---

# 34. Test-Pair Sampling

The final sampling procedure shall be designed to preserve the distribution of relevant evidence characteristics.

At minimum, sampling shall consider:

* claim class;
* source tier;
* temporal interval;
* type of evidence change;
* pre-training cutoff relationship;
* question type;
* clinical topic.

The exact sampling proportions are:

**`[TO BE SPECIFIED]`**

The random seed is:

**`[TO BE SPECIFIED]`**

The target number of valid FRB pairs is:

**`[TO BE SPECIFIED AFTER PILOT AND POWER ANALYSIS]`**

The target sizes for negative-control, contested, and other evaluation sets are:

**`[TO BE SPECIFIED]`**

---

# 35. Pair Attrition and Eligibility

A candidate pair may be excluded because of:

* inability to verify claim equivalence;
* unavailable provenance;
* insufficient temporal evidence;
* source-tier mismatch;
* excessive length mismatch;
* insufficient topical similarity;
* non-comparability;
* uncertain change status;
* leakage;
* duplicate content;
* inadequate metadata.

Every exclusion must have a recorded reason.

The final pair-selection criteria are:

**`[TO BE SPECIFIED]`**

---

# 36. Pre-Training Cutoff Analysis

The temporal-counterfactual design depends partly on the relationship between evidence dates and the knowledge available to the model used to generate training labels.

Each pair shall therefore retain:

* older evidence date;
* newer evidence date;
* change-point date;
* model identity;
* relevant model knowledge/pre-training cutoff information where available.

The final cutoff-stratification rule is:

**`[TO BE SPECIFIED]`**

The number and proportion of usable pairs in each temporal stratum shall be reported before the final test set is frozen.

---

# 37. Primary Bias-Probe Experiment

The primary experiment compares:

```text
Matched Older Passage ──┐
                        ├── Baseline Filter ── Δbaseline
Matched Newer Passage ──┘

Matched Older Passage ──┐
                        ├── Entailment Filter ── Δsupport
Matched Newer Passage ──┘
```

The candidate evidence, query, retrieval, reranking, model backbone, and experimental conditions are held constant except for the filter label function.

The primary comparison is:

$$
\Delta_{\mathrm{baseline}}
\quad\text{vs.}\quad
\Delta_{\mathrm{support}}
$$

---

# 38. Validity Controls

The following controls shall be included or explicitly resolved before final experimentation.

| Control                   | Purpose                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| Candidate replay          | Isolate admission from retrieval                                   |
| Backbone replication      | Test whether the effect generalises beyond one filter model        |
| Negative-control set      | Distinguish claim-recency effects from general era/prose effects   |
| Decoupled verifier        | Reduce correlated evaluation errors                                |
| Date-manipulation control | `[FINAL DESIGN TO BE SPECIFIED]`                                   |
| Date-annotated condition  | Evaluate effects of explicit temporal information where applicable |

The original proposal specifies a permutation control in which publication dates are reassigned. Its precise use in a date-blind condition requires methodological resolution because the filter cannot respond to dates it does not receive. Therefore the final control design is:

**`[TO BE SPECIFIED]`**

---

# 39. Experimental Variants

The complete experimental comparison shall include the following conceptual arms.

| ID | Variant                                    | Purpose                          |
| -- | ------------------------------------------ | -------------------------------- |
| B0 | Closed-book baseline                       | Establish no-retrieval reference |
| B1 | Retrieval + reranking without filtering    | Measure retrieval contribution   |
| B2 | Original RAG²                              | Primary baseline                 |
| B3 | Contemporary support-supervised comparator | External comparison              |
| B4 | Agentic/reference system                   | Contextual comparison            |
| P  | SCAF                                       | Proposed system                  |

The exact implementation of B3 and B4 is:

**`[TO BE SPECIFIED]`**

The proposal explicitly includes these comparative conditions.

---

# 40. Ablation Plan

The following ablations are specified conceptually.

| ID  | Ablation                              | Purpose                              |
| --- | ------------------------------------- | ------------------------------------ |
| A1  | Perplexity label vs. entailment label | Primary causal comparison            |
| A2  | Date/control condition                | Validity                             |
| A3  | Second filter backbone                | Replication                          |
| A4  | No filter                             | Determine filtering contribution     |
| A5  | Support without currency              | Measure currency contribution        |
| A6  | Date-blind vs. date-annotated prompt  | Separate temporal cue effects        |
| A7  | Currency without support              | Measure support contribution         |
| A8  | Contested state removed               | Measure disagreement handling        |
| A9  | Abstention removed                    | Measure abstention contribution      |
| A10 | Teacher vs. distilled student         | Measure distillation trade-off       |
| A11 | Parameter sensitivity                 | Assess robustness                    |
| A12 | Authority variants                    | Assess authority/recency interaction |
| A13 | Hard supersession                     | Assess soft-supersession design      |
| A14 | Generator transfer                    | Assess generality                    |

The exact subset designated as primary versus secondary is:

**`[TO BE SPECIFIED]`**

---

# 41. Evaluation Metrics

## 41.1 Primary Bias Metric

$$
\Delta =
E[
P(\mathrm{admit}\mid older)
-
P(\mathrm{admit}\mid newer)
]
$$

with:

* point estimate;
* 95% confidence interval;
* pair-level analysis.

## 41.2 Admission Metrics

* older admission rate;
* newer admission rate;
* pairwise admission difference;
* proportion of pairs favouring older evidence;
* proportion favouring newer evidence;
* tie rate.

## 41.3 Retrieval Metrics

* Recall@k;
* Precision@k;
* MRR;
* nDCG@10;
* evidence coverage;
* redundancy.

## 41.4 Evidence and Claim Metrics

* unsupported-claim rate;
* contradicted-claim rate;
* superseded-claim rate;
* citation precision;
* citation recall;
* entity-attribution error;
* evidence insufficiency.

## 41.5 Currency Metrics

* outdated-evidence rate;
* temporal correctness;
* superseded-answer rate;
* guideline-version correctness;
* update-flip rate.

## 41.6 Safety and Abstention Metrics

* abstention rate;
* appropriate abstention;
* over-abstention;
* risk–coverage area;
* coverage at fixed accuracy.

## 41.7 Efficiency Metrics

* latency;
* computational cost;
* cost per correct answer.

These metric categories are established in the proposal's evaluation framework.

---

# 42. Dual Reporting of Answer-Conditional Metrics

Any metric affected by abstention must be reported in two forms:

1. conditional on the system providing an answer;
2. with abstentions counted as failures.

This prevents increased abstention from artificially improving conditional performance.

This requirement is part of the established evaluation design.

---

# 43. Statistical Analysis

## 43.1 Admission Asymmetry

Primary analysis:

* paired bootstrap;
* 10,000 resamples;
* matched pair as unit of analysis;
* 95% confidence interval.

## 43.2 Claim-Level Outcomes

For repeated claims nested within questions:

* mixed-effects logistic regression;
* arm as a fixed effect;
* question as a random effect;
* claim-level clustering where appropriate.

## 43.3 Retrieval–Admission Decoupling

The relationship between retrieval and unsupported claims shall be examined using:

* item-level association between retrieval-recall change and unsupported-claim change;
* mixed-effects mediation where appropriate;
* analysis of items where admission differs but retrieval recall remains unchanged.

## 43.4 Paired Currency Outcomes

For paired categorical outcomes:

* McNemar's exact test;
* effect size using Cohen's \(h\).

## 43.5 Expert Ratings

For ordinal expert ratings:

* mixed-effects cumulative-link model;
* effect-size analysis using Cliff's delta where appropriate.

These statistical structures are specified in the proposal.

---

# 44. Multiple Comparisons

The primary hypothesis family shall use:

* family-wise significance level: \(\alpha = 0.05\);
* Holm–Bonferroni correction.

Secondary/exploratory analyses shall be clearly identified as such.

Any alternative multiplicity policy is:

**`[TO BE SPECIFIED BEFORE FINAL ANALYSIS]`**

---

# 45. Evaluation of Automatic Judges

Automatic claim labels shall be validated against human annotations on a stratified sample.

The validation procedure shall report:

* sample size;
* sampling method;
* agreement statistic;
* validation precision/recall where appropriate.

The proposal specifies Cohen's kappa as the agreement measure and proposes a minimum kappa criterion of 0.6 for retaining automatic claim labels as primary evaluation measures.

The final validation sample size is:

**`[TO BE SPECIFIED]`**

---

# 46. Clinical Evaluation

Where expert evaluation is conducted, outputs shall be:

* anonymised with respect to system arm;
* presented in randomised order;
* evaluated under a pre-generated assignment key;
* assessed using predefined rating criteria.

The proposed rating dimensions include:

* factual correctness;
* evidence support;
* currency;
* clinical appropriateness;
* safety;
* completeness.

Additional binary safety/appropriateness items may be included according to the final evaluation instrument.

The exact evaluator composition is:

**`[TO BE SPECIFIED]`**

The final rating instrument is:

**`[TO BE SPECIFIED]`**

---

# 47. Inter-Rater Reliability

Inter-rater reliability shall be assessed before interpreting between-arm expert-rating differences.

The proposed statistic is:

**Krippendorff's alpha**

The minimum reliability threshold is:

**0.4**

Dimensions falling below the threshold shall be reported descriptively unless an alternative criterion is specified in advance.

This follows the evaluation framework established in the proposal.

---

# 48. Error Taxonomy

Every relevant failure should be assigned to a predefined error category where sufficient logged evidence exists.

The proposed categories are:

1. retrieval failure;
2. reranking failure;
3. filter over-rejection;
4. filter under-rejection;
5. temporal failure;
6. evidence insufficiency;
7. hallucination;
8. contradiction;
9. clinical reasoning error;
10. abstention error.

The taxonomy is intended to distinguish failures occurring at different stages of the RAG pipeline.

---

# 49. Success Criteria

## 49.1 Bias-Probe Success

The primary bias finding is considered supported only if:

1. the primary admission asymmetry is measurable;
2. the pair construction satisfies the predefined eligibility criteria;
3. the effect meets the pre-specified statistical criterion;
4. the negative-control result supports interpretation as a claim-recency effect rather than a generic era effect;
5. the result is not attributable to retrieval differences;
6. the result is sufficiently replicated across the specified backbone conditions.

The precise minimum effect of interest is:

**`[TO BE SPECIFIED]`**

## 49.2 SCAF Success

SCAF is considered successful if it demonstrates the pre-specified improvement in relevant evidence-selection outcomes while satisfying the pre-specified constraints on:

* accuracy;
* over-rejection;
* over-abstention;
* evidence insufficiency;
* safety.

The exact success thresholds are:

**`[TO BE SPECIFIED]`**

## 49.3 Null Result

A null result shall be considered scientifically valid if:

* the study is adequately powered;
* the evaluation set satisfies the predefined criteria;
* controls function as expected;
* the experimental conditions were executed as specified.

A null result shall not be treated as experimental failure.

---

# 50. Reproducibility Requirements

Every experimental result must be traceable to:

1. a corpus snapshot;
2. a pair/test-set manifest;
3. a model version;
4. a filter version;
5. a SCAF configuration;
6. a prompt version;
7. a retrieval configuration;
8. a random seed where applicable;
9. a candidate-set snapshot;
10. an analysis configuration.

The following artefacts shall be versioned:

### Research

* research specification;
* research ledger;
* hypotheses;
* preregistration;
* deviation record.

### Data

* corpus manifest;
* pair manifest;
* train/validation/test manifests;
* metadata;
* exclusion log.

### Models

* model identifiers;
* checkpoint identifiers;
* tokenizer versions;
* training configurations;
* filter checkpoints;
* SCAF configuration.

### Experiments

* prompts;
* decoding parameters;
* retrieval parameters;
* candidate sets;
* admission decisions;
* generated outputs;
* verification outputs;
* evaluation labels;
* statistical-analysis scripts/configuration.

---

# 51. Stage Interfaces

## 51.1 Stage 1 → Stage 2

Stage 1 shall provide:

* frozen corpus snapshot;
* document identifiers;
* passage/chunk identifiers;
* publication dates;
* date provenance;
* source tiers;
* guideline/version metadata;
* supersession information;
* retraction/withdrawal status;
* claim classes;
* provenance information;
* corpus manifest.

## 51.2 Stage 2 → Stage 3

Stage 2 shall provide:

* final temporal-counterfactual pairs;
* negative-control set;
* contested set;
* additional evaluation sets;
* pair provenance;
* matching metadata;
* change-point metadata;
* train/validation/test assignments;
* sampling information;
* exclusion log;
* leakage audit.

## 51.3 Stage 3 → Stage 4

Stage 3 shall provide:

* baseline admission asymmetry;
* alternative-filter admission asymmetry;
* confidence intervals;
* control results;
* backbone replication results;
* temporal-stratum results;
* identified failure modes;
* evidence supporting or failing to support the proposed mechanism.

SCAF implementation must remain conditional on the scientific interpretation of the Stage-3 findings.

## 51.4 Stage 4 → Stage 5

Stage 4 shall provide:

* frozen SCAF implementation;
* frozen parameters;
* model checkpoints;
* support model;
* currency configuration;
* authority configuration;
* contested-state configuration;
* abstention configuration;
* verifier;
* validation results;
* final configuration manifest.

---

# 52. Frozen Components

The following components shall be frozen for the primary comparative experiments.

## Retrieval

* corpus snapshot;
* index;
* retriever;
* retrieval parameters;
* candidate-set size;
* balanced retrieval policy;
* reranker;
* reranking procedure.

## Generation

* generator model;
* prompt;
* decoding configuration;
* temperature;
* maximum output configuration.

## Evaluation

* final test set;
* metric definitions;
* annotation protocol;
* statistical analysis plan.

## Baseline

* verified RAG² implementation;
* baseline filter architecture;
* baseline training procedure;
* baseline label-generation procedure.

---

# 53. Tunable Components

The following may be tuned only using development/validation data:

* filter training hyperparameters;
* entailment thresholds;
* SCAF weights;
* admission threshold;
* currency half-life;
* supersession discount;
* temporal-sensitivity parameters;
* authority parameters;
* contested-state thresholds;
* abstention threshold;
* verifier thresholds.

The final configuration must be frozen before final test execution.

---

# 54. Pre-Registration Requirements

Before final test evaluation, the following shall be pre-registered:

* research questions;
* hypotheses;
* primary outcomes;
* secondary outcomes;
* pair-selection criteria;
* sampling procedure;
* train/validation/test split;
* statistical tests;
* significance level;
* multiplicity correction;
* non-inferiority margin;
* minimum effect of interest;
* control definitions;
* SCAF tuning procedure;
* pivot criteria;
* exclusion rules.

The proposal explicitly establishes pre-registration before contact with final test data.

---

# 55. Methodological Decision Register

The following items remain deliberately unspecified until sufficient justification is available.

| Decision                          | Specification                                  |
| --------------------------------- | ---------------------------------------------- |
| Thesis title                      | `[TO BE SPECIFIED]`                            |
| Pair passage provenance           | `[TO BE SPECIFIED]`                            |
| Same-claim criterion              | `[TO BE SPECIFIED]`                            |
| Change-point definition           | `[TO BE SPECIFIED]`                            |
| Length tolerance                  | `[TO BE SPECIFIED]`                            |
| Topical-similarity criterion      | `[TO BE SPECIFIED]`                            |
| Date precision handling           | `[TO BE SPECIFIED]`                            |
| Non-comparability rule            | `[TO BE SPECIFIED]`                            |
| Contradiction criterion           | `[TO BE SPECIFIED]`                            |
| Pre-training cutoff rule          | `[TO BE SPECIFIED]`                            |
| Final pair sample size            | `[TO BE SPECIFIED AFTER PILOT/POWER ANALYSIS]` |
| Sampling proportions              | `[TO BE SPECIFIED]`                            |
| Development/validation/test split | `[TO BE SPECIFIED]`                            |
| RAG² reproduction details         | `[TO BE SPECIFIED AFTER SOURCE VERIFICATION]`  |
| RAG² perplexity target            | `[TO BE SPECIFIED]`                            |
| Entailment model                  | `[TO BE SPECIFIED]`                            |
| Entailment thresholds             | `[TO BE SPECIFIED]`                            |
| Discriminativeness usage          | `[TO BE SPECIFIED]`                            |
| Currency half-life                | `[TO BE SPECIFIED]`                            |
| Supersession discount             | `[TO BE SPECIFIED]`                            |
| Temporal-sensitivity function     | `[TO BE SPECIFIED]`                            |
| Source-authority function         | `[TO BE SPECIFIED]`                            |
| Contested window                  | `[TO BE SPECIFIED]`                            |
| Contested threshold               | `[TO BE SPECIFIED]`                            |
| SCAF weights                      | `[TO BE SPECIFIED]`                            |
| SCAF admission threshold          | `[TO BE SPECIFIED]`                            |
| Abstention threshold              | `[TO BE SPECIFIED]`                            |
| Non-inferiority margin δ          | `[TO BE SPECIFIED]`                            |
| Judge-validation sample size      | `[TO BE SPECIFIED]`                            |
| Expert-evaluation protocol        | `[TO BE SPECIFIED]`                            |
| Final validity-control design     | `[TO BE SPECIFIED]`                            |

---

# 56. Methodological Principles

The following principles govern implementation of the specification.

### Principle 1 — Isolate the variable

Where an experiment is intended to compare admission policies, all upstream components must remain fixed.

### Principle 2 — Measure before intervening

The existence and characteristics of the proposed bias must be measured before the corrective policy is interpreted as necessary.

### Principle 3 — Preserve the distinction between retrieval and admission

A change in retrieved evidence is not evidence of an admission-policy effect.

### Principle 4 — Do not equate recency with correctness

Newer evidence may legitimately differ from older evidence because the underlying evidence has changed.

### Principle 5 — Do not equate disagreement with supersession

A live scientific dispute must not automatically be resolved in favour of the newest source.

### Principle 6 — Protect the final test set

No final-test observation may influence model, threshold, pair, or hypothesis selection.

### Principle 7 — Prefer explicit uncertainty to invented precision

Any methodological value not yet justified shall remain a documented placeholder until specified.

### Principle 8 — Design for an informative null

The methodology must remain scientifically interpretable if the predicted bias is not detected.

---

# 57. Final Experimental Sequence

The intended sequence is:

```text
Stage 1
Frozen Alzheimer's Corpus
        │
        ▼
Stage 2
Pilot Pair Construction
        │
        ├── Eligibility / Attrition Analysis
        ├── Temporal Feasibility
        ├── Pre-training-Cutoff Analysis
        └── Power Analysis
                │
                ▼
        Final Pair Construction
                │
                ▼
Stage 3
Filter Recency-Bias Probe
        │
        ├── Baseline Filter
        ├── Entailment Filter
        ├── Backbone Replication
        └── Validity Controls
                │
                ▼
Stage 4
SCAF Development and Validation
        │
        ├── Support
        ├── Currency
        ├── Authority
        ├── Contested State
        └── Abstention
                │
                ▼
        Freeze SCAF
                │
                ▼
Stage 5
Final Comparative Evaluation
        │
        ├── Baseline
        ├── Retrieval-only
        ├── RAG²
        ├── Contemporary Comparator
        ├── SCAF
        └── Ablations
                │
                ▼
Stage 6
Statistical Analysis
Error Analysis
Clinical Evaluation
Thesis Write-up
```

---

# 58. Specification Freeze Conditions

Before implementation of each stage begins, the following conditions must be satisfied.

## Step 2 may begin when:

* pair definition is specified;
* provenance is specified;
* matching criteria are specified;
* change-point definition is specified;
* sampling procedure is specified;
* leakage rules are specified.

## Step 3 may begin when:

* the final or pilot pair methodology is frozen;
* baseline RAG² implementation is verified;
* filter-label definitions are frozen;
* control design is frozen;
* candidate replay is operational;
* statistical analysis is specified.

## Step 4 may begin when:

* Stage-3 findings are available;
* SCAF components are formally defined;
* tuning data are separated from test data;
* tunable parameters are identified;
* validation procedure is frozen.

## Step 5 may begin when:

* SCAF is frozen;
* final test sets are locked;
* all model and retrieval configurations are frozen;
* evaluation metrics are frozen;
* statistical analysis is frozen;
* non-inferiority margin is specified;
* preregistration is complete.

---

# 59. Specification Status

This document defines the methodological framework for Steps 2–5.

It intentionally does not assign values to parameters, thresholds, sample sizes, or procedures that have not yet been established by the research record.

All `[TO BE SPECIFIED]` fields represent genuine methodological decisions that must be resolved before the corresponding component is frozen.

No placeholder shall be populated solely for implementation convenience.

The final experimental implementation shall conform to this specification unless a formally recorded methodological amendment is made before the affected experiment is conducted.
