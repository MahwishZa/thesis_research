# Proposal Scope Amendment

The approved proposal (`MS Thesis Research Proposal · Measuring Recency Bias in
Evidence Utility Signals`) describes a larger design than the thesis now
executes. This note lists exactly what must change in it, and what must not.

It exists because the proposal is a PDF held outside this repository; the edits
below are for the student to apply to the source document. **Nothing here
changes a result — no results exist yet.**

## The good news first

The proposal names the **Filter Recency-Bias Probe as the primary
contribution (C1)** and **SCAF as the secondary contribution (C2)**. The scope
reduction cuts components of C2. **C1 is untouched and remains fully
executable.** The thesis keeps its primary contribution intact; what shrinks is
the corrective policy, which the proposal itself already framed as secondary.

## Edits required

### 1. Abstract (p. 4)

Currently: *"The secondary contribution is a corrective admission policy, SCAF,
that replaces confidence-derived utility with entailment-derived support, adds
a three-state currency model (current, superseded, contested) and treats source
authority as a tested variable rather than an assumed ordering."*

Every element named in that sentence is out of scope. Replace with:

> The secondary contribution is a corrective admission policy that adds an
> explicit evidence-recency signal to the reranker relevance the baseline
> already uses, and tests whether making evidence age visible to the admission
> step reduces the measured asymmetry without degrading answer quality.

### 2. Contribution C2 (§8, p. 24)

Currently defines SCAF by entailment-derived support, the three-state currency
model with a contested condition, and source authority as a tested variable.
Replace with:

> **C2.** A recency-aware admission policy: a two-signal scoring rule combining
> rank-normalised reranker relevance with an explicit recency score, evaluated
> against both the RAG² baseline and an unfiltered control.

### 3. Contribution C3 (§8, p. 24)

Remove "the supersession table" (not built; no supersession metadata exists).
Soften "FRB-PAIRS with its permutation control" — the permutation control was
shown not to discriminate for the primary comparison (ledger R-2). Released
artefacts are: the test-pair set with its exclusion log and attrition table,
the frozen evaluation specification with its content hash, and the
pre-registered analysis plan.

### 4. Contribution C1 (§8, p. 24)

C1 says "replicated across two backbones and validated by a permutation
control". Two changes:

* **the permutation control cannot validate C1** — neither filter receives a
  date as input, so permuting dates changes no filter output and drives the
  statistic to zero by construction (ledger R-2/R-3). Replace with the
  negative-control condition (claims that did not change across the window).
* **two backbones is a compute decision, not a scientific one.** Keep it as
  stated if the budget allows; if it does not, report one backbone and state
  the single-backbone limitation explicitly rather than quietly dropping it.

### 5. Objectives (§1.5, p. ~10)

Objective 5 currently reads "Specify and implement SCAF: entailment-derived
support, three-state currency, tested authority". Replace with "Specify and
implement a recency-aware admission policy combining reranker relevance with an
explicit recency score."

Objective 6 currently evaluates against "RAG-squared and the current
state-of-the-art medical filtering system". Drop the second comparator to
related-work discussion; reproducing a third system is out of budget.

### 6. Architecture and mathematics (§4.3–4.5, Algorithm 1 "SCAF-ADMIT")

Replace `A(s) = w1·σ(s) + w2·γ(s) + w3·ρ(s) + w4·τ(s)` with:

> `A(s) = (1 − λ)·ρ(s) + λ·R(s, q, t_q)`, admit if `A(s) ≥ θ`

One weight rather than four. A pair of free weights has a redundant degree of
freedom that the threshold absorbs; a single `λ ∈ [0,1]` states exactly how
much of the score is recency, and `λ = 0` is the pure-relevance ablation that
isolates the temporal contribution. Both signals are already in [0,1], so no
extra normalisation is introduced.

Remove σ, τ and the supersession table `T` from the notation table (§4.1).
Keep `t_q`, `ψ(q)` may be dropped or marked secondary.

### 7. Terminology throughout

| Old | New |
|---|---|
| SCAF | recency-aware admission (no acronym) |
| currency score, γ | recency score, R |
| framework | admission policy |
| three-state currency model | *(remove)* |

### 8. Evaluation and clinician study (§7)

The blinded clinician rating study moves to future work: its recruitment lead
time does not fit the schedule. Keep the automatic metrics under the existing
mandatory dual reporting, and keep the judge-validation guard. A small targeted
clinician review of the Alzheimer's case-study items may remain if feasible.

### 9. Output states

Three-state output (GROUNDED / FLAGGED / CONTESTED) reduces to GROUNDED /
ABSTAIN, with CONTESTED produced only under the secondary contested analysis.

## What must NOT change

* **The research question and title.** The reduced design answers the original
  question; it was reduced to make it executable, not to replace it.
* **The background and related work.** Still correct and still motivating.
* **The provenance firewall (§5.2)** and the frozen-candidate-set control —
  these are the thesis's strongest validity guarantees.
* **The pre-declared disappointing outcomes.** Stating in advance that
  saturated benchmarks will barely move, and that automatic hallucination
  metrics are fragile, is good practice and should survive intact.
* **The Alzheimer's framing**, with one clarification: the primary probe uses
  externally-authored general-medical items, and the Alzheimer's corpus is the
  retrieval population plus the case-study domain. The proposal's own firewall
  already implies this (ledger D-20).

## What must not be claimed after the reduction

That the thesis validates contested-evidence handling, source-authority
weighting, entailment-derived support, or supersession as mechanisms; that it
outperforms current state of the art; or that outputs were clinician-rated.
Each becomes a stated limitation with its reason.
