# SCAF extension

**Not started, deliberately.** Placeholder for the proposed scoring function
`A(s) = w1·σ + w2·γ + w3·ρ + w4·τ` (support, currency, corroboration,
authority), with its three-state currency term and abstention/contested gate.

When it is built, it arrives as a new `rag2.filtering.base.EvidenceFilter`
implementation registered under its own key — the same seam the paper's own
"RAG² w/o filter" ablation uses. The baseline filter stays untouched and
selectable, so the two can be compared on one replayed candidate set.

Not implemented anywhere in this repository yet: recency weighting, authority
weighting, currency scoring, supersession, contested-evidence handling,
abstention, temporal counterfactuals. Establish and measure the baseline first.
