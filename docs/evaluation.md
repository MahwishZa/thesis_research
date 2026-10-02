# Evaluation and Experimental Design

What is measured and how a difference is judged real. For what is compared see `methodology.md`; for
the full protocol, gates and decision rules see `experiment_plan.md`; for how to run an evaluation
see `reproducibility.md`.

## 1. Primary outcome: verdict accuracy

The question each system answers is whether a medical hypothesis is SUPPORTED, REFUTED or has NOT
ENOUGH INFORMATION. **Verdict accuracy** is the share of items whose parsed verdict equals the gold
verdict of the newest Cochrane review version (`experiments/medchange/analyze.py`).

* The generator must open with `VERDICT: <label>`; the verdict is parsed from that line by a regular
  expression, never inferred from prose. An answer with no parsable verdict counts as wrong and is
  counted separately (`unparsed`). There is no judge model on the primary path.
* It is independent of the mechanism under test: the Temporal Filter changes which passages are
  admitted, while the outcome compares the answer with a gold label that no admission rule sees.

## 2. Supporting outcomes

| Outcome | Definition | Role |
|---|---|---|
| Outdated-verdict rate | verdict equals the previous version's verdict (changed items) | key secondary: direct measure of outdated answers |
| Accuracy on unchanged items | as the primary, on controls whose verdict never changed | safety: non-inferiority margin 5 pp |
| Update-window share | share of admitted passages that surely first appeared after the previous version and on or before the newest | retrieval-level manipulation check |
| Items with update-window evidence | share of items where any admitted passage is in the window | retrieval-level manipulation check |
| Mean admitted-passage age; overlap with B1 | years from passage to t_q; Jaccard of admitted sets | retrieval-level manipulation checks |
| Blinded human hallucination rate | claims unsupported by or contradicting a common reference, on a stratified subset | **planned**; `evaluation/annotation.py`, `stats.har`, `stats.coverage` |

**Manipulation checks are never outcomes.** Showing that an arm admits more recent passages
demonstrates that the mechanism acts as designed; it says nothing about whether answers improve.
Earlier in the project the primary metric was *currency* (the mean temporal score of admitted
evidence) and the fitting objective was the same quantity, so the proposed system would have won it
by construction; that computation survives only in the archived v1 runner
(`_archive/alzheimers_pilot_v1/`). The active retrieval-level metrics above replace it. Token F1,
exact match, ROUGE-L, context precision/recall and token-overlap groundedness are implemented in
`evaluation/rag_metrics.py` as standard RAG diagnostics for the framework's fixture runs; they are not
evidence of improvement here (a single verbatim reference sentence is a weak target, and groundedness
depends on each arm's own admitted evidence).

## 3. Statistical procedure (`evaluation/stats.py`, standard library only)

* **Exact McNemar test** on paired discordant verdict outcomes (two-sided exact binomial).
* **Paired bootstrap 95% CI** on the difference, resampling items as units.
* **Holm correction** across the confirmatory family: P vs B1, P vs B2, P vs B3.
* Planned: a mixed-effects logistic model `correct ~ helpfulness × recency + (1 | item)`, the
  changed-vs-unchanged interaction, one-sided non-inferiority on unchanged items, and P vs the
  shuffled-date control C1.

A difference is read as real only if the pre-declared test says so at α = 0.05 after correction; an
average gap alone is not sufficient. If about 30% of answers differ between arms (an assumption until
dev results exist), 353 confirmatory changed items give 84% power at the corrected α for a true 10 pp
difference and 61% for 8 pp (simulated; `experiment_plan.md` §7); smaller effects are reported as
inconclusive with their intervals, not as absence of effect.

## 4. Controls

* **B0** (no evidence) shows whether retrieval helps or hurts at all.
* **C1** (dates shuffled within each pool) tests specificity: a gain that survives shuffling is not
  temporal.
* **Unchanged items** test that recency does not hurt where nothing changed.
* **Generator-sensitivity gate G2** (B1 must change ≥ 20% of dev verdicts relative to B0) tests
  whether the generator uses evidence at all; if not, no admission rule can matter.

## 5. Abstention and coverage

An arm that admits nothing still produces an answer and is scored on it. Rates are always reported
with coverage (`stats.coverage`, `stats.har`) so that a high threshold cannot manufacture a good-looking
result by answering fewer questions.

## 6. Error analysis

Each wrong answer on a changed item is assigned one cause — retrieval miss, admission miss, generator
override, parse failure, or gold-label error — and reported by change type, update-window length and arm
(**planned**; `experiment_plan.md` §8).

## 7. Evaluation status (2026-10-02)

| Component | Status |
|---|---|
| Benchmark, dev candidate pools, helpfulness scores, arms, generation harness, analysis script | built; 6 real generations run for timing (n = 3 items, no accuracy conclusion) |
| Gate G0 (evidence headroom) | passed (94.0% of changed dev items) |
| Gates G1, G2, G3; any accuracy result | pending: no accuracy result exists yet |
| Confirmatory split (pools, helpfulness, generation) | not started |
| Human hallucination annotation; error analysis; second generator; Alzheimer's case study | planned |

The only earlier real-data outputs (the Alzheimer's pilot with an extractive stand-in generator and an
unvalidated baseline checkpoint) are archived in `_archive/alzheimers_pilot_v1/results/`; they showed the
pipeline ran, not that anything improved.
