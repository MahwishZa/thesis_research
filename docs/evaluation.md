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
* In stage 2 the primary outcome is accuracy over **all confirmatory items** (353 changed + 175 unchanged),
  chosen before any confirmatory data existed because pooling raises power; accuracy on the changed items alone
  is the key secondary. For the hybrid arms the verdict is the layer's prediction, not generated text.

## 2. Supporting outcomes

| Outcome | Definition | Role |
|---|---|---|
| Outdated-verdict rate | verdict equals the previous version's verdict (changed items) | key secondary: direct measure of outdated answers |
| Accuracy on unchanged items | as the primary, on controls whose verdict never changed | safety: non-inferiority margin 5 pp |
| Update-window share | share of admitted passages that surely first appeared after the previous version and on or before the newest | retrieval-level manipulation check |
| Items with update-window evidence | share of items where any admitted passage is in the window | retrieval-level manipulation check |
| Mean admitted-passage age; overlap with B1 | years from passage to t_q; Jaccard of admitted sets | retrieval-level manipulation checks |
| Recall of each gold class; macro-F1; share of NOT ENOUGH INFORMATION answers | per-class recall over all items, mean F1 of the three classes, how often an arm abstains | stage 2: the main behaviour retrieval changes (SUPPORTED recall falls, abstention rises) and the one a layer can correct |
| Wording agreement; irrelevant-paper control; invalid-output rate; stance-direction AUC against the gold labels (dev) | agreement of the two wordings; share of control papers rated "neither"; share of unusable outputs on real papers; how well the signed stance separates gold SUPPORTED from REFUTED | stage-2 pilot (gate 1), `stance_check.py` |
| Label reproducibility; stated-verdict consistency | agreement and kappa of an independent model's re-labelling with the gold labels (`label_audit.py`); agreement of an independent judge with the stated verdict on a sample (`consistency_auto.py`) | **built**, not yet run; replace the human checks |
| Automatic faithfulness proxies | citations point to admitted passages; entailment by a second-family model | optional, not built; the human hallucination annotation is dropped (`evaluation/annotation.py` stays as dormant framework code) |

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
* **Holm correction.** Stage 1 (dev, exploratory): P vs B1, P vs B2, P vs B3. Stage 2 (confirmatory, run once): the
  primary family RQ1 (B1 vs B0) and RQ2 (the selected hybrid vs B1R) at family-wise α = .05, and a secondary
  family (selected vs B1; vs H0; vs its date-shuffled control; S0 vs B1; changed-items-only RQ1 and RQ2)
  corrected among themselves (`analyze_stage2.py`). Single-arm accuracies carry Wilson 95% intervals.
* **Selection and fitting on dev only.** The hybrid is selected by the pre-stated rule on repeated 5-fold
  cross-validation (50 repeats, seed 20261003); dev comparisons of the stage-2 arms use out-of-fold predictions
  and are exploratory.
* Planned: a mixed-effects logistic model `correct ~ helpfulness × recency + (1 | item)`, the
  changed-vs-unchanged interaction, one-sided non-inferiority on unchanged items, and P vs the
  shuffled-date control C1.

A difference is read as real only if the pre-declared test says so at α = 0.05 after correction; an
average gap alone is not sufficient. If about 30% of answers differ between arms (an assumption until
dev results exist), 353 confirmatory changed items give 84% power at the corrected α for a true 10 pp
difference and 61% for 8 pp (simulated; `experiment_plan.md` §7); smaller effects are reported as
inconclusive with their intervals, not as absence of effect. For stage 2 (528 items; a hybrid that differs from
the RAG answer it refines on 15–25% of items) a true +3 pp is detected 26–38% of the time, +4 pp 42–62% and
+5 pp 61–82%: the benchmark can confirm only effects of about 5 pp or more.

## 4. Controls

* **B0** (no evidence) shows whether retrieval helps or hurts at all.
* **C1** (dates shuffled within each pool) tests specificity: a gain that survives shuffling is not
  temporal.
* **Unchanged items** test that recency does not hurt where nothing changed.
* **B1R** gives the same fitting to the RAG answer alone, so a hybrid's gain over it cannot come from fitting
  a decision rule on dev. **H1C / H3C** shuffle publication dates, so a gain from recency weights must beat them.
  **S0** against **S1–S3** and **H0** against **H1–H3** isolate the weights. The **irrelevant-paper control**
  (each question judged against papers of another item) checks that the stance step reads the paper.
* **Generator-sensitivity gate G2** (B1 must change ≥ 20% of dev verdicts relative to B0) tests
  whether the generator uses evidence at all; if not, no admission rule can matter.

## 5. Abstention and coverage

An arm that admits nothing still produces an answer and is scored on it. Rates are always reported
with coverage (`stats.coverage`, `stats.har`) so that a high threshold cannot manufacture a good-looking
result by answering fewer questions. For verdict accuracy the analogue is the share of NOT ENOUGH INFORMATION
answers, reported beside accuracy: on this benchmark abstaining is a legitimate answer (31% of gold verdicts),
and a method that merely says it more often is shown as such by the per-class recall.

## 6. Error analysis

Each wrong answer on a changed item is assigned one cause — retrieval miss, admission miss, generator
override, parse failure, or gold-label error — and reported by change type, update-window length and arm.
Stage 1, dev, changed items: **done** (`error_analysis.py`; `experiment_plan.md` §8, `log.md` Phase 26):
no retrieval misses for any arm; nearly all wrong answers are "evidence admitted, still wrong". Stage 2 adds
per-class recall, macro-F1 and predicted-class shares (**built**, `analyze_stage2.py`), and confusion matrices,
stance accuracy on decisive flips and the conflict feature against gold NOT ENOUGH INFORMATION (**planned**).
Gold-label error is not checked.

## 7. Evaluation status (2026-10-03)

| Component | Status |
|---|---|
| Benchmark, dev candidate pools, helpfulness scores, six stage-1 arms, generation harness, analysis script | built; dev run of all six arms done |
| Gate G0 (evidence headroom) | passed (94.0% of changed dev items) |
| Dev run of B0 and B1 (151 changed + 75 unchanged items) | done: accuracy on changed items 41.1% (B0) vs 49.7% (B1), difference +8.6 pp, 95% CI [0.7, 16.6], exact McNemar p = 0.060; unchanged 53.3% vs 60.0%, p = 0.42; 0 unparsed answers; gate G2 passed (B1 changed 35.8% of verdicts) |
| Dev run of B2, B3, P, C1 (changed items, accuracy; difference vs B1 with exact McNemar p) | B2 37.1% (−12.6 pp, p = 0.001), B3 45.7% (−4.0 pp, p = 0.38), P 35.8% (−13.9 pp, p = 0.0008), C1 37.7%; P − B2 = −1.3 pp, P − C1 = −2.0 pp, P − B3 = −9.9 pp (p = 0.017); unchanged items 58.7–61.3% for all arms, no significant differences; 0 unparsed |
| Gate G3 | **failed** (needs P − B2 ≥ +5 pp and P − C1 ≥ +2.5 pp; observed −1.3 and −2.0) |
| Error analysis, stage 1, dev changed items | done (`log.md` Phase 26) |
| Gate G1 (format validity) | parse rate 100% on dev (0 of 1,356 unparsed); the human consistency sheet is retired and replaced by the automatic check (`consistency_auto.py`, built, not yet run) |
| Stage 2: stance, pilot checks, diagnostics, synthesis layer, confirmatory analysis, label audit, consistency check, pipeline | code built and unit-tested (synthetic data) |
| Stage 2 pilot (gate 1, machine checks, 40 dev items, 960 papers) | **PASS**: wording agreement 85.3%, control papers "neither" 97.8%, invalid 0.16% on real papers (2.5% pooled with control papers), 6.76 s per paper |
| Stage 2: P0 diagnostics | done (2026-10-03): 18.5% of helpfulness inputs over 512 tokens (31.1% of B2's admitted papers); 36.4% of candidates have labelled RESULTS/CONCLUSIONS; a systematic review in the top 8 for 41.6% of items |
| Stage 2: full dev stance (both wordings) + fit (gate 2) | pending (`pipeline dev`) |
| Confirmatory split (pools, B0/B1 answers, stance, frozen-model prediction) | not started; nothing from it has been analysed |
| Second generator; automatic Alzheimer's case study | planned |

The only earlier real-data outputs (the Alzheimer's pilot with an extractive stand-in generator and an
unvalidated baseline checkpoint) are archived in `_archive/alzheimers_pilot_v1/results/`; they showed the
pipeline ran, not that anything improved.
