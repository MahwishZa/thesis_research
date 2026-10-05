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
* In the realigned study (`experiment_plan.md`) the primary outcome is the same accuracy over all items of the
  held-out split, compared between the proposed R2V and the adapted RAG² baseline R2. The criteria arms (R2C,
  R2V, R2V-ND) answer in three lines and their verdict is parsed from the `FINAL VERDICT:` line by a regular
  expression (`rag2.parse_final_verdict`); a verification output without one keeps the draft verdict and is
  counted as invalid; an R2C output without one is unparsed and counted wrong.

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
| Label reproducibility; stated-verdict consistency | agreement and kappa of an independent model's re-labelling with the gold labels (`label_audit.py`); agreement of an independent judge with the stated verdict on a sample (`consistency_auto.py`) | independent-model audits that replace the human checks |
| Automatic faithfulness proxies | citations point to admitted passages; entailment by a second-family model | optional; outside the primary analysis; the human hallucination annotation is dropped (`evaluation/annotation.py` stays as dormant framework code) |
| Anachronism rate (realigned study) | the answer mentions a year later than the question date's year; no admitted study, all published before the question date, can support it (approximate: a four-digit count is read as a year) | indicator of unsupported, parametric or future knowledge |
| Unsupported decisive verdict (realigned study) | SUPPORTED or REFUTED while citing none of the admitted studies ("[n]" in a standard answer; the DIRECT STUDIES line in the three-line format) | indicator of unsupported answers; questions without admitted evidence are excluded |
| Verifier behaviour (realigned study) | share of valid outputs; share of verdicts changed from R2's; changes that fixed or broke an answer, and their direction | mechanism of the proposed component |
| Retrieval metrics (realigned study) | number admitted and share of questions with none; evidence-type mix (systematic review or meta-analysis / trial / other); update-window share; mean age; overlap with B1; optional directness@k (share of admitted abstracts an independent second-family model judges to test the question's intervention and outcome) | retrieval evaluated separately from generation; descriptive |

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
* Optional secondary analyses, outside the confirmatory family: a mixed-effects logistic model `correct ~ helpfulness × recency + (1 | item)`, the
  changed-vs-unchanged interaction, one-sided non-inferiority on unchanged items, and P vs the
  shuffled-date control C1.

A difference is read as real only if the pre-declared test says so at α = 0.05 after correction; an
average gap alone is not sufficient. If about 30% of answers differ between arms (an assumption until
dev results exist), 353 confirmatory changed items give 84% power at the corrected α for a true 10 pp
difference and 61% for 8 pp (simulated in the stage-1 protocol); smaller effects are reported as
inconclusive with their intervals, not as absence of effect. For stage 2 (528 items; a hybrid that differs from
the RAG answer it refines on 15–25% of items) a true +3 pp is detected 26–38% of the time, +4 pp 42–62% and
+5 pp 61–82%: the benchmark can confirm only effects of about 5 pp or more.

* **Realigned study** (`experiment_plan.md` §7): one primary test, R2V − R2 over all held-out items (exact
  McNemar and a paired bootstrap 95% interval); a secondary family Holm-corrected among themselves (R2 − B1,
  R2V − B1, R2C − R2, R2V − R2C, R2V − R2V-ND, R2V − R2 on changed items); an ablation family on dev only
  (R2 − R2-RQ, R2 − R2-BR, R2 − R2-NF). The +1 pp requirement is read in advance: *met and confirmed*
  (difference ≥ 1.0 pp, p < .05, interval above 0), *met as a point estimate, not confirmed* (≥ 1.0 pp
  otherwise) or *not met*. With 528 questions and 10–25% of them answered correctly by only one of the two
  systems, the standard error of the difference is 1.4–2.2 pp: an observed +1 pp arises by chance alone
  23–32% of the time, and confirming a real +1 pp would need about 3,800–9,600 questions (computed).

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
* **Realigned study.** R2C (the criteria without a draft) separates what the evidence criteria achieve from
  what verifying a draft adds; R2V-ND (no dates, no currency clause) isolates the temporal criterion; R2-RQ,
  R2-BR and R2-NF remove one RAG² component each; B1 shows whether the adapted RAG² retrieval beats standard
  retrieval; the label-stable subset checks that a difference is not carried by unreliable labels.

## 5. Abstention and coverage

An arm that admits nothing still produces an answer and is scored on it. Rates are always reported
with coverage (`stats.coverage`, `stats.har`) so that a high threshold cannot manufacture a good-looking
result by answering fewer questions. For verdict accuracy the analogue is the share of NOT ENOUGH INFORMATION
answers, reported beside accuracy: on this benchmark abstaining is a legitimate answer (31% of gold verdicts),
and a method that merely says it more often is shown as such by the per-class recall.

## 6. Error analysis

Each wrong answer on a changed item is assigned one cause — retrieval miss, admission miss, generator
override, parse failure, or gold-label error — and reported by change type, update-window length and arm.
Stage 1, dev, changed items (`error_analysis.py`; `log.md` Phase 26):
no retrieval misses for any arm; nearly all wrong answers are "evidence admitted, still wrong". Stage 2 adds
per-class recall, macro-F1 and predicted-class shares (`analyze_stage2.py`); confusion matrices,
stance accuracy on decisive flips and the conflict feature against gold NOT ENOUGH INFORMATION are optional additions outside the primary analysis.
Gold-label error is not checked.

## 7. Development-split results (exploratory)

All numbers are from the 226 dev items; none is a confirmatory result.

| Item | Result |
|---|---|
| Gate G0 (evidence headroom) | passed (94.0% of changed dev items) |
| B0 vs B1 (151 changed + 75 unchanged items) | accuracy on changed items 41.1% (B0) vs 49.7% (B1), difference +8.6 pp, 95% CI [0.7, 16.6], exact McNemar p = 0.060; unchanged 53.3% vs 60.0%, p = 0.42; 0 unparsed answers; gate G2 passed (B1 changed 35.8% of verdicts) |
| B2, B3, P, C1 (changed items, accuracy; difference vs B1 with exact McNemar p) | B2 37.1% (−12.6 pp, p = 0.001), B3 45.7% (−4.0 pp, p = 0.38), P 35.8% (−13.9 pp, p = 0.0008), C1 37.7%; P − B2 = −1.3 pp, P − C1 = −2.0 pp, P − B3 = −9.9 pp (p = 0.017); unchanged items 58.7–61.3% for all arms, no significant differences; 0 unparsed |
| Gate G3 | **failed** (needs P − B2 ≥ +5 pp and P − C1 ≥ +2.5 pp; observed −1.3 and −2.0) |
| Error analysis, stage 1, changed items | no retrieval misses for any arm; nearly all wrong answers are "evidence admitted, still wrong" (`log.md` Phase 26) |
| Gate G1 (format validity) | parse rate 100% on dev (0 of 1,356 unparsed); an independent judge agrees with the stated verdict on 99.0% of the sample (`consistency_auto.py`, diagnostic) |
| Stage 2 pilot (gate 1, machine checks, 40 dev items, 960 papers) | **PASS**: wording agreement 85.3%, control papers "neither" 97.8%, invalid 0.16% on real papers (2.5% pooled with control papers), 6.76 s per paper |
| Stage 2 P0 diagnostics | 18.5% of helpfulness inputs over 512 tokens (31.1% of B2's admitted papers); 36.4% of candidates have labelled RESULTS/CONCLUSIONS; a systematic review in the top 8 for 41.6% of items |
| Stage 2 dev fit (gate 2) | **FAIL**: stance-direction AUC 0.624 (passes ≥ 0.60), but the selected hybrid H0 reaches 51.8% against 52.2% for B1R (needs +1.0 pp) and S0 reaches 43.6% against a constant-guess 46.5%; the confirmatory run therefore tests RQ1 only (`results/DEV_REPORT.md`) |
| Label audit (independent model, dev) | agreement 83.2%, kappa 0.7422; 188 of 226 items label-stable |
| Result tables and figures in the base paper's layout | `results/report/REPORT.md` (dev split, exploratory) |

The only earlier real-data outputs (the Alzheimer's pilot with an extractive stand-in generator and an
unvalidated baseline checkpoint) are archived in `_archive/alzheimers_pilot_v1/results/`; they showed the
pipeline ran, not that anything improved.

## 8. Confirmatory results (528 items, run once)

The confirmatory split was run once, after the stage-2 model was frozen, with the analysis rules of
§9–§10 of the stage-2 protocol (commit 92e3aaf). Gate 2 had failed on dev, so only RQ1 was tested and RQ2 was not run
(`results/FINDINGS.md`, `results/stage2_analysis_confirm.md`, `results/report/REPORT.md`).

| Item | Result |
|---|---|
| Probe (gate G0) | passed: 80.2% of changed items have a trial or review in the update window within the top 50 candidates (93.8% within the top 200); no item has zero hits |
| Primary family: RQ1, B1 vs B0, all 528 items | B1 48.1% (95% CI 43.9–52.4) vs B0 46.6% (42.4–50.9); difference **+1.5 pp**, 95% CI −2.8 to +6.1, 74 / 66 discordant items, exact McNemar p = 0.554, Holm p = 0.554: **not confirmed** |
| Secondary: RQ1 on the 353 changed items | B1 45.0% vs B0 41.9%; **+3.1 pp**, 95% CI −2.5 to +8.5, 55 / 44 discordant items, p = 0.315: **not confirmed** |
| Unchanged items (175) | B1 54.3% vs B0 56.0% (−1.7 pp) |
| Outdated-verdict rate (changed items) | B0 32.6%, B1 32.9%: no change |
| Where the answers move (all items; recall SUPPORTED / REFUTED / NOT ENOUGH INFORMATION) | B0 85.0% / 3.2% / 25.6%; B1 70.1% / 8.7% / 47.0%; macro-F1 32.9% → 39.5% (descriptive, outside the confirmatory family). Retrieval mainly moves answers from SUPPORTED to NOT ENOUGH INFORMATION |
| Pre-declared reading | RQ1 **not confirmed**; RQ2 **not run** |
| Closed-book models from the benchmark release (existing work, same questions) | 50.8–52.8% on all items, against 48.1% for B1; the intervals overlap widely |
| Label reproducibility (independent model) | agreement 81.4%, kappa 0.7161; 430 of 528 items label-stable; agreement per gold class: SUPPORTED 86.8%, REFUTED 91.3%, NOT ENOUGH INFORMATION 66.7%; 72.5% of the 353 label changes between versions reproduced |
| Stated-verdict consistency | parse rate 100.0%; the independent judge agrees with the stated verdict on 97.7% of 399 sampled answers (diagnostic) |

**Reading.** The point estimate of retrieval over no retrieval is positive but small, and the interval includes zero,
so the confirmatory data cannot distinguish it from no effect. They also do not rule out a benefit of up to about
6 pp (the benchmark can confirm only effects of about 5 pp or more, §3). The +8.6 pp seen on the dev changed items
(§7) did not replicate on the larger split; the dev value is the one chosen for attention after seeing it and
should not be quoted as the effect. Neither extension tested on dev (recency-aware admission, evidence synthesis)
beat standard retrieval, so no confirmatory claim is made for them.
