# Evaluation and experiment plan: adapted RAG² with evidence-criteria verification

This document is the protocol and the evaluation of the realigned study in one place: what is compared, what is
measured, how a difference is judged real, the rules fixed before any result existed, and the results of record.
The protocol was fixed on 2026-10-05, before any output of the systems below existed. It replaces the earlier
protocol of stage 1 (recency-aware admission) and stage 2 (an evidence-synthesis layer); that text is kept in the
repository history (`git show 92e3aaf:docs/experiment_plan.md`) and its results are summarised in §10. Numbers are
*measured* (this project's runs), *computed* (from files in this project) or *estimated* (derived from a
measurement).

Layout. §1–§5: requirement, diagnosis, benchmark, systems, what is held constant. §6–§7: outcomes, metrics,
statistics, controls and the reading of the requirement. §8: procedure, gates and compute. §9: decisions taken
after seeing data. §10: earlier stages. §11: amendments. §12–§14: error analysis and the results of the
development split and of the confirmatory split. Section numbers are cited from the code and from the other
documents, so they are stable. What is compared is described in `methodology.md`; how to run it in
`reproducibility.md`.

## 1. Requirement, questions and what can be claimed

**Requirement (agreed with the supervisor).** The proposed system must improve verdict accuracy by at least
1 percentage point over the selected baseline, an adapted RAG² system. The improvement must be genuine: no
tuning on the test split, no change of metric, no selection of questions.

**Primary question.** On held-out Cochrane questions asked as of the newest review's date, does
evidence-criteria verification (R2V, §4) give a higher verdict accuracy than the adapted RAG² baseline (R2)?

**Secondary questions.** (a) Does the adapted RAG² baseline beat standard retrieval (B1) and no retrieval
(B0)? (b) How much of any gain comes from stating the evidence criteria alone (R2C) and how much from
verifying a draft answer (R2V against R2C)? (c) Does the currency criterion (publication dates and "newer
evidence takes precedence") contribute (R2V against R2V-ND)? (d) How do the evidence sets that R2 admits differ
from B1's (retrieval metrics, §6)?

**What can be claimed** is fixed in §7: an improvement is *confirmed* only when the paired test says so; a
difference of 1 point or more that the test cannot separate from zero is reported as a point estimate that
meets the requirement but is not demonstrated. A 1-point difference is far below what 528 questions can
confirm (§7), so the design targets the largest error source rather than small refinements.

## 2. Why the study was realigned (diagnosis)

All figures are computed from committed files; item-level inspection used the dev split only.

1. **The bottleneck is the verdict decision, not retrieval.** On dev, no wrong answer of any arm was a
   retrieval miss; nearly all were "evidence admitted, still wrong" (`results/error_analysis_dev.md`).
2. **The answering task and the gold labels define the classes differently.** The gold labels follow the
   benchmark's labelling rubric: NOT ENOUGH INFORMATION *only* when not enough studies were found, REFUTED
   when the result is "at least partially not supported or similar to placebo", and SUPPORTED or REFUTED
   "even if the certainty is low" (`experiments/medchange/label_audit.py`). The answering prompt used for
   every arm so far (the benchmark's own evaluation style) gives none of this. Result on dev, B1: REFUTED
   recall 10% (5 of 49), NOT ENOUGH INFORMATION predicted for 38% of items against 32% gold, SUPPORTED for
   58% against 46% gold; 44 of B1's 106 errors are gold-REFUTED items answered SUPPORTED or NOT ENOUGH
   INFORMATION. The confirmatory aggregate shows the same pattern (REFUTED recall 8.7%).
3. **Indirect evidence is read as support.** Dev examples (MC-00127, MC-00272, MC-00354, MC-00036): the
   model answers SUPPORTED from studies of another intervention, another outcome, surrogate measures or
   non-randomised designs, where the Cochrane review (randomised evidence) concluded the opposite. The
   candidate pools are 72% "other" designs, 23% trials, 6% systematic reviews (dev, P0 diagnostics).
4. **Date-based admission had nothing to act on.** Newer evidence exists for changed and unchanged questions
   alike (a trial or review in the update window for 94.0% of changed and 93.3% of unchanged dev items;
   93.8% and 92.6% on the confirmatory split), and an old trial is not outdated: it stays part of the updated
   meta-analysis. Recency re-ranking lowered accuracy (B3 −4.0 pp against B1 on dev) and shuffling the dates
   changed nothing (C1 against P). What does go out of date is a *synthesis* that newer trials supersede,
   which is a reading task, not an admission rule.
5. **There is no working RAG² baseline in the pipeline.** B1 is standard retrieval; B2 is an untrained
   Flan-T5 stand-in whose input was cut for 31% of the papers it admitted, and it lost 12.6 pp against B1.
   RAG²'s trained filter is not distributed and a local retraining learned only the class prior (`log.md` Phases 13–18).
6. **The evaluation cannot confirm 1 point.** With 528 paired questions the smallest effect detectable with
   80% power is about 4–6 pp (§7). Earlier gains measured on 226 dev items did not replicate (retrieval: +8.6
   pp on dev changed items, +3.1 pp on the confirmatory changed items, not significant).
7. **Previously tried decision-side fixes were weak.** Refitting B1's verdict (B1R) did not beat B1 (52.2%
   against 53.1%, dev cross-validation); per-paper stance predicted the gold direction only weakly (AUC
   0.624); the same five papers in another order keep the verdict 86% of the time, so voting over orders can
   fix at most a small share of answers.

**Consequence.** A component with a realistic chance of a genuine gain must act on the verdict decision and
target the dominant errors: evidence that does not address the question, null results read as "not enough
information", and the near-absence of REFUTED. The temporal idea survives in its defensible form, a currency
criterion applied while reading the evidence, and is tested by ablation.

## 3. Benchmark, splits and reuse of the confirmatory split

MedChange as-of benchmark (`experiments/medchange/manifest.json`): dev 226 items (151 changed, 75 unchanged),
confirmatory 528 items (353 changed, 175 unchanged); question date t_q = the newest review's date; evidence =
PubMed records first public before t_q, Cochrane records excluded; the as-of candidate sets
(`data/pubmed_g0/`, `data/abstracts.jsonl`) already exist for both splits, so no new network access is needed.

**Reuse disclosure.** The confirmatory split was used once before, for B0 against B1 (§10). Its items and
labels have not been used to design anything below; the B0/B1 aggregate confusion matrices of that split were
computed on 2026-10-05 before this design and show the same pattern as dev. The new systems' confirmatory
outputs do not exist yet. Every design choice below rests on the dev split and on the benchmark's published
labelling rubric. This is a weaker guarantee than an untouched split, and the thesis states it.

Alzheimer's disease is reported as a descriptive case study (the `ad_related` items of both splits; 9 changed
and 5 unchanged in total), not tested.

## 4. Systems

All systems use Meta-Llama-3-8B-Instruct (Q4_K_M, llama.cpp, CPU, greedy decoding, seed 42), the same
as-of candidate records of each question, and at most 5 admitted abstracts. Prompts are in `experiments/medchange/prompts.py`
(standard answer) and `experiments/medchange/rag2.py` (everything new); their hashes are recorded with every
output.

| Arm | What it is | Role |
|---|---|---|
| B0 | no evidence, standard answer prompt | reference (existing answers reused) |
| B1 | question → MedCPT dense → cross-encoder → top 5, standard prompt | standard RAG (existing answers reused) |
| **R2** | **adapted RAG²**: rationale query, balanced retrieval, LLM filter, standard prompt | **the baseline** |
| R2-RQ, R2-BR, R2-NF | R2 without the rationale query / without balancing / without the filter | ablations of the baseline (dev) |
| R2C | R2's evidence, read once with the evidence criteria and design/date labels | criteria control |
| **R2V** | R2's answer as a draft, checked against the evidence with the criteria (evidence-criteria verification) | **proposed** |
| R2V-ND | R2V without dates and without the currency criterion | temporal ablation |

**Adapted RAG² (R2).** RAG²'s three components, kept in kind and adapted in implementation:

1. *Rationale-based query formulation.* The generator writes a short rationale for the question (closed book,
   ≤ 128 new tokens, starting with population, intervention, comparison and outcome, as RAG²'s chain-of-thought
   prompt starts with a summary). The rationale alone, not the question, is the dense query (MedCPT query
   encoder; its 64-token limit keeps the opening summary). Adaptation: the corpus searched is the question's
   as-of PubMed candidate set (up to 200 records per question), not a 564 GB index.
2. *Balanced retrieval.* RAG² takes an equal number of snippets from each of four corpora so that the large
   corpus does not crowd out the small ones. With one dated corpus, the strata are evidence types from PubMed
   publication types: systematic reviews/meta-analyses, trials, other designs. Up to 8 per stratum by the
   rationale's dense score; their union is re-ranked by the MedCPT cross-encoder against the original question
   (as in RAG²); the top 8 go to the filter.
3. *Rationale-guided filtering.* RAG²'s filter is a Flan-T5 model trained on perplexity-based labels; its
   checkpoint is not distributed; RAG² reports that a GPT-4o filter matched its trained filter (its Table 3),
   and a local 8B judge is a weaker substitute. Adaptation: the generator judges each of the 8 abstracts (title, results and conclusions,
   ≤ 200 words) with "Does this document contain information that helps answer the question? Yes or No";
   P(yes) is read from the first token's probabilities. Admitted: those with P(yes) ≥ 0.5, in cross-encoder
   order, at most 5. If none passes, the answer is given without evidence. An invalid judgement (yes + no
   probability < 0.5) keeps the paper.

**Evidence-criteria verification (R2V, proposed).** A second pass of the same model receives the question,
the admitted abstracts labelled with design and publication year, R2's draft answer, and these criteria:
directness (only studies of the question's intervention, population and outcome count); design weight
(randomised trials and systematic reviews first); the class definitions (SUPPORTED when the direct evidence
shows at least partial benefit even with low certainty; REFUTED when it shows no benefit, an effect similar to
placebo or control, or harm; NOT ENOUGH INFORMATION only when there are no direct studies); and, when direct
studies disagree, the weight of randomised evidence with newer evidence taking precedence over older evidence
it may have superseded. It answers in three lines (direct studies; findings; final verdict). The class
definitions restate the benchmark's labelling rubric; R2C measures what they achieve without a draft.
Rules fixed now: without admitted evidence the draft stands (nothing to verify against); an output without a
FINAL VERDICT line keeps the draft verdict and is counted as invalid.

**R2C** uses the same evidence, labels, criteria and three-line format in a single pass (no draft); without
admitted evidence it gives R2's answer. **R2V-ND** is R2V with the publication years removed and the currency
clause dropped from the criteria (the randomised-evidence clause stays).

**Settings fixed now, not tuned:** rationale 128 new tokens; 8 per stratum; 8 to the filter; threshold 0.5;
5 admitted; answers and verification 160 new tokens, context 6,144 tokens; filter context 1,536 tokens.

## 5. What is held constant

Generator file and decoding; the standard answer prompt for B0, B1 and R2; the as-of candidate sets; the
budget of 5; the scorer of correctness (the parsed verdict line against the newest gold label; no judge).
Retrieval differs between B1 and the R2 family only; R2, R2C, R2V and R2V-ND admit identical evidence for
each question, so their differences are differences in how the evidence is used.

## 6. Outcomes and metrics

### 6.1 Primary outcome: verdict accuracy

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
* In the realigned study the primary outcome is the same accuracy over all items of the held-out split, compared
  between the proposed R2V and the adapted RAG² baseline R2. The criteria arms (R2C, R2V, R2V-ND) answer in three
  lines and their verdict is parsed from the `FINAL VERDICT:` line by a regular expression
  (`rag2.parse_final_verdict`); a verification output without one keeps the draft verdict and is counted as
  invalid; an R2C output without one is unparsed and counted wrong.

### 6.2 Generation metrics

Secondary: accuracy on changed and on unchanged items; macro-F1; recall of each class; share of each predicted
class; outdated-verdict rate (the answer equals the previous version's verdict, changed items). The automatic
indicators of unsupported answers and the verifier's behaviour are defined in the table below, with the other
supporting outcomes.

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
| Automatic faithfulness proxies | citations point to admitted passages; entailment by a second-family model | optional; outside the primary analysis; the human hallucination annotation is dropped (its code was removed with the original framework, 2026-10-06; Git history) |
| Anachronism rate (realigned study) | the answer mentions a year later than the question date's year; no admitted study, all published before the question date, can support it (approximate: a four-digit count is read as a year) | indicator of unsupported, parametric or future knowledge |
| Unsupported decisive verdict (realigned study) | SUPPORTED or REFUTED while citing none of the admitted studies ("[n]" in a standard answer; the DIRECT STUDIES line in the three-line format) | indicator of unsupported answers; questions without admitted evidence are excluded |
| Verifier behaviour (realigned study) | share of valid outputs; share of verdicts changed from R2's; changes that fixed or broke an answer, and their direction | mechanism of the proposed component |

### 6.3 Retrieval metrics

Evaluated separately from generation, per arm, descriptive: number admitted and share of questions with none;
evidence-type mix of the admitted set (systematic review or meta-analysis / trial / other); update-window share and
share of questions with update-window evidence; mean age of the evidence; overlap with B1 (Jaccard); optionally
*directness@5*, the share of admitted abstracts that an independent second-family model (Qwen2.5-7B-Instruct)
judges to directly test the question's intervention and outcome.

**Manipulation checks are never outcomes.** Showing that an arm admits more recent passages
demonstrates that the mechanism acts as designed; it says nothing about whether answers improve.
Earlier in the project the primary metric was *currency* (the mean temporal score of admitted
evidence) and the fitting objective was the same quantity, so the proposed system would have won it
by construction; that computation survived only in the first pilot's runner (removed 2026-10-06; Git history). The active retrieval-level metrics above replace it. Token F1,
exact match, ROUGE-L, context precision/recall and token-overlap groundedness are implemented only in the original
framework (removed 2026-10-06; Git history; standard RAG diagnostics for its fixture runs); the current pipeline does not compute them, because a single verbatim reference sentence is a weak target
and groundedness depends on each arm's own admitted evidence. Generation is measured here by verdict accuracy,
per-class recall, macro-F1 and the unsupported-answer indicators above.

## 7. Statistical analysis and the reading of the requirement

### 7.1 Procedure

Standard-library statistics (`evaluation/stats.py`), paired over the same questions:

* **Exact McNemar test** on paired discordant verdict outcomes (two-sided exact binomial).
* **Paired bootstrap 95% CI** on the difference, resampling items as units (4,000 resamples with a fixed seed in
  the realigned analysis). Single-arm accuracies carry Wilson 95% intervals.
* **Holm correction** within pre-declared families:
  * Stage 1 (dev, exploratory): P vs B1, P vs B2, P vs B3.
  * Stage 2 (confirmatory, run once): the primary family RQ1 (B1 vs B0) and RQ2 (the selected hybrid vs B1R) at
    family-wise α = .05, and a secondary family (selected vs B1; vs H0; vs its date-shuffled control; S0 vs B1;
    changed-items-only RQ1 and RQ2) corrected among themselves (`analyze_stage2.py`).
  * **Realigned study** (`analyze_rag2.py`): **primary** R2V − R2 over all items, one test at α = .05;
    **secondary family** (Holm among themselves): R2 − B1; R2V − B1; R2C − R2; R2V − R2C; R2V − R2V-ND;
    R2V − R2 on changed items; **ablation family** (dev only, Holm among themselves, exploratory): R2 − R2-RQ;
    R2 − R2-BR; R2 − R2-NF.
  * Descriptive: the label-stable subset (both labelers agree), the Alzheimer's items, all retrieval metrics.
* **Selection and fitting on dev only (stage 2).** The hybrid is selected by the pre-stated rule on repeated 5-fold
  cross-validation (50 repeats, seed 20261003); dev comparisons of the stage-2 arms use out-of-fold predictions
  and are exploratory.
* Optional secondary analyses, outside the confirmatory family: a mixed-effects logistic model
  `correct ~ helpfulness × recency + (1 | item)`, the changed-vs-unchanged interaction, one-sided non-inferiority on
  unchanged items, and P vs the shuffled-date control C1.

A difference is read as real only if the pre-declared test says so at α = 0.05 after correction; an
average gap alone is not sufficient. If about 30% of answers differ between arms (an assumption until
dev results exist), 353 confirmatory changed items give 84% power at the corrected α for a true 10 pp
difference and 61% for 8 pp (simulated in the stage-1 protocol); smaller effects are reported as
inconclusive with their intervals, not as absence of effect. For stage 2 (528 items; a hybrid that differs from
the RAG answer it refines on 15–25% of items) a true +3 pp is detected 26–38% of the time, +4 pp 42–62% and
+5 pp 61–82%: the benchmark can confirm only effects of about 5 pp or more.

### 7.2 Pre-declared reading of the requirement

Δ = R2V − R2 on the confirmatory split:

| Reading | Condition |
|---|---|
| met and confirmed | Δ ≥ +1.0 pp, McNemar p < .05 and the 95% interval above 0 |
| met as a point estimate, not confirmed | Δ ≥ +1.0 pp otherwise |
| not met | Δ < +1.0 pp |

The thesis reports whichever applies, with the interval. A dev result is never reported as meeting it.

### 7.3 Power

Computed with a normal approximation for n = 528; d is the share of questions on which exactly one of the two
systems is right:

| d | SE of Δ | P(Δ̂ ≥ 1 pp) if true Δ = 0 / 2 / 3 / 5 pp | P(confirmed) if true Δ = 3 / 5 pp | 80%-power effect |
|---|---|---|---|---|
| 0.10 | 1.4 pp | 0.23 / 0.77 / 0.93 / 1.00 | 0.59 / 0.95 | 3.9 pp |
| 0.15 | 1.7 pp | 0.28 / 0.72 / 0.88 / 0.99 | 0.43 / 0.84 | 4.7 pp |
| 0.25 | 2.2 pp | 0.32 / 0.68 / 0.82 / 0.97 | 0.28 / 0.63 | 6.1 pp |

An observed +1 pp arises by chance alone 23–32% of the time, so a point estimate meeting the requirement is
weak evidence; confirming an observed +1 pp would need about 3,800–9,600 questions. With 528 the benchmark can
confirm only effects of about 4–6 pp or more.

### 7.4 Controls

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

### 7.5 Abstention and coverage

An arm that admits nothing still produces an answer and is scored on it. Rates are always reported
with coverage (`stats.coverage`, `stats.har`) so that a high threshold cannot manufacture a good-looking
result by answering fewer questions. For verdict accuracy the analogue is the share of NOT ENOUGH INFORMATION
answers, reported beside accuracy: on this benchmark abstaining is a legitimate answer (31% of gold verdicts),
and a method that merely says it more often is shown as such by the per-class recall.

## 8. Procedure, gates and compute

1. **Dev run** (`rag2_pipeline dev`): rationales, R2 candidate lists, filter judgements, answers (R2, R2C, R2V,
   R2V-ND, and with `--ablations` R2-RQ, R2-BR, R2-NF), analysis, dev report, design record.
2. **Dev check** (pre-declared): (a) every arm parses ≥ 95% and the verifier output is valid ≥ 95%;
   (b) the baseline works: R2 ≥ B1 − 5.0 pp on dev; (c) direction: R2V − R2 ≥ 0 on dev. A failure of (a) or (b)
   is a defect: it is fixed, recorded in §9 and dev is rerun. A failure of (c) allows **one** recorded revision
   of the verification prompt on dev. Then the design is frozen whatever dev shows.
3. **Freeze**: `results/rag2_design.json` (every setting, prompt hash and the model file hash) is committed and
   pushed; the confirmatory phase refuses to start unless that file on origin/main equals the current design.
4. **Confirmatory run, once** (`rag2_pipeline confirm --go`): rationales, lists, filter, R2, R2C, R2V, R2V-ND,
   analysis, findings. Nothing is changed afterwards; the result is reported whatever it is.

**Compute** (*estimated* from measured rates: 61 s per B1 answer, 15 s per B0 answer, 6.8 s per one-token
judgement and ≈ 50 s per question for MedCPT encoding and re-ranking on this laptop): rationale ≈ 25 s, candidate
lists ≈ 45 s, filter ≈ 55 s, each evidence answer or verification ≈ 60–75 s per question, about 7 minutes per
question in all. Dev ≈ 26 h (+ ≈ 14 h with the ablations); confirmatory ≈ 60 h (≈ 49 h without R2V-ND); the
optional directness judge ≈ 3 h (dev) and 7 h (confirmatory); the Alzheimer's/dementia set ≈ 25 h (§11). **Measured** on the target laptop with the directness judge: dev 20.4 h (5.4 min per question), held-out 49.2 h (5.6 min per question). A
three-question smoke test of the realigned steps on the target laptop (2026-10-05; its output was not committed)
measured a rationale at ≈ 20 s, candidate lists at ≈ 89 s (which includes one load of the encoders, so an upper
bound), the filter at ≈ 59 s and each answer at ≈ 60 s, about 6.8 minutes per question for the four answers: in
line with the estimate above. The totals stay estimates until the dev run's own timings exist. Every step is
resumable. R2V-ND may be left out of the confirmatory run for time, decided
before the run starts (`--no-temporal-ablation`); then no confirmatory claim is made about the currency
criterion.

## 9. Decisions taken after seeing dev data (forking-path ledger)

Every choice made after any result existed, so that a reader can judge how much the design was shaped by the
data it is tested on.

| Date | Decision | Data seen before it |
|---|---|---|
| 2026-10-05 | Realign to an adapted RAG² baseline with a verification extension (this plan) | stage 1 and stage 2 dev results; the confirmatory B0/B1 result; aggregate B0/B1 confusion matrices of the confirmatory split |
| 2026-10-05 | Put the benchmark's class definitions into the verification criteria; add R2C to measure them alone | dev errors (§2); the rubric, known since the label audit |
| 2026-10-05 | Balance by evidence type; filter with a zero-shot LLM judgement | P0 diagnostics on dev; the failed filter retraining |
| 2026-10-05 | Keep the currency idea only as a reading criterion with an ablation (R2V-ND) | stage-1 dev results; G0 on both splits |
| 2026-10-06 | Freeze the design without using the one allowed prompt revision (dev check: direction failed, R2V − R2 = −0.4 pp) | the dev run's results and the pattern of R2V's changes (fixes and breaks nearly cancelled); no held-out realigned result |

Earlier ledgers (stages 1 and 2) are in the repository history (§10).

## 10. Earlier stages (results of record)

* **Stage 1, recency-aware admission (Temporal Filter, arm P), dev:** not better than standard retrieval
  (P 35.8% against B1 49.7% on changed items); gate G3 failed. Recency re-ranking alone (B3) −4.0 pp.
* **Stage 2, evidence-synthesis layer, dev:** gate 2 failed (stance AUC 0.624; hybrid 51.8% against 52.2%
  for B1R); not run on the confirmatory split.
* **Confirmatory split, B1 against B0 (run once, 2026-10-05):** +1.5 pp (95% CI −2.8 to +6.1), not confirmed;
  changed items +3.1 pp (−2.5 to +8.5).

Details: §13–§14 and `experiments/medchange/results/`.

## 11. Amendments

**2026-10-05, Alzheimer's/dementia secondary test set** (decided before the dev run; the only realigned output
that existed was a three-question smoke test of the pipeline on dev, whose answers were not analysed). The
research proposal named Alzheimer's disease as the domain, and the 14 Alzheimer's items of the main benchmark
are too few for any test. A second held-out set is therefore added: every MedRevQA question whose text names
dementia, Alzheimer's disease, mild cognitive impairment or cognitive decline, from a Cochrane review that is not
in the dev or confirmatory split (by study group **and** by Cochrane ID), exact duplicates removed
(`experiments/medchange/ad_benchmark.py`; counts in `experiments/medchange/manifest_ad.json`): **208 questions from
159 reviews** (gold: 88 NOT ENOUGH INFORMATION, 68 REFUTED, 52 SUPPORTED); 202 come from reviews with a single
version, so the outdated-verdict rate and the update-window metrics do not apply to them, and none has a changed
verdict (computed). By the wording of the question, 48 name Alzheimer's disease and 156 dementia (15 name both) and
19 name neither (cognitive impairment after stroke, in Parkinson's disease or vascular disease, mild cognitive
impairment, delirium), so this is a dementia and cognitive-impairment set in which fewer than a quarter of the
questions name Alzheimer's disease, and the thesis should call it that. Its reviews are older than the main
benchmark's: 48 of the 208 are dated before 2005 (14 of the main benchmark's 762 are), 46 dates are to the year
only (read as 1 January) and the earliest is 2000. As-of evidence will therefore be thinner for many of them; the
pools have not been built, so their sizes are not known. It is a fresh set: no item, label or answer of it
has been seen. It is run **once, after the freeze**, with the frozen design (B0, B1, R2, R2C, R2V; no ablations),
by `rag2_pipeline ad --go`, which has the same guards as the confirmatory phase. The requirement is read on it with
the same rule (§7) as a **secondary** result; the confirmatory split stays primary. With 208 questions the
standard error of R2V − R2 is about 2–3.5 pp depending on how often the two systems disagree (the normal
approximation of §7.3), so only effects of roughly 6–10 pp can be confirmed, and an observed +1 pp arises by chance
alone 32–39% of the time. Several questions
can come from one review (208 questions, 159 reviews); the paired tests treat questions as independent, which
slightly understates the uncertainty, and the thesis states it. Compute (*estimated*): as-of records ≈ 0.4 h
(network) and frozen pools with abstracts ≈ 2.9 h, B0/B1 ≈ 4.4 h, the realigned arms ≈ 14.4 h (R2, R2C, R2V at the held-out run's measured rates) and the judge ≈ 2.4 h; about 22 h in all, 25 h with the judge.

*Correction, 2026-10-05, before any output of this set existed.* The first build contained **212** questions from
163 reviews (89 / 69 / 54). The audit that reproduced it found four questions (AD-14453, AD-15149, AD-15568,
AD-15979) that are 2000–2003 versions of reviews whose newer versions are in the confirmatory split: they sit in
`MedRevQA` as ungrouped rows, so excluding by study group alone let them through. The builder now also excludes
by Cochrane ID; these four are the only difference. The rule is about review membership, not labels or results,
and a regression test and a manifest-consistency test cover it. The 212-question manifest was reproduced
hash-for-hash from the released files before the change; after it the builder gives the 208-question set, which
shares no Cochrane ID, study group, question text or review PMID with dev or confirm (computed).

## 12. Error analysis

Each wrong answer on a changed item is assigned one cause — retrieval miss, admission miss, generator
override, parse failure, or gold-label error — and reported by change type, update-window length and arm.
Stage 1, dev, changed items (`error_analysis.py`; `log.md` Phase 26):
no retrieval misses for any arm; nearly all wrong answers are "evidence admitted, still wrong". Stage 2 adds
per-class recall, macro-F1 and predicted-class shares (`analyze_stage2.py`); confusion matrices,
stance accuracy on decisive flips and the conflict feature against gold NOT ENOUGH INFORMATION are optional additions outside the primary analysis.
Gold-label error is not checked.

## 13. Results: development split (exploratory)

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
| Result tables and figures in the base paper's layout | `results/report/report_data_dev.json` (the data of the dev version); `REPORT.md` is rewritten by each run of `report`, and the committed one is the confirmatory version (§14) |

The only earlier real-data outputs (the Alzheimer's pilot with an extractive stand-in generator and an
unvalidated baseline checkpoint) were in the first pilot's results folder (removed 2026-10-06; Git history); they showed the
pipeline ran, not that anything improved.

## 14. Results: confirmatory split (528 items, run once)

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
6 pp (the benchmark can confirm only effects of about 5 pp or more, §7.1). The +8.6 pp seen on the dev changed items
(§13) did not replicate on the larger split; the dev value is the one chosen for attention after seeing it and
should not be quoted as the effect. Neither extension tested on dev (recency-aware admission, evidence synthesis)
beat standard retrieval, so no confirmatory claim is made for them.

## 15. Results: realigned study, confirmatory split (528 items, run once)

Run once on 2026-10-06/07 with the design frozen as it stood after the dev run (§8, §9); nothing was changed
afterwards (`results/RAG2_FINDINGS.md`, `results/rag2_analysis_confirm.md`, `results/report/REPORT.md`).

| Item | Result |
|---|---|
| Accuracy, all items | B0 46.6%, B1 48.1%, R2 48.7% (44.4–52.9), R2C 48.5%, **R2V 50.0%** (45.8–54.2), R2V-ND 49.0% |
| **Requirement, R2V − R2** | **+1.3 pp**, 95% CI −1.9 to +4.7; 44 questions right only with R2V, 37 only with R2; exact McNemar p = 0.5052. Pre-declared reading: **met as a point estimate, not confirmed** |
| Changed items (353) | R2V 47.0% vs R2 43.6%: +3.4 pp (−0.8 to +7.4), p = 0.148, Holm 0.888: not confirmed |
| Secondary family (Holm) | R2 − B1 +0.6; R2V − B1 +1.9; R2C − R2 −0.2; R2V − R2C +1.5; R2V − R2V-ND +0.9 pp: none confirmed |
| Where the answers move | R2V raises recall of REFUTED (11.9% → 19.1%) and NOT ENOUGH INFORMATION (33.3% → 42.9%) and lowers SUPPORTED (79.5% → 71.8%); macro-F1 39.5% → 44.0% (descriptive) |
| Verifier behaviour | valid output 100%; changed 22.4% of R2's verdicts, 44 fixes against 37 breaks |
| Unsupported answers | decisive verdicts citing no admitted study: R2 15.4%, R2C/R2V/R2V-ND 0.0% (the format requires a DIRECT STUDIES line, so this is partly built in); anachronism rate 1.9% for all evidence arms |
| Retrieval | R2 admits 3.3 abstracts on average (none for 15.3% of questions), 57.3% judged direct by the independent model against 28.6% for B1 |
| Label-stable subset (430) | R2V − R2 = +1.9 pp; R2 − B1 = +1.4 pp (descriptive) |
| Closed-book models, same questions | 50.8–52.8% on all items (Table 1 of `REPORT.md`); the local 8B systems are at or below them |

**Reading.** The point estimate meets the +1 point requirement, but the data cannot separate it from zero, and with
528 questions only effects of about 4–6 pp could be confirmed (§7.3): an observed +1.3 pp arises by chance alone
about a quarter of the time. R2V's gain over standard retrieval (+1.9 pp) is also unconfirmed, and R2 itself is not
better than standard retrieval (+0.6 pp). The evidence-criteria controls do not isolate a single cause: R2C
(criteria without a draft) equals R2, and R2V − R2C is +1.5 pp, not confirmed. The dev estimate (−0.4 pp) and the
held-out estimate (+1.3 pp) differ by less than their intervals, which is what noise around a small effect looks
like. The thesis should report the point estimate with its interval and the sample size that would be needed.
The 208-question Alzheimer's/dementia set has not been run.
