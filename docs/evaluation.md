# Evaluation

How the systems are scored, compared and interpreted: the metrics for generation and retrieval, the comparison
methodology, the statistical analysis, what the sample size can and cannot confirm, the rules for reading a result,
and the results obtained so far. The design that these rules apply to is in `protocol.md`.

## 1. Outcomes and metrics

### 1.1 Primary outcome: verdict accuracy

Each system answers whether a medical hypothesis is SUPPORTED, REFUTED or has NOT ENOUGH INFORMATION. **Verdict
accuracy** is the share of questions whose parsed verdict equals the gold verdict of the newest Cochrane review
version.

* Standard answers must open with `VERDICT: <label>`; the criteria arms (R2C, R2V, R2V-ND) end with
  `FINAL VERDICT: <label>`. The verdict is read from that line by a regular expression
  (`prompts.parse_verdict`, `rag2.parse_final_verdict`), never inferred from prose. There is no judge model on the
  primary path.
* An answer with no parsable verdict counts as wrong and is counted separately (`unparsed`). A verification output
  without a final verdict keeps the draft verdict and is counted as invalid.
* The outcome is independent of the mechanism under test: the gold label is not visible to any system.
* The primary comparison is accuracy over **all** questions of the held-out split, between the proposed system R2V and
  the adapted RAG² baseline R2 (pooling changed and unchanged questions raises power).

### 1.2 Generation metrics

| Metric | Definition | Role |
|---|---|---|
| Accuracy on changed / unchanged questions | as the primary, on questions whose verdict changed / never changed | secondary; the unchanged questions check that no system loses accuracy where nothing changed |
| Per-class recall; macro-F1; share of answers per class | recall of each gold class; mean F1 of the three classes; how often a system answers each class | shows whether accuracy comes from reading evidence or from answering NOT ENOUGH INFORMATION more often |
| Outdated-verdict rate | the verdict equals the previous version's verdict (changed questions) | direct measure of outdated answers |
| Anachronism rate | the answer mentions a year later than the question date's year (approximate: a four-digit number is read as a year) | indicator of unsupported or future knowledge |
| Unsupported decisive verdict | SUPPORTED or REFUTED while citing no admitted study (`[n]` in a standard answer; the DIRECT STUDIES line in the three-line format); questions without admitted evidence are excluded | indicator of unsupported answers; partly built into the three-line format |
| Verifier behaviour | share of valid outputs; share of verdicts changed from R2's; changes that fixed or broke an answer | mechanism of the proposed component |
| Truncated answers | answers written after the abstracts were shortened to fit the context window | transparency |

### 1.3 Retrieval metrics

Evaluated separately from generation, per system, descriptive only: number of abstracts admitted and the share of
questions with none; the evidence-type mix of the admitted set (systematic review or meta-analysis / trial / other);
the update-window share (admitted abstracts that surely first appeared after the previous review version and on or
before the newest) and the share of questions with such evidence; mean age of the evidence; overlap with B1
(Jaccard); and optionally *directness@k*, the share of admitted abstracts that an independent second-family model
(Qwen2.5-7B-Instruct) judges to test the question's intervention and outcome directly.

**Retrieval measures are never outcomes.** They show that a mechanism acts as designed; they say nothing about whether
answers improve. An earlier primary metric, the mean temporal score of the admitted evidence, was also the fitting
objective of the first pilot, so the proposed system would have won it by construction; it was retired.

### 1.4 Reliability of the labels and of the stated verdicts

* **Label reproducibility** (`label_audit.py`). The gold labels are model-generated. An independent model of another
  family re-labels every question from the same conclusions with the authors' rubric; the report gives agreement,
  Cohen's kappa, per-class agreement and whether label changes between versions are reproduced. Questions on which it
  agrees with the gold label are *label-stable*; the analysis repeats the key differences on that subset (descriptive).
  This measures reproducibility, not medical truth: two models can share a bias. No clinician has validated the labels. The audit
  covers the development and held-out splits; the `ad` audit is run separately (`label_audit --split ad`), and until it exists the `ad` analysis has no label-stable subset.
* **Stated-verdict consistency** (`consistency_auto.py`). On a seeded sample of standard answers, an independent judge
  states which verdict the answer expresses; its agreement with the parsed verdict is a diagnostic of the parser.

### 1.5 Metrics not used

Token F1, exact match, ROUGE-L, context precision/recall and token-overlap groundedness belonged to the removed first
design; they are not computed, because a single verbatim reference sentence is a weak target and groundedness depends
on each system's own admitted evidence. A human-annotated hallucination rate was dropped: nobody involved can judge
medical evidence, and clinician validation is a stated limitation.

## 2. Comparison methodology

* **Paired design.** Every system answers the same questions with the same generator, decoding and candidate records;
  comparisons are between systems on the same question (`protocol.md` §5).
* **Controlled comparisons.** B0, B1 and R2 (what retrieval and the RAG² components add); R2, R2C, R2V, R2V-ND (how the
  same evidence is read); the completed stage-1 and stage-2 systems (negative results of record).
* **Existing models (context only).** The benchmark authors released the closed-book answers of other language models
  (Qwen2.5-7B, Mistral-24B, Llama-3.3-70B, GPT-4o-mini, DeepSeek-V3) to these questions. They are scored with the same
  rule on the same questions and shown beside the local systems (`results/report/REPORT.md`). They differ in size,
  training and prompt and use no retrieval, so they place the local 8B system among existing models; they are not a
  controlled comparison and are not tested.
* **Published methods.** RAG² as published is not comparable (different task, scale, corpus and filter); the adapted
  baseline R2 is the stand-in. Recency-aware retrieval (TempRALM) corresponds to the stage-1 arms. The literature is
  positioned in `methodology.md` §11.
* **Further models were not run.** Each extra generator costs days of CPU time on the study's hardware, and larger ones
  do not fit; a stronger generator, a trained filter and a larger sample are the natural next steps.

## 3. Statistical analysis

Standard-library statistics (`evaluation/stats.py`, `experiments/medchange/scoring.py`), paired over the same questions:

* **Exact McNemar test** on the discordant pairs (two-sided exact binomial).
* **Paired bootstrap 95% interval** of the difference, resampling questions as units (4,000 resamples, fixed seed).
  Single accuracies carry Wilson 95% intervals.
* **Holm correction** within pre-declared families, at family-wise α = .05 (`analyze_rag2.py`):
  * *primary:* R2V − R2 over all questions, a single test;
  * *secondary family:* R2 − B1; R2V − B1; R2C − R2; R2V − R2C; R2V − R2V-ND; R2V − R2 on changed questions;
  * *ablation family* (development split only, exploratory; the ablations have not been run): R2 − R2-RQ; R2 − R2-BR;
    R2 − R2-NF;
  * *descriptive:* the label-stable subset, the Alzheimer's-related items of the main benchmark, all retrieval metrics.
* **Clustering.** Several Alzheimer's/dementia questions can come from one review (208 questions, 159 reviews); the
  tests treat questions as independent, which slightly understates the uncertainty.

## 4. Power and what can be confirmed

Computed with a normal approximation for n = 528; d is the share of questions on which exactly one of the two systems
is right:

| d | SE of Δ | P(Δ̂ ≥ 1 pp) if true Δ = 0 / 2 / 3 / 5 pp | P(confirmed) if true Δ = 3 / 5 pp | 80%-power effect |
|---|---|---|---|---|
| 0.10 | 1.4 pp | 0.23 / 0.77 / 0.93 / 1.00 | 0.59 / 0.95 | 3.9 pp |
| 0.15 | 1.7 pp | 0.28 / 0.72 / 0.88 / 0.99 | 0.43 / 0.84 | 4.7 pp |
| 0.25 | 2.2 pp | 0.32 / 0.68 / 0.82 / 0.97 | 0.28 / 0.63 | 6.1 pp |

An observed +1 pp arises by chance alone 23–32% of the time, so a point estimate that meets the requirement is weak
evidence; confirming an observed +1 pp would need about 3,800–9,600 questions. With 528 questions only effects of about
4–6 pp or more can be confirmed. For the 208-question Alzheimer's/dementia set the standard error is 2.2–3.5 pp, the
80%-power effect 6–10 pp, and an observed +1 pp arises by chance alone 32–39% of the time.

## 5. Interpretation rules

* **Confirmed.** A comparison is *confirmed* only if its Holm-adjusted p is below .05, its 95% interval excludes 0 and
  the difference is positive. Everything else is an estimate with an interval, not a finding.
* **Reading of the requirement.** Δ = R2V − R2 on the held-out split, fixed before any held-out output existed:

  | Reading | Condition |
  |---|---|
  | met and confirmed | Δ ≥ +1.0 pp, McNemar p < .05 and the 95% interval above 0 |
  | met as a point estimate, not confirmed | Δ ≥ +1.0 pp otherwise |
  | not met | Δ < +1.0 pp |

  Whichever reading applies is reported with the interval. A dev result is never reported as meeting it. On the
  Alzheimer's/dementia set the same rule gives a secondary reading; the held-out split stays primary.
* **Dev versus held-out.** Dev results are exploratory (226 questions, intervals of about ±6 to ±8 points) and are
  used to find defects and to apply the pre-declared dev check, not to claim effects.
* **Abstention and coverage.** A system that admits nothing still answers and is scored on it. For verdict accuracy the
  analogue is the share of NOT ENOUGH INFORMATION answers, reported beside accuracy; on this benchmark abstaining is a
  legitimate answer (32% of the gold verdicts of the development and held-out splits), and a method that merely says it more often shows in the per-class recall.
* **Mechanism is not outcome.** A change in what is retrieved or how often the verdict changes does not by itself
  indicate improvement (§1.3).
* **Gold-label noise.** The independent model reproduces 81.4% of the held-out gold labels (§6.2), which bounds what any
  accuracy can mean; gold-label error is not corrected.

## 6. Results to date

### 6.0 What each result is

The research aims at Alzheimer's disease. An Alzheimer's-specific question set was specified on 2026-10-09 (`protocol.md` §8) and its
construction was stopped at gate 1 the same day (§6.8), so no Alzheimer's-specific confirmatory evaluation exists. The pre-declared
confirmatory evidence for the requirement remains the held-out split of the as-of Cochrane benchmark (general medicine); the
dementia-wide set is secondary and its Alzheimer's-named part exploratory. Nothing below is an Alzheimer's-specific evaluation.

| Result set | Questions | Class | Scope |
|---|---|---|---|
| As-of Cochrane questions, held-out split (§6.2) | 528 | completed once; pre-declared and confirmatory under the protocol of 2026-10-05 | general medicine |
| Dementia and Alzheimer's set (§6.6) | 208 | completed once; secondary | dementia-wide: 48 questions name Alzheimer's disease |
| As-of Cochrane questions, development split (§6.1) | 226 | exploratory; used to find defects | general medicine |
| Ablation arms, retrieval diagnostics, label audits (§6.2) | | secondary or descriptive | general medicine |
| Released answers of eight other models (§6.2, §6.6) | 8 | context; other prompts, no retrieval | not a controlled comparison |
| Earlier stages (§6.3) | | negative results of record | general medicine |
| Alzheimer's-named part of the dementia set (§6.7) | 48 | exploratory; defined after the results were known | Alzheimer's disease (description only) |
| Alzheimer's-specific question set AD-KQA (§6.8) | | construction stopped at gate 1; no system was run on it | Alzheimer's disease |

### 6.1 Realigned study, development split (226 questions; exploratory)

| System | All | Changed | Unchanged |
|---|---|---|---|
| B0 | 45.1% | 41.1% | 53.3% |
| B1 | 53.1% | 49.7% | 60.0% |
| R2 (baseline) | 50.0% | 48.3% | 53.3% |
| R2C | 49.1% | 43.0% | 61.3% |
| R2V (proposed) | 49.6% | 42.4% | 64.0% |
| R2V-ND | 52.2% | 46.4% | 64.0% |

R2V − R2 = −0.4 pp (95% CI −5.8 to +4.9; 18 questions right only with R2V, 19 only with R2; exact McNemar p = 1.0).
On changed questions −6.0 pp (−11.9 to +0.0; p = 0.09). The verifier changed 23.3% of R2's verdicts, fixing 18 and
breaking 19. Dev check: (a) parsing and validity ≥ 95% passed; (b) R2 ≥ B1 − 5 pp passed (R2 is 3.1 pp below B1);
(c) direction R2V − R2 ≥ 0 failed. The one allowed revision was not used (`protocol.md` §7). Full tables:
`experiments/medchange/results/rag2_analysis_dev.md`.

### 6.2 Realigned study, held-out split (528 questions; run once, 2026-10-06 to 2026-10-08)

| System | All (95% CI) | Changed (353) | Unchanged (175) |
|---|---|---|---|
| B0 | 46.6% (42.4–50.9) | 41.9% | 56.0% |
| B1 | 48.1% (43.9–52.4) | 45.0% | 54.3% |
| R2 (baseline) | 48.7% (44.4–52.9) | 43.6% | 58.9% |
| R2C | 48.5% (44.2–52.7) | 45.0% | 55.4% |
| **R2V (proposed)** | **50.0%** (45.8–54.2) | 47.0% | 56.0% |
| R2V-ND | 49.1% (44.8–53.3) | 45.3% | 56.6% |

| Comparison | Difference | 95% CI | Exact McNemar p (Holm) | Reading |
|---|---|---|---|---|
| **R2V − R2 (primary)** | **+1.3 pp** | −1.9 to +4.7 | 0.51 | **met as a point estimate, not confirmed** |
| R2V − R2, changed questions | +3.4 pp | −0.8 to +7.4 | 0.15 (0.89) | not confirmed |
| R2 − B1 | +0.6 pp | −3.0 to +4.2 | 0.84 (1.0) | not confirmed |
| R2V − B1 | +1.9 pp | −2.5 to +6.1 | 0.44 (1.0) | not confirmed |
| R2C − R2 | −0.2 pp | −3.6 to +3.4 | 1.0 (1.0) | not confirmed |
| R2V − R2C | +1.5 pp | −1.7 to +4.7 | 0.41 (1.0) | not confirmed |
| R2V − R2V-ND | +0.9 pp | −1.1 to +3.0 | 0.49 (1.0) | not confirmed |

* **Where answers move.** R2V raises recall of REFUTED (11.9% → 19.0%) and of NOT ENOUGH INFORMATION (33.3% → 42.9%) and
  lowers SUPPORTED (79.5% → 71.8%); macro-F1 rises from 39.5% to 44.0% (descriptive). The verifier output was valid for
  100% of questions and changed 22.4% of R2's verdicts (44 fixes, 37 breaks).
* **Unsupported answers.** Decisive verdicts citing no admitted study: R2 15.4%, R2C/R2V/R2V-ND 0.0% (partly built into
  the three-line format); anachronism rate 1.9% for all systems with evidence.
* **Retrieval.** R2 admits 3.3 abstracts on average and none for 15.3% of questions; 57.3% of its admitted abstracts
  are judged direct by the independent model, against 28.6% for B1.
* **Label-stable subset (430 questions).** R2V − R2 = +1.9 pp; R2 − B1 = +1.4 pp (descriptive).
* **Label reproducibility.** Agreement 81.4%, kappa 0.7161; 430 of 528 questions label-stable; per gold class:
  SUPPORTED 86.8%, REFUTED 91.3%, NOT ENOUGH INFORMATION 66.7%; 72.5% of the 353 label changes between versions
  reproduced.
* **Existing models, closed-book, same questions** (released answers): Qwen2.5-7B and GPT-4o-mini 52.8%,
  DeepSeek-V3 52.3%, Llama-3.3-70B and Mistral-24B 50.8%, OLMo-13B 50.0%, BioMistral 45.1%, PMC-LLaMA 34.8%. The local
  8B systems (46.6–50.0%) are below the five best, level with OLMo-13B at best and above the two medical models.

Full tables: `experiments/medchange/results/rag2_analysis_confirm.md`, `RAG2_FINDINGS.md`,
`results/report/REPORT.md`.

### 6.3 Earlier stages (completed, superseded)

| Stage | Result | Reading |
|---|---|---|
| 1, recency-aware admission, dev (changed questions) | B2 37.1% (−12.6 pp vs B1, p = 0.001), B3 45.7% (−4.0), P 35.8% (−13.9, p = 0.0008), C1 37.7% against B1 49.7%; P − B2 = −1.3 pp, P − C1 = −2.0 pp | gate G3 **failed** |
| 1, B0 against B1, dev | changed 41.1% vs 49.7%, +8.6 pp (95% CI 0.7 to 16.6), p = 0.060; unchanged 53.3% vs 60.0% | gate G2 passed; the gain did not replicate |
| 2, evidence-synthesis layer, dev | stance AUC 0.624; selected hybrid 51.8% against 52.2% for the refit B1 answer | gate 2 **failed**; not run on the held-out split |
| 2, B1 against B0, held-out (RQ1, run once) | +1.5 pp (95% CI −2.8 to +6.1; 74 / 66 discordant, p = 0.55); changed +3.1 pp (−2.5 to +8.5) | not confirmed |
| Label audit, dev | agreement 83.2%, kappa 0.7422; 188 of 226 questions label-stable | |
| Stated-verdict consistency | parse rate 100% (dev 0 of 1,356 unparsed); the independent judge agrees with the stated verdict on 99.0% (dev) and 97.7% (held-out, 399 sampled) | diagnostic |

Files: `experiments/medchange/results/earlier_stages/` (records) and `results/` (`label_audit_*`, `consistency_auto_*`).

### 6.4 Reading

The point estimate of R2V over R2 on the held-out split meets the +1 point requirement, but the data cannot separate it
from zero, and with 528 questions only effects of about 4–6 pp could be confirmed (§4): a gap of +1.3 pp or more in
R2V's favour arises by chance alone about a quarter of the time if the two systems are equally good (one-sided
p ≈ 0.25). R2V's gain over standard retrieval (+1.9 pp) is also unconfirmed, and R2
itself is not better than standard retrieval (+0.6 pp). The controls do not isolate a single cause: R2C (criteria
without a draft) equals R2, and R2V − R2C is +1.5 pp, not confirmed. The dev estimate (−0.4 pp) and the held-out
estimate (+1.3 pp) differ by less than their intervals, which is what noise around a small effect looks like. The
earlier gain of retrieval over no retrieval (+8.6 pp on dev) did not replicate (+1.5 pp held-out). The point
estimate is reported with its interval, together with the sample size that would be needed to settle it. The
208-question Alzheimer's/dementia set was run afterwards, once (§6.6); it gives the same reading, with a larger and equally unconfirmed gain.

### 6.5 Rounding in the committed analysis files

The committed analysis tables `rag2_analysis_dev.md` and `rag2_analysis_confirm.md` were written when stored rates carried four
decimals before being shown with one, so a few cells differ by 0.1 from the exact value used in this document. Held-out split:
the REFUTED recall of R2C and R2V is 19.0% (shown 19.1%) and the accuracy of R2V-ND is 49.1%, 259 of 528 (shown 49.0%).
Development split: the REFUTED recall of R2C is 32.7% (shown 32.6%), its share of NOT ENOUGH INFORMATION answers 23.5% (23.4%),
the outdated-verdict rate of R2V 37.7% (37.8%) and the systematic-review or meta-analysis share of R2's admitted abstracts 25.3%
(25.2%). The code now stores six decimals, and rerunning `analyze_rag2` where the frozen pools exist regenerates the files with
exact values.

### 6.6 Dementia and Alzheimer's set (secondary, run once)

208 questions from 159 reviews, none with a changed verdict; gold NOT ENOUGH INFORMATION 42.3%, REFUTED 32.7%, SUPPORTED
25.0%. Pools were complete (median 20 candidates, none empty); R2 admitted at least one abstract for 177 of 208
questions (85.1%).

| System | Accuracy (95% Wilson interval) |
|---|---|
| Constant answer NOT ENOUGH INFORMATION (no model) | 42.3% |
| B0 | 33.2% (27.1–39.8) |
| B1 | 30.8% (24.9–37.3) |
| R2 | 34.6% (28.5–41.3) |
| R2C | 39.4% (33.0–46.2) |
| R2V | 38.0% (31.7–44.7) |

Requirement: R2V − R2 = +3.4 pp (95% CI −1.9 to +8.7; 19 / 12 discordant, exact McNemar p = 0.28), reading **met as a point
estimate, not confirmed**; with 208 questions only 6–10 pp can be confirmed (§4). Secondary comparisons (Holm-adjusted p):
R2 − B1 +3.8 pp (0.46), R2V − B1 +7.2 pp (0.25), R2C − R2 +4.8 pp (0.26), R2V − R2C −1.4 pp (0.68); none is confirmed.
R2V changed 20.9% of the answers it verified (19 fixed, 12 broken); its commonest change was SUPPORTED to NOT ENOUGH
INFORMATION (17). None of the five local systems scores above the constant answer. The released answers of the eight models score
Qwen2.5-7B 42.3%, GPT-4o-mini 37.0%, DeepSeek-V3 39.4%, Llama-3.3-70B 43.3%, Mistral-24B 42.3%, OLMo-13B 41.3%, BioMistral
34.1% and PMC-LLaMA 43.3%; Llama-3.3-70B and PMC-LLaMA are two questions above the constant answer. The labels of this set have not been audited by a second model, and the update-window and
age columns of `rag2_analysis_ad.md` do not apply to it. Files: `results/RAG2_FINDINGS_AD.md`, `rag2_analysis_ad.*`,
`results/report_ad/REPORT.md`.

### 6.7 Alzheimer's-named part of the dementia set (exploratory; no model run)

The 48 of the 208 dementia-set questions whose text names Alzheimer's disease, analysed from the answers on file
(`python -m experiments.medchange.subgroup_ad`; `results/ad_subgroup_alzheimer.*`). The subgroup was defined after the run, by the
wording of the question alone, so it is exploratory. Gold NOT ENOUGH INFORMATION 21, REFUTED
19, SUPPORTED 8.

| System | Accuracy (95% Wilson interval) |
|---|---|
| Constant answer NOT ENOUGH INFORMATION (no model) | 43.8% |
| B0 | 31.2% (19.9–45.3) |
| B1 | 18.8% (10.2–31.9) |
| R2 | 27.1% (16.6–41.0) |
| R2C | 31.2% (19.9–45.3) |
| R2V | 31.2% (19.9–45.3) |

Requirement-style comparison: R2V − R2 +4.2 pp (95% CI −6.2 to +14.6; 4 / 2 discordant, exact p = 0.69). Others: R2C − R2 +4.2 pp (95% CI −8.3 to +16.7; 5 / 3 discordant, exact p = 0.73); R2V − R2C +0.0 pp (95% CI −10.4 to +12.5; 4 / 4 discordant, exact p = 1.00); R2 − B1 +8.3 pp (95% CI +0.0 to +18.8; 5 / 1 discordant, exact p = 0.22); R2V − B1 +12.5 pp (95% CI +0.0 to +25.0; 8 / 2 discordant, exact p = 0.11); R2V − B0 +0.0 pp (95% CI −12.5 to +14.6; 6 / 6 discordant, exact p = 1.00).
With 48 questions only paired differences of about 13 to 20 points or more could be confirmed (§4), so none of these is
confirmed and the 1-point requirement is neither met nor failed here. All five systems score below the constant answer, and standard
retrieval (B1) scores lowest.
Per-verdict view of the 48 (exploratory; the metric was chosen after the accuracy results): macro-F1 R2 0.245, R2V 0.305 (R2V − R2 +0.060, 95% interval -0.037 to +0.168); REFUTED recall R2 5.3%, R2V 15.8%; B1 answers SUPPORTED 81.2% of the time against a gold share of 16.7%. R2V changed 6 of the 48 answers (4 fixed, 2 broken; `python -m experiments.medchange.ad_case_study` lists them; its outputs hold question text and are not committed).

### 6.8 Construction of an Alzheimer's-specific question set (AD-KQA): stopped at gate 1 (negative result of record)

Built from 710 PubMed systematic reviews, meta-analyses and guidelines on Alzheimer's disease published from 2023-04-01 (four of seven
areas had enough sources; `protocol.md` §8). On the development split (150 records) gate 1 failed: 11 questions kept (7.3%; gate 40% and
60 questions), verdicts SUPPORTED 10, REFUTED 1, NOT ENOUGH INFORMATION 0 (gate: at least 25% each), claim-only classifier equal to the
majority class. Two model families agree that the source is skewed: of 99 well-formed drafts Qwen2.5-7B proposed SUPPORTED 88, REFUTED 10,
NOT ENOUGH INFORMATION 1, and the independent verifier Phi-3.5-mini labelled 75, 8 and 16; they agree on 77 (72 SUPPORTED, 5 REFUTED, none
NOT ENOUGH INFORMATION). Published reviews of Alzheimer's disease lean positive and seldom state that evidence is insufficient, so a
verdict-balanced question set of useful size cannot be built from this source, and no change of the rules could reach the verdict floors. The
researcher stopped the construction (2026-10-09). No test question was drafted; no system was run on any AD-KQA question. The code is in
Git history (commit `4ab9d1b`) and the counts are in `experiments/medchange/results/earlier_stages/adkqa/`.


