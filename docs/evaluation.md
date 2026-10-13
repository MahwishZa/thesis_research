# Evaluation

How the systems are scored, compared and interpreted: the metrics for generation and retrieval, the comparison
methodology, the statistical analysis, what the sample size can and cannot confirm, the rules for reading a result,
the results obtained, the answers to the research questions and the limitations. The design that these rules apply to is in `protocol.md`.

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
* The comparison is accuracy over **all** questions of a set, between the proposed system R2V and the adapted RAG² baseline R2
  (pooling changed and unchanged questions raises power). It is read on the Alzheimer's disease set (the primary set) and on the held-out split (secondary set) by the same rule (§5).
* The pre-registered test on fresh questions (`protocol.md` §10, results in §6.9) has a different primary outcome, the macro-F1 of
  R2V − R2 (§1.2), because it asks a different question: whether verification makes the three verdicts more balanced. The two outcomes are
  not interchangeable and neither is re-read from the other.

### 1.2 Generation metrics

| Metric | Definition | Role |
|---|---|---|
| Accuracy on changed / unchanged questions | as the primary, on questions whose verdict changed / never changed | secondary; the unchanged questions check that no system loses accuracy where nothing changed |
| Per-class recall; macro-F1; share of answers per class | recall of each gold class; unweighted mean of the F1 of the three classes (0 for a class never predicted correctly); how often a system answers each class | shows whether accuracy comes from reading evidence or from answering NOT ENOUGH INFORMATION more often; macro-F1 is the primary outcome of the pre-registered test (§6.9) |
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
  covers the development and held-out splits, the Alzheimer's disease set (§6.6) and, for the pre-registered test, a seeded sample of 300 fresh questions (§6.9).
* **Stated-verdict consistency** (`verdict_consistency.py`). On a seeded sample of standard answers, an independent judge
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
* **Existing models (context only).** The benchmark authors released the closed-book answers of eight other language models
  (Qwen2.5-7B, Mistral-24B, Llama-3.3-70B, GPT-4o-mini, DeepSeek-V3, OLMo-13B, BioMistral, PMC-LLaMA) to these questions. They are scored with the same
  rule on the same questions and shown beside the local systems (`results/report_confirm/REPORT.md`, `results/report_ad/REPORT.md`). They differ in size,
  training and prompt and use no retrieval, so they place the local 8B system among existing models; they are not a
  controlled comparison and are not tested.
* **Published methods.** RAG² as published is not comparable (different task, scale, corpus and filter); the adapted
  baseline R2 is the stand-in. Recency-aware retrieval (TempRALM) corresponds to the stage-1 arms. The literature is
  positioned in `methodology.md` §11.
* **Further models were not run.** Each extra generator costs days of CPU time on the study's hardware, and larger ones
  do not fit; a stronger generator, a trained filter and a larger sample are the natural next steps.

## 3. Statistical analysis

Paired over the same questions. The accuracy comparisons use standard-library statistics (`experiments/medchange/stats.py`,
`experiments/medchange/scoring.py`); the macro-F1 and per-verdict analyses use numpy for the bootstrap (`class_balance.py`, `analyze_fresh.py`):

* **Exact McNemar test** on the discordant pairs (two-sided exact binomial).
* **Paired bootstrap 95% interval** of the difference, resampling questions as units (4,000 resamples, fixed seed).
  Single accuracies carry Wilson 95% intervals. The macro-F1 differences of the pre-registered test use 10,000 resamples of questions with a seed
  hashed from the text `fresh-macro-f1` (`analyze_fresh.py`); `class_balance.py` (exploratory) uses 10,000 resamples seeded from the first 8 bytes of its
  seed text, so its comparisons share one stream of resamples (the intervals are valid and are kept as committed).
* **Holm correction** within pre-declared families, at family-wise α = .05 (`analyze_rag2.py`):
  * *primary:* R2V − R2 over all questions, a single test;
  * *secondary family:* R2 − B1; R2V − B1; R2C − R2; R2V − R2C; R2V − R2V-ND; R2V − R2 on changed questions;
  * *ablation family* (development split only, exploratory; the ablations have not been run): R2 − R2-RQ; R2 − R2-BR;
    R2 − R2-NF;
  * *descriptive:* the label-stable subset, the Alzheimer's-related items of the main benchmark, all retrieval metrics.
* **Clustering.** Several Alzheimer's disease and dementia questions can come from one review (208 questions, 159 reviews); the
  tests treat questions as independent, which slightly understates the uncertainty.

## 4. Power and what can be confirmed

Computed with a normal approximation for n = 528; d is the share of questions on which exactly one of the two systems
is right:

| d | SE of Δ | P(confirmed) if true Δ = 3 / 5 pp | 80%-power effect |
|---|---|---|---|
| 0.10 | 1.4 pp | 0.59 / 0.95 | 3.9 pp |
| 0.15 | 1.7 pp | 0.43 / 0.84 | 4.7 pp |
| 0.25 | 2.2 pp | 0.28 / 0.63 | 6.1 pp |

With the 528 held-out questions (secondary set) only effects of about 4–6 pp or more can be confirmed, so a smaller gain may be real and still go
unconfirmed. For the 208-question Alzheimer's disease set (primary set) the standard error is 2.2–3.5 pp and the 80%-power effect 6–10 pp, so it can confirm still less.

For the macro-F1 hypothesis of the pre-registered test the standard error of R2V − R2 is about 0.41 / √N, estimated from the intervals of
`class_balance.py`; with N = 850 questions it is about 0.014, the chance of confirming a true gain of 0.045 (about what the held-out split
showed) is 89%, of 0.04 81%, of 0.03 57% and of 0.02 30%, and the smallest gain confirmable with 80% power is about 0.039 (`protocol.md` §10.3). A true gain
below about 0.03 may therefore go unconfirmed, and the interval is then reported as it is.

## 5. Interpretation rules

* **Confirmed.** A comparison is *confirmed* only if its Holm-adjusted p is below .05, its 95% interval excludes 0 and
  the difference is positive. Everything else is an estimate with an interval, not a finding.
* **Reading of the accuracy difference.** Δ = R2V − R2 on a set. The condition for *confirmed* is the one fixed on 2026-10-05, before
  any held-out output existed, and it is applied unchanged to every set:

  | Reading | Condition |
  |---|---|
  | confirmed | McNemar p < .05 and the 95% interval above 0 |
  | positive but not confirmed | Δ > 0 otherwise |
  | no evidence | Δ ≤ 0 |

  Whichever reading applies is reported with the interval. A dev result is never read as a finding. The Alzheimer's disease set (primary set) and the held-out split (secondary set) are read by this one rule. The protocol of
  2026-10-05 had named the held-out split as the primary test and the Alzheimer's disease set as a secondary reading; the order of presentation was changed on 2026-10-13, after every result was known (`protocol.md` §8).
* **Pre-registered test.** Its hypothesis is *confirmed* only if the lower end of the 95% bootstrap interval of the macro-F1 difference R2V − R2 is above 0;
  otherwise it is *positive but not confirmed* (point estimate above 0) or *no evidence*, and when the upper end is below 0.02 a gain of 0.02 or more is
  excluded (`protocol.md` §10.2). Every other outcome of that test is secondary and carries no confirmatory claim.
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

The research domain is Alzheimer's disease and related dementias. Its evidence, the **primary set**, is the 208-question Alzheimer's disease set (§6.6; 48 questions name Alzheimer's disease, 156 dementia (15 name both), 19 other cognitive impairment), reported whole and with the Alzheimer's-named subgroup. General medicine is the **secondary set**: the held-out split of the as-of Cochrane benchmark (§6.2), its development split (§6.1) and the 850 fresh questions of the pre-registered test (§6.9). The Alzheimer's disease set was presented as the primary set on 2026-10-13, after every result was known; the protocol of 2026-10-05 had named the held-out split as the primary (pre-declared, confirmatory) test and the Alzheimer's disease set as a secondary result (`protocol.md` §1, §8). An Alzheimer's-only question set was specified on 2026-10-09 (`protocol.md` §8) and its
construction was stopped at gate 1 the same day (§6.8), so no Alzheimer's-only evaluation exists; the Alzheimer's-named part of the primary set is exploratory. The macro-F1 hypothesis registered afterwards is tested on
850 fresh general-medicine questions (§6.9). Nothing below is an Alzheimer's-only evaluation.

| Result set | Questions | Class | Scope |
|---|---|---|---|
| The Alzheimer's disease set, i.e. Alzheimer's disease and related dementias (§6.6) | 208 | primary set; completed once (the protocol of 2026-10-05 called it a secondary result) | Alzheimer's disease and related dementias: 48 questions name Alzheimer's disease, 156 dementia (15 name both), 19 other cognitive impairment |
| The questions of that set that name Alzheimer's disease (§6.7) | 48 | exploratory; defined after the results were known | Alzheimer's disease (description only) |
| As-of Cochrane questions, held-out split (§6.2) | 528 | secondary set; completed once; the pre-declared primary and confirmatory test of the protocol of 2026-10-05 | general medicine |
| Pre-registered test on fresh questions (§6.9) | 850 | secondary set; completed once; registered before its data existed; analysed blind; confirmatory for the macro-F1 hypothesis only (not confirmed) | general medicine |
| As-of Cochrane questions, development split (§6.1) | 226 | secondary set; exploratory; used to find defects | general medicine |
| Ablation arms, retrieval diagnostics, label audits (§6.2) | | secondary or descriptive | general medicine |
| Released answers of eight other models (§6.2, §6.6) | 8 | context; other prompts, no retrieval | not a controlled comparison |
| Earlier stages (§6.3) | | negative results of record | general medicine |
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

### 6.2 Realigned study, held-out split (528 questions; secondary set; run once, 2026-10-06 to 2026-10-08)

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
| **R2V − R2** | **+1.3 pp** | −1.9 to +4.7 | 0.51 | **positive but not confirmed** |
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
* **Reference without a model.** Always answering SUPPORTED, the most frequent verdict of the reviews, scores 44.3% (234 of 528); the six local systems score 2.3 (B0) to 5.7 (R2V) points above it.
* **Existing models, closed-book, same questions** (released answers): Qwen2.5-7B and GPT-4o-mini 52.8%,
  DeepSeek-V3 52.3%, Llama-3.3-70B and Mistral-24B 50.8%, OLMo-13B 50.0%, BioMistral 45.1%, PMC-LLaMA 34.8%. The local
  8B systems (46.6–50.0%) are below the five best, level with OLMo-13B at best and above the two medical models.

Full tables: `experiments/medchange/results/rag2_analysis_confirm.md`, `RAG2_FINDINGS_CONFIRM.md`,
`results/report_confirm/REPORT.md`.

### 6.3 Earlier stages (completed, superseded)

| Stage | Result | Reading |
|---|---|---|
| 1, recency-aware admission, dev (changed questions) | B2 37.1% (−12.6 pp vs B1, p = 0.001), B3 45.7% (−4.0), P 35.8% (−13.9, p = 0.0008), C1 37.7% against B1 49.7%; P − B2 = −1.3 pp, P − C1 = −2.0 pp | gate G3 **failed** |
| 1, B0 against B1, dev | changed 41.1% vs 49.7%, +8.6 pp (95% CI 0.7 to 16.6), p = 0.060; unchanged 53.3% vs 60.0% | gate G2 passed; the gain did not replicate |
| 2, evidence-synthesis layer, dev | stance AUC 0.624; selected hybrid 51.8% against 52.2% for the refit B1 answer | gate 2 **failed**; not run on the held-out split |
| 2, B1 against B0, held-out (RQ1, run once) | +1.5 pp (95% CI −2.8 to +6.1; 74 / 66 discordant, p = 0.55); changed +3.1 pp (−2.5 to +8.5) | not confirmed |
| Label audit, dev | agreement 83.2%, kappa 0.7422; 188 of 226 questions label-stable | |
| Stated-verdict consistency | parse rate 100% (dev 0 of 1,356 unparsed); the independent judge agrees with the stated verdict on 99.0% (dev) and 97.7% (held-out, 399 sampled) | diagnostic |

Files: `experiments/medchange/results/earlier_stages/` (records) and `results/` (`label_audit_*`, `verdict_consistency_*`).

### 6.4 Reading

**Primary set (Alzheimer's disease and related dementias, §6.6).** R2V scored 3.4 points above R2 (38.0% against 34.6%), but the data cannot separate that from zero (95% interval −1.9 to +8.7, p = 0.28), and with 208
questions only effects of about 6–10 pp could be confirmed (§4). The set was run once, after the held-out split, and none of the five systems scores above the constant answer (42.3%). R2V's gain over standard retrieval
(+7.2 pp, Holm-adjusted p = 0.25) and R2's (+3.8 pp, 0.46) are also unconfirmed. The controls do not isolate a single cause: R2C (criteria without a draft) is 4.8 points above R2 (0.26) and R2V is 1.4 points below R2C (0.68),
and none of these is confirmed.

**Secondary set (general medicine).** R2V scored 1.3 points above R2 on the held-out split, but the data cannot separate that from zero, and with 528
questions only effects of about 4–6 pp could be confirmed (§4): a gap of +1.3 pp or more in R2V's favour arises by chance
alone about a quarter of the time if the two systems are equally good (one-sided p ≈ 0.25). R2V's gain over standard retrieval (+1.9 pp) is also unconfirmed, and R2
itself is not better than standard retrieval (+0.6 pp). The controls do not isolate a single cause: R2C (criteria
without a draft) equals R2, and R2V − R2C is +1.5 pp, not confirmed. The dev estimate (−0.4 pp) and the held-out
estimate (+1.3 pp) differ by less than their intervals, which is what noise around a small effect looks like. The
earlier gain of retrieval over no retrieval (+8.6 pp on dev) did not replicate (+1.5 pp held-out). The point
estimate is reported with its interval. The pre-registered test on 850 new general-medicine questions (§6.9) then found no evidence of a gain in macro-F1.

### 6.5 Rounding in the committed analysis files

The first versions of `rag2_analysis_dev.md` and `rag2_analysis_confirm.md` were written when stored rates carried four decimals before
being shown with one, so a few cells were 0.1 away from the exact value. Held-out split: the REFUTED recall of R2C and R2V (exactly
19.0%) was shown as 19.1% and the accuracy of R2V-ND (exactly 49.1%, 259 of 528) as 49.0%. Development split: the REFUTED recall of R2C
(32.7%) was shown as 32.6%, its share of NOT ENOUGH INFORMATION answers (23.5%) as 23.4%, the outdated-verdict rate of R2V (37.7%) as
37.8% and the systematic-review or meta-analysis share of R2's admitted abstracts (25.3%) as 25.2%. The code now stores six decimals,
and on 2026-10-12 both analyses (`.json` and `.md`) were regenerated from the committed answers with the same code, so their cells equal
the values used in this document. Four figures per file describe the abstracts that standard retrieval (B1) admitted (the evidence mix, the
update-window share, the share of questions with update-window evidence and the mean age); they need the frozen candidate pools, which exist
only on the researcher's computer, and were carried over unchanged from the first versions. Regenerating the first versions with four
decimals in the same way gave the committed files again, byte for byte (the development file only lacked the key `context_truncated`,
added later), and the Alzheimer's disease set's analysis, written with six decimals from the start, is reproduced exactly by the same
procedure. On 2026-10-13 `analyze_rag2` was rerun on the researcher's computer, where the pools exist: it computed the four figures exactly, with six decimals, and the committed
files are those. The only change that shows in a table is the third share of B1's evidence mix in the development file (70.3% became 70.4%; `log.md`, Phase 79).

### 6.6 Alzheimer's disease and related dementias set (primary set, run once)

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

Verdict accuracy: R2V − R2 = +3.4 pp (95% CI −1.9 to +8.7; 19 / 12 discordant, exact McNemar p = 0.28), reading **positive but not
confirmed**; with 208 questions only 6–10 pp can be confirmed (§4). Secondary comparisons (Holm-adjusted p):
R2 − B1 +3.8 pp (0.46), R2V − B1 +7.2 pp (0.25), R2C − R2 +4.8 pp (0.26), R2V − R2C −1.4 pp (0.68); none is confirmed.
R2V changed 20.9% of the answers it verified (19 fixed, 12 broken); its commonest change was SUPPORTED to NOT ENOUGH
INFORMATION (17). None of the five local systems scores above the constant answer. The released answers of the eight models score
Qwen2.5-7B 42.3%, GPT-4o-mini 37.0%, DeepSeek-V3 39.4%, Llama-3.3-70B 43.3%, Mistral-24B 42.3%, OLMo-13B 41.3%, BioMistral
34.1% and PMC-LLaMA 43.3%; Llama-3.3-70B and PMC-LLaMA are two questions above the constant answer.

Label reproducibility (`label_audit --split ad`, run on 2026-10-13): the independent model reproduces 81.2% of the labels of this set (kappa 0.7134; 208 labelled, none unparsed); per gold
class SUPPORTED 73.1% (52 questions), REFUTED 100.0% (68) and NOT ENOUGH INFORMATION 71.6% (88); 169 of the 208 questions are label-stable. On those 169, R2V − R2 = +6.5 pp and
R2 − B1 = +1.8 pp (descriptive; the primary analysis keeps all 208). No question of this set has a changed verdict, so the reproduction of label changes cannot be computed.
The update-window column of `rag2_analysis_ad.md` does not apply to it either. Files: `results/RAG2_FINDINGS_AD.md`, `rag2_analysis_ad.*`, `label_audit_ad.*`,
`results/report_ad/REPORT.md`.

### 6.7 The questions of the Alzheimer's disease set that name Alzheimer's disease (exploratory; no model run)

The 48 of the 208 questions of the Alzheimer's disease set whose text names Alzheimer's disease, analysed from the answers on file
(`python -m experiments.medchange.ad_subgroup`; `results/ad_subgroup_alzheimer.*`). The subgroup was defined after the run, by the
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

R2V against R2: R2V − R2 +4.2 pp (95% CI −6.2 to +14.6; 4 / 2 discordant, exact p = 0.69). Others: R2C − R2 +4.2 pp (95% CI −8.3 to +16.7; 5 / 3 discordant, exact p = 0.73); R2V − R2C +0.0 pp (95% CI −10.4 to +12.5; 4 / 4 discordant, exact p = 1.00); R2 − B1 +8.3 pp (95% CI +0.0 to +18.8; 5 / 1 discordant, exact p = 0.22); R2V − B1 +12.5 pp (95% CI +0.0 to +25.0; 8 / 2 discordant, exact p = 0.11); R2V − B0 +0.0 pp (95% CI −12.5 to +14.6; 6 / 6 discordant, exact p = 1.00).
With 48 questions only paired differences of about 13 to 20 points or more could be confirmed (§4), so none of these is
confirmed. All five systems score below the constant answer, and standard
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

### 6.9 Pre-registered test on fresh questions (850 questions; secondary set; in force since 2026-10-10; run once, completed 2026-10-12)

**What it asks.** Whether evidence-criteria verification (R2V) raises the macro-F1 of the adapted RAG² baseline (R2), on questions that played no part in
the design or in choosing the metric. The questions are 850 general-medicine questions (the secondary set) of reviews dated 2010 or later that no other split uses,
chosen by a seeded hash order (`data.md` §3); the arms are R2, R2C (the criteria control) and R2V; the design, settings and prompts are those of
the held-out run (settings hash `1743afd7045b`, `results/rag2_design.json`). The decision rule is fixed in `protocol.md` §10.2 and summarised in §5.

**Where the hypothesis comes from (exploratory, from answers on file; `class_balance.py`, `results/class_balance.*`).** All systems answer SUPPORTED far
more often than the reviews do and almost never REFUTED, and R2V raises the REFUTED recall and the macro-F1:

| Set | R2V − R2, macro-F1 (95% interval) | R2C − R2 | R2V − R2C |
|---|---|---|---|
| Held-out split (528) | +0.045 (+0.010 to +0.080) | +0.010 (−0.033 to +0.053) | +0.035 (−0.002 to +0.072) |
| Alzheimer's disease set (208) | +0.080 (+0.024 to +0.135) | +0.088 (+0.030 to +0.147) | −0.008 (−0.055 to +0.037) |

The metric was chosen after these results were known, and R2C gained as much as R2V on the Alzheimer's disease set, so part of the gain may come from
reading the evidence by criteria and not from the verification step. That is why the test includes R2C and the comparison R2V − R2C.

**Secondary outcomes (no confirmatory claim).** R2V − R2C macro-F1; REFUTED recall and the share of SUPPORTED answers of R2V and R2; verdict accuracy of
R2V − R2 and R2V − R2C, the first read by the rule of §5; how many answers R2V changed and how many it fixed or broke; the label audit of a
seeded sample of 300 reference labels; and, descriptively, the macro-F1 difference on the Alzheimer's disease set with the statement whether its sign
agrees. The test cannot show an effect for Alzheimer's disease: the fresh questions contain practically none (`data.md` §3).

**Result.** The hypothesis is **not confirmed**: there is no evidence that evidence-criteria verification raises the macro-F1 of the adapted RAG² baseline. The
analysis computed nothing until all 850 questions had an answer for all three arms (10,000 bootstrap resamples; `results/rag2_analysis_fresh.json`,
`results/RAG2_FINDINGS_FRESH.md`).

| System | Accuracy (95% Wilson interval) | Macro-F1 | Predicted SUPPORTED | Recall: SUPPORTED / REFUTED / NOT ENOUGH INFORMATION |
|---|---|---|---|---|
| R2 (baseline) | 47.2% (43.8–50.5) | 0.395 | 70.0% | 84.7% / 14.6% / 29.6% |
| R2C | 42.9% (39.7–46.3) | 0.359 | 70.5% | 79.9% / 15.3% / 24.2% |
| R2V (proposed) | 45.4% (42.1–48.8) | 0.393 | 61.6% | 72.4% / 16.1% / 34.8% |

Gold verdicts of the 850 questions: SUPPORTED 36.2%, REFUTED 16.1%, NOT ENOUGH INFORMATION 47.6%.

* **Primary (H1).** Macro-F1, R2V − R2 = **−0.002** (95% interval −0.027 to +0.023): **no evidence**. The interval excludes a gain as large as the held-out
  estimate (+0.045) but not a gain of 0.02, because its upper end is 0.023.
* **Secondary (no confirmatory claim).** Macro-F1, R2V − R2C: +0.034 (+0.004 to +0.065). REFUTED recall, R2V − R2: +1.5 points (−2.1 to +5.1). Verdict accuracy,
  R2V − R2: −1.8 points (95% interval −4.2 to +0.8; 51 / 66 discordant; exact McNemar p = 0.195), reading by §5: no evidence. Verdict accuracy, R2V − R2C: +2.5 points
  (+0.0 to +4.9; 66 / 45 discordant; p = 0.057).
* **What the verification did.** It changed 19.5% of the 697 answers it verified (136): 51 fixed, 66 broken, 19 stayed wrong. Most of its changes (90 of 136) turned SUPPORTED
  into NOT ENOUGH INFORMATION. R2C, which states the criteria without checking a draft, changed 22.2% of the same answers: 44 fixed, 80 broken.
* **Retrieval (descriptive; added after the run, `protocol.md` §10.9, item 10).** R2 admitted 3.0 abstracts per question on average and none for 18.0% of the questions (held-out split: 3.3 and 15.3%); of the admitted abstracts 20.9% are systematic reviews or meta-analyses, 33.9% trials and 45.2% other studies, and their mean age at the review date is 8.58 years. R2C and R2V read the same abstracts. The update-window measures of the other analyses do not apply: 783 of the 850 questions come from reviews with a single version.
* **Alzheimer's disease set (descriptive).** Macro-F1, R2V − R2 on its 208 questions: +0.080 (+0.024 to +0.135); its sign differs from the fresh result.
* **Reference without a model.** Always answering NOT ENOUGH INFORMATION, the most frequent verdict of the reviews, would score 47.6% (405 of 850): R2 (47.2%), R2V (45.4%) and R2C (42.9%) are all below it, while their macro-F1 (0.395, 0.393 and 0.359) is far above that of the constant answer (0.215).
* **Reference labels.** In the sample of 300, agreement with the independent model 83.0%, kappa 0.7332 (`results/label_audit_fresh.md`).

**Reading.** With 850 questions the test could confirm a true gain of 0.04 with a probability of 81% and a gain of 0.045 with 89% (`protocol.md` §10.3); it found
none. The gains of the exploratory analyses (§6.2, §6.6) therefore did not hold on new questions. The REFUTED recall, which drove them, rose by 7.1 points on the
held-out split (11.9% to 19.0%) and by 16.2 points on the Alzheimer's disease set (5.9% to 22.1%), but by 1.5 points here (14.6% to 16.1%). The criteria alone (R2C),
which gained as much as R2V on the Alzheimer's disease set, did worse than the baseline here (macro-F1 0.359, accuracy 42.9%). R2V made the model more cautious: the recall
of NOT ENOUGH INFORMATION rose (29.6% to 34.8%) and that of SUPPORTED fell (84.7% to 72.4%); the two cancelled in macro-F1, and the net cost was 1.8 points of
accuracy, which cannot be told from chance. The result is for one generator (an 8B model on a laptop), one prompt design and general-medicine questions; it says nothing
about other models, and the fresh questions contain practically none on Alzheimer's disease (`data.md` §3).

## 7. Answers to the research questions

The two objectives of the README and the questions of `protocol.md` §1 and §10, answered from the results of §6. Where a difference was not confirmed, its interval is the
answer: it shows which gains the data rule out and which they leave open.

| Question | Answer | Reading |
|---|---|---|
| Primary set (`protocol.md` §1, asked of the Alzheimer's disease set; 208 questions): does evidence-criteria verification (R2V) give a higher verdict accuracy than the adapted RAG² baseline (R2)? | 38.0% against 34.6%: +3.4 points, 95% interval −1.9 to +8.7, exact McNemar p = 0.28 (§6.6) | positive but not confirmed |
| The same on the held-out split of general-medicine questions (secondary set; 528 questions; the primary test of the protocol of 2026-10-05) | 50.0% against 48.7%: +1.3 points, −1.9 to +4.7, p = 0.51 (§6.2) | positive but not confirmed |
| Pre-registered (`protocol.md` §10; secondary set): on 850 new general-medicine questions, does R2V raise the macro-F1 of R2? | 0.393 against 0.395: −0.002, 95% interval −0.027 to +0.023 (§6.9) | no evidence; the hypothesis is not confirmed |
| (a) Does R2 beat standard retrieval (B1) and no retrieval (B0)? | Alzheimer's disease set: R2 34.6%, B1 30.8%, B0 33.2%; R2 − B1 = +3.8 (−1.4 to +9.1). Held-out: R2 48.7%, B1 48.1%, B0 46.6%; R2 − B1 = +0.6 points (−3.0 to +4.2) | not confirmed |
| (b) How much of a gain comes from stating the criteria alone (R2C) and how much from verifying a draft (R2V − R2C)? | Alzheimer's disease set: R2C − R2 = +4.8 (+0.0 to +9.6) and R2V − R2C = −1.4 (−5.8 to +2.9). Held-out: −0.2 (−3.6 to +3.4) and +1.5 (−1.7 to +4.7). Fresh macro-F1: R2V − R2C = +0.034 (+0.004 to +0.065), a secondary outcome | none confirmed; on the Alzheimer's disease set the criteria alone gained as much as the verification, on the fresh questions they did worse than the baseline |
| (c) Does the currency criterion (publication dates, newer evidence takes precedence) contribute (R2V − R2V-ND)? | Held-out (secondary set): +0.9 points (−1.1 to +3.0); not run on the other sets | not confirmed |
| (d) How do the evidence sets that R2 admits differ from those of B1? | Held-out (secondary set): R2 admits 3.3 abstracts per question against 5.0 and none for 15.3% of the questions; systematic reviews or meta-analyses are 24.6% of its abstracts against 10.9%, trials 32.9% against 17.1%; 57.3% are judged direct by the independent model against 28.6% (§6.2) | descriptive |

**Objective 1, to implement and validate the proposed system with predefined metrics for retrieval and generation.** The adapted RAG² baseline (R2), the criteria control (R2C), the
verification step (R2V) and its ablation (R2V-ND) are implemented (`methodology.md` §5 to §7) and were run under one frozen design (`results/rag2_design.json`, settings hash
`1743afd7045b`). The metrics, the comparisons and the rules for reading them were fixed before the held-out and fresh runs (§1 to §5, `protocol.md` §10), the code is tested
automatically, and the three question sets are rebuilt identically from the public release (`reproducibility.md` §2).

**Objective 2, to determine the extent to which the proposed system improves retrieval and generation compared with baselines and existing works.** Generation: no improvement was
confirmed on any of the three sets. The point estimates of verdict accuracy were +3.4 points on the primary set (Alzheimer's disease) and +1.3 on the held-out split of the secondary set (general medicine), and the macro-F1 difference on the secondary set's new questions was −0.002; the intervals include no gain at all
and, at most, gains of 8.7, 4.7 and 0.023 (macro-F1) respectively. Retrieval: the verification step reads the abstracts that the baseline admits (R2C and R2V use R2's lists), so the
proposed step does not change what is retrieved; the retrieval that differs from standard retrieval is the adapted baseline's, as in (d). Existing works: RAG² as published is not comparable (a different
task, scale, corpus and filter; `methodology.md` §5.1), so the adapted baseline stands in for it. The closed-book answers that the MedChange authors released for eight other models give context: on the
Alzheimer's disease set they score from 34.1% (BioMistral) to 43.3% (Llama-3.3-70B and PMC-LLaMA), and the best local system (39.4%) scores below five of them (§6.6); on the
held-out questions they score from 34.8% (PMC-LLaMA) to 52.8% (Qwen2.5-7B and GPT-4o-mini), and the local systems (46.6% to 50.0%) are below the five best of them (§6.2).

## 8. Limitations and threats to validity

What the results do and do not support, in one place. The first group limits what the evidence can show, the second how far it reaches, the third what was not done.

**What the evidence can show.**

1. *Reference labels are model-made.* The verdicts were labelled by gpt-4o-mini from the reviews' conclusions. An independent model of another family reproduces 81.4% of the held-out
   labels (kappa 0.7161), 83.2% of the development labels, 83.0% of a sample of 300 fresh labels (kappa 0.7332) and 81.2% of the labels of the Alzheimer's disease set (kappa 0.7134). No clinician
   validated any of them (nobody involved in the study is a medical expert), so label reproducibility bounds what any accuracy can mean (§1.4).
2. *Sample sizes.* With 208 questions (primary set) only gains of roughly 6 to 10 points can be confirmed, with 528 (secondary set) only 4 to 6 (§4), and with 850 fresh questions a macro-F1 gain of about 0.039 (`protocol.md` §10.3). A true gain
   below these may go unconfirmed; the Alzheimer's-named subgroup (48 questions) is too small for any conclusion.
3. *Questions are not independent.* Several questions can come from one review (208 questions come from 159 reviews), and the paired tests treat them as independent, which slightly understates the uncertainty.
4. *The order of discovery.* The labels and the standard-retrieval results of the held-out split had been seen before the design was fixed, and the Alzheimer's disease set was run before the pre-registered test and
   helped form its hypothesis (`data.md` §8). Only the 850 fresh questions played no part in choosing the hypothesis, and the metric (macro-F1) was chosen after the accuracy results were known; the pre-registered test is what
   makes it a result at all. The Alzheimer's disease set was presented as the primary set on 2026-10-13, after every result was known; the protocol of 2026-10-05 had named the held-out split as the primary test (`protocol.md` §8).
5. *The model may have seen the answers.* The reviews are older than the generator's training cut-off, so it may have seen some conclusions; the as-of rule limits only the evidence it is shown (`protocol.md` §10.6).
6. *As-of evidence is approximate.* A record is admitted only if its publication dates prove that it preceded the review; the records that cannot be proven are dropped. The text of an abstract is the one PubMed holds
   today, and only records that have an abstract can be retrieved (full texts are not used).

**How far the results reach.**

7. *One generator, one prompt design.* All systems use one 8-billion-parameter model quantised to 4 bits, run on a CPU with greedy decoding and one seed; the prompts were fixed once. The results say nothing about other
   models, sizes, prompts or seeds, and the variation over seeds and prompts was not measured. The generator is also close to the level of the most frequent verdict (§6.2, §6.6, §6.9): on the Alzheimer's
   disease set and on the fresh questions no system scores above it in accuracy, so there is little room for a verification step to raise accuracy.
8. *An adaptation, not a reproduction, of RAG².* The trained filter, the corpus and the task differ from the published method (`methodology.md` §5.1, §12), so no claim is made about RAG² itself.
9. *The primary set is Alzheimer's disease and related dementias, not Alzheimer's disease alone.* The Alzheimer's disease set (208 questions) is a dementia and cognitive-impairment set of which 48 questions name
   Alzheimer's disease. The secondary set is general medicine: its benchmark has 14 Alzheimer's-related questions and the 850 fresh questions contain practically none, so the pre-registered test checks the mechanism outside the domain.
   An Alzheimer's-only question set could not be built (§6.8; 7 usable new reviews were found, 30 were needed), so no confirmatory result for Alzheimer's disease exists.
10. *The primary set has no questions with a change of verdict.* Of the 208 questions of the Alzheimer's disease set, 202 come from reviews with a single version, and so do 783 of the 850 fresh questions. They test whether the answer matches the
    review's verdict, not whether a system notices that newer evidence has changed a verdict; only the development and held-out splits of the secondary set contain changed verdicts (353 of the 528 held-out questions).

**What was not done.**

11. *Not run:* the ablations of the baseline (R2-RQ, R2-BR, R2-NF; optional, `protocol.md` §5), the consistency check of the stated verdicts on the Alzheimer's disease set and the
    fresh questions, and the directness judge on the fresh questions. No result of any of them is claimed.
12. *Only automatic checks.* By design no step needs a person to label, check or edit anything, so the answers were not read for medical correctness; the indicators of unsupported answers (§6.2) are automatic and rough.
