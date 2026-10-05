# Experiment plan: adapted RAG² with evidence-criteria verification

This is the protocol of the realigned study (fixed 2026-10-05, before any output of the systems below
existed). It replaces the earlier protocol of stage 1 (recency-aware admission) and stage 2 (an
evidence-synthesis layer); that text is kept in the repository history (`git show 92e3aaf:docs/experiment_plan.md`)
and its results are summarised in §10. Numbers are *measured* (this project's runs), *computed* (from files
in this project) or *estimated* (derived from a measurement).

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
   RAG²'s trained filter is not distributed and a local retraining learned only the class prior (archived).
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

**Generation.** Primary: verdict accuracy over all items of the split. Secondary: accuracy on changed and on
unchanged items; macro-F1; recall of each class; share of each predicted class; outdated-verdict rate (the
answer equals the previous version's verdict, changed items). Automatic indicators of unsupported answers:
*anachronism rate* (the answer mentions a year later than the question date's year, which no admitted study can
support); *unsupported decisive verdict* (SUPPORTED or REFUTED while citing no admitted study: no "[n]" in a
standard answer, "DIRECT STUDIES: NONE" or no numbers in the three-line format); *verifier changes and
invalid outputs*.

**Retrieval** (per arm, descriptive): number admitted and share of questions with none; evidence-type mix of
the admitted set; update-window share and share of questions with update-window evidence; mean age of the
evidence; overlap with B1 (Jaccard); optionally *directness@5*, the share of admitted abstracts that an
independent second-family model (Qwen2.5-7B-Instruct) judges to directly test the question's intervention
and outcome.

## 7. Statistical analysis and the reading of the requirement

Paired comparisons over the same questions: exact McNemar test on the discordant pairs, paired bootstrap 95%
interval of the accuracy difference (4,000 resamples, fixed seed), Wilson intervals for single arms.

* **Primary:** R2V − R2, all items. One test, α = .05.
* **Secondary family** (Holm among themselves): R2 − B1; R2V − B1; R2C − R2; R2V − R2C; R2V − R2V-ND;
  R2V − R2 on changed items.
* **Ablation family** (dev only, Holm among themselves, exploratory): R2 − R2-RQ; R2 − R2-BR; R2 − R2-NF.
* Descriptive: the label-stable subset (both labelers agree), the Alzheimer's items, all retrieval metrics.

**Pre-declared reading of the requirement** (Δ = R2V − R2 on the confirmatory split):

| Reading | Condition |
|---|---|
| met and confirmed | Δ ≥ +1.0 pp, McNemar p < .05 and the 95% interval above 0 |
| met as a point estimate, not confirmed | Δ ≥ +1.0 pp otherwise |
| not met | Δ < +1.0 pp |

The thesis reports whichever applies, with the interval. A dev result is never reported as meeting it.

**Power** (computed, normal approximation, n = 528; d = share of questions on which exactly one of the two
systems is right):

| d | SE of Δ | P(Δ̂ ≥ 1 pp) if true Δ = 0 / 2 / 3 / 5 pp | P(confirmed) if true Δ = 3 / 5 pp | 80%-power effect |
|---|---|---|---|---|
| 0.10 | 1.4 pp | 0.23 / 0.77 / 0.93 / 1.00 | 0.59 / 0.95 | 3.9 pp |
| 0.15 | 1.7 pp | 0.28 / 0.72 / 0.88 / 0.99 | 0.43 / 0.84 | 4.7 pp |
| 0.25 | 2.2 pp | 0.32 / 0.68 / 0.82 / 0.97 | 0.28 / 0.63 | 6.1 pp |

An observed +1 pp arises by chance alone 23–32% of the time, so a point estimate meeting the requirement is
weak evidence; confirming an observed +1 pp would need about 3,800–9,600 questions. With 528 the benchmark can
confirm only effects of about 4–6 pp or more.

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
optional directness judge ≈ 3 h (dev) and 7 h (confirmatory). Every step is resumable. R2V-ND may be left out of the confirmatory run for time, decided
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

Earlier ledgers (stages 1 and 2) are in the repository history (§10).

## 10. Earlier stages (results of record)

* **Stage 1, recency-aware admission (Temporal Filter, arm P), dev:** not better than standard retrieval
  (P 35.8% against B1 49.7% on changed items); gate G3 failed. Recency re-ranking alone (B3) −4.0 pp.
* **Stage 2, evidence-synthesis layer, dev:** gate 2 failed (stance AUC 0.624; hybrid 51.8% against 52.2%
  for B1R); not run on the confirmatory split.
* **Confirmatory split, B1 against B0 (run once, 2026-10-05):** +1.5 pp (95% CI −2.8 to +6.1), not confirmed;
  changed items +3.1 pp (−2.5 to +8.5).

Details: `docs/evaluation.md` §7–§8 and `experiments/medchange/results/`.

## 11. Amendments

None yet. Any change after 2026-10-05 is added here with its date and the data seen before it.
