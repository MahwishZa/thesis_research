# Experimental Protocol

This document fixes the design of the study: the requirement, the systems compared, the controls, the procedure and
decision rules, the decisions taken after data had been seen, and the amendments. Metrics, statistics and the rules for
reading results are in `evaluation.md`; the approach is explained step by step in `methodology.md`; the data in
`data.md`; the commands in `reproducibility.md`.

The protocol was fixed on 2026-10-05, before any output of the systems below existed. It replaces the protocol of
stage 1 (recency-aware admission) and stage 2 (an evidence-synthesis layer), which is kept in the repository history
(`git show 92e3aaf:docs/experiment_plan.md`) and summarised in §9. Numbers are *measured* (this project's runs),
*computed* (from files in this project) or *estimated* (derived from a measurement).

## Status

| Part | Status |
|---|---|
| Development run, 226 questions | run 2026-10-05 to 2026-10-06; the pre-declared dev check failed on direction (R2V − R2 = −0.4 pp) |
| Design freeze | completed 2026-10-06 without using the one allowed prompt revision; the design record is committed |
| Held-out run, 528 questions | started 2026-10-06 after the freeze, completed once on 2026-10-08 (results: `evaluation.md` §6) |
| Baseline ablations R2-RQ, R2-BR, R2-NF (development split, optional) | implemented and tested, **not run**; no result exists |
| Alzheimer's/dementia run, 208 questions | **planned, not yet run** (§8) |

## 1. Requirement and questions

**Requirement (agreed with the supervisor).** The proposed system must improve verdict accuracy by at least
1 percentage point over the selected baseline, an adapted RAG² system. The improvement must be genuine: no tuning on
the test split, no change of metric, no selection of questions.

**Primary question.** On held-out Cochrane questions asked as of the newest review's date, does evidence-criteria
verification (R2V, §4) give a higher verdict accuracy than the adapted RAG² baseline (R2)?

**Secondary questions.** (a) Does the adapted RAG² baseline beat standard retrieval (B1) and no retrieval (B0)?
(b) How much of any gain comes from stating the evidence criteria alone (R2C) and how much from verifying a draft
answer (R2V against R2C)? (c) Does the currency criterion (publication dates and "newer evidence takes precedence")
contribute (R2V against R2V-ND)? (d) How do the evidence sets that R2 admits differ from B1's
(`evaluation.md` §1.3)?

What can be claimed from the answers is fixed in `evaluation.md` §5: an improvement is *confirmed* only when the paired
test says so; a difference of 1 point or more that the test cannot separate from zero is reported as a point
estimate that meets the requirement but is not demonstrated.

## 2. Basis of the design

The realignment followed from the results of the earlier stages (§9). All figures are computed from committed files;
item-level inspection used the dev split only.

1. **The bottleneck is the verdict decision, not retrieval.** On dev, no wrong answer of any arm was a retrieval
   miss; nearly all were "evidence admitted, still wrong" (`experiments/medchange/results/earlier_stages/error_analysis_dev.md`).
2. **The answering task and the gold labels define the classes differently.** The gold labels follow the
   benchmark's labelling rubric: NOT ENOUGH INFORMATION *only* when not enough studies were found, REFUTED when the
   result is "at least partially not supported or similar to placebo", and SUPPORTED or REFUTED "even if the
   certainty is low" (`experiments/medchange/label_audit.py`). The answering prompt used for every arm so far gives
   none of this. On dev, B1: REFUTED recall 10% (5 of 49), NOT ENOUGH INFORMATION predicted for 38% of items against
   32% gold, SUPPORTED for 58% against 46% gold. The confirmatory aggregate shows the same pattern (REFUTED recall
   8.7%).
3. **Indirect evidence is read as support.** Dev examples (MC-00127, MC-00272, MC-00354, MC-00036): the model answers
   SUPPORTED from studies of another intervention, another outcome, surrogate measures or non-randomised designs,
   where the Cochrane review (randomised evidence) concluded the opposite. The candidate pools are 72% "other"
   designs, 23% trials, 6% systematic reviews (dev).
4. **Date-based admission had nothing to act on.** Newer evidence exists for changed and unchanged questions alike
   (a trial or review in the update window for 94.0% of changed and 93.3% of unchanged dev items; 93.8% and 92.6% on
   the held-out split), and an old trial is not outdated: it stays part of the updated meta-analysis. Recency
   re-ranking lowered accuracy (B3 −4.0 pp against B1 on dev) and shuffling the dates changed nothing (C1 against P).
   What does go out of date is a *synthesis* that newer trials supersede, which is a reading task, not an admission
   rule.
5. **There was no working RAG² baseline in the pipeline.** B1 is standard retrieval; B2 was an untrained Flan-T5
   stand-in whose input was cut for 31% of the papers it admitted, and it lost 12.6 pp against B1. RAG²'s trained
   filter is not distributed and a local retraining learned only the class prior (`log.md` Phases 13–18).
6. **The evaluation cannot confirm 1 point.** With 528 paired questions the smallest effect detectable with 80% power
   is about 4–6 pp (`evaluation.md` §4). Earlier gains measured on 226 dev items did not replicate (retrieval: +8.6 pp
   on dev changed items, +3.1 pp on the held-out changed items, not significant).
7. **Decision-side fixes tried earlier were weak.** Refitting B1's verdict (B1R) did not beat B1 (52.2% against 53.1%,
   dev cross-validation); per-paper stance predicted the gold direction only weakly (AUC 0.624); the same five papers
   in another order keep the verdict 86% of the time, so voting over orders can fix at most a small share of answers.

**Consequence.** A component with a realistic chance of a genuine gain must act on the verdict decision and target
the dominant errors: evidence that does not address the question, null results read as "not enough information", and
the near-absence of REFUTED. The temporal idea survives in its defensible form, a currency criterion applied while
reading the evidence, and is tested by ablation.

## 3. Benchmark, splits and reuse of the held-out split

MedChange as-of benchmark (`experiments/medchange/manifest.json`; `data.md` §1): dev 226 items (151 changed, 75
unchanged), held-out ("confirm") 528 items (353 changed, 175 unchanged). The question date t_q is the newest review's
date; the evidence is PubMed records first public before t_q, Cochrane records excluded. The as-of candidate sets
already existed for both splits, so the realigned dev and held-out runs needed no new network access.

**Reuse disclosure.** The held-out split was used once before, for B0 against B1 (§9). Its items and labels were not
used to design anything below; the B0/B1 aggregate confusion matrices of that split were computed on 2026-10-05
before this design and show the same pattern as dev. Every design choice rests on the dev split and on the benchmark's
published labelling rubric. This is a weaker guarantee than an untouched split, and it is declared here. The
Alzheimer's/dementia set (§8) is the only split no result has touched.

## 4. Systems

All systems use Meta-Llama-3-8B-Instruct (Q4_K_M, llama.cpp, CPU, greedy decoding, seed 42), the same as-of candidate
records of each question, and at most 5 admitted abstracts. How each system works is explained in `methodology.md`
§§4–7; prompts are in `experiments/medchange/prompts.py` (standard answer) and `experiments/medchange/rag2.py`
(everything else), and their hashes are recorded with every output.

| Arm | What it is | Role |
|---|---|---|
| B0 | no evidence, standard answer prompt | reference |
| B1 | question → MedCPT dense → cross-encoder → top 5, standard prompt | standard RAG |
| **R2** | **adapted RAG²**: rationale query, balanced retrieval, LLM filter, standard prompt | **baseline** |
| R2-RQ, R2-BR, R2-NF | R2 without the rationale query / without balancing / without the filter | ablations of the baseline (dev, optional; not run) |
| R2C | R2's evidence, read once with the evidence criteria and design/date labels | criteria control |
| **R2V** | R2's answer as a draft, checked against the evidence with the criteria (evidence-criteria verification) | **proposed system** |
| R2V-ND | R2V without dates and without the currency criterion | temporal ablation |

**Rules fixed in advance (not tuned).**

* R2: rationale of at most 128 new tokens; up to 8 abstracts per evidence type; the top 8 after cross-encoder
  re-ranking go to the filter; threshold P(yes) ≥ 0.5; at most 5 admitted; if none passes, R2 answers without
  evidence; an invalid filter judgement (yes + no probability < 0.5) keeps the paper.
* R2C, R2V, R2V-ND: a three-line answer (direct studies; findings; `FINAL VERDICT:`). Without admitted evidence the
  R2 answer stands. A verification output without a final verdict keeps the draft verdict and is counted as invalid; an
  R2C output without one is unparsed and counted wrong.
* Answers and verification: 160 new tokens, context 6,144 tokens; filter context 1,536 tokens. If a prompt does not
  fit the context window, the abstracts are cut to 200 words once and the answer is flagged `context_truncated`; the
  analysis counts such answers.
* `rag2.SETTINGS` and every prompt text are hashed into `results/rag2_design.json` (§6).

## 5. Controls, ablations and what is held constant

* **B0** shows whether retrieval helps or hurts at all; **B1** shows whether the adapted RAG² retrieval beats standard
  retrieval.
* **R2C** (criteria, no draft) separates what the evidence criteria achieve from what verifying a draft adds;
  **R2V-ND** (no dates, no currency clause) isolates the temporal criterion; **R2-RQ, R2-BR, R2-NF** remove one RAG²
  component each.
* The **label-stable subset** (both labelers agree on the newest label) checks that a difference is not carried by
  unreliable labels; **unchanged items** check that no system loses accuracy where nothing changed.

**Held constant:** the generator file and decoding; the standard answer prompt for B0, B1 and R2; the as-of candidate
sets; the budget of 5; the scorer of correctness (the parsed verdict line against the newest gold label; no judge
model). Retrieval differs between B1 and the R2 family only; R2, R2C, R2V and R2V-ND admit identical evidence for each
question, so their differences are differences in how the evidence is used. How this is enforced:
`methodology.md` §9.

## 6. Procedure

1. **Development run** (`rag2_pipeline dev`): rationales, R2 candidate lists, filter judgements, answers (R2, R2C, R2V,
   R2V-ND, and with `--ablations` R2-RQ, R2-BR, R2-NF), analysis, dev report, design record.
2. **Dev check** (pre-declared): (a) every arm parses at least 95% and the verifier output is valid at least 95%;
   (b) the baseline works: R2 ≥ B1 − 5.0 pp on dev; (c) direction: R2V − R2 ≥ 0 on dev. A failure of (a) or (b) is a
   defect: it is fixed, recorded in §7 and dev is rerun. A failure of (c) allows **one** recorded revision of the
   verification prompt on dev. Then the design is frozen whatever dev shows. *Outcome:* (a) and (b) passed, (c) failed
   (−0.4 pp); the revision was not used (§7).
3. **Freeze:** `results/rag2_design.json` (every setting, prompt hash and the generator file's hash) is committed and
   pushed. Both held-out phases refuse to start unless that file on origin/main equals the current design.
4. **Held-out run, once** (`rag2_pipeline confirm --go`): rationales, lists, filter, R2, R2C, R2V, R2V-ND, analysis,
   findings. Nothing is changed afterwards; the result is reported whatever it is.
5. **Alzheimer's/dementia run, once, after the freeze** (`ad_benchmark`, then `rag2_pipeline ad --go`): §8.
6. **Commits and pushes.** The pipelines commit results (`--commit`) but never push; the researcher runs
   `git push origin main`, which is also what puts the design record where the guard looks for it.

**Compute (measured on the target laptop, with the directness judge).** Development run 20.4 h (5.4 min per
question); held-out run 49.2 h (5.6 min per question). The Alzheimer's run is *estimated* at about 22 h, or 25 h with
the judge (§8). Every step is resumable. R2V-ND may be left out of a held-out run for time
(`--no-temporal-ablation`), decided before the run; then no claim is made about the currency criterion.

## 7. Decisions taken after seeing data (forking-path ledger)

Every choice made after any result existed, so that a reader can judge how much the design was shaped by the data it
is tested on.

| Date | Decision | Data seen before it |
|---|---|---|
| 2026-10-05 | Realign to an adapted RAG² baseline with a verification extension | stage 1 and stage 2 dev results; the held-out B0/B1 result; aggregate B0/B1 confusion matrices of the held-out split |
| 2026-10-05 | Put the benchmark's class definitions into the verification criteria; add R2C to measure them alone | dev errors (§2); the rubric, known since the label audit |
| 2026-10-05 | Balance by evidence type; filter with a zero-shot LLM judgement | P0 diagnostics on dev; the failed filter retraining |
| 2026-10-05 | Keep the currency idea only as a reading criterion with an ablation (R2V-ND) | stage-1 dev results; G0 on both splits |
| 2026-10-06 | Freeze the design without using the one allowed prompt revision (dev check: direction failed, R2V − R2 = −0.4 pp) | the dev run's results and the pattern of R2V's changes (fixes and breaks nearly cancelled); no held-out realigned result |

Earlier ledgers (stages 1 and 2) are in the repository history (§9).

## 8. Amendments

**2026-10-05, Alzheimer's/dementia secondary test set** (decided before the dev run; the only realigned output that
existed was a three-question smoke test of the pipeline on dev, whose answers were not analysed). The research
proposal named Alzheimer's disease as the domain, and the 14 Alzheimer's items of the main benchmark are too few for
any test. A second held-out set is therefore added: every MedRevQA question whose text names dementia, Alzheimer's
disease, mild cognitive impairment or cognitive decline, from a Cochrane review that is not in the dev or held-out
split (by study group **and** by Cochrane ID), exact duplicates removed (`experiments/medchange/ad_benchmark.py`;
counts in `experiments/medchange/manifest_ad.json`): **208 questions from 159 reviews** (gold: 88 NOT ENOUGH
INFORMATION, 68 REFUTED, 52 SUPPORTED). 202 come from reviews with a single version, so the outdated-verdict rate and
the update-window metrics do not apply, and none has a changed verdict (computed). By the wording of the question, 48
name Alzheimer's disease and 156 dementia (15 name both) and 19 name neither (cognitive impairment after stroke, in
Parkinson's disease or vascular disease, mild cognitive impairment, delirium), so this is a dementia and
cognitive-impairment set in which fewer than a quarter of the questions name Alzheimer's disease, and it should be
called that. Its reviews are older than the main benchmark's: 48 of the 208 are dated before 2005 (14 of the
main benchmark's 762 are), 46 dates are to the year only (read as 1 January) and the earliest is 2000. As-of evidence
will therefore be thinner for many of them; the pools have not been built, so their sizes are not known.

It is a fresh set: no item, label or answer of it has been seen. It is run **once, after the freeze**, with the frozen
design (B0, B1, R2, R2C, R2V; no ablations), by `rag2_pipeline ad --go`, which has the same guards as the held-out
phase. The requirement is read on it with the same rule (`evaluation.md` §5) as a **secondary** result; the held-out
split stays primary. With 208 questions the standard error of R2V − R2 is about 2–3.5 pp depending on how often the
two systems disagree, so only effects of roughly 6–10 pp can be confirmed, and an observed +1 pp arises by chance
alone 32–39% of the time (`evaluation.md` §4). Several questions can come from one review (208 questions, 159
reviews); the paired tests treat questions as independent, which slightly understates the uncertainty (a limitation
of the analysis). *Estimated compute:* as-of records ≈ 0.4 h (network), frozen pools with abstracts ≈ 2.9 h, B0/B1 ≈ 4.4 h,
the realigned arms ≈ 14.4 h at the held-out run's measured rates, the judge ≈ 2.4 h; about 22 h in all, 25 h with the
judge.

*Correction, 2026-10-05, before any output of this set existed.* The first build contained **212** questions from 163
reviews (89 / 69 / 54). Reproducing it from the released files found four questions (AD-14453, AD-15149, AD-15568,
AD-15979) that are 2000–2003 versions of reviews whose newer versions are in the held-out split: they sit in
`MedRevQA` as ungrouped rows, so excluding by study group alone let them through. The builder now also excludes by
Cochrane ID; these four are the only difference. The rule concerns review membership, not labels or results, and a
regression test and a manifest-consistency test cover it. The 212-question manifest was reproduced hash-for-hash
before the change; after it the builder gives the 208-question set, which shares no Cochrane ID, study group, question
text or review PMID with dev or held-out (computed).

## 9. Earlier stages (completed, superseded)

Both stages were run to a pre-declared gate and did not pass it. Their code was removed from the active tree on
2026-10-08 after it was verified that the current pipeline needs none of it; it is in Git history at commit `f721bbb`
(`git checkout f721bbb`), and their outputs are kept in `experiments/medchange/results/earlier_stages/`.

* **Stage 1, recency-aware admission (the Temporal Filter; arms B2, B3, P, C1).** Passages were admitted by a score
  combining relevance with the age of the evidence (`methodology.md` §10). Gates on the dev split: G0 (evidence headroom:
  newer evidence is retrievable), G1 (format validity), G2 (the generator uses evidence: B1 changes at least 20% of
  B0's verdicts) and G3 (P beats B2 by at least 5 pp and the shuffled-date control C1 by at least 2.5 pp). G0–G2 passed;
  **G3 failed** (P − B2 = −1.3 pp, P − C1 = −2.0 pp).
* **Stage 2, evidence-synthesis layer.** One judgement per paper (supports / contradicts / neither) from the first
  eight pool papers, condensed into four features and combined with B1's verdict by a regularised logistic regression
  fitted on dev and frozen. Gate 1 (machine checks of a 40-question pilot) passed; **gate 2 failed** (stance AUC 0.624,
  hybrid 51.8% against 52.2% for the refit B1 answer), so the layer was not run on the held-out split.
* **Held-out split, B1 against B0 (run once, 2026-10-05).** Pre-specified as RQ1 of stage 2: +1.5 pp (95% CI −2.8 to
  +6.1), not confirmed.

Results of these stages: `evaluation.md` §6.3.
