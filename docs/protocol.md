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
| Alzheimer's/dementia run, 208 questions | completed once, 2026-10-08 to 2026-10-09 (about 22.7 h with the judge; results: `evaluation.md` §6.6) |
| Added comparisons (all eight released-answer models, constant-answer baseline, dementia label audit) | added 2026-10-09 (§8); the dementia label audit is not run |
| Alzheimer's-specific question set (AD-KQA) | specified 2026-10-09 (§8); the development build failed gate 1 and the construction was stopped the same day (§8, close-out); no test question was drafted and no system was run on it |

The research domain is Alzheimer's disease and related dementias (ADRD; Alzheimer's disease is the most common dementia). The ADRD set is the 208-question dementia set (§8), of which 48 questions name Alzheimer's disease; it is not an Alzheimer's-only set. On 2026-10-09 an Alzheimer's-specific question set was specified and its construction was
stopped at gate 1 (§8, close-out), so the questions, systems and results in §1 to §7 and in `evaluation.md` §6 (the realigned study on the
as-of Cochrane benchmark, general medicine) remain the evidence for the requirement, with the dementia set secondary; no
Alzheimer's-specific confirmatory evaluation exists.

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
| 2026-10-09 | Add the three remaining released-answer models (BioMistral, PMC-LLaMA, OLMo-13B) and a constant-answer baseline to the comparisons; audit the dementia labels with the same judge | the held-out and `ad` results of the local systems (§8) |
| 2026-10-09 | Re-declare the primary evaluation as an Alzheimer's-specific question set (AD-KQA); keep the as-of Cochrane results as secondary evidence | all held-out and dementia results of every system and the label audits; no AD-KQA question, label or output exists; the reason is scope, not outcome (§8) |
| 2026-10-09 | Correct the Stage 0 instrument after run 1 (prevention qualifier, warning filter, information-only counts); thresholds and window unchanged | run 1 counts per area were seen (total 866; prevention 0 by the query error; progression 58; care 20); the criterion was not changed to fit them; the decision after run 2 is the researcher's (§8) |
| 2026-10-09 | Cover only the core areas with at least 100 source records (four of seven); Stage 0 needs 4 covered areas and 650 covered records; test minimum 200 (up to 300) | the counts of Stage 0 runs 1 and 2 were seen (original criterion failed twice); no question, label or output exists; every other threshold is unchanged; the amendment is a scope decision, not a test of the original criterion (§8) |
| 2026-10-09 | Fix the AD-KQA templates, reading cues, topic-cluster split and seed, source freeze and secondary measures (§8) | nothing about the questions: no abstract read, no draft, no answer; the cues come from the audit prompt's definitions and general wording; the spec is in `spec.py` and compared with the protocol text by a test |
| 2026-10-09 | Read a verifier reply by `spec.parse_verdict` ("LABEL:" anywhere, or a first line that begins with the verdict) and re-score the stored replies; thresholds unchanged | qualification run 1 (55.8% unparsed; 85.0% agreement on the 100 read) and eight unparsed replies were seen; the repaired reading is applied to every reply alike (§8) |
| 2026-10-09 | Widen the conclusion rule (discussion tail, last three sentences of an unlabelled abstract) and raise the development share in steps of 0.05 until 150 records; all gates and thresholds unchanged | `prepare` counts (292 of 710 without a conclusion; development pool 110) and `diagnose` labels and sentence starts were seen; no draft or output exists (§8, amendment B) |
| 2026-10-09 | Use short-form templates when a copied span names Alzheimer's disease | five development drafts were read (reply text only); no verdict of any system, no retrieval and no answer exists; every gate and threshold unchanged (§8, amendment C) |
| 2026-10-09 | Stop the construction of the Alzheimer's-specific question set after gate 1 of the development build; primary evaluation reverts to the pre-declared held-out split; report the Alzheimer's-named part of the dementia set as an exploratory subgroup | the whole development build (11 of 150 kept; verdict mix 10 / 1 / 0), the diagnostic verifier pass and the dementia-run answers were seen; the subgroup was defined by the wording of the question after the run (§8, close-out) |
| 2026-10-09 | Draft a pre-registered confirmatory test of the verification effect on fresh as-of questions (§10; not in force) | the exploratory per-verdict analysis of the held-out and dementia answers was seen and gave the hypothesis; the test uses questions outside every existing split; nothing is selected or run yet |

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

*Outcome, 2026-10-09.* The set was run once with the frozen design; the reading is **met as a point estimate, not
confirmed** (R2V − R2 = +3.4 pp, p = 0.28). The results are in `evaluation.md` §6.6.

**2026-10-09, added comparisons** (decided after the held-out and `ad` results of the local systems were known, so
they are not part of the pre-declared design). (1) The released answers of all eight models of the MedChange release are
scored on the held-out and dementia questions, not only the five first chosen. (2) The best constant answer (always
the most frequent gold class) is reported as a no-model reference. (3) The 208 dementia labels are audited with the
same independent judge as the main benchmark's (`label_audit --split ad`). All three are descriptive context. They are
included whatever they score; nothing is added, removed or tuned according to a result, and none of them changes how
the requirement is read.

**2026-10-09, an Alzheimer's-specific question set (AD-KQA): specified, built to a gate, stopped (code archived; this is the condensed record).**
After the held-out and dementia results of every system were known, the researcher re-declared the primary evaluation as an Alzheimer's-specific
question set, because the as-of Cochrane benchmark is general medicine and the dementia set is dementia-wide (48 of its 208 questions name
Alzheimer's disease). *Design:* template questions with a three-way verdict from PubMed systematic reviews, meta-analyses and guidelines (MeSH major
topic Alzheimer Disease, MEDLINE, English, abstract, not retracted, published from 2023-04-01, after the generator's stated knowledge cutoff), one per
source record, asked as of the source's date; the frozen design (settings hash `1743afd7045b`) unchanged; reference verdicts from the abstract's
conclusion read by a rule, a drafting model (Qwen2.5-7B) and an independent verifier (Phi-3.5-mini); pre-declared gates for the source counts (Stage
0), a 150-record development trial (40% survival, verdict floors of 25% each, verifier agreement, claim-only classifier, pool size, closed-book
headroom, same-seed hash) and the full build. The requirement (R2V at least 1 point above R2) was unchanged; the power at about 300 test questions was
computed (smallest confirmable gain about 6 points; a true +1 point confirmed with probability about 7%).

*Stage 0 (counts only, committed in `experiments/medchange/results/earlier_stages/adkqa/`).* Run 1: 866 eligible records, areas treatment 314,
prevention 0, diagnosis 278, causes 216, progression 58, symptoms 171, care 20; four of seven areas reached 100, six were required: no-go. The
prevention count came from a query error ("prevention & control"), corrected, with the criterion unchanged. Run 2: prevention 21; the same four areas
covered (710 distinct records, not the 634 first stated, which came from an allocation that gave shared records to the uncovered areas): no-go again.
*Amendment after the counts (a scope decision, not a pass of the original test):* cover the four areas with at least 100 records, require 4 areas and
650 covered records (200 test + 60 development questions at 40% survival), a test minimum of 200 (up to 300); every other threshold unchanged.

*Specification fixed before any draft:* three question templates (effect, association, test) with short forms when a copied span names the disease
(added after five development drafts showed the disease named twice), rule-based cue lists for the conclusion, topic clusters by the first other major
MeSH descriptor, a seeded split (`adkqa-split-v1`) and the source list frozen from Stage 0 run 2. *Amendment B (after the first `prepare` and
`diagnose` counts, no draft yet):* 292 of 710 records had no conclusion under the first rule (245 without section labels); the conclusion became the
labelled section, else the last three sentences of a discussion section, else of an unlabelled abstract (703 of 710 eligible), and the development
share rose from 0.25 in steps of 0.05 until the pool held 150 records (0.35: 156 development, 554 test records).

*Verifier qualification (Phi-3.5-mini-instruct Q4_K_M, SHA-256 `3EF53267...BB03BCA`; the existing label audit on the 226 development questions).* As
run, 126 of 226 replies were unparsed (55.8%; limit 2%) because the audit required a "LABEL:" prefix and the model leads with the verdict word. The
reading rule was corrected after the replies were seen (declared as such) and the stored replies re-scored with the thresholds unchanged: 0 unparsed,
agreement 77.9% (limit 75%), kappa 0.6641 (limit 0.60), NOT ENOUGH INFORMATION the weak class (65.3%): qualified, a repaired instrument on the same data.

*Close-out, same day: the set is not built.* The development build (150 records) failed gate 1: 11 questions kept (7.3% survival; gate 40% and 60
questions); verdicts SUPPORTED 10, REFUTED 1, NOT ENOUGH INFORMATION 0 (gate: at least 25% each); verifier agreement 91.7% on 12 labelled drafts (met);
claim-only classifier equal to the majority class (met); the pool, closed-book and same-hash checks were not made. Where the drafts were lost: cue
reading gave no verdict 84, copied span not in the abstract 28, length rules 13, no claim 6, no conclusion 3, rule disagreed 3, verifier disagreed 1,
unparsed 1. A diagnostic pass (the verifier labelling every well-formed draft, never used by the keep rules) showed the verdict mix is a property of
the source: of 99 well-formed drafts the drafter proposed SUPPORTED 88, REFUTED 10, NOT ENOUGH INFORMATION 1, the verifier 75, 8 and 16, and they agree
on 77 (72 SUPPORTED, 5 REFUTED, none NOT ENOUGH INFORMATION). Letting the cue reading act only as a veto would raise survival to about 50% but leave
about 94% of the kept questions SUPPORTED. No change of the rules can reach the verdict floors from this source. Decision of the researcher: stop. The
test split was never drafted and no system was run on any AD-KQA question. *Consequences:* the primary evaluation reverts to the pre-declared
held-out split of the protocol of 2026-10-05 (reading unchanged: met as a point estimate, not confirmed); the dementia set is secondary and its 48
Alzheimer's-named questions are an exploratory subgroup (`evaluation.md` §6.7, from answers on file, no model run). No Alzheimer's-specific
confirmatory evaluation exists, and none is claimed. The code, tests and long amendment text left the active tree (Git history, commit `4ab9d1b`;
`git checkout 4ab9d1b -- experiments/adkqa` restores the code, which `evaluation.md` §6.8 summarises).

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

## 10. Pre-registration of a confirmatory test on fresh questions (DRAFT: not in force)

**Status.** This section is a draft written on 2026-10-09. It binds nothing until the freeze of §10.8 is committed, and no fresh question has been
selected, run or looked at. Nothing in it changes a result already obtained.

### 10.1 Why, and what is already known

The held-out and dementia runs found no confirmed gain in verdict accuracy (R2V − R2 = +1.3 and +3.4 points, intervals including zero). A look at the
per-verdict behaviour *after* those runs showed that all systems answer SUPPORTED far more often than the reviews do (dementia set: 62 to 70% predicted
against 25% gold) and almost never answer REFUTED (recall 3 to 6% for B0, B1 and R2), and that R2V raises REFUTED recall (dementia set 6% to 22%,
held-out 12% to 19%, from the committed answers) and the macro-F1 (R2V − R2 = +0.080, interval +0.024 to +0.135, and +0.045, interval +0.010 to
+0.080; `python -m experiments.medchange.class_balance`, `results/class_balance.*`). The criteria-only control R2C shows a similar gain on the dementia set
(R2C − R2 = +0.088; R2V − R2C = −0.008, interval −0.055 to +0.037) but not on the held-out set (R2C − R2 = +0.010; R2V − R2C = +0.035, interval −0.002 to
+0.072), so part of the gain may come from reading the evidence by criteria, not from the verification step. That metric was chosen after the data were seen and no correction was made for the intervals looked at, so it is a hypothesis, not a result.
This section tests it on questions that played no part in choosing it. Whatever the outcome, it is reported.

### 10.2 Hypothesis and decision rule

*Primary (H1).* On fresh questions, the macro-F1 of R2V is higher than that of R2. Macro-F1 is the unweighted mean of the F1 of the three verdicts
(an F1 is 0 when the verdict is never predicted correctly). The paired difference R2V − R2 has a 95% interval from a bootstrap over questions
(10,000 resamples, seed `fresh-macro-f1`, questions as units). **H1 is confirmed if the lower end of the interval is above 0.** Otherwise it is
reported as "positive but not confirmed" (point estimate above 0) or "no evidence" (point estimate at or below 0), with the upper end stated: if
it is below 0.02, a gain of 0.02 or more is excluded.

*Secondary (reported with intervals; no confirmatory claim).* REFUTED recall and the predicted-SUPPORTED share of R2V and R2; verdict accuracy of
R2V − R2 read by the three rules of `evaluation.md` §5 (the 1-point requirement keeps its meaning and is not re-read from this test alone); changed
**R2V − R2C macro-F1 (does the verification step add anything beyond reading the evidence by criteria?)**; changed
versus unchanged questions if both occur; the share of answers changed by R2V and how many changes fix or break an answer.

*Alzheimer's disease (descriptive, pre-specified).* The fresh pool holds practically no Alzheimer's or dementia questions (§10.3). The 208 dementia-set questions (48 name Alzheimer's disease) were seen and helped form H1, so
they are not part of the confirmatory family. Their macro-F1 difference and interval are reported next to the fresh result, with the statement
whether the sign agrees. No claim is made that H1 is confirmed for Alzheimer's disease.

### 10.3 Questions

A question is *fresh* when neither its study group, nor its Cochrane ID, nor its wording appears in any split of `benchmark.jsonl` (dev, confirm, ad)
and its wording does not name dementia or Alzheimer's disease (those were all used). The count was made by `python -m experiments.medchange.fresh_supply` (counts only;
`results/fresh_supply.json`): 16,501 MedRevQA rows, 970 used, 10,672 fresh, all of kind unchanged; **7,880 from 2010** (6,577 reviews; SUPPORTED 38.1%,
REFUTED 16.9%, NOT ENOUGH INFORMATION 45.1%), 4,798 from 2015, 9,695 from 2005. The held-out and development reviews have a median year of 2014 and 79
to 83% are dated 2010 or later, which is why the threshold is 2010. *No Alzheimer's stratum:* of the fresh questions only 7 (from 2010; 9 from 2005)
name Alzheimer's disease or dementia anywhere in the question, objectives or conclusions, so a fresh Alzheimer's-domain test cannot be built from this
source; the fresh test is general medicine. The review must be dated from **2010-01-01**; if fewer than the sample size qualify, all qualifying questions are used and the power
is stated, and the date is not moved earlier than 2005-01-01. From the qualifying questions the first **N** in the order of the SHA-256 of
`fresh-v1|<MedRevQA row>` are taken; no label or topic is used. **N = 1,500** (decided by the researcher on 2026-10-09, on the recommendation to plan for a true gain smaller than the one observed). Power for H1
with N = 1,500 (standard error of the difference about 0.41 divided by the square root of N, from the two intervals above; normal approximation):
97% if the true gain is 0.04, 81% if 0.03, 47% if 0.02; a gain of 0 is "confirmed" in about 2.5% of tests; the smallest gain confirmable with 80%
power is 0.030. For comparison N = 1,200 gives 92%, 72% and 39%, and N = 2,000 gives 91% at 0.03.

### 10.4 Systems and settings

The frozen design (settings hash `1743afd7045b`, `results/rag2_design.json`), unchanged: the arms R2, R2C and R2V (R2V verifies R2's draft; R2C is the criteria
control; B0, B1, R2V-ND and the directness judge are not run in this test). Every question is analysed whatever happens to it (an unparsed answer counts as wrong; a
question without admitted abstracts is analysed as the pipeline answers it). A defect found while running is fixed and recorded, never tuned on outcomes.
The split name `fresh` is registered in the pipeline (a mechanical change, tested, that does not touch the other splits); the as-of records and the abstracts
are fetched as for the other splits, the MedCPT candidate pools of the earlier splits are not frozen (the R2 candidate lists are built by `rag2_run lists`), and the
analysis is the blinded `analyze_fresh` (§10.2), which computes nothing until all questions have answers for all three arms.

### 10.5 Cost

From the measured steps of the dementia run (records, pools, rationales, candidate lists, filter and the R2 and R2V answers), about 3.9 minutes per
question for R2 and R2V plus 52 seconds for R2C, 4.8 minutes [estimate]: about 120 hours for 1,500 questions (96 for 1,200, 160 for 2,000), on the
laptop, resumable; the abstracts-only step saves the MedCPT pool freezing of the earlier splits (about 50 seconds per question, an estimate from the
held-out run), so about 100 hours is likely.

### 10.6 Limits stated in advance

(1) Macro-F1 was chosen after exploratory looks; this test is what makes it a result or not. (2) The fresh set is general medicine, not Alzheimer's
disease; a confirmed H1 would support the mechanism, not an Alzheimer's-specific gain. (3) The reference verdicts are model-made (an independent model
reproduces 81% of them); a fresh label audit of a sample is part of the report. (4) The true gain may be smaller than the one observed, because the
observed one was selected; the sample size is set with that in mind. (5) The reviews are older than the generator's training cutoff, so the model may
have seen some conclusions; the as-of rule limits only the evidence it is shown.

### 10.7 What may not change after the freeze

The hypothesis, the primary metric and decision rule, the sample size, the selection rule and seed, the arms, every setting and prompt. Not tried: other
metrics as primary, other thresholds, other seeds, dropping questions after seeing answers.

### 10.8 Freeze procedure

(1) The counts of `fresh_supply` are committed (done: 7,880 qualifying questions). (2) N = 1,500 is fixed here; `python -m experiments.medchange.fresh_benchmark
--medchange-dir ..\MedChange` selects the questions, writes them to `benchmark.jsonl` and writes `manifest_fresh.json` (the selected MedRevQA rows and the hash of the item
list), which is committed and pushed before any answer exists. (3) The tests of the new split pass and the design record equals the current design (the pipeline
checks it). (4) `python -m experiments.medchange.fresh_benchmark --freeze` marks this section IN FORCE (it refuses unless the manifest is on origin/main); the change is
committed and pushed. (5) A random sample of 300 questions (the first 300 in the order of the SHA-256 of `fresh-audit-v1|<item id>`) has its reference verdicts
audited by an independent model (`label_audit --split fresh --sample 300`); this uses no answer. Only then is the run started: `rag2_pipeline fresh --go` refuses
to start unless this section is IN FORCE, the manifest is on origin/main and the items match it.


