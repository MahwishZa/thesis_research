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
| Alzheimer's-specific primary evaluation (AD-KQA) | re-declared 2026-10-09 (§8); Stage 0 was no-go under the original criterion twice and is go under the amendment of 2026-10-09 (four covered areas, 200 to 300 test questions); the specification is fixed before any question is drafted; no question exists |

Since 2026-10-09 the primary evaluation of the research is Alzheimer's-specific (§8). The questions, systems and results in
§1 to §7 and in `evaluation.md` §6 are those of the realigned study on the as-of Cochrane benchmark (general medicine); they
are completed work and now serve as secondary evidence for that aim.

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

**2026-10-09, primary evaluation re-declared: an Alzheimer's-specific question set (AD-KQA, working name).** Decided by the
researcher (the supervisor has left research decisions to the researcher) after the held-out and dementia results of every
system were known; no AD-KQA question, label or output exists. *Reason (scope, not outcome):* the research is about
Alzheimer's disease, but the as-of Cochrane benchmark is general medicine and the dementia set is dementia-wide (48 of its 208
questions name Alzheimer's disease), so neither can support Alzheimer's-specific claims. Their results stay, unchanged, as
completed secondary and exploratory evidence (`evaluation.md` §6.0). Existing datasets were checked first (counts verified in
the released files unless marked): the Alzheimer's subsets of exam benchmarks (ADQA 446 items, 297 naming Alzheimer's; ADRD-Bench
1,438 index entries and 149 caregiving items) are multiple choice or true/false with no evidence for retrieval evaluation and no dates;
MedQuAD has 47 Alzheimer's pairs and PubMedQA 4 of 1,000; BioASQ has expert answers and expert-selected evidence but its
Alzheimer's share could not be counted (registration needed) and most of its questions need answer types the frozen design does
not score. None combines Alzheimer's focus, a verdict task, dates and evidence, so a controlled set is built from authoritative
sources instead, and the datasets above are not pooled with it.

*Design* (as declared; sizes and areas amended below). About 300 test and 60 development questions on Alzheimer's disease, built from PubMed systematic reviews,
meta-analyses and guidelines (MeSH major topic Alzheimer Disease; MEDLINE-indexed; English; abstract with a conclusion; not
retracted) published from 2023-04-01, after the generator's stated knowledge cutoff (March 2023), one question per source record.
Each question is made from a fixed template out of terms copied from the abstract and has a three-way verdict (SUPPORTED, REFUTED,
NOT ENOUGH INFORMATION) as of the source's publication date; retrieval uses only earlier records and never the source itself.
Areas, sampled by MeSH qualifiers: treatment, prevention, diagnosis, causes and risk factors, progression, symptoms, care and
management; background/history and terminology are added only if the trial run passes. *Reference answers:* the verdict and the
key sentence of the source conclusion (stored as offsets and a hash, never as text). A question is kept only if every term is
found in the abstract, a rule-based reading of the conclusion and the drafting model (Qwen2.5-7B-Instruct) agree, an independent
verifier that is neither the drafter nor the generator under test agrees when it reads only the quoted sentence, and an audit pass
confirms it; agreement and kappa are reported as for the other splits. The generator under test never drafts, labels, screens or
selects questions. *Splits:* development and test by source record and topic with a seeded hash; no question is added or removed
because of any system's answer.

*Frozen design.* Prompts, settings and arms are those of the design record (settings hash `1743afd7045b`); AD-KQA is a test of the
frozen design on new data. Only the split registry, the exclusion of the question's own source from retrieval and the audit
prompt's input change, none of which affects the earlier splits.

*Requirement.* Unchanged: R2V at least 1 percentage point of verdict accuracy above R2, read by the three rules of
`evaluation.md` §5. *Power (computed from the discordance of 14.9% to 16.4% observed in all three completed runs):* with 300 test
questions the standard error of a paired difference is about 2.3 points, so only gaps of about 6 points can be confirmed with 80%
power; a true gain of +1 point would be confirmed with probability about 7%, +1.5 points 10%, +3 points 26%, +5 points 60%; a
point estimate of at least +1 point has probability 33% when the true gain is zero. The requirement can therefore be read as met
as a point estimate, not confirmed, or as not met, and no conclusion beyond the reading will be drawn.

*Independent verifier.* A third-family model: Phi-3.5-mini-instruct (MIT licence; about 2.2 to 2.4 GB at Q4_K_M, a size reported
by the quantisers; chat template with a system turn), used only if it passes a qualification run that uses the existing label audit
unchanged on the development split in a separate data folder: agreement with the gold labels at least 75%, kappa at least 0.60,
unparsed answers at most 2%. If it fails, Mistral-7B-Instruct-v0.3 (Apache-2.0; about 4.4 GB) is tried on the same terms; a
second download needs a new approval. Without a qualified verifier the fallback is the drafting model with a different prompt,
which is weaker independence and is then reported as a limitation.

*Verifier qualification, run 1 (2026-10-09, Phi-3.5-mini-instruct Q4_K_M, SHA-256 `3EF53267...BB03BCA` as downloaded; the existing label audit on the
development split, 226 items, replies limited to 12 tokens):* **not qualified as run.** The audit's strict pattern ("LABEL: <verdict>")
read 100 of 226 replies; 126 (55.8%) were unparsed against the limit of 2%. On the 100 read replies agreement was 85.0% and kappa
0.7424, which says little about the other 126. The first eight unparsed replies were inspected: each begins with a verdict word
without the "LABEL:" prefix (for example "SUPPORTED: ..." or "REFUTED; ..."), so the fault looked like a reading rule, not quality.
*Correction, declared before the replies were re-read:* a reply is read as its verdict if it contains "LABEL: <verdict>" or if its first
non-empty line begins with the verdict word (`spec.parse_verdict`); nothing is guessed from the rest of a reply. The same rule is
used by the builder's verification step. The stored replies (first 60 characters) are re-scored without calling the model
(`python -m experiments.adkqa.qualify`), with the thresholds unchanged (agreement at least 75%, kappa at least 0.60, unparsed at most
2%). The correction was designed after seeing run 1 and eight of its replies, so the re-score is a repaired instrument on the same
data, not an independent test; its result is recorded here. If it fails, the fallback is Mistral-7B-Instruct-v0.3 on the same terms.
*Result of the re-score (2026-10-09, `results/adkqa_verifier_qualification.json`):* **qualified.** All 226 replies were read (0 unparsed;
limit 2%); agreement 77.9% (limit 75%); kappa 0.6641 (limit 0.60). By gold class: SUPPORTED 81.9% (105 items), REFUTED 87.8% (49),
NOT ENOUGH INFORMATION 65.3% (72). The margin over the agreement limit is 2.9 points, and NOT ENOUGH INFORMATION is the weak class; for
comparison Qwen2.5-7B reached 83.2% and kappa 0.7422 on the same split. A kept question needs the verifier to agree with the drafter
and the cue reading, so a verifier that errs more removes more questions; it does not make a kept question less valid.

**Amendment B of 2026-10-09 (conclusion rule and development share), after the first `prepare` and `diagnose`.** Decided by the researcher
on counts only: no abstract text beyond section labels and the first words of last sentences was read, and no draft, verdict or system output
exists. *Why:* of 710 source records 292 (41%) had no conclusion under the rule above: 245 are abstracts without section labels and 47 are
structured without a conclusion label (most with a discussion section); most unlabelled abstracts end with a statement about future
research, so the marker rule rejected conclusions that exist. The development pool was 110 records, fewer than the 150 the trial run
drafts, because a few large topic clusters (the largest, 115 records, hash position 0.93) fell to test. *Changes:* (B1) the conclusion is
the section labelled conclusion(s), interpretation or authors' conclusions; else the last three sentences of a section whose label
begins with "discussion" or is "implications"; else, for an abstract with no section label at all, its last three sentences (the
repository's convention for unlabelled abstracts, `abstracts.key_text`); otherwise the record gives no question. The cue reading,
the drafter and the verifier must still agree, so a wrong span costs a question and cannot add one. The verifier's input is that span.
(B2) The development share starts at 0.25 and is raised by 0.05, up to 0.60, until the development pool holds at least 150 records
(`spec.dev_fraction`; depends on cluster sizes only; seed and hash unchanged, so every earlier development cluster stays). *Checked
before applying:* the ten largest clusters have hash positions of 0.61 or more, so the rise cannot pull a large cluster into
development at once; the development pool grows from the small clusters, to about 150 records at a share of about 0.35 to 0.45
(expected, to be read from `prepare`), leaving about 530 to 560 test records. *Unchanged:* 150 drafted, 60 kept, the 40% survival
gate, every other gate and threshold. *Costs:* wider spans meet hedged wording more often, so fewer drafts will survive; a larger
development share shrinks the test pool, where 40% survival would give about 215 to 225 questions against a minimum of 200.

**Amendment C of 2026-10-09 (short-form templates), after a look at five development drafts.** The first five drafts of the development split
(text of the replies only; no verdict of any system, no retrieval, no answer) showed a grammatical defect of the templates: when a copied span
already names Alzheimer's disease, the closing phrase names it twice (for example "... on Alzheimer's disease in people with Alzheimer's
disease?"), and no automatic check caught it. *Change:* when `x` or `y` contains "alzheimer" (case-insensitive) the question uses the short form of
its template, without the closing phrase:
- `Is {x} effective for {y}?`
- `Is there any effect of {x} on {y}?`
- `Can {x} be used for {y}?`

Every fixed word of the short forms is a stop word of the frozen query builder, as before. In the short form the query takes "alzheimer" from the
span, which must lie within the first 8 query terms; the limit of 6 content words copied, the length of a span and the rest of the rules are
unchanged. The drafting prompt is unchanged, so the five drafts stay valid. The same five drafts also showed what the gates will have to
measure and are not changed here: only 2 of 5 cleared the length rules, and all 5 proposed SUPPORTED. This amendment is not the one redesign that
gate 1 allows.

*Gates (fixed before any question is drafted). They concern the source records and the quality of the questions; the one exception, the closed-book headroom check on the 60 development questions, is made once for the whole set and never removes a question.*
(0) Stage 0 (`python -m experiments.adkqa.stage0`) [original wording, replaced by the amendment below: at least 700 eligible source records in the window and at least 100 in each
of at least 6 core areas; areas below 100 are reported as not covered]. (1) Trial run of 60 development questions: at least 40% of
drafts survive the automatic checks; verifier agreement at least 85% on kept questions; a claim-only classifier (5-fold) at most
the majority class plus 5 points; each verdict at least 25%; the same seed gives the same manifest hash; a median candidate pool of
at least 15 abstracts and at most 5% empty pools; closed-book B0 between the constant answer plus 5 points and 80% on the
evidence-sensitive areas (a ceiling area is reported, not dropped). One documented redesign is allowed, only before the test
split is sealed. (2) Full build: at least 300 test questions [replaced by the amendment below: at least 200], at least 25 per covered area, audit agreement at least 85% on test
questions, manifest hash committed before any system sees a test question. (3) Run: parse rate and verifier validity at least
95%; a defect is fixed and recorded, never tuned on outcomes. A failed gate stops the build and is reported.

*Stage 0, run 1 (2026-10-09, `results/stage0_counts.json`): no-go under the criterion above.* 866 eligible records in the window
(2023-04-01 to 2026-10-09). Distinct records per area: treatment 314, prevention 0, diagnosis 278, causes_risk 216, progression
58, symptoms 171, care_management 20; four areas reached 100, six were required. The prevention count of 0 was an instrument
error: the qualifier was written "prevention & control", which PubMed did not find (its translation record showed it). Europe PMC
reference lists: 19 of 20 sampled records had at least 10 (threshold 60%). *Revision 2 of the instrument* corrects the prevention
qualifier ("prevention and control"), drops PubMed's routine messages from the recorded warnings, and adds information-only
counts that do not enter the criterion (records naming Alzheimer's only in the title, the pre-cutoff window, records in no area,
and the supply when each record serves one area only). The thresholds (700, 100, 6) and the window are unchanged. One corrected
re-run is made (`results/stage0_counts_r2.json`); what follows a second no-go (fewer covered areas, a wider window flagged by
`post_generator_cutoff`, a smaller test set, or stopping) is decided by the researcher and recorded here.

*Stage 0, run 2 (2026-10-09, `results/stage0_counts_r2.json`, instrument revision 2): no-go again under the unchanged criterion.*
866 records. Distinct per area (qualifier frame, the gate): treatment 314, prevention 21, diagnosis 278, causes_risk 216,
progression 58, symptoms 171, care_management 20; covered (at least 100): treatment, diagnosis, causes_risk, symptoms (4 of the 6
required). Information only: with one question per record the qualifier-frame supply is 162 / 21 / 185 / 143 / 57 / 144 / 20
(in the four covered areas the distinct records number 710; the figure 634 first given here was an allocation that gave records shared with the uncovered areas to those areas, and is withdrawn); records in no qualifier area 131; the pre-cutoff window 2021-04-01..2023-03-31 holds 525 further
records. The heading frame (MeSH main headings instead of qualifiers) was recorded for information only; using it as the gate would
be a change of criterion after the counts were seen and would have to be declared as such. Nothing is built. The next step is the
researcher's decision, recorded here.

**Amendment of 2026-10-09 (after Stage 0 run 2): the areas covered, and the thresholds that depend on them.** Decided by the
researcher on the counts of run 2, which had been seen; it is therefore a scope decision made after the counts, not an independent
test of the original criterion, and the original criterion stays recorded as failed twice. *What changes:* (a) Only the core areas
that reach 100 distinct source records are covered: treatment (314), diagnosis (278), causes and risk factors (216) and symptoms
(171). Prevention (21), progression (58) and care and management (20) are reported as not covered, and no claim about them is made;
nothing is added to them from dementia-wide sources, from the pre-cutoff window (525 records) or from the MeSH-heading frame, which
was counted for information only. The optional strata cannot be added: background/history has 0 records and terminology 3.
(b) Stage 0 criterion: at least 700 eligible records (unchanged: 866), at least 100 distinct records in each covered area
(unchanged), **at least 4 covered areas (was 6)**, and **at least 650 distinct records in the covered areas (new: 200 test + 60
development questions at the 40% draft survival of gate 1 is 260 / 0.40)**. The covered areas hold 710 distinct records, so Stage 0
is go under the amended criterion (`python -m experiments.adkqa.stage0 --recheck experiments/adkqa/results/stage0_counts_r2.json`).
(c) Test set: **at least 200 test questions (was 300); every question that passes the gates is kept, up to 300**; at least 25 per
covered area (unchanged); 60 development questions (unchanged); the test and development questions share the 710 records, one
question per record, a record in two covered areas serving the scarcer one. (d) Unchanged: the 3-way verdict, the as-of rule, the
frozen design, the requirement of 1 point, the verifier and its qualification thresholds (75%, 0.60, 2%), the trial-run gates (40%
survival, 85% verifier agreement, claim-only classifier at most the majority class plus 5 points, each verdict at least 25%, the
same seed gives the same hash, median pool at least 15 abstracts and at most 5% empty, B0 between constant plus 5 points and 80%),
the audit agreement of 85%, and the run validity of 95%. *Reach:* the evaluation covers treatment, diagnosis, causes and risk
factors and symptoms of Alzheimer's disease; it does not cover prevention, progression or care. *Power (normal approximation to the
paired difference, discordance 15%, recomputed for all sizes with one formula; it replaces the figures above, which differ by one or
two points):* with 200 test questions the standard error is about 2.7 points and gaps of about 7.7 points can be confirmed with
80% power; with 300 it is 2.2 points and 6.3. A true gain of +1 point is confirmed with probability 7% at both sizes, +1.5 points
9% to 10%, +3 points 20% to 27%, +5 points 45% to 62%; a point estimate of at least +1 point has probability 36% (200) or 33% (300)
when the true gain is zero. The requirement therefore cannot be confirmed at either size unless the true gain is several points;
the reading rules are unchanged.

**Amendment of 2026-10-09 (templates, reading cues, split and source freeze), fixed before any question is drafted.** Nothing
below was tuned on an abstract, an answer or a system output; no abstract has been read. The same values are in
`experiments/adkqa/spec.py`, and a test compares this text with that file. A change after the trial run is allowed only as the one
documented redesign of gate 1, before the test split is sealed.

*Templates.* A question is one of three fixed templates filled with two spans copied from the abstract (`x`, `y`):
- effect of an intervention: `Is {x} effective for {y} in people with Alzheimer's disease?`
- effect or association of a factor, exposure or biomarker: `Is there any effect of {x} on {y} in people with Alzheimer's disease?`
- usefulness of a test for a diagnostic purpose: `Can {x} be used for {y} in people with Alzheimer's disease?`

The drafting model picks the template and the spans and proposes the verdict; it writes no other question text. Each span has 1 to
8 words, must occur verbatim (case-insensitive) in the title or abstract, and `x` and `y` together may have at most 6 content words
(words the retrieval query builder does not drop). *Why these words:* every fixed word of the templates is a stop word of the
frozen query builder (`pubmed_asof._STOP`) except "alzheimer" and "disease", which come last, so the retrieval query is the copied
terms plus those two and no filler word restricts it; a draft whose query would lose "alzheimer" or "disease" within the first 8
terms is not made. Reviews that report only prevalence, incidence or other descriptive results state no claim and give no question.
The drafter's prompt is committed and hashed in the manifest before the first development draft.

*Verdicts.* As for the other splits (the audit prompt's definitions): SUPPORTED means the authors' conclusion at least partially
supports the claim in the question (an effect, an association, a useful test); REFUTED that it at least partially does not (no
effect, no association, similar to placebo, poor accuracy); NOT ENOUGH INFORMATION only when the authors state that the studies or
the evidence are insufficient, not when the certainty is low. *Conclusion:* the abstract section labelled conclusion(s),
interpretation or authors' conclusions; for an unstructured abstract its last sentence, only if it begins with a conclusion marker
(in conclusion, we conclude, overall, taken together, these/our results/findings, this review/meta-analysis/study); otherwise the
record gives no question [replaced by amendment B below]. *Objectives* for the audit and verifier prompts: the section labelled objective(s), aim(s), purpose or
background, else the first sentence.

*Reading cues (rule-based).* The conclusion is lower-cased; negative phrases are removed first so that "not effective" is not also
read as positive; the verdict is the single class that has a hit; a hit in two classes, or any hedge cue, gives no verdict and no
question. These are regular expressions as in `spec.py`:
```
negative:
  no (statistically )?significant (effects?|differences?|improvements?|benefits?|associations?|reductions?|changes?)
  not (significantly )?(effective|beneficial|associated|superior|supported|useful|accurate)
  (did|does|do) not (improve|reduce|show|demonstrate|differ|prevent|slow|affect|increase|decrease)
  (no|without) (clear |apparent |definite )?(benefits?|effects?|differences?|associations?|advantages?|improvements?)
  ineffective
  failed to
  similar to placebo
  not recommended
  (poor|low) (diagnostic )?(accuracy|performance|sensitivity|specificity)
positive:
  significantly (improved?|improves|reduced?|reduces|increased?|increases|enhanced?|slowed?|delayed?|lower|higher|better)
  \b(effective|efficacious|beneficial|benefits?)\b
  associated with (a |an )?(higher|increased|lower|reduced|greater|decreased|elevated|increase|decrease|risk)
  (high|good|excellent|acceptable|promising) (diagnostic )?(accuracy|sensitivity|specificity|performance)
  positive (effects?|associations?)
  superior to
insufficient:
  insufficient (evidence|data)
  (evidence|data) (is|are|remains?) (insufficient|lacking|scarce|inconclusive)
  not enough (evidence|data|studies)
  inconclusive
  too few (studies|trials)
  lack of (evidence|data)
  no (firm|definitive) conclusions?
  (cannot|can not|could not) be (drawn|made|determined)
  remains? unclear
  unclear whether
  no (eligible|relevant) (studies|trials)
hedge:
  \bhowever\b
  \bmixed\b
  \binconsistent\b
```
They were written from the audit prompt's definitions and from general wording, not from the abstracts of this set. Their coverage
(the share of drafts for which the rule gives a verdict) is a secondary measure of the trial run; a low coverage lowers the
survival rate, and gate 1 then decides.

*Topic cluster and split.* The cluster of a record is its first major-topic MeSH descriptor other than Alzheimer Disease
(alphabetical, case-insensitive), or the record itself when it has none, so reviews of one intervention or topic fall in one
split. A cluster is development if the first 8 hexadecimal digits of the SHA-256 of `adkqa-split-v1|cluster|<cluster>`, read as a
fraction, are below 0.25 (25%), and test otherwise [the share is raised by amendment B below when the pool is too small]. The seed is `adkqa-split-v1`; no other seed is tried, and the pool sizes are
recorded before any draft. Development records are drafted in the order of the SHA-256 of `adkqa-split-v1|draft|<pmid>`: the first
150, and the development set is the first 60 of them that are kept. Gate 1 is therefore stated on these 150 drafts: at least 40%
survive, which is the 60 needed. Test records are not read or drafted until gate 1 has passed. Test: every record of the test pool
is drafted; all that pass the gates are kept, at least 200 and at most 300 (if more than 300 pass, the first 300 in the order of the
SHA-256 of `adkqa-split-v1|keep|<pmid>`).

*Source freeze.* The source records are the identifiers in the covered areas of `experiments/adkqa/results/stage0_counts_r2.json`
(search of 2026-10-09: 2023-04-01 to 2026-10-09); the builder reads that file and does not search again, so records indexed later
are not added. The date of a source is the publication date PubMed reports (`sortpubdate`), with a missing month or day set to 01 as
for the other splits, and retrieval admits only records dated strictly earlier. The candidate pools are frozen once, with the date
recorded, before any system sees a test question.

*Secondary measures (reported, never gates; no new test).* Per covered area: accuracy of each arm with its interval; retrieval
recall of the source's own reference list (the probe found 19 of 20 sampled records with at least 10 references, above the 60%
needed); rule coverage; share of records excluded for having no conclusion; parse rate. Per-area differences are descriptive, with
no p-values and no claim.

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
