# Experiment plan: as-of evaluation on Cochrane verdict changes

This is the protocol for the primary experiment. **Stage 1** (recency-aware admission) was fixed in
code and text before any answer was generated; its dev result was negative. **Stage 2** (an
evidence-synthesis layer, §1 and §4–§10) was fixed on 2026-10-03, after the stage-1 dev results and
**before any confirmatory pool, answer or stance output existed** (see §13 for every amendment, each
dated, and for the list of decisions taken after seeing dev data). Status labels used throughout:
**done** (run, result recorded in `log.md`), **built** (code and tests exist, not yet run on
the full data), **planned** (not implemented). Numbers are *measured* (this project's runs),
*computed* (from files in this project) or *estimated* (derived from a measurement).

## 1. Question and what can be claimed

**Stage 1 (done; dev split only).**

> On medical questions whose Cochrane verdict changed between review versions, asked as of the
> newest review's publication date with only evidence published before that date, does
> recency-weighted evidence admission make a local LLM give the current verdict more often than
> standard RAG, a helpfulness-ranked admission, and a published-style recency reranking, without
> lowering accuracy on questions whose verdict did not change?

Answer on dev (`log.md` Phases 24–26): no. The mechanism works (recency raised the share of admitted
passages from the update window from 51% to 74%) but did not improve verdicts (gate G3 failed:
P − B2 = −1.3 pp, P − C1 = −2.0 pp; B3 − B1 = −4.0 pp, p = 0.38). This result is kept in the thesis
as a first finding; it is not re-tested on the confirmatory split.

**Why stage 2** (all *computed* on the 226 dev items, from the committed answers; see `log.md`
Phase 27). (a) Verdict accuracy here mostly measures how readily the model says SUPPORTED versus NOT
ENOUGH INFORMATION: always answering SUPPORTED scores 44.4% on changed items against 49.7% for the best
arm, and recall of REFUTED is 0–11% for every arm. (b) Retrieval's gain comes with a shift towards NOT
ENOUGH INFORMATION (recall 15% for B0, 53% for B1) at the cost of SUPPORTED recall (81% → 69%). (c)
Reading five abstracts at once is noisy: the same five papers in a different order change 14% of
verdicts (86.1% agreement, 72 pairs). (d) Recency carries no information about the gold verdict in this
data: the mean age of B1's evidence is 10.6, 11.4 and 10.7 years for gold SUPPORTED, REFUTED and NOT
ENOUGH INFORMATION, and adding age features to the verdict lowers cross-validated accuracy (52.7% →
50.1%). (e) Recalibrating hard verdicts (52.7% vs 53.1% raw) and majority voting over evidence sets
(47.0% vs 49.7%) gave no gain.

**Stage 2 (planned; two pre-specified questions, each tested once on the confirmatory split).**

> **RQ1.** Does giving a small local LLM as-of retrieved evidence (standard RAG, B1) make its verdicts
> more accurate than giving it none (B0)?
>
> **RQ2.** Does an evidence-synthesis layer — one narrow stance judgement per retrieved paper
> (supports / contradicts / neither), summarised into four numbers, optionally weighting newer and
> stronger study types more, and combined with the RAG answer by a small fixed logistic regression —
> raise verdict accuracy over the same RAG answer passed through the same fitting procedure (B1R)?

RQ1 is a replication of a dev difference (+8.6 pp, p = 0.06) and is the study's most likely positive
result. RQ2 is the proposed system; its *recency and study-type weights* are tested only as ablations
(§4), because stage 1 and (d) give no reason to expect recency to help.

Recency-aware retrieval and conflict handling in RAG are crowded areas (TempRALM; AionRAG;
FRESCO; ConflictRAG; EvoTrustRAG; "Contradictions in Context / Toward Safer RAG in
Healthcare", arXiv 2511.06668; DriftMedQA, EMNLP 2025 Findings), and per-passage stance
classification and logistic stacking are standard techniques. **No novelty is claimed for any
single ingredient.** What this work can contribute, if the results hold, is a controlled as-of
evaluation on real verdict changes with falsification controls, a measurement of how a small local
model uses retrieved evidence (largely through abstention), and an honest test of whether decomposing
evidence reading helps. A search for an identical study was not exhaustive (*unverified*). No claim
about general hallucination reduction is made.

## 2. Benchmark (`experiments/medchange/`, **done**)

Source: MedChange (Vladika et al., EMNLP 2025 Findings), cloned locally; its text is not
redistributed (no licence is stated). `build_benchmark` rebuilds MedChangeQA from
`MedRevQA.csv` + `AllStudyGroups.csv` by the authors' rule (lowest row index in a DOI group is the
newest version, checked against publication dates in 1,534 of 1,535 groups) and **refuses to
continue unless all 512 released items are reproduced with identical labels**. Every item thereby
carries both versions' dates and PMIDs, which the released CSV lacks.

| Property (computed) | Value |
|---|---|
| Changed items | 512 rebuilt; 8 excluded as label noise (conclusions near-identical, similarity ≥ 0.85); **504 usable** |
| Change types | 397 involve NOT ENOUGH INFORMATION; 114 are decisive SUPPORTED ↔ REFUTED reversals |
| Unchanged controls | 250 sampled (seeded) from groups whose versions all share one label |
| Splits (seeded 20261001, stratified by kind × change type) | dev 151 changed + 75 unchanged = **226**; confirmatory 353 + 175 = **528** (usable) |
| Gold labels | gpt-4o-mini labels of each abstract's conclusions: model labels, not human; ~100 to be human-checked |
| Dates | both versions dated for every item; 13 newest versions are year-only (cutoff set to Jan 1: conservative) |
| Alzheimer's-related | 9 changed + 5 unchanged: descriptive only |

Without retrieval, five released models (7B to DeepSeek-V3) give the current verdict on changed
items 49–50% of the time and the outdated one 25–32% (computed, `headroom.py`); size barely
matters, which points to a ceiling from label ambiguity. No review group or Cochrane ID appears in
both splits (computed: 0 of 762) and each review contributes one item; the median newest-review year
is 2014 in both splits. The newest reviews pre-date the generator's training data, so memorisation of
their conclusions is possible (*unverified*; it affects every arm alike).

## 3. As-of protocol and candidate pools (**done** for dev; **planned** for confirmatory)

For each item the question date t_q is the newest review's publication date. The system's only
input is the question. Candidates are PubMed records first public **strictly before** t_q
(`pubmed_asof`), Cochrane Database records excluded, so neither the review nor its versions can
appear. PubMed's publication-date filter matches the print *or* electronic date, so availability
is taken as the earliest known date, kept only if its upper bound precedes t_q; boundary-ambiguous
records are dropped, and a run aborts if more than 15% of a result set cannot be shown to precede
the cutoff. Abstracts are fetched, ranked by MedCPT dense similarity to the question, the top 50
reranked with the MedCPT cross-encoder, and the top 20 frozen with date bounds and an
order-sensitive hash (`freeze_candidates`). No label, verdict or generated text is read at this
stage. One pool per item is shared by every arm.

Gate G0 (headroom, **passed**, dev, n = 151 changed): 94.0% of changed items have a trial or
systematic review inside the update window among the as-of candidates (80.1% within the top 50;
median 72 in-window records); unchanged controls look the same (93.3%). Availability is therefore
necessary but not sufficient evidence that new studies carry the verdict change. Dev pools: 226
frozen, none empty, median size 20. The confirmatory pools (528 items) are **planned**; stage 2 needs
the probe and the freezing steps but **not** the helpfulness scores, because arms B2, P and C1 are not
run on the confirmatory split.

## 4. Arms and fixed settings (**built**)

Arms differ only in which ≤ 5 passages they admit, never in prompt, generator or decoding.
Settings are in `experiments/medchange/arms.py`, hashed into every answer record, and **not
tuned**: budget 5; recency T = 2^(−age/H), H = 1,095 days, age from the passage's availability
midpoint to t_q; relevance rank-normalised within the pool; λ = 0.5. The score
A = (1 − λ)·ρ + λ·T is computed by the reference implementation in `src/proposed/`
(`TemporalPolicy`, `AdmissionScorer`); the arms apply it to the top-5 by score with no θ
threshold. Passages are shown without dates in every arm.

| Arm | Relevance signal | Recency | Role |
|---|---|---|---|
| B0 | none (no evidence) | – | the model's own knowledge |
| B1 | MedCPT cross-encoder rank | no | standard RAG |
| B2 | zero-shot Flan-T5 P(yes), ranked | no | helpfulness-ranked admission |
| B3 | cross-encoder rank | yes | TempRALM-style recency reranking |
| **P** | zero-shot Flan-T5 P(yes), ranked | yes | proposed |
| C1 | as P, dates shuffled within the pool | fake | falsification control |

Stage 1 arms are run on the dev split only. B1/B2/B3/P form a 2 × 2 (helpfulness signal × recency);
the interaction asks whether the combination adds anything. **B2 and P are not RAG².** RAG²'s trained filter is not distributed and
the local retraining attempt failed (archived; `log.md` Phases 13–18). B2 uses the unmodified
Flan-T5-large asked RAG²'s prompt plus "Answer yes or no.", scored as P(yes), and ranks by it
rather than thresholding as RAG² does; RAG²'s rationale-as-query step is also dropped so that all
arms share one pool. A code-reading finding to be verified (diagnostics step P0, below): the
helpfulness prompt puts the question *after* the abstract and truncates inputs at 512 tokens from the
end, so for long abstracts the question and the "Answer yes or no" instruction may have been cut off.
The stage-1 conclusions about recency are unaffected (P − B2, P − C1 and B3 − B1 compare arms that share
the same scores), but the statement that the helpfulness score is a weak selector is confounded by it.

**Stage 2 arms (planned).** The evidence-synthesis layer works on the first 8 candidates of each
frozen pool in cross-encoder order and never changes what the generator reads.

| Arm | What it is |
|---|---|
| B0, B1 | as in stage 1 (no evidence; cross-encoder top-5) |
| **B1R** | B1's verdict passed through the same logistic fitting on dev as the hybrids (fairness control: same fitting, no stance information) |
| S0 | stance features only, all papers weighted equally |
| S1 / S2 / S3 | S0 with recency weights / study-type weights / both |
| H0–H3 | hybrid: B1's verdict (one-hot) plus the stance features of S0–S3 |
| H1C, H3C | H1 and H3 with publication dates shuffled within the pool (falsification controls) |

*Stance of one paper.* The model reads the question as a claim, then the paper's title and its RESULTS
and CONCLUSIONS sections (an abstract without labelled sections contributes its last three sentences; at
most 200 words, conclusion-biased), and answers one letter: the study supports the claim, contradicts it
(no benefit, harm or an opposite effect), or says nothing clear. The three probabilities p_sup, p_con,
p_nei come from the model's first-token distribution over the three letters (a hard-label fallback exists).
Two wordings with different letter orders exist for the pilot; the full runs use the one with the higher
hand-check accuracy (ties: wording A).

*Four features.* With paper weights w_i and W = Σ w_i:
f1 = Σ w_i (p_sup − p_con) / W (weighted signed stance);
f2 = Σ w_i p_nei / W (share carrying no clear stance);
f3 = 2·min(Σ w_i p_sup, Σ w_i p_con) / W (conflict);
f4 = ln(1 + Σ w_i (p_sup + p_con)) (informative evidence mass).
Weights: S0/H0 w = 1; S1/H1 w = 2^(−age/H) with H = 1,095 days and age measured to the question date t_q
only (the same `TemporalPolicy` as stage 1, never the previous version's date); S2/H2 w = 3 for
"Meta-Analysis" or "Systematic Review", 2 for "Randomized Controlled Trial", "Controlled Clinical Trial"
or "Clinical Trial", else 1 (PubMed publication types stored in the pools); S3/H3 the product.
No feature uses an item's kind, change type, previous-version date or label.

*The fitted layer.* A multinomial logistic regression (numpy only) on standardised features with a fixed
L2 penalty of 5.0 on the summed log-loss, intercept unpenalised, solved by Newton's method to convergence; **nothing is tuned**.
It is fitted on all 226 dev items (changed and unchanged, as in the evaluation) and frozen. The four
variants H0–H3 form the only forking budget: the *selected* hybrid is H0 unless another variant's dev
cross-validated accuracy exceeds H0's by at least 1.0 pp, in which case the best such variant is selected.
The selection, every fitted coefficient and the stance wording are written to one frozen model file and
committed **before** any confirmatory stance run.

## 5. Generation (**built**)

Llama-3-8B-Instruct, Q4_K_M GGUF (bartowski), llama.cpp on CPU, greedy decoding, 160 new tokens,
one prompt (`prompts.py`). The model must open with `VERDICT: SUPPORTED | REFUTED | NOT ENOUGH
INFORMATION` followed by at most three sentences. Measured on the target laptop: ≈ 18 s per answer
without evidence and ≈ 66 s with five passages, so ≈ 350 s per item over six arms: dev ≈ 22 h,
confirmatory ≈ 51 h (*estimated*). A second generator (Qwen2.5-7B-Instruct) is **planned** as a
robustness check.

Stage 2 reuses these generations: B0 and B1 are generated once on the confirmatory split (≈ 2.1 h and
≈ 8.9 h, *estimated*). The stance step uses the same Llama-3-8B-Instruct Q4_K_M model with a context of
1,536 tokens, temperature 0, one output token and the top-20 log-probabilities of that token
(`logits_all`, ≈ 0.5 GB extra memory); its cost per paper is *not measured* — about 5–8 s is expected
(≈ 250 prompt tokens at a prompt-processing speed of ≈ 40 tokens/s inferred from the B1 timings, plus overhead) — and is replaced by the pilot's
measurement. One declared fallback exists: if the pilot fails, the same pilot is repeated once with
Flan-T5-large (≈ 1 s per pair on CPU) as the stance model; a second failure ends stage 2's synthesis arm.

## 6. Outcomes

* **Primary: verdict accuracy** — the parsed verdict equals the gold newest verdict. No judge
  model: an answer without a parsable verdict counts as wrong and is reported separately.
* **Primary, stage 2: verdict accuracy on all confirmatory items** (353 changed + 175 unchanged = 528).
  Pooling is chosen, before any confirmatory data exists, because it raises power and the unchanged
  items are legitimate benchmark items; the changed-items-only accuracy is reported beside it as the
  key secondary.
* **Key secondary: outdated-verdict rate** — the verdict equals the previous version's (changed items).
* **Stage 2 diagnostics:** recall for each gold class, macro-F1 and the NOT ENOUGH INFORMATION rate (the
  main behaviour retrieval changes); the stance step's hand-checked accuracy, wording agreement and
  irrelevant-paper control (§9).
* **Safety: accuracy on unchanged items** — non-inferiority margin 5 pp.
* **Retrieval-level (manipulation checks, never outcomes; built, `analyze.py`):** share of admitted
  passages that surely first appeared inside the update window, items with any such passage, mean
  passage age, overlap with B1's admitted set.
* **Hallucination / faithfulness (planned):** blinded human annotation of claims against a common
  reference (gold conclusion plus the shared pool) on a stratified subset, using
  `evaluation/annotation.py`; reported with coverage.
* Sensitivity (planned): the 114 decisive-flip items alone; an LLM judge on a sample; excluding
  non-Cochrane systematic reviews from candidates.

## 7. Statistics (**built**: `analyze.py`, `evaluation/stats.py`)

**Stage 1 (dev, exploratory).** Family, Holm-corrected: P vs B1, P vs B2, P vs B3 on the primary
outcome — exact McNemar with a question-resampled bootstrap 95% CI.

**Stage 2 (confirmatory, run once).** Primary family, Holm-corrected at family-wise α = .05 (two-sided):
**RQ1** B1 vs B0 and **RQ2** the selected hybrid vs B1R, on all 528 items, exact McNemar with a paired
bootstrap CI. Because RQ1 is expected to be strongly significant, Holm leaves RQ2 close to α = .05. Secondary
family (Holm among themselves; descriptive): the selected hybrid vs B1; vs H0 (do the weights add?); vs its
date-shuffled control H1C/H3C (are real dates specific? only if the selected variant uses recency); S0 vs B1
(does stance alone match reading?); and the changed-items-only versions of RQ1 and RQ2. A result is called
*confirmed* only when the Holm-adjusted p < .05 **and** the 95% CI excludes 0; otherwise the estimate and
interval are reported as not confirmed. Single-arm accuracies carry Wilson 95% intervals. Also planned: a mixed-effects logistic model
`correct ~ helpfulness × recency + (1 | item)` (the interaction is the contribution test), the
changed-vs-unchanged interaction, one-sided non-inferiority on unchanged items, and P vs C1.
Power (*simulated*, exact McNemar, α = .05). Stage 1 assumed ≈ 30% of answers differing between two
arms: 353 changed items gave 84% power at Holm-corrected α (92% at α = .05) for a true 10 pp difference,
61% (76%) for 8 pp and 33% (50%) for 6 pp. Stage 2, 528 items: a hybrid layer differs from the RAG answer
it refines on perhaps 15–25% of items, so a true +3 pp is detected 26–38% of the time, +4 pp 42–62% and +5 pp
61–82%; two unrelated methods (35% differing) detect +5 pp 46% of the time and +8 pp 86%. **This benchmark can
only confirm effects of about 5 pp or more; a real gain of 1–3 pp would read as "suggestive, not confirmed".**
Expected outcome (*judgment, not a computed fact*): RQ1 is confirmed with probability ≈ 58% (prior on the true
effect N(+5.5, 3.5) pp, 29% differing); RQ2 is confirmed with probability ≈ 15% (range 8–25%; scenario
weights 50% no effect, 25% +1 pp, 15% +3 pp, 10% +5 pp). Smaller effects will read as inconclusive and are
reported with their intervals, not as "no effect". Anything outside these lists is exploratory and labelled so.

## 8. Error analysis (stage 1 **done** on dev; stage 2 **planned**)

Each wrong answer on a changed item gets one cause: (1) retrieval miss — no update-window evidence in
the pool; (2) admission miss — present but not admitted; (3) generator override — admitted, answer
contradicts it; (4) parse/format failure; (5) gold-label error (human check of ~100 labels). Stage 1,
dev, changed items (`error_analysis.py`; `log.md` Phase 26): retrieval misses 0 for every arm; admission
misses B1 13, B3 1, P 2; "evidence admitted, answer still wrong" B1 63, B3 81, P 95 — the bottleneck is how
the generator uses evidence (or the gold label), not retrieval. Stage 2 adds, on dev and on the
confirmatory split: per-class recall and confusion matrices for every arm; stance accuracy on the 35 (dev)
decisive flips; stance of update-window versus older papers (diagnostic only, using the oracle window);
and the conflict feature against gold NOT ENOUGH INFORMATION. Gold-label error (cause 5) is not checked.

## 9. Gates and decision rules (dev split only; thresholds fixed in advance)

**Stage 1 gates.**

| Gate | Pass condition | Status |
|---|---|---|
| G0 evidence headroom | ≥ 50% of changed dev items have a trial or review in the update window | **passed** (94.0%) |
| G1 format validity | ≥ 95% of answers parse AND a human check of 50 answers finds the stated verdict consistent with its justification in ≥ 90% (`consistency.py`) | pending |
| G2 generator uses evidence | B1 changes the verdict of ≥ 20% of dev items relative to B0 | **passed** (35.8%) |
| G3 dev effect | on changed dev items P − B2 ≥ +5 pp AND P − C1 ≥ +2.5 pp | **failed** (−1.3 pp and −2.0 pp) |

G2 and G3 are small-n sanity gates deciding whether the confirmatory split is run at all; they do not
establish that the method works. If G2 fails, no admission rule can matter with this generator and
prompt, and the work stops there.

*Operating characteristics of G3* (simulated; 151 changed dev items; ≈ 30% of answers differ between
arms, so the standard error of a paired difference is ≈ 4.5 pp). The P − B2 ≥ +5 pp condition is met
in 13% of runs when the true effect is zero, 50% when it is +5 pp and 87% when it is +10 pp; the
P − C1 ≥ +2.5 pp condition in 30%, 73% and 96%. G3 therefore screens out a clearly absent effect; it
cannot confirm a small one, and a real 5 pp effect fails it about as often as it passes.

**Stage 2 steps and gates (planned; thresholds fixed here, before any stage-2 result).**

| Step | What | Pass condition |
|---|---|---|
| P0 diagnostics (no model) | on dev pools: share of helpfulness inputs over 512 tokens; share of abstracts with labelled RESULTS/CONCLUSIONS; study-type composition; B1 verdicts by composition | informational: decides whether the stage-1 helpfulness caveat is confirmed, and whether the study-type weight has anything to act on (a systematic review in the top 8 for at least 30% of items) |
| P1 pilot (Gate 1) | 40 seeded dev items × top-8 papers, both wordings, plus an irrelevant-paper control (each question scored against papers of another pilot item); the researcher hand-labels 40 papers | **all of:** hand-check accuracy ≥ 70% (40 papers); wording agreement ≥ 80%; ≥ 70% of control papers rated "neither"; invalid outputs ≤ 2%; ≤ 10 s per paper (Llama) or ≤ 4 s (Flan-T5) |
| P2 full dev stance + fit (Gate 2) | all 226 dev items × top-8, the chosen wording; repeated 5-fold CV (50 repeats, seed 20261003) | **all of:** stance-direction AUC (signed score S0, gold SUPPORTED vs REFUTED) ≥ 0.60; selected hybrid CV accuracy ≥ B1R CV accuracy + 1.0 pp; S0 CV accuracy above the constant-prior CV accuracy |
| Confirmatory preparation | probe, candidate freezing for the 528 confirmatory items (no helpfulness) | none; run for RQ1 whatever Gate 2 says |
| Confirmatory run | B0 and B1 answers; stance for the top-8 of every pool; frozen model applied | stance run only if Gate 2 passed; RQ1 is run regardless |

*Operating characteristics* (computed). Hand-check, 40 papers, pass at ≥ 28 correct: a stance step that is
truly 60% accurate passes 13% of the time, 70% accurate 58%, 80% accurate 96%. Gate 2 (b), standard error
≈ 2.8 pp (226 items, ≈ 18% differing): with no true gain it passes 36% of the time, with a true +1 pp 50%,
+3 pp 76%, +5 pp 92%. The gates therefore screen out clear failures; they do not confirm anything. They
are lenient on purpose: after Gate 2 the extra cost of the synthesis arm is only the confirmatory stance run
(≈ 6–9 h), because the confirmatory preparation and B0/B1 are needed for RQ1 anyway.

*Data hygiene rules.* (1) The confirmatory labels and outcomes are not analysed before the frozen model
file is committed. (2) Every design choice made after seeing dev data is listed in §13. (3) At most the
declared variants, two stance wordings and one stage-model swap exist; anything else is a new fork and must
be added to the §13 list with its date. (4) Prediction on the confirmatory split refuses to run without
the frozen model file, and records that file's hash.

## 10. Evidence that would justify an international paper

Stage 2, confirmatory split, run once. **Strongest outcome:** RQ1 and RQ2 both confirmed (Holm p < .05, CI
excluding 0), the selected hybrid beats its date-shuffled control where it uses recency, the outdated-verdict
rate falls and the direction holds with a second generator — a method-plus-evaluation paper, still lacking a
second dataset and a trained-filter baseline (not available; the first is only partly addressed by the
Alzheimer's case study). **Likely useful outcome:** RQ1 confirmed, RQ2 not — an evaluation paper: as-of
retrieval helps a small local model, mostly by enabling appropriate abstention, while recency, study-type
weighting and stance synthesis add nothing detectable; together with the stage-1 negative result, the as-of
benchmark and the error analysis this is reportable and honest. **Weakest outcome:** RQ1 not confirmed — the
dev difference did not replicate, and that is reported as such. Because the benchmark can confirm only
effects of about 5 pp or more (§7), a true gain of 1–3 pp would be reported with its interval and called
suggestive, never as an improvement.

## 11. Alzheimer's disease case study (secondary, **planned — not implemented**)

The 99 verdict-labelled Alzheimer's questions (80 test, 19 validation; `experiments/shared/questions/`)
re-run time-consistently — t_q = the cited review's date, passages restricted to earlier ones — with the
same arms. Descriptive only (≈ 15–18 pp detectable). Needed before it can run: an as-of freezing step
over the local 4.4M-chunk index (the earlier pilot used the run date as t_q for every question; see
`_archive/alzheimers_pilot_v1/`) and an adapter for questions that have no previous-version verdict.
The 14 Alzheimer's-related MedChange items are reported individually.

## 12. Known limitations

Gold labels are model-generated; most changes involve NOT ENOUGH INFORMATION (116 of 151 changed dev
items), a vague boundary; abstracts only (no full text); one generator family at 4-bit, whose decoding is not
bitwise reproducible (identical prompts gave different wording in 9 of 17 pairs, with the same verdict);
PubMed best-match candidate generation is lexical; the RAG²-style arms are untrained stand-ins and may have
been scored on truncated inputs (§4); the stance step is a zero-shot judgement by a 4-bit 8B model whose
quality is checked only by a 40-paper hand check by a non-expert (confidence interval about ±14 pp);
the fitted layer learns the benchmark's class mix from dev (SUPPORTED 46%, REFUTED 22%, NOT ENOUGH
INFORMATION 32%), which may not transfer to other distributions; confirmatory power is limited to
≈ 5 pp (RQ2) to ≈ 8 pp (two unrelated methods).

## 13. Amendments (dated)

* **2026-10-01** — scope widened from Alzheimer's-only to MedChangeQA (primary) with Alzheimer's as a
  case study, after the Alzheimer's evaluation was found to be time-inconsistent (55 of 99 verdict-labelled
  questions cite pre-2010 reviews while 71% of the corpus is from 2020 or later), underpowered (90 test
  questions detect ≈ 17–20 pp) and circular (currency was the primary metric and the fitting objective).
* **2026-10-02, before any generation** — (a) the primary outcome is a parsed `VERDICT:` line, not a judge
  model; G1 changed accordingly; (b) one pool per item shared by all arms, RAG²'s rationale-as-query
  dropped, B2/P named as untrained helpfulness-ranked stand-ins; (c) arm settings and gate definitions
  frozen as in §4 and §9.
* **2026-10-02, audit** — arms delegate the formula to `src/proposed` (relevance ties are now broken by
  cross-encoder rank instead of sharing a normalised score); no outcome existed, so no result is affected.
* **2026-10-02, audit** — added the simulated power of the confirmatory test (§7) and the operating
  characteristics of G3 (§9); no threshold, setting or rule was changed.
* **2026-10-02, audit** — `generate_answers` now records the generator configuration (model file hash,
  context size, token limit, temperature, seed, prompt hashes) beside the answers and refuses to extend an
  answers file under a different one; no generation setting was changed.
* **2026-10-03, stage 2, before any confirmatory pool, answer or stance output** — after gate G3 failed on
  dev, the study pivots from recency-aware admission to the evidence-synthesis layer of §1 and §4–§10. The
  confirmatory split's labels and outcomes had not been analysed. **Decisions taken after seeing dev data**
  (the forking-path ledger): (1) dropping P as the proposed system; (2) RQ1 as a replication anchor; (3)
  B1R as the fairness control; (4) a pooled all-items primary outcome; (5) the top-8, snippet, feature and
  weight definitions; (6) the four-variant selection rule and its 1.0 pp margin; (7) the gate thresholds in
  §9; (8) the single-stance-wording rule. Nothing about the confirmatory data informed any of them.
