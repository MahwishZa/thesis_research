# Experiment plan: as-of evaluation on Cochrane verdict changes

This is the protocol for the primary experiment. It was fixed in code and text **before any
answer was generated** (see §13 for amendments, each dated). Status labels used throughout:
**done** (run, result recorded in `log.md`), **built** (code and tests exist, not yet run on
the full data), **planned** (not implemented). Numbers are *measured* (this project's runs),
*computed* (from files in this project) or *estimated* (derived from a measurement).

## 1. Question and what can be claimed

> On medical questions whose Cochrane verdict changed between review versions, asked as of the
> newest review's publication date with only evidence published before that date, does
> recency-weighted evidence admission make a local LLM give the current verdict more often than
> standard RAG, a helpfulness-ranked admission, and a published-style recency reranking, without
> lowering accuracy on questions whose verdict did not change?

Recency-aware retrieval and conflict handling in RAG are crowded areas (TempRALM; AionRAG;
FRESCO; ConflictRAG; EvoTrustRAG; "Contradictions in Context / Toward Safer RAG in
Healthcare", arXiv 2511.06668; DriftMedQA, EMNLP 2025 Findings). Adding a time-decay term to an
admission score is **not claimed as a novel method**. What this work can contribute, if the
results hold, is a controlled as-of evaluation on real verdict changes with a falsification
control, and an honest measurement of whether the signal helps. A search for an identical study
was not exhaustive (*unverified*). The mechanism can plausibly reduce *outdated-evidence*
errors; no claim about general hallucination reduction is made.

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
matters, which points to a ceiling from label ambiguity.

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
frozen, none empty, median size 20.

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

B1/B2/B3/P form a 2 × 2 (helpfulness signal × recency); the interaction asks whether the
combination adds anything. **B2 and P are not RAG².** RAG²'s trained filter is not distributed and
the local retraining attempt failed (archived; `log.md` Phases 13–18). B2 uses the unmodified
Flan-T5-large asked RAG²'s prompt plus "Answer yes or no.", scored as P(yes), and ranks by it
rather than thresholding as RAG² does; RAG²'s rationale-as-query step is also dropped so that all
arms share one pool.

## 5. Generation (**built**)

Llama-3-8B-Instruct, Q4_K_M GGUF (bartowski), llama.cpp on CPU, greedy decoding, 160 new tokens,
one prompt (`prompts.py`). The model must open with `VERDICT: SUPPORTED | REFUTED | NOT ENOUGH
INFORMATION` followed by at most three sentences. Measured on the target laptop: ≈ 18 s per answer
without evidence and ≈ 66 s with five passages, so ≈ 350 s per item over six arms: dev ≈ 22 h,
confirmatory ≈ 51 h (*estimated*). A second generator (Qwen2.5-7B-Instruct) is **planned** as a
robustness check.

## 6. Outcomes

* **Primary: verdict accuracy** — the parsed verdict equals the gold newest verdict. No judge
  model: an answer without a parsable verdict counts as wrong and is reported separately.
* **Key secondary: outdated-verdict rate** — the verdict equals the previous version's (changed items).
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

Confirmatory family, Holm-corrected: P vs B1, P vs B2, P vs B3 on the primary outcome — exact
McNemar with a question-resampled bootstrap 95% CI. Also planned: a mixed-effects logistic model
`correct ~ helpfulness × recency + (1 | item)` (the interaction is the contribution test), the
changed-vs-unchanged interaction, one-sided non-inferiority on unchanged items, and P vs C1.
Power (*simulated*, exact McNemar, assuming ≈ 30% of answers differ between the two arms — an
assumption until dev results exist): 353 confirmatory changed items give 84% power at Holm-corrected α
(92% at α = .05) for a true 10 pp difference, 61% (76%) for 8 pp and 33% (50%) for 6 pp. Smaller
effects will read as inconclusive and are reported with their intervals, not as "no effect".
Anything outside these lists is exploratory and labelled so.

## 8. Error analysis (**planned**)

Each wrong answer on a changed item gets one cause: (1) retrieval miss — no update-window evidence in
the pool; (2) admission miss — present but not admitted; (3) generator override — admitted, answer
contradicts it; (4) parse/format failure; (5) gold-label error (human check of ~100 labels). Reported
by change type, update-window length and arm, with confusion matrices and examples.

## 9. Gates and decision rules (dev split only; thresholds fixed in advance)

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

## 10. Evidence that would justify an international paper

On the confirmatory split (run once), all of: P beats B1, B2 and B3 on verdict accuracy by ≥ 8 pp
(Holm p < .05, CI excluding 0); the outdated-verdict rate falls; P is non-inferior on unchanged
items; C1 shows no comparable gain; the direction holds with a second generator; coverage is matched.
P beating B1/B2 but not B3 means recency helps but this method adds nothing over existing recency
reranking — an evaluation paper, not a method paper. Nothing beating B1 is an honest null; the as-of
benchmark and the error breakdown are still reportable. Reviewers will also expect a second dataset
and a trained-filter baseline; the second is not available (§4) and the first is only partly
addressed by the Alzheimer's case study.

## 11. Alzheimer's disease case study (secondary, **planned — not implemented**)

The 99 verdict-labelled Alzheimer's questions (80 test, 19 validation; `experiments/shared/questions/`)
re-run time-consistently — t_q = the cited review's date, passages restricted to earlier ones — with the
same arms. Descriptive only (≈ 15–18 pp detectable). Needed before it can run: an as-of freezing step
over the local 4.4M-chunk index (the earlier pilot used the run date as t_q for every question; see
`_archive/alzheimers_pilot_v1/`) and an adapter for questions that have no previous-version verdict.
The 14 Alzheimer's-related MedChange items are reported individually.

## 12. Known limitations

Gold labels are model-generated; most changes involve NOT ENOUGH INFORMATION; abstracts only (no full
text); one generator family at 4-bit; PubMed best-match candidate generation is lexical; the
RAG²-style arms are untrained stand-ins; confirmatory power is limited to ≈ 10 pp.

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
