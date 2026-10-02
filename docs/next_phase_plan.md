# Next phase: an as-of evaluation on real Cochrane verdict changes

**Status (2026-10-01): design fixed, benchmark built, nothing generated.**
Supersedes the earlier version of this file (author-contact route, dropped).
*Measured* = computed in this repository; *estimate* = derived from a measured
figure; *unverified* = not yet checked. Thresholds below were written before any
result existed and must not be changed after seeing one.

## 1. Why the scope changed

The Alzheimer's-only design could not produce valid evidence for a temporal
claim (log Phases 19-21):

* only 5 of 113 usable questions are known to have a changed verdict;
* 55 of the 99 verdict-labelled questions cite reviews from before 2010, while
  71% of the corpus is from 2020 or later, and every question was asked "as of"
  the run date. A system preferring current evidence would be graded against
  15-25-year-old conclusions;
* 90 test questions detect only ~17-20 pp differences;
* no answer-level result exists yet (the pilot used an extractive stand-in).

MedChangeQA (Vladika et al., EMNLP 2025 Findings) provides Cochrane reviews whose
verdict changed between published versions. It is the primary dataset;
Alzheimer's remains a secondary case study.

## 2. What the inspection found (all measured)

| Check | Result |
|---|---|
| Rebuild of MedChangeQA from the release | 512/512 items reproduced with identical labels; the authors' rule (lowest row = newest version) agrees with publication dates in 1,534/1,535 groups |
| Dates for both versions | recovered for every item (749 day-precision, 13 year-only; year-only cutoffs are conservative) |
| Gold labels | gpt-4o-mini labels of each abstract's conclusions, not human labels |
| Label noise | 8 changed pairs have near-identical conclusions (similarity >= 0.85) - excluded; 482 of 512 are substantially rewritten (< 0.6) |
| Change types (504 usable) | 397 involve NOT ENOUGH INFORMATION (the fuzziest boundary); 114 are decisive SUPPORTED <-> REFUTED flips |
| Update window | median 9 years between versions; newest versions mostly 2010-2019 |
| Headroom without retrieval (authors' released answers) | on changed items, current-verdict accuracy 49-50% for Qwen2.5-7B, Llama-3.3-70B, Mistral-24B, GPT-4o-mini and DeepSeek-V3 alike; 25-32% give the outdated verdict. Unchanged items: 54-60%. Model size barely matters, which points to a ceiling from label ambiguity |
| Alzheimer's within MedChange | 9 changed and 5 unchanged items - far too few for statistics |
| Evidence availability | **unverified** - PubMed is blocked from the session that built this; G0 below measures it |

## 3. Novelty: what can and cannot be claimed

Recency-aware retrieval and conflict handling in RAG are active areas (e.g.
TempRALM; AionRAG; FRESCO; ConflictRAG; EvoTrustRAG; "Contradictions in Context /
Toward Safer RAG in Healthcare", arXiv 2511.06668; DriftMedQA, EMNLP 2025 Findings).
Adding a time-decay term to an admission score is **not a novel method** and is not
claimed as one.

What is defensible, if it holds up: a controlled **as-of evaluation on real
Cochrane verdict changes**, comparing standard, helpfulness-filtered (RAG²-style)
and recency-aware evidence admission under identical retrieval and generation,
with a falsification control and an Alzheimer's case study. I found no paper doing
exactly this, but the search was not exhaustive (*unverified*).

**Name of the method:** *recency-weighted admission* on top of a RAG²-style
helpfulness filter - descriptive, not a brand. In text: "a RAG²-style pipeline with
recency-weighted evidence admission".

## 4. Research question

On questions whose Cochrane verdict changed, and asked as of the newest review's
publication date with only evidence published before that date, does
recency-weighted admission make a local LLM give the current verdict more often
than standard RAG, a RAG²-style helpfulness filter, and published-style recency
reranking - without lowering accuracy on questions whose verdict did not change?

## 5. Design

**As-of protocol.** For each item: t_q = newest version's date; candidates are
PubMed records published strictly before t_q; all Cochrane Database records are
excluded (the review and its versions); the gold answer is the newest verdict. The
system's input is the question only.

**Benchmark** (`experiments/medchange/`, seed 20261001, splits fixed before any
generation): 504 usable changed items (154 dev / 358 confirmatory after
stratification by change type) and 250 unchanged controls (75 / 175). The
derived text stays local (`experiments/medchange/data/`, gitignored: the release
states no licence); `manifest.json` holds input hashes, counts and split ids.

**Arms** - same candidate pool (top 20, as-of), budget 5, generator, prompt and
greedy decoding:

| Arm | Query | Admission | Dates | Role |
|---|---|---|---|---|
| B0 | none | none | no | model memory alone (the MedChange setting) |
| B1 | question | MedCPT rerank top-5 | no | standard medical RAG |
| B2 | question + model rationale | zero-shot [HELPFUL]/[NOT_HELPFUL] filter with RAG²'s prompt, top-5 by P(helpful) | no | RAG²-style baseline (adapted, untrained; named as such) |
| B3 | question | relevance + recency (TempRALM-style) | yes | closest published idea |
| **P** | as B2 | P(helpful) + lambda * recency | yes | proposed |
| C1 | as P | as P with dates randomly permuted within the pool | fake | falsification: a gain that survives shuffling is not temporal |

B1/B3/B2/P form a 2x2 (helpfulness filter x recency); the interaction tests whether
the combination adds anything. B3 and P get the same tuning budget on dev, chosen
by verdict accuracy - never by currency.

## 6. Outcomes

* **Primary: verdict accuracy** - the answer's verdict equals the newest gold
  verdict. The answer's verdict is read by a judge of a different model family that
  sees only the question and the answer (not the arm, evidence, dates or gold);
  validated against >= 150 human-labelled answers (require Cohen's kappa >= 0.7).
* **Key secondary: outdated-verdict rate** - the answer equals the previous
  version's verdict.
* **Safety: verdict accuracy on unchanged items** - non-inferiority, margin 5 pp.
* **Sensitivity:** the decisive-flip subset (114 items); human re-scoring of the
  judged subset; excluding non-Cochrane systematic reviews from candidates.
* **Supporting:** blinded human annotation of claims supported or contradicted
  by the shared candidate pool (60 items x arms B2, B3, P; existing
  `evaluation/annotation.py`); coverage and abstention beside every rate.
* **Manipulation checks only, never outcomes:** share of admitted evidence from
  the update window, currency, admitted-set overlap.

## 7. Statistics

* Confirmatory family, Holm-corrected: P vs B1, P vs B2, P vs B3 on the primary
  outcome (exact McNemar), with paired differences and question-resampled
  bootstrap 95% CIs.
* Mixed-effects logistic regression `correct ~ filter * recency + (1 | item)`;
  the interaction coefficient answers whether the combination is the contribution.
* Changed-vs-unchanged interaction; one-sided non-inferiority on unchanged; C1 vs P.
* Power (*computed*): 358 confirmatory changed items with ~30% discordance detect
  ~10 pp at Holm alpha; smaller real effects will read as inconclusive, and are
  reported with their intervals, not as "no effect".
* Everything outside this list is exploratory and labelled so.

## 8. Error analysis

Each wrong answer on a changed item is assigned one cause: (1) retrieval miss - no
update-window evidence among candidates; (2) admission miss - present, not
admitted; (3) generator override - admitted, answer contradicts it; (4) judge
error; (5) gold-label error (human check of 100 sampled gold labels). Reported by
change type, update-window length and arm, with confusion matrices and examples.

## 9. Gates (dev split only; thresholds fixed now)

| Gate | Pass condition | If it fails |
|---|---|---|
| G0 evidence availability (`pubmed_asof.py`) | >= 50% of changed dev items have a trial or systematic review published inside the update window among the as-of candidates (top 200) | the as-of corpus cannot carry the change; redesign retrieval before any generation |
| G1 judge | kappa >= 0.7 vs human labels | fix the judge or judge by hand |
| G2 sensitivity | B1 changes >= 20% of verdicts vs B0 on dev | the generator ignores evidence: no admission rule can matter; stop |
| G3 dev effect | P - B2 >= +5 pp and C1 does not reproduce it | report a null on dev; do not run the confirmatory split |

## 10. Compute (estimates; benchmark in G1 first)

PubMed probe and abstract fetch: hours. MedCPT encoding of ~100k abstracts on
CPU: hours. Helpfulness filter (top-10 x ~760 items): ~20-40 h. Generation (~760
items x 6 arms, short answers, 1-2 min each): ~75-150 h. Judge: ~15 h. In total
1-2 weeks of resumable laptop time, dev split first.

## 11. Alzheimer's case study (secondary)

The 99 verdict-labelled Alzheimer's questions, re-run **time-consistently**
(t_q = the cited review's date; corpus restricted to earlier passages) with the
same arms. Descriptive only (detects ~15-18 pp), plus the 14 Alzheimer's-related
MedChange items reported individually.

## 12. What counts as meaningful evidence

On the confirmatory split, all of: P beats B1, B2 and B3 on verdict accuracy by
>= 8 pp (Holm p < .05, CI excluding 0); the outdated-verdict rate falls; P is
non-inferior on unchanged items; C1 shows no comparable gain; the direction
holds with a second generator; coverage is matched.

* P beats B1/B2 but not B3: recency helps, but this method adds nothing over
  existing recency reranking - an evaluation paper, not a method paper.
* Nothing beats B1: an honest null; the as-of benchmark and the error breakdown
  are still reportable.

## 13. Fixed before any generation (2026-10-02) - amendments to sections 5-9

These were decided in code (`experiments/medchange/arms.py`, `prompts.py`,
`analyze.py`) before a single answer was generated; the arm-settings hash is stored
in every answer record.

**Two disclosed changes from the earlier design.**
1. *No judge model for the primary outcome.* The generator must open with
   `VERDICT: SUPPORTED | REFUTED | NOT ENOUGH INFORMATION`; the primary outcome is that
   parsed verdict against the gold verdict (the same three-way task the MedChange
   authors used). Unparsed answers count as wrong and are reported. This removes judge
   noise and cost. G1 therefore becomes: >= 95% of answers parse AND a human check of 50
   answers finds the stated verdict consistent with the justification in >= 90%. A judge
   model is kept only as a sensitivity analysis.
2. *Pools are built from the question alone and shared by all arms.* RAG²'s
   rationale-as-query step is dropped, so arms differ only in admission (the repository's
   core design rule). B2 is therefore "a zero-shot helpfulness filter", **not RAG²**:
   unmodified Flan-T5-large asked RAG²'s prompt plus "Answer yes or no.", scored as
   P(yes). B2 and P cannot be described as reproductions of RAG².

**Arm settings (not tuned).** Budget 5 passages; recency T = 2^(-age/H) with H = 1,095
days, age measured from the question date (the newest review's date) to the midpoint of a
passage's availability bounds; relevance signal rank-normalised to [0,1] within the pool
of 20; lambda = 0.5 for B3, P and C1. Passages are shown without dates in every arm.
Greedy decoding, 160 new tokens, one prompt template.

| Arm | Relevance signal | Recency |
|---|---|---|
| B0 | none (no evidence) | - |
| B1 | MedCPT cross-encoder rank | no |
| B2 | zero-shot Flan-T5 P(yes) | no |
| B3 | cross-encoder rank | yes |
| P | zero-shot Flan-T5 P(yes) | yes |
| C1 | as P, dates shuffled within the pool | fake |

**Gate definitions (dev split, final).**
* G0 (passed 2026-10-02): >= 50% of changed dev items have a trial or review in the
  update window among the as-of candidates. Result 94.0% (n = 151).
* G1: as above (parse rate >= 95%; human consistency >= 90% on 50 answers).
* G2: B1 changes the verdict of >= 20% of dev items relative to B0.
* G3: on changed dev items, P - B2 >= +5 pp AND P - C1 >= +2.5 pp.

G2 and G3 are dev-only sanity gates with small n; they decide whether the confirmatory
split is run at all, not whether the method works.
