# Next phase: evidence that the proposed system improves RAG reliability

**Status: proposal for supervisor review. Nothing in this plan has been run.**
Written 2026-10-01 after the filter-training negative result (log Phases 13-19).
Numbers marked *measured* come from this project's own runs; *estimate* means
derived from a measured figure; *unverified* means not checked.

## 0. The question this phase must answer

> Does the proposed system produce more correct, less hallucinated answers
> than appropriate existing baselines, measured by an outcome that does not
> depend on the mechanism under test?

It must be able to return a real improvement **or** an honest null. Showing that
the mechanism changes which passages are admitted (currency, admitted-set
overlap) is a manipulation check, never the result.

## 1. Faithful RAG² reproduction: what is missing

"Faithful reproduction" has two meanings; only the second is needed here.

* **A. Reproduce the paper's own results** (four biomedical corpora, rationale
  queries, GPU training, MIRAGE-style benchmarks). Not feasible locally and not
  needed for this thesis.
* **B. A faithful RAG² filter as a baseline arm** on our frozen candidate sets.
  Arms differ only in which passages are admitted, so this needs only the
  trained filter's decisions on our candidates.

| Artifact | Released by the authors? | Needed for | Local cost if obtained |
|---|---|---|---|
| Trained Flan-T5 filter checkpoint | **No.** README: "not available for distribution; use `classifier/` to train an equivalent filter on your own labeled data" | B | inference only: 1.34 s/pair *measured* (50 pairs in 67 s) -> 2,260 pairs ~ 50 min |
| Full labelled set | **No.** 5 examples shipped; ids run to `llama3_5%_23600` (so ~23.6k) | B (by retraining) | 106 s per 16-example step *measured* -> ~43 h/epoch; 40 epochs ~ 72 days; 3 epochs ~ 5.4 days *estimate* |
| Label generation (CoT prompt, perplexity code, rationale code) | Filter-training and retriever code yes; rationale generation, evaluation code and full prompts reported as not released (per README as read by a fetch tool - *verify by reading the README*) | regenerating labels | ~30 days at the measured 109 s/pair; and our Q4 labeller was shown to produce near-noise labels (log Phases 15-18) |
| Corpora (PubMed, PMC, CPG, textbooks) | Not hosted (multi-GB, partly licence-restricted) | A only | not needed for B |
| Retriever (MedCPT query encoder, MIPS, cross-encoder rerank) | Yes | both | already implemented here |

**Conclusion.** The only artifact that makes B realistic is the checkpoint, or the
filter's decisions on our candidates. Retraining at the authors' scale on a laptop
is not possible, and regenerating the labels is neither possible nor reliable.

## 2. Is contacting the authors worthwhile? Yes - cost ~30 minutes

The README states a checkpoint policy, so a bare checkpoint request may be refused.
Ask in tiers so a refusal of the first still leaves useful answers:

1. the checkpoint, for non-commercial research use under any terms they set;
2. **if they will not distribute it: run their filter on a frozen list of our
   (question, passage) pairs and return the labels and the two logits.** One run,
   no distribution, and it is exactly what the baseline arm needs;
3. at minimum: the CoT prompt, the perplexity computation, and the real size of
   the `5%` split.

Before sending: read the replies on GitHub issues #1 (perplexity calculation, 5
comments) and #2 (data and models) - they may already answer part of this. The
replies could not be read by the tool used here, so this is *unverified*. Do not
let the project wait on a reply; plan as if none comes. The email text is kept outside the repository.

## 3. If the checkpoint is unavailable: the strongest practical path

### 3.1 What the proposed system can credibly claim

The mechanism is recency-aware admission. It can plausibly reduce **temporal /
outdated-evidence errors**. There is no reason to expect it to reduce hallucination
in general. A claim of general hallucination reduction would need a different
mechanism (for example evidence-conditioned abstention or verification) and is out
of scope here.

### 3.2 Baselines (all runnable locally)

| Arm | Role | Status |
|---|---|---|
| No retrieval (generator alone) | Does retrieval help or hurt at all? | new, trivial |
| No-Filter (top-5 reranked) | standard RAG | exists |
| Relevance-only (lambda = 0) | does the *temporal term* add anything? | exists |
| Temporal Filter, 2 pre-declared settings | the proposal | exists |
| **Temporal Filter with shuffled dates** | specificity control: a gain that survives date shuffling is not temporal | new, small |
| Hard date window (e.g. last 5 years) | the common practical heuristic | new, small |
| Zero-shot text-only filter (Flan-T5 or Llama-3), *named as such* | stand-in for "text-only filtering"; **not RAG²** | new, 1-10 h estimate |
| RAG² via the authors' filter decisions | the intended baseline | only if authors respond |

Self-RAG / CRAG-style systems use different generators and trained critics;
postpone.

### 3.3 Outcomes: not circular

* **Primary: verdict accuracy** against the latest Cochrane verdict
  (SUPPORTED / REFUTED / NOT ENOUGH INFORMATION), available for 99 of the 113
  usable questions (80 test, 19 validation; the 14 MedQuAD items have none). A
  separate judge reads **only the answer text** and classifies its verdict - it
  never sees the arm, the evidence, the dates or the reference. Different model
  family from the generator where possible; validated against >= 50
  human-labelled answers (report kappa; if kappa < 0.6, use human judging).
  Also report balanced accuracy (labels are ~38% NEI / 34% REFUTED / 27% SUPPORTED).
* **Key secondary: outdated-answer rate** on questions whose verdict changed
  between review versions (only 5 of 113 are known to be such questions).
* **Hallucination: blinded human annotation** on a stratified subset (existing
  `evaluation/annotation.py` blinding and agreement code). Claims are judged
  against a *common* reference (gold conclusion plus the shared candidate pool), not
  each arm's own admitted evidence, so an arm cannot win by admitting less.
  Always reported with abstention/coverage (existing `har` / `coverage`).
* **Manipulation checks only (never outcomes):** currency, admitted-set overlap,
  age distribution.
* **Dropped as evidence:** token F1 / ROUGE against one verbatim sentence and
  token-overlap "groundedness" (weak validity; arm-dependent).

### 3.4 How to avoid winning because the metric was tuned for it

1. No fitting on currency. Prefer **no tuning at all**: pre-declare 2 temporal
   settings, run both, correct for multiplicity (Holm). Sensitivity grids are
   exploratory, on development data only.
2. The shuffled-date arm and the irrelevant-evidence control (a gain that does not
   need dates or relevant text is not the mechanism).
3. Subgroup prediction written down in advance: effect on verdict-changed /
   revised questions, none or non-inferior (within 5 pp) on the rest.
4. Judge prompt frozen on development answers; arm identity hidden; analysis plan
   committed (git tag) **before** any test-split generation.
5. Report every pre-declared arm, with confidence intervals, whatever the result.

### 3.5 Is the current question pool enough? No

Exact paired test (McNemar), 80% power, *computed*:

| Discordant share | 5 pp | 8 pp | 10 pp | 15 pp | 20 pp |
|---|---|---|---|---|---|
| 0.2 (alpha .05) | 626 | 243 | 155 | 68 | 37 |
| 0.3 (alpha .05) | 940 | 366 | 234 | 103 | 57 |
| 0.3 (Holm .0167) | 1254 | 489 | 312 | 137 | 76 |

Our pilots saw 20-30% of answers change when any passage was added, so a discordant
share of ~0.3 is realistic. Consequences:

* 80 test questions detect only differences of ~17-20 pp; 250 questions ~10-11 pp.
* **A null on this pool is inconclusive**, not "no effect"; report the interval.
* The temporal subgroup is the weak point: only 5 of 113 questions are known
  verdict-changed, and `temporal_candidate` only means "review revised".
* Headroom in the source data: MedRevQA holds 281 dementia/Alzheimer's questions
  (201 from revised reviews) versus 113 in the pool; MedChangeQA (512
  verdict-changed questions across medicine) contains only 9 dementia-related
  ones. If MedChangeQA lists all changes, Alzheimer's alone cannot supply a
  powered temporal subgroup (*unverified*).
* **Stage-2 option (needs supervisor decision):** use MedChangeQA's cross-disease
  verdict-changed questions as the temporal benchmark, with per-question candidate
  pools from dated PubMed abstracts and a leakage firewall (exclude the Cochrane
  review and its versions). Larger and more convincing, but it widens the thesis
  beyond Alzheimer's.

## 4. Minimum experiment before heavy investment (decision gates)

Fix the pass/fail rules before running; they are go/no-go sanity gates, not
hypothesis tests.

| Gate | What | Cost | Proceed only if |
|---|---|---|---|
| G0 | Freeze candidates for all 113 questions on the full index (retrieval only). Headroom: how often do relevance-only and temporal top-5 differ, and by how much age? No generation. | hours, CPU | admitted sets differ on a meaningful share of questions (if they are nearly identical, no answer difference is possible) |
| G1 | Wire a local (llama.cpp) generator to the existing `Generator` interface; 10-minute speed benchmark with 5-passage prompts. Build and validate the verdict judge on ~50 human-labelled answers. | ~1 day | judge kappa >= 0.6; measured s/answer known |
| G2 | **Generator sensitivity** on the 23 validation questions: no retrieval / top-5 / irrelevant-5. | 1-2 days | evidence moves answers: >= 20% of verdicts change between none and top-5 **and** top-5 beats irrelevant-5 by >= 5 pp (point estimate). Otherwise stop: no admission rule can matter with this generator/prompt. |
| G3 | Pilot comparison on the 99 verdict-labelled questions, ~6 arms (section 3.2). Gives effect sizes, variance and the discordant share. | 2-4 days *estimate* | go to Stage 2 if temporal vs relevance-only >= +8 pp **and** the shuffled-date arm does not reproduce it; otherwise report the pilot as an honest null/inconclusive result |

Generation cost *estimate* from the measured ~60 s per label-run generation:
1.5-3.5 min per 5-passage answer, so 99 questions x 6 arms is ~15-35 h; 250 x 6 is
~38-88 h. Benchmark in G1 before trusting this.

## 5. What can run locally / what to postpone

**Local (CPU/RAM, resumable):** retrieval and freezing, admission analysis,
llama.cpp generation (Llama-3-8B Q4; Qwen2.5-7B as second generator), the verdict
judge, statistics, blinded human annotation packets.

**Postpone:** thesis writing; any filter training or label regeneration; Self-RAG /
CRAG; pool expansion until G2-G3 show headroom (candidate human review can start in
parallel because it costs no compute); GPU offload tuning; the as-of-date
simulation.

## 6. What would justify an international paper

All of the following on **held-out questions not used in the pilot** (>= 200
questions, or >= 60 verdict-changed ones):

1. Temporal arm beats the **strongest** non-temporal comparator (relevance-only,
   no-filter, and the text-only filter / RAG² if available) by >= 8-10 pp verdict
   accuracy, or cuts the blinded human hallucination rate by a comparable margin;
   exact McNemar with Holm-corrected p < .05 and a bootstrap CI excluding 0;
2. the shuffled-date control does **not** show a comparable gain (and differs from
   the real-date arm);
3. the effect is concentrated in verdict-changed/revised questions and the arm is
   non-inferior (within 5 pp) on the rest;
4. it replicates with a second generator;
5. coverage is matched (no gain from answering fewer questions).

Reviewers will still ask for a second dataset and a trained-filter baseline; the
MedChangeQA stage and the authors' filter decisions address those. Without them, a
workshop-level paper is the realistic ceiling.

**A well-powered null is still a valid result.** Report the confidence interval and
frame the contribution as a measurement of how often medical RAG returns outdated
conclusions and whether recency-aware admission changes that. That framing is
weaker as a "method" paper but honest and publishable at workshop level.

## 7. First actions (no long computation)

1. Read GitHub issues #1 and #2 (replies); send the author email.
2. Supervisor decisions: adopt verdict accuracy as the primary outcome (retiring
   currency as primary and as the fitting objective - a disclosed methodological
   change made before any test-split result exists); decide on the MedChangeQA
   stage.
3. Commit the analysis plan (arms, outcomes, gates, thresholds) and tag it before
   G1.
4. Then implement G0/G1.
