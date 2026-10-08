# Data

The datasets the study uses, where they come from, how they were constructed and preprocessed, what temporal information
they carry, how each is used, what is committed to the repository and what is rebuilt locally, and their limitations. How
retrieval consumes them is explained in `methodology.md`; how to rebuild anything is in `reproducibility.md`. Numbers are
*computed* from the benchmark file or the committed manifests unless marked otherwise.

| Dataset | Role | Source | In the repository? | Rebuilt by |
|---|---|---|---|---|
| MedChange as-of benchmark: 754 usable Cochrane questions (development 226, held-out 528) | primary evaluation: questions and gold verdicts | MedChange release (`MedRevQA`, `AllStudyGroups`, `MedChangeQA`) | no (no licence stated); only `experiments/medchange/manifest.json` | `build_benchmark` |
| Dementia and Alzheimer's set (`ad`): 208 questions | secondary held-out evaluation | the same release | no; only `experiments/medchange/manifest_ad.json` | `ad_benchmark` |
| As-of PubMed records, abstracts, frozen pools | the evidence systems may read | PubMed (NCBI E-utilities) | no (publisher text) | `pubmed_asof`, `freeze_candidates` |
| Study outputs (rationales, filter judgements, answers, analyses, reports, design record) | results of the runs | produced by the pipelines | yes, in `experiments/medchange/results/` (no source text) | the pipelines |
| Earlier-stage outputs | results of record of stages 1 and 2 | produced by the removed stage code | yes, in `results/earlier_stages/` | not rebuilt (code in Git history) |

## 1. MedChange benchmark (primary)

**Source.** Vladika, Dhaini and Matthes, *Facts Fade Fast: Evaluating Memorization of Outdated Medical Knowledge in Large
Language Models* (Findings of EMNLP 2025); repository `github.com/jvladika/MedChange`. `MedRevQA` holds 16,501 questions
derived from Cochrane systematic-review abstracts; each carries a verdict label (SUPPORTED, REFUTED or NOT ENOUGH
INFORMATION) that gpt-4o-mini assigned to the review authors' conclusions. `AllStudyGroups` groups the versions of one
review. `MedChangeQA` lists the 512 questions whose verdict changed between versions, with the newest and an outdated label.
Question texts were written by a model from the reviews' objectives.

**Construction** (`experiments/medchange/build_benchmark.py`). The released files do not say which review version each label
belongs to or when it was published. The builder reproduces MedChangeQA by the authors' rule and **refuses to continue
unless all 512 released items match label for label**. It then attaches both versions' dates (parsed from the Cochrane
citation) and PMIDs, flags 8 changed items whose two conclusions are near-identical text (similarity ≥ 0.85) as label
noise and excludes them (504 usable changed questions), samples 250 unchanged controls (754 usable questions in all) and
fixes seeded splits stratified by kind and change type. Counts and the hashes of the input files (normalised to LF line
endings, so a Windows checkout reproduces them) are in `manifest.json`. A question is *changed* when its newest verdict
differs from an earlier version's and *unchanged* when it never did.

**Splits** (seed 20261001).

| Split | Questions | Changed / unchanged | Gold verdicts (SUPPORTED / REFUTED / NOT ENOUGH INFORMATION) |
|---|---|---|---|
| development (`dev`) | 226 | 151 / 75 | 105 / 49 / 72 (46% / 22% / 32%) |
| held-out (`confirm`) | 528 | 353 / 175 | 234 / 126 / 168 (44% / 24% / 32%) |

No review group or Cochrane ID appears in both splits (computed: 0 of 762 items). **The held-out split has been used before
the realigned study:** the stage-1/2 run scored B0 and B1 on it once, so its labels and that result had been seen when the
realigned design was fixed. The reuse is declared in the decision ledger (`protocol.md` §§3, 7), and no setting of the
realigned design was tuned on it.

**Temporal information.** Each question has a *newest* review version (its date is the question date *t_q* and its label the
gold verdict) and, for changed questions, a *previous* version with its own date and label. The newest reviews are dated 2004
to 2024 (median 2014 in both splits; 26% are from 2018 or later, 14 before 2005). For changed questions the median
difference between the years of the two versions is 9 on the development split and 8 on the held-out split (range 0 to 23). Dates are known to
the day for 741 of the 754 newest versions and to the year only for 13; the previous version's date is known to the year only
for 266 of the 504 changed questions, so the update window (the interval after the previous version and up to the newest) is
approximate for them. A year-only date is read as 1 January.

**Limitations.** (1) Gold labels are model-generated, not human-verified. An independent model of another family reproduces
81.4% of the held-out and 83.2% of the development labels (`evaluation.md` §1.4); no clinician has validated them. (2) 397 of
the 504 usable changes involve NOT ENOUGH INFORMATION, the vaguest boundary. (3) Only 14 questions (9 changed, 5 unchanged)
are Alzheimer's-related, too few for any test; the `ad` set exists for that reason (§2). (4) The question text was written by
a model.

## 2. Dementia and Alzheimer's set (secondary; split `ad`)

**Why it exists.** The research proposal names Alzheimer's disease as the domain, and the main benchmark holds only 14
related questions. `experiments/medchange/ad_benchmark.py` builds a second held-out set from the same release by the same
rules (`protocol.md` §8).

**Construction.** Every `MedRevQA` question whose text matches `dementia`, `alzheimer`, `mild cognitive impairment`,
`cognitive decline`, `cognitively impaired` or `cognitive impairment`, from a review that is in neither the development nor
the held-out split (excluded by study group **and** by Cochrane ID), with exact duplicate questions kept once. Each question
is asked as of its review's publication date, with the release's label for that review as the gold verdict. The questions are
appended to `benchmark.jsonl` with split `ad` and summarised in `manifest_ad.json` (counts, labels, an item-id hash and the
input-file hashes). The manifest was reproduced from the released files on 2026-10-05, and a first build of 212 questions was
corrected to 208 before any use (`protocol.md` §8).

**Composition.** 208 questions from 159 reviews; gold NOT ENOUGH INFORMATION 88 (42%), REFUTED 68 (33%), SUPPORTED 52 (25%).
All 208 are unchanged-verdict questions: 202 come from reviews with a single version (the `previous` field repeats `newest`),
and the other 6 have two versions with the same verdict. By the wording of the question, 48 name Alzheimer's disease, 156
dementia (15 name both) and 19 name neither (cognitive impairment after stroke, in Parkinson's disease or vascular disease,
mild cognitive impairment, delirium).

**Limitations.** (1) It is a dementia and cognitive-impairment set in which fewer than a quarter of the questions name
Alzheimer's disease, and the thesis calls it that, not an Alzheimer's-only benchmark. (2) No question has a changed verdict,
so changed-question and update-window results cannot be computed for it. (3) Its reviews are old: 48 of the 208 are dated
before 2005 (14 of the main benchmark's 762 are), 46 dates are to the year only and the earliest is 2000, so as-of evidence
will be thinner for many questions; the pools have not been built, so their sizes are unknown. (4) 208 questions come from
159 reviews, and the paired tests treat questions as independent, which slightly understates the uncertainty. (5) Its power is
low: only effects of about 6 to 10 points can be confirmed (`evaluation.md` §4). (6) Its labels have not been audited by an
independent model (the audit covers the development and held-out splits), so it has no label-stable subset. (7) The set has not
been run.

## 3. Evidence: as-of PubMed records

**Source.** PubMed, through the NCBI E-utilities (ESearch for record identifiers, ESummary for dates and publication types,
EFetch for abstracts), at most 3 requests per second without an API key and 10 with one.

**Construction and preprocessing** (`pubmed_asof.py`, `freeze_candidates.py`). For each question: a query built from its
content words, restricted to records published before the question date and excluding the Cochrane Database (so the review
and its versions cannot be retrieved); up to 200 records; for each record, bounds on the date it first became public, from
its print and electronic dates, keeping only records whose *latest possible* first-public date is on or before *t_q*;
abstracts of at least 200 characters; retractions, errata, comments, editorials, letters, news items and patient handouts
removed. A large share of boundary-ambiguous records stops the run, because it would mean the date parsing is wrong
(`methodology.md` §3.2 gives the details). *Computed:* on the development and held-out splits a question has on average 140 and
148 eligible records (minimum 22 and 4; none has zero). Abstracts are publisher text and are neither committed nor
redistributed: they are fetched per question and cached under the gitignored `experiments/medchange/data/`.

**Temporal information.** Every record carries `lower` and `upper` bounds on its first-public date, which determine eligibility
and the *update-window* retrieval metrics (`evaluation.md` §1.3). The year of a record is shown to the verifier in R2V, and
withheld in R2V-ND.

## 4. Study outputs and what is committed

The runs write their working files to `experiments/medchange/data/` (gitignored: MedChange-derived questions, abstracts,
frozen pools). Outputs that contain **no source text** and are expensive to regenerate are copied to
`experiments/medchange/results/` and committed, so that a run of many hours does not exist only on one laptop:

* the answers and their generator records (`answers_<split>.jsonl` for B0 and B1, `rag2_answers_<split>.jsonl` for the R2
  family, each with `.config.json`), the rationales, filter judgements and the candidate lists without titles or abstracts
  (PMIDs, ranks, scores, dates), and the directness judgements;
* the analyses and reports (`rag2_analysis_<split>.*`, `RAG2_DEV_REPORT.md`, `RAG2_FINDINGS.md`, `report/`), the label-audit
  and consistency outputs (ids, labels and agreement only), the design record `rag2_design.json` (committed **before** the
  held-out run, which refuses to start otherwise) and the environment records;
* `results/earlier_stages/` holds the outputs of stages 1 and 2 as results of record (`methodology.md` §10).

`answers_dev.jsonl` also contains the stage-1 answers of B2, B3, P and C1 for the development split. The file list is in
`experiments/medchange/results/README.md`.

## 5. Usage

| Data | Used by | Purpose |
|---|---|---|
| `benchmark.jsonl` (local) | every step | questions, dates, gold verdicts, splits |
| as-of records and abstracts (local) | `freeze_candidates`, `rag2_run lists` | candidate evidence |
| frozen pools (local) | `generate_answers` (B0, B1) | the 20 candidates behind B1 |
| rationales, lists, filter judgements | `rag2_run answers` | the evidence R2, R2C, R2V and R2V-ND read |
| gold labels | `analyze_rag2`, `label_audit` | scoring only; no system reads them |

No part of the study trains a model on any question set; the sets only evaluate. The one exception in the earlier stages was
the stage-2 logistic layer, with a few dozen coefficients fitted on the 226 development questions and frozen; it is not part of
the current systems.

## 6. Removed: the Alzheimer's evidence corpus and question pool

Both belonged to the first design and are **not used** by the current study; they remain in Git history (commit `5e03540`,
the `alzheimers_framework` subfolder of the former `_archive` folder). Facts computed before their removal:

* **The corpus** was a pipeline over PubMed Central full text and public-health pages: 114,256 PMC records, 70.9% published in
  2020 or later, split into 4,377,041 chunks with a dense index. It could not support an as-of test, because 71% of it is from
  2020 or later while the Alzheimer's questions mostly cite older reviews (`log.md` Phase 21).
* **The question pool** held 123 reviewed questions of which 113 were usable (23 validation, 90 test); 99 carried a verdict label,
  only 5 were known verdict changes, 55 cited reviews dated before 2010 and only 17 were from 2018 onward. The `ad` set (§2)
  replaces it as the Alzheimer's evaluation.

Local leftovers of that design (`corpus\`, `experiments\results\index\` and `_archive\`, among them 16.6 hours of
filter-training labels) are gitignored; a `git pull` does not delete them (`reproducibility.md` §7).

## 7. Known limitations of the data

1. Gold labels (main benchmark and `ad` set) are model-generated; their reproducibility is measured by an independent model, and
   no clinician has validated them. Nobody involved in the study is a medical expert, which is a documented limitation.
2. Identifiers (PMIDs, DOIs) were taken from the source datasets and not independently resolved in every build environment.
3. The held-out split's labels and its B0/B1 results had been seen before the realigned design was fixed (§1); only the `ad` set
   is untouched, and it is secondary.
4. MedChange states no licence and abstracts are publisher text, so neither is redistributed: a reader rebuilds both from the
   manifests and the public sources.
5. Only 36.4% of the 4,520 candidates in the development pools have labelled RESULTS or CONCLUSIONS sections
   (`results/earlier_stages/diagnostics_dev.md`); for the others the filter reads the last three sentences.
