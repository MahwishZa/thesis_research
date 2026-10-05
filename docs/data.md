# Data

The datasets used, their provenance, what is committed and what is rebuilt locally, and their
limitations. For how retrieval consumes them see `methodology.md`; for how to rebuild anything see
`reproducibility.md`.

| Dataset | Role | Committed? | Rebuilt by |
|---|---|---|---|
| MedChange (MedRevQA, AllStudyGroups, MedChangeQA) | primary evaluation questions and gold verdicts | no (no licence stated); only `experiments/medchange/manifest.json` | `build_benchmark` from a clone of `jvladika/MedChange` |
| PubMed abstracts, as-of candidate pools | primary evidence | no (publisher text) | `pubmed_asof` → `freeze_candidates` |
| Per-paper stance records (`stance_<split>.jsonl`, `stance_pilot.jsonl`), P0 diagnostics | stage-2 judgements of the first eight pool papers; checks of the inputs | yes, copied to `results/` after a run (PMIDs, ranks, probabilities and timings only; no source text) | `stance`, `diagnostics` |
| Label-audit and consistency outputs (`label_audit_<split>`, `consistency_auto_<split>`) | an independent model's re-labelling of the gold labels and judgement of stated verdicts | yes (ids, labels and agreement only) | `label_audit`, `consistency_auto` |
| Frozen synthesis model (`synthesis_model.json`) | the fitted stage-2 layer: coefficients, standardisation, selected hybrid, stance wording | yes, and **before** any confirmatory stance run | `synthesis fit` (dev only) |
| Realigned study: rationales, candidate lists, filter judgements, answers, directness judgements (`rag2_*_<split>.jsonl`) | the adapted RAG² baseline and the verification arms | yes, copied to `results/` by `rag2_pipeline` (model text, PMIDs, ranks, scores, dates; the candidate lists without titles or abstracts) | `rag2_run` |
| Design record (`rag2_design.json`) | every setting and prompt hash of the realigned systems and the generator file's hash | yes, and **before** the held-out run, which refuses to start otherwise | `rag2_pipeline dev` |
| Alzheimer's/dementia test set (split `ad`) | a fresh secondary held-out set for the realigned study: 212 MedRevQA questions on dementia and Alzheimer's disease from 163 reviews outside dev and confirm | no; only `experiments/medchange/manifest_ad.json` | `ad_benchmark` after `build_benchmark` |
| Alzheimer's evidence corpus | secondary evidence | provenance only (metadata, reports, logs) | stages 01–07 in `corpus/scripts/` |
| Alzheimer's question pool | secondary evaluation questions | yes (`experiments/shared/questions/`) | `build_pool`, human review (done 2026-09-20; not extended), `split` |

## 1. MedChange benchmark (primary)

**Source.** Vladika, Dhaini and Matthes, *Facts Fade Fast: Evaluating Memorization of Outdated Medical
Knowledge in Large Language Models* (EMNLP 2025 Findings), repository `github.com/jvladika/MedChange`.
`MedRevQA` has 16,501 questions derived from Cochrane systematic-review abstracts; each carries a
verdict label (SUPPORTED, REFUTED or NOT ENOUGH INFORMATION) that gpt-4o-mini assigned to the
review's authors' conclusions. `AllStudyGroups` groups the versions of one review; `MedChangeQA` lists
the 512 questions whose verdict changed between versions, with the newest and an outdated label.

**What this project adds** (`experiments/medchange/build_benchmark.py`): the released files do not say
which versions the labels belong to or when they were published. The builder reproduces MedChangeQA by
the authors' rule and **refuses to continue unless all 512 items match the release label for label**; it
then attaches both versions' dates (parsed from the Cochrane citation) and PMIDs, flags 8 changed pairs
whose conclusions are near-identical text (similarity ≥ 0.85) as label noise and excludes them, samples
250 unchanged controls, and fixes seeded dev/confirmatory splits stratified by kind and change type.
Counts are in `experiments/medchange/manifest.json` (input hashes are line-ending-normalised so a
Windows checkout reproduces them).

**Evidence.** For each question, PubMed records first public strictly before the newest version's date,
excluding the Cochrane Database, with abstracts fetched through E-utilities (`log.md` Phases 21–22). The
realigned study reuses these cached records and abstracts; it needs no new network access.
Abstracts are publisher text and are not redistributed. Stage 2 reads the first eight candidates of each
frozen pool one at a time, from the title and the RESULTS and CONCLUSIONS sections of the abstract (the
last three sentences when an abstract has no labelled sections); the records it writes contain only PMIDs,
ranks, probabilities, timings and a hash of the snippet, so they can be committed.

**Splits.** Seeded 20261001, stratified by kind and change type: dev 151 changed + 75 unchanged = 226 items
(gold labels SUPPORTED 105, REFUTED 49, NOT ENOUGH INFORMATION 72), confirmatory 353 + 175 = 528 items. No
review group or Cochrane ID appears in both splits (computed: 0 of 762) and the median newest-review year is
2014 in both. The confirmatory labels have not been analysed, and their class mix is deliberately not
inspected (it should be close to dev's because of the stratification).

**Limitations.** Gold labels are model-generated, not human-verified; their reproducibility is measured by an
independent model (`label_audit.py`) and clinician validation is unavailable (a stated limitation). 397 of 504 usable changes involve NOT ENOUGH INFORMATION, the vaguest boundary. Only 14 items
(9 changed, 5 unchanged) are Alzheimer's-related. Question text was written by a model from review
objectives. For stage 2, the P0 diagnostics (`experiments/medchange/diagnostics.py`) measured on dev that 36.4% of
candidates have labelled RESULTS and CONCLUSIONS sections and that a systematic review is among the first
eight candidates for 41.6% of items.

## 2. Alzheimer's evidence corpus (secondary)

**Status: built and frozen.** A seven-stage pipeline (`corpus/scripts/01_pubmed_download.py` through
`07_claim_classification.py`) over PubMed/PMC full text plus government public-health pages.

| Stage | What it does |
|---|---|
| 01 PubMed | retrieves a PMID list (no text or dates; see stage 02) |
| 02 PMC retrieval + finalisation | downloads full text from PMC, resolves publication dates from JATS XML |
| 03 Guidelines / textbooks | optional; not populated (no document could be independently source-verified) |
| 04 Normalise | text cleanup, Alzheimer's-relevance tagging |
| 05 Deduplicate | near-duplicate removal (content id + token-Jaccard) |
| 06 Chunk | real `ncbi/MedCPT-Article-Encoder` tokenizer, 256-token windows with 32-token overlap |
| 07 Claim classification | tags every chunk by claim type and evidence level |

Computed from the tracked provenance: 114,256 PMC records in `corpus/metadata/pmc.csv`, of which about
71% were published in 2020 or later (48% in 2020–24, 23% in 2025 and after); 4,377,041 chunks. The
dense index holds 4,376,141 chunks × 768: 900 chunks that repeated an id were dropped
(`keep_first`; see `experiments/shared/retrieval/corpus.py`). Every tracked provenance file
(`corpus/metadata/*.csv`, `corpus/reports/*.csv`, `corpus/logs/*.log`) is well-formed and free of
unresolved errors. The corpus text itself (`corpus/data/**`) is not committed: it is large, built
locally and gitignored by design; only the synthetic offline fixture
`corpus/data/raw/pubmed/records.example.jsonl` (ten `FIXTURE-*` records) is tracked so a fresh clone can
run stages 04–07.

**Known gaps.** The guideline/textbook source (stage 03) is empty. The PMC redistribution-licence gate
is recorded per record but not enforced, which affects redistribution of corpus text, not research use.
The corpus is Alzheimer's-scoped and therefore **not used by the primary experiment**.

## 3. Alzheimer's question pool (secondary)

**Provenance rule.** Every reference answer must be traceable to a real published record; nothing is
invented or model-authored. Pipeline: external source record → factual proposition → candidate
question → verbatim reference answer → citation, locator and date → automatic validation and
deduplication → human review (the only source of approval) → final question.

**Sources.** Cochrane systematic reviews via MedRevQA (treatment, diagnosis, prevention, prognosis; 128
of the 150 candidates) and NIH public-health pages via MedQuAD (22 candidates). **Human review** judged
every candidate ACCEPT / REVISE / REJECT / HOLD; no code path can produce an approved question.
Result: 123 reviewed, 113 usable (ACCEPT + REVISE), split by seed 20260921 into 23 validation and 90 test
questions (`splits.json`), stratified by topic and by whether the cited review was revised.

**Labels.** 99 of the 113 usable questions carry a verdict label (the 14 MedQuAD items do not). The
`temporal_candidate` flag means the cited Cochrane review has been revised at least once (`.pub2` or
higher); it does **not** assert that the verdict changed, and only 5 of the 113 questions are known
verdict changes. The cited reviews are mostly old: of the 99 verdict-labelled questions, 55 cite reviews
dated before 2010 and only 17 are from 2018 onward, while 71% of the corpus is from 2020 or later. This
mismatch is why the Alzheimer's evaluation had to be redone as-of (`log.md` Phase 21).

## 4. Known limitations of the data

1. Identifiers (PMIDs, DOIs) were transcribed from source datasets and not independently resolved in
   every build environment; this is flagged per record.
2. The Alzheimer's pool is predominantly Cochrane (~85%) and short by design: reference answers are single
   verbatim sentences.
3. Some public-health pages carry no publication date; the retrieval date is recorded and flagged rather
   than treated as a real date.
4. No part of this project trains on any question pool; the pools only evaluate. The one exception is the
   stage-2 logistic layer, which has a few dozen coefficients fitted on the 226 dev items and then frozen; it
   never sees the confirmatory split before prediction.
5. The stage-2 stance step is a zero-shot model judgement that no human validated: it is checked by machine
   checks, a negative control (irrelevant papers) and its predictive value for the gold labels on dev. Nobody
   involved is a medical expert and no clinician is available; this is a documented limitation.
6. The fitted layer learns the dev class mix (SUPPORTED 46%, REFUTED 22%, NOT ENOUGH INFORMATION 32%), which
   may not match another benchmark or the confirmatory split.
