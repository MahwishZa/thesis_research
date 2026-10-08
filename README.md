# Thesis Title (Temporary Placeholder)

*The thesis title has not yet been decided by the supervisor and the student. Working subject: retrieval-augmented medical
question answering evaluated as of a date.*

## Overview

This repository holds the code, documentation and results of an MS thesis on retrieval-augmented generation (RAG) for
medical questions. A question asks whether a medical claim is supported, refuted, or has not enough information; the
reference answer is the verdict of the newest Cochrane systematic review, and each question is asked *as of* that review's
date, so a system may read only PubMed abstracts published before it (the MedChange benchmark, Vladika et al., 2025). Every
system uses one small local model, Meta-Llama-3-8B-Instruct (4-bit, run on a CPU laptop). The baseline is an **adapted
RAG²** system (Sohn et al., 2025): a rationale written by the model as the retrieval query, retrieval balanced across
evidence types and an LLM filter. It is not a reproduction of RAG². The **proposed system** adds **evidence-criteria
verification**: a second pass in which the model checks the baseline's answer against the admitted abstracts using fixed
criteria (directness, study design, the meaning of each verdict, and the currency of the evidence).

The study is built around one requirement agreed with the supervisor: the proposed system must improve verdict accuracy by
at least 1 percentage point over the adapted RAG² baseline, genuinely and reproducibly. The design was fixed before the
held-out run, which was completed once; a secondary test set of 208 dementia and Alzheimer's questions has been built but
not yet run. Two earlier approaches, recency-aware evidence admission and a per-paper evidence-synthesis layer, did not beat
standard retrieval and are kept as negative results of record.

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant
   baseline models and existing works.

| Objective | Addressed by | Status |
|---|---|---|
| 1 | the pipeline in `experiments/medchange/`; metrics, statistics and decision rules fixed in advance (`docs/evaluation.md`, `docs/protocol.md`); a unit-test suite | implemented and evaluated on the development and held-out splits; the secondary dementia and Alzheimer's set is not yet run |
| 2 | paired comparisons with the adapted RAG² baseline, standard retrieval and no retrieval; controls (R2C, R2V-ND); the earlier approaches; existing models' released answers as context | held-out comparison completed once: the improvement is a point estimate, not confirmed (below) |

## Research Results

Unless stated otherwise, figures are verdict accuracy against the newest Cochrane verdict, taken from the committed results
(`experiments/medchange/results/`); the metrics and the rules for reading them are in `docs/evaluation.md`.

**Held-out split (primary): 528 questions, run once.**

| System | All (95% CI) | Changed (n = 353) | Unchanged (n = 175) |
|---|---|---|---|
| B0: no evidence | 46.6% (42.4–50.9) | 41.9% | 56.0% |
| B1: standard retrieval, top 5 | 48.1% (43.9–52.4) | 45.0% | 54.3% |
| R2: adapted RAG² (baseline) | 48.7% (44.4–52.9) | 43.6% | 58.9% |
| R2C: R2's evidence read with the criteria, no draft (control) | 48.5% (44.2–52.7) | 45.0% | 55.4% |
| **R2V: evidence-criteria verification (proposed)** | **50.0%** (45.8–54.2) | 47.0% | 56.0% |
| R2V-ND: R2V without dates (ablation) | 49.1% (44.8–53.3) | 45.3% | 56.6% |

**The requirement and the paired comparisons (held-out).** The reading of the requirement was fixed before any held-out output
existed: *met and confirmed* (difference of at least 1.0 pp, exact McNemar p < .05 and the 95% interval above 0), *met as a
point estimate, not confirmed* (at least 1.0 pp otherwise) or *not met*.

| Comparison | Difference | 95% CI | Exact McNemar p (Holm) | Reading |
|---|---|---|---|---|
| **R2V − R2 (requirement: at least +1.0 pp)** | **+1.3 pp** | −1.9 to +4.7 | 0.51 | **met as a point estimate, not confirmed** |
| R2V − R2, changed questions | +3.4 pp | −0.8 to +7.4 | 0.15 (0.89) | not confirmed |
| R2 − B1 | +0.6 pp | −3.0 to +4.2 | 0.84 (1.0) | not confirmed |
| R2V − B1 | +1.9 pp | −2.5 to +6.1 | 0.44 (1.0) | not confirmed |
| R2C − R2 | −0.2 pp | −3.6 to +3.4 | 1.0 (1.0) | not confirmed |
| R2V − R2C | +1.5 pp | −1.7 to +4.7 | 0.41 (1.0) | not confirmed |
| R2V − R2V-ND | +0.9 pp | −1.1 to +3.0 | 0.49 (1.0) | not confirmed |

**Development split (exploratory): 226 questions.** R2V − R2 = −0.4 pp (95% CI −5.8 to +4.9); the pre-declared dev check failed
on direction, and the design was frozen without using the one allowed prompt revision.

| System | B0 | B1 | R2 | R2C | R2V | R2V-ND |
|---|---|---|---|---|---|---|
| All (n = 226) | 45.1% | 53.1% | 50.0% | 49.1% | 49.6% | 52.2% |

**Retrieval (held-out, descriptive).** R2 admits fewer abstracts than B1 but more of them test the question directly.

| System | Abstracts admitted (mean) | Questions with none | Systematic review or meta-analysis / trial / other | Judged direct by an independent model |
|---|---|---|---|---|
| B1 | 5.0 | 0.0% | 10.9% / 17.1% / 72.0% | 28.6% |
| R2, R2C, R2V, R2V-ND (same evidence) | 3.3 | 15.3% | 24.6% / 32.9% / 42.5% | 57.3% |

**Existing models, context only** (the benchmark authors' released closed-book answers to the same 528 questions, no retrieval;
different size, training and prompt, so not a controlled comparison): Qwen2.5-7B 52.8%, GPT-4o-mini 52.8%, DeepSeek-V3 52.3%,
Llama-3.3-70B 50.8%, Mistral-24B 50.8%. The local 8B systems (46.6–50.0%) are at or below them.

**Status of the experiments.**

| Experiment | Status |
|---|---|
| Development run, 226 questions | run 2026-10-05 to 2026-10-06; dev check failed on direction |
| Design freeze | completed 2026-10-06; design record committed |
| Held-out run, 528 questions | started 2026-10-06 after the freeze; completed once, 2026-10-08 |
| Label reproducibility audit | completed: an independent model reproduces 81.4% of the held-out and 83.2% of the development gold labels |
| Baseline ablations R2-RQ, R2-BR, R2-NF | implemented, **not run** |
| Dementia and Alzheimer's set, 208 questions | built, **not yet run** |
| Recency-aware admission and evidence synthesis (earlier approaches) | completed; negative; results of record |

**Reading.** On the held-out split the proposed system's point estimate is 1.3 points above the baseline, which meets the
1-point requirement as a point estimate, but the test cannot separate it from zero: a gap of +1.3 points or more in R2V's favour
arises by chance alone about a quarter of the time if the two systems are equally good (one-sided p ≈ 0.25), and with 528
questions only effects of roughly 4 to 6 points or more can be confirmed.
The improvement is therefore **not demonstrated**. The baseline itself is not better than standard retrieval (+0.6 points),
the criteria without a draft equal the baseline, and on the development split the difference was −0.4 points. The gold labels
are model-generated and an independent model reproduces only 81.4% of them, which bounds what any accuracy here can mean. Full
tables: `docs/evaluation.md` §6 and `experiments/medchange/results/RAG2_FINDINGS.md`.

## Repository Structure

```
.
├── README.md
├── pyproject.toml              packaging and the dependency list
├── docs/
│   ├── data.md                 datasets: sources, construction, preprocessing, temporal information, usage
│   ├── methodology.md          the approach step by step: adapted RAG², verification, models, literature
│   ├── protocol.md             experimental design: systems, controls, procedure, decision ledger, amendments
│   ├── evaluation.md           metrics, comparison methodology, statistics, interpretation rules, results to date
│   ├── reproducibility.md      install, tests, commands, costs, recorded configuration
│   └── log.md                  dated research log: decisions, progress, problems, changes
├── src/
│   ├── common/                 the Evidence and Candidate record types
│   └── temporal_filter/        the recency formula of the earlier approach (result of record)
├── evaluation/
│   ├── stats.py                exact McNemar test, paired bootstrap interval, Holm correction
│   └── tests/                  the unit-test suite and the check that it leaves the repository unchanged
└── experiments/medchange/      the pipeline
    ├── build_benchmark.py, ad_benchmark.py, benchmark.py, manifest.json, manifest_ad.json    the benchmark and its splits
    ├── pubmed_asof.py, freeze_candidates.py, encoders.py, abstracts.py                       as-of evidence and retrieval models
    ├── prompts.py, arms.py, generate_answers.py                                              standard answering (B0, B1)
    ├── rag2.py, rag2_run.py, rag2_pipeline.py, runner.py                                     adapted RAG² and verification
    ├── scoring.py, analyze_rag2.py, report.py, headroom.py, label_audit.py, consistency_auto.py   analysis and audits
    └── results/                committed results without source text; results/earlier_stages/ holds the earlier approaches
```

`experiments/medchange/README.md` maps each module to its step, and `experiments/medchange/results/README.md` lists the result
files. The working data (`experiments/medchange/data/`) and model files (`models/`) are not committed.

## How to Run

Commands are run from the repository root; model files and platform notes are in `docs/reproducibility.md`. The `dev` and
`confirm` phases below have been run (results above); the `ad` phase has not.

```bash
# Install (numpy only; the model backends come with the extras) and run every test
pip install -e .
python -m unittest discover -s evaluation -t .

# Rebuild the benchmark from the MedChange release (once)
git clone https://github.com/jvladika/MedChange ../MedChange
python -m experiments.medchange.build_benchmark --medchange-dir ../MedChange
python -m experiments.medchange.ad_benchmark --medchange-dir ../MedChange

# Run the study: one resumable command per phase; --commit commits the results and never pushes
python -m experiments.medchange.rag2_pipeline dev --model-path models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit
python -m experiments.medchange.rag2_pipeline confirm --go --model-path models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit
python -m experiments.medchange.rag2_pipeline ad --go --model-path models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --judge-path models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --commit

# Re-analyse and report (no model is run; needs the rebuilt benchmark and the run's files in experiments/medchange/data/)
python -m experiments.medchange.analyze_rag2 --split confirm --out-dir experiments/medchange/results
python -m experiments.medchange.report --split confirm --medchange-dir ../MedChange
```

The held-out and `ad` phases refuse to start without `--go`, without the development report, and unless the design record on
`origin/main` equals the current design. `--dry-run` prints a phase's steps without running them, and `rag2_pipeline status`
shows which phase is done.
