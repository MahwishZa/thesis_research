# Thesis Title (Temporary Placeholder)

## Overview

AI chatbots are increasingly used to answer medical questions, but they can be wrong, base an answer on studies that do not
really address the question, or repeat advice that newer research has overturned. A common remedy is *retrieval-augmented
generation* (RAG): before answering, the system looks up scientific studies and reads them.

This research tests whether a small language model running on an ordinary laptop answers medical questions more correctly when
RAG gets one extra step: after drafting an answer from the studies it found, the model checks the draft against plain rules of
evidence (only studies that really test the treatment count, trials and systematic reviews weigh most, newer evidence takes
precedence). The step is compared with an adapted version of a published medical RAG method (RAG²; not a full reproduction of
it), with standard retrieval and with no retrieval. The research focuses on Alzheimer's disease, and its requirement is an
improvement of at least 1 percentage point in the share of correct answers.

The complete academic account (background, related work, methodology, detailed results, statistical analysis, discussion,
limitations and references) is in the accompanying research paper: [title and link to be added].

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant
   baseline models and existing works.

## Research Results

**What was measured.** Two sets of medical-evidence questions taken from Cochrane systematic reviews, each asked as of its
review's date so that only earlier studies can be used: 528 general-medicine questions and 208 dementia-related questions, of
which 48 name Alzheimer's disease. They are not an Alzheimer's-specific evaluation. Each answer is one of three verdicts
(*supported*, *refuted*, *not enough information*) and counts as correct when it matches the verdict of the Cochrane review;
*accuracy* is the share of correct answers, and differences are given in percentage points ("points" for short).

| System (same language model, same studies) | General medicine (n = 528) | Dementia-related (n = 208) |
|---|---|---|
| No retrieval | 46.6% | 33.2% |
| Standard retrieval | 48.1% | 30.8% |
| Adapted RAG² (baseline) | 48.7% | 34.6% |
| Proposed extra check | 50.0% | 38.0% |

| Proposed extra check minus baseline | Difference (points) | 95% range | p-value |
|---|---|---|---|
| General medicine | +1.3 | −1.9 to +4.7 | 0.51 |
| Dementia-related | +3.4 | −1.9 to +8.7 | 0.28 |

**How to read this.** The 95% range is the span of differences that fits the data, and the p-value is the chance of seeing a gap at
least this large if the two systems were really equally accurate (below 0.05 would count as evidence). Both gaps are above the
1-point requirement as point estimates, but both ranges include zero and negative values, so they cannot be told apart from
chance: the requirement is **met as a point estimate, not confirmed**. With these numbers of questions only gains of roughly 4 to
6 points (general medicine) or 6 to 10 points (dementia-related) could be confirmed. The results do not show that the extra
check improves accuracy, and they do not describe Alzheimer's disease specifically. The reference answers were produced by a
language model; a second, independent model reproduces 81.4% of them on the general-medicine questions. Further tables,
controls and the earlier approaches that did not work are in `docs/evaluation.md`.

## Repository Structure

```
.
├── README.md
├── pyproject.toml              packaging and the dependency list
├── docs/
│   ├── data.md                 datasets: sources, construction, preprocessing, temporal information, usage
│   ├── methodology.md          the approach step by step: adapted RAG², verification, models, literature
│   ├── protocol.md             experimental design: systems, controls, procedure, decision ledger, amendments
│   ├── evaluation.md           metrics, comparison methodology, statistics, interpretation rules, results
│   ├── reproducibility.md      install, tests, commands, costs, recorded configuration
│   └── log.md                  dated research log: decisions, progress, problems, changes
├── src/
│   ├── common/                 the Evidence and Candidate record types
│   └── temporal_filter/        the recency formula of the earlier approach (result of record)
├── evaluation/
│   ├── stats.py                exact McNemar test, paired bootstrap interval, Holm correction
│   └── tests/                  the unit-test suite and the check that it leaves the repository unchanged
└── experiments/
    └── medchange/              the pipeline and the general medical (as-of) benchmark
        ├── build_benchmark.py, ad_benchmark.py, benchmark.py, manifest.json, manifest_ad.json    the benchmark and its splits
        ├── pubmed_asof.py, freeze_candidates.py, encoders.py, abstracts.py                       as-of evidence and retrieval models
        ├── prompts.py, arms.py, generate_answers.py                                              standard answering (B0, B1)
        ├── rag2.py, rag2_run.py, rag2_pipeline.py, runner.py                                     adapted RAG² and verification
        ├── scoring.py, analyze_rag2.py, analyze_fresh.py, subgroup_ad.py, ad_case_study.py, class_balance.py, fresh_supply.py, cochrane_ad_supply.py, fresh_benchmark.py, report.py, headroom.py, label_audit.py, consistency_auto.py   analysis and audits
        └── results/            committed results without source text; results/earlier_stages/ holds the earlier approaches
```

The working data (`experiments/medchange/data/`) and model files (`models/`) are not committed.

## How to Run

Commands are run from the repository root; model files and platform notes are in `docs/reproducibility.md`.

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

# Questions of the dementia set that name Alzheimer's disease (from the answers on file; no model is run)
python -m experiments.medchange.subgroup_ad

# Per-verdict behaviour: predicted shares, recall per verdict, macro-F1 (from the answers on file; no model is run)
python -m experiments.medchange.class_balance

# Count the questions of MedRevQA that no split uses yet (counts only)
python -m experiments.medchange.fresh_supply --medchange-dir ../MedChange

# Count new Cochrane dementia/Alzheimer's reviews published after MedRevQA's newest review (counts only; needs the network)
python -m experiments.medchange.cochrane_ad_supply --medchange-dir ../MedChange

# Select the fresh test questions, then run and analyse the test (steps and freeze: docs/reproducibility.md section 11)
python -m experiments.medchange.fresh_benchmark --medchange-dir ../MedChange

# Re-analyse and report (no model is run; needs the rebuilt benchmark and the run's files in experiments/medchange/data/)
python -m experiments.medchange.analyze_rag2 --split confirm --out-dir experiments/medchange/results
python -m experiments.medchange.report --split confirm --medchange-dir ../MedChange
```

The held-out (`confirm`) and `ad` phases refuse to start without `--go`, without the development report, and unless the design
record on `origin/main` equals the current design. `--dry-run` prints a phase's steps without running them, and
`rag2_pipeline status` shows which phase is done.
