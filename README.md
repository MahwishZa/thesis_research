# Thesis Title (Temporary Placeholder)

## Overview

AI chatbots can give wrong medical answers, rely on studies that do not address the question, or repeat advice that newer research has overturned. This research tests whether a small language model answers medical questions more correctly when *retrieval-augmented generation* (RAG), which looks up scientific studies before answering, adds one check of its draft against plain rules of evidence (for example, newer evidence takes precedence). The research domain is Alzheimer's disease and related dementias. 

The full academic account is in the accompanying research paper: [title and link to be added].

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant baseline models and existing works.

## Research Results

**What was done.** The proposed system was built and run end to end, with the baseline and four control systems, under one frozen design, on Cochrane-review questions asked as of each review's date (only earlier studies can be used). The **primary set** is the research domain: 208 questions on Alzheimer's disease and related dementias, of which 48 name Alzheimer's disease (23%), 156 name dementia (15 name both) and 19 name neither; it is not an Alzheimer's-only evaluation. The **secondary set** is general medicine: 528 held-out questions (226 more were used while the study was built) and, for a pre-registered test, 850 new questions. Presenting the Alzheimer's disease set as primary was decided after all results were known: the protocol had named the held-out general-medicine questions as the primary test (`docs/protocol.md` §8), and one rule reads a difference on every set. Accuracy is the share of answers that match the verdict of the Cochrane review (*supported*, *refuted*, *not enough information*); differences are in percentage points ("points"). The baseline, adapted RAG², is this study's adaptation of a published medical RAG method (RAG²), not a full reproduction of it.

| System (same language model, same studies) | Primary set: Alzheimer's disease (n = 208) | Secondary set: general medicine (n = 528) |
|---|---|---|
| Always the most frequent verdict (no model) | 42.3% | 44.3% |
| No retrieval | 33.2% | 46.6% |
| Standard retrieval | 30.8% | 48.1% |
| Adapted RAG² (baseline) | 34.6% | 48.7% |
| Proposed system | 38.0% | 50.0% |

| Proposed system minus baseline | Difference (points) | 95% range | p-value |
|---|---|---|---|
| Primary set: Alzheimer's disease | +3.4 | −1.9 to +8.7 | 0.28 |
| Secondary set: general medicine | +1.3 | −1.9 to +4.7 | 0.51 |

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
├── tests/                      the test suite: one file per module (test_<module>.py), test_repository_guards.py (documents, committed results and layout),
│                               fixtures.py (synthetic data shared by the tests) and check_hermetic.py, which fails if the suite changes any file
└── experiments/
    └── medchange/              the pipeline and the general medical (as-of) benchmark
        │                       commands (each is run as a module of experiments.medchange, see How to Run)
        ├── build_benchmark.py, ad_benchmark.py, fresh_supply.py, fresh_benchmark.py, ad_cochrane_supply.py   the benchmark, the Alzheimer's disease set, the unused questions and the fresh split (counts only for new reviews)
        ├── pubmed_asof.py, freeze_candidates.py                                                              as-of evidence and the candidate pools
        ├── generate_answers.py, rag2_run.py, rag2_pipeline.py                                                standard answering (B0, B1), adapted RAG² and verification, and one command per phase
        ├── analyze_rag2.py, analyze_fresh.py, class_balance.py, report.py                                    analysis and tables
        ├── ad_subgroup.py, ad_case_study.py                                                                  the Alzheimer's-named questions of the Alzheimer's disease set
        ├── label_audit.py, verdict_consistency.py                                                            audits of the reference labels and of the stated verdicts
        │                       libraries (imported by the commands; no library imports a command)
        ├── benchmark.py, selection.py, released_answers.py                                                   the MedChange release as as-of items, the questions no split uses, the release's closed-book answers
        ├── eutils.py, abstracts.py, encoders.py                                                              PubMed access, what is read from a record, the retrieval models
        ├── prompts.py, arms.py, generator.py, files.py                                                       prompts and verdict parsing, the standard arms B0 and B1, the local model with its recorded configuration, file helpers
        ├── rag2.py, pipeline_support.py                                                                      adapted RAG² and verification, helpers of the phase driver
        ├── analysis.py, scoring.py, stats.py                                                                 measurements on the answers, scoring and statistics
        ├── manifest.json, manifest_ad.json, manifest_fresh.json                                              counts, hashes and item lists of the three question sets (no question text)
        └── results/            committed results without source text; results/earlier_stages/ holds the results of the earlier approaches (their code is in Git history)
```

The working data (`experiments/medchange/data/`) and model files (`models/`) are not committed.

## How to Run

Commands are run from the repository root; model files and platform notes are in `docs/reproducibility.md`.

```bash
# Install (numpy only; the model backends come with the extras) and run every test
pip install -e .
python -m unittest discover -s tests -t .

# Rebuild the benchmark from the MedChange release (once)
git clone https://github.com/jvladika/MedChange ../MedChange
python -m experiments.medchange.build_benchmark --medchange-dir ../MedChange
python -m experiments.medchange.ad_benchmark --medchange-dir ../MedChange

# Run the study: one resumable command per phase; --commit commits the results and never pushes
python -m experiments.medchange.rag2_pipeline dev --model-path models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit
python -m experiments.medchange.rag2_pipeline confirm --go --model-path models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit
python -m experiments.medchange.rag2_pipeline ad --go --model-path models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --judge-path models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --commit

# Questions of the Alzheimer's disease set that name Alzheimer's disease (from the answers on file; no model is run)
python -m experiments.medchange.ad_subgroup

# Per-verdict behaviour: predicted shares, recall per verdict, macro-F1 (from the answers on file; no model is run)
python -m experiments.medchange.class_balance

# Count the questions of MedRevQA that no split uses yet (counts only)
python -m experiments.medchange.fresh_supply --medchange-dir ../MedChange

# Count new Cochrane reviews on Alzheimer's disease and dementia published after MedRevQA's newest review (counts only; needs the network)
python -m experiments.medchange.ad_cochrane_supply --medchange-dir ../MedChange

# The pre-registered test on fresh questions: select the questions, then (after the freeze, docs/reproducibility.md section 11) run and analyse it
python -m experiments.medchange.fresh_benchmark --medchange-dir ../MedChange
python -m experiments.medchange.rag2_pipeline fresh --go --model-path models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit

# Re-analyse and report (no model is run; needs the rebuilt benchmark and the run's files in experiments/medchange/data/)
python -m experiments.medchange.analyze_rag2 --split confirm --out-dir experiments/medchange/results
python -m experiments.medchange.report --split confirm --medchange-dir ../MedChange
```

The held-out (`confirm`), `ad` and `fresh` phases refuse to start without `--go`, without the development report, and unless the design
record on `origin/main` equals the current design; the `fresh` phase also needs the pre-registration marked in force and its question
list on `origin/main`. `--dry-run` prints a phase's steps without running them, and `rag2_pipeline status` shows which phase is done.
