# Thesis Title (Temporary Placeholder)

## Overview

AI chatbots can give wrong medical answers, rely on studies that do not address the question, or repeat advice that newer research has overturned. This research tests whether a small language model answers medical questions more correctly when *retrieval-augmented generation* (RAG), which looks up scientific studies before answering, adds one check of its draft against plain rules of evidence (for example, newer evidence takes precedence). The research domain is Alzheimer's disease and related dementias. The full academic account is in the accompanying research paper: [title and link to be added].

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant baseline models and existing works.

## Research Results

**What was measured.** Two sets of medical-evidence questions taken from Cochrane systematic reviews, each asked as of its
review's date so that only earlier studies can be used: 528 general-medicine questions and 208 questions on Alzheimer's disease and related dementias, of
which 48 name Alzheimer's disease (23%), 156 name dementia (15 name both) and 19 name neither (other cognitive impairment). The 48 are reported as a subgroup; the 208 are not an Alzheimer's-only evaluation. Each answer is one of three verdicts
(*supported*, *refuted*, *not enough information*) and counts as correct when it matches the verdict of the Cochrane review;
*accuracy* is the share of correct answers, and differences are given in percentage points ("points" for short).

| System (same language model, same studies) | General medicine (n = 528) | Alzheimer's disease set (n = 208) |
|---|---|---|
| No retrieval | 46.6% | 33.2% |
| Standard retrieval | 48.1% | 30.8% |
| Adapted RAG² (baseline) | 48.7% | 34.6% |
| Proposed extra check | 50.0% | 38.0% |

| Proposed extra check minus baseline | Difference (points) | 95% range | p-value |
|---|---|---|---|
| General medicine | +1.3 | −1.9 to +4.7 | 0.51 |
| Alzheimer's disease set | +3.4 | −1.9 to +8.7 | 0.28 |

**How to read this.** The 95% range is the span of differences that fits the data, and the p-value is the chance of seeing a gap at
least this large if the two systems were really equally accurate (below 0.05 would count as evidence). The extra check scored 1.3 points
higher on general medicine and 3.4 points higher on the Alzheimer's disease set, but both ranges include zero and negative values, so
the gaps cannot be told apart from chance: they are **positive but not confirmed**. With these numbers of questions only gains of
roughly 4 to 6 points (general medicine) or 6 to 10 points (Alzheimer's disease set) could be confirmed. The results do not show that
the extra check improves accuracy, and the Alzheimer's-named subgroup (48 questions) is too small to describe Alzheimer's disease on its own. The reference answers were produced by a
language model; a second, independent model reproduces 81.4% of them on the general-medicine questions.

**The pre-registered test.** The last test asked the question again on 850 new general-medicine questions, with a measure that gives each of the three verdicts the
same weight (macro-F1). The extra check did not change it: the proposed check minus the baseline was −0.002 (95% range −0.027 to +0.023), which counts as **no evidence**
of an improvement. Accuracy was 45.4% with the extra check and 47.2% for the baseline (−1.8 points, range −4.2 to +0.8, which cannot be told apart from chance). The gains
seen earlier on the other two sets, which led to this test, did not repeat on new questions. Further tables, controls, the earlier approaches that did not work, the answers to the research questions (§7) and the limitations (§8) are in
`docs/evaluation.md`.

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
        ├── analyze_rag2.py, analyze_fresh.py, class_balance.py, report.py, headroom.py                       analysis and tables
        ├── ad_subgroup.py, ad_case_study.py                                                                  the Alzheimer's-named questions of the Alzheimer's disease set
        ├── label_audit.py, verdict_consistency.py                                                            audits of the reference labels and of the stated verdicts
        │                       libraries (imported by the commands; no library imports a command)
        ├── benchmark.py, selection.py, released_answers.py                                                   the MedChange release as as-of items, the questions no split uses, the release's closed-book answers
        ├── eutils.py, abstracts.py, encoders.py                                                              PubMed access, what is read from a record, the retrieval models
        ├── prompts.py, arms.py, generator.py, files.py                                                       prompts and verdict parsing, the arms, the local model with its recorded configuration, file helpers
        ├── rag2.py, pipeline_support.py                                                                      adapted RAG² and verification, helpers of the phase driver
        ├── analysis.py, scoring.py, stats.py                                                                 measurements on the answers, scoring and statistics
        ├── manifest.json, manifest_ad.json, manifest_fresh.json                                              counts, hashes and item lists of the three question sets (no question text)
        ├── stage1/             the evidence records and the recency formula of the earlier approach (result of record)
        └── results/            committed results without source text; results/earlier_stages/ holds the earlier approaches
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
