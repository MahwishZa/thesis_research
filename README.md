# Thesis Title (Temporary Placeholder)

## Overview

AI chatbots are increasingly used to answer medical questions, but they can be wrong, base an answer on studies that do not
really address the question, or repeat advice that newer research has overturned. A common remedy is *retrieval-augmented
generation* (RAG): before answering, the system looks up scientific studies and reads them.

This research tests whether a small language model running on an ordinary laptop answers medical questions more correctly
when RAG gets one extra step: after drafting an answer from the studies it found, the model checks the draft against plain
rules of evidence (only studies that really test the treatment count, trials and systematic reviews weigh most, newer
evidence takes precedence). The idea is compared with an adapted version of a published medical RAG method (RAG²; not a full
reproduction of it) on real questions taken from Cochrane systematic reviews, each asked as of the day the review was
published, so that only earlier studies can be used. The goal agreed with the supervisor is an improvement of at least 1
percentage point in the share of correct answers.

## Research Objectives

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance compared with relevant
   baseline models and existing works.

## Research Results

**What was measured.** Every system answered the same 528 medical questions. They were kept apart while the method was
designed, and the method was run on them once. Each answer is one of three verdicts: *supported*, *refuted* or *not enough
information*. An answer counts as correct when it matches the verdict of the newest Cochrane review for that question, and
*accuracy* is the share of questions answered correctly. Differences are given in percentage points ("points" for short).
The figure in brackets is the 95% range: the span of accuracies that fits the data, so a wide span means the figure is
imprecise.

| System | What it does | Accuracy (95% range) |
|---|---|---|
| No evidence (B0) | answers from the model's own knowledge | 46.6% (42.4–50.9) |
| Standard retrieval (B1) | looks up the 5 most relevant studies, then answers | 48.1% (43.9–52.4) |
| **Baseline: adapted RAG² (R2)** | searches more carefully and filters out unhelpful studies, then answers | 48.7% (44.4–52.9) |
| Rules of evidence only (R2C) | reads the baseline's studies with the rules of evidence, without a first draft | 48.5% (44.2–52.7) |
| **Proposed system (R2V)** | takes the baseline's answer and checks it against the rules of evidence | **50.0%** (45.8–54.2) |
| Proposed system without dates (R2V-ND) | the same check, without publication dates and the "newer evidence first" rule | 49.1% (44.8–53.3) |

**Main finding.** The proposed system answered 264 of the 528 questions correctly and the baseline 257: seven more, a gain
of **1.3 percentage points**. On the 81 questions where the two systems disagreed, the proposed system was right on 44 and
the baseline on 37.

**Did it reach the goal?** The goal was judged by a rule fixed before the test: *met and confirmed* if the gain is at least
1 point and the p-value is below 0.05 (the usual bar for calling a difference real rather than luck); *met as a point
estimate, not confirmed* if the gain is at least 1 point but the p-value is not below 0.05; *not met* otherwise. The gain
reaches 1 point, but it is too small to rule out luck: if the two systems were equally good, a difference at least this
large in either direction would appear about half of the time (p = 0.51). The goal is therefore **met as a point estimate,
not confirmed**. With 528 questions, only gains of roughly 4 to 6 points or more could be confirmed.

**Are the other differences real?** The same comparison for other pairs of systems on the same questions. A difference is
*confirmed* only if its p-value is below 0.05 and its 95% range stays above zero; p-values after the first row are adjusted
for making several comparisons. None of the differences is confirmed.

| Comparison | Difference (points) | 95% range | p-value | Conclusion |
|---|---|---|---|---|
| **Proposed vs baseline (R2V − R2)** | **+1.3** | −1.9 to +4.7 | 0.51 | **met as a point estimate, not confirmed** |
| Proposed vs baseline, only the 353 questions whose correct verdict changed between review versions | +3.4 | −0.8 to +7.4 | 0.89 | not confirmed |
| Baseline vs standard retrieval (R2 − B1) | +0.6 | −3.0 to +4.2 | 1.0 | not confirmed |
| Proposed vs standard retrieval (R2V − B1) | +1.9 | −2.5 to +6.1 | 1.0 | not confirmed |
| Rules of evidence only vs baseline (R2C − R2) | −0.2 | −3.6 to +3.4 | 1.0 | not confirmed |
| Proposed vs rules of evidence only (R2V − R2C) | +1.5 | −1.7 to +4.7 | 1.0 | not confirmed |
| Proposed vs proposed without dates (R2V − R2V-ND) | +0.9 | −1.1 to +3.0 | 1.0 | not confirmed |

**What the extra check does.** For the 447 questions where the baseline had studies to read, the check changed the
baseline's verdict 100 times: 44 changes turned a wrong answer into a right one and 37 did the opposite. It moved answers
away from *supported* towards *refuted* and *not enough information*. Among the 126 questions whose reference verdict is
*refuted*, the proposed system got 24 right against 15 for the baseline; among the 168 *not enough information* questions,
72 against 56; among the 234 *supported* questions, 168 against 186.

**Practice run.** While the method was being finalised, the same systems were tried on 226 other questions. There the
proposed system scored 49.6% against 50.0% for the baseline (−0.4 points; 95% range −5.8 to +4.9), the opposite direction to
the final test, which is one more reason to treat the +1.3 points with caution.

| System | Accuracy on the 226 practice questions |
|---|---|
| No evidence (B0) | 45.1% |
| Standard retrieval (B1) | 53.1% |
| Baseline: adapted RAG² (R2) | 50.0% |
| Rules of evidence only (R2C) | 49.1% |
| Proposed system (R2V) | 49.6% |
| Proposed system without dates (R2V-ND) | 52.2% |

**Search quality.** The baseline's search read fewer studies than standard retrieval but better ones. "Directly test" was
judged by a second, independent AI model.

| System | Studies read per question | Questions with no study found | Studies that directly test the question's treatment |
|---|---|---|---|
| Standard retrieval (B1) | 5.0 | 0.0% | 28.6% |
| Baseline and proposed system (same studies) | 3.3 | 15.3% | 57.3% |

**For context.** The authors of the benchmark published the answers of five other AI models, some of them far larger, to the
same questions without any lookup: Qwen2.5-7B 52.8%, GPT-4o-mini 52.8%, DeepSeek-V3 52.3%, Llama-3.3-70B 50.8% and
Mistral-24B 50.8%. The small laptop model scored 46.6–50.0%, numerically below all five. This is not a like-for-like
comparison, because the models differ in size, training and instructions.

**How reliable are the reference answers?** The reference verdicts were themselves produced by an AI model from the reviews'
conclusions. A second, independent AI model reproduced 81.4% of them on these questions, so some references are debatable
and small differences between systems should be read with that in mind.

**In short.** The extra check gave a small gain (+1.3 points) that cannot be told apart from luck, the baseline was not
clearly better than simple retrieval (+0.6 points), and in the practice run the proposed system was slightly behind. The
results therefore do not show that the extra check improves accuracy. Full tables: `docs/evaluation.md` §6.

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
└── experiments/medchange/      the pipeline
    ├── build_benchmark.py, ad_benchmark.py, benchmark.py, manifest.json, manifest_ad.json    the benchmark and its splits
    ├── pubmed_asof.py, freeze_candidates.py, encoders.py, abstracts.py                       as-of evidence and retrieval models
    ├── prompts.py, arms.py, generate_answers.py                                              standard answering (B0, B1)
    ├── rag2.py, rag2_run.py, rag2_pipeline.py, runner.py                                     adapted RAG² and verification
    ├── scoring.py, analyze_rag2.py, report.py, headroom.py, label_audit.py, consistency_auto.py   analysis and audits
    └── results/                committed results without source text; results/earlier_stages/ holds the earlier approaches
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

# Re-analyse and report (no model is run; needs the rebuilt benchmark and the run's files in experiments/medchange/data/)
python -m experiments.medchange.analyze_rag2 --split confirm --out-dir experiments/medchange/results
python -m experiments.medchange.report --split confirm --medchange-dir ../MedChange
```

The held-out (`confirm`) and `ad` phases refuse to start without `--go`, without the development report, and unless the design
record on `origin/main` equals the current design. `--dry-run` prints a phase's steps without running them, and
`rag2_pipeline status` shows which phase is done.
