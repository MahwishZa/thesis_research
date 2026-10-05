# `experiments/medchange/` — the primary pipeline

An as-of evaluation, on Cochrane questions whose verdict changed between review versions, of evidence
admission (stage 1) and of an evidence-synthesis layer (stage 2).
The protocol, settings and gates are fixed in `docs/experiment_plan.md`; this file maps code to steps.
Commands and costs: `docs/reproducibility.md` §3.

| Module | Step | Network / models | Writes (`data/`, gitignored) |
|---|---|---|---|
| `benchmark.py`, `build_benchmark.py` | rebuild MedChangeQA from the MedChange release (refuses unless 512/512 items match), attach version dates, flag label noise, seed splits | none | `benchmark.jsonl`; `manifest.json` (tracked) |
| `headroom.py` | how often released models give the current verdict without retrieval | none | stdout |
| `pubmed_asof.py` | as-of PubMed hits per item; gate G0 | PubMed E-utilities | `pubmed_g0/<item>.json` |
| `freeze_candidates.py` | abstracts, MedCPT dense + cross-encoder rerank, frozen pool of 20 | E-utilities, MedCPT | `abstracts.jsonl`, `frozen_<split>.jsonl` |
| `helpfulness.py` | zero-shot Flan-T5 P(yes) per pool candidate (untrained; **not RAG²**) | Flan-T5-large | `helpfulness_<split>.jsonl` |
| `arms.py` | admission rules B0, B1, B2, B3, P, C1 with fixed settings; formula delegated to `src/proposed/` | none | – |
| `prompts.py` | the one answer prompt and the `VERDICT:` parser | none | – |
| `generate_answers.py` | one answer per (item, arm) with llama.cpp, resumable; records the generator configuration and refuses to resume under a different one | GGUF model | `answers_<split>.jsonl`, `answers_<split>.config.json` |
| `analyze.py` | accuracy, outdated rate, retrieval-level metrics, paired McNemar + Holm, gates G2/G3 | none | stdout or `--out` |
| `error_analysis.py` | where wrong answers on changed items go wrong: retrieval miss, admission miss, evidence admitted but still wrong; accuracy with/without update-window evidence | none | `results/error_analysis_<split>.json`, `.md` |
| `label_audit.py` | independent re-labelling of every gold label with the authors' rubric by a second-family model (agreement, kappa, label-stable items); confirmatory split only after the frozen model exists | second GGUF model | `label_audit_<split>.jsonl`; `results/label_audit_<split>.json`, `.md` |
| `consistency_auto.py` | parse-rate gate and an independent judge's agreement with the stated verdict on a seeded sample | second GGUF model | `consistency_auto_<split>.jsonl`; `results/consistency_auto_<split>.*` |
| `report.py` | paper-style result tables (systems × benchmark; one generator × admission methods; per-class recall; dev cross-validation), a LaTeX table and four figures from the committed results and the benchmark authors' released closed-book answers; no experiment is run | clone of the MedChange release (closed-book rows); matplotlib (figures) | `results/report/` |
| `findings.py` | writes `results/DEV_REPORT.md` and `results/FINDINGS.md` in plain language from the computed results | none | `results/DEV_REPORT.md`, `results/FINDINGS.md` |
| `pipeline.py` | one resumable command per phase (`dev`, `confirm --go`, `status`): integrity checks, gates, steps, reports, optional commit and push; refuses to start the confirmatory phase unless the frozen model is on `origin/main` and the generator equals the dev one | the models above | the files of the steps it runs |
| `dev_audit.py` | the figures behind the stage-2 decision, recomputed from the committed dev answers: constant-answer baseline, per-class recall and abstention, order sensitivity, evidence age by gold class, does refitting help, majority vote; dev only | none | `results/dev_audit.json`, `.md` |
| `diagnostics.py` | stage-2 step P0, no model: is the helpfulness input truncated at 512 tokens, how many abstracts have labelled RESULTS / CONCLUSIONS, study types among the top 8, B1 verdicts by evidence composition | tokenizer (optional) | `results/diagnostics_<split>.json`, `.md` |
| `stance.py` | one stance judgement (supports / contradicts / neither) per paper for the first 8 pool candidates, from title + RESULTS + CONCLUSIONS; two wordings; first-token probabilities; `--pilot` = 40 dev items, both wordings and the irrelevant-paper control; resumable, refuses a changed configuration | GGUF model (or Flan-T5-large as the declared fallback) | `stance_<split>.jsonl`, `stance_pilot.jsonl`, with `.config.json` |
| `stance_check.py` | gate 1 of the pilot: `report` (machine checks only: wording agreement, control papers, invalid rate on real papers, speed; gate verdict) | none | (`--out` for the saved report) |
| `synthesis.py` | the evidence-synthesis layer: features (signed stance, no-stance share, conflict, informative mass), paper weights, regularised multinomial logistic regression; `fit` = repeated cross-validation on dev, gate 2, selection, frozen model; `predict` = apply the frozen model to the confirmatory split (refuses without it) | none | `results/synthesis_model.json`, `results/synthesis_cv_dev.md`, `synthesis_<split>.jsonl` |
| `analyze_stage2.py` | stage-2 analysis: accuracy with Wilson intervals, per-class recall, macro-F1, NOT ENOUGH INFORMATION share; primary family RQ1 (B1 vs B0) and RQ2 (selected hybrid vs B1R) with Holm; secondary family | none | `results/stage2_analysis_<split>.json`, `.md` |

`data/` is gitignored because the MedChange release states no licence and abstracts are publisher text.
Outputs worth keeping (answers, helpfulness scores, saved analyses) are copied into `results/`, which is
tracked; see `docs/reproducibility.md` §5. `manifest.json` records the input-file hashes, counts and seed
that rebuild `data/benchmark.jsonl` identically.

Tests: `evaluation/tests/unit/test_medchange_benchmark.py`, `test_medchange_arms.py`,
`test_medchange_stance.py` (stance, pilot checks, diagnostics, fakes in place of the model),
`test_medchange_synthesis.py` (features, weights, fitting, selection, gate 2, the confirmatory analysis and its pre-declared reading),
`test_medchange_dev_audit.py`, `test_medchange_audits.py`, `test_medchange_report.py` and `test_medchange_pipeline.py`; all use synthetic data.
