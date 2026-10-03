# `experiments/medchange/` — the primary pipeline

An as-of evaluation, on Cochrane questions whose verdict changed between review versions, of evidence
admission (stage 1, done on dev) and of an evidence-synthesis layer (stage 2, code built, pilot pending).
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
| `consistency.py` | gate G1 human check of stated verdicts | none | `consistency_sheet.csv` |
| `dev_audit.py` | the figures behind the stage-2 decision, recomputed from the committed dev answers: constant-answer baseline, per-class recall and abstention, order sensitivity, evidence age by gold class, does refitting help, majority vote; dev only | none | `results/dev_audit.json`, `.md` |
| `diagnostics.py` | stage-2 step P0, no model: is the helpfulness input truncated at 512 tokens, how many abstracts have labelled RESULTS / CONCLUSIONS, study types among the top 8, B1 verdicts by evidence composition | tokenizer (optional) | `results/diagnostics_<split>.json`, `.md` |
| `stance.py` | one stance judgement (supports / contradicts / neither) per paper for the first 8 pool candidates, from title + RESULTS + CONCLUSIONS; two wordings; first-token probabilities; `--pilot` = 40 dev items, both wordings and the irrelevant-paper control; resumable, refuses a changed configuration | GGUF model (or Flan-T5-large as the declared fallback) | `stance_<split>.jsonl`, `stance_pilot.jsonl`, with `.config.json` |
| `stance_check.py` | gate 1 of the pilot: `report` (machine checks), `export` (the 40-paper hand-check sheet and its key), `score` (accuracy of each wording, the chosen wording, gate verdict) | none | `stance_handcheck.csv`, `stance_handcheck_key.json`, `stance_choice.json` |
| `synthesis.py` | the evidence-synthesis layer: features (signed stance, no-stance share, conflict, informative mass), paper weights, regularised multinomial logistic regression; `fit` = repeated cross-validation on dev, gate 2, selection, frozen model; `predict` = apply the frozen model to the confirmatory split (refuses without it) | none | `results/synthesis_model.json`, `results/synthesis_cv_dev.md`, `synthesis_<split>.jsonl` |
| `analyze_stage2.py` | stage-2 analysis: accuracy with Wilson intervals, per-class recall, macro-F1, NOT ENOUGH INFORMATION share; primary family RQ1 (B1 vs B0) and RQ2 (selected hybrid vs B1R) with Holm; secondary family | none | `results/stage2_analysis_<split>.json`, `.md` |

`data/` is gitignored because the MedChange release states no licence and abstracts are publisher text.
Outputs worth keeping (answers, helpfulness scores, saved analyses) are copied into `results/`, which is
tracked; see `docs/reproducibility.md` §5. `manifest.json` records the input-file hashes, counts and seed
that rebuild `data/benchmark.jsonl` identically.

Tests: `evaluation/tests/unit/test_medchange_benchmark.py`, `test_medchange_arms.py`,
`test_medchange_stance.py` (stance, pilot checks, diagnostics, fakes in place of the model),
`test_medchange_synthesis.py` (features, weights, fitting, selection, gate 2, the confirmatory analysis) and
`test_medchange_dev_audit.py`; all use synthetic data. **Status:** the stage-2 modules have been run only on synthetic data; the first run
on real pools is the P0 diagnostics and the pilot.
