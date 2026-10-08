# `experiments/medchange/` — the pipeline

An as-of evaluation, on Cochrane questions (a primary benchmark and a secondary dementia and Alzheimer's set), of an adapted
RAG² baseline and an evidence-criteria verification extension. The design, settings and decision rules are fixed in
`docs/protocol.md`, the approach is explained in `docs/methodology.md`, the metrics and results are in `docs/evaluation.md`, and
the commands and costs are in `docs/reproducibility.md`. This file maps code to steps. The earlier approaches (recency-aware
admission and the evidence-synthesis layer) are not part of the active code; their outputs are in `results/earlier_stages/` and
their code is in Git history (commit `f721bbb`).

| Module | Step | Network / models | Writes (`data/`, gitignored) |
|---|---|---|---|
| `benchmark.py`, `build_benchmark.py` | rebuild MedChangeQA from the MedChange release (refuses unless 512/512 items match), attach version dates, flag label noise, sample controls, seed the splits | none | `benchmark.jsonl`; `manifest.json` (tracked) |
| `ad_benchmark.py` | the dementia and Alzheimer's secondary set: MedRevQA questions naming dementia, Alzheimer's disease, mild cognitive impairment or cognitive decline, from reviews outside the development and held-out splits (by study group and by Cochrane ID), duplicates removed; appended as split `ad` | none (MedChange clone) | `benchmark.jsonl`; `manifest_ad.json` (tracked) |
| `headroom.py` | the benchmark authors' released closed-book answers of five language models, scored on the same questions (used by `report.py`) | none | stdout |
| `pubmed_asof.py` | as-of PubMed records per question: query, publication-date bounds, Cochrane records excluded, boundary-ambiguous records dropped | PubMed E-utilities | `pubmed_g0/<item>.json` |
| `freeze_candidates.py` | abstracts; MedCPT dense ranking and cross-encoder re-ranking; the frozen pool of 20 behind B1 | E-utilities, MedCPT | `abstracts.jsonl`, `frozen_<split>.jsonl` |
| `encoders.py` | the MedCPT query and article encoders and the cross-encoder (torch loaded lazily), and the stand-ins the tests use | MedCPT (Hugging Face) | – |
| `abstracts.py` | what the model reads of an abstract (title, RESULTS, CONCLUSIONS, at most 200 words) and the classification of a record's study design | none | – |
| `prompts.py` | the one standard answer prompt and the `VERDICT:` parser (B0, B1, R2) | none | – |
| `arms.py` | the admission rules and the arm-settings hash recorded with every B0 and B1 answer; also the completed stage-1 rules (B2, B3, P, C1), whose development answers are in `results/answers_dev.jsonl`; formula delegated to `src/temporal_filter/`. The current pipeline runs only B0 and B1 | none | – |
| `generate_answers.py` | one answer per (question, arm) with llama.cpp, resumable; records the generator configuration and refuses to resume under a different one. Used for B0 and B1 | GGUF model | `answers_<split>.jsonl`, `answers_<split>.config.json` |
| `rag2.py` | adapted RAG² and verification, pure logic: the rationale prompt, balanced retrieval by evidence type, cross-encoder re-ranking, the filter's P(yes) and admission, the evidence criteria and verification prompts, the `FINAL VERDICT:` parser, the unsupported-answer indicators, the design record | none | – |
| `rag2_run.py` | the resumable steps `rationale`, `lists`, `filter`, `answers` (R2, R2-RQ, R2-BR, R2-NF, R2C, R2V, R2V-ND) and optional `judge` (directness); records each step's configuration and refuses to resume under a different one | GGUF model; MedCPT for `lists`; no network | `rag2_rationales_<split>.jsonl`, `rag2_lists_<split>.jsonl`, `rag2_filter_<split>.jsonl`, `rag2_answers_<split>.jsonl`, `rag2_directness_<split>.jsonl` |
| `rag2_pipeline.py` | one resumable command per phase (`dev`, `confirm --go`, `ad --go`, `status`): integrity checks, steps, dev check, design record, findings, publishing, an environment record, optional commit (never pushes); refuses a held-out run unless the design record on `origin/main` equals the current design | as the steps | `results/RAG2_DEV_REPORT.md`, `results/rag2_design.json`, `results/RAG2_FINDINGS.md`, `results/RAG2_FINDINGS_AD.md`, `results/rag2_environment_<phase>.json` |
| `runner.py` | what the phase driver shares: expected item counts, command builder, step executor, the "is it on `origin/main`" guard, and a commit that never pushes | git | – |
| `scoring.py` | accuracy with Wilson intervals, per-class recall and macro-F1, paired comparisons (exact McNemar and paired bootstrap from `evaluation/stats.py`), Holm families, the sample size that can confirm an effect | none | – |
| `analyze_rag2.py` | the analysis: accuracy, per-class recall, outdated rate, unsupported-answer indicators, verifier behaviour, retrieval metrics, the primary test and the pre-declared reading of the +1 point requirement, secondary and ablation families with Holm, the label-stable subset, the Alzheimer's items of the main benchmark | none | `results/rag2_analysis_<split>.json`, `.md` |
| `label_audit.py` | independent re-labelling of every gold label with the authors' rubric by a second-family model (agreement, kappa, label-stable questions); held-out split only after the design record exists | second GGUF model | `label_audit_<split>.jsonl`; `results/label_audit_<split>.json`, `.md` |
| `consistency_auto.py` | parse-rate check and an independent judge's agreement with the stated verdict on a seeded sample | second GGUF model | `consistency_auto_<split>.jsonl`; `results/consistency_auto_<split>.*` |
| `report.py` | result tables in the layout of the base paper (systems against the benchmark, one generator with different admission methods, per-class recall), a LaTeX table and two figures, from the committed results and the released closed-book answers; no experiment is run | MedChange clone (closed-book rows); matplotlib (figures) | `results/report/` |

`data/` is gitignored because the MedChange release states no licence and abstracts are publisher text. Outputs worth keeping
are copied to `results/`, which is tracked (`results/README.md`; `docs/data.md` §4). `manifest.json` and `manifest_ad.json`
record the input-file hashes, counts and seed that rebuild `data/benchmark.jsonl` identically.

Tests (all use synthetic data, no network and no models): `evaluation/tests/unit/test_medchange_benchmark.py`,
`test_medchange_abstracts.py`, `test_medchange_arms.py`, `test_medchange_scoring.py`, `test_medchange_runner.py`,
`test_medchange_audits.py`, `test_medchange_report.py` and `test_medchange_rag2.py` (retrieval lists, filter, criteria prompts and
parsers, the step runner with fake models, the analysis and its reading of the requirement, the pipeline's checks and guards, the
`ad` builder, and a fake-model run of every step). The statistics and the recency formula are tested in `test_stats.py` and
`test_temporal_filter.py`; `test_scope_invariants.py` guards the documentation, the layout and the package list.
