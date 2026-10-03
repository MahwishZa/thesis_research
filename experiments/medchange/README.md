# `experiments/medchange/` — the primary pipeline

An as-of evaluation of evidence-admission rules on Cochrane questions whose verdict changed between
review versions. The protocol, settings and gates are fixed in `docs/experiment_plan.md`; this file maps
code to steps. Commands and measured costs: `docs/reproducibility.md` §3.

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

`data/` is gitignored because the MedChange release states no licence and abstracts are publisher text.
Outputs worth keeping (answers, helpfulness scores, saved analyses) are copied into `results/`, which is
tracked; see `docs/reproducibility.md` §5. `manifest.json` records the input-file hashes, counts and seed
that rebuild `data/benchmark.jsonl` identically.

Tests: `evaluation/tests/unit/test_medchange_benchmark.py`, `test_medchange_arms.py`.
