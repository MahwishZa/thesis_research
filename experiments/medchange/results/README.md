# MedChange results (tracked)

Outputs of real runs that contain **no source text** and are expensive to regenerate: generated answers
(`answers_<split>.jsonl`, with admitted PMIDs, and its generator record `answers_<split>.config.json`), zero-shot
helpfulness scores (`helpfulness_<split>.jsonl`), saved analyses (`analysis_<split>.json`, written by
`analyze --out`) and, for stage 2, the per-paper stance records (`stance_*.jsonl` with `.config.json`), the
the synthesis predictions (`synthesis_<split>.jsonl`) and the
reports below. Copy them here after a run and commit them; the frozen pools and abstracts stay in the
gitignored `../data/`. See
`docs/reproducibility.md` §5.

Committed: `answers_dev.jsonl`, `answers_dev.config.json` and `analysis_dev.json` (all six arms, dev split; `docs/log.md` Phase 25), and
`dev_audit.json` / `.md` (`dev_audit.py`: the descriptive figures behind the stage-2 decision, recomputed from the
committed answers; `docs/log.md` Phase 27).

Written here directly by a command with `--out-dir` (not yet committed, because the runs are pending or the
file is still on the researcher's laptop): `error_analysis_dev.json` / `.md` (`error_analysis.py`; `docs/log.md`
Phase 26), `diagnostics_dev.json` / `.md` (`diagnostics.py`, step P0), `synthesis_cv_dev.md` and
`synthesis_model.json` (`synthesis fit`; the model file is committed **before** any confirmatory stance run) and
`stage2_analysis_<split>.json` / `.md` (`analyze_stage2.py`; the confirmatory one is produced once, after
the confirmatory run).
