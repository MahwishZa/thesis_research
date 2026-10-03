# MedChange results (tracked)

Outputs of real runs that contain **no source text** and are expensive to regenerate. The frozen pools and
abstracts stay in the gitignored `../data/` and are rebuilt from the manifest (`docs/reproducibility.md` §5).
`pipeline.py` copies the stage-2 outputs here and, with `--commit`, commits and pushes them.

| File | Produced by | Status |
|---|---|---|
| `answers_dev.jsonl`, `answers_dev.config.json` | `generate_answers` (all six stage-1 arms, dev) | committed (`docs/log.md` Phase 25) |
| `analysis_dev.json` | `analyze --split dev --out` | committed |
| `error_analysis_dev.json` / `.md` | `error_analysis` | committed (Phase 26) |
| `dev_audit.json` / `.md` | `dev_audit` | committed (Phase 27) |
| `diagnostics_dev.json` / `.md` | `diagnostics` (step P0) | committed (Phase 28) |
| `stance_pilot.jsonl`, `stance_dev.jsonl` (+ `.config.json`), `synthesis_dev.jsonl`, `synthesis_cv_dev.md`, `synthesis_model.json`, `label_audit_dev.*`, `consistency_auto_dev.*`, `DEV_REPORT.md` | `pipeline dev` | pending (the pilot file is still only on the researcher's laptop until `pipeline dev --commit`) |
| `answers_confirm.*`, `stance_confirm.*`, `synthesis_confirm.*`, `label_audit_confirm.*`, `consistency_auto_confirm.*`, `stage2_analysis_confirm.*`, `FINDINGS.md` | `pipeline confirm --go` | pending; produced once, after the frozen model is on `origin/main` |
