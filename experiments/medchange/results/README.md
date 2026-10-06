# MedChange results (tracked)

Outputs of real runs that contain **no source text** and are expensive to regenerate. The frozen pools and
abstracts stay in the gitignored `../data/` and are rebuilt from the manifest (`docs/reproducibility.md` §5).
`pipeline.py` copies the stage-2 outputs here and, with `--commit`, commits them (you push by hand).

| File | Produced by | Notes |
|---|---|---|
| `answers_dev.jsonl`, `answers_dev.config.json` | `generate_answers` (all six stage-1 arms, dev) | `docs/log.md` Phase 25 |
| `analysis_dev.json` | `analyze --split dev --out` | |
| `error_analysis_dev.json` / `.md` | `error_analysis` | Phase 26 |
| `dev_audit.json` / `.md` | `dev_audit` | Phase 27 |
| `diagnostics_dev.json` / `.md` | `diagnostics` (step P0) | Phase 28 |
| `report/` (`REPORT.md`, `tables.tex`, `report_data_dev.json`, `report_data_confirm.json`, four PNG figures) | `report` | `REPORT.md`, `tables.tex` and the figures are rewritten by each run, so the committed ones are those of the last run, the confirmatory split (stage-1/2 run); `report_data_<split>.json` is kept per split. The dev version was first written in Phase 31 |
| `stance_pilot.jsonl`, `stance_dev.jsonl` (+ `.config.json`), `synthesis_dev.jsonl`, `synthesis_cv_dev.md`, `synthesis_model.json`, `label_audit_dev.*`, `consistency_auto_dev.*`, `DEV_REPORT.md` | `pipeline dev` | dev split |
| `answers_confirm.*`, `label_audit_confirm.*`, `consistency_auto_confirm.*`, `stage2_analysis_confirm.*`, `FINDINGS.md` | `pipeline confirm --go` | written once (2026-10-05), after the frozen model was on `origin/main`. Gate 2 had failed on dev, so only RQ1 was tested and no `stance_confirm.*` or `synthesis_confirm.*` file exists |
| `rag2_rationales_<split>.jsonl`, `rag2_lists_<split>.ids.jsonl`, `rag2_filter_<split>.jsonl`, `rag2_answers_<split>.jsonl`, `rag2_directness_<split>.jsonl` (+ `.config.json`), `rag2_analysis_<split>.*`, `RAG2_DEV_REPORT.md`, `rag2_design.json`, `RAG2_FINDINGS.md`, `RAG2_FINDINGS_AD.md` and the `ad` versions of these files (plus `answers_ad.*`) | `rag2_pipeline dev` / `confirm --go` / `ad --go` | realigned study; the candidate lists without titles or abstracts; the design record is committed before the held-out run |
