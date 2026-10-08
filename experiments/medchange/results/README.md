# MedChange results (tracked)

Outputs of real runs that contain **no source text** and are expensive to regenerate. The frozen pools and abstracts stay in the
gitignored `../data/` and are rebuilt from the manifests (`docs/reproducibility.md` §3). The phase drivers copy their outputs
here and, with `--commit`, commit them; the researcher pushes by hand.

| File | Produced by | Notes |
|---|---|---|
| `answers_dev.jsonl`, `answers_dev.config.json` | `generate_answers` | B0 and B1 of the development split (reused by every comparison) and the completed stage-1 arms B2, B3, P and C1 |
| `answers_confirm.jsonl`, `answers_confirm.config.json` | `generate_answers` | B0 and B1 of the held-out split |
| `rag2_rationales_<split>.jsonl`, `rag2_lists_<split>.ids.jsonl`, `rag2_filter_<split>.jsonl`, `rag2_answers_<split>.jsonl`, `rag2_directness_<split>.jsonl` (each with `.config.json`) | `rag2_run` through `rag2_pipeline` | adapted RAG² and verification, for `dev` and `confirm`: the rationales, the candidate lists without titles or abstracts, the filter judgements, the answers of R2, R2C, R2V and R2V-ND, and the directness judgements |
| `rag2_analysis_<split>.json`, `.md` | `analyze_rag2` | the analysis of each split, with the pre-declared reading of the requirement |
| `RAG2_DEV_REPORT.md` | `rag2_pipeline dev` | the development report with the pre-declared dev check (status REVISE ONCE: direction failed) |
| `rag2_design.json` | `rag2_pipeline dev` | the design record: settings, prompt hashes and the generator file's hash, committed before the held-out run |
| `RAG2_FINDINGS.md` | `rag2_pipeline confirm` | the held-out findings, written once (2026-10-08) |
| `rag2_environment_<phase>.json` | `rag2_pipeline` | package versions, platform and code commit of a phase (informational) |
| `label_audit_<split>.*`, `consistency_auto_<split>.*` | `label_audit`, `consistency_auto` | the independent re-labelling of the gold labels and the check of stated verdicts (`dev` and `confirm`) |
| `report/` | `report` | `REPORT.md` (tables of the held-out split), `tables.tex`, `report_data_confirm.json`, two figures |
| `earlier_stages/` | the removed stage code | results of record of stage 1 and stage 2 (below) |

**Not present because not run:** `rag2_*` files for the ablations R2-RQ, R2-BR and R2-NF (optional, development split), and every
file of the dementia and Alzheimer's set (`answers_ad.*`, `rag2_*_ad.*`, `rag2_analysis_ad.*`, `RAG2_FINDINGS_AD.md`,
`rag2_environment_ad.json`) until `rag2_pipeline ad --go` has been run.

## `earlier_stages/`

Outputs of the two completed, superseded approaches (`docs/methodology.md` §10, `docs/protocol.md` §9, `docs/evaluation.md`
§6.3). Their code is not in the active tree; `git checkout f721bbb` restores it.

| Files | What they are |
|---|---|
| `DEV_REPORT.md`, `FINDINGS.md`, `stage2_analysis_confirm.*` | the stage-2 development report, the plain-language findings and the held-out analysis of B1 against B0 (RQ1) |
| `analysis_dev.json`, `error_analysis_dev.*`, `dev_audit.*`, `diagnostics_dev.*` | the stage-1 analysis of the development split, the error analysis, the audit of the development answers and the input diagnostics (`docs/log.md` Phases 25 to 28) |
| `stance_pilot.*`, `stance_dev.*`, `synthesis_dev.jsonl`, `synthesis_cv_dev.md`, `synthesis_model.json` | the per-paper judgements, the layer's predictions, its cross-validation and the frozen model of stage 2 |

## Notes on generated files

* Generated files keep the text they were generated with. Their pointers to documents use the names current at the time:
  `docs/experiment_plan.md` (until 2026-10-05) and `docs/experimentation.md` (until 2026-10-08) are now `docs/protocol.md` and
  `docs/evaluation.md`; the files in `earlier_stages/` also cite the stage-2 protocol, which is in Git history
  (`git show 92e3aaf:docs/experiment_plan.md`).
* The analysis tables of `dev` and `confirm` (`rag2_analysis_*.md`, `.json`) were written when stored rates were rounded to four
  decimals before being shown with one, so a few cells differ by 0.1 from the exact value: on the held-out split R2C and R2V recall of
  REFUTED are 19.0% (shown 19.1%) and R2V-ND accuracy is 49.1% (259 of 528; shown 49.0%); on the development split R2C recall of
  REFUTED is 32.7% (32.6%), R2C's share of NOT ENOUGH INFORMATION answers 23.5% (23.4%), R2V's outdated-verdict rate 37.7% (37.8%) and the
  SR/MA share of R2's admitted abstracts 25.3% (25.2%). The code now stores six decimals; `python -m experiments.medchange.analyze_rag2
  --split <dev|confirm> --out-dir experiments/medchange/results --label-audit experiments/medchange/results/label_audit_<split>.jsonl`,
  run where the frozen pools exist, regenerates the files with exact values. The documents in `docs/` quote the exact values.
* `report/REPORT.md`, `tables.tex` and `report_data_confirm.json` were regenerated on 2026-10-08 with the current code from the
  committed answers: no number changed (the rate precision aside); a stale reference and a formatting slip in one caption were removed.
