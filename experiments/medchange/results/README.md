# MedChange results (tracked)

Outputs of real runs that contain **no source text** and are expensive to regenerate: generated answers
(`answers_<split>.jsonl`, with admitted PMIDs, and its generator record `answers_<split>.config.json`), zero-shot
helpfulness scores (`helpfulness_<split>.jsonl`),
and saved analyses (`analysis_<split>.json`, written by `analyze --out`). Copy them here after a run and
commit them; the frozen pools and abstracts stay in the gitignored `../data/`. See
`docs/reproducibility.md` §5.

Committed: `answers_dev.jsonl`, `answers_dev.config.json` and `analysis_dev.json` (all six arms, dev split; `docs/log.md` Phase 25).
