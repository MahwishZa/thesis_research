# Reproducibility

How to install, test and run this project, what each step costs (measured on the target laptop:
Windows, 15.2 GB RAM, RTX 2050 with 4 GB VRAM, CPU-only generation), and where results live. Commands
are run from the repository root; Windows PowerShell paths use `\`, but `python -m ...` commands are
identical everywhere.

## 1. Install

```bash
pip install -e .                    # the active test suite and the pipelines' pure logic: numpy only
pip install -e ".[models]"          # torch, transformers, sentencepiece (MedCPT, Flan-T5)
pip install -e ".[medchange]"       # the models extra plus llama-cpp-python (generation)
pip install -e ".[report]"          # matplotlib, for the figures of `report` only
pip install -e ".[archive]"         # PyYAML, requests, pypdf: only to run the archived work (§4, §7)
```

Python ≥ 3.10; `pyproject.toml` is the single source of dependency truth. Environment notes that cost
time once:

* **llama-cpp-python on Windows** has no source build that works without a toolchain; install the
  prebuilt CPU wheel: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`.
* **TensorFlow with Keras 3 installed** makes `transformers` fail on import. Training and inference here
  use PyTorch only: set `USE_TF=0` (PowerShell: `$env:USE_TF = "0"`) instead of installing `tf-keras`.
* **numpy 2.x breaks torch 2.4.x** and TensorFlow 2.17; keep `numpy<2` (and a scipy built for it, e.g.
  `scipy==1.13.1`) in that environment.
* **Generator model.** Download `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` (≈ 4.6 GB) from
  `bartowski/Meta-Llama-3-8B-Instruct-GGUF` into `models/` (gitignored). `generate_answers` hashes the
  file itself and records the SHA-256 beside the answers (§8).
* **Keep a long run alive on a laptop:** plug in and disable sleep
  (`powercfg /change standby-timeout-ac 0`).

## 2. Tests

```bash
python -m unittest discover -s evaluation -t .      # active suite: no network, no models, no real data
python -m unittest discover -s _archive -t .         # archived work (Alzheimer's framework, RAG² filter attempt, v1 pilot, ...)
```

The active suite needs neither torch nor transformers nor network; model-dependent code is exercised
through interfaces and fixtures, and the realigned pipeline is run end to end on fake models in
`test_medchange_rag2.py`. Checked on 2026-10-05, after the reorganisation: in a fresh virtualenv holding only the
base dependency (numpy), with outbound socket connections blocked, the active suite passes (280 tests; the one
figure test is skipped because matplotlib is the optional `report` extra); in a fresh virtualenv with the `archive`
extra the archived suite passes (592 tests, 465 of them the Alzheimer's framework's); and the package installed
into the numpy-only environment makes all 38 active modules importable from outside the repository
(`log.md`, Phase 39).

The archived suite includes the tests of the corpus stage scripts, which work in a lean scaffold copy
(`_archive/alzheimers_framework/tests/corpus_scaffold.py`) that never copies the corpus's data folder, so they
behave identically on a fresh clone and on a machine holding the multi-GB real corpus; tests write only to
temporary directories. `python -m evaluation.tests.check_hermetic` verifies that: it runs both suites and fails if
any file in the repository, ignored files included, was created, modified or deleted. (It once caught tests
overwriting stage 06's resume marker `.chunk_progress.json` in the corpus's data folder on 2026-10-02; if that
file exists with `"documents_done": 10`, it is such an artefact and can be deleted.) The optional static
check used in the audit is `python -m pyflakes src evaluation experiments` (`pip install pyflakes`; not a project
dependency; it flags one intentional availability import in `encoders.py`).

## 3. Primary pipeline: MedChange as-of benchmark

All outputs of steps below go to `experiments/medchange/data/` (gitignored), except where a step is given
`--out-dir` or writes a report into `results/`. Every long step is resumable: rerun the same command after
an interruption.

| # | Command | Cost (measured unless noted) |
|---|---|---|
| 1 | `git clone https://github.com/jvladika/MedChange ..\MedChange` | one-off |
| 2 | `python -m experiments.medchange.build_benchmark --medchange-dir ..\MedChange` | seconds; refuses unless all 512 items reproduce; afterwards `git status` must show no change to `experiments/medchange/manifest.json` |
| 3 | `python -m experiments.medchange.headroom --medchange-dir ..\MedChange` (optional) | seconds; released models' answers without retrieval |
| 4 | `python -m experiments.medchange.pubmed_asof --split dev` | needs network; ≈ 22 min for the 226 dev items; prints the pre-stated G0 verdict |
| 5 | `python -m experiments.medchange.freeze_candidates --split dev --device cpu` | downloads abstracts (37,375 for dev), then MedCPT dense + cross-encoder; ≈ 50 s per item for encoding and reranking, ≈ 3 h for dev |
| 6 | `python -m experiments.medchange.helpfulness --split dev --device cpu` | zero-shot Flan-T5 on 20 pairs per item; 5,946 s for 223 items (1.33 s/pair) |
| 7 | `python -m experiments.medchange.generate_answers --split dev --arms B0 B1 --model-path models\<file>.gguf` | llama.cpp CPU: ≈ 18 s per B0 answer, ≈ 66 s with five passages; add `--limit 3` for a timing test |
| 8 | `python -m experiments.medchange.analyze --split dev` | per-arm accuracy, retrieval-level metrics, paired tests with Holm, gates G2/G3 |
| 9 | *(retired, archived in `_archive/medchange_human_checks/`)* the human check of stated verdicts | replaced by `consistency_auto` (step 14); the researcher is not asked to label anything |
| 10 | `python -m experiments.medchange.error_analysis --split dev --out-dir experiments\medchange\results` | seconds; uses the existing answers and frozen pools, no generation |
| 10b | `python -m experiments.medchange.dev_audit --out-dir experiments\medchange\results` | about 2 s; needs only `benchmark.jsonl` and the committed `results\answers_dev.jsonl` (dev only; refuses `--split confirm`); recomputes the figures quoted in `experimentation.md` §2 |

Dev is run first. For arms B2, P and C1 run step 6 before step 7. The full dev run of all six arms is
≈ 22 h (*estimated* from the measured per-answer times). Steps 4 and 5 call NCBI E-utilities and accept an
optional `--api-key` (3 requests per second without a key, 10 with one), which changes their duration, not
their results.

**Stage 2 (evidence-synthesis layer; fully automated, no human labelling).** Two one-time preparations: download a
second-family GGUF for the audits (Qwen2.5-7B-Instruct Q4_K_M, about 4.7 GB, into `models\`) with
`python -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/Qwen2.5-7B-Instruct-GGUF', 'Qwen2.5-7B-Instruct-Q4_K_M.gguf', local_dir='models')"` and have the MedChange
clone from step 1. Then two commands run everything; each is resumable (rerun the same command after an
interruption), stops at the first failed step or gate, and with `--commit` commits and pushes the results to `main`.

| # | Command | What it does and costs (*estimated* unless marked measured) |
|---|---|---|
| 11 | `python -m experiments.medchange.diagnostics --split dev --out-dir experiments\medchange\results` | P0, **done**: seconds, no model |
| 12 | `python -m experiments.medchange.stance --split dev --pilot --n-threads 6 --model-path models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` | pilot, **done**: 960 papers, 6.76 s per paper (measured, ≈ 1.8 h) |
| 13 | `python -m experiments.medchange.stance_check report` | gate 1, machine checks only: **PASS** on the pilot |
| 14 | `python -m experiments.medchange.pipeline dev --model-path models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --judge-path models\Qwen2.5-7B-Instruct-Q4_K_M.gguf --medchange-dir ..\MedChange --commit` | checks gate 1 and integrity; dev label audit (≈ 1 h); automatic consistency check (≈ 0.6 h); stance on all dev papers with both wordings (3,616 papers, ≈ 6.8 h); `synthesis fit` with repeated cross-validation, gate 2 and the frozen model; writes `results\DEV_REPORT.md`; commits and pushes the frozen model. **Stops here** |
| 15 | read `results\DEV_REPORT.md`; if you agree: `python -m experiments.medchange.pipeline confirm --go --model-path models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --judge-path models\Qwen2.5-7B-Instruct-Q4_K_M.gguf --medchange-dir ..\MedChange --commit` | refuses unless the frozen model is on `origin/main` and the generator is the dev one; confirmatory probe and pools (≈ 8 h), B0 and B1 answers (≈ 11 h), stance with both wordings and the frozen-model prediction (≈ 16 h; only if gate 2 passed, otherwise RQ1 only), the after-freeze audits (≈ 2.6 h), the analysis and `results\FINDINGS.md` |
| 16 | `python -m experiments.medchange.pipeline status` | which gates passed and which files exist |
| 17 | `python -m experiments.medchange.report --medchange-dir ..\MedChange` (add `--split confirm` after step 15) | seconds; no experiment: result tables in the base paper's layout, a LaTeX table and four figures in `results\report\` (figures need `pip install -e ".[report]"`; the tables do not); `pipeline` runs it at the end of each phase |

Add `--dry-run` to either phase to print its steps without running anything. The individual modules
(`stance`, `synthesis fit`, `synthesis predict`, `label_audit`, `consistency_auto`, `analyze_stage2`) can also be
run by hand with the flags shown in their docstrings. The confirmatory split is run once, in this order, with no
setting changed after the frozen model is committed. In total about 46 h of unattended laptop time (≈ 30 h if gate 2
fails); the six-arm confirmatory run of stage 1 (≈ 51 h) is no longer planned. `analyze_stage2 --split dev` gives an
exploratory dev version from the out-of-fold predictions.

**Realigned study (adapted RAG² + evidence-criteria verification; no network, no human labelling).** It reuses
the benchmark, the as-of PubMed records and abstracts from steps 2–5 (`data\pubmed_g0\`, `data\abstracts.jsonl`)
and the B0/B1 answers. Protocol: `docs/experimentation.md`.

| # | Command | What it does and costs (*estimated* from measured per-step times) |
|---|---|---|
| 18 | `python -m experiments.medchange.rag2_pipeline dev --model-path models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit` | integrity checks; rationales (≈ 20–25 s per question); candidate lists with MedCPT (≈ 45–90 s); filter (≈ 55–60 s); answers R2, R2C, R2V, R2V-ND (≈ 4–4.5 min); analysis; dev report with the pre-declared dev check; design record; environment record; commit and push. ≈ 26 h for the 226 dev questions (a three-question test gave ≈ 6.8 min per question). `--ablations` adds R2-RQ, R2-BR and R2-NF (≈ +14 h); `--judge-path models\Qwen2.5-7B-Instruct-Q4_K_M.gguf` adds the directness judge (≈ +3 h) |
| 19 | read `results\RAG2_DEV_REPORT.md`; if the dev check says READY: `python -m experiments.medchange.rag2_pipeline confirm --go --model-path models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit` | refuses to start unless the design record on origin/main equals the current design; then the same steps on the 528 held-out questions for R2, R2C, R2V and R2V-ND, analysis, `RAG2_FINDINGS.md`; ≈ 60 h (≈ 49 h with `--no-temporal-ablation`, decided before the run) |
| 20 | `python -m experiments.medchange.rag2_pipeline status` | which phase is done and whether the design record is pushed |
| 21 | `python -m experiments.medchange.rag2_run answers --split dev --limit 3 --model-path models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` | a timing test of one step on three questions (each step: `rationale`, `lists`, `filter`, `answers`, `judge`) |
| 22 | `python -m experiments.medchange.analyze_rag2 --split dev` | the analysis alone, printed |
| 23 | `python -m experiments.medchange.ad_benchmark --medchange-dir ..\MedChange` | seconds; appends the 208 Alzheimer's/dementia questions to `benchmark.jsonl` as split `ad` (rerunnable); `experiments/medchange/manifest_ad.json` must show no change in `git status` (it was reproduced from the released files on 2026-10-05) |
| 24 | after step 19: `python -m experiments.medchange.rag2_pipeline ad --go --model-path models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit` | same guards as step 19; as-of PubMed records (network, ≈ 0.4 h), frozen pools with abstracts (≈ 2.9 h), B0/B1 (≈ 4.4 h), R2, R2C, R2V (≈ 20.5 h), analysis, `RAG2_FINDINGS_AD.md`; ≈ 28 h in all. Many of its reviews are old (48 of 208 before 2005), so some pools may be small or empty: nothing aborts, but for a question with no admitted evidence R2 answers without evidence and R2C and R2V keep R2's answer, so they cannot differ there; the analysis reports the share of questions without evidence per arm, and `freeze_candidates` prints the number of empty pools |

## 4. Archived: the Alzheimer's corpus, question pool and index

The first design (a local Alzheimer's corpus of 4.4 million chunks with a dense index, a 113-question reviewed
pool and a three-arm runner) is archived in `_archive/alzheimers_framework/` and is **not used by the current
study**; the Alzheimer's evaluation is now the `ad` test set (step 23). Its README lists what it holds, where each
part used to be and how to run it, for example the fixture demo
`python -m _archive.alzheimers_framework.experiments.shared.runners.run_end_to_end` (synthetic data, no corpus,
no model; it is not a thesis result) and the dense-index build
`python -m _archive.alzheimers_framework.experiments.shared.retrieval.build_index --device cpu`. The corpus
scripts and the archived tests that exercise them need `pip install -e ".[archive]"`.

The corpus text and the index were built locally and are not tracked; they were never moved. A computer that built
them before 2026-10-05 still holds them in the old `corpus\data\` and `experiments\results\index\` folders, which
the root `.gitignore` keeps out of Git; the archive's README says how to reuse, move or delete them.

## 5. What is gitignored, and results policy

| Path | Why |
|---|---|
| `experiments/medchange/data/` | MedChange-derived questions, abstracts, frozen pools, raw working files |
| `models/`, `checkpoints/` | model files |
| `_archive/alzheimers_framework/corpus/data/` (except the fixture), `_archive/alzheimers_framework/experiments/results/index/` | the archived corpus text and retrieval index (~12.5 GB): large, built locally. The same folders at their pre-2026-10-05 location (`corpus\` and `experiments\results\index\`) are ignored too, so `git add -A` is safe on a computer that still holds them |
| `_archive/rag2_filter_reproduction/filter_training/labels/`, `.textbook_index_cache/` | labels embed textbook passages |

**Results worth keeping are committed.** After a run, copy the files that contain no source text — the
answers (`answers_<split>.jsonl`: generated text and PMIDs) with their generator record
(`answers_<split>.config.json`), the helpfulness scores (`helpfulness_<split>.jsonl`) and the saved analysis (`python -m experiments.medchange.analyze --split
dev --out experiments\medchange\results\analysis_dev.json`) — into `experiments/medchange/results/` and
commit them; a 22-hour run must not exist only on one laptop. The frozen pools and abstracts stay
local and are rebuilt from the manifest. Stage 2 adds the same kind of files (the pipeline copies them): `stance_pilot.jsonl`,
`stance_dev.jsonl` and `stance_confirm.jsonl` with their `.config.json` records, the label-audit and consistency files, `synthesis_dev.jsonl` and `synthesis_confirm.jsonl` (the arms'
verdicts and probabilities), and the reports `diagnostics_dev.*`, `synthesis_cv_dev.md`, `stage2_analysis_*.json/.md`
(written straight into `results/` by `--out-dir`). `synthesis_model.json` is the frozen model and is
committed **before** any confirmatory stance run, so the history shows that it preceded the confirmatory
data. The dev error analysis
(`error_analysis_dev.md`, `.json`) is committed the same way when it is saved with `--out-dir`.

## 6. Hardware notes

MedCPT and Flan-T5 inference are comfortable on CPU (Flan-T5-large: 1.33 s per pair measured). The 8B
generator cannot run on the 4 GB GPU; llama.cpp on CPU needs ≈ 6–7 GB RAM. The stage-2 stance step loads the
same model with `logits_all` (needed for the first-token log-probabilities) and a 1,536-token context, which
should add about 0.5 GB (*estimated*, not yet measured); close other programs before it, and use
`--hard-labels` if memory is short. Full fine-tuning of Flan-T5-large
(archived attempt) was ≈ 106 s per 16-example optimiser step on CPU. The streaming index build keeps memory
bounded at the full corpus scale.

## 7. Archived work

`_archive/` holds directions that were tried and replaced: the Alzheimer's-specific framework
(`alzheimers_framework/`: corpus, question pool, three-arm runner, annotation workflow, with its tests), the RAG²
filter-reproduction attempt (`rag2_filter_reproduction/`: label generation with a 4-bit Llama-3, filter
training, diagnostics, with its tests), the superseded Alzheimer's pilot runners and results
(`alzheimers_pilot_v1/`), and earlier exploratory work; see `_archive/README.md`. Local files from the filter-reproduction attempt that git does
not track (`labels/`, `.textbook_index_cache/`) belong in
`_archive/rag2_filter_reproduction/filter_training/`; the `labels` file is 16.6 hours of compute, so back it
up outside the repository.

## 8. Reproducing a specific run exactly

Every answer record carries the arm-settings hash, the prompt hash, the admitted PMIDs and the wall-clock
seconds; every frozen pool carries an order-sensitive hash and the encoder names. The generator is recorded
once per answers file in `answers_<split>.config.json`: the model file's SHA-256, context size, token limit,
temperature, seed, the arm-settings hash and hashes of the system prompt and template (and, for information,
thread count, GPU layers and the llama-cpp-python version). `generate_answers` refuses to extend an answers
file whose recorded model or result-relevant settings differ from the current ones, so one file never mixes
generator configurations; an answers file made before this record existed (the six timing answers) is
adopted with a note, its original configuration being unknown. `analyze` copies the record into its saved
report. A run also needs the manifest's input hashes (the MedChange files) and the `numpy`/`torch` versions.
Finished (item, arm) pairs are skipped on a resume, never overwritten.

Stage 2 follows the same rule. Every stance record carries the PMID, the pool rank, the wording, the
backend, the three probabilities (or none, when the output was invalid), the probability mass found on the
three letters, the seconds and a hash of the snippet that was shown. `stance_<split>.config.json` records
the model file's SHA-256, the hashes of the system prompt and of both wordings, the context size,
temperature 0, the seed, the top-k and the snippet limit; `stance` refuses to extend a file written under a
different configuration. `synthesis_model.json` records the stance wording and backend, the fixed settings
(penalty 5.0, half-life, study-type weights, feature list, variants), the cross-validation results, the
selected hybrid, every coefficient and a hash of the dev item list; `synthesis predict` refuses to run
without it and writes its SHA-256 beside the confirmatory predictions (`synthesis_confirm.config.json`), and
refuses to overwrite predictions made with a different model file. The fitting is deterministic (seeded
folds, a Newton solver run to convergence), so the same dev stance file gives the same model file; the
generator is not bit-for-bit reproducible (identical prompts gave different wording in 9 of 17 pairs, with
the same verdict), so a rerun of the stance step itself may differ slightly in a probability (not measured).
