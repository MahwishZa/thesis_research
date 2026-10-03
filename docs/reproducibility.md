# Reproducibility

How to install, test and run this project, what each step costs (measured on the target laptop:
Windows, 15.2 GB RAM, RTX 2050 with 4 GB VRAM, CPU-only generation), and where results live. Commands
are run from the repository root; Windows PowerShell paths use `\`, but `python -m ...` commands are
identical everywhere.

## 1. Install

```bash
pip install -e .                    # tests, framework, corpus scripts: PyYAML, numpy, requests, pypdf
pip install -e ".[models]"          # torch, transformers, sentencepiece (MedCPT, Flan-T5)
pip install -e ".[medchange]"       # the models extra plus llama-cpp-python (generation)
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
python -m unittest discover -s evaluation -t .      # active suite: no network, no models, no real corpus
python -m unittest discover -s _archive -t .         # archived work (RAG² filter attempt, v1 pilot runner)
```

The active suite needs neither torch nor transformers nor network; model-dependent code is exercised
through interfaces and fixtures. Checked on 2026-10-03: in a fresh virtualenv holding only the four
base dependencies, with outbound socket connections blocked, both suites pass (649 active and 125 archived
tests; `log.md`, Phase 27).

Tests that run the corpus stage scripts work in a lean scaffold copy
(`evaluation/tests/corpus_scaffold.py`) that never copies `corpus/data/`, so they behave identically on a
fresh clone and on a machine holding the multi-GB real corpus, and tests write only to temporary
directories. `python -m evaluation.tests.check_hermetic` verifies that: it runs both suites and fails if
any file in the repository, ignored files included, was created, modified or deleted. (It caught tests
overwriting stage 06's resume marker `corpus/data/chunks/.chunk_progress.json` on 2026-10-02; if that
file exists with `"documents_done": 10`, it is such an artefact and can be deleted.) The optional static
check used in the audit is `python -m pyflakes src evaluation experiments corpus/scripts` (`pip install
pyflakes`; not a project dependency; it flags one intentional availability import in `encoders.py`).

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
| 9 | `python -m experiments.medchange.consistency export --n 50`, fill the CSV, `... score` | G1 human check of the stated verdicts |
| 10 | `python -m experiments.medchange.error_analysis --split dev --out-dir experiments\medchange\results` | seconds; uses the existing answers and frozen pools, no generation |
| 10b | `python -m experiments.medchange.dev_audit --out-dir experiments\medchange\results` | about 2 s; needs only `benchmark.jsonl` and the committed `results\answers_dev.jsonl` (dev only; refuses `--split confirm`); recomputes the figures quoted in `experiment_plan.md` §1 |

Dev is run first. For arms B2, P and C1 run step 6 before step 7. The full dev run of all six arms is
≈ 22 h (*estimated* from the measured per-answer times). Steps 4 and 5 call NCBI E-utilities and accept an
optional `--api-key` (3 requests per second without a key, 10 with one), which changes their duration, not
their results.

**Stage 2 (evidence-synthesis layer; code built and unit-tested, every step below still pending).** The order
matters: the gates are checked in order, each step stops the chain if it fails, and the frozen model file is
committed before any confirmatory stance output exists (`experiment_plan.md` §9). Costs marked *estimated*
assume 5–8 s per stance judgement, which the pilot replaces by a measurement.

| # | Command | Cost and what it decides |
|---|---|---|
| 11 | `python -m experiments.medchange.diagnostics --split dev --out-dir experiments\medchange\results` | P0, seconds, no model (add `--no-tokenizer` offline): is the helpfulness input truncated, how many abstracts have labelled RESULTS / CONCLUSIONS, how many systematic reviews are in the top 8 |
| 12 | `python -m experiments.medchange.stance --split dev --pilot --model-path models\<file>.gguf` | pilot: 40 items × 8 papers × 2 wordings + the irrelevant-paper control = 960 judgements, ≈ 1.3–2.1 h (*estimated*); if log-probabilities fail, repeat with `--hard-labels` |
| 13 | `python -m experiments.medchange.stance_check report` | seconds: the machine checks of gate 1 (invalid outputs, seconds per paper, wording agreement, control) |
| 14 | `python -m experiments.medchange.stance_check export`, fill `stance_handcheck.csv` (S, C or N for 40 papers), then `... score` | ≈ 30 min of reading by the researcher (*estimated*); picks the wording and finishes gate 1 |
| 15 | `python -m experiments.medchange.stance --split dev --model-path models\<file>.gguf` | gate 1 passed first; 226 × 8 = 1,808 judgements ≈ 2.5–4 h (*estimated*) |
| 16 | `python -m experiments.medchange.synthesis fit` | seconds to minutes: repeated cross-validation, gate 2, the selected hybrid; writes `results\synthesis_model.json` and `results\synthesis_cv_dev.md`. **Commit both before step 19** |
| 17 | `python -m experiments.medchange.pubmed_asof --split confirm`, then `freeze_candidates --split confirm --device cpu` | confirmatory preparation, needed for RQ1 whatever gate 2 says; ≈ 8 h (*estimated* from the dev per-item times of ≈ 6 s and ≈ 49 s); helpfulness (step 6) is **not** needed |
| 18 | `python -m experiments.medchange.generate_answers --split confirm --arms B0 B1 --model-path models\<file>.gguf` | ≈ 2.1 h + ≈ 8.9 h (*estimated* from 14.6 s and 60.9 s per answer) |
| 19 | `python -m experiments.medchange.stance --split confirm --model-path models\<file>.gguf` | only if gate 2 passed and step 16's files are committed; 528 × 8 = 4,224 judgements ≈ 6–9 h (*estimated*) |
| 20 | `python -m experiments.medchange.synthesis predict --split confirm`, then `python -m experiments.medchange.analyze_stage2 --split confirm --out-dir experiments\medchange\results` | seconds; `predict` refuses to run without the frozen model and records its hash; the analysis refuses an incomplete confirmatory run |

The confirmatory split is run once, in this order, with no setting changed after step 16. The confirmatory
run needs about 11 h of B0/B1 generation, 8 h of preparation and, if gate 2 passes, 6–9 h of stance (all
*estimated*); the six-arm confirmatory run of stage 1 (≈ 51 h) is no longer planned. `analyze_stage2` on
`--split dev` gives an exploratory dev version from the out-of-fold predictions of step 16 (needs the
dev B0, B1 answers and `synthesis_dev.jsonl`).

## 4. Secondary: Alzheimer's corpus, question pool and index

```bash
# corpus (queries and configs are versioned in corpus/config/; data is built locally and gitignored)
python corpus/scripts/01_pubmed_download.py --all        # PMID lists
python corpus/scripts/02_pmc_download.py                 # full text and dates (then --finalize)
python corpus/scripts/04_normalize.py                    # reads metadata/pmc.csv + XML
python corpus/scripts/05_deduplicate.py
python corpus/scripts/06_chunk.py                        # MedCPT tokenizer, 256 tokens / 32 overlap
python corpus/scripts/07_claim_classification.py
python corpus/scripts/04_normalize.py --input data/raw/pubmed/records.example.jsonl   # offline fixture instead

# dense index over corpus/data/chunks/chunks.jsonl (4,376,141 x 768, ~12.5 GB, memory-bounded, resumable)
python -m experiments.shared.retrieval.build_index --device cuda   # or cpu

# question pool (already built, reviewed and split; committed under experiments/shared/questions/)
python -m experiments.shared.questions.build_pool --medrevqa <DS_MedRevQA.csv> --medquad <MedQuAD dir> --retrieved-on 2026-09
python -m experiments.shared.questions.export_review
python -m experiments.shared.questions.split

# framework demo on synthetic fixtures (no corpus, no model): all arms, evaluation, ablation
python -m experiments.shared.runners.run_end_to_end
```

The corpus is complete and the index is built; the as-of Alzheimer's case-study run is **not
implemented** (`experiment_plan.md` §11). `run_end_to_end` demonstrates the original three-arm framework
(`src/`, `evaluation/runner.py`) on fixtures; it is not the thesis result.

## 5. What is gitignored, and results policy

| Path | Why |
|---|---|
| `corpus/data/` (except the fixture) | the corpus text: large, built locally |
| `experiments/results/index/` | the full retrieval index (~12.5 GB) |
| `experiments/medchange/data/` | MedChange-derived questions, abstracts, frozen pools, raw working files |
| `models/`, `checkpoints/` | model files |
| `_archive/rag2_filter_reproduction/filter_training/labels/`, `.textbook_index_cache/` | labels embed textbook passages |

**Results worth keeping are committed.** After a run, copy the files that contain no source text — the
answers (`answers_<split>.jsonl`: generated text and PMIDs) with their generator record
(`answers_<split>.config.json`), the helpfulness scores (`helpfulness_<split>.jsonl`) and the saved analysis (`python -m experiments.medchange.analyze --split
dev --out experiments\medchange\results\analysis_dev.json`) — into `experiments/medchange/results/` and
commit them; a 22-hour run must not exist only on one laptop. The frozen pools and abstracts stay
local and are rebuilt from the manifest. Stage 2 adds the same kind of files: `stance_pilot.jsonl`,
`stance_dev.jsonl` and `stance_confirm.jsonl` with their `.config.json` records, `stance_choice.json` (the
hand-check result and the chosen wording), `synthesis_dev.jsonl` and `synthesis_confirm.jsonl` (the arms'
verdicts and probabilities), and the reports `diagnostics_dev.*`, `synthesis_cv_dev.md`, `stage2_analysis_*.json/.md`
(written straight into `results/` by `--out-dir`). `synthesis_model.json` is the frozen model and is
committed **before** any confirmatory stance run, so the history shows that it preceded the confirmatory
data. `stance_handcheck.csv` shows paper text and stays local. The dev error analysis
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

`_archive/` holds directions that were tried and replaced: the RAG² filter-reproduction attempt
(`rag2_filter_reproduction/`: label generation with a 4-bit Llama-3, filter training, diagnostics, with its
tests), the superseded Alzheimer's pilot runners and results (`alzheimers_pilot_v1/`), and earlier
exploratory work; see `_archive/README.md`. Local files from the filter-reproduction attempt that git does
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
