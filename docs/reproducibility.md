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
  `bartowski/Meta-Llama-3-8B-Instruct-GGUF` into `models/` (gitignored) and record
  `Get-FileHash models\<file>.gguf -Algorithm SHA256` with every run.
* **Keep a long run alive on a laptop:** plug in and disable sleep
  (`powercfg /change standby-timeout-ac 0`).

## 2. Tests

```bash
python -m unittest discover -s evaluation -t .      # active suite: no network, no models, no real corpus
python -m unittest discover -s _archive -t .         # archived work (RAG² filter attempt, v1 pilot runner)
```

The active suite needs neither torch nor transformers nor network; model-dependent code is exercised
through interfaces and fixtures. Checked on 2026-10-02: in a fresh virtualenv holding only the four
base dependencies, with outbound socket connections blocked, both suites pass (counts: `log.md`,
Phase 23).

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

All outputs of steps below go to `experiments/medchange/data/` (gitignored). Every long step is
resumable: rerun the same command after an interruption.

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

Dev is run first. The confirmatory split (`--split confirm`) is run only after the gates pass, once,
following the plan frozen in `experiment_plan.md`. For arms B2, P and C1 run step 6 before step 7. The
full dev run of all six arms is ≈ 22 h and the confirmatory run ≈ 51 h (*estimated* from the measured
per-answer times). Steps 4 and 5 call NCBI E-utilities and accept an optional `--api-key` (3 requests
per second without a key, 10 with one), which changes their duration, not their results.

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
answers (`answers_<split>.jsonl`: generated text and PMIDs), the helpfulness scores
(`helpfulness_<split>.jsonl`) and the saved analysis (`python -m experiments.medchange.analyze --split
dev --out experiments\medchange\results\analysis_dev.json`) — into `experiments/medchange/results/` and
commit them; a 22-hour run must not exist only on one laptop. The frozen pools and abstracts stay
local and are rebuilt from the manifest.

## 6. Hardware notes

MedCPT and Flan-T5 inference are comfortable on CPU (Flan-T5-large: 1.33 s per pair measured). The 8B
generator cannot run on the 4 GB GPU; llama.cpp on CPU needs ≈ 6–7 GB RAM. Full fine-tuning of Flan-T5-large
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
seconds; every frozen pool carries an order-sensitive hash and the encoder names. A run also needs the
manifest's input hashes (the MedChange files), the GGUF file's SHA-256, `numpy`/`torch`/`llama-cpp-python`
versions, and the seed. Output files are never silently overwritten by a different run: collisions are
refused or resumed, not merged.
