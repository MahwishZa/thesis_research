# Reproducibility

How to install, test and run the project, what each step costs (measured on the target laptop: Windows, 15.2 GB RAM, an
RTX 2050 with 4 GB of video memory that cannot hold the generator, so generation runs on the CPU), what is recorded so that
a run can be checked, and where results live. Commands are run from the repository root. Paths use the Windows form
(`models\...`); the `python -m ...` commands are identical on every platform. Numbers are *measured* (this project's runs),
*computed* (from files in this project) or *estimated* (derived from a measurement).

## 1. Install

```bash
pip install -e .                    # the test suite and the pipelines' pure logic: numpy only
pip install -e ".[medchange]"       # adds torch and transformers (MedCPT) and llama-cpp-python (generation)
pip install -e ".[report]"          # adds matplotlib, for the figures of `report` only
```

Python ≥ 3.10. `pyproject.toml` is the single source of dependency truth, and a test checks that its package list equals the
active packages. No dependency version is pinned; `rag2_environment_<phase>.json` records the versions actually used (§6).
Environment notes that cost time once:

* **llama-cpp-python on Windows** has no source build that works without a toolchain; install the prebuilt CPU wheel:
  `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`.
* **TensorFlow with Keras 3 installed** makes `transformers` fail on import. The study uses PyTorch only: set `USE_TF=0`
  (PowerShell: `$env:USE_TF = "0"`) instead of installing `tf-keras`.
* **numpy 2.x breaks torch 2.4.x** and TensorFlow 2.17; keep `numpy<2` (and a scipy built for it, for example `scipy==1.13.1`)
  in that environment.
* **Generator.** Download `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` (about 4.6 GB) from `bartowski/Meta-Llama-3-8B-Instruct-GGUF`
  into `models\` (gitignored). The pipelines hash the file and record the SHA-256 (§6).
* **Independent judge** (label audit, consistency check, directness): `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (about 4.7 GB) into
  `models\`: `python -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/Qwen2.5-7B-Instruct-GGUF', 'Qwen2.5-7B-Instruct-Q4_K_M.gguf', local_dir='models')"`.
* **Keep a long run alive on a laptop:** plug in and disable sleep (`powercfg /change standby-timeout-ac 0`).

## 2. Tests and checks

```bash
python -m unittest discover -s evaluation -t .      # no network, no models, no real data
python -m evaluation.tests.check_hermetic            # the suite must leave the repository unchanged
python -m pyflakes src evaluation experiments        # optional static check (pip install pyflakes)
```

The suite needs neither torch nor transformers nor network: model-dependent code is exercised through interfaces and fake
models, and the realigned pipeline is run end to end on fakes in `test_medchange_rag2.py`. It passes in a fresh virtual
environment that holds only numpy, with outbound socket connections blocked (one figure test is skipped there because
matplotlib is an optional extra), and in a fresh clone of the repository (the dated checks are in `log.md`).
`check_hermetic` fails if any file in the repository, ignored files included, was created, modified or deleted by the suite.
`pyflakes` reports one intentional availability import in `encoders.py`. Real-model code paths cannot be tested without the
models; they were exercised only by the researcher's smoke test and runs.

## 3. Rebuilding the data (once)

All working files go to `experiments/medchange/data/` (gitignored). Every long step is resumable: rerun the same command after
an interruption. The development and held-out as-of records, frozen pools and B0/B1 answers already existed when the realigned
study was run (committed answers: `results/answers_<split>.jsonl`); a reader who wants to rebuild them runs steps 1 to 7.

| # | Command | Cost |
|---|---|---|
| 1 | `git clone https://github.com/jvladika/MedChange ..\MedChange` | one-off |
| 2 | `python -m experiments.medchange.build_benchmark --medchange-dir ..\MedChange` | seconds; refuses unless all 512 items reproduce; afterwards `git status` must show no change to `manifest.json` |
| 3 | `python -m experiments.medchange.ad_benchmark --medchange-dir ..\MedChange` | seconds; appends the 208 ADRD (Alzheimer's disease and related dementias) questions as split `ad` (rerunnable); `manifest_ad.json` must show no change in `git status` |
| 4 | `python -m experiments.medchange.pubmed_asof --split dev` (then `--split confirm`) | needs network; about 22 min for the 226 development questions (measured); an optional `--api-key` changes the duration, not the results |
| 5 | `python -m experiments.medchange.freeze_candidates --split dev --device cpu` (then `--split confirm`) | downloads the abstracts (37,375 for dev), then MedCPT encoding and re-ranking at about 50 s per question, about 3 h for dev (measured) |
| 6 | `python -m experiments.medchange.generate_answers --split dev --arms B0 B1 --model-path models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` (then `--split confirm`) | about 15 s per B0 answer and 61 s per B1 answer (measured, held-out); `--limit 3` gives a timing test |
| 7 | `python -m experiments.medchange.label_audit --split dev --medchange-dir ..\MedChange --model-path models\Qwen2.5-7B-Instruct-Q4_K_M.gguf` (and `consistency_auto`) | independent re-labelling of the gold labels (`evaluation.md` §1.4); about an hour for dev (*estimated*); `--split confirm` and `--split ad` are accepted only after `results/rag2_design.json` exists |

The `ad` phase (§4.1) fetches its own as-of records and builds its own pools, so steps 4 to 6 are needed only for the
development and held-out splits; step 3 must have been run before the `ad` phase.

## 4. Running the study

### 4.1 The three phases

One resumable command per phase (`rag2_pipeline`). Each stops at the first failed step, prints what it runs, and with
`--commit` commits the results to the local `main` branch. **It never pushes**: the researcher runs `git push origin main`.
`--dry-run` prints a phase's steps without running anything, and `status` shows which phase is done and whether the design
record is pushed.

| Phase | Command | What it does | Cost |
|---|---|---|---|
| dev | `python -m experiments.medchange.rag2_pipeline dev --model-path models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit` | checks; rationales; candidate lists; filter; answers R2, R2C, R2V, R2V-ND; analysis; dev report with the pre-declared dev check; design record; environment record; commit. `--ablations` adds R2-RQ, R2-BR and R2-NF; `--judge-path` adds the directness judge | **measured 20.4 h** for the 226 questions with the judge (5.4 min per question) |
| held-out | `python -m experiments.medchange.rag2_pipeline confirm --go --model-path ... --commit` | the same steps on the 528 held-out questions, then `RAG2_FINDINGS.md`; `--no-temporal-ablation` leaves R2V-ND out (decided before the run) | **measured 49.2 h** with the judge (5.6 min per question) |
| `ad` | `python -m experiments.medchange.rag2_pipeline ad --go --model-path models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --judge-path models\Qwen2.5-7B-Instruct-Q4_K_M.gguf --commit` | as-of PubMed records (network); candidate pools and abstracts; B0 and B1 answers; then rationales, lists, filter and answers R2, R2C, R2V; analysis; `RAG2_FINDINGS_AD.md` | **measured 22.7 h** with the judge: records 0.4 h, pools 3.7 h, B0 and B1 4.3 h, rationales 0.8 h, lists 2.8 h, filter 2.4 h, answers 6.1 h, judge 2.1 h |

Measured per question in the held-out run, from the recorded durations: rationale 16 s, filter 48 s (eight judgements), the four
answers 175 s (R2 54 s, R2C 52 s, R2V 14 s, R2V-ND 55 s) and the directness judge 42 s; the remainder of the 5.6 minutes is
candidate lists with MedCPT and model loading.

### 4.2 Guards

* The held-out and `ad` phases refuse to start without `--go`, without `results/RAG2_DEV_REPORT.md`, unless the design record
  `results/rag2_design.json` on `origin/main` equals the current design (every setting, every prompt hash, the generator file's
  hash), and unless the design record is pushed. This is why the researcher pushes the dev phase's commit before the next phase.
* Each step records its configuration beside its output (model SHA-256, context size, token limit, temperature, seed, settings
  and prompt hashes) and refuses to resume under a different one, so one file never mixes configurations.
* A preflight checks the benchmark counts (226, 528, 208), the as-of records, the abstract cache and the B0/B1 answers.
* The `ad` phase has no empty-pool abort: many of its reviews are old (48 of 208 before 2005), so some pools may be small or
  empty. For a question with no admitted evidence R2 answers without evidence and R2C and R2V keep R2's answer, so they cannot
  differ there; the analysis reports the share of questions without evidence per system, and `freeze_candidates` prints the
  number of empty pools.

### 4.3 Individual steps

`python -m experiments.medchange.rag2_run <step> --split <dev|confirm|ad> --model-path ...` runs one step: `rationale`,
`lists` (MedCPT, no model path), `filter`, `answers` (`--arms`) or `judge`; `--limit 3` is a timing test. The modules can be run
by hand with the flags shown in their docstrings.

## 5. Re-analysing without re-running models

```bash
python -m experiments.medchange.analyze_rag2 --split confirm --out-dir experiments\medchange\results --label-audit experiments\medchange\results\label_audit_confirm.jsonl
python -m experiments.medchange.report --split confirm --medchange-dir ..\MedChange
```

`analyze_rag2` computes the accuracy tables, the primary test, the pre-declared reading of the 1-point requirement, the secondary
family with Holm correction and the retrieval metrics from the answers and records; it runs a model nowhere and takes seconds. It
reads `benchmark.jsonl`, `answers_<split>.jsonl`, `rag2_answers_<split>.jsonl`, `rag2_directness_<split>.jsonl` and, when present,
`frozen_<split>.jsonl` from `experiments\medchange\data\` (without the frozen pools the evidence-type, age and update-window cells of
B1 stay empty); the committed copies in `results\` can be copied there.
`report --split ad` writes the ADRD set's tables to `results\report_ad\`. `report` writes the result tables, a LaTeX table and two figures to `results\report\` (the closed-book rows of existing models
need the MedChange clone; the figures need the `report` extra). The phase drivers do not run it: it is run by hand after a phase,
and the committed `results\report\` is the held-out version.

## 6. What is recorded, and what is and is not reproducible

* **Per answer:** the arm's settings hash, the prompt hash, the admitted PMIDs (for the R2 family also their date bounds and
  evidence types) and the wall-clock seconds. **Per output file:** `<file>.config.json` with the generator's SHA-256, context size, token limit,
  temperature 0, seed 42 and the hashes of the system prompt and template (for information also the thread count, GPU layers and
  the llama-cpp-python version). Finished (question, system) pairs are skipped on a resume, never overwritten.
* **Per frozen pool:** an order-sensitive hash and the encoder names. **Per benchmark:** the hashes of the input files, normalised
  to LF line endings.
* **Per phase:** `rag2_environment_<phase>.json` with Python and package versions, the platform, the code's git commit and
  whether tracked code was modified. It is informational; nothing compares it.
* **Deterministic:** the benchmark builders (seeded; the manifests are reproduced hash for hash), the as-of rule, the scoring, the
  statistics (bootstrap seed fixed) and the analyses and reports, which regenerate the committed tables from the committed
  records.
* **Not bit-for-bit reproducible:** the generated text. Greedy decoding with a fixed seed removes sampling, but llama.cpp output
  can differ across machines, thread counts and library versions; an earlier check found identical prompts giving different
  wording in 9 of 17 pairs with the same verdict. A rerun can therefore differ slightly in individual answers. PubMed changes
  over time (records and abstracts are added or revised), so rebuilt as-of records can differ slightly from the committed runs;
  the frozen pools and answers of the runs are the record.
* MedChange states no licence and abstracts are publisher text, so neither is redistributed: a reader rebuilds both from the
  manifests and the public sources.

## 7. Files: committed, ignored and local leftovers

| Path | Status |
|---|---|
| `experiments/medchange/data/` | gitignored: MedChange-derived questions, abstracts, frozen pools, working files |
| `models/`, `checkpoints/`, `build/`, `dist/` | gitignored |
| `experiments/medchange/results/` | committed: results without source text (file list: `data.md` §4) |
| `experiments/medchange/results/earlier_stages/` | committed: results of record of stages 1 and 2 |
| `corpus\`, `experiments\results\index\`, `_archive\` | gitignored local leftovers of the removed first design |

A computer that built the first design still holds its corpus text, retrieval index (about 12.5 GB) and filter-training labels
in `corpus\data\`, `experiments\results\index\` and `_archive\`. The root `.gitignore` keeps them out of Git, so `git add -A` is
safe. **A `git pull` does not delete them; do not delete them by hand unless you are sure**: the filter-training labels file is
16.6 hours of compute and the index took days. The layout checks in the tests look at files, not folders, for this reason.

## 8. Hardware notes

MedCPT inference is comfortable on a CPU. The 8B generator cannot run on the 4 GB GPU; llama.cpp on the CPU needs about 6 to 7 GB
of RAM. The filter and directness steps load a model with the log-probabilities of the first output token enabled (about 0.5 GB
more, *estimated*), so close other programs before a run. MedCPT encoding on the GPU (`--device cuda`) was never tried.

## 9. Earlier stages and removed work

The code of stages 1 and 2 (recency-aware admission and the evidence-synthesis layer) was removed from the active tree on
2026-10-08 after the current pipeline was verified not to need it; `git checkout f721bbb` restores that state, and
`git show f721bbb:docs/reproducibility.md` has the commands. Their outputs are in `results/earlier_stages/`. The first,
Alzheimer's-specific design (corpus, question pool, three-arm runner, filter retraining) was removed on 2026-10-06
(`git show 5e03540:_archive/README.md`; `git checkout 5e03540 -- _archive` restores the folder, whose tests need PyYAML,
requests and pypdf).

## 10. Archived: the Alzheimer's-specific question-set builder

The counting script, specification, builder and verifier check of the Alzheimer's-specific question set left the active tree after the construction was
stopped at gate 1 (`protocol.md` §8). They are in Git history: `git checkout 4ab9d1b -- experiments/adkqa evaluation/tests/unit` restores the code and its
tests (the committed counts-only results are in `experiments/medchange/results/earlier_stages/adkqa/`). Local build data in the git-ignored folder experiments/adkqa/data, if any,
may be deleted.

Exploratory subgroup of the dementia run (the questions naming Alzheimer's disease): `python -m experiments.medchange.subgroup_ad` uses
`benchmark.jsonl` (`--data-dir`) and the committed answers (`--results-dir`), runs no model and writes `results/ad_subgroup_alzheimer.*`.
Questions of MedRevQA that no split uses yet (counts only; nothing is selected): `python -m experiments.medchange.fresh_supply --medchange-dir ..\MedChange`
reads `benchmark.jsonl` (`--data-dir`) and the MedChange files and writes `results/fresh_supply.json` (it also counts questions about other neurodegenerative diseases, by question wording and by review text, with their verdicts).
New Cochrane dementia and Alzheimer's reviews after the MedRevQA snapshot (counts only; needs PubMed): `python -m experiments.medchange.cochrane_ad_supply --medchange-dir ..\MedChange`
writes `results/cochrane_ad_supply.json` (counts and PubMed identifiers, no titles) and prints the usable titles; 30 or more usable reviews = go, 20 to 29 = a small pilot, fewer = too few.

## 11. The pre-registered test on fresh questions (protocol §10)

Order of work, in PowerShell from the repository folder (the run itself is about 57 to 68 hours [estimate], resumable):
```
python -m experiments.medchange.fresh_benchmark --medchange-dir ..\MedChange       # selects the 850 questions, writes manifest_fresh.json
git add experiments\medchange\manifest_fresh.json ; git commit -m "Fresh question selection" ; git push origin main
python -m experiments.medchange.fresh_benchmark --freeze                            # marks protocol section 10 IN FORCE (needs the push above)
git add docs\protocol.md ; git commit -m "Pre-registration in force" ; git push origin main
python -m experiments.medchange.label_audit --split fresh --sample 300 --medchange-dir ..\MedChange --model-path models\Qwen2.5-7B-Instruct-Q4_K_M.gguf
python -m experiments.medchange.rag2_pipeline fresh --go --model-path models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --commit
python -m experiments.medchange.analyze_fresh                                       # again, to include the audit; prints only a count until all answers exist
```
The pipeline refuses to start unless the freeze is complete. `analyze_fresh` is blinded: it computes nothing until all 850 questions have answers for R2, R2C and R2V, and
it writes `rag2_analysis_fresh.json` and `RAG2_FINDINGS_FRESH.md` only then. Do not open partial answer files.

