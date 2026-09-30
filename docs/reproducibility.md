# Reproducibility

How to install, test, and run this project, and exactly what is and is not
reduced-scale right now.

## 1. Install

```bash
pip install -e .
# Optional, only needed to run real models (not needed for the test suite):
pip install -e ".[models]"
# Optional, only needed for filter-training label generation - pick ONE:
pip install -e ".[filter-training-gguf]"  # local, CPU, no VRAM requirement
pip install -e ".[filter-training-hf]"    # needs a real GPU (~6GB VRAM) or ~16GB RAM
```

Python ≥ 3.10. `pyproject.toml` is the single source of dependency truth.

## 2. Run the tests

```bash
python -m unittest discover -s evaluation/tests -t .
```

No network access and no ML stack (`torch`/`transformers`) are required for
the test suite to pass — model-dependent code paths are exercised through
interfaces and fixtures, not real weights.

## 3. Run the pipeline

```bash
# Fixture demo — every arm, evaluation, and the ablation sweep, no real
# corpus or model needed:
python -m experiments.shared.runners.run_end_to_end

# Real data — fit lambda/theta/H on validation, report on held-out test
# (requires a built retrieval index; see step 4):
python -m experiments.shared.runners.fit_and_evaluate --device cpu
```

Both print two comparisons: `main_evaluation` (baseline vs. the full
proposed system) and `ablation_study` (full proposed system vs. `λ = 0`).

## 4. Rebuilding the retrieval index

```bash
python -m experiments.shared.retrieval.build_index --device cuda
```

Requires the local corpus (`corpus/data/chunks/chunks.jsonl` — gitignored,
built locally; see `data.md`). Recommended on GPU: the MedCPT encoder is
small (~0.44 GB) and fits modest hardware; the resulting embedding matrix
is large in memory at full corpus scale (see §6).

## 5. What is currently reduced-scale, and why it matters

The committed pilot-scale run under `experiments/results/fit_and_evaluate/` used a
pilot-scale index, an undertrained filter checkpoint, and an extractive
stand-in generator. Status as of 2026-09-27:

| Component | Current state | Full-scale requires |
|---|---|---|
| Retrieval index | **Complete** — full corpus, 4,376,141 × 768, built via the streaming/checkpointed build (§4) | — |
| RAG² baseline filter checkpoint | Not trained. Tooling ready (§4b). Prior pilot checkpoint (18 training examples, 1 epoch) is unusable — `validation_accuracy = 0.0`, and the saved archive did not even include a model weight file | Real-scale label generation (§4a) + local filter training (§4b) |
| Generator (main evaluation) | Extractive stand-in (verbatim top-admitted passage) | A real generative model under the contract in `methodology.md` §9. `fit_and_evaluate.py` currently hardcodes the stand-in — wiring in a real generator is unimplemented, not just unrun |

None of these are methodology changes in the design sense — but two are not
"just needs compute time" either: the filter checkpoint needs the
label-generation and training steps below actually built and run, and the
real-generator wiring in `fit_and_evaluate.py` needs writing before it can
be run at all.

## 4a. Generating filter-training labels (local, GGUF)

`experiments/baseline/filter_training/rationale.py`'s HF/`bitsandbytes` path
needs ~5.5–6 GB VRAM (4-bit) or ~16 GB RAM (CPU fallback) — infeasible on a
4 GB VRAM / 15.2 GB RAM laptop (verified 2026-09-27, see `log.md`). For that
hardware, `--scorer gguf` runs the same model (Llama-3-8B-Instruct) 4-bit
quantized via `llama.cpp` on CPU RAM instead (~6–7 GB) — a disclosed
execution-backend deviation, not a model or method change; see
`rationale_gguf.py`'s module docstring for exactly what is and isn't
identical to the HF path.

**Obtaining a GGUF file.** A widely-used, actively-maintained community
quantization exists at
[`bartowski/Meta-Llama-3-8B-Instruct-GGUF`](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF)
(`Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`, ~4.92 GB) — a 4-bit k-quant
conversion of the same model you already have license access to. Record the
exact file's sha256 alongside your run, the same discipline this project
already applies to pinned model revisions (§9).

**Always calibrate before a real run:**

```bash
python -m experiments.baseline.filter_training.build_labels \
    --scorer gguf --gguf-model-path /path/to/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    --n-questions 500 --output experiments/baseline/filter_training/labels/medqa_filter_labels.json \
    --model-revision <pinned-commit-sha> --calibrate 10
```

Scores 10 pairs, reports measured throughput and an extrapolated estimate
for the full `--n-questions`, and writes nothing — no output file, no
checkpoint. Repeat freely; only drop `--calibrate` once the estimate is
acceptable. The real run is checkpointed automatically
(`<output>.progress/`) and resumes with `--resume` if interrupted.

## 4b. Filter training (local)

`train.py`'s paper recipe (full AdamW fine-tuning of Flan-T5-large) needs
~12 GB on CPU, tight against 15.2 GB RAM, and does not fit the 4 GB GPU. The
implemented local recipe (each switch is recorded as a deviation in the
checkpoint record):

```
python -m experiments.baseline.filter_training.train \
    --labels experiments/baseline/filter_training/labels/medqa_filter_labels.json \
    --output-dir checkpoints/rag2_filter --epochs N \
    --cpu --precision fp32 --optimizer adafactor --gradient-checkpointing \
    --early-stopping-patience 3
```

* `--calibrate-steps N` runs N optimizer steps, prints s/step and memory, and
  writes no checkpoint - use it to choose `--epochs`.
* `--resume` continues an interrupted run; refused if any setting or the
  labels file (sha256) changed; an incomplete (mid-save) checkpoint is
  skipped automatically.
* Validation uses the *deployed* two-way [HELPFUL]/[NOT_HELPFUL] logit rule
  and reports accuracy, balanced accuracy, and the majority-class baseline.
  A checkpoint is "usable" only if it beats the majority baseline and has
  balanced accuracy > 0.5 (labels are ~71% NOT_HELPFUL, so "accuracy > 0.5"
  alone is satisfied by rejecting everything).
* Verified offline on a tiny T5 (real torch): calibrate, train, kill
  mid-run, resume. Not yet run on real Flan-T5-large - that is the
  student's next step.

Fixed bug (found before any real training): the old validation metric
compared `generate()` output column 0 - which is the decoder-start id - to the
label id, so it reported 0.0 accuracy even for a model that had learned the
task (reproduced: old metric 0.0 vs deployed-rule 0.79).

## 6. Hardware notes

MedCPT encoder/reranker inference and Flan-T5 filter inference are feasible
on a modest local GPU (a few GB VRAM). Flan-T5 filter *training* needs more
memory than that — see §4b. Llama-3-8B-Instruct label generation needs a
real GPU or ~16 GB RAM via the HF path (§4a) — see §4a for the local GGUF
alternative used on hardware without either. Building the full-corpus
retrieval index produces an embedding matrix on the order of several GB in
memory; the streaming build (§4) keeps this off the peak-memory path, but
watch for memory pressure on constrained hardware regardless.

## 7. What is gitignored, and why

| Path | Why |
|---|---|
| `corpus/data/` | The corpus text itself — large, built locally, regenerated by `corpus/scripts/`, never committed |
| `checkpoints/` | Trained model checkpoints — large, regenerated by `experiments/baseline/filter_training/` |
| Local pilot/sample corpus and index folders | Large, regenerable, machine-local |

Small, legitimate result files (`experiments/results/fit_and_evaluate/`,
`experiments/results/smoke_test_NOT_FINAL/`) are committed — they are not covered by any
blanket ignore rule, so a real committed result stays visible to `git
status` rather than silently disappearing alongside large regenerable
artifacts sitting next to it.

## 8. Reviewing the question pool

Instructions for a human reviewer working through
`experiments/shared/questions/review.csv` are archived at
`_archive/docs_legacy/question_review.md` (the review itself is complete;
see `data.md` §3). The worksheet's ACCEPT/REVISE/REJECT/HOLD taxonomy and
review criteria there are unchanged and still apply if the pool is ever
re-reviewed or extended.

## 9. Reproducing a specific run exactly

Every run records: the model id and a pinned commit sha (never a branch
name), quantisation settings, the prompt template, the context budget, the
fitted `λ`/`θ`/`H`, the baseline filter's identity and whether it is
trained, and a corpus-snapshot id. An output directory is never silently
overwritten by a repeated run — a collision is refused rather than merged.
