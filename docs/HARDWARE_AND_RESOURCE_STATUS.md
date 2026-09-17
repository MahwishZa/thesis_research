# Hardware and Resource Status

**Updated:** 2026-09-17 · Local environment now **VERIFIED**.

Labels: **VERIFIED** (measured on the student's laptop and reported here),
**MEASURED** (observed in the audit container), **ESTIMATED** (arithmetic from
published model sizes), **UNKNOWN**.

---

## 1. Verified local environment

| Property | Value | Label |
|---|---|---|
| OS | 64-bit Windows, x64 | VERIFIED |
| CPU | AMD Ryzen 5 7535HS @ 3.30 GHz | VERIFIED |
| RAM | 16 GB total, 15.2 GB usable | VERIFIED |
| GPU | NVIDIA GeForce RTX 2050 | VERIFIED |
| VRAM | **4096 MiB** | VERIFIED |
| NVIDIA driver | 592.82 | VERIFIED |
| Driver-reported CUDA | **13.1** | VERIFIED |
| Python | 3.12.6 | VERIFIED |
| PyTorch | 2.4.1+cu121 | VERIFIED |
| PyTorch build CUDA | **12.1** | VERIFIED |
| `torch.cuda.is_available()` | **True** | VERIFIED |
| PyTorch-detected GPU | NVIDIA GeForce RTX 2050 (count 1) | VERIFIED |
| transformers / accelerate / faiss / huggingface_hub | installed | VERIFIED |
| Free disk (C:) | **30.07 GB** (445.87 GB used) | VERIFIED |

The corpus download is in progress and was not touched.

### 1.1 The CUDA version mismatch is not a problem

Two different numbers, two different meanings:

* **Driver-reported CUDA 13.1** — the maximum CUDA runtime this driver can
  support. It is a ceiling, not the version in use.
* **PyTorch build CUDA 12.1** — the runtime PyTorch was compiled against, and
  the one actually used.
* **Functional status: WORKING** — `torch.cuda.is_available()` returns `True`
  and PyTorch names the RTX 2050.

NVIDIA drivers are backward compatible with older CUDA runtimes, so a 13.1-capable
driver runs a cu121 build normally. **Do not reinstall CUDA or PyTorch because
the numbers differ.** The only evidence that matters is that CUDA initialises
and the GPU is detected, and it does. Reinstalling risks breaking a working
stack and consuming scarce disk for no gain.

### 1.2 Where this audit ran

The audit itself ran in a **Linux container with no GPU and no ML stack**
(**MEASURED**). It is not the laptop. All runtime figures below remain
**ESTIMATED**: no model was executed on either machine.

---

## 2. Feasibility by component

| Component | Verdict | Basis |
|---|---|---|
| **MedCPT query encoder** inference | **FEASIBLE LOCALLY** | ≈109 M params, ≈0.44 GB fp32 (ESTIMATED); fits 4 GB with room, CPU-viable |
| **MedCPT reranker** inference | **FEASIBLE LOCALLY** | same class; cost scales with candidates per question, not corpus size |
| **Flan-T5 filter** inference | **FEASIBLE LOCALLY** | ≈1.6 GB fp16 for the large variant (ESTIMATED); fits 4 GB |
| **Flan-T5 filter** *training* | **REQUIRES REMOTE/CLOUD HARDWARE** | ≈12.4 GB for weights + gradients + AdamW moments before activations (ESTIMATED) — see `RAG2_CLASSIFIER_FEASIBILITY.md` |
| **FAISS retrieval** | **RESOURCE-CONSTRAINED** | faiss installed (VERIFIED). Index RAM and disk scale with the final corpus, whose size is **UNKNOWN**. CPU index is the right choice; 16 GB RAM is the limit to watch |
| **Alzheimer's corpus processing** | **FEASIBLE LOCALLY, SLOW** | CPU-bound text processing; already running |
| **Llama-3-8B-Instruct** inference | **NOT PRACTICAL LOCALLY — see §3** | |
| **100-question orchestration** | **FEASIBLE LOCALLY** | the runner, freezing, hashing, annotation and statistics are pure Python; 229 tests pass with no ML stack |

---

## 3. Llama-3-8B-Instruct

What can be said from arithmetic alone (**ESTIMATED**):

* **BF16/FP16 GPU-only does not fit.** ≈16 GB of weights against 4 GB of VRAM.
* **Ordinary 4-bit GPU-only is also likely beyond 4 GB.** Weights alone are
  ≈4.5–5 GB before any KV cache, activations or runtime overhead — and RAG
  prompts are long, so the KV cache is not negligible.
* **CPU or CPU/GPU-offload execution may technically be possible** — llama.cpp
  with partial offload, or CPU-only on the Ryzen 5. Nothing here rules that out.
* **Actual runtime has not been measured.** No tokens/second figure is given.
* **Remote or cloud execution may therefore be necessary** for the final
  experiment.

**The open question is not "can it run" but "can it run reproducibly, within a
sensible time budget, twice" — once per arm, with re-runs if something fails.**
That is what decides the venue.

**Do not replace Llama-3-8B because local execution is awkward.** Decide the
venue first. If a remote GPU is used for generation, the scientific comparison
is unchanged: same questions, same frozen candidate sets, same prompts. Only
the machine differs, and that is a resource limitation to document, not a
methodological one.

---

## 4. Storage

**30.07 GB free, with the corpus still growing.** Budget before downloading
anything (all **ESTIMATED**):

| Item | Size | Needed now? |
|---|---|---|
| Llama-3-8B BF16 | ≈16 GB | **No** — cannot run on 4 GB |
| Llama-3-8B Q4_K_M GGUF | ≈4.9 GB | **No** — decide venue first |
| Flan-T5-large | ≈3 GB | **No** — not until filter inference is wired up |
| Flan-T5 filter checkpoint (yours, after training) | ≈3 GB | Later |
| MedCPT encoder + reranker | ≈0.9 GB | Yes, when retrieval is built |
| FAISS index | **UNKNOWN** — scales with corpus | Later |
| HF cache overhead + temp files | ≈1.5× model size during download | Account for it |
| Generated outputs (600 answers, JSONL) | a few MB | Negligible |
| Corpus growth | **UNKNOWN** | Reserve headroom |

A naive "download everything" would need roughly 25 GB before the index and
before the corpus finishes — most of the free space. **Nothing was downloaded
during this work.**

---

## 5. Recommended execution strategy

**Locally:** corpus build · question sourcing and review · evidence freezing and
hashing · annotation packets · statistics · tests · MedCPT and Flan-T5
*inference*. None needs more than 4 GB VRAM.

**Remotely (free-tier 16 GB GPU session):** filter *training*, once; generation
for both arms, if the venue decision goes that way.

The runner already takes a frozen manifest in and writes JSONL out, so the only
things crossing machines are two small files. Execution venue is a deployment
detail, and the experiment is already built that way.
