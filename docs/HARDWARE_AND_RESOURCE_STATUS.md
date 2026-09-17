# Hardware and Resource Status

**Updated:** 2026-09-17

Labels used throughout: **REPORTED** (given by the student, not verified here),
**MEASURED** (observed by a command), **ESTIMATED** (arithmetic from published
model sizes), **UNKNOWN**.

---

## 1. A limitation of this audit, stated first

The audit runs in a **Linux cloud container with no GPU**. The thesis runs on a
**Windows laptop**. They are different machines.

| | Audit container (**MEASURED**) | Student laptop (**REPORTED**) |
|---|---|---|
| OS | Linux x86_64 | 64-bit Windows |
| CPU | Intel Xeon @ 2.10 GHz, 4 cores | AMD Ryzen 5 7535HS @ 3.30 GHz |
| RAM | 15 GiB | 16 GB (15.2 GB usable) |
| GPU | none | NVIDIA RTX 2050, **4 GB VRAM** |
| Free disk | 30 G | **~31 GB** |
| torch / transformers | **not installed** | UNKNOWN |
| HF token | **absent** | UNKNOWN |

**I could not verify the laptop's CUDA version, driver, PyTorch build, or
whether the GPU is visible to Python.** Nothing below claims otherwise. Run
§6 on the laptop before acting on any of it.

---

## 2. Actual hardware (as reported)

AMD Ryzen 5 7535HS (6 cores / 12 threads, Zen 3+) · 16 GB RAM · NVIDIA RTX 2050
**4 GB VRAM** · AMD Radeon iGPU · 477 GB disk with ~31 GB free · Windows x64.

The 4 GB VRAM figure is the binding constraint for every decision in this
document.

---

## 3. What can realistically run locally

| Component | Size | Verdict |
|---|---|---|
| All data preparation, question building, freezing, hashing | pure Python | **Yes** — already runs; 201 tests pass with no ML stack |
| Annotation packet generation, agreement, statistics | pure Python | **Yes** |
| MedCPT query encoder (~109 M params) | ≈ 0.44 GB fp32 | **Yes** — ESTIMATED, fits 4 GB or CPU |
| MedCPT cross-encoder reranker (~109 M) | ≈ 0.44 GB fp32 | **Yes** — ESTIMATED |
| Flan-T5-large **inference** (~780 M) | ≈ 1.6 GB fp16 | **Yes** — ESTIMATED, fits 4 GB |

The evidence filter can be **run** locally. That is worth knowing: only its
*training* is blocked.

## 4. What cannot realistically run locally

| Component | Requirement | Why it fails on 4 GB |
|---|---|---|
| **Llama-3-8B-Instruct, BF16** | ≈ 18–20 GB VRAM (ESTIMATED) | 5× the available VRAM |
| **Llama-3-8B-Instruct, 4-bit** | ≈ 4.5–5 GB weights **plus** KV cache (ESTIMATED) | Weights alone exceed 4 GB before any context |
| **Flan-T5-large full fine-tune** | ≈ 12 GB+ for weights, gradients and AdamW states, before activations (ESTIMATED) | 3× the available VRAM. See `RAG2_CLASSIFIER_FEASIBILITY.md` |

**Llama-3-8B does not fit this GPU in any configuration**, including 4-bit.
A partial CPU/GPU offload via llama.cpp would run, and CPU-only inference on a
6-core Zen 3+ chip would also run — but speed is **UNKNOWN** and RAG prompts
are long, so prompt processing, not token generation, would dominate. No
tokens/second figure is given because none was measured.

## 5. Storage risk

~31 GB free **while the corpus is still downloading**. The corpus's final size
is UNKNOWN. Treat disk as the scarcest resource.

Before any download, state its size, its VRAM need, whether it is needed *now*,
and whether a smaller option exists. Current standing answers:

| Download | Size (ESTIMATED) | Needed now? |
|---|---|---|
| Llama-3-8B Q4_K_M GGUF | ≈ 4.9 GB | **No** — see §4; decide execution venue first |
| Llama-3-8B BF16 weights | ≈ 16 GB | **No** — cannot run on 4 GB |
| Flan-T5-large | ≈ 3 GB | Not yet — only when filter inference is wired up |
| MedCPT encoder + reranker | ≈ 0.9 GB total | Yes, when retrieval is built |

## 6. Verification commands — run these on the laptop

```powershell
python --version
pip --version
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"
python -c "import transformers, accelerate; print(transformers.__version__, accelerate.__version__)"
huggingface-cli whoami
```

Record the output in this file. Until then §3–4 remain estimates.

## 7. Recommended execution strategy

**Split by cost, not by convenience. The science is identical either way; only
the machine changes, and that is a resource limitation to document, not a
methodological one.**

**Locally (laptop):** corpus build · question construction and review · evidence
freezing and hashing · annotation packets · all statistics · all tests. None of
this needs a GPU.

**Elsewhere (free-tier cloud GPU, e.g. a 16 GB T4 session):** filter training ·
the generation runs for both arms.

This works because the runner already takes a frozen manifest in and writes
JSONL out. The frozen manifest and the results file are the only things that
need to move between machines, and both are small. **Design the experiment so
execution venue is a deployment detail** — it already is.

**Do not** redesign the experiment because the GPU is small. Reduce the
generator if you must, and document that as a deviation.
