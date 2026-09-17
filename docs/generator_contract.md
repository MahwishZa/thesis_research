# Generator Execution Contract

**Decided 2026-09-17** (ledger D-38). This is the exact, reproducible
specification of how answers are generated. Both arms run under it; nothing in
it differs between them.

---

## 1. The decision, and why

**Model: `meta-llama/Meta-Llama-3-8B-Instruct` — RAG²'s own generator, kept.**
**Venue: free-tier remote GPU (Colab or Kaggle T4, 16 GB).**
**Precision: 4-bit NF4.**

The local machine (RTX 2050, 4 GB VRAM) cannot run an 8B model at any
precision:

| Precision | VRAM needed incl. KV cache at RAG-length contexts | Fits 4 GB? |
|---|---|---|
| BF16 | ≈ 18–20 GB | no |
| 8-bit | ≈ 10 GB | no |
| 4-bit NF4 | ≈ 6–8 GB | **no** |

CPU-only inference is technically possible and practically unusable for a run
that has to be repeatable.

**That rules out the venue, not the model.** The tempting move is to swap in a
model small enough for 4 GB. It is the wrong one: it would trade away fidelity
to RAG² to solve a problem that a free T4 already solves. A T4 fits Llama-3-8B
at 4-bit with headroom, so the model stays.

Quantisation is the one deviation from the paper, and it is a precision
change applied **identically to both arms**, not a different system. It is
reported as a limitation.

**Declared fallback:** Llama-3 is gated on Hugging Face. If the licence cannot
be obtained, `Qwen/Qwen2.5-7B-Instruct` (ungated, same parameter scale)
substitutes, and the substitution is stated in the thesis rather than made
quietly. Accepting the licence is a student action, not a technical blocker.

## 2. Why greedy decoding, and why that is not a knob

`do_sample=False`, no temperature, no top-p, no top-k.

With sampling, one run per arm would be a *sample* from a distribution, and
separating a real HAR difference from decoding variance would need many runs
per question — which the compute budget does not allow. With greedy decoding,
one run per arm **is** the measurement, and a repeat run reproduces it exactly.

`GenerationConfig` raises if `do_sample=True` rather than permitting it
quietly, and raises if temperature or top-p are set while greedy, because
recording parameters the run ignored would misdescribe it.

There is no random seed, because nothing samples. A seed that does not matter
is worse than no seed: it implies a control that is not there.

## 3. The contract

| Field | Value | Recorded in |
|---|---|---|
| Model id | `meta-llama/Meta-Llama-3-8B-Instruct` | `ModelSpec.model_id` |
| Revision | **a commit sha — to be pinned at download** | `ModelSpec.revision` |
| Quantization | `nf4`, double quant, bf16 compute | `ModelSpec.quantization` |
| Prompt template | shared; `assert_prompt_parity` | `RunConfig` |
| Chat template | the checkpoint's own, applied to the built prompt | generator metadata |
| Decoding | greedy, `max_new_tokens=256` | `GenerationConfig` |
| Context budget | 5 admitted passages, both arms | `assert_budget_parity` |
| Generator instance | **one object, shared** | `assert_generator_parity` |
| Software versions | transformers, torch, bitsandbytes, python | run manifest |

**`revision` must be a commit sha and `ModelSpec` refuses `"main"`.** A branch
name resolves to different weights over time, so a result recorded against one
could not be reproduced. The sha is read off the Hub at download time and
written into the run manifest. It is deliberately **not** pre-filled in the
code: a placeholder would be a fabricated provenance record.

## 4. What must be identical between arms

Everything in §3. The generator is not part of the intervention — the
admission rule is — so there is no case in this design where the arms should
differ in any of it. Three assertions in `experiments/evaluation/runner.py`
enforce the ones that could drift silently.

## 5. Expected cost

| Quantity | Estimate | Basis |
|---|---|---|
| Weights download | ≈ 16 GB | published parameter count — **ESTIMATED** |
| VRAM at 4-bit, RAG-length context | ≈ 6–8 GB | **ESTIMATED** |
| Answers per full run | ≈ 200 (100 questions × 2 arms) | question budget |
| Tokens generated per run | ≈ 51k at 256 max new tokens | arithmetic |

**Generation speed is not estimated here, because it has not been measured.**
No tokens/second figure appears in this repository until one is produced on
the actual venue. The first real action on the remote machine is a short
timing check (§6), not the experiment.

## 6. Execution procedure

1. Accept the Llama-3 licence on Hugging Face; create a read token.
2. Open a T4 session. Install pinned `transformers`, `bitsandbytes`,
   `accelerate`.
3. Download the model **recording the resolved commit sha**.
4. **Timing check:** generate 5 answers, record wall-clock and tokens/second.
   This is the first measurement and it decides whether the full run fits the
   session limit. It is engineering measurement, not a result.
5. Fit λ, θ, H on the **validation split only**.
6. Run both arms over the frozen test manifest, one shared generator object.
7. Download the raw JSONL. It is never overwritten (`run_experiment` refuses
   an existing path).

Steps 5–7 require the corpus, the approved questions and the filter
checkpoint. None has been run.

## 7. Status

**DECIDED. NOT EXECUTED.**

- `HuggingFaceGenerator` is implemented and unit-tested without loading a
  model.
- No model has been downloaded. No generation has been performed. No
  tokens/second figure exists.
- Blocking: filter checkpoint (`docs/filter_training.md`), corpus, approved
  questions, and the Hugging Face licence.
