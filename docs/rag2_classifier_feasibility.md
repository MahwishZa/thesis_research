# RAG² Filter (Classifier) — Recipe and Feasibility

**Updated:** 2026-09-17 · **Source:** the official repository, read directly —
`classifier/README.md`, `classifier/run_classifier.py`,
`classifier/run/run_large_train_xl_000.sh`. Nothing here is from memory.

---

## 1. The actual recipe

The filter is a **Flan-T5 seq2seq model** fine-tuned with HF Accelerate. The two
labels are added as special tokens and the embedding matrix is resized:

```python
new_tokens = ["[HELPFUL]", "[NOT_HELPFUL]"]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
```

**Training command, verbatim from the released launch script:**

| Setting | Value | Source |
|---|---|---|
| Base model | a Flan-T5 checkpoint with the label tokens added | `MODEL=/classifier/model/updated_flan_t5_model` |
| Learning rate | **3e-5** | launch script |
| Optimizer | **AdamW** | `torch.optim.AdamW` in `run_classifier.py:528` |
| Max sequence length | **512** | launch script |
| Doc stride | **128** | launch script |
| Train batch size per device | **16** | launch script |
| Epochs | **40** | launch script |
| Checkpointing | **per epoch** | launch script |
| Weight decay | **0.0** (default) | `run_classifier.py:283` |
| Warmup steps | **0** (default) | `run_classifier.py:306` |
| Gradient accumulation | configurable, default **1** | `run_classifier.py:293` |
| LR scheduler | configurable (HF default linear) | `run_classifier.py:299` |
| Precision | **not hardcoded** — mixed precision comes from `accelerate config` | `accelerator.use_fp16`, line 499 |
| Devices | single GPU in the script (`GPU=1`); Accelerate supports multi-GPU | launch script |

**Training data format** — a JSON list, one object per example:

```json
{
  "id": "llama3_5%_23600",
  "answer": "[NOT_HELPFUL]",
  "dataset_name": "llama3_5%",
  "question": "Given the following evidence, determine whether it helps answer the provided question.\n\nEvidence: ...\n\nQuestion: ..."
}
```

Labels are the two tokens; supervision comes from perplexity-based labels of
rationales. **Inference** takes a softmax over the `[HELPFUL]` /
`[NOT_HELPFUL]` token logits and predicts the higher-probability label — which
is exactly what `systems/baseline/admission.py` already implements.

**One ambiguity, recorded rather than resolved.** The launch script sets
`MODELNAME=flant5` and points at a generic directory; its filename mentions both
"large" and "xl". The paper record in this repository's ledger states
Flan-T5-large (≈780 M). **The released script does not pin the base size.**
Confirm against the paper before stating a size in the thesis.

---

## 2. Feasibility on the actual hardware

Parameter-count arithmetic below is **ESTIMATED**; nothing was trained.

### A. Flan-T5-large training on the RTX 2050 (4 GB VRAM) — **NO**

For ≈780 M parameters with AdamW, mixed precision still keeps fp32 master
weights and two optimizer moments:

| Component | Memory |
|---|---|
| fp32 master weights | ≈ 3.1 GB |
| Gradients | ≈ 3.1 GB |
| AdamW moments (×2) | ≈ 6.2 GB |
| **Subtotal, before activations** | **≈ 12.4 GB** |
| Activations at batch 16 × seq 512 | several GB more |

**≈ 3× over budget before a single activation.** Not a tuning problem — batch 1
with gradient accumulation still leaves the ~12.4 GB optimizer footprint.

### B. CPU training on 16 GB RAM — **technically possible, practically not**

The optimizer state fits in 16 GB, so it would not crash. But the recipe is
**40 epochs at batch 16, seq 512** on a 780 M seq2seq model. On 6 CPU cores
this is days to weeks. **Not feasible within the thesis schedule.**

### C. Is another machine required? — **Yes, for training only**

The job is small by modern standards: a free-tier 16 GB T4 session handles
Flan-T5-large fine-tuning comfortably. **Filter *inference* runs locally**
(≈1.6 GB fp16 fits 4 GB), so only the one-off training needs to move.

### D. Smaller defensible configurations

Ordered by how much they deviate from the paper:

1. **Flan-T5-large, LoRA, 4-bit base.** Same architecture and labels; different
   optimization. Might fit 4 GB at batch 1–2. Deviation: parameter-efficient
   rather than full fine-tuning.
2. **Flan-T5-base (≈250 M), full fine-tune.** Same architecture family, same
   label scheme, same data. Deviation: smaller capacity. Fits a free T4 easily
   and may fit 4 GB at small batch.
3. **Fewer epochs.** 40 epochs on a 5% sample is heavy; report the epoch count
   actually used and the validation accuracy reached.

All three keep the `[HELPFUL]`/`[NOT_HELPFUL]` decision procedure intact, which
is the part the thesis measures. **None substitutes a different architecture.**

### E. Minimum defensible baseline adaptation

The original checkpoint is not distributed, so an identical baseline is
impossible for anyone. Ranked:

1. **Best:** train Flan-T5-large on a free cloud GPU using the recipe in §1,
   report validation accuracy, then run inference locally. Closest to the paper.
2. **Acceptable:** Flan-T5-base under the same recipe, with the size deviation
   stated.
3. **Weakest, and only with an explicit caveat:** zero-shot Flan-T5 as the
   filter. This is *not* the paper's method — the paper's contribution is the
   perplexity-trained filter — and a thesis using it must say the baseline is
   an untrained approximation, not RAG²'s filter.

**Whichever is chosen, the no-filter control is the honest floor**: if the
trained filter underperforms it, the baseline was weak and any advantage for
the proposed solution must be read against that, not celebrated.

---

## 3. What must be documented in the thesis

The checkpoint is unavailable · the filter was retrained by the student, with
which base size, epochs and validation accuracy · the training ran on
different hardware than the rest of the pipeline · whether the base size
matches the paper · that this makes the baseline an **adaptation**, not a
reproduction.
