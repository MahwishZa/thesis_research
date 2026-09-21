# Step 4: Training the RAG² Filter Checkpoint

**Status:** Ready to execute. All code is in place with CPU-safe guards. This is the critical blocker before running end-to-end experiments.

## Overview

The RAG² baseline filter must be trained on general-medical MedQA questions (never the thesis's Alzheimer's questions). The pipeline has two phases:

1. **Label Generation:** Retrieve passages, score with Llama-3-8B-Instruct, apply the label decision tree
2. **Checkpoint Training:** Fine-tune Flan-T5-large on the generated labels

## Phase 1: Generate Training Labels

**Platform:** Kaggle or Google Colab (GPU required, CPU is too slow)

**Command:**
```bash
python -m experiments.filter_training.build_labels \
    --n-questions 20 \
    --output experiments/filter_training/labels/medqa_filter_labels.json \
    --model-revision <pinned-commit-sha> \
    --max-new-tokens 96 \
    --n-textbook-passages 3000
```

**Parameters:**
- `--n-questions 20`: Scope recommendation. At ~1.5 tokens/sec per generation on CPU, 20 questions is ~40–55 min. Do NOT attempt 500+ without GPU.
- `--max-new-tokens 96`: Rationale length (default 256). Reduce to speed up CPU runs proportionally.
- `--n-textbook-passages 3000`: Index size (default 50,000). The smaller size still covers medical diversity and trains faster (~15–20 min index build vs ~5 h for default).
- `--model-revision`: Pinned commit SHA for Llama-3-8B-Instruct (gated model, requires HF token). Example: `"e2bde3gf7e2a51e4c5d2f1a9b8c7d6e5"` (use the real SHA from the model card).

**Troubleshooting:**

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | HF token not authenticated | In Kaggle/Colab: `from huggingface_hub import login; login(token=your_token)` before importing the script |
| `AssertionError: Torch not compiled with CUDA enabled` | Kaggle `pip install` replaced CUDA torch with CPU-only build | Re-run the setup cell: `pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| `No space on device` | Kaggle session disk limit hit | Delete old checkpoints or reduce `--n-textbook-passages` further (1000 still works, just less diversity in the index) |
| Model fails to load / OOM | 16GB Llama-3-8B doesn't fit in ~13GB free RAM | Use a smaller substitute for this step only (e.g., Llama-2-7B-Chat), document the deviation in status_and_decisions.md |

**Output:** `experiments/filter_training/labels/medqa_filter_labels.json`  
**Expected:** ~20–50 KB JSON file with labeled question–passage pairs

## Phase 2: Train the Checkpoint

**Platform:** Same GPU session (or copy the labels.json and run locally if available)

**Command:**
```bash
python -m experiments.filter_training.train \
    --labels experiments/filter_training/labels/medqa_filter_labels.json \
    --output-dir checkpoints/rag2_filter \
    --epochs 3
```

**Parameters:**
- `--labels`: Path to the JSON file from Phase 1
- `--output-dir`: Where to save the trained checkpoint (creates `pytorch_model.bin`, `config.json`, etc.)
- `--epochs`: Number of training epochs. The paper uses 40; for a 20-question label set and Colab's free tier, 3–5 is more realistic. Record your choice in `status_and_decisions.md`.

**Expected Output:**
```
Saved checkpoint to checkpoints/rag2_filter
Wrote CheckpointRecord to checkpoints/rag2_filter/checkpoint_record.json
```

The checkpoint record includes:
- Training configuration (epochs, batch size, learning rate)
- Label distribution
- Timestamp

## Phase 3: Integrate the Checkpoint

Once training is complete:

1. **Copy the checkpoint** to your repo: `checkpoints/rag2_filter/`
2. **Update `current_objectives.md`**: Change "no trained checkpoint exists" to "checkpoint trained on [date] with [N] labels, [E] epochs"
3. **Update `status_and_decisions.md`**: Add a dated entry (e.g., 2026-09-22) documenting:
   - How many labels were generated
   - Any deviations (e.g., smaller model, fewer epochs)
   - The checkpoint path and test accuracy (if available)

4. **Run a validation test:**
   ```bash
   python -m experiments.runners.run_end_to_end \
       --output-dir experiments/outputs/checkpoint_validation \
       --n-questions 5 \
       --rag2-checkpoint checkpoints/rag2_filter
   ```
   This should show `baseline_is_trained_rag2: true` in the report (instead of the default `false`).

## Next Step After Checkpoint

Once the checkpoint is integrated:

1. **Fit λ, θ, and the half-life** on the validation split (23 questions)
2. **Run the full three-arm evaluation** on the test set (90 questions)
3. **Analyze results** and write up findings

## Honest Risks & Limitations

- **RAM on Kaggle/Colab free tier:** Llama-3-8B needs ~16 GB; free sessions have ~13 GB. If `from_pretrained` fails, use a smaller model as an explicit deviation (document it).
- **Label scale:** 20 questions is small. The paper's original training set is much larger. Record this as a methodology limitation, not a hidden substitution.
- **Quantization:** The code defaults to `nf4` (4-bit) on CUDA, `float32` on CPU. Both are correct; record which was used.
- **Training time:** On Colab's free T4 GPU, 20 labels × 3 epochs is roughly 10–20 minutes. On a P100 or better, faster. Record your actual timing.

---

**See also:** `experiments/filter_training/build_labels.py`, `experiments/filter_training/train.py`, and `experiments/filter_training/config.py` for full documentation and hyperparameter details.
