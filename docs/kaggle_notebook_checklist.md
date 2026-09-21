# Kaggle Notebook Setup Checklist for Step 4

**Use this when opening a new Kaggle notebook for label generation and checkpoint training.**

## Pre-Notebook Setup

- [ ] Ensure GPU is enabled in Kaggle settings (Settings → Compute Options → Toggle GPU ON)
- [ ] Verify at least 10 GB free disk space
- [ ] Have your Hugging Face token ready (`hf_*` string from https://huggingface.co/settings/tokens)

---

## Notebook Setup (Cell 0)

```python
# Clone the repository
!git clone https://github.com/MahwishZa/thesis_research.git
%cd thesis_research
!git checkout main
```

Expected: Repository cloned, main branch checked out

---

## Cell 1: Install Dependencies & Authenticate HF

```python
# Install dependencies (if not already present in Kaggle)
!pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers datasets bitsandbytes accelerate

# Authenticate with Hugging Face (required for gated Llama-3-8B-Instruct)
from huggingface_hub import login
login(token="<your_hf_token>")  # Replace with your actual token

print("✓ Dependencies installed and HF authenticated")
```

**Why:** Kaggle sometimes has outdated torch (CPU-only). This ensures CUDA-enabled version. The `login()` call is **critical** — without it, Llama-3-8B will return 401 Unauthorized.

---

## Cell 2: Generate Labels (Phase 1)

```bash
python -m experiments.filter_training.build_labels \
    --n-questions 20 \
    --output experiments/filter_training/labels/medqa_filter_labels.json \
    --model-revision e2bde3gf7e2a51e4c5d2f1a9b8c7d6e5 \
    --max-new-tokens 96 \
    --n-textbook-passages 3000 \
    --seed 42
```

**Expected output:**
```
loaded 3000 textbook passages
built textbook index: 3000 x 768 in 20s
loaded 20 MedQA questions
  ...scoring...
  ...20/20 pairs scored (1200s elapsed, ~0s remaining)
scored 20 pairs
label distribution:
{...}
wrote 20 labelled examples to experiments/filter_training/labels/medqa_filter_labels.json
```

**Duration:** ~20–25 minutes  
**Disk space used:** ~1–2 GB (index) + ~50 KB (labels JSON)

**If it fails:**
- **`401 Unauthorized`**: Go back to Cell 1, add `login()` call
- **`AssertionError: Torch not compiled with CUDA`**: Re-run Cell 1 (the pip install)
- **`CUDA out of memory`**: Reduce `--max-new-tokens` to 64, or `--n-textbook-passages` to 1500
- **`No space on device`**: Delete old checkpoints or kernel and restart with less data

---

## Cell 3: Verify Labels Were Generated

```python
import json
from pathlib import Path

labels_path = Path("experiments/filter_training/labels/medqa_filter_labels.json")
labels = json.loads(labels_path.read_text())

print(f"✓ {len(labels)} labels generated")
print(f"First label keys: {list(labels[0].keys())}")
print(f"Label distribution: {[l.get('label') for l in labels[:5]]}")
```

Expected: 20 label records, each with fields like `pair_id`, `question`, `label`, etc.

---

## Cell 4: Train Checkpoint (Phase 2)

```bash
python -m experiments.filter_training.train \
    --labels experiments/filter_training/labels/medqa_filter_labels.json \
    --output-dir checkpoints/rag2_filter \
    --epochs 3 \
    --seed 42
```

**Expected output:**
```
Training on 18 examples, validating on 2
Epoch 1/3: loss=..., val_loss=...
Epoch 2/3: loss=..., val_loss=...
Epoch 3/3: loss=..., val_loss=...
Saved checkpoint to checkpoints/rag2_filter
Wrote CheckpointRecord to checkpoints/rag2_filter/checkpoint_record.json
```

**Duration:** ~10–15 minutes  
**Disk space used:** ~500 MB (checkpoint files)

---

## Cell 5: Verify Checkpoint & Download

```python
from pathlib import Path
import json

checkpoint_dir = Path("checkpoints/rag2_filter")
assert (checkpoint_dir / "pytorch_model.bin").exists(), "Model not found!"
assert (checkpoint_dir / "config.json").exists(), "Config not found!"
assert (checkpoint_dir / "checkpoint_record.json").exists(), "Record not found!"

# Read the checkpoint record
with open(checkpoint_dir / "checkpoint_record.json") as f:
    record = json.load(f)
    print(f"✓ Checkpoint trained on {record['timestamp']}")
    print(f"  Epochs: {record['epochs']}")
    print(f"  Labels: {record['label_count']}")
    print(f"  Label distribution: {record['label_distribution']}")
```

**Then download the entire `checkpoints/rag2_filter/` folder to your local machine.**

---

## After Notebook: Integrate Locally

Once you've downloaded the checkpoint:

```bash
# In your repo (main branch)
cp -r /path/to/downloaded/rag2_filter checkpoints/rag2_filter

# Test the checkpoint
python -m experiments.runners.run_end_to_end \
    --output-dir experiments/outputs/checkpoint_validation \
    --n-questions 5 \
    --rag2-checkpoint checkpoints/rag2_filter

# Should show: baseline_is_trained_rag2: true (instead of false)

# Commit
git add checkpoints/rag2_filter
git commit -m "Add trained RAG2 filter checkpoint (20-label, 3-epoch)"
git push origin main
```

---

## Troubleshooting Reference

See full details in `docs/step_4_filter_training.md` — this checklist is the quick path.

**Key gotchas:**
1. **Missing `login(token=...)` → 401 Unauthorized** — Cell 1 must run first
2. **`pip install` replaces torch with CPU-only → CUDA error** — Re-run the pip line from Cell 1
3. **Model load failure → OOM** — Use smaller model (document as deviation) or reduce batch size in config
4. **Timeout → Process killed** — Kaggle has 12-hour per notebook limit; should be fine for 20 questions, but monitor progress

---

## Post-Training

Once checkpoint is committed to main:

1. **Update `docs/current_objectives.md`**: Remove "no trained checkpoint exists"
2. **Update `docs/status_and_decisions.md`**: Add entry like:
   ```
   | 2026-09-22 | **RAG² filter checkpoint trained** — 20 MedQA labels, 3 epochs, Flan-T5-large. Deviation: smaller label set than paper (20 vs ~5000), explicitly recorded. Checkpoint validated against 5-question fixture, baseline_is_trained_rag2 now true. |
   ```
3. **Next step:** Fit λ/θ/H on validation split (23 questions)

---

## One More Thing

If you get stuck at any point, the repo has full diagnostics:

- `experiments/filter_training/build_labels.py` → full docstring with timing estimates
- `experiments/filter_training/train.py` → full docstring with config details
- `experiments/filter_training/config.py` → FilterTrainingConfig reference
- `docs/step_4_filter_training.md` → detailed troubleshooting table
