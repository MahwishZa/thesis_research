# Research Roadmap Status Snapshot — 2026-09-21

**Last updated:** 2026-09-21  
**Timeline context:** ~7 days until end of week. Full roadmap is 14 steps; current blockers outlined below.

## Completed (Steps 1–3)

| Step | Title | Status | Notes |
|---|---|---|---|
| 1 | Build Alzheimer's corpus | ✅ DONE | 7 pipeline stages verified end-to-end; 4.3M chunks frozen |
| 2 | Index & retrieval setup | ✅ DONE | MedCPT encoder, DenseIndex, reranker all ready |
| 3 | Question pool audit & split | ✅ DONE | 123 original → 82 repaired + 31 merged + 10 removed; 113 usable → 23 validation / 90 test |

## Critical Blocker (Step 4)

### **Train RAG² Filter Checkpoint**

**Blocker Status:** ❌ NOT STARTED — requires external GPU compute

**What's ready:**
- Code: `experiments/filter_training/build_labels.py`, `train.py`, all CPU-safe guards in place
- Instructions: `docs/step_4_filter_training.md` (detailed 3-phase guide)
- Data sources: MedQA dataset + MedRAG textbooks (public, no license issues)

**What's needed:**
1. Run label generation on Kaggle/Colab (20 questions × 2 generations recommended = ~40–55 min with `--max-new-tokens 96`)
2. Train Flan-T5-large checkpoint (3–5 epochs, ~10–20 min on Colab T4)
3. Integrate checkpoint into repo and validate with `run_end_to_end.py`

**Estimated time:** 1–2 hours of compute (mostly I/O and waiting), ~30 min hands-on

**Known risks:**
- Llama-3-8B load failure if free-tier RAM < 16GB → fallback to smaller model (recorded deviation)
- Kaggle/Colab torch version gotcha (documented workaround in troubleshooting table)

**Next action:** Go to `docs/step_4_filter_training.md` and follow Phase 1 on Kaggle/Colab

---

## Next (Steps 5–14, ready after Step 4)

| Step | Title | Readiness | Dependency |
|---|---|---|---|
| 5 | Fit λ, θ, H on validation | ⚙️ CODE READY | Needs checkpoint + validation set (ready) |
| 6 | Freeze test set | ⚙️ CODE READY | Needs Step 5 complete |
| 7–10 | Run 3 arms + ablations | ⚙️ CODE READY | Needs checkpoint + frozen test |
| 11 | Analyze hallucination/outdated | ⚙️ CODE READY | Needs Step 7–10 output |
| 12 | Per-subgroup metrics | ⚙️ CODE READY | Needs Step 7–10 output |
| 13 | Statistical test | ⚙️ CODE READY | Needs Step 7–10 output |
| 14 | Write-up | 📝 OUTLINE ONLY | Needs Step 11–13 complete |

**Critical path:** Step 4 → Steps 5–6 (sequential, ~1 h) → Steps 7–10 (parallel setup, ~8–16 h compute) → Steps 11–14 (sequential, ~6–12 h, can overlap with running time)

---

## Can This Finish Before Week's End?

**Given:** 7 days remaining, Step 4 blocking everything  
**Realistic:** Depends entirely on when Step 4 runs

### Scenario A: Step 4 runs TODAY
- Step 4: 1–2 h (today/tomorrow)
- Steps 5–6: 1 h (tomorrow)
- Steps 7–10: 8–16 h compute (can run while you write)
- Steps 11–14: 6–12 h writing + analysis
- **Finish date:** Early next week (Wednesday–Friday), assuming continuous work

### Scenario B: Step 4 runs end of week
- **Finish date:** Week AFTER next

### Scenario C: Reduce scope
- Use only 20–30 test questions instead of full 90
- Skip some ablations (e.g., half-life sensitivity)
- **Estimated savings:** 4–8 h
- **New finish date:** Friday–Saturday if Step 4 starts today

---

## Honest Assessment

**The code is ready.** Every blocker is now external:
- Your GPU access (Kaggle/Colab)
- Time to run the label generation
- Time to train and integrate

**No methodology or design work remains.** All decisions are recorded in `docs/status_and_decisions.md` and `current_objectives.md`.

**What matters most right now:** Starting Step 4. The 20-question label set is small (a deviation from the paper's scale), but it's better than no checkpoint and will prove the pipeline works end-to-end before scaling up later if needed.

---

## Files to Reference

- **What to do next:** `docs/step_4_filter_training.md`
- **Why we made these choices:** `docs/status_and_decisions.md` (§3.2–3.4, entries 2026-09-21k through 2026-09-21i)
- **The research question:** `docs/current_objectives.md`
- **Specification:** `docs/research_experimental_specification.md` §10 (filter training)

---

## Running End-to-End Tests (No GPU Needed)

To verify the pipeline is wired correctly without training:

```bash
python -m experiments.runners.run_end_to_end \
    --output-dir experiments/outputs/sanity_check \
    --n-questions 4 \
    --ablation-lambdas 0,1.0
```

Expected output: 4 questions × 4 arms (3 proposed λ values + 1 baseline) = 16 records, all `status: ok`. Metrics computed on fixture, report generated. This takes <1 min and proves the end-to-end wiring works before GPU costs start.

This is safe to run anytime; it touches nothing real (only fixtures and temp output dirs).
