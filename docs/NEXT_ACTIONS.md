# Next Actions

**Updated:** 2026-09-17 · Repository at 206 passing tests.

---

## DONE

* **Abstention asymmetry resolved.** `AbstentionPolicy` added and configurable;
  default `ANSWER_ALWAYS` matches what the baseline already does when its
  filter admits nothing. New `UNGROUNDED` output state so an answer produced
  from an empty evidence block is not mislabelled `GROUNDED`. Rationale in
  `docs/EXPERIMENTAL_PARITY_AUDIT.md` §2.
* **Abstention accounting.** `evaluation/stats.py` reports answered, abstained,
  hallucinated and non-hallucinated separately, with `answer_coverage` beside
  every rate. `compare_systems()` returns `interpretable: false` and emits no
  headline difference when a system answered nothing.
* **Prompt parity enforced** before a run starts, not assumed.
* **RAG² filter recipe recorded from source** (`docs/RAG2_CLASSIFIER_FEASIBILITY.md`):
  AdamW, lr 3e-5, seq 512, stride 128, batch 16, 40 epochs, checkpoint per epoch.
* **Hardware status documented** (`docs/HARDWARE_AND_RESOURCE_STATUS.md`),
  distinguishing the audit container from the laptop.
* **Question schema extended** with `ambiguity_candidate`, a reference-answer
  thinness check, a corpus-support check, an answerability screen and a pool
  summary.
* **206 tests pass.** 15 added this round.

## IN PROGRESS

* **Step 1 — corpus build**, running on the laptop. Not touched by this work.
* **Step 2 — question pool.** Infrastructure is ready; no questions sourced yet.

## BLOCKED

| Blocked on | What it blocks | Why |
|---|---|---|
| **Laptop verification** (§6 of the hardware doc) | every VRAM/runtime claim | CUDA, driver and torch build are UNKNOWN; this audit ran on a different machine |
| **Generator decision** | Steps 4–6, generation parameters, seed | Llama-3-8B does not fit 4 GB VRAM in any configuration, including 4-bit |
| **Filter training venue** | the baseline arm | Flan-T5-large training needs ≈12 GB before activations; the released checkpoint does not exist |
| **Corpus completion** | Step 3 onward | Retrieval, candidate sets, evidence freezing |
| **Retrieval + reranking** | Step 3 | Not implemented; `retrieval_external: True` |
| **Annotator availability** | Step 7, κ | One annotator is workable; document it if the second is unavailable |

## NEXT — in order

**1. Verify the laptop (15 minutes, unblocks the most).**

```powershell
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"
huggingface-cli whoami
```

Paste the output into `docs/HARDWARE_AND_RESOURCE_STATUS.md` §6.

**2. Decide the generator, and write the decision down.** Llama-3-8B will not
run on this GPU. Three honest routes: run generation on a free cloud GPU and
keep the base paper's model; use a smaller instruct model locally and document
the deviation; or CPU inference locally and accept the wall-clock cost. **Do
not download any model until this is decided** — 31 GB free with a corpus still
downloading.

**3. Start sourcing evaluation questions.** Longest-lead item, needs no GPU and
no corpus. Target ~150–200 candidates, human-reviewed down to ~100. Per
question: transcribe the reference answer from a named dated source, never
generate it.

```bash
python -c "from evaluation.questions import EvaluationQuestion, validate; ..."
```

Record `reference_source` and `reference_date` for every item; the firewall
check will later refuse any question whose reference evidence also appears in
its candidate set.

**4. Pick the filter training venue** (free T4 session is sufficient) and decide
base size — see `RAG2_CLASSIFIER_FEASIBILITY.md` §2D.

**5. Implement the concrete generator** once step 2 is decided, behind the
existing `Generator` interface. Fix greedy decoding (temperature 0) so a single
run suffices.

**6. Build retrieval and reranking** to produce frozen candidate sets, once the
corpus completes.

**Do not** start a pilot study, add metrics beyond HAR and accuracy, or tune θ
on anything but the validation split.

---

## Standing limitations to state in the thesis

1. The baseline is an **adaptation** of RAG², not a reproduction: different
   corpus, retrained filter, thesis-authored prompt, and balanced multi-corpus
   retrieval not reproduced.
2. The ~100-question evaluation set is a **practical budget, not a powered
   sample size**. Report the detectable effect, do not claim adequate power.
3. If only one annotator is available, say so and omit κ rather than
   substituting a number.
4. Execution hardware differs from the paper's; this is a resource limitation,
   not a methodological one.
