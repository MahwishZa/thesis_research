# Feasibility and Alignment Audit

**Date:** 2026-09-17 · **Repository state audited:** `0298d0d` (clean working tree)

> **Note added 2026-09-18:** the FAISS-based retrieval plan this audit
> discusses (§5C and elsewhere) was superseded by D-37: retrieval is
> implemented as a flat **exact** index, deliberately not FAISS — see
> `docs/next_steps.md` D-37, `docs/system_specification.md` §3, and
> `experiments/retrieval/index.py` for why. FAISS is no longer part of
> the plan; the point-in-time hardware figures below are otherwise
> unaffected and left as measured.

Every quantity below carries a label: **MEASURED** (observed in this
environment), **ESTIMATED** (derived from published model specifications),
**ASSUMED** (a stated premise), or **UNKNOWN / REQUIRES VERIFICATION**.
No runtime figure in this document was produced by running a model.

---

## 1. Executive summary

The baseline system and the proposed solution/system are **scientifically
aligned** on every cross-arm control that matters: identical prompt template,
identical context formatting, identical evidence ordering, identical generator
instance, and a candidate set supplied from outside the arms. This was verified
by reading the code, not by trusting the documentation.

Two findings qualify that, and one is a genuine blocker.

1. **The proposed solution/system can abstain; the baseline cannot.** When
   nothing is admitted it returns `prediction=None` and
   `output_state=ABSTAIN`. An abstention contains no claims, so it cannot
   contain a hallucinated claim. Scored naively, **abstention lowers the
   hallucination rate for free**. This must be handled by dual reporting
   before any comparison is run.
2. **Nothing in the model stack is installed, and no GPU is present** in this
   audit environment. The pipeline cannot be executed or benchmarked here. Any
   runtime number produced from this environment would be fabricated, so none
   is given.

The thesis implementation is an **adaptation** of RAG², not a reproduction,
and must be described that way. The RAG² repository itself states it "is not
a full, one-command reproduction of the paper" — the corpora are not
redistributable and the trained filter checkpoint is no longer available.

---

## 2. Actual hardware and software environment

**All MEASURED in the audit container on 2026-09-17.**

| Property | Value | Label |
|---|---|---|
| CPU | Intel Xeon @ 2.10GHz | MEASURED |
| Cores / threads | 4 / 4 (1 thread per core) | MEASURED |
| System RAM | 15 GiB total, ~15 GiB available | MEASURED |
| GPU | **none** — `nvidia-smi` not present | MEASURED |
| VRAM | not applicable | MEASURED |
| CUDA toolkit | **not present** — `nvcc` absent | MEASURED |
| NVIDIA driver | not applicable | MEASURED |
| Python | 3.11.15 | MEASURED |
| Disk | 252 G total, ~30 G available | MEASURED |
| Hugging Face cache | absent | MEASURED |

Installed packages relevant to the pipeline (**MEASURED**):

| Package | Status |
|---|---|
| numpy | 2.4.6 |
| PyYAML | present (declared dependency) |
| torch | **NOT INSTALLED** |
| transformers | **NOT INSTALLED** |
| sentence-transformers | **NOT INSTALLED** |
| faiss | **NOT INSTALLED** |
| accelerate / bitsandbytes / vllm | **NOT INSTALLED** |
| huggingface_hub / datasets | **NOT INSTALLED** |
| scipy / scikit-learn / pandas | **NOT INSTALLED** |

### 2.1 What this environment is, and is not

> **This container is the audit environment. It is not the student's machine.**

No corpus process is running here (**MEASURED**: no matching process). The
Alzheimer's corpus build is running elsewhere, on hardware this audit cannot
see. Consequently:

* **The student's CPU, RAM, GPU, VRAM, CUDA version and driver are
  UNKNOWN / REQUIRES VERIFICATION.**
* Every runtime and VRAM figure in sections 7–13 is **ESTIMATED** from
  published model specifications, never measured.
* The student must run the verification commands in section 21 on their own
  machine before any of this is treated as settled.

`pyproject.toml` declares model backends as an **optional** extra
(`[project.optional-dependencies] models = ["torch>=2.0",
"transformers>=4.40"]`), which is why the package imports and the 140 tests
pass with no model stack installed (**MEASURED**: 140 tests, all passing).

---

## 3. RAG² paper versus official repository

Read from the official repository's README (**MEASURED**, fetched
2026-09-17) and the paper record.

**Pipeline, as the repository states it:**

```
question
  -> [1] rationale generation (LLM); the rationale becomes the retrieval query
  -> [2] retrieval + reranking (MedCPT + FAISS) across 4 corpora, rerank to top-k
  -> [3] filtering (Flan-T5): each (question, snippet) -> [HELPFUL] / [NOT_HELPFUL]
  -> [4] answer generation (LLM), conditioned on filtered evidence
```

**What the repository releases:** the retriever pipeline, the filter training
and evaluation code, and one representative perplexity-labelled training
artifact (a 5% sample).

**What it does not release, in its own words:**

* the four biomedical corpora (PubMed, PMC, CPG, Textbooks) and their
  embeddings — "multi-GB and partly license-restricted";
* **the exact trained filtering-model checkpoint** — "not available for
  distribution; use `classifier/` to train an equivalent filter on your own
  labeled data."

The README states directly: **"This repository is not a full, one-command
reproduction of the paper."**

**Metrics reported by the paper:** accuracy on three medical question-answering
benchmarks ("improvements of up to 6.1%"). Its only open-ended evaluation is
25 queries scored with ROUGE-L and BERTScore.

**A point the thesis should make explicitly.** The RAG² abstract motivates the
work by stating that LLMs "struggle with hallucinations and outdated
knowledge" — and then measures neither. Measuring hallucination directly is
therefore not a detour from the base paper; it addresses the base paper's own
unmeasured claim.

---

## 4. Baseline alignment

| Component | Original RAG² | Current thesis implementation | Match? | Required change | Scientific impact |
|---|---|---|---|---|---|
| Question input | Medical QA item | Passed to `System.run(question=...)` | Yes | none | none |
| Rationale generation | LLM chain-of-thought, used as retrieval query | **Not implemented.** `rationale` is passed *in* and recorded, never regenerated | No | Produce rationales upstream, or document their absence | If omitted, retrieval differs from RAG²; the arm is no longer a faithful baseline |
| Retrieval | MedCPT + FAISS, balanced over 4 corpora | **Not implemented in repo**; metadata flags `retrieval_external: True` | No | Build the frozen-candidate-set stage | Evidence population differs — the central adaptation |
| Reranking | MedCPT cross-encoder | **Not implemented**; `rerank_score` / `rerank_rank` are supplied as inputs | No | Supply real rerank scores upstream | Ranks drive budget selection in all arms |
| Evidence filtering | Flan-T5 `[HELPFUL]`/`[NOT_HELPFUL]` | `FlanT5RAG2Filter` implemented with real `transformers` code (`systems/baseline/admission.py`, 271 lines) | Structurally yes | Train a checkpoint | Checkpoint unavailable upstream — see §10 |
| Evidence ordering | Not specified in paper | Select by `(rerank_rank, evidence_id)`, then **restore candidate-list order** | Documented deviation | none | Deliberate fairness control; identical in all three arms |
| Final generator | Llama-3-8B-Instruct and others | `Generator` ABC + `CallableGenerator` adapter. **No concrete model generator exists** | No | Implement one | Blocks every run |
| Prompt | Not fully published | `"Answer the question using the provided evidence.\n\nQuestion: …\n\nEvidence:\n…"` | Reimplemented | none | Must be recorded as thesis-authored, not paper-faithful |
| Generation parameters | Not fully published | Not set anywhere | No | Fix and record | Must be identical across arms |
| Output format | Answer text | `ExperimentResult` with `prediction`, `generated_text`, `admitted_evidence_ids`, metadata | Superset | none | Good — supports the required JSONL output |

**Verdict: ADAPTATION.** The corpus, the retrieval collection, the filter
checkpoint, the prompt and the generation parameters all differ from or are
absent in the original. The thesis must not claim reproduction.

---

## 5. Proposed-system alignment

| Component | Baseline | Proposed solution/system | Same / Different | Scientific implication |
|---|---|---|---|---|
| Question | `question` argument | identical | **Same** | Comparison valid |
| Question ID | `sample_id` | identical | **Same** | Pairing possible |
| Candidate evidence | `candidates` argument | identical argument | **Same** | The core control |
| Generator | injected `Generator` | same injected instance | **Same** | No model confound |
| Rationale | passed in, recorded | passed in, recorded | **Same** | Invariant, verifiable post hoc |
| Retrieval / reranking | external, upstream | external, upstream | **Same** | Frozen upstream |
| Evidence filtering | Flan-T5 binary label | score `A(s) = (1−λ)·ρ(s) + λ·R(s,q,t_q)`, admit if `A(s) ≥ θ` | **Different — this is the intervention** | Intended |
| Evidence ordering | rank-select, restore candidate order | decisions built by iterating `candidates` in order; prompt emits admitted-in-candidate-order | **Same** | Verified by reading `decide()` and `build_prompt()` |
| Prompt template | default string | **byte-identical default string** | **Same** | Verified by direct comparison |
| Context formatting | `[id] text` joined by blank lines | identical | **Same** | Verified |
| Generation parameters | not set | not set | Same (both unset) | Must be fixed before running |
| **Abstention** | none — always answers | `prediction=None`, `output_state=ABSTAIN` when nothing admitted | **Different** | **See §18.1 — affects the primary metric** |
| Output state | `None` | `GROUNDED` / `CONTESTED` / `ABSTAIN` | Different | Recording only |

**Does the comparison isolate the intervention?** Yes on every axis except
abstention, which is a property of the intervention rather than a flaw, but
which biases the primary metric if not handled. Two things must change before
running:

1. **Dual reporting of HAR** — conditional on answering, and with abstentions
   counted as non-hallucinated but reported separately with the abstention
   rate. Never report a single HAR number without the abstention rate beside it.
2. **A cross-arm prompt-parity assertion.** All three arms accept a
   `context_prompt` override and nothing checks that the arms were configured
   with the same one. Parity currently holds by default but can be broken
   silently by configuration. Assert equality of the prompt template hash at
   run start.

---

## 6. Alzheimer's corpus adaptation analysis

**1. Which components can remain unchanged?** Rationale generation, MedCPT
query encoding, the MedCPT cross-encoder reranker, the Flan-T5 filter
*architecture* and its `[HELPFUL]`/`[NOT_HELPFUL]` decision procedure, and the
answer-generation step. None of these is corpus-specific in its mechanics.

**2. Which must use the Alzheimer's corpus?** The retrieval collection and its
index. RAG² retrieves evenly across four biomedical corpora; this thesis
retrieves from one Alzheimer's-anchored corpus.

**3. Does replacing the evidence collection change the baseline's scientific
meaning?** **Yes, and this must be stated.** RAG²'s balanced multi-corpus
retrieval is one of its three claimed innovations, introduced specifically to
mitigate retriever bias toward a single source corpus. Retrieving from one
corpus removes that mechanism. The arm therefore reproduces RAG²'s *filtering*
faithfully while *not* reproducing its retrieval design.

**4. Is the adaptation defensible?** Yes, for this thesis, because the research
question concerns evidence admission and answer hallucination, not retrieval
breadth, and because both arms receive the identical frozen candidate set — so
the corpus change cannot favour either arm. It would not be defensible in a
thesis claiming to reproduce RAG²'s reported gains.

**5. Differences that must be documented.** Single AD corpus instead of four
biomedical corpora; balanced multi-corpus retrieval not reproduced; filter
trained by the student rather than the released checkpoint; thesis-authored
prompt; generator and decoding settings chosen by the thesis; corpus snapshot
identifier and date.

**6. Hidden confounders introduced.**

* **Corpus recency skew.** An AD corpus assembled now is denser in recent
  publications than RAG²'s snapshot. Since the proposed intervention scores
  recency, a corpus skewed recent could inflate or mask its effect. Report the
  corpus publication-date distribution.
* **Topical narrowness.** All candidates being AD-anchored raises inter-passage
  similarity, which may compress the relevance signal `ρ(s)` relative to a
  diverse corpus, changing the effective meaning of `θ`.
* **Filter domain shift.** The Flan-T5 filter is trained on general medical QA
  and applied to AD passages. This is a deliberate design choice (it removes
  the suspicion that gains come from domain-specific fine-tuning) but it is a
  documented deviation, not a neutral one.

---

## 7. Llama-3-8B-Instruct feasibility

**Cannot be measured here** — no GPU, no `torch`, no `transformers`
(**MEASURED**). The following are **ESTIMATED** from the model's published
parameter count (8.03 B) and standard precision arithmetic.

| Quantity | Value | Label |
|---|---|---|
| Weights, BF16/FP16 | ≈ 16 GB | ESTIMATED |
| Weights, 8-bit | ≈ 8 GB | ESTIMATED |
| Weights, 4-bit | ≈ 4.5–5 GB | ESTIMATED |
| VRAM incl. KV cache and activations, BF16, RAG-length contexts | ≈ 18–20 GB | ESTIMATED |
| VRAM, 4-bit, RAG-length contexts | ≈ 6–8 GB | ESTIMATED |
| Disk / cache | ≈ 16 GB download | ESTIMATED |
| Host RAM during load | ≈ 16–20 GB | ESTIMATED |

**Consequences.** BF16 requires a ≥24 GB GPU. 4-bit quantisation brings it
within reach of a 12 GB card. CPU-only inference is technically possible and
practically unusable for this experiment.

Generation speed is **UNKNOWN** and depends entirely on hardware this audit
cannot see. **No tokens/second figure is given, because none was measured.**

Access is gated on Hugging Face and **REQUIRES VERIFICATION** that the student
holds an accepted licence and a valid token.

---

## 8. MedCPT feasibility

| Quantity | Value | Label |
|---|---|---|
| Query encoder | BERT-base class, ≈ 109 M parameters | ESTIMATED |
| Weights, FP32 | ≈ 0.44 GB | ESTIMATED |
| VRAM, inference | ≈ 1–2 GB | ESTIMATED |
| CPU-only inference for 100 queries | feasible | ESTIMATED |

The query encoder is small enough that it is not a blocker on any plausible
hardware, including CPU-only. Encoding 100 questions is trivial work.

---

## 9. Reranker feasibility

| Quantity | Value | Label |
|---|---|---|
| Cross-encoder | BERT-base class, ≈ 109 M parameters | ESTIMATED |
| VRAM, inference | ≈ 1–2 GB | ESTIMATED |
| Cost model | one forward pass per (query, passage) pair | ESTABLISHED |

Cost scales with candidates per question, not with corpus size. For 100
questions × k candidates the total pair count is small. Feasible on CPU if
slow; comfortable on any GPU.

**Retrieval / index** (Part 5C): FAISS is **NOT INSTALLED** (**MEASURED**).
Index size, build time and RAM scale with the final corpus, whose size is
**UNKNOWN** until the build completes. Query latency for 100 questions is not
the constraint; index construction RAM is. **REQUIRES VERIFICATION** once the
corpus lands.

---

## 10. Flan-T5 evidence filter feasibility

**Implementation check (MEASURED).** `systems/baseline/admission.py` implements
`FlanT5RAG2Filter` with real `transformers` code: it loads
`AutoModelForSeq2SeqLM`, builds decoder input ids, and scores the
`[HELPFUL]` / `[NOT_HELPFUL]` tokens under `torch.no_grad()`. This matches the
RAG² filtering approach structurally.

**Checkpoint availability.** The original trained checkpoint is **not
available** — stated by the RAG² repository itself. **An equivalent filter must
be trained.** This is the single largest hidden cost in the thesis and it sits
on the critical path.

| Quantity | Value | Label |
|---|---|---|
| Flan-T5-large parameters | ≈ 780 M | ESTABLISHED |
| Fine-tuning VRAM, BF16 + Adam | ≈ 24–40 GB | ESTIMATED |
| Fine-tuning VRAM with LoRA / 8-bit optimiser | ≈ 12–16 GB | ESTIMATED |
| Checkpoint size, FP32 | ≈ 3 GB | ESTIMATED |
| Inference VRAM | ≈ 2–3 GB | ESTIMATED |
| Inference cost | one forward pass per (question, passage) | ESTABLISHED |

**Training hyperparameters — max sequence length, batch size, learning rate,
epochs — are UNKNOWN from the audit and must be read from the released
`classifier/` code before being fixed.** They are not invented here.

**Training data.** RAG² labels come from a correctness-flip decision tree with
a perplexity differential as tie-breaker. Perplexity labels are computed
against a specific base model, which implies a filter per generator backbone.
The repository ships a 5% sample of labelled pairs in the exact schema; the
full labelling run must be reproduced by the student.

**Feasibility verdict: UNKNOWN until the student's GPU is known.** On a 24 GB
card, LoRA fine-tuning is plausible. With no GPU it is not feasible.

---

## 11. Training requirements (summary)

Only one component requires training: the Flan-T5 filter. Nothing else in the
pipeline is trained. The proposed solution/system has **no trained
parameters** — `λ`, `θ` and `H` are selected on development data, not learned.
This is a meaningful feasibility advantage and should be stated in the thesis.

## 12. Inference requirements (summary)

Per question: 1 rationale generation (LLM) → 1 query encode (MedCPT) → 1 index
search → k rerank passes → k filter passes → 1 answer generation (LLM). The two
LLM calls dominate; everything else is small-model work.

---

## 13. Estimated runtime for 100 questions

**No runtime was measured. The following are ESTIMATED and carry wide
uncertainty because the student's hardware is UNKNOWN.**

| Phase | Estimate | Confidence | Main uncertainty |
|---|---|---|---|
| Model downloads (first run only) | 30–90 min | Low | network bandwidth |
| Model load per session | 2–10 min | Low | disk speed, quantisation |
| Index build | UNKNOWN | — | final corpus size |
| Retrieval + rerank, 100 questions | minutes | Medium | k per question |
| Filter, 100 questions × k | minutes | Medium | k per question |
| Generation, 2 LLM calls × 100 questions | **the dominant term; UNKNOWN** | — | tokens/sec on unknown hardware |
| **Baseline, 100 questions, end to end** | **2–6 hours on a single modern GPU** | **Low** | generation speed |
| **Proposed, 100 questions, end to end** | **similar to baseline; the intervention is arithmetic, not a model** | **Low** | same |

The proposed solution/system adds **no model call** over the baseline — it
replaces a Flan-T5 forward pass with a scored threshold, so its per-question
cost should be **lower**, not higher. This is worth stating in the thesis.

Disk: ≈ 20 GB for model caches (ESTIMATED) plus the corpus and index (UNKNOWN).
Output size for 600 answers with full metadata: a few MB (ESTIMATED) — JSONL
output is not a storage concern.

---

## 14–17. Configurations

| Component | Minimum feasible | Recommended | Paper-faithful | Feasible on student hardware? | Est. runtime | Blocker? |
|---|---|---|---|---|---|---|
| Generator | Llama-3-8B-Instruct, 4-bit, ~8 GB VRAM | 8B BF16, 24 GB VRAM | Llama-3-8B-Instruct BF16 + other paper models | **UNKNOWN** | dominant | **Yes, if no GPU** |
| MedCPT query encoder | CPU | any GPU | as published | Likely yes | trivial | No |
| Retrieval / index | FAISS CPU | FAISS CPU, ample RAM | 4 corpora, balanced | **UNKNOWN** (corpus size) | UNKNOWN | Possible |
| MedCPT reranker | CPU (slow) | any GPU | as published | Likely yes | minutes | No |
| Flan-T5 filter — inference | ~3 GB VRAM or CPU | any GPU | released checkpoint | Likely yes | minutes | No |
| Flan-T5 filter — **training** | LoRA, ~12–16 GB VRAM | 24 GB VRAM | full FT on full label set | **UNKNOWN** | hours–days | **Yes, likely** |
| Baseline pipeline | above combined | — | not reproducible (no corpora, no checkpoint) | UNKNOWN | 2–6 h | see above |
| Proposed pipeline | same minus filter training | — | n/a (novel) | UNKNOWN | ≤ baseline | No |

**Paper-faithful configuration is not attainable** — the corpora are not
distributable and the checkpoint does not exist. This is a property of the
upstream release, not a shortcoming of the thesis.

---

## 18. Scientific risks and confounders

### 18.1 Abstention asymmetry — highest priority

The proposed solution/system abstains when nothing clears `θ`; the baseline
never abstains. An abstention makes no claims and therefore cannot hallucinate.
**A threshold set high enough produces HAR = 0 and answers nothing.** Mitigation:
report HAR conditional on answering *and* with abstentions counted as failures,
always with the abstention rate. Never report a bare HAR.

### 18.2 Prompt-parity drift
Each arm accepts a `context_prompt` override with no cross-arm check.
Mitigation: hash the prompt template per arm and assert equality at run start.

### 18.3 Filter fidelity
A student-trained filter is not the paper's filter. If it is weaker, the
baseline is unfairly weak and the intervention looks better than it is.
Mitigation: report the filter's own validation performance and treat the
no-filter control as the honest floor.

### 18.4 Reference-answer leakage
If the passage used to establish a reference answer also enters the candidate
set, the evaluation is circular. Mitigation: the provenance firewall in Part 12,
enforced as a machine check, not a convention.

### 18.5 Corpus recency skew
See §6.6. Report the corpus date distribution alongside results.

### 18.6 Annotator blinding
Arm identity must be hidden and order randomised, with the mapping stored
separately. Without it the primary result is not defensible.

### 18.7 Generator knowledge cutoff
The generator may answer correctly from parametric memory without using the
evidence. This is not hallucination but it compresses the difference between
arms. Record the generator's cutoff relative to the reference time and treat it
as a documented condition.

---

## 19. Immediate corpus-independent work

Everything here can be built and tested with synthetic fixtures:

1. Evaluation-question schema, validation, provenance, duplicate and
   near-duplicate detection, stable IDs, human-review export.
2. Frozen evidence manifest, candidate-set hashing, cross-arm hash equality
   check that **fails** the run on mismatch.
3. Configuration hashing and run manifests.
4. Annotation schema, blinding and randomisation, agreement statistics.
5. Result serialisation to JSONL, append-only.
6. Statistical analysis: McNemar exact, paired bootstrap CI, subtype
   distributions, error-analysis cross-tabulations.
7. Baseline and proposed runners over a supplied candidate set.
8. A concrete generator behind the existing interface.
9. Engineering smoke test over synthetic fixtures.
10. Unit tests for all of the above.

## 20. Corpus-dependent blockers

* Final retrieval and candidate-set construction.
* Corpus-scale indexing and its RAM profile.
* Final evidence freezing against real passages.
* Linking each approved question to corpus evidence.
* The end-to-end experiment.
* Filter training (blocked on hardware, not on the corpus).

---

## 21. Exact next steps

**Run these on the student's own machine and record the output.** Until then,
sections 7–13 remain estimates.

```bash
python -c "import platform;print(platform.platform());import sys;print(sys.version)"
nvidia-smi                       # GPU model, VRAM, driver
nvcc --version                   # CUDA toolkit
free -h                          # system RAM
df -h .                          # free disk
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then, in order:

1. Record the hardware above; decide BF16 versus 4-bit from the measured VRAM.
2. Verify Hugging Face access to Llama-3-8B-Instruct.
3. Read `classifier/` in the RAG² repository and record the filter's real
   training hyperparameters — do not assume them.
4. Build the corpus-independent infrastructure in §19.
5. Run the engineering smoke test on synthetic fixtures.
6. Resume the corpus-dependent path once the corpus completes.

---

*Figures labelled ESTIMATED are derived from published model specifications,
not from execution. No model was run during this audit.*
