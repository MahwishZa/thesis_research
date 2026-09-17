# Next Actions

**Updated:** 2026-09-17 · 229 tests passing.

---

## DONE

* **Local environment verified.** RTX 2050 4 GB, driver 592.82, PyTorch
  2.4.1+cu121, `torch.cuda.is_available() == True`. The driver-reported CUDA
  13.1 versus PyTorch's 12.1 is **not a fault** — drivers are backward
  compatible and CUDA initialises. No reinstall.
* **Repository consolidated** to one top-level `experiments/`; `evaluation/`
  moved to `experiments/evaluation/` with `git mv`. No stale references.
* **Question-source protocol established** *before* sourcing
  (`docs/QUESTION_SOURCE_AND_PROVENANCE_PROTOCOL.md`).
* **Candidate pool built from two inspected sources** — 170 sourced, 150 after
  id-collision removal, **123 auto-validated awaiting human review**, 27
  auto-rejected as near-duplicates. Every record carries a concrete locator
  (PMID/DOI/CD, or MedQuAD id + CUI + URL) and a date.
* **Flan-T5 base-size ambiguity resolved as far as the repository allows:**
  `classifier/model/token_add.ipynb` has *empty* `from_pretrained("")` strings,
  so the released code pins no base model. Only the paper is authoritative;
  ledger E1 records Flan-T5-large (770 M) read from it.
* **Abstention policy** resolved and configurable (previous round).
* Nothing downloaded; corpus untouched.

## IN PROGRESS

* **Step 1 — corpus download**, running on the laptop.
* **Step 2 — question pool.** 123 candidates await human review.

## BLOCKED

| Blocked on | What it blocks | Note |
|---|---|---|
| **Generator venue decision** | Steps 4–6 | Llama-3-8B: BF16 needs ≈16 GB; 4-bit ≈4.5–5 GB before KV cache, against 4 GB VRAM. CPU/offload may work; **runtime unmeasured** |
| **Filter training venue** | the baseline arm | ≈12.4 GB before activations; a free 16 GB session suffices. Inference stays local |
| **Corpus completion** | corpus-support check, Step 3 onward | |
| **Retrieval + reranking** | Step 3 | not implemented; `retrieval_external: True` |
| **Human review** | the final ~100 questions | 123 candidates ready in `pool/review.csv` |
| **Identifier verification** | question approval | PubMed/doi.org blocked in the build environment; PMIDs transcribed, not resolved |

## NEXT — in order

**1. Keep the corpus downloading.** Do not restart it.

**2. Review the candidate pool.** Open
`experiments/question_sources/pool/review.csv` (123 rows). For each: confirm
the reference answer actually answers the question and matches the cited
record, then set `reviewer_decision` to accept / reject / revise / verify.
Spot-check a sample of PMIDs first — they were transcribed, not resolved.
Target ~100 accepted.

**3. Decide the generator venue.** The question is not "can Llama-3-8B run"
but "can it run reproducibly, twice, in a sensible time". Either measure a
short local generation, or commit to a remote GPU. **Download nothing until
this is decided** — 30.07 GB free with the corpus still growing.

**4. Decide the filter training venue** and base size — see
`RAG2_CLASSIFIER_FEASIBILITY.md` §2D.

**5. Only then download** the models the decision actually requires.

**6. When the corpus finishes**, run the corpus-support check against the
approved questions. `corpus_support_expected` is currently an *expectation*;
this is where it becomes a finding.

**7. Freeze evidence** — `experiments/evaluation/freezing.py` hashes the
candidate set and gates cross-arm equality.

**8. Run baseline, then the proposed solution/system**, same frozen manifest.

**9. Blind annotation → HAR → QA accuracy → paired comparison → error
analysis.**

Do not add a pilot study, extra metrics, or extra baselines.

## Standing limitations for the thesis

1. The baseline is an **adaptation** of RAG², not a reproduction.
2. ~100 questions is a **practical budget, not a powered sample size**.
3. Question sources are ~85% Cochrane; a guideline source would strengthen it.
4. Identifiers were transcribed from an inspected dataset, not resolved.
5. Execution hardware differs from the paper's — a resource limitation, not a
   methodological one.
