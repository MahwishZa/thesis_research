# Next Actions

**Updated:** 2026-09-17 · 282 tests passing.

---

## DONE

* **Local environment verified.** RTX 2050 4 GB, driver 592.82, PyTorch
  2.4.1+cu121, `torch.cuda.is_available() == True`. The driver-reported CUDA
  13.1 versus PyTorch's 12.1 is **not a fault** — drivers are backward
  compatible and CUDA initialises. No reinstall.
* **Repository consolidated** to one top-level `experiments/`; `evaluation/`
  moved to `experiments/evaluation/` with `git mv`. No stale references.
* **Question-source protocol established** *before* sourcing
  (`docs/question_sources.md`).
* **Candidate pool built from two inspected sources** — 170 sourced, 150 after
  id-collision removal, **123 auto-validated, currently under review by one
  medically trained reviewer**, 27 auto-rejected as near-duplicates. Every
  record carries a concrete locator (PMID/DOI/CD, or MedQuAD id + CUI + URL)
  and a date.
* **Flan-T5 base-size ambiguity resolved as far as the repository allows:**
  `classifier/model/token_add.ipynb` has *empty* `from_pretrained("")` strings,
  so the released code pins no base model. Only the paper is authoritative;
  ledger E1 records Flan-T5-large (770 M) read from it.
* **Abstention policy** resolved and configurable.
* **Steps 3–12 infrastructure complete and tested on synthetic fixtures**
  (`docs/methodology.md`, `docs/repository_structure.md`'s "Steps 3–12"
  section): evidence freezing with corpus-version tracking, a real
  baseline+proposed system run together end to end, the full results/
  annotation/QA-accuracy/statistics/error-analysis pipeline. See the summary
  table below for exactly what remains blocked within each step.
* Nothing downloaded; corpus untouched; no final or result-generating run
  performed.

## IN PROGRESS

* **Step 1 — corpus download**, running on the laptop.
* **Step 2 — question pool.** 123 candidates under review by one medically
  trained reviewer.

## Steps 3–12: what's ready vs. what's blocked

| Step | Infrastructure | Blocked on |
|---|---|---|
| 3. Freeze evidence | `freezing.py` — hashing, parity gate, firewall, `from_question()`, `corpus_snapshot` tracking. Tested on fixtures. | Real candidate evidence (needs corpus + retrieval) |
| 4/5. Run baseline / proposed | Real `RAG2System` + `RecencyAwareSystem` run together via `run_experiment()`, tested end to end (`tests/integration/test_baseline_and_proposed.py`). | A concrete generator (venue decision, below); retrieval/reranking to produce real candidates |
| 6. Collect results | Full JSONL schema; `read_results()` / `group_by_system()` for question-by-question comparison. | Same as 4/5 |
| 7. Annotation | Schema, blinding, `read_annotations()`, `unblind_annotations()`. Tested on fixtures. | Generated answers to annotate |
| 8. HAR | `har()`, `coverage()`, `compare_systems()`, `hallucination_outcomes()` glue. Tested with known expected values. | Real annotations |
| 9. QA accuracy | `accuracy.py` (`QAJudgment`), `qa_accuracy()`. Tested. Correctness is judged, not auto-scored — see `docs/methodology.md`. | A correctness-judging protocol (human or rule) applied to real answers |
| 10. Statistics | `mcnemar()`, `paired_bootstrap_ci()`, `holm()` — same functions serve both HAR and accuracy. Tested with known expected values. | Real paired outcomes |
| 11. Error analysis | `outcome_crosstab()`, `error_analysis()`. Tested. | Real outcomes to analyse |
| 12. Thesis prep | `docs/methodology.md` — methods skeleton, cross-referenced, no results. | Results themselves |

## BLOCKED

| Blocked on | What it blocks | Note |
|---|---|---|
| **Generator venue decision** | Steps 4–6 | Llama-3-8B: BF16 needs ≈16 GB; 4-bit ≈4.5–5 GB before KV cache, against 4 GB VRAM. CPU/offload may work; **runtime unmeasured** |
| **Filter training venue** | the baseline arm | ≈12.4 GB before activations; a free 16 GB session suffices. Inference stays local |
| **Corpus completion** | corpus-support check, Step 3 onward | |
| **Retrieval + reranking** | Step 3 | not implemented; `retrieval_external: True` |
| **Human review** | the final ~100 questions | 123 candidates under review in `experiments/questions/review.csv` |
| **Identifier verification** | question approval | PubMed/doi.org blocked in the build environment; PMIDs transcribed, not resolved |

## NEXT — in order

**1. Keep the corpus downloading.** Do not restart it.

**2. Let review of the candidate pool finish.** `experiments/questions/
review.csv` (123 rows), decisions ACCEPT / REVISE / REJECT / HOLD per
`docs/question_review.md`. Spot-check a sample of PMIDs first — they were
transcribed, not resolved. Target ~100 accepted, not a fixed count.

**3. Decide the generator venue.** The question is not "can Llama-3-8B run"
but "can it run reproducibly, twice, in a sensible time". Either measure a
short local generation, or commit to a remote GPU. **Download nothing until
this is decided** — 30.07 GB free with the corpus still growing.

**4. Decide the filter training venue** and base size — see
`rag2_classifier_feasibility.md` §2D.

**5. Only then download** the models the decision actually requires.

**6. When the corpus finishes**, run the corpus-support check against the
approved questions. `corpus_support_expected` is currently an *expectation*;
this is where it becomes a finding.

**7. Freeze evidence** — `from_question()` builds each `FrozenItem` from an
approved question plus its retrieved candidates and the corpus snapshot id;
`write_manifest()` hashes and gates cross-arm equality.

**8. Run baseline, then the proposed solution/system**, same frozen manifest,
via `run_experiment()` (already proven end to end on fixtures).

**9. Blind annotation → HAR → QA accuracy → paired comparison → error
analysis.** Every step in this chain already has tested code; what's missing
is real data to run it on.

Do not add a pilot study, extra metrics, or extra baselines.

## Standing limitations for the thesis

1. The baseline is an **adaptation** of RAG², not a reproduction.
2. ~100 questions is a **practical budget, not a powered sample size**.
3. Question sources are ~85% Cochrane; a guideline source would strengthen it.
4. Identifiers were transcribed from an inspected dataset, not resolved.
5. Execution hardware differs from the paper's — a resource limitation, not a
   methodological one.
6. QA accuracy correctness is judged (human or a named rule), not computed by
   an automatic scorer — see `docs/methodology.md`.
