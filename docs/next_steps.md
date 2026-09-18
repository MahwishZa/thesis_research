# Next Actions

**Updated:** 2026-09-18 · 395 tests passing.

**PMC retrieval is now EXECUTED** — see "PMC finalization" below. Nothing
else in this repository is EXECUTED.

Status vocabulary, used strictly: **IMPLEMENTED** (code exists) · **TESTED**
(tested on fixtures) · **VALIDATED** (whole pathway exercised end to end) ·
**READY FOR REAL EXECUTION** (only inputs missing) · **EXECUTED** (actually
run).

---

## Research question

> Does the proposed solution/system reduce the rate of hallucinated answers in
> Alzheimer's disease question answering, relative to the baseline system,
> under identical question and evidence conditions, while maintaining
> comparable QA accuracy?

Primary: hallucination rate. Secondary: QA accuracy. No third objective.
`docs/frozen_scope.md` governs.

---

## Decisions taken (so the student does not have to)

| # | Decision | Where |
|---|---|---|
| D-36 | Research question is HAR + QA accuracy; admission asymmetry superseded | `frozen_scope.md` |
| D-37 | MedCPT retrieval + reranking over a flat **exact** index (not FAISS approximate — determinism) | `experiments/retrieval/` |
| D-38 | Generator stays **Llama-3-8B-Instruct**, 4-bit NF4, on a free remote T4. Local GPU is ruled out at every precision | `generator_contract.md` |
| D-39 | Filter trained by us: **Flan-T5-large**, paper's recipe, effective batch preserved by accumulation | `filter_training.md` |
| D-40 | Training set is a subsample; the label *function* stays exact, the weaker filter is a stated limitation | `filter_training.md` |
| — | Context budget 5, candidate-set size 20, retrieval depth 50 | `system_specification.md` §8 |

λ, θ and H remain **deliberately unfitted** — `validate()` raises rather than
default them. They are fitted on the validation split, never on test.

## Step status

| Step | Status | Waiting on |
|---|---|---|
| 1. Corpus — PMC source | **EXECUTED** (retrieval + finalization) | normalize/dedup/chunk still to run over it |
| 1. Corpus — other sources, normalize/dedup/chunk | IN PROGRESS | student's build |
| 2. Questions | IN PROGRESS | human review of 123 candidates |
| 3. Freeze evidence | READY FOR REAL EXECUTION | corpus + approved questions |
| 3a. Retrieval/rerank | IMPLEMENTED, TESTED | corpus; `torch` on the run machine |
| 4. Baseline | READY, except the filter checkpoint | filter training |
| 5. Proposed | READY | λ/θ/H fitting (needs validation split) |
| 6. Collect results | READY | steps 4–5 |
| 7. Annotation | IMPLEMENTED, TESTED | generated answers |
| 8. HAR | IMPLEMENTED, TESTED | annotations |
| 9. QA accuracy | IMPLEMENTED, TESTED | a correctness-judging protocol applied to real answers |
| 10. Statistics | IMPLEMENTED, TESTED | real paired outcomes |
| 11. Error analysis | IMPLEMENTED, TESTED | real outcomes |
| 12. Write-up | methods skeleton only, no results | results |

**The whole pathway is VALIDATED end to end on synthetic fixtures**
(`tests/integration/test_controlled_validation.py`): corpus → index →
retrieval → rerank → freeze → both arms → runner → JSONL, with all thirteen
control properties asserted. That is software validation, not a pilot, and it
produces no number that could be read as a result.

## PMC finalization — EXECUTED 2026-09-18

**Fixed a producer/consumer log-evidence mismatch** (ledger D-41): the
finalizer's reader required the newest of three message wordings the script
had used across retrieval runs, so it recovered 0 of 102 legitimately
unavailable PMCIDs from `retrieval.log` and refused to write a manifest on
every attempt. The fix widened the reader to match the core statement common
to all three wordings; no corpus file was touched.

The student re-ran `--finalize` after pulling the fix. Independently verified
against the committed `metadata/pmc.csv` (114,256 rows) and `logs/retrieval.log`:

| Check | Result |
|---|---|
| Row count | 114,256 (predicted from repair evidence before the run) |
| Duplicate (pmcid, version) pairs | 0 |
| `already_verified` rows | 114,157 |
| `unavailable_current_dataset` rows | 99 |
| Rows missing a required path for their status | 0 |
| xml_path count | 114,157 — matches the independently-reported local XML file count |
| json_path count | 114,850 — matches the independently-reported local JSON file count |
| Reconciliation: 114,859 local dirs − 9 stale − 693 non-OA | 114,157 (exact) |

All three cross-checks — predicted row count, XML file count, JSON file
count — closed exactly against numbers reported independently of the
manifest itself. **This is the first EXECUTED artifact in the repository.**

**What this does not mean:** Step 1 as a whole is not complete. PubMed
(676 records), guidelines (0) and textbooks (0) metadata are unchanged, and
`data/normalized/`, `data/deduplicated/` and `data/chunks/` still hold only
the 10-record synthetic fixture — the normalize → deduplicate → chunk stages
have not run over the real PMC data yet.

---

## Blockers

| Blocked on | Blocks | Note |
|---|---|---|
| **Corpus completion** | 3 onward | student's machine |
| **Human review** | the final ~100 questions | 123 candidates in `experiments/questions/review.csv` |
| **Filter training** | the baseline arm | strategy decided; needs one free-tier GPU session |
| **Llama-3 licence** | generation | a click-through on Hugging Face, then a read token |
| **Generation speed** | run planning | **unmeasured**; the timing check produces it |
| **Identifier verification** | question approval | PMIDs transcribed, not resolved |

## NEXT — in order

**1. Finish the corpus.** PMC retrieval is done; PubMed/guidelines/textbooks
and the normalize → deduplicate → chunk stages remain. Do not restart PMC.
When the whole corpus completes, say so — the retrieval stage then indexes
it.

**2. Finish question review.** ACCEPT / REVISE / REJECT / HOLD per
`docs/question_review.md`. Spot-check a sample of PMIDs first. Target ~100
accepted, not a fixed count.

**3. Accept the Llama-3 licence** on Hugging Face and create a read token.

Steps 4 onward are assistant work once 1–3 land:

4. Build the index (`python -m experiments.retrieval.build_index`), recording
   the corpus snapshot id.
5. Timing check on the remote GPU — the first real measurement.
6. Generate filter labels on a subsample; check the label distribution; train
   Flan-T5-large; record validation accuracy in a `CheckpointRecord`.
7. Freeze evidence for the approved questions.
8. Fit λ, θ, H on the **validation split only**, then freeze them.
9. Run both arms over the frozen test manifest with one shared generator.
10. Blind annotation → HAR → QA accuracy → paired statistics → error analysis.

Do not add a pilot study, extra metrics, or extra baselines.

## Standing limitations for the thesis

1. The baseline is an **adaptation** of RAG², not a reproduction — the
   checkpoint is not distributed and the released training file is a 5-example
   format sample.
2. The filter is trained on a **subsample**, so it is weaker than the paper's.
   The no-filter control is the floor that keeps this visible.
3. The generator runs **4-bit quantised**, identically for both arms.
4. ~100 questions is a **practical budget, not a powered sample size**.
5. Question sources are ~85% Cochrane; a guideline source would strengthen it.
6. Identifiers were transcribed from an inspected dataset, not resolved.
7. Execution hardware differs from the paper's — a resource limitation, not a
   methodological one.
8. QA accuracy correctness is judged (human or a named rule), not computed by
   an automatic scorer.
