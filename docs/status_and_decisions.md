# Status, Decisions and Record

**Authoritative status and decision record. Consolidated 2026-09-19** from
`research_ledger.md`, `next_steps.md`, `hardware_and_resources.md`,
`question_pool_status.md`, `repository_structure.md`,
`experiment_outputs.md`, `feasibility_and_alignment_audit.md`,
`research_understanding.md` and `proposal_scope_amendment.md`.

Scope is governed by `docs/current_objectives.md`. The method is specified in
`docs/research_experimental_specification.md`. **This document records what is
true right now, what was decided and why, and what is still missing.**

Status vocabulary, used strictly: **IMPLEMENTED** (code exists) · **TESTED**
(tested on fixtures) · **VALIDATED** (whole pathway exercised end to end) ·
**READY FOR REAL EXECUTION** (only inputs missing) · **EXECUTED** (actually
run on real data).

Evidence tags: `[DOC]` demonstrated by the provided documents · `[PUB]`
reported by published research · `[INF]` methodological inference · `[STU]`
student assumption · `[REC]` assistant recommendation.

**Terminology note (2026-09-20).** The proposed method is now consistently
called the **Temporal Filter**, and its signal the **temporal score**. Older
decisions and facts below (D-17, D-25, D-26, D-28, E15, K5) were written when
the code called this "recency" — they are left in their original wording as
an accurate record of what was decided *then*, per this document's own rule
that history is not rewritten. Nothing active uses "recency" any more; see
`docs/current_objectives.md` and `_archive/README.md`.

---

## 1. Headline status

| | |
|---|---|
| **Alzheimer's corpus** | **COMPLETE / FROZEN** (verified 2026-09-19). Do not rerun stages 01–07. |
| **Evaluation questions** | 123 candidates, **under human review**. 0 approved. |
| **RAG² baseline filter** | Code complete and reachable; **no trained checkpoint exists**. |
| **Proposed system** | Code complete; **λ, θ, H unfitted** by design. |
| **Generator** | Contract pinned and wired; **real generation confirmed working** with a small model (see §1.1) — Llama-3-8B-Instruct itself has not been downloaded or run. |
| **Metrics / runner / ablation** | IMPLEMENTED, TESTED, VALIDATED on fixtures; **confirmed consuming real (non-mock) generation output** (§1.1). |
| **Experimental result** | **None exists.** No number in this repository is a research finding. |

**No real (non-fixture, non-thesis-model) experimental run has been
performed.** The whole pathway is VALIDATED end to end on synthetic fixtures
(`tests/integration/test_controlled_validation.py`): corpus → index →
retrieval → rerank → freeze → every arm → runner → JSONL, with all control
properties asserted. §1.1 additionally confirms the same pathway runs
cleanly with a real, non-mock generator producing real text — the one link
software validation alone cannot exercise. Neither is a pilot, and neither
produces a number that could be read as a result.

### 1.1 Real-generator smoke test — CONFIRMED 2026-09-19

The one component the fixture-based validation in §1 cannot exercise — an
actual language model loading, receiving the runner's built context, and
generating real text that flows into a real evaluation record — was
confirmed working end to end on the student's own machine, using a small
stand-in model in place of the thesis generator (Llama-3-8B-Instruct itself
needs a 16 GB GPU the student's laptop does not have — see
`research_experimental_specification.md` §9).

**Command run** (`--real-model` with a small, ungated, architecturally
identical stand-in):

```powershell
python -m experiments.runners.run_end_to_end `
  --output-dir experiments\outputs\smoke_test_NOT_FINAL `
  --n-questions 4 --ablation-lambdas 0,1.0 `
  --real-model --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
  --model-revision fe8a4ea1ffedaf415f4da2f062534de366a451e6 `
  --quantization nf4
```

**Result:** `Ran 4 arms over 4 questions (0 errors)`. All 16 records `status:
ok`, non-empty `generated_answer` (e.g. `"The current value for question 0
is 0."`), correct `model: hf:TinyLlama/TinyLlama-1.1B-Chat-v1.0` stamped on
every record. Metrics computed correctly on real output: on this fixture,
gold evidence ids are present (unlike the real-data case — see
`rag_metrics.py`'s `None`-vs-`0.0` distinction in the specification §11),
so `context_precision/recall = 0.000` for the three arms that admitted the
wrong passage and `1.000` for the arm that admitted the right one — real
measured zeros, not the `n/a` case. `main_evaluation`/`ablation_study`
verdicts were produced from real generation for the first time, though they
remain fixture output on a stand-in model and are not a thesis result.

**Environment defect found and fixed by the student, not this repository:**
the student's environment had drifted to `transformers` 5.17.0, which
refuses to load PyTorch below 2.5 and silently disables all model loading
(`AutoModelForCausalLM` becomes undefined) rather than raising a clear
error at the point of use — every one of an initial 16 generations failed
with `NameError: name 'torch' is not defined` before any model code ran.
Pinning `transformers==4.51.3` (compatible with the student's PyTorch
2.4.1+cu121, and still above this repository's declared floor of
`transformers>=4.40` in `pyproject.toml`) resolved it. No repository code
was at fault and none was changed for this.

**What this confirms:** `HuggingFaceGenerator` (`systems/interfaces/
hf_generator.py`) is a real, working implementation — not merely code that
type-checks — for loading a causal LM, applying its chat template,
quantizing to NF4, generating greedily, and returning real text through the
exact same code path Llama-3-8B-Instruct will use. Retrieval → admission
(all three arms) → context construction → generation → record → metrics →
report is now confirmed working with genuine model output, on top of the
software-only validation in §1.

**What this does NOT confirm:** nothing about Llama-3-8B-Instruct's own
memory footprint, load time, generation speed, or output quality — a 1.1B
model's resource behaviour does not transfer. The timing check specified
in `research_experimental_specification.md` §9.3 step 4 (on the actual
target model, on the actual T4 venue) remains unperformed. TinyLlama does
not appear in, and must never be cited as, a thesis result.

## 2. Corpus stage — COMPLETE / FROZEN

All seven pipeline stages have run for real against the live corpus. The
evidence chain below was verified end to end by cross-checking each stage's
*output count* against the *next* stage's *input count* as independently
reported by that next stage's own log line — not by trusting any stage's
self-report in isolation:

```
PMC finalize:   matched_pmcids=112260 = accounted_pmcids=112260, manifest_rows=114256  (exact)
04 normalize:   PMC verified=114157 = parsed=114157 = normalized=114157                (exact)
05 deduplicate: input 114157, duplicates=2842, unique=111315   (114157 − 2842)         (exact)
06 chunk:       documents=111315 (matches 05 exactly)  →  chunks=4377041               (exact)
07 classify:    chunks=4377041 (matches 06 exactly)    →  all 4377041 tagged           (exact)
```

| Stage | Result |
|---|---|
| 01 PubMed | **EXECUTED** — 676 records |
| 02 PMC retrieval + finalization | **EXECUTED** — 114,256-row manifest, 0 duplicate (pmcid, version) pairs, 114,157 `already_verified` + 99 `unavailable_current_dataset`, every row's paths consistent with its status |
| 03 guidelines / textbooks | IMPLEMENTED, TESTED, **0 rows curated** — optional, not blocking |
| 04 normalize | **EXECUTED** — 114,157 normalized (67,608 AD-relevant, 46,549 excluded) |
| 05 deduplicate | **EXECUTED** — 111,315 unique, 2,842 duplicates recorded |
| 06 chunk | **EXECUTED** — 4,377,041 chunks, real `ncbi/MedCPT-Article-Encoder` tokenizer, 256/32/224 spec, ~7,892 s (≈2 h 11 m) |
| 07 claim classification | **EXECUTED** — 4,377,041 chunks tagged, all 26 `claim_types` and all 17 `evidence_levels` matched at least once, ~4,520 s at ~975 chunks/sec |

Also checked and found clean: every tracked CSV (`metadata/*.csv`,
`reports/*.csv`) parses as well-formed CSV with a consistent column count on
every row (verified programmatically); `quality_control.log` and
`deduplication.log` contain no `ERROR`, `CRITICAL` or traceback lines;
`retrieval.log`'s 227 `ERROR` lines are all transient retries from the
multi-day retrieval campaign, all preceding its own final authoritative
`Stage 02 finalization completed successfully` line with zero reconciliation
discrepancy — resolved history, not a live problem.
`tests/integration/test_corpus_to_retrieval.py` runs the real 04→05→06→07
chain and confirms `experiments/retrieval/corpus.py::read_passages()` loads
the result, every passage carries the fields retrieval and admission need,
chunk_ids are unique, and Stage 07's tags survive into what retrieval hands
onward — proof, not assumption, that the corpus schema is what the retrieval
layer expects.

**Conclusion: no genuine integrity problem was found; nothing was corrected,
rerun or regenerated.**

**The real intermediate JSONL files exist only on the machine that produced
them.** `alzheimer_corpus/data/**` is gitignored by design ("research data is
never committed"); only the logs, reports and registries above are tracked
provenance.

### 2.1 Two known, non-blocking corpus gaps

* **Stage 03 registries are header-only (0 rows).** `03_guidelines.py` is
  implemented and tested but no document has been curated into it. Stage 04's
  real-data path treats both registries as optional, so nothing upstream is
  affected. Real open-access guidance exists (US government work is public
  domain by statute), but this environment's egress blocks `who.int` and
  `nia.nih.gov`, so no specific title, URL or licence could be independently
  verified — and fabricating one was prohibited from the start. Populating a
  row remains a per-document curation step for the student (D-45).
* **The PMC licensing gate is not enforced.** `04_normalize.py` stamps
  `redistribution_allowed` on every PMC record but does not act on it — unlike
  the guidelines/textbooks path, which correctly excludes a restricted row's
  text. On the finalized manifest roughly **19,669 of 114,256 rows (~17%)**
  carry a licence outside `_common.DISTRIBUTABLE` (`CC BY-NC-ND`, `TDM`, blank
  or missing) and would be affected if the gate were enforced. Stage 04 has
  already run, so this affects real generated output, not a future run. It is
  a **dataset-composition decision**, not a bug fix to apply silently: it
  changes which real documents' text is in the corpus. It affects
  redistributability, not internal use — the corpus JSONL is gitignored and
  never leaves the machine — so it does not block research use now, but it
  must be decided before any external publication or redistribution of the
  corpus text, and Stage 04 re-run if the gate is enforced.

## 3. Component readiness

| Component | Status | Waiting on |
|---|---|---|
| Retrieval / rerank (`experiments/retrieval/`) | IMPLEMENTED, TESTED | `torch` on the run machine |
| Evidence freezing (`freezing.py`) | READY FOR REAL EXECUTION | approved questions |
| Baseline arm (`systems/baseline/`) | READY except the checkpoint | filter training |
| Proposed arm (`systems/proposed/`) | READY | λ/θ/H fitting on a validation split |
| Generator interface (`hf_generator.py`) | IMPLEMENTED, TESTED, **confirmed loading and generating with a real model** (§1.1) | the Llama-3-8B licence + a pinned sha + a GPU that fits it |
| Standard metrics (`rag_metrics.py`) | IMPLEMENTED, TESTED, **confirmed scoring real generation output** (§1.1) | approved questions with gold annotation |
| End-to-end runner (`run_end_to_end.py`) | IMPLEMENTED, TESTED, VALIDATED, **confirmed end-to-end with a real generator** (§1.1) | Llama-3-8B-Instruct itself + the remaining blockers below |
| Ablation (`λ = 0` arm, always included) | IMPLEMENTED, TESTED | a fitted full-system λ to ablate against |
| Annotation / HAR / statistics | IMPLEMENTED, TESTED — out of the critical path | real outcomes |

### 3.1 Blockers to a real (non-fixture) run

| Blocked on | Blocks | Note |
|---|---|---|
| ~~**Human question review**~~ | ~~the final ~100 questions~~ | **DONE (2026-09-20).** All 123 decided: 90 ACCEPT, 23 REVISE, 10 REJECT — see §3.3. |
| ~~**Validation/test split**~~ | ~~parameter fitting vs. final evaluation~~ | **DONE (2026-09-21).** `experiments/questions/splits.json` — see §3.3. |
| **Filter training** | a *trained* RAG² baseline | strategy decided; needs one free-tier GPU session |
| **Llama-3 licence** | generation | a click-through on Hugging Face, then a read token |
| **λ/θ/H fitting** | the proposed arm's real configuration | validation split ready (§3.3); still needs a trained RAG² checkpoint and a GPU session to actually fit on |
| **23 REVISE questions still need their wording/answer corrected** | the validation and test splits both contain some | the underlying claim is sound (reviewer's own judgement); the wording fix itself has not been done — see §3.3 |
| **Generation speed** | run planning | **unmeasured**; the timing check produces it |
| **Identifier verification** | question approval | PMIDs transcribed, not resolved |
| **PMC licensing gate** | corpus composition (§2.1) | decision, not a bug fix |

None of these blocks running, validating and ablating the system against the
fixture. They block treating any number as a real result.

### 3.2 Next steps, in order

1. ~~Finish question review.~~ **DONE (2026-09-20)** — see §3.3.
2. ~~Split the reviewed questions into validation and test.~~ **DONE
   (2026-09-21)** — see §3.3.
3. **Accept the Llama-3 licence** on Hugging Face; create a read token.
4. ~~Measure the real corpus's null publication-date rate.~~ **DONE
   (2026-09-21).** Read-only pass over the real, local
   `alzheimer_corpus/data/chunks/chunks.jsonl` (4,377,041 chunks, the
   student's own machine — never available in this environment; see §1):
   **100.00% valid `publication_date` (4,377,041 / 4,377,041), 0 missing,
   0 invalid.** `dated_only()`'s undated-passage drop (specification §15.3)
   removes nothing — every chunk PMC's JATS XML dates deterministically
   resolved, and Stage 06 carried that date onto every chunk it produced.
   This closes the one open question the 2026-09-20 audit flagged before
   fitting `H`: candidate-set size at N=20 is not threatened by missing
   dates. No corpus action was needed or taken.
5. Build the index (`python -m experiments.retrieval.build_index`), recording
   the corpus snapshot id.
6. Timing check on the remote GPU — the first real measurement.
7. Generate filter labels on a subsample; check the label distribution; train
   Flan-T5-large; record validation accuracy in a `CheckpointRecord`.
8. Freeze evidence for the split's questions (validation and test
   separately — never mix the two manifests).
9. Fit λ, θ, H on the **validation split only**, then freeze them.
10. Run every arm over the frozen **test** manifest with one shared generator
    (`run_end_to_end.py`, see the specification §17). Include an `H`-sensitivity
    sweep (2-3 `--half-life` values at the fitted λ) as the ablation study's
    second configuration — already CLI-supported, no new code.
11. Score with the standard metrics; report `main_evaluation` (objective 3),
    `ablation_study` (objective 2, both configurations), and the
    `temporal_subgroup` breakdown (specification §11.2) separately.

### 3.3 Question review and the validation/test split

**Question review — DONE, 2026-09-20.** The student reviewed all 123
candidates directly against `docs/question_review.md`'s ACCEPT / REVISE /
REJECT / HOLD taxonomy, with real source verification (something this
pipeline's own automated passes could never do — Cochrane Library and
PubMed are network-egress-blocked in this build environment). Recorded in
`experiments/questions/review.csv`: **90 ACCEPT, 23 REVISE, 10 REJECT, 0
HOLD.**

**Validation/test split — DONE, 2026-09-21.** Implemented as
`experiments/questions/split.py`
(`python -m experiments.questions.split`), output committed at
`experiments/questions/splits.json`.

Only two splits, not three: nothing in this thesis trains on the
123-question pool (the RAG² filter trains on general-medical MedQA,
specification §10; the generator is used off the shelf). The only thing
this pool feeds is the manual fit of `λ`, `θ`, `H`, which needs one held-out
set to fit on and a separate, untouched set to report the final result on —
a validation/test split, not train/validation/test. A third, unused
"training" partition would only shrink the two splits that matter, for no
scientific reason.

REJECT and HOLD questions never enter the split (0 HOLD currently, so only
the 10 REJECT are excluded). ACCEPT and REVISE both enter it — REVISE's own
definition in `question_review.md` is that the underlying claim is sound and
only wording needs fixing, not a REJECT-in-waiting — but every REVISE item
is flagged `pending_revision: true` in `splits.json`, and **that wording fix
has not been done yet**: it needs the actual cited source text, which
requires a human (or an environment with real network access) to pull the
corrected sentence. Freezing evidence (next-steps step 8) for a REVISE item
before its wording is fixed would freeze a known-defective reference answer.

Split: **113 usable questions → 23 validation / 90 test** (20%, stratified
by topic then by `temporal_candidate` within each topic, seed `20260921`,
fully reproducible — `write_split()` refuses to silently change an existing
`splits.json` if the reviewed pool has moved since). Both splits contain
every topic that has ≥2 usable questions, and both contain a comparable mix
of temporal and non-temporal questions (validation 52% temporal, test 60%)
so a `λ` fit on validation is not fit against an unrepresentative slice of
what test will contain. Full per-topic/per-temporal counts and the
question-by-question assignment are in `splits.json` itself.

`export_review.py` now refuses to overwrite a `review.csv` that already has
recorded decisions unless `--force` is passed — the export always emits
blank decision columns, and without this guard a careless re-run would have
silently erased the completed human review.

Do not add a pilot study, extra metrics, or extra baselines.

## 4. Question pool status

**Built 2026-09-17.** Reproduce with:

```bash
python -m experiments.questions.build_pool \
    --medrevqa <path>/MedRevQA.csv --medquad <path>/MedQuAD --retrieved-on 2026-09
```

> This table describes the pool as *built* (2026-09-17), before human review.
> **Human review is now complete — see §3.3 for the reviewed counts (90
> ACCEPT / 23 REVISE / 10 REJECT / 0 HOLD) and the validation/test split.**
> The figures below (e.g. "Approved / final: 0") are the pre-review
> snapshot and are kept for provenance, not updated in place.

| | |
|---|---|
| Sourced from adapters | 170 |
| After dropping identical-id repeats | 150 |
| **Auto-validated, awaiting human review** | **123** |
| Auto-rejected (all `near_duplicate_of_earlier_candidate`) | 27 |
| Near-duplicate pairs detected | 35 |
| Distinct source records | 133 |
| AD-anchored / determinate | 150 / 150 |
| Temporal candidates / ambiguity candidates | 74 / 47 |
| Missing provenance | 0 — the schema refuses a record without source, locator and date |
| **Approved / final (as built, pre-review)** | **0** |

Target is roughly 100 after review; 123 candidates gives room to reject weak
ones without dropping below it. Rejected records are kept in
`candidates.jsonl` with their reason, so counts reconcile. The actual
post-review outcome landed at 113 usable (90 ACCEPT + 23 REVISE), slightly
above that informal target — see §3.3.

**Topics:** treatment 79, diagnosis 17, management 17, prevention 9, general
7, disease_characteristics 6, mechanism 6, disease_course 5, epidemiology 4.
Skewed to treatment because Cochrane is intervention-heavy. **Not corrected**
— forcing balance would mean dropping sound items or inventing weak ones. A
reviewer wanting more diagnosis or epidemiology coverage should add a
guideline source rather than rebalance this one.

**Source types:** peer_reviewed_evidence_synthesis 128,
government_public_health 22.

### 4.1 Known limitations of the pool

1. **Identifiers were transcribed, not resolved.** PubMed E-utilities and
   doi.org are blocked in the build environment, so no PMID or DOI was
   confirmed to resolve. Every record carries `verification_required: true`.
   Spot-checking these is the first review task.
2. **Two sources, ~85% Cochrane.** A guideline source would strengthen it.
3. **MedQuAD items carry no publication date**; the retrieval date is recorded
   and flagged as such. Two of its AD source sites have been retired into
   MedlinePlus and their URLs may redirect.
4. **`corpus_support_expected` is an expectation, not a check.**
5. **Reference answers are one-sentence verbatim extracts** — short by design,
   for copyright and judgeability, but a reviewer should confirm each one
   *answers* its question rather than merely relating to it.

## 5. Hardware and execution venue

**Local environment VERIFIED 2026-09-17** (measured on the student's laptop):
64-bit Windows on AMD Ryzen 5 7535HS, 16 GB RAM (15.2 GB usable), NVIDIA RTX
2050 with **4096 MiB VRAM**, driver 592.82 (driver-reported CUDA 13.1),
Python 3.12.6, PyTorch 2.4.1+cu121 with `torch.cuda.is_available() == True`,
transformers/accelerate/huggingface_hub installed, 30.07 GB free on C:.

**The CUDA version mismatch is not a problem.** Driver-reported CUDA 13.1 is a
*ceiling*; PyTorch's build CUDA 12.1 is what is *used*; NVIDIA drivers are
backward compatible. `torch.cuda.is_available()` returns True and PyTorch
names the RTX 2050. **Do not reinstall CUDA or PyTorch because the numbers
differ** — it risks breaking a working stack and consuming scarce disk for no
gain.

All runtime figures below are **ESTIMATED**: no model has been executed on
either machine, and the audits themselves ran in a Linux container with no GPU
and no ML stack.

| Component | Verdict | Basis |
|---|---|---|
| MedCPT query encoder / reranker inference | **FEASIBLE LOCALLY** | ≈109 M params, ≈0.44 GB fp32; cost scales with candidates per question, not corpus size |
| Flan-T5 filter **inference** | **FEASIBLE LOCALLY** | ≈1.6 GB fp16 |
| Flan-T5 filter **training** | **REQUIRES REMOTE GPU** | ≈12.4 GB before activations — specification §10.4 |
| Retrieval index | **FEASIBLE** | flat exact index (D-37), CPU; 16 GB RAM is the limit to watch |
| Corpus processing | **EXECUTED LOCALLY** | CPU-bound; see §2 |
| Llama-3-8B inference | **NOT PRACTICAL LOCALLY** | specification §9 |
| ~100-question orchestration | **FEASIBLE LOCALLY** | pure Python; the full suite passes with no ML stack |

**Storage budget (ESTIMATED, nothing downloaded):** Llama-3-8B BF16 ≈16 GB
(not needed), Q4_K_M GGUF ≈4.9 GB, Flan-T5-large ≈3 GB, the trained filter
checkpoint ≈3 GB, MedCPT encoder + reranker ≈0.9 GB, HF cache overhead ≈1.5×
model size during download, generated outputs a few MB. A naive "download
everything" would need ~25 GB of the 30 GB free.

**Strategy.** Locally: corpus build, question sourcing and review, evidence
freezing and hashing, annotation packets, statistics, tests, MedCPT and
Flan-T5 *inference*. Remotely (free-tier 16 GB GPU session): filter training,
once; generation for every arm. The runner takes a frozen manifest in and
writes JSONL out, so only two small files cross machines. **Execution venue is
a deployment detail; the scientific comparison is unchanged.**

## 6. Repository layout

```
thesis_research/
├── README.md                      the one root readme
├── pyproject.toml
├── docs/                          four authoritative documents (§6.1)
├── _archive/                      earlier research direction, not active — see _archive/README.md
├── alzheimer_corpus/              the evidence corpus
│   ├── config/                    MeSH vocabulary, claim taxonomy, queries
│   ├── scripts/                   01_pubmed … 07_claim_classification
│   ├── metadata/ reports/ logs/   tracked provenance
│   └── data/                      gitignored — research data is never committed
├── systems/                       the experimental arms
│   ├── interfaces/                Evidence, Candidate, ExperimentResult,
│   │                              Generator, Retriever, System, HF generator
│   ├── baseline/                  no-filter control · RAG²-style Flan-T5 filter
│   └── proposed/                  the Temporal Filter
├── experiments/
│   ├── outputs/                   run artefacts (never overwritten)
│   ├── runners/run_end_to_end.py  the single entry point for pipeline steps 2–4
│   ├── evaluation/                questions · freezing · runner · rag_metrics ·
│   │                              accuracy · annotation · stats
│   ├── questions/                 the candidate question pool + review export
│   ├── retrieval/                 MedCPT retrieval + reranking + flat exact index
│   └── filter_training/           RAG² filter labels and training config
└── tests/                         unit + integration
```

### 6.1 `docs/`

| File | Purpose |
|---|---|
| `current_objectives.md` | **Canonical scope.** The three objectives, the pipeline, what is out of the critical path, and the known blockers. Governs every other document. |
| `research_experimental_specification.md` | **The method.** What each arm does, what is held constant, the parameters, the generator contract, filter training, metrics, statistics, question provenance, and how to run it reproducibly. |
| `status_and_decisions.md` | **This file.** What is executed, decided, measured, blocked and limited, plus the decision/fact/risk record and the change log. |
| `question_review.md` | **Reviewer instructions** for `experiments/questions/review.csv` — a human-facing worksheet, not a design document. |

Seventeen earlier documents were consolidated into these on 2026-09-19; the
change log (§13) records what went where.

### 6.2 What is not in the repository, and why

| Not vendored | Reason |
|---|---|
| The Alzheimer's corpus data | size; built locally; ignored by `alzheimer_corpus/.gitignore` |
| MedRevQA / MedChangeQA CSVs | no upstream licence file; Cochrane abstract text |
| MedQuAD checkout | CC BY 4.0 but large; cloned on demand |
| Model weights | disk; see §5 |

Each source adapter takes a path and fails with a message naming what to
fetch, rather than silently falling back to thesis-authored material.

### 6.3 `_archive/stage2_pilot_outputs/` is not research data

Those files were produced by running the superseded Stage-2 pilot against the
10-document fixture corpus. Every candidate is excluded, 0 pairs are eligible,
the questions are placeholders, and `check_a_interpretable: false` because the
change points are corpus publication dates rather than evidence-change dates.
**No number in that directory may be cited as a research finding.**

## 7. Established facts

| # | Fact | Tag | Source |
|---|---|---|---|
| E1 | RAG²'s filter is Flan-T5-large (≈770 M), trained on labels from a correctness-flip decision tree with a perplexity differential as tie-breaker at τ = top 25% | `[DOC]` | RAG² §3.2, Fig. 2, Eq. 3 |
| E2 | The primary label criterion is the correctness flip; perplexity was introduced explicitly "to address" cases where accuracy is unchanged | `[DOC]` | RAG² §3.2 |
| E3 | Perplexity is computed over the model's generated **rationale**; the paper's Eq. 4 notation is ambiguous and appears to score the query | `[DOC]` | RAG² §1, §5.1 vs Eq. 4 |
| E4 | RAG² corpus: 37.6 M docs / 116.7 M passages / 564.2 GB across PubMed, PMC, CPG, 18 textbooks | `[DOC]` | RAG² Table A1 |
| E5 | Reported gains +6.1 / +3.8 / +0.9 average accuracy points (Llama-3-8B 60.2→66.3; Meerkat-7B 68.6→72.4; GPT-4o 86.0→86.9) | `[DOC]` | RAG² Table 2 |
| E6 | Retrieval uses the **rationale** as query (original query excluded for length); reranking uses the **original query**. MedCPT at both stages | `[DOC]` | RAG² §3.3, §3.4 |
| E7 | RAG² contains no temporal representation anywhere — corpus, index, retriever, reranker or filter | `[INF]` | Absence across the paper |
| E8 | RAG²'s only open-ended evaluation is ClinicalQA25 — 25 queries, ROUGE-L and BERTScore | `[DOC]` | RAG² §A.4 |
| E9 | RAG² code is released at `github.com/dmis-lab/RAG2`; **the trained checkpoint is not distributed**, and the README states "this repository is not a full, one-command reproduction of the paper" | `[DOC]` | repository README, fetched 2026-09-17 |
| E10 | The paper demonstrates cross-**dataset** filter transfer, not cross-**backbone** transfer, and never states how the GPT-4o filter was obtained | `[DOC]` | RAG² §4.2, §4.3 |
| E11 | The paper's own stated limitations: snippets are labelled individually, ignoring combined effects; Flan-T5's context limit means one snippet at a time | `[DOC]` | RAG² Limitations |
| E12 | Largest medical-RAG expert evaluation to date: 18 clinicians, 80,502 annotations, 800 outputs; 22% top-16 relevance, 31% of queries with no relevant passage | `[PUB]` | arXiv:2511.06738 |
| E13 | MedChangeQA: 512 changed-verdict pairs derived from MedRevQA (16,501 pairs, Cochrane census 2000–2024) | `[PUB]` | Vladika et al. 2025 |
| E14 | Cochrane CD016297 (April 2026, 17 trials, 20,342 participants) concluded anti-amyloid mAbs probably produce little/no clinically meaningful cognitive difference at 18 months; contested within days | `[DOC]` | proposal §1.3 |
| E15 | **The RAG² abstract motivates the work by stating that LLMs "struggle with hallucinations and outdated knowledge" — and then measures neither.** Measuring evidence recency directly is therefore not a detour from the base paper; it addresses the base paper's own unmeasured claim | `[DOC]` | RAG² abstract vs §4 |

## 8. Decisions

Decisions before D-36 were taken under superseded framings. They are kept for
provenance; where one concerns what is *measured* rather than what is *built*,
D-36 and then `current_objectives.md` supersede it. Decisions concerning what
is *built* are still in force and are implemented as described in the
specification.

### 8.1 In force

| # | Decision |
|---|---|
| D-1 | Freeze everything upstream of admission; arms are distinct functions over one cached candidate list. **The central internal-validity guarantee.** |
| D-2 | **Provenance firewall:** the primary claim rests on externally-authored material; thesis-curated material supports replication and case study only. |
| D-3 | Soft supersession (down-weight); hard rejection only for retraction/withdrawal — claim-class tagging precision is unverified, and a hard gate would silently delete correct evidence in proportion to tagging error. |
| D-8 | The main comparison uses a prompt template **without** date annotations, so gains attribute to which passages were admitted rather than to date cues. |
| D-10 | Mandatory dual reporting of every rate (conditional on answering; and with abstentions counted). |
| D-12 | Rank-based rather than min-max normalisation of `ρ(s)`, so a global θ is well defined across queries. |
| D-13 | The filter is trained on general medical QA and applied to AD items, removing any suspicion that gains come from domain-specific fine-tuning. |
| D-14 | Contested evidence is **capped** by the common context budget, never exempted from it; preserved and dropped contested positions are recorded in run metadata. Exempting it would give the proposed arm more context than the baseline. |
| D-18 | Dev/validation/test assignment is a pure function of `question_id` and a recorded seed, with the **question** as the unit. |
| D-21 | `question_date` is the dataset's own date when supplied, otherwise one experiment-wide `evaluation_as_of_date`. **Never** derived from a passage in the pair; the schema rejects such a value. |
| D-26 | Terminology fixed: "recency", not "currency"; "admission policy", not "framework". One term per concept. |
| D-27 | `A(s) = (1 − λ)·ρ(s) + λ·R(s,q,t_q)`, admit if `A(s) ≥ θ`. One weight removes a redundant degree of freedom and makes `λ = 0` a built-in ablation. Tunables: λ, θ, H — all fitted on validation. |
| D-28 | The recency score is **plain age decay only**. Retraction, supersession and time-invariance are secondary, opt-in and recorded when enabled — folding validity rules into the same scalar would make an observed effect unattributable to age. |
| D-29 | The no-filter control arm is implemented: without it, neither comparison shows whether filtering helps at all. |
| D-30 | `λ = 0` is an **internal ablation**, not a fourth arm: it still admits at θ, so it is relevance-thresholded admission, not unfiltered. |
| D-33 | Candidate sets contain **only dated passages**, applied identically to every arm at construction, keeping the tunable count at three. |
| D-37 | Retrieval and reranking are MedCPT + a deterministic **flat exact** index — an approximate index introduces build-order-dependent recall, and a domain slice is small enough that exact search removes that hazard for free. |
| D-38 | The generator stays **Llama-3-8B-Instruct** at 4-bit NF4 on a free-tier remote T4, under a pinned contract. Local execution is ruled out at every precision — that rules out the *venue*, not the *model*. Declared fallback: Qwen2.5-7B-Instruct, reported as a limitation. |
| D-39 | The RAG² filter is trained by us: Flan-T5-large, the paper's recipe, effective batch preserved by accumulation. The released 5%-split file was downloaded and **contains 5 examples** — a format sample — so label regeneration is unavoidable. |
| D-40 | The labelled training set is a subsample: an honest budget, not a target. The weaker filter is a limitation on the **baseline's strength**, not a threat to the comparison's validity, and the no-filter control keeps it visible. |
| D-44 | `claim_status`, `temporal_status` and `disease_relevance` are deliberately **not** tagged by Stage 07. The first two need cross-passage comparison keywords cannot establish; temporal status is the thesis's own experimental treatment, and pre-baking a weaker keyword version would duplicate it with a worse method; disease relevance is already decided with a full rule trace by Stage 04. |
| D-45 | Guideline/textbook acquisition is implemented (`03_guidelines.py --download` + a Stage 04 extraction bridge), but **no specific document was added** — no title, URL or licence could be independently verified from this environment, and fabricating one was prohibited. PDF only: a scraped HTML page cannot be reliably separated from its boilerplate. |

### 8.2 Superseded, recorded for provenance

| # | Decision | Fate |
|---|---|---|
| D-4 to D-7, D-9, D-11 | Contested-before-superseded ordering · ψ-conditioned decay · a SOTA arm · source authority as a tested variable · judge-validation guard · phase ordering | Components moved out of the primary pipeline by D-25 and `current_objectives.md` |
| D-15, D-16, D-19, D-20, D-22 to D-24, D-31, D-32, D-34, D-35 | The matched-pair / admission-asymmetry machinery: inherited claim equivalence, interval-based temporal eligibility, claim classes as diagnostics only, no target pair count, frozen hashed Stage-2 spec, the unchanged-claim negative control, one primary backbone, the verified MedChangeQA join, and the measured AD census ceiling (only 9 of 512 items are AD-domain, exact power 0.000) | Moved to `_archive/test_pairs/` (2026-09-20); produce no thesis outcome |
| D-17 | `question_date` falls back to the newer passage's date | **Superseded by D-21** — it set the newer passage's recency to exactly 1 in every pair, maximising the contrast by construction |
| D-25 | Scope reduction to recency + rank-normalised relevance under one weight; support and authority become ablations | Absorbed into the current scope |
| D-36 | The research question becomes hallucination rate (primary) and QA accuracy (secondary); admission asymmetry ceases to be an outcome | **Superseded by `current_objectives.md`** (2026-09-18), which makes the RAG² comparison on standard RAG metrics the main contribution |
| D-41 | Stage 02 finalization failed on a producer/consumer log-evidence mismatch, not corpus damage; the reader was widened to the wordings actually written | Discharged — the manifest was produced (D-42) |
| D-42 | Stage 02 finalization executed and independently verified | Superseded by the fuller §2 verification |
| D-43 | Stages 04–07 rewritten to consume the real corpus; 01/02/03 deliberately left untouched | Executed — see §2 |

## 9. Assumptions

| # | Assumption | Status |
|---|---|---|
| A1 | Newer evidence lies outside the label model's parametric priors | **Resolved — false, and retired as blocking.** Measured: change points span 2004–2024, peaking 2013–2015; **1 of 512** falls after the Llama-3-8B cutoff. Re-scoped to the *generator* analysis, where it remains a real constraint (see K7) |
| A3 | 150–300 matched pairs are obtainable from 512 MedChangeQA items | **Resolved — supported.** All 512 reconstruct with both PMIDs, dates and texts; separation min 1 y, median 12 y, max 23 y |
| A5 | Perplexity labels require per-backbone filter retraining | Open (cost) |
| A6 | Baseline labels are predominantly confidence-derived | Open — instrument the label pipeline and report branch proportions |
| A7 | A 20–30 k subsample suffices to reproduce the baseline filter | Open — cost-verify on 100 items first |
| A8 | Claim-class tagging precision is adequate for down-weighting | Open — 300-passage audit; report P/R |
| A9 | Guideline PDFs yield poorer date coverage than XML | Open — report per-corpus null-date rates |
| A13 | The available hardware suffices | **Resolved for everything but generation and filter training** — see §5 |

## 10. Risks and confounders

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| K1 | **Abstention asymmetry.** The proposed arm can abstain; the baseline cannot. A threshold set high enough answers nothing and cannot hallucinate | **Resolved** | `ANSWER_ALWAYS` is the default, coverage is reported beside every rate, and a fully-abstaining system yields `interpretable: false` (specification §7) |
| K2 | **Prompt-parity drift.** Each arm accepts a `context_prompt` override | **Resolved** | `assert_prompt_parity` before the first item |
| K3 | **Filter fidelity.** A student-trained filter is not the paper's; if it is weaker, the baseline is unfairly weak and the intervention looks better than it is | **High, standing** | Report the filter's own validation accuracy; treat the no-filter control as the honest floor; label every report with which filter actually ran |
| K4 | **Reference-answer leakage.** A passage used to establish a reference answer entering the candidate set makes the evaluation circular | **Resolved** | The provenance firewall, enforced as a machine check (`assert_firewall`), not a convention |
| K5 | **Corpus recency skew** | Medium | Report the corpus date distribution alongside results |
| K6 | **Annotator blinding** | Medium | Arm identity hidden, order randomised, mapping stored separately |
| K7 | **Generator knowledge cutoff.** The generator may answer correctly from parametric memory without using the evidence — not hallucination, but it compresses the difference between arms | Medium | Record the cutoff relative to the reference time; treat it as a documented condition |
| K8 | **Inconclusive rather than negative outcome** | **High** | Report honestly either way; `current_objectives.md` explicitly requires reporting a negative result as such |
| K9 | **Pair-count pressure breaching the provenance firewall** — the cheapest way to more items is to author them | **High** | D-2; no code path produces an approved question |
| K10 | **Corpus irreproducibility** from live-database snapshots | Medium | Frozen query strings, MeSH settings, pull dates and per-collection counts are recorded in `alzheimer_corpus/` |
| K12 | **Reproduction cost overrun** — label regeneration is GPU-hours per item | **High** | Subsample budgeted from the measured timing check, not assumed |

## 11. Standing limitations for the thesis

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
8. Judged correctness (where used) is human or rule-based, not computed by an
   automatic scorer; the automatic metrics in `rag_metrics.py` are reported as
   what they are.
9. The corpus is an Alzheimer's domain slice, not RAG²'s 564 GB
   general-medical corpus. The *method* is unchanged; the *corpus* is the
   declared adaptation.

## 12. Test suite

```bash
python -m unittest discover -s tests -t .
```

**558 tests. Two fail, both pre-existing and both for the same reason: they
compare a tracked real artifact against what a fixture-scale run produces.**
Neither indicates a defect in the system under test, and neither should be
"fixed" by editing the tracked artifact.

| Failing test | Why |
|---|---|
| `test_corpus_normalize_pipeline.FixturePathRegressionTests.test_duplicates_registry_matches_and_is_valid_csv` | Compares a freshly-run 10-record fixture `duplicates.csv` against the committed **real** one (2,842 rows from the executed corpus). The committed file is correct; the fixture cannot reproduce it. |

`test_review_export.CommittedPoolTests.test_decision_columns_start_empty`
(asserted the committed `review.csv` shipped with every reviewer column
blank) was **replaced, not left failing**, once question review actually
completed (§3.3) — see `test_decision_columns_are_now_fully_recorded` in
`tests/unit/test_review_export.py`.

**The suite does not write to the corpus's tracked logs.** `tests/__init__.py`
redirects `ALZHEIMER_CORPUS_LOGS` to a temporary directory for the duration of
a run; `tests/unit/test_corpus_log_isolation.py` keeps that true and also
asserts that a *real* run — which never sets the variable — still logs to
`alzheimer_corpus/logs/`.

## 13. Change log

| Date | Change |
|---|---|
| 2026-09-21d | **Fixed a real Windows bug in `experiments/retrieval/corpus.py::read_passages`, found when the student ran `build_index.py --dry-run` against the real corpus for the first time.** `read_passages` loaded the entire `chunks.jsonl` into memory in one call (`path.read_text().splitlines()`); against the real, local 4.3M-chunk file this hit a genuine CPython/Windows limitation (`OSError: [Errno 22] Invalid argument`, raised when a single text-mode read exceeds roughly 2 GB) — reproduced verbatim in the student's traceback. Fixed by streaming the file line by line through an open handle instead of reading it whole; behaviour and every existing test are unchanged (28 `test_retrieval.py` tests still pass), and a new regression test (`test_never_reads_the_whole_file_into_memory_at_once`) asserts `read_passages` never calls `Path.read_text()` on the chunk file, so this cannot silently regress. `snapshot_id` was already safe (it hashed the file in 1 MB binary blocks) and needed no change. This was a genuine defect in the repository's code, not a problem with the student's corpus, Python install, or command. 500 tests pass; the same 1 pre-existing, unrelated corpus-fixture failure. |
| 2026-09-21c | **Roadmap step 3: validation/test split implemented over the reviewed question pool.** New `experiments/questions/split.py` (`python -m experiments.questions.split`) joins `candidates.jsonl` with the completed `review.csv`, keeps only ACCEPT/REVISE questions (REJECT and HOLD are excluded by construction — `assert_valid` raises if either ever leaks in), and assigns each to a **validation** or **test** split. Two splits, not three: nothing in this thesis trains on the 123-question pool (the RAG² filter trains on general-medical MedQA per specification §10; the generator is used off the shelf, never fine-tuned) — the pool only feeds the manual fit of λ/θ/H, which needs a held-out validation set and a separate, untouched test set, not a training set. Split is stratified by topic, then by `temporal_candidate` within each topic (an earlier version stratified by topic alone and left validation 74% temporal against test's 54%, which would have made a λ fit on validation look better than it generalises); seeded (`20260921`) and fully deterministic — `write_split()` refuses to silently change an on-disk `splits.json` if the reviewed pool has moved since, catching accidental drift. Result, committed at `experiments/questions/splits.json`: **113 usable (90 ACCEPT + 23 REVISE) → 23 validation / 90 test**; every topic with ≥2 usable questions appears in both splits; validation is 52% temporal vs. test's 60%. Every REVISE item is flagged `pending_revision: true` in the split file — the reviewer's own taxonomy treats REVISE as "sound claim, wording needs fixing," not a REJECT-in-waiting, but the wording fix itself has not been performed (it needs the actual cited source text) and evidence must not be frozen for a REVISE item before that happens. **Also fixed a latent hazard found while touching this code**: `export_review.py`'s `run()` always wrote a blank `review.csv`, with nothing stopping a re-run from silently erasing the now-completed human review; it now refuses to overwrite a `review.csv` that already has recorded decisions unless `--force` is passed. 24 new tests (`tests/unit/test_question_split.py`) plus 4 new/updated tests in `test_review_export.py` (the stale `test_decision_columns_start_empty`, which asserted the committed review was blank, was replaced with a check that it is now fully and validly recorded, per the entry below). 499 tests pass; the same 1 pre-existing, unrelated corpus-fixture failure (the second historical failure was this run's own stale test, now fixed, not a persisting defect). |
| 2026-09-21b | **Question review actually completed by the student, directly on `main`: all 123 questions decided.** Recorded in `experiments/questions/review.csv`: **90 ACCEPT, 23 REVISE, 10 REJECT, 0 HOLD** — real source-checked judgement (Cochrane Library / PubMed), which this pipeline's own automated passes could never perform, since both are network-egress-blocked in this build environment (confirmed by direct test). This supersedes and discards an earlier, explicitly-invalidated automated substitute that had been proposed on a feature branch; that branch's PR was closed without merging once the real review landed, specifically so it could not overwrite it. §3.1's "Human question review" blocker is closed; §3.2 step 1 marked done. |
| 2026-09-21 | **Corpus date-coverage pre-flight measurement: DONE.** The 2026-09-20 audit flagged that the real corpus's null publication-date rate had never been measured. The student ran a read-only, offline check against the real, local `alzheimer_corpus/data/chunks/chunks.jsonl` (4,377,041 chunks — this file has never existed in this environment; `data/**` is gitignored by design) using a temporary, non-repository script: **100.00% valid `publication_date` (4,377,041/4,377,041), 0 missing, 0 invalid.** No corpus file was read, written, moved, or regenerated by the check itself, and no corpus stage was rerun. §3.2 step 3 marked done; §9's A9 assumption (guideline-vs-XML date coverage specifically) remains open only because no guideline PDF has been curated (0 rows) to compare against — unrelated to this measurement. This closes the last pre-flight question before λ/θ/H fitting: `dated_only()` (specification §15.3) drops nothing, so candidate-set size at N=20 is unaffected by missing dates. |
| 2026-09-20 | **Renamed "recency" to "Temporal Filter"/"temporal score" throughout the active repository, and archived the machinery that does not belong to the current thesis.** Purely a clarity pass for a beginner MS student — no methodology, formula, or experimental result changed. **Renamed:** `systems/proposed/recency.py` → `temporal.py` (`RecencyPolicy`→`TemporalPolicy`, `RecencyResult`→`TemporalResult`, `RecencyState`→`TemporalState`); `RecencyAwareAdmissionPolicy`→`TemporalFilterPolicy`, `RecencyAwareSystem`→`TemporalFilterSystem` (`admission.py`); `AdmissionScorer.recency_weight`→`temporal_weight`, its `score(recency=...)` parameter →`temporal=`; the proposed arm's `name` constant `"P_RECENCY"`→`"RAG2_TEMPORAL"`. Every doc, docstring, comment, and test updated to match; a repository-wide search for `recency` (case-insensitive) now returns zero hits outside `_archive/` and this document's own historical entries (D-17, D-25, D-26, D-28, E15, K5 — left in their original wording, since they record what was decided *then*; see the terminology note above §1). **Archived** (`_archive/`, see `_archive/README.md`): `experiments/test_pairs/` (the matched old-vs-new evidence pair design, its 8 scripts, and its data directory), `systems/proposed/{contested.py,verifier.py}` (contested-evidence detection and answer verification — both already `SECONDARY`-labelled and off by default), `experiments/configs/stage2_pilot.yaml`, `experiments/outputs/stage2_pilot/` → `_archive/stage2_pilot_outputs/`, and the 5 test files that exercised only that archived code (`test_eligibility.py`, `test_attrition_split.py`, `test_schema.py`, `test_audit_invariants.py`, `test_pilot_pipeline.py`, all moved to `_archive/test_pairs/tests/`). **Closed the one dependency the move would otherwise have created**: `systems/proposed/__init__.py` re-exported `ContestedDetector`/`ClaimVerifier`/etc. from the now-archived files — those re-exports were removed (nothing outside `__init__.py` imported them; verified by grep before removing). `pyproject.toml`'s package list and `OutputState.CONTESTED`/`PassageDecision.contested` (unreachable dead code once `contested.py` left the tree - nothing could ever set `is_contested=True`) were removed with it. Verified with an AST-based repository-wide import scan (`tests/unit/test_scope_invariants.py::test_active_code_does_not_import_the_archive`) that **zero active files import from `_archive/`**. `docs/research_experimental_specification.md` §14 (99 lines detailing the now-archived external-data contract) replaced with a 6-line pointer to the archive. README rewritten around the plain-language research question or **whether a Temporal Filter improves RAG² for Alzheimer's QA** rather than the implementation-first framing it had. 470 tests pass (103 fewer than before this pass — exactly the 5 files moved to `_archive/test_pairs/tests/`, which are no longer discovered under `tests/`); the same 2 pre-existing, unrelated failures. |
| 2026-09-20 | **Research-realignment audit: re-read the base paper directly (all 15 pages of the NAACL 2025 PDF, not from memory) and re-inspected the repository against it.** Requested because the earlier (now-superseded) admission-asymmetry design had drifted through incremental additions; the finding is that the *current* design (`current_objectives.md`, adopted 2026-09-18) already **is** the minimal "RAG² baseline → temporal filter → controlled experiments → evaluation → ablation → statistics" methodology this audit was asked to (re)establish — confirmed rather than rebuilt. Concrete findings from the re-read and audit: (1) the paper's own evaluation never reports a retrieval-only ranking metric (Recall@K/MRR/nDCG) anywhere, even while sweeping top-k in Figure 3 — it validates retrieval only through downstream accuracy, which the existing `token_f1`/`rouge_l_f1`/`context_precision/recall` already do; formalised as a decision *not* to add ranking metrics, with reasoning, in specification §11.1 (the structural reason: retrieval is frozen and identical across every arm by construction, D-1, so a ranking metric cannot vary with the thing being compared). (2) `systems/proposed/{contested.py,verifier.py}` re-confirmed genuinely isolated — both self-labelled `SECONDARY` in their own docstrings, off by default, not imported anywhere in the critical path (`run_end_to_end.py` mentions them only in a comment listing what is out of scope) — no further pruning needed. (3) **Corpus temporal-metadata mechanics confirmed sound but two things newly found and recorded**: `_parse_pub_date()` picks one of possibly several `<pub-date>` elements by a documented, deterministic priority (`epub > pub > ppub > collection`) and returns "no date" rather than fabricating one - -but `01_pubmed_download.py` **only ever retrieves a PMID list, never abstracts or dates** (self-documented in `04_normalize.py`'s own top-of-file comment); all real corpus text and every real publication date come from PMC's JATS XML, not from the 676 PubMed PMIDs, which serve only to shape the PMC search and contribute no independent content. And: **the real corpus's null-date rate has never been measured** — `dated_only()` (specification §15.3) silently drops undated passages before either arm sees them, which is the correct behaviour, but nobody has run the one cheap count (non-empty `publication_date` across the real `chunks.jsonl`) that would show whether that drop is negligible or material before λ/θ/H are fitted. Recorded as an open pre-flight measurement, not assumed either way. (4) Verified the question pool already gives the temporal question a real, non-trivial contrast to test: 74/150 sourced candidates carry `temporal_candidate=True` (a Cochrane review cited at `.pub2`+, i.e. actually revised over time) — sourced from real, cited records, not invented. **Added, as the one concrete gap the audit found worth closing**: `temporal_candidate` was tracked on every `EvaluationQuestion` and summarised at the pool level but dropped at freezing and never used downstream. It now survives into `FrozenItem` (`from_question()`) and `run_end_to_end.py`'s report gains a `temporal_subgroup` breakdown — the same per-system metrics, split by whether each question's evidence base has actually been revised over time versus not (specification §11.2) — the direct, zero-new-data-collection test of whether the Temporal Filter's effect concentrates where it should matter rather than sitting flat across the pool. On the module's synthetic fixture (not sourced from a real Cochrane republication) the temporal subgroup is honestly empty, not populated to look exercised. (5) Confirmed the ablation study needs no new infrastructure beyond what already existed: `lambda=0` (component removed, mathematically verified correct in the 2026-09-19 session) plus a half-life (`H`) sensitivity sweep answer "does the chosen temporal window matter" using the `--half-life` flag that already exists — both configurations are already runnable, zero new code. 4 new tests (`FrozenItem.temporal_candidate` pass-through, the subgroup-breakdown grouping logic with a genuine mixed contrast, and that the report key is always present). 573 tests pass; the same 2 pre-existing failures. No corpus rerun, no architecture change, no new dataset, no new model. |
| 2026-09-19 | **First real-generator smoke test confirmed working end to end** — see §1.1. `HuggingFaceGenerator` loads a real causal LM, applies its chat template, quantizes to NF4, generates greedily and returns real text through `run_end_to_end.py --real-model`, verified with a small stand-in (TinyLlama-1.1B) since the thesis generator needs a GPU the student's laptop does not have. 0 errors across 16 real-model records; metrics scored the real output correctly, including the annotated-fixture case of `context_precision/recall` (0.000/1.000, not `n/a` — gold evidence ids are present in this fixture, unlike the real-data case). No repository defect found; the one failure encountered (`transformers` 5.17.0 silently disabling PyTorch below 2.5) was an environment-drift issue outside this repository, resolved by the student pinning `transformers==4.51.3`. No thesis result was produced or claimed. |
| 2026-09-19 | **`admit_threshold` (θ) range-checked; θ/H/budget made reachable from the CLI.** `AdmissionConfig.validate()` refused an unresolved θ but not an out-of-range one, although `A(s)` is a convex combination of two `[0, 1]` quantities and θ > 1 or θ < 0 is silently degenerate (admits nothing, or everything, on every question) rather than merely wrong. `validate()` now rejects θ outside `[0, 1]`. Separately, θ, the temporal half-life and the context budget were module-level constants in `run_end_to_end.py` with no CLI override, so a real run would have used the fixture placeholders (θ=0.5, H=365, budget=1) regardless of a validation-split fit — defeating `validate()`'s deliberate refusal to default θ. Added `--theta`/`--half-life`/`--budget`; the report now records them in a readable `system_config` block flagged `theta_and_half_life_are_fitted: false`, rather than only inside an opaque config hash. Two existing tests that used θ=1.1 as a shortcut for "nothing clears θ" were rewritten to produce the same condition in range. `λ=0` was separately audited and needs no correction: it zeroes the temporal term exactly, leaving `A(s) = ρ(s)`. 569 tests pass. |
| 2026-09-19 | **Corpus provenance logs were being contaminated by the test suite, and are not any more.** `_common.get_logger()` resolved its directory from the loaded module's own `__file__`, so a test calling a stage's `main()` in-process against a temp corpus still appended to the real, git-tracked `alzheimer_corpus/logs/quality_control.log` — a suite run added Stage-06 whitespace-tokenizer lines naming `/tmp` paths to the file the §2 evidence chain is read out of. `get_logger()` now honours `ALZHEIMER_CORPUS_LOGS`; `tests/__init__.py` sets it once for the whole run; `tests/unit/test_corpus_log_isolation.py` (4 tests) guards both the redirection and the unchanged real-run default. The contaminating lines were reverted; every tracked corpus artifact matches the student's own pushed commits. |
| 2026-09-19 | **Documentation consolidated from 21 Markdown files to 4.** `system_specification.md`, `generator_contract.md`, `filter_training.md`, `rag2_classifier_feasibility.md`, `experimental_parity_audit.md`, `methodology.md`, `question_sources.md` and `external_evaluation_data.md` were merged into `research_experimental_specification.md` (whose superseded admission-asymmetry contents were replaced — they are recorded as provenance in `current_objectives.md` and §8.2 above). `research_ledger.md`, `next_steps.md`, `hardware_and_resources.md`, `question_pool_status.md`, `repository_structure.md`, `experiment_outputs.md`, `feasibility_and_alignment_audit.md`, `research_understanding.md` and `proposal_scope_amendment.md` were merged into this file. `frozen_scope.md`'s live content (interface requirements, reporting requirements, the formula) moved into the specification and its §7 superseded-design paragraph into `current_objectives.md`. `current_objectives.md` and `question_review.md` were kept. Every code and test reference to a merged document was updated to its new home in the same commit. |
| 2026-09-19 | **Readiness defects corrected before any real run.** (1) The real `FlanT5RAG2Filter` was unreachable from `run_end_to_end.py` — the baseline could only ever be the all-HELPFUL stand-in, and the report did not say so; added `--rag2-checkpoint`, plus `baseline_filter`/`baseline_is_trained_rag2` in the report and a console warning. (2) `rag_metrics.context_scores` returned 0.0 for an unannotated question, which under the provenance firewall is the *expected* real-data case — every arm would have reported context precision 0.000 as if measured; it now returns `None`, `aggregate()` averages over only the annotated rows and reports `context_scored_n`. (3) `--real-model` constructed `ModelSpec(name=...)`, a field that does not exist, so it raised immediately; fixed with `--model-name`/`--model-revision`/`--quantization` (defaulting to the contract's `nf4`). Also: `RAG2Config` accepted a non-positive context budget where the other two arms rejected it. 553 tests pass. |
| 2026-09-19 | **Alzheimer's corpus marked COMPLETE / FROZEN** after independent end-to-end verification of all seven stages — see §2. No integrity problem found; nothing corrected or rerun. Added `tests/integration/test_corpus_to_retrieval.py`. |
| 2026-09-18 | Stage 07 found to have the same architectural flaw Stage 06 had (whole corpus loaded into one list, nothing written until the end, no progress logging) at 4.3 M-chunk scale; rewritten to stream to a temp file, replace atomically, log every 100,000 chunks and keep only an 8-field projection for the stratified sample. A second real bottleneck — per-keyword regex scans over full chunk text — was replaced by one tokenization plus set membership, with a regex fallback for phrase keywords (≈12× faster on a realistic 30 k-chunk sample). Verified field-for-field identical to the original algorithm, preserved as a reference implementation in the test. |
| 2026-09-18 | Scope realigned to the three current objectives (`current_objectives.md`). Added the missing single entry point `experiments/runners/run_end_to_end.py` and `experiments/evaluation/rag_metrics.py`. Repository reduced to a single `main` branch; feature branches and pull requests are no longer used. |
| 2026-09-18 | Stage 06 rewritten for performance and observability after a ~1 h run produced no output: batched encode/decode (same token boundaries), streamed output, periodic progress, `--resume` refusing a mismatched tokenizer/size/overlap. Chunking specification unchanged. |
| 2026-09-18 | Removed `alzheimer_corpus/README.md` and the `tools/` benchmark; relocated the PMC licensing-gate finding rather than losing it. Rewrote the root `README.md` against the current scope. |
| 2026-09-17 | Generator contract (D-38), filter-training strategy (D-39/40) and the parity/abstention audit decided and implemented. |
| 2026-09-16 | Design freeze under the admission-asymmetry framing. One genuine fairness defect fixed: the no-filter control emitted context in rank-sorted order while the other arms emitted candidate-list order. D-30 to D-35 recorded. |
| 2026-09-15 | Stage-2 readiness audit and methodological review: D-15 to D-29 recorded; D-17 superseded by D-21; scope reduced (D-25). |
| 2026-09-11 | Ledger created from the proposal and the RAG² paper. |
