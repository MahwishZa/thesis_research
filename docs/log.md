# Research Log

A chronological record of the implementation and research work performed on
this repository's `main` branch, from project start through the current
repository layout. Reconstructed from the commit history for documentation
purposes. Each entry states what changed and why; it does not restate full
technical detail already covered in `methodology.md`, `data.md`,
`evaluation.md`, and `reproducibility.md` — see those for the current,
authoritative description of the system. This log is the history of how the
project got there.

Dates are the date of the commit(s) described. Where a phase corrected an
earlier decision, both the original decision and the correction are recorded
— nothing here is retroactively cleaned up to look like the final design was
obvious from the start.

---

## Phase 1 — Project initialization and PubMed/PMC acquisition (Aug 30 – Sep 4)

- Repository initialized; a PubMed acquisition pipeline (ESearch + EFetch)
  and its integration tests were the first working code.
- A read-only PMC open-access/reuse inventory script was added, followed by
  a safe, resumable PMC full-text XML downloader. Acquisition and
  reconciliation were run and the results recorded, including a retry pass
  that resolved 70 of 71 stale-MD5 records.
- A PMC XML parser was implemented with tests, then hardened through two
  rounds of edge-case and structural-check fixes surfaced by a full-corpus
  QC report run before and after each fix.
- A Windows-specific `OverflowError` reading the PubMed CSV was fixed
  (Python's default CSV field-size limit was too small for some records).
- Corpus organization and metadata were finalized for milestones M1–M4, and
  a retrieval-ready chunk layer with provenance validation was added.
- Initial MedCPT retrieval infrastructure was added (with candidate replay
  for reproducibility), and generated JSON output was made
  byte-deterministic across platforms so runs could be diffed reliably.

## Phase 2 — RAG² baseline integration and thesis architecture (Sep 4)

- The original RAG² reproduction (Sohn et al., NAACL 2025) was audited
  component-by-component against the published paper and the authors'
  released code (`f0c9f98`). Verdict: faithful, with two fixes applied —
  notably adding a test pinning the equivalence between the reproduction's
  single-forward-pass scoring path and the release's
  `generate(...).scores[0]`, which had been claimed but never actually
  tested.
- The MedCPT embedding step was made GPU-explicit and resumable.
- A first pass at the thesis's own system architecture layer was added on
  top of the integrated RAG² baseline.

## Phase 3 — Research re-scoping to an Alzheimer's-specific design (Sep 11)

- A research understanding report and a research ledger were added to track
  design decisions going forward (later superseded by the current four-doc
  structure — see Phase 6).
- The corpus requirements were re-derived specifically for an Alzheimer's
  disease scope: a Stage 1 specification, feasibility measurements (M1–M4),
  a claim-class taxonomy, Alzheimer's-specific corpus configuration and
  query families, and a relevance gate with decision traces and tests.
- A corpus card and provenance registries were documented, and the
  Alzheimer's corpus work was consolidated into a single folder.
- Two divergent branch histories were reconciled by merge.

## Phase 4 — Alzheimer's corpus pipeline and a scope freeze (Sep 11 – Sep 16)

- The Alzheimer's corpus pipeline itself was implemented, then corrected:
  MeSH terms, search scope, and the claim taxonomy were fixed after review.
- Methodological defects were found and fixed in the admission-scoring
  design of the time: operation order in the scoring formula, filter
  fidelity to RAG², and a budget policy that should have been a "cap" but
  wasn't.
- A "Stage-2" line of work (temporal test-pairs, contestedness/authority
  signals) was built out with its own readiness audit, then its
  methodological issues were resolved by **reducing its scope** rather than
  patching around them — four further defects found in a consistency audit
  were fixed at the same time.
- **Scope freeze:** the design was frozen at a two-component admission
  formula (relevance + a temporal/recency term), down from an earlier
  four-signal design that also included entailment-based "support" and
  source-authority terms. A cross-arm context-ordering defect was fixed as
  part of freezing it. An updated MS thesis proposal was generated
  reflecting the frozen scope (decisions D-26 through D-33 in the
  then-current research ledger).

## Phase 5 — Corpus-independent evaluation infrastructure and objective refinement (Sep 16 – Sep 18)

- `MedChangeQA` was evaluated as a possible question source and its
  structure verified; an Alzheimer's-restricted primary pool built from it
  was rejected as unsuitable, motivating the project's own question-pool
  construction (see Phase 8).
- A feasibility audit was run, and evaluation infrastructure that could be
  built and tested independently of the real corpus was completed —
  allowing the test suite to run without network access or ML dependencies.
- An asymmetry in how the pipeline handled abstention (a system admitting
  no evidence) was resolved, and the real RAG² filter's training recipe was
  recorded precisely.
- The project's various experiment scripts were consolidated into one
  `experiments/` tree, and the candidate question pool was built. Reviewer-
  facing files were made neutral (stripped of any signal that could bias
  human review), and folder/file names across `experiments/` and `docs/`
  were simplified.
- The research question was reframed and formalized in stages ("Step A"
  through "Step F"): first making a hallucination-rate framing
  authoritative, then the system specification and MedCPT retrieval stage,
  then the generator execution contract and filter-training strategy,
  finishing with an end-to-end validation of the whole pathway and a status-
  doc update. (The hallucination-rate framing from Step A was itself later
  narrowed — see Phase 8 — once the Temporal Filter framing was finalized;
  the underlying annotation infrastructure built for it was kept, not
  deleted, and remains available as documented in `evaluation/annotation.py`.)
- Context-budget and generator parity were enforced in code across every
  experiment arm, closing a gap where arms could otherwise have silently
  differed on something meant to be held constant.

## Phase 6 — Building the real corpus, Stages 01–07 (Sep 18 – Sep 19)

- The PMC download workflow was updated and a Stage 02 finalization bug was
  fixed (the stage was not reading the unavailable-evidence records it
  itself had written). The finalized PMC manifest was verified and PMC
  retrieval marked executed.
- A repository-wide audit fixed packaging (`pyproject.toml` had been empty,
  which silently broke `pip install -e .`), restored a deleted test
  fixture, and removed dead code.
- Stages 04–07 (normalize, deduplicate, chunk, claim-classify) were
  rewritten to consume the real corpus instead of fixtures, fixing a
  licence-matching bug in the process. Guideline/textbook acquisition
  (Stage 03) was implemented and run, but added no documents — no candidate
  could be independently source-verified in the build environment, and this
  was recorded as a known, non-blocking gap rather than papered over.
- Stage 06 (chunking) was rewritten to batch tokenizer calls and stream
  output rather than holding everything in memory, and its keyword matching
  in Stage 07 was optimized (precompiled regex, then a word-set match)
  after profiling showed it was a bottleneck.
- The pipeline and documentation were realigned to three objectives and a
  refined 5-step objective/ablation framing; a repository-wide audit against
  that refined pipeline corrected a wrong assumption about corpus status
  that had been carried in the docs. A memory/visibility bug affecting both
  Stage 06 and Stage 07 was found and fixed in both places.
- Documentation was consolidated from many working documents down to four
  authoritative files (the precursors of today's `methodology.md`,
  `data.md`, `glossary.md`, `evaluation.md`), and the test suite was fixed
  so it no longer wrote corpus logs as a side effect of running.

## Phase 7 — Corpus freeze, RAG² baseline reachability, Temporal Filter naming, question-pool audit (Sep 19 – Sep 20)

- The Alzheimer's corpus was verified and frozen, and retrieval was proven
  able to consume it end to end.
- Two readiness defects were found and fixed in the experimental setup
  (`8c3e8c2`): the runner had no way to reach the real trained RAG² filter
  (it was hardcoded to an all-HELPFUL stand-in with no report field saying
  so), and `context_precision`/`context_recall` returned `0.0` — read as a
  real bad result — for the expected case of a question with no
  gold-evidence annotation, rather than `None`. Both were fixed: a
  `--rag2-checkpoint` path was added with the report stamping which filter
  actually ran, and unscored metrics now report `None` with a visible
  `context_scored_n` coverage count instead of a misleading zero.
- `theta` was made range-checked, and fitted parameters were made reachable
  from the CLI rather than hardcoded.
- The first confirmed smoke test against a real (not stand-in) generator
  was recorded.
- A research-realignment audit re-read the RAG² paper against the current
  implementation, confirmed the design, and added the temporal-subgroup
  breakdown that the evaluation now reports separately (per
  `evaluation.md` §3).
- **Naming cleanup, no methodology change:** "recency" was renamed
  "Temporal Filter" throughout code, tests, and docs (`RecencyPolicy` →
  `TemporalPolicy`, `R(s,q,t_q)` → `T(s,q,t_q)`, etc.), and code that no
  longer answered the current research question — the temporal
  counterfactual test-pair work, a claim-contestedness detector, a post-hoc
  answer verifier, and a question-pool audit that had explored an
  alternative 82-question pool — was moved to `_archive/`, not deleted.
  `_archive/README.md` records what each piece was for and why it was
  archived rather than kept active.
- Real-corpus date coverage was checked and found 100% valid — a pre-flight
  condition for the temporal signal to be meaningful at all.
- The 123-question Alzheimer's evaluation pool was audited and repaired,
  and a `review_decision` (ACCEPT/REVISE/REJECT/HOLD) was recorded for
  every question. The validation/test split (23 / 90, stratified by topic
  and temporal-candidate status) was then built over the reviewed pool.
- Separately, a Windows-specific `OSError` reading the real corpus was
  fixed, corpus read+hash was consolidated into a single pass with progress
  output, a memory-doubling risk in MedCPT encoding was found and fixed
  proactively (before the next real run hit it), and real duplicate
  `chunk_id`s surfaced by the actual corpus build were diagnosed and
  handled.

## Phase 8 — RAG² filter-training pipeline (Sep 21)

- A readiness audit for filter training confirmed existing wiring, added a
  `--rag2-device` option, and applied safe encoder speed fixes.
- The RAG² filter-training pipeline itself was implemented: label
  generation (from general-medical MedQA data, per the paper's
  correctness-flip recipe) and the training script.
- CUDA-availability error handling was clarified, and filter-label
  generation was made CPU-safe for the (common) case of no local GPU.
- Documentation was added for running filter training on a free GPU tier:
  a step-by-step guide with troubleshooting, a roadmap status snapshot
  recording the GPU-access blocker and timeline scenarios, and a Kaggle
  notebook setup checklist that was itself corrected twice (fixing the
  torch/CUDA install, adding `sentencepiece`, adding early validation
  checkpoints) as the actual setup was worked through.

## Phase 9 — First real-data pilot run and parameter fitting (Sep 22 – Sep 23)

- A trained RAG² filter checkpoint was produced on Kaggle (per the
  `20260922T...` timestamp in the uploaded archive) and integrated, then
  the upload archive was removed after extraction (repository hygiene —
  the checkpoint itself is gitignored, per `reproducibility.md` §7). As
  documented in the runner scripts, this checkpoint was trained on a
  reduced label set (20 labels, 1 epoch) and reached `validation_accuracy
  = 0.0` — i.e. it is not yet a usable classifier; full training on the
  complete label set was deferred past this deadline-driven pass.
- Under a same-night deadline, `run_real_evaluation.py` was added: a
  reduced-scope runner assembled entirely from already-tested components
  (retrieval, MedCPT encoders, `FlanT5RAG2Filter`, `TemporalFilterSystem`,
  `rag_metrics`) to get a first real-data baseline-vs-proposed comparison,
  over the real 113-question reviewed pool against a **pilot-scale** corpus
  index (~1% of the full corpus), with an **extractive stand-in** in place
  of a real generative model. Every simplification was stated explicitly
  in the script and written into each report's limitations block.
- This run's first hyperparameters (`theta`, `half_life`) were **unfit
  placeholders**, and degenerated at high `lambda`: `theta=0.5` admitted
  almost nothing once the temporal term dominated, producing a false
  "does not improve" reading that was really "these values were never
  tuned." This was fixed by adding `fit_and_evaluate.py`
  (`0708f1d`): a proper validation/test procedure that grid-searches
  `theta`/`half_life`/`lambda` on the 23-question validation split only
  (no model calls needed — the admission math is pure arithmetic over
  already-retrieved candidates), then reports baseline vs. proposed on the
  held-out 90-question test split using the fitted configuration.
- The fitting objective itself was corrected twice more: the evaluation
  target was changed from textual overlap with a fixed reference to
  scoring **currency** directly (`93753e0`), and a guard was added so a
  configuration could not win by admitting a near-empty, degenerate
  evidence set that happened to score an artificially high currency gain
  (`95db8de`) — an earlier version of the objective had no such guard. A
  real date bug was fixed and the fitting grid widened, with a divergence
  diagnostic added (`a1d757d`).
- **Statistical rigor fix:** the reported verdict had been decided by an
  arbitrary magnitude threshold on the relative currency change (e.g. a
  0.85% change), which could not distinguish a real small-but-consistent
  per-question effect from a couple of outlier questions moving the
  aggregate. This was replaced with a **paired sign test** over
  per-question currency outcomes on the temporal-candidate test questions
  (`884e2af`), reusing the same exact-binomial machinery already used for
  McNemar's test — not new statistical infrastructure. The verdict is now
  IMPROVES / DOES NOT IMPROVE / NO SIGNIFICANT DIFFERENCE based on
  `p ≤ 0.05` and direction; the relative-magnitude figure is kept only as a
  secondary "how big" diagnostic.
- A data-integrity mismatch between `review.csv` and `splits.json` was
  found and fixed, and results from the fitted, sign-test-based run were
  committed under `experiments/results/fit_and_evaluate/`.
- **Pilot-scale result (recorded here for the record, not as a thesis
  finding):** on this pilot-scale run — reduced corpus, the un-trained
  (`validation_accuracy = 0.0`) filter checkpoint, and an extractive
  stand-in generator — the paired sign test found **no significant
  difference** between the Temporal Filter and the RAG² baseline
  (`p ≈ 0.89` on the main comparison; `p ≈ 0.89` on the λ=0 ablation, 26
  wins / 28 losses out of 54 temporal-candidate questions). This
  demonstrates that the pipeline and statistical procedure execute
  correctly end to end on real data; per `evaluation.md` §7 and
  `reproducibility.md` §5, it is explicitly **not** the reported comparison
  the thesis will use, because the checkpoint, corpus scale, and generator
  are all reduced-scale stand-ins by design at this stage.

## Phase 10 — Repository reorganization for thesis presentation (Sep 23 – Sep 24)

- Repository hygiene: stale docs removed, `.gitignore` tightened, and
  documentation updated to accurately describe the pilot result as an
  engineering checkpoint rather than a finding.
- The repository was reorganized into a research-paper-friendly top-level
  layout: `evaluation/` moved out of `src/` to the top level, `tests/`
  moved under `evaluation/`, and `results/` moved under `experiments/`,
  leaving `src/`, `corpus/`, `experiments/`, `evaluation/`, and `docs/` as
  the only top-level folders. Import paths, path constants, and
  `pyproject.toml`'s package list were updated to match; a broken link was
  fixed and the intermittent-looking `test_corpus_log_isolation` failure
  was root-caused (a missing `-t .` flag in the invocation, not a real
  flake) during the same audit.
- The README's methodology diagram was simplified from an 8-node graph to
  a clearer, minimal flow, and the standing status table in the evaluation
  section was removed as unnecessary.

---

## Current status (as of this log)

- **Methodology, data pipeline, and software infrastructure:** complete,
  tested (536/537 tests passing — the one failure is an expected fixture-
  vs-real-corpus registry comparison, not a defect), and unchanged in
  substance since the Phase 4 scope freeze.
- **Corpus:** built and frozen (Phase 6); the guideline/textbook source
  (Stage 03) remains empty as a known, non-blocking gap.
- **Question pool:** 123 questions, fully human-reviewed, split into 23
  validation / 90 test questions (Phase 7).
- **What is still reduced-scale, pending a full run:** the RAG² baseline
  checkpoint (needs training on the full label set, not the 20-label/
  1-epoch pilot checkpoint), the retrieval index (needs building over the
  full corpus, not the ~1% pilot slice), and the generator (needs a real
  generative model in place of the extractive stand-in). None of these are
  methodology changes — see `reproducibility.md` §5 for exactly what each
  requires.
- **Reported thesis result:** not yet produced. The only real-data result
  committed so far is the Phase 9 pilot run, which is explicitly a pipeline
  validation, not the comparison the thesis will report.
