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

## Phase 11 — Full-scale index build, and local filter-training tooling (Sep 24 – Sep 27)

- **Full retrieval index built.** The original `build_index.py` held every
  passage's text and the full ~12.8 GB vector matrix in process memory at
  once; a diagnostic run on the student's 16 GB-RAM laptop (a live
  memory-monitor log, not a guess) showed this pushed free RAM to ~0 and
  caused sustained OS paging before any model or vectors were even
  involved. Rewritten as a two-pass, memory-bounded, checkpointed build
  (`streaming_index_build.py`): a cheap planning pass fixes row order and
  count, then vectors are written directly into a disk-backed memmap
  instead of a resident array. Verified bit-identical to the original
  in-memory output. Run for real: **4,376,141 × 768, 12.52 GB**,
  reload-verified. A related repository-hygiene gap was fixed at the same
  time: the resulting 12.5 GB directory was untracked but not gitignored,
  which was both slow (`git status` scanning it) and a real risk of an
  accidental multi-GB commit.
- **Filter-training investigation.** The pilot checkpoint (Phase 9) was
  inspected directly, not assumed: the saved archive contains no model
  weight file at all, on top of the already-known 18-training-example/
  0.0-accuracy problem — nothing in it is reusable. The existing label-
  generation path (`rationale.py`, Llama-3-8B-Instruct via `bitsandbytes`)
  needs ~5.5–6 GB VRAM (4-bit) or ~16 GB RAM (CPU) — both infeasible on
  the student's 4 GB VRAM / 15.2 GB RAM laptop. A free-tier remote GPU
  would trivially satisfy this with zero methodology deviation (the
  20-label pilot file was almost certainly produced exactly that way,
  per the student's own earlier Colab notebook), but running anything
  remotely was ruled out by explicit instruction — the student chose to
  run the entire remaining pipeline locally.
- **GGUF/`llama.cpp` adopted for local label generation**, after
  reassessing (not assuming) that it was necessary: every non-GGUF local
  alternative either doesn't fit the hardware (the existing HF paths;
  reducing question count, which doesn't touch peak memory at all) or is
  a materially larger deviation (swapping to a smaller substitute model
  changes which model reasons over the label-generation prompts, vs.
  GGUF only changing the quantization/execution backend for the *same*
  model). Implemented as `rationale_gguf.py` (`GGUFRationaleScorer`),
  matching `Llama3RationaleScorer`'s prompt template, chat formatting
  (via the real HF tokenizer, weights never loaded), greedy decoding, and
  answer-extraction exactly — the only difference is the execution
  backend, disclosed via the scorer's recorded `name`. 15 tests, all
  against fakes (neither `llama_cpp` nor `transformers` is installed in
  the development environment, by design).
- **Checkpoint/resume and a calibration mode added to `build_labels.py`**,
  which previously wrote its whole output in one shot at the end — a real
  gap given a local CPU-based label-generation run is expected to take
  hours, unattended. `--calibrate N` scores a small sample, reports
  measured throughput and an extrapolated full-run estimate, and writes
  neither an output file nor a checkpoint, so it can be repeated freely
  before committing to the real run. 10 tests, including a simulated
  mid-run crash verifying no pair is silently re-scored or dropped on
  resume.
- **Packaging gap fixed:** `pyproject.toml` declared no dependencies at
  all for filter training (`bitsandbytes`, `datasets`, `sentencepiece`,
  `accelerate` were imported but never listed) — found during audit, not
  previously known. Split into two installable extras
  (`filter-training-hf`, `filter-training-gguf`) so installing the local
  path doesn't pull in `bitsandbytes`, which is CUDA-oriented and useless
  on this hardware.
- **Audit correction:** an earlier status summary in this project's
  working history stated 552/553 tests passing; the actual count was
  551/552 (537 baseline + 15 new, not 553) — an arithmetic error caught
  by re-running the suite rather than trusting the prior figure.

## Phase 12 — GGUF path exercised on real hardware, and a pre-flight audit (Sep 28)

- **Two real bugs found only by actually running `GGUFRationaleScorer` on
  the student's laptop** — neither was caught by the fake-based unit
  tests, because the fakes didn't model the specific behaviour that
  broke: (1) the `Llama` constructor's `logits_all=False` default made
  `create_completion(logprobs=...)` raise outright in the installed
  `llama-cpp-python` version — perplexity computation cannot work
  without it, so this was a hard blocker, not a degradation; (2)
  Llama-3's chat template already emits a leading `<|begin_of_text|>`,
  and llama.cpp adds its own by default, producing a real "duplicate
  leading" warning and a prompt that wasn't what it was supposed to be.
  Both fixed; the fakes were then rewritten to mirror the real chat
  template shape so this specific class of bug would be caught next time
  without needing a real run.
- **The textbook retrieval index (50,000 passages) took ~1.8-2 hours to
  encode on CPU** — measured twice for real (~6624-7518s), not estimated.
  Every retry of an unrelated later failure (the two bugs above included)
  was redoing this from scratch. Added a cache keyed on
  `(n_textbook_passages, seed)`, validated against a freshly (and
  cheaply) reloaded passage-id list before being trusted, so a mismatched
  or stale cache rebuilds rather than silently reusing the wrong vectors.
- **Calibration run, for real** (10 pairs, `--scorer gguf`, CPU-only, 256
  max tokens): 1092.4s, 109.24s/pair, extrapolating to **~15.2 hours**
  for the planned 500-question real run. Recorded here as the actual
  measured figure this project is committing to, not a preliminary
  guess.
- **Pre-flight audit before committing to that 15-hour run** surfaced two
  further gaps, both fixed before any real-scale run started: (1)
  `load_checkpoint` had no handling for a truncated trailing line in the
  checkpoint file — a real failure mode for a multi-hour unattended run
  (power loss, a closed laptop lid), and would have crashed a resume
  attempt instead of recovering; fixed to drop an incomplete last line
  and re-score that one pair, while still raising loudly on damage
  anywhere else in the file (not explainable by an interrupted write).
  (2) the resume fingerprint checked `scorer_name`/`n_questions`/`seed`/
  `n_textbook_passages` but not `max_new_tokens` or which model file/
  revision was in use - a resume with an accidentally different flag on
  any of those would have silently mixed incompatible pairs into one
  label set rather than being caught. Both now covered, with tests.

## Phase 13 — Real label generation completed; filter-training tooling fixed (Sep 30)

**Labels.** The student's local GGUF run finished: 500 questions x 2
generations (no evidence / with evidence), 256 max new tokens, 59,618 s
(16.6 h vs the 15.2 h calibration estimate). Distribution: 143 HELPFUL / 357
NOT_HELPFUL (28.6% / 71.4%). Flips: flip_to_wrong (97) > flip_to_correct
(56) - retrieved evidence hurt Llama-3-8B-Q4 more often than it helped, which
is what makes a filter worthwhile but also makes the classes imbalanced.
The labels file embeds textbook passages; back it up outside git.

**train.py audit (real torch, tiny T5, before any real training).**
(1) Validation metric bug confirmed: old metric 0.0 vs deployed-rule 0.79 on
the same model (T5 `generate()` starts with the decoder-start id). Replaced by
the deployed two-way rule at decoder position 0, plus balanced accuracy and
majority baseline. (2) `is_usable` now requires beating the majority baseline
and balanced accuracy > 0.5. (3) Added Adafactor / gradient checkpointing /
CPU / fp32 switches (all reported as deviations), `--calibrate-steps`,
fingerprinted `--resume`, best-epoch-by-balanced-accuracy, optional early
stopping, label single-token check. (4) A kill mid-save left a partial
`checkpoint-N` that HF's `get_last_checkpoint` selected and then crashed on;
resume now skips incomplete checkpoints (test-proven with a real kill).
Not yet done: training on real Flan-T5-large.

## Current status (as of this log)

- **Methodology, data pipeline, and software infrastructure:** complete,
  tested (599/600 tests passing — the one failure is the same expected
  fixture-vs-real-corpus registry comparison as always, not a defect),
  unchanged in substance since the Phase 4 scope freeze.
- **Corpus:** built and frozen (Phase 6); the guideline/textbook source
  (Stage 03) remains empty as a known, non-blocking gap.
- **Question pool:** 123 questions, fully human-reviewed, split into 23
  validation / 90 test questions (Phase 7).
- **Retrieval index:** complete — full corpus, 4,376,141 × 768 (Phase 11).
- **RAG² baseline filter checkpoint:** not yet produced. Label-generation
  tooling for a fully local (GGUF-based) path exists, is bug-fixed
  against a real run, and has a measured throughput (~15.2h/500
  questions) — the real-scale run itself has not been started yet
  (Phase 11-12). Local filter training (`train.py`'s Adafactor/gradient-
  checkpointing change) remains unbuilt.
- **Main-evaluation generator:** still the extractive stand-in.
  `fit_and_evaluate.py` has no wiring for a real generator yet — this is
  unimplemented, not merely unrun.
- **Reported thesis result:** not yet produced. The only real-data result
  committed so far is the Phase 9 pilot run, which is explicitly a
  pipeline validation, not the comparison the thesis will report.
