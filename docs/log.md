# Research Log

A chronological record of the implementation and research work performed on
this repository's `main` branch, from project start through the current
repository layout: decisions, progress, changes, problems and how the work stayed
aligned with the research objectives. Reconstructed from the commit history for
documentation purposes. Each entry states what changed and why; it does not
restate full technical detail already covered in `methodology.md`, `data.md`,
`protocol.md`, `evaluation.md` and `reproducibility.md` — see those for the
current, authoritative description of the study. This log is the history of how
the project got there. Entries name files, documents and commands as they were
called when the entry was written: `experiment_plan.md` and `experimentation.md`
became `protocol.md` and `evaluation.md` on 2026-10-08 (Phase 42), and the code of
the earlier stages is in Git history at commit `f721bbb`. Generated result files
keep the text they were generated with, so some cite documents by the older names.

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

## Phase 14 — First real filter-training run: negative result (Sep 30)

Run 1 (local CPU, Adafactor, fp32, gradient checkpointing, 450 train / 50
val, 20 epochs planned, early-stopping patience 4 on balanced accuracy):
calibration 106 s/optimizer step (~50 min/epoch incl. eval). It stopped after
6 epochs (5 h 1 min). Result: **no usable filter.** Deployed-rule validation:
accuracy 0.700 = majority baseline 0.700, balanced accuracy 0.538 (2 of 15
HELPFUL recalled), eval_loss 0.320 vs a prior-only floor of 0.306 - i.e. the
model learned the class prior (71% NOT_HELPFUL) and nothing else. The
restored "best" epoch (2) is noise on ~15 validation positives. The record
correctly says `usable = False`; this checkpoint must not be used.

Causes not yet separated: (a) weak/noisy labels (347 of 500 labels come from
the perplexity-percentile tie-break; the HELPFUL class is ~61% "top-25% of
perplexity reduction" by construction; 4-bit labeller, 256-token rationales),
(b) too little data (450 vs the paper's far larger set), (c) early stopping
too eager - balanced accuracy is constant 0.5 while the model predicts only
the majority class, so it cannot signal that learning is about to start (my
design flaw). Changes made: `--best-metric` (default eval_loss),
prior-only-floor report, `diagnose_labels.py` (labels by rule; TF-IDF
learnability probe). The 6-epoch run is kept as a documented negative result.

## Phase 15 — Why the filter did not learn: label diagnostics (Sep 30)

Read-only diagnostics on the 500 labels (`diagnose_labels.py`): rationale
answer correct 334/500 without evidence vs 293/500 with retrieved evidence;
TF-IDF+LR 5-fold balanced accuracy 0.529 (all labels) and 0.547 (153 flip
labels) - no lexical signal; MedCPT relevance vs label AUC 0.519 (all), 0.429
(flips), 0.568 (tie-break); Spearman(relevance, perplexity reduction) +0.095.
Caveat: retrieved passages are all top-1 (relevance sd 2.1), so this probe has
little range. Working hypothesis, NOT yet tested: flips reflect prompt
sensitivity of greedy 4-bit decoding rather than evidence content. Test added:
`label_noise_control.py` re-scores a seeded 150-question subset with an
irrelevant (derangement-assigned) passage and compares flip rates; it writes
no labels. No further filter training until that result is in.

## Phase 16 — Label-noise control result; answer-extraction audit added (Oct 1)

`label_noise_control.py`, n=150 questions, same generator. Accuracy: no
evidence 0.700, retrieved passage 0.600, irrelevant passage 0.553. Flips:
retrieved 17 to-correct / 32 to-wrong (rate 32.7%); irrelevant 11 / 33 (rate
29.3%). An irrelevant passage flips the labeller about as often as the
retrieved one and lowers accuracy at least as much; the only visible content
effect is 6 more flips to correct for retrieved passages (17 vs 11), not
distinguishable from noise at n=150. So most correctness flips here reflect
the generator's sensitivity to the prompt, not whether the passage helps -
the label function is mostly noise at this scale, consistent with the
filter learning only the class prior.
Candidate mechanism (untested): `correct` is False when no letter is
extracted, and `extract_answer_letter` takes the last standalone A-D token
(including the article "A"), so rationales truncated at 256 tokens before
their "Answer:" line are scored wrong or get a stray letter - and passages
lengthen rationales. `rationale_audit.py` measures truncation rate, explicit
"Answer:" rate, and greedy reproducibility on a seeded sample.

## Phase 17 — Answer-extraction audit result (Oct 1)

`rationale_audit.py`, 40 seeded questions, same generator: generations cut
off at 256 tokens 27.5% without evidence / 25.0% with the retrieved passage;
explicit "Answer: X" line 72.5% / 75.0%; scored letter a fallback or
disagreeing with the explicit line 27.5% / 25.0%; no letter at all 7.5% /
5.0%; greedy regeneration matched the original run 100%. The earlier
hypothesis that passages lengthen rationales (more truncation with evidence)
is REFUTED (rates are equal). What the audit does show: about a quarter of
all generations are scored from a guessed letter, which injects arbitrary
correct/wrong bits into the labels regardless of evidence. Next:
`--analyze-only` for how many observed flips involve a guessed generation,
then a repaired-measurement pilot (larger `--max-new-tokens`, explicit
"Answer:" only, with the irrelevant-passage control) before any relabelling.
Any such change is a disclosed deviation.

**Phase 17 follow-up (analysis of the 40-question audit file).** 16 of 40
pairs flipped under lenient extraction; 9 of those 16 (56%) involved a guessed
(fallback) generation. Only 25 of 40 pairs had an explicit "Answer:" line in
both generations; among them 3 flipped to correct and 4 to wrong (28%, wide
interval at n=25). Measurement artifacts explain over half the flips here,
but clean pairs still flip. **Repaired-measurement pilot - decision rule fixed
BEFORE seeing results:** 80 seeded questions, `--max-new-tokens 768`,
`--with-control`; analysed on matched triples (explicit answer in all three
conditions) with strict scoring. Labels are treated as repairable only if
truncation is < 5% AND retrieved-passage accuracy exceeds irrelevant-passage
accuracy by >= 10 percentage points on the matched triples. Low power (about
50-70 triples) - a negative result will be reported as "no detectable
evidence signal", not as proof of none.

## Phase 18 — Repaired-measurement pilot: pre-stated rule NOT met (Oct 1)

`rationale_audit.py --n 80 --max-new-tokens 768 --with-control`, strict
scoring on matched triples (77 of 80 pairs with an explicit answer in all
three conditions). Measurement artifact fixed: truncation 0% / 1.25% / 0%
(without / retrieved / irrelevant); 0 of the 16 retrieved-vs-none flips
involve a guessed generation. Accuracy: no evidence 0.753, retrieved passage
0.675, irrelevant passage 0.675. Flips: retrieved 5 to-correct / 11 to-wrong;
irrelevant 8 / 14. Pre-stated rule (truncation < 5% AND retrieved beats
irrelevant by >= 10 pp): truncation PASS, accuracy gap 0 pp FAIL.
Conclusion: with the measurement repaired, the retrieved top-1 passage is
statistically indistinguishable from an irrelevant one for this labeller
(Llama-3-8B-Instruct Q4_K_M, textbook-substitute corpus, MedQA); any added
passage lowers accuracy ~8 pp. Correctness flips therefore do not measure
passage helpfulness here. Limits: n=77 triples (cannot detect effects much
under ~10 pp); the perplexity tie-break (69% of labels) was not separately
controlled - the audit did not store perplexities. Decision: no relabelling and
no further filter training on this label pipeline; the RAG² baseline arm needs a
supervisor-level decision (see the options in the session notes).

## Phase 19 — Publishability review of the evaluation design (Oct 1)

Findings (checked against the repo, the source datasets, and the authors'
public repository; nothing here is a test-split result):
1. **RAG² release.** The authors' README states the trained filter checkpoint
   "is not available for distribution"; only a 5-example sample of the
   labelled data ships (`5%-train.json`, ids up to `llama3_5%_23600`, which
   implies a label set far larger than our 450). Reproducing their scale
   locally would take ~30 days of label generation at the measured 109 s/pair.
2. **Circular primary metric.** Currency = mean T(s) of admitted evidence, and
   `fit_and_evaluate.py` selects lambda/theta/H to maximise currency gain on
   validation. The Temporal Filter therefore wins currency by construction; it
   shows the mechanism acts as designed, not that answers improve. It must be a
   manipulation check, not the primary outcome or the fitting objective.
3. **Few real answer changes.** Only 5 of the 113 usable questions appear in
   MedChangeQA (verdicts that changed between review versions; exact-text
   match, so a lower bound). `temporal_candidate` (review revised >= once: 54 of
   90 test questions) does not imply the verdict changed, so for most of the
   subgroup older and newer evidence would agree and recency cannot help.
4. **Power.** Exact two-sided sign test needs a >= 65% win rate among 54
   non-tied pairs (>= 69% with Holm over 3 comparisons); generators tie on many
   questions, so effective n is smaller. Only large effects are detectable.
5. **Novelty.** The admission score is a recency prior interpolated with
   relevance, close to published time-aware retrieval (e.g. TempRALM, Gade &
   Jetcheva). The defensible contribution is the domain, the leakage-safe
   protocol, and an honest finding.
6. **Source-data headroom.** MedRevQA holds 281 dementia/Alzheimer's-related
   questions (201 from revised reviews) versus 113 usable in the pool.

Proposed (NOT yet implemented or approved): make answer-level verdict accuracy
vs the latest Cochrane verdict the primary outcome and fitting objective (with
a validated judge), keep currency as a manipulation check, add generator
sensitivity controls (no / irrelevant / admitted evidence), add comparators
(recency-only, date-window, zero-shot Flan-T5 filter named as such), enlarge the
pool toward verdict-changed questions, and fix the analysis plan before any
test-split run. Contact the RAG² authors to request checkpoint/labels.

## Phase 20 — Next-phase plan: evidence of real improvement (Oct 1) - PROPOSAL

No experiment was run. `docs/next_phase_plan.md` (renamed `docs/experiment_plan.md` in Phase 23)
records: what a
faithful RAG² baseline needs (only the checkpoint, or the authors' filter
decisions on our frozen candidates; their README states the checkpoint is "not
available for distribution", only 5 label examples ship, ids reach 23,600);
measured costs if artifacts arrive (inference 1.34 s/pair; training ~43 h/epoch
at 23.6k labels, so 40 epochs ~72 days); a tiered author request; and the fallback design: verdict
accuracy vs the latest Cochrane verdict as primary outcome (judge sees only the
answer), blinded human hallucination on a subset, shuffled-date and
irrelevant-evidence controls, pre-declared settings instead of fitting, decision
gates G0-G3 with thresholds fixed beforehand, and a power table (a 90-question
test set detects only ~17-20 pp differences; 250 questions ~10-11 pp). GitHub
issue replies (#1, #2 of dmis-lab/RAG2) could not be read here and are unverified.

## Phase 21 — Scope expanded to MedChangeQA (primary) with Alzheimer's as case study (Oct 1)

Inspection before committing to the expanded scope (all measured):
* MedChangeQA rebuilt from the release (`AllStudyGroups.csv` + `MedRevQA.csv`):
  512/512 items identical, so both review versions' dates and PMIDs are known
  for every item. Gold labels are gpt-4o-mini labels of abstract conclusions.
* 8 changed pairs with near-identical conclusions excluded as label noise; 397 of
  504 usable changes involve NOT ENOUGH INFORMATION; 114 are decisive flips.
* Without retrieval, five released models (7B to DeepSeek-V3) give the current
  verdict on changed items 49-50% of the time and the outdated one 25-32%.
* Alzheimer's items in MedChange: 9 changed, 5 unchanged.
* The current Alzheimer's evaluation is time-inconsistent: 55/99 verdict-labelled
  questions cite pre-2010 reviews, the corpus is 71% post-2020, and t_q was the
  run date for every question.
* Recency-aware RAG is an active area (TempRALM, AionRAG, FRESCO, ConflictRAG,
  DriftMedQA, arXiv 2511.06668): no novelty is claimed for the time-decay term.

Built (no network, no model): `experiments/medchange/` - benchmark builder that
verifies itself against the release, headroom script over the authors' released
answers, and the G0 PubMed as-of availability probe (to run locally; E-utilities
are blocked in the cloud session). 13 unit tests. Design, gates and thresholds: the plan
(then `docs/next_phase_plan.md`, now `docs/experiment_plan.md`). Nothing generated yet.

## Phase 22 — MedChange pipeline built; first measurements on the student's machine (Oct 2)

Built in the cloud session in `experiments/medchange/` and run on the student's laptop (E-utilities and
the models are not available in the cloud session). Nothing here is an accuracy result.

1. **G0 probe, first run.** It aborted with "a record postdates the cutoff": PubMed's publication-date
   filter matches the print OR the electronic date, so an article e-published before the cutoff but dated to
   a later print issue passes the search. It was available in time. The probe now computes each record's
   earliest availability from `pubdate` and `epubdate` as bounds, keeps only records proven to precede the
   cutoff (uncertain month-boundary records are dropped, never admitted) and aborts only if more than 15% of
   a result set cannot be shown to precede it. The rebuilt manifest also differed on Windows
   (` M manifest.json`), most likely CRLF conversion of the input CSVs (hashes) and of the manifest itself;
   hashes now normalise CRLF and the manifest is written with LF. Five tests added (18 in total).
2. **G0 on five items, then a crash.** 5/5 changed items had trials or reviews inside the update window
   among the as-of candidates (median 137 in-window records in the top 200). The full dev run then
   crashed on a PubMed date string the parser did not handle (season and range forms). `date_bounds` now
   never raises: season words, year-wrapping ranges ("Dec-Jan") and day ranges are widened to bounds that
   can only make a record look later, so the worst case is a dropped record. Tests for these forms added.
3. **G0 passed on the full dev split** (151 usable changed + 75 unchanged items). Changed: 94.0% have a
   trial or systematic review published inside the update window among the as-of PubMed candidates (top 200;
   80.1% within the top 50), median 72 in-window records; unchanged: 93.3% / 65.3%, median 50. The
   pre-stated threshold (>= 50%) was met. Caveats: (a) availability is necessary, not sufficient - the
   unchanged controls look the same, so G0 does not show that the new evidence carries the verdict change;
   that is what G2/G3 and the error analysis test; (b) these are PubMed lexical best-match candidates before
   MedCPT reranking, abstracts only. The ` M` on `manifest.json` was line-ending noise (`git diff` showed no
   content change; the splits were identical). Added `freeze_candidates.py` (abstracts via efetch, MedCPT
   dense rank and cross-encoder rerank, a frozen pool of 20 with date bounds and an order-sensitive hash;
   no outcome is read) and 4 tests (22 in total).
4. **Dev candidate pools frozen:** 226 items, 0 empty, median pool size 20 (37,375 abstracts fetched; about
   3 h on CPU). The generation and analysis harness was added before any answer existed: `arms.py`
   (B0/B1/B2/B3/P/C1 admission rules with fixed settings), `prompts.py` (an explicit `VERDICT:` line, parsed
   deterministically; the earlier plan's judge model is dropped for the primary outcome), `helpfulness.py`
   (zero-shot Flan-T5 P(yes); NOT RAG², and rationale-as-query is dropped so that all arms share one pool),
   `generate_answers.py` (llama.cpp, resumable, arms interleaved) and `analyze.py` (per-arm accuracy, exact
   McNemar, bootstrap CI, gates G2/G3). Settings and gate definitions were fixed in the plan (section 13)
   before any generation. 18 tests added (40 in `experiments/medchange` in total).
5. **Timing measured.** Zero-shot helpfulness scores computed for all 226 dev items (27 s per item, 1.33 s
   per pair, as predicted). First real generations (3 changed dev items x B0 and B1, Llama-3-8B Q4_K_M, CPU,
   greedy, 160 tokens): 18 s per answer without evidence and 66.5 s with five passages; 6/6 answers parsed.
   One no-evidence arm plus five evidence arms is about 350 s per item: dev (226 items) about 22 h,
   confirmatory (528 items) about 51 h, about 73 h in total, against an earlier estimate of 75-150 h. (The
   first version of this entry said ~52 h and ~74 h, from a rounded ~530-item split; 528 x 350 s is 51.3 h.)
   n = 3 says nothing about accuracy. Added `consistency.py` (G1 human check: a seeded sample of 50 answers,
   arm hidden, Y/N consistency; pass = parse >= 95% and consistency >= 90%).

## Phase 23 — Repository audit and reorganisation (Oct 2)

An audit of the GitHub repository and the working clone before any change: every tracked file; the branch
list; the import graph (to establish what is active and what is obsolete rather than assume it);
`pyproject.toml`; `.gitignore` behaviour (`git check-ignore`); the test suite and what it touches on disk;
line endings; and every path and `python -m` command named in the documentation. GitHub holds one branch,
`main`. The working clone's `main` was at the same commit; the clone also held two local branches from
earlier sessions, both fully merged into `main`, whose remote counterparts no longer exist.

Confirmed and corrected:

* **Tests read machine-local state.** Five test modules copied the whole `corpus/` directory - on a
  machine holding the real corpus that includes the multi-GB `data/` - and two of them compared a fixture
  run with the real corpus; one such comparison failed permanently and was recorded as "expected" (603/604)
  in the previous version of this log's status block. Tests now copy a lean scaffold
  (`evaluation/tests/corpus_scaffold.py`: `data/` contributes only its empty directories and one fixture
  file; a test checks that local data is never copied), and the fixture-versus-real-corpus comparisons
  became determinism tests (two independent scaffolds give identical outputs). The active suite has no
  standing failure and does not depend on whether the real corpus is present. A before/after snapshot
  of the whole tree then showed that one more test class still wrote into the real tree: the
  `iter_chunks` tests overwrote stage 06's resume marker `corpus/data/chunks/.chunk_progress.json`
  (a genuine interrupted run resumes from it). Stage-script output paths are now redirected to a
  temporary directory in the test helper, and `python -m evaluation.tests.check_hermetic` repeats the
  snapshot check on demand (it fails on the old test version and passes on the new one).
* **Abandoned code sat among active code.** The RAG² filter-reproduction package
  (`experiments/baseline/filter_training/`) and the superseded Alzheimer's v1 runners and results
  (`experiments/shared/runners/fit_and_evaluate.py`, `run_real_evaluation.py`, `experiments/results/`
  pilot outputs, `_archive/superseded_outputs/`), with their 11 test files, were moved with `git mv`
  (history preserved) to `_archive/rag2_filter_reproduction/` and `_archive/alzheimers_pilot_v1/`. Imports
  were rewritten; the 125 archived tests still pass (`python -m unittest discover -s _archive -t .`); a guard
  test checks that no active module imports `_archive`.
* **The admission formula was implemented twice** (`src/proposed/` and `experiments/medchange/arms.py`).
  `arms.py` now calls `TemporalPolicy` and `AdmissionScorer`; the arm settings and their hash are unchanged.
* **`analyze.py` lacked two things the plan states:** the Holm correction over the confirmatory family and
  the retrieval-level manipulation checks. Both implemented (`confirmatory_family`, `retrieval_metrics`)
  with tests. `--out` added because PowerShell's `>` redirection writes UTF-16.
* **Packaging and ignore rules.** `pyproject.toml` described the earlier project and did not package
  `experiments.medchange`; rewritten (version 0.2.0, extras `models` and `medchange`). It also omitted
  `requests` and `pypdf`, which `corpus/scripts` imports at module level: with both absent, 90 tests
  errored on import (the student's machine had both installed, so this was invisible there). Both are now
  base dependencies. `.gitignore`
  rewritten and checked with `git check-ignore`: a first draft used trailing comments, which git does not
  support, so the 4.6 GB `models/` directory was not ignored. `.gitattributes` added (LF in the
  repository and in checkouts; the four CRLF CSVs are left byte-identical).
* **Documentation described the abandoned design.** README, methodology, data, evaluation, reproducibility,
  glossary, the plan (renamed `docs/experiment_plan.md`), `_archive/README.md` and the two results READMEs
  were rewritten against the code, and `experiments/medchange/README.md` added. Statements that the code
  did not support were removed, e.g. that currency and a paired sign test are in active code (they exist
  only in the archived runner); the plan's statement that Holm was built was made true by implementing it
  in `analyze.py`. A `DocumentationIntegrityTests` class now fails if a path or `python -m` command named in
  a current document does not exist, and `test_scope_invariants.py` guards the README's objectives, its
  no-novelty statement and its "not a RAG² reproduction" statement.
* **Answers were not bound to the generator that made them.** Records carried the arm-settings hash and a
  prompt hash but nothing about the model file, context size, token limit or seed, and `generate_answers`
  appended to an existing answers file under whatever settings were current, so a 22-hour run resumed with
  another GGUF file or `--max-new-tokens` would have mixed configurations undetectably. It now writes
  `answers_<split>.config.json` (model SHA-256, decoding settings, hashes of the arm settings and prompts;
  thread count, GPU layers and the llama-cpp-python version for information), refuses to extend a file
  under a different model or result-relevant setting before loading the model, adopts pre-existing
  answers with a note, and `analyze` copies the record into its saved report. Ten tests added.
* **Feasibility arithmetic made explicit.** The plan's power claim and the dev gates were re-derived by
  simulation (exact McNemar; ≈ 30% of answers differing between arms is an assumption until dev results
  exist). The confirmatory split (353 changed items) has 84% power at Holm-corrected α for a true 10 pp
  difference, 61% for 8 pp and 33% for 6 pp. Gate G3 on 151 changed dev items has a standard error of
  ≈ 4.5 pp: the P − B2 ≥ +5 pp condition is met in 13% of runs with no true effect, 50% with a true +5 pp
  effect and 87% with +10 pp, so G3 is a coarse screen, not a test (plan §7 and §9; no threshold changed).
* **Dead code and dangling references.** The `Retriever`/`Reranker` interfaces in `src/common/retriever.py`
  (self-declared superseded, referenced nowhere in code, tests or documents) were archived as
  `_archive/retriever_interfaces.py`; two unreferenced helpers (`helpfulness.attach`,
  `retrieval.corpus.iter_texts`) were removed. Six comments and docstrings that cited paths from before
  the 2026-09-24 reorganisation (`systems/...`, `docs/next_phase_plan.md`) were corrected, and a test now
  checks every file path cited in a code comment or docstring.
* **Smaller.** Test files opened files without closing them (ResourceWarnings); a docstring cited a
  nonexistent `docs/research_log.md`; comments and CUDA error messages referred to Colab/Kaggle notebooks and
  `!nvidia-smi` (notebook syntax) although the project runs locally.

Deliberately not changed: the Alzheimer's corpus, question pool and index; the MedChange protocol and
settings (fixed before any answer existed); the history in this log.

Verification after the changes: 569 active tests and 125 archived tests pass, also in a fresh virtualenv
holding only the base dependencies with outbound socket connections blocked; both suites leave the
repository tree unchanged (`check_hermetic`); `pyflakes` reports only an
intentional availability import in `encoders.py` and unused imports inside archived code; every path and
command named in the current documents exists. No experiment was run and no result was produced in this
phase.

## Current status (2026-10-03)

* **Research direction:** two stages on MedChangeQA, asked as of each newest Cochrane review's date. Stage 1,
  the Temporal Filter (recency in evidence admission), is done on the dev split and negative (Phase 25).
  Stage 2, an evidence-synthesis layer (one stance judgement per retrieved paper, combined with the RAG
  answer by a small logistic regression fitted on dev and frozen), is pre-specified in `experiment_plan.md`
  with two questions: RQ1, does as-of retrieval (B1) beat no evidence (B0); RQ2, does the selected hybrid beat
  the same RAG answer put through the same fitting (B1R). The Alzheimer's study stays a secondary case study.
* **Built and tested:** the MedChange benchmark (504 usable changed and 250 unchanged items, seeded
  dev/confirmatory splits), the PubMed as-of probe, frozen dev candidate pools (226 items), zero-shot
  helpfulness scores for the dev pools, the six stage-1 arms, the generation harness, the analysis (accuracy,
  retrieval-level checks, McNemar with Holm, gates), the G1 consistency check and the dev error analysis.
  Stage 2 (Phases 27 and 29): `diagnostics.py`, `stance.py`, `stance_check.py`, `synthesis.py`, `analyze_stage2.py`,
  `label_audit.py`, `consistency_auto.py`, `findings.py`, `pipeline.py`, unit-tested on synthetic data (the pilot and
  diagnostics have also run on real data). Alzheimer's corpus (114,256 PMC records, 4,377,041 chunks), question
  pool (113 usable) and dense index (4,376,141 x 768): built, secondary.
* **Gates (updated through Phase 29):** stage 1: G0 passed; G2 passed; **G3 failed** (P − B2 = −1.3 pp, P − C1 =
  −2.0 pp); G1: parse rate 100% on dev, the human consistency sheet retired. Stage 2: P0 diagnostics done; gate 1
  (stance pilot, machine checks) **passed**; gate 2 (full dev stance with the fitted layer) pending.
* **Results:** dev split only, all six stage-1 arms (Phases 24-25): standard RAG (B1) is the most accurate arm
  on changed items (49.7%); the proposed Temporal Filter P is 13.9 pp lower (p = 0.0008). No stage-2 result
  exists. The confirmatory split (528 items) has not been built, run or analysed; it is run once, after the
  frozen model file is committed.
* **Not built:** frozen pools for the confirmatory split; the full dev stance output; the automatic faithfulness
  proxies; a second generator; the automatic Alzheimer's case study. The human hallucination annotation and all
  human labelling steps are dropped (Phase 29).
* **Abandoned and archived:** the RAG² filter reproduction (the checkpoint is not distributed; local
  retraining learned only the class prior) and the first Alzheimer's pilot runners (circular primary metric).
* **Known limitations of the design** (see `methodology.md`): the B2/P helpfulness score is an untrained
  stand-in, not RAG², and may have been scored on truncated inputs (unverified; the P0 diagnostics measure
  it); gold verdicts are model-generated; the generator is a 4-bit 8B model on CPU; the stance step is a
  zero-shot judgement checked only by a 40-paper hand check by a non-expert; the decisive-flip subgroup is
  small (114 items); the 528 confirmatory items can confirm only effects of about 5 pp or more.

## Phase 24 — Dev run of B0 and B1 (Oct 2)

Run on the student's laptop with the pre-audit harness (commit bb89609; no generator-configuration
record, so this answers file is adopted with a note when extended). 446 new answers plus the 6 timing
answers = 452 (226 items x B0, B1); 0 unparsed. Mean time per answer 14.6 s (B0) and 60.9 s (B1), faster
than the earlier estimate (about 320 s per item over six arms: dev about 20 h, confirmatory about 47 h).

| | Changed (n = 151) | Unchanged (n = 75) |
|---|---|---|
| B0 accuracy | 41.1% | 53.3% |
| B1 accuracy | 49.7% | 60.0% |
| B1 − B0 | +8.6 pp, 95% CI [0.7, 16.6], 27 vs 14 discordant, exact McNemar p = 0.060 | +6.7 pp, CI [−6.7, 20.0], p = 0.42 |

Outdated-verdict rate on changed items: 36.4% (B0), 31.8% (B1). Gate G2 passed: B1 changed the verdict of
35.8% of items (threshold 20%). Reading: retrieval of as-of evidence appears to help (the lower CI bound is
barely above zero and the test is not significant at 0.05), so the generator does use evidence; this is a
dev comparison without correction and says nothing yet about the proposed system. Next: G1 (the student
fills `consistency_sheet.csv`, then `consistency score`), then B2, B3, P, C1 on dev for G3.

## Phase 25 — Dev run of all six arms; gate G3 failed (Oct 2-3)

904 further answers (B2, B3, P, C1 x 226 items) on the student's laptop with the current harness
(`answers_dev.config.json`: Llama-3-8B-Instruct Q4_K_M, SHA-256 8ba9baf3..., n_ctx 4096, 160 tokens, greedy,
seed 42, llama-cpp-python 0.3.35); 0 unparsed answers. Committed in `experiments/medchange/results/`.

Accuracy on changed items (n = 151), with exact McNemar against B1 (uncorrected unless stated):

| Arm | Accuracy | Outdated-verdict rate | vs B1 | Update-window share of admitted passages |
|---|---|---|---|---|
| B0 no evidence | 41.1% | 36.4% | | 0 |
| B1 cross-encoder | 49.7% | 31.8% | | 51.0% |
| B2 helpfulness | 37.1% | 38.4% | −12.6 pp, p = 0.001 | 51.7% |
| B3 cross-encoder + recency | 45.7% | 33.1% | −4.0 pp, p = 0.38 | 74.3% |
| P helpfulness + recency | 35.8% | 40.4% | −13.9 pp, p = 0.0008 (Holm 0.0024) | 74.7% |
| C1 P with shuffled dates | 37.7% | 39.1% | | |

P − B2 = −1.3 pp (p = 0.80); P − C1 = −2.0 pp (p = 0.65); P − B3 = −9.9 pp (p = 0.017, Holm 0.033). On
unchanged items all arms score 53-61% with no significant difference. **Gate G3 failed** (needs +5 pp and
+2.5 pp). G2 had passed (B1 changed 35.8% of verdicts relative to B0).

What this does and does not show. (1) The mechanism works as designed: recency raised the share of admitted
passages from the update window from about 51% to about 74% and cut mean passage age from 10.7 to 6.3
years. (2) It did not make answers more correct: adding recency to the helpfulness ranking changed nothing
(P about B2 about C1), and adding it to the cross-encoder ranking was slightly, not significantly, worse
(B3 below B1). (3) The zero-shot Flan-T5 helpfulness score is a worse selector than the cross-encoder: B2 shares
only 21% of its passages with B1 and loses 12.6 pp. The most damaging factor for P is therefore its relevance
component, not its recency component. (4) Dev has 151 changed items (standard error about 4.5 pp), so small
effects cannot be excluded, but a gain of the pre-stated size is not there. (5) Per the plan, the confirmatory
split is not run for P as designed. Any redesigned variant (for example recency on top of the cross-encoder)
would be a post-hoc change, must be labelled exploratory, and needs fresh data. Next: gate G1 (the student
fills `consistency_sheet.csv`), then error analysis on the existing answers (no new generation): is the
update-window evidence in the pool, is it admitted, does the generator follow it.

## Phase 26 — Error analysis of the dev answers (Oct 3)

`error_analysis.py` on the 151 changed dev items (output saved by the student in
`experiments/medchange/results/error_analysis_dev.md`).

* The frozen pool contained at least one update-window passage for 100% of changed items; retrieval misses = 0
  for every arm. Admission misses: B1 13, B3 1, P 2. Nearly every wrong answer (B1 63, B3 81, P 95) is in the
  group "update-window evidence admitted, answer still wrong". The bottleneck is therefore not retrieval or
  admission but what the generator does with the evidence (or the gold label itself).
* B1 accuracy 51.9% when it admitted an update-window passage (n = 131) vs 35.0% when it did not (n = 20); P
  admitted one in 149 of 151 items and scored 36.2%, so more window evidence is not the same as more useful
  evidence.
* Verdict behaviour: without evidence the model says SUPPORTED for 117 of 151 changed items (REFUTED 2, NEI 32).
  B1 says NEI 60 times, B3 72 times, P 49 times. 116 of the 151 changed items involve NEI, so part of B1's gain is
  plausibly more appropriate use of NEI rather than reading the newer evidence; this is an interpretation,
  not tested. On the 35 decisive flips every retrieval arm is at or below B0 (B0 48.6%, B1 45.7%, B3 34.3%,
  P 37.1%; n is small).
* Not checked: whether the gold labels (gpt-4o-mini, from abstract conclusions) are right.


## Phase 27 — Audit of the dev results, stage-2 design and code (Oct 3)

**Why.** Gate G3 had failed (Phase 25) and the error analysis (Phase 26) located the bottleneck in how the
generator uses evidence, not in retrieval. Before choosing a next step the dev results were audited once more
and a redesign was checked for feasibility; no repository file changed during the audit.

**Audit findings** (computed on the 226 dev items from the committed answers; no new generation; recomputed
by the committed `experiments/medchange/dev_audit.py`, output `results/dev_audit.md`).

* Always answering SUPPORTED scores 44.4% on changed items (the best arm, B1, scores 49.7%); recall of
  REFUTED is 0-11% for every arm. Accuracy here depends largely on how readily the model says SUPPORTED
  rather than NOT ENOUGH INFORMATION.
* B1's gain over B0 comes with a change in behaviour: NOT ENOUGH INFORMATION answers rise from 21% to 40% of
  answers (recall 15% to 53%) while SUPPORTED recall falls from 81% to 69%.
* Reading five abstracts at once is noisy: two arms that admitted the same five papers in a different order
  gave the same verdict in 86.1% of 72 pairs (14% flips); partial overlaps agree 80.1% (overlap 0.25-0.66,
  1,515 pairs) and 63.9% (overlap below 0.25, 656 pairs); identical lists in identical order gave identical
  generated text in only 8 of 17 pairs (the same verdict in all 17), so greedy decoding is not bitwise
  reproducible.
* Evidence age carries no visible signal: the mean age of B1's evidence is 10.6, 11.4 and 10.7 years for gold
  SUPPORTED, REFUTED and NOT ENOUGH INFORMATION; adding age features to a refit of B1's verdict lowers
  cross-validated accuracy (52.2% to 49.6%). Refitting the hard verdict alone (52.2% against 53.1% raw) and a
  majority vote of B1, B2 and B3 (47.0% against 49.7%) do not help.
* Correction recorded: these figures were first computed with one-off scripts that were not saved. The
  committed script reproduces every figure exactly except the two cross-validated accuracies, which moved by
  0.5 pp (52.7% to 52.2%, 50.1% to 49.6%) because it uses the stage-2 fitting; no conclusion changed.

**Decision (the researcher's).** Stage 2: an evidence-synthesis layer instead of further work on recency.
Two pre-specified questions: RQ1, does as-of retrieval (B1) beat no evidence (B0), a replication of the dev
difference (+8.6 pp, p = 0.06) and the most likely positive result; RQ2, does the selected hybrid beat B1 put
through the same fitting (B1R). Recency and study-type weights are ablations only. The protocol was written
and committed first (`experiment_plan.md`, commit 26984cf: arms, features, fixed settings, stage gates P0 / gate
1 / gate 2 with their operating characteristics, Holm families, data-hygiene rules, and the forking-path
ledger of eight decisions taken after seeing dev data), then the code, before any confirmatory pool, answer
or stance output existed. Judgement recorded there, not computed fact: RQ1 is confirmed with probability about
58%, RQ2 about 15% (range 8-25%); the 528 confirmatory items can confirm only effects of about 5 pp or more.

**Built** (unit-tested on synthetic data only; 649 active and 125 archived tests pass, also in a fresh
virtualenv holding only the four base dependencies with outbound socket connections blocked; the suites leave
the repository tree unchanged, `check_hermetic`; `pyflakes` reports only the intentional availability import
in `encoders.py`).
`experiments/medchange/diagnostics.py` (P0, no model); `stance.py` (one stance letter per paper for the first
eight pool candidates, two wordings, first-token probabilities, the irrelevant-paper control, resumable,
configuration recorded; Flan-T5-large as the one declared fallback); `stance_check.py` (gate 1: machine checks,
the 40-paper hand-check sheet, scoring and the wording choice); `synthesis.py` (four features, paper weights,
regularised multinomial logistic regression fitted on dev and frozen, repeated cross-validation, selection rule,
gate 2, prediction that refuses to run without the frozen model); `analyze_stage2.py` (RQ1 and RQ2 as one Holm
family, a secondary family, per-class recall); `dev_audit.py`. `generate_answers.check_config` was generalised so
that stance and synthesis files carry the same configuration record as answers.

**Not done, and not verified.** Nothing has been run on real data: the llama-cpp log-probability path was
written from the library source and has not met a real model (`--hard-labels` is the fallback); the cost per
stance judgement (expected 5-8 s) is unmeasured; the confirmatory split has no pools yet. A code-reading
suspicion, unconfirmed: `helpfulness.py` puts the question after the abstract and truncates at 512 tokens from
the end, so for long abstracts the question and the "Answer yes or no" line may have been cut off. This
leaves the stage-1 recency conclusions unchanged (the compared arms share the scores) but weakens the
statement that the helpfulness score is a weak selector; the P0 diagnostics measure it. The saved dev error
analysis (`error_analysis_dev.md`, Phase 26) is on the student's laptop and is not yet committed.

**Next** (`reproducibility.md` §3, steps 11-20): pull, P0 diagnostics, the 40-item stance pilot with
`stance_check report`, the researcher's hand check of 40 papers (gate 1), the full dev stance run, `synthesis
fit` (gate 2), then commit `results/synthesis_model.json` before any confirmatory preparation. G1 (the
consistency sheet) is still pending.

## Phase 28 — P0 diagnostics on the dev pools (Oct 3)

`diagnostics.py` run on the student's laptop (output `results/diagnostics_dev.md`, no model).

* **Helpfulness truncation confirmed.** 837 of 4,520 stage-1 helpfulness inputs (18.5%) exceed 512 tokens, so
  the question and the "Answer yes or no" line were cut off for them; 352 of the 1,130 papers B2 admitted
  (31.1%) were affected. Stage-1 conclusions about recency stand (the compared arms share the scores), but
  the statement that the zero-shot helpfulness score is a weak selector is confounded by this defect, and B2, P
  and C1 were partly scored on inputs that did not contain the question. Not fixed: stage 2 does not use these
  scores, and re-running stage 1 is not planned.
* **Stance inputs.** Only 36.4% of candidates have labelled RESULTS or CONCLUSIONS sections; the rest
  contribute their last three sentences (median 89 words, maximum 202). The pilot's hand check will show
  whether this is good enough.
* **Study types.** Systematic reviews or meta-analyses are 5.7% of the pool and appear in the top 8 for 41.6% of
  items (33.6% in B1's admitted five), above the 30% bar, so the study-type weight has material to act on. Gold
  REFUTED is about as frequent with and without a review in B1's evidence (16 of 76, 33 of 150): no visible
  signal (descriptive).

## Phase 29 — Pilot result, removal of every human-labelling step, and the automated pipeline (Oct 3-4)

**Pilot (run on the student's laptop, 40 dev items x 8 papers, both wordings, plus the irrelevant-paper control: 960
papers, 6.76 s per paper on average, 6 threads).** Wording agreement 85.3% (needs 80%), irrelevant control papers
rated "neither" 97.8% (needs 70%), mean 6.76 s (needs 10 s), invalid outputs 2.5% pooled with control papers: wording A
0.3% and B 0% on real papers, 7.2% on control papers. Real-paper class shares: supports 22%, contradicts 17%, no clear
stance 62%. The human hand check was never run (see below).

**Decision (the researcher's, 2026-10-04).** The researcher is not a medical expert, has no access to clinicians and will
not label, review or validate medical content; human involvement is limited to research decisions, approving
predefined changes, running commands and interpreting results. Every human judgement step was therefore audited
(read-only, whole repository including the archive) and removed or replaced: the stance hand check and the human
consistency sheet (retired), the planned ~100-label human check of the gold labels (replaced by an independent-model
label audit with a label-stable sensitivity analysis), the planned blinded human hallucination annotation (dropped),
and the human-reviewed Alzheimer's question pool (kept as history, not extended; case study redesigned to be
automatic). The existing Alzheimer's review turned out to be weaker than it looked: all 123 decisions are by the
researcher, the free-text notes were lost in an overwrite and the decisions restored from a split file, 23 of the
113 usable questions are flagged `pending_revision` and were never fixed. Clinician validation of the gold labels
cannot be automated and is a stated limitation. Amendments are in `experiment_plan.md` §13 (items 9-14).

**Checked and not adopted: changing the answer prompt to the labelling rubric.** The authors' labelling prompt tells
gpt-4o-mini to answer NOT ENOUGH INFORMATION only when too few studies were found, but their own answering prompt
(`Code/answer_questions.ipynb`) defines the labels as loosely as ours, so our prompt follows the benchmark's evaluation
protocol; all stage-2 comparisons are within one prompt and the fitted layer absorbs calibration. Changing it would have
required regenerating dev B1 and B0 (about 4.7 h) and broken comparability with stage 1. Kept as a stated limitation.

**Built (unit-tested on synthetic data; 681 active tests).** `stance` runs both wordings and `synthesis` averages them
(the stance wording is no longer chosen by a person); `stance_check` is machine-only with the invalid-output rule on real
papers; `analyze_stage2.decide` implements the pre-declared genuine-positive reading (also better than raw B1, macro-F1
not lower, label-stable direction, recency earned) and an RQ1-only mode for a failed gate 2; `label_audit.py`,
`consistency_auto.py`, `findings.py` (DEV_REPORT.md, FINDINGS.md) and `pipeline.py` (one resumable command per phase,
integrity checks, the frozen-model-on-origin/main guard, a same-generator check between dev and confirmatory runs,
optional commit and push). Nothing from the confirmatory split has been generated or analysed. Not yet run on real
data: the audits (need a second GGUF model), the dev stance run of both wordings and everything after it.

**Next:** `pipeline dev` (about 8.4 h), read `results/DEV_REPORT.md`, then `pipeline confirm --go` (about 38 h; about
22 h if gate 2 fails); commands in `reproducibility.md` §3.

## Phase 30 — Repository audit and cleanup (Oct 4)

Whole-repository audit (local working tree and `origin/main`, both at the same commit; only `main` exists on GitHub).
Changes: the retired human consistency check (`consistency.py`, never run) and its tests moved to
`_archive/medchange_human_checks/`; `experiments/medchange/results/README.md` rewritten (it had a broken sentence and
listed the committed diagnostics and error-analysis files as uncommitted); placeholder model paths in the run
commands replaced by the real file names, with the Qwen download command. First real run of `pipeline dev` on the
student's laptop passed all six integrity checks and stopped, as designed, because the audit model was not yet
downloaded. Verification: 679 active and 127 archived tests pass (also in a fresh virtualenv with only the base
dependencies and network blocked); the suites leave the tree unchanged; `pyflakes` reports only the intentional
import in `encoders.py`. No experiment was run and no result changed in this phase.

## Phase 31 — Can the base paper's results be matched? Result tables and figures (Oct 4)

**Feasibility of results comparable to the base paper (RAG², Sohn et al., NAACL 2025): no, for four reasons that are
facts, not assumptions.** (1) Different task: they report multiple-choice accuracy on MedQA, MedMCQA and MMLU-Med (their
Llama-3-8B-Instruct goes from 57.7 to 64.6 on MedQA); this thesis measures as-of verdict accuracy against model-generated
Cochrane labels that another model reproduces only 83.2% of the time (Phase 29), so the headroom is of a different kind.
(2) Their gain comes from a Flan-T5 filter trained on perplexity-based labels for MedQA/MedMCQA training questions, rationale
queries and balanced retrieval over a 564 GB index of four corpora, with training on one H100 GPU; the checkpoint is not
distributed and the local retraining attempt (archived, Phases 13-18) learned only the class prior. (3) The thesis runs on a
CPU laptop with a 4-bit 8B model. (4) The measured stage-1 and stage-2 results are negative: the untrained helpfulness stand-in
is 12.6 points below standard retrieval (and its inputs were truncated for 18.5% of papers), recency re-ranking is
not better, and the evidence-synthesis layer failed gate 2. What is comparable is the design of their Tables 2 and 3.

**Built.** `report.py` produces those tables from the committed results with nothing typed in: Table 1 (systems, in the
layout of their Table 2) adds the benchmark authors' released closed-book answers for five LLMs scored on the same
questions, which gives the comparison with existing work on identical inputs; Table 2 (one generator, different admission
methods, in the layout of their Table 3); per-class recall; the dev cross-validation of the layer; a LaTeX table; and four
figures (accuracy by system, per-class recall, the layer's cross-validation, mechanism against outcome). `pipeline`
runs it at the end of each phase; `pip install -e ".[report]"` adds matplotlib for the figures.

**What the dev tables show (exploratory, n = 226, intervals about ±6 to ±8 points).** Standard retrieval with the 8B model
(B1, 53.1% on all items) is within noise of the much larger closed-book models on the same questions (50.4% to 54.9%), and
8.6 points above the same model without retrieval on changed items (49.7% vs 41.1%); the temporal mechanism worked (update-window
share 51% to 74%) but did not raise accuracy; no stage-2 variant beat B1R. Confirmatory results do not exist yet; nothing
from the confirmatory split has been generated or analysed.


## Phase 32 — Confirmatory preflight relaxed for the RQ1-only run (Oct 4)

The confirmatory run built all 528 pools and then stopped at the preflight: one pool has fewer than 8 candidates. The
8-candidate minimum exists only for the stance step, which is skipped because gate 2 failed. `pipeline.py` now requires
8 only when stance will run and 5 (the B1 budget) otherwise; the short pool stays in the evaluation. Decided after seeing
pool sizes, before any confirmatory answer or label was generated or analysed; recorded in `experiment_plan.md` §13.

The short pool is MC-00372 (4 candidates), so the minimum for the RQ1-only run was lowered again from 5 to 1; the item
stays in the evaluation (see `experiment_plan.md` §13). Still decided before any confirmatory answer or label was generated.

## Phase 33 — Status snapshots removed from the documentation (Oct 5)

At the researcher's request (university requirement), the dated status snapshots, status rows and columns, and
"pending / not yet run" statements were removed from `docs/evaluation.md` (§7 is now the dev-split
results), `docs/methodology.md`, `docs/data.md`, `docs/experiment_plan.md`, `experiments/medchange/README.md` and
`experiments/medchange/results/README.md` (the root `README.md` was rewritten by the researcher separately). No result, rule, threshold, limitation or dev/exploratory label was
changed; §7 of `docs/evaluation.md` now also records the dev gate-2 outcome (FAIL) that the old snapshot predated.
The `done / built / planned` labels in the headings of `docs/experiment_plan.md` and in this log were kept: they
separate implemented parts of the protocol from unimplemented ones.

## Phase 34 — Confirmatory run completed; docs rebuilt (Oct 5)

`pipeline confirm --go` ran once on the 528 confirmatory items (answers B0 and B1, label audit, consistency check,
analysis, findings, report) and its results were pushed to `main` (commit 5ddd201). Pre-declared reading: RQ1 **not
confirmed** (B1 − B0 = +1.5 pp, 95% CI −2.8 to +6.1, Holm p = 0.554; changed items +3.1 pp, CI −2.5 to +8.5); RQ2 not
run because gate 2 failed on dev. The dev effect of +8.6 pp on changed items did not replicate. Label agreement 81.4%
(kappa 0.7161); consistency judge 97.7%. Details: `docs/evaluation.md` §8.

The `docs/` folder had been deleted from `main` by the researcher (commit 608f94b) while the run was in progress.
This folder was rebuilt from the repository history (`608f94b^`, plus the status-removal edits of Phase 33) with the
confirmatory results added; no protocol rule, threshold or earlier result was changed.

## Phase 35 — Diagnosis and realignment: adapted RAG² + evidence-criteria verification (Oct 5)

**Requirement.** The supervisor requires the proposed system to beat the selected baseline (an adapted RAG²
system) by at least 1 percentage point of verdict accuracy, genuinely and reproducibly.

**Diagnosis (computed from committed files; item-level inspection on dev only).** (1) The bottleneck is the
verdict decision: no dev error of any arm was a retrieval miss. (2) The answering prompt and the gold labels
define the classes differently: the benchmark's labelling rubric keeps NOT ENOUGH INFORMATION for "not enough
studies found" and calls placebo-like results REFUTED; B1's REFUTED recall was 10% on dev (5 of 49) and 8.7% on
the confirmatory aggregate; 44 of B1's 106 dev errors are gold-REFUTED items. (3) Indirect evidence (other
interventions, other outcomes, surrogate measures, non-randomised designs) is read as support (dev examples
MC-00127, MC-00272, MC-00354, MC-00036). (4) Date-based admission had nothing to act on: update-window evidence
exists for 94.0% of changed and 93.3% of unchanged dev items (93.8% / 92.6% confirmatory), and old trials are
not outdated. (5) The pipeline had no working RAG² baseline (B2 was an untrained, truncated stand-in). (6) With
528 questions only effects of about 4–6 pp can be confirmed; an observed +1 pp arises by chance 23–32% of the
time. (7) Recalibrating B1's verdict, per-paper stance and order voting had each shown little to gain on dev.

**Decision.** Realign to an adapted RAG² baseline (R2: rationale query, balanced retrieval across evidence
types, zero-shot LLM filter) and one proposed component, evidence-criteria verification (R2V), with a criteria
control (R2C), a temporal ablation (R2V-ND) and component ablations; MedChange stays the primary benchmark,
Alzheimer's a descriptive case study. The protocol, written before any realigned output, replaces the stage-2
protocol in `docs/experiment_plan.md` (the researcher had deleted the old file, commit aa2c62b; its text stays in
the history at commit 92e3aaf). The confirmatory split is reused once for the new comparison; the reuse and the
aggregates seen beforehand are disclosed in the plan's ledger.

**Built.** `rag2.py`, `rag2_run.py`, `analyze_rag2.py`, `rag2_pipeline.py` and 34 tests on synthetic data; the
README, methodology, evaluation, data, reproducibility and glossary describe the realigned study, and the README
guard test was retargeted to it. No realigned output exists yet; nothing was run on any real data in this phase.

## Phase 36 — Alzheimer's/dementia secondary test set (Oct 5)

The proposal named Alzheimer's disease as the domain; the main benchmark holds only 14 such items. Checked first:
MedRevQA has 212 distinct questions naming dementia, Alzheimer's disease, mild cognitive impairment or cognitive
decline from 163 Cochrane reviews outside the dev and confirmatory splits (computed; gold 89 NOT ENOUGH
INFORMATION, 69 REFUTED, 54 SUPPORTED; 206 from single-version reviews, so no changed verdicts). The main
benchmark was rebuilt in the cloud from a fresh MedChange clone and matched `manifest.json` exactly before the new
set was appended. Built: `ad_benchmark.py`, split `ad` in the existing tools, the `rag2_pipeline ad --go` phase
(same freeze guards as confirm), tests, and the dated amendment in `experiment_plan.md` §11 (made before the dev
run; only a three-question smoke test existed). It is a secondary, fresh held-out test run once after the freeze;
with 212 questions only effects of about 7–9 pp can be confirmed.

## Phase 37 — Evaluation and experiment plan merged into one document (Oct 5)

At the researcher's request, `docs/evaluation.md` was merged into `docs/experiment_plan.md`, now "Evaluation and
experiment plan". The plan's §1–§11 keep their numbers (the code and the other documents cite them); the former
evaluation sections moved to §6 (primary outcome, supporting outcomes, retrieval metrics), §7 (statistical
procedure, controls, abstention and coverage) and §12–§14 (error analysis, development-split results,
confirmatory results). No result, rule or threshold was changed. References in the README, `methodology.md` and
the documentation guard tests were updated; earlier entries of this log still name `evaluation.md` as it was then.

## Phase 38 — `experiment_plan.md` renamed `experimentation.md` (Oct 5)

At the researcher's request the merged protocol and evaluation document was renamed from
`docs/experiment_plan.md` to `docs/experimentation.md`; its content and section numbers are unchanged. Current
references (README, `methodology.md`, `glossary.md`, `reproducibility.md`, the medchange READMEs, the realigned
modules and the documentation guard tests) point to the new name. References to the old stage-1/stage-2 protocol
still name the file `experiment_plan.md`, because that is its name at commit 92e3aaf. Committed run outputs
(`results/FINDINGS.md`) keep the text they were generated with, which names the old file. Earlier entries of this
log use the names that were current when they were written.

## Phase 39 — Repository audit, reorganisation and corrections (Oct 5)

At the researcher's request the whole repository, local and GitHub, was audited before anything was changed.

**Inspected.** Git: local `main` and `origin/main` were identical (`fa4cb61`); GitHub holds one branch, seven pull
requests from September that are all closed and unmerged, and no tags. Structure and imports: an AST-based
analysis of every Python file, with third-party imports classified as module-level or lazy. Documents against code
and files: every named path, command, count and cross-reference. Packaging: `pyproject.toml` against what is
imported. Both test suites, including a fresh virtualenv with only numpy and sockets blocked. The two benchmark
manifests, rebuilt from the released MedChange files.

**Reorganised (nothing deleted; Git history keeps everything; recovery point: the commit `fa4cb61`).** The import
analysis showed that the realigned pipeline imports nothing of the Alzheimer's-specific framework of the first
design. That framework moved to `_archive/alzheimers_framework/` with its tests: the local corpus (33 tracked
files), the question pool, dense retrieval and index build, the three-arm runner with its freezing contract, the
annotation workflow, the RAG metrics, the hallucination-rate statistics (`evaluation/stats.py` was split: the
active file keeps the exact McNemar test, the paired bootstrap and Holm, now with their own `test_stats.py`; the
hallucination-rate accounting is `har_stats.py` in the archive), the RAG² baseline slot and the
threshold-and-budget admission.
Kept active: `src/common/evidence.py`, `evaluation/stats.py`, the MedCPT encoders (moved to
`experiments/medchange/encoders.py`, where the pipeline uses them) and the Temporal Filter formula (renamed
`src/proposed/` → `src/temporal_filter/`: it is a stage-1 result of record, no longer the proposed system). Test
accounting by class and method name: 730 active and 127 archived tests before, 280 and 592 after, none lost, 15
added by this audit. The package list in `pyproject.toml` was brought to the new layout and is now checked by a
test; the base dependencies were cut from four to numpy, which is all the active code and tests import at module
level (checked by the import analysis), and an `archive` extra was added for PyYAML, requests and pypdf.

**Corrected (each confirmed by a check, not assumed).**

1. *The Alzheimer's/dementia set contained four questions it should not have.* Rebuilding it from the released
   files reproduced the 212-question manifest hash for hash, and also showed that four questions were 2000–2003
   versions of reviews whose newer versions are in the confirmatory split (ungrouped rows in `AllStudyGroups`, so
   the exclusion by study group missed them). The builder now also excludes by Cochrane ID: **208 questions from 159
   reviews**. This was done before any output of the set existed and is recorded in `experimentation.md` §11. The
   preflight's expected counts were a hand-typed copy of the manifests; a test now ties them together.
2. *A prompt longer than the context window would have stopped the run.* llama.cpp raises an error for it. The
   answer step now retries once with each abstract cut to 200 words, never touches a prompt that fits, marks the
   answer `context_truncated` and the analysis counts them. Not observed in the three-question smoke test.
3. *Files the study writes now use LF line endings on every platform* (a Windows run would otherwise write CRLF
   into committed results and the manifest).
4. *Traceability:* each phase now writes `rag2_environment_<phase>.json` (package versions, Python, platform, the
   code's commit and whether tracked code was modified); informational, nothing compares it.
5. *Documents that contradicted the repository:* `data.md` said the confirmatory labels had not been analysed
   (they were, in the stage-1/2 run) and gave no class mix; three documents called the committed `REPORT.md` a dev
   report although the confirmatory run had overwritten it; the AD set was described as Alzheimer's questions
   although fewer than a quarter name Alzheimer's disease (most name dementia); the results README said confirmatory
   files existed only after the command ran; `_archive/README.md` cited a section that now holds something else
   and hid why the `test_pairs` tests cannot run (they import a path that no longer exists). Also removed: a dead
   function in `rag2.py` and the fixture-demo command from the README (the demo is archived). The per-step compute
   figures were reconciled with the three-question smoke test (≈ 6.8 minutes per question for four answers).
6. *Hygiene:* `.gitignore` now ignores `build/` and `dist/` (pip builds in the source tree), the archived index
   and the old `corpus/` location at the repository root, so `git add -A` is safe on a computer that still holds
   the multi-GB corpus text and index from before the move.

**Guards added or retargeted** (`test_scope_invariants.py`): the layout is checked by named files, not folders (a
computer that pulls this change still holds the ignored `corpus/data/` and `experiments/results/index/`); a
current document may name moved code only by its archive path; the package list must equal the active packages;
the archive's READMEs are checked for the `_archive/...` paths and commands they give.

**Verified.** Active suite 280 tests and archived suite 592 tests pass; in a fresh virtualenv with only numpy,
sockets blocked, the active suite passes (one figure test skipped: matplotlib is optional), the archived suite
passes with the `archive` extra, and all 38 active modules import from the installed package outside the
repository; `check_hermetic` reports both suites leave the tree unchanged; pyflakes reports one intentional import
in the active code (and seven cosmetic notes in archived code, left as written).

**Not changed.** The realigned design (settings and prompts are hashed into `rag2_design.json` after the dev
phase), every committed result and every committed answer file. Archived code was moved, not edited, apart from
import paths and the files named above.

**Risks identified (not defects; *confirmed* = computed or read from a file, *assumed* = not yet measured).**
*Confirmed:* a 1-point difference cannot be confirmed with 528 questions (only about 4–6 points can) or 208 (about
6–10); a true effect of exactly zero still shows +1 point or more in 23–32% of runs (32–39% with 208), which the pre-declared
three-way reading of the requirement accounts for. The gold labels are model-generated and an independent model
reproduces 81.4% of the confirmatory ones, which bounds what any accuracy can mean. The confirmatory split's labels
and its B0/B1 results were seen before the realigned design was fixed (declared in §9), so only the `ad` set is
untouched, and it is secondary. The verification criteria restate the benchmark's labelling rubric; R2C measures
that. The `ad` set has no changed verdicts and is old (48 of 208 reviews before 2005). Compute is about 114 h
(dev ≈ 26, held-out ≈ 60, `ad` ≈ 28, *estimated*); no dependency is pinned (the environment record captures
versions); llama.cpp output is not bit-for-bit reproducible across machines; MedChange states no licence, so its
data are rebuilt, not redistributed. *Assumed:* that the as-of pools of the old `ad` questions are large enough
(unmeasured until the pools are built); that MedCPT encoding can use the 4 GB GPU (never tried; `--device cuda`).

**Still to do (researcher).** Pull this change; run `rag2_pipeline dev` (≈ 26 h by the estimate) and read
`RAG2_DEV_REPORT.md`; the held-out and `ad` phases follow only after that report, each once.

## Phase 40 — The `_archive` folder removed (Oct 6)

At the researcher's request, and after checking that it was not required: nothing in the pipeline or the active tests
imports it (checked by the import guard and a scan), no document the thesis depends on lives only there (the
negative results are written up in this log, Phases 13–19, and in `methodology.md`), and Git history keeps every
file (commit `5e03540`; `git checkout 5e03540 -- _archive` restores the folder). What went: the Alzheimer's
framework (corpus scripts, question pool, three-arm runner, annotation, RAG metrics), the RAG² filter-reproduction
attempt, the first pilot, the matched-pair work, the legacy documents and 592 archived tests. The README, data,
methodology, glossary, reproducibility and protocol documents, `pyproject.toml` (the `archive` extras), the
hermetic check and the guard tests were updated; the three archive-specific guards were dropped (active suite: 277
tests, passing also in a numpy-only environment with sockets blocked). `.gitignore` now ignores `_archive/` as a
whole, so local leftovers of the first design (corpus text, index, filter-training labels) cannot be committed; a
`git pull` does not delete them. Earlier entries of this log name `_archive` paths as they were then.

## Phase 41 — Realigned dev check, freeze and confirmatory result (Oct 5–8)

Dev run (226 questions, with the directness judge): R2 50.0%, R2V 49.6%, R2V − R2 = −0.4 pp; the pre-declared
check failed only on direction (status REVISE ONCE). An analysis of R2 → R2V verdict changes showed fixes and
breaks nearly cancelling (e.g. SUPPORTED → NOT ENOUGH INFORMATION: 10 fixes, 9 breaks), so the one allowed prompt
revision was not used and the design was frozen (decision of the researcher, 2026-10-06; recorded in §9).
The pipelines were also changed to commit but never push (the researcher pushes by hand). Confirmatory run, once:
R2V − R2 = +1.3 pp (95% CI −1.9 to +4.7, p = 0.51): met as a point estimate, not confirmed (§15 of the protocol).
`report.py` now puts R2, R2C, R2V and R2V-ND beside the released closed-book models (existing work) in Table 1 and
in the paired and per-class tables; `REPORT.md` is the confirmatory version. Not yet run: the Alzheimer's/dementia
set (208 questions).

## Phase 42 — Repository audit, cleanup and documentation realignment before the dementia/Alzheimer's run (Oct 8)

At the researcher's request, before the `ad` phase is started, the whole repository was audited, cleaned and its documents
realigned with the state of the research. Recovery point: commit `f721bbb` (everything removed below is in it).

**Method (each step a check, not an assumption).** (1) An inventory of every tracked file and an import graph of every Python
module, from the entry points of the current pipeline (`rag2_pipeline`, `rag2_run`, `analyze_rag2`, `report`, `label_audit`,
`consistency_auto`, `build_benchmark`, `ad_benchmark`, `pubmed_asof`, `freeze_candidates`, `generate_answers`, `headroom`) and of
the tests; a module that none of them reaches was a candidate for removal. (2) For each candidate, what the current pipeline still
used from it was found first and moved: `abstracts.py` (the abstract snippet and study-type rules), `scoring.py` (accuracy,
intervals, paired tests and Holm families) and `runner.py` (the phase driver's step executor, the pushed-design guard and the
commit helper), each with tests. (3) The analysis and the report were regenerated from the committed records before and after the
change and compared: identical to within rounding. (4) Every path, `python -m` command, flag and section number that the current
documents cite, and every figure the README and `docs/evaluation.md` quote from the committed analyses, is now checked by tests
that stay in the suite.

**Removed** (not needed by the current pipeline; all in Git history): the code of the two completed, superseded approaches
(`stance.py`, `stance_check.py`, `synthesis.py`, `analyze_stage2.py`, `diagnostics.py`, `dev_audit.py`, `error_analysis.py`,
`analyze.py`, `helpfulness.py`, `findings.py`, `pipeline.py`) with their four test files; the documents `glossary.md` and
`related_work.md` (folded into `methodology.md`); two stale figures and the development report data of the old report. Moved to
`results/earlier_stages/` as results of record: the 18 output files of those approaches. Kept on purpose: `arms.py` and
`src/temporal_filter/`, because the settings hash of `arms.py` is stamped into every committed B0 and B1 answer and must not change
(it equals the hash recorded with the answers); `headroom.py`, which `report.py` imports. Unused dependency removed: `sentencepiece`
(nothing active imports it). Counts: 139 tracked files before, 126 after; 278 tests before, 200 after (97 tests went with the four
removed test files, 15 tested the removed stage-1 analysis, 21 were added for the new modules, the documentation guards were
rewritten, 15 replaced by 28; none of the tests of the current pipeline was dropped).

**Documents.** `README.md` now has exactly the agreed sections (title placeholder, overview, objectives, results, repository
structure, how to run); `docs/` holds `data.md`, `methodology.md` (rewritten step by step, with the adapted RAG² approach, the
literature and a glossary), `protocol.md` and `evaluation.md` (the former `experimentation.md`, split without duplication),
`reproducibility.md` and this log. The statements about RAG² were checked against the published paper and the authors' repository;
the bibliographic details of the other cited works against public listings.

**Inconsistencies found and corrected** (each confirmed in the files):

1. *Dates of the runs.* The protocol and the results README gave the held-out run as 2026-10-06/07; its environment record says it
   finished on 2026-10-08 (11:00 UTC) after starting on 2026-10-06; the development run ran 2026-10-05 to 2026-10-06.
2. *Rounding.* The analysis stored rates to four decimals and then showed one, so a few cells were off by 0.1 (R2V-ND accuracy
   259/528 shown as 49.0%, exactly 49.05%; the REFUTED recall of R2C and R2V 19.1%, exactly 19.0%; four cells of the development
   analysis). The code now stores six decimals; the committed dev and held-out analyses are not regenerated here (the B1 evidence-type
   cells need the frozen pools that only the laptop holds), the documents quote the exact values, and `evaluation.md` §6.5 lists the
   cells.
3. *A wrong sentence in generated output.* The analysis report said "with about 500 questions only differences of roughly 4–6 pp can
   be confirmed" whatever the split; for the 208-question `ad` set that would have been wrong. The figure is now computed from the
   number of questions (4–6 for 528, 6–10 for 208).
4. *`results/report/REPORT.md`* contained an unfilled placeholder (`{d['split']}`) and a reference to a file that does not exist;
   regenerated from the committed answers with the corrected code (no number changed).
5. *Stale references.* Docstrings and generated texts named `docs/experimentation.md` and `docs/experiment_plan.md`; `encoders.py`
   still described the removed Alzheimer's corpus; the package metadata mentioned the removed Flan-T5 steps. All corrected;
   the generated analysis and findings files keep the names current when they were written (noted in `data.md` §4).
6. *The ablations R2-RQ, R2-BR and R2-NF* are part of the design but were never run; the documents now say so wherever they are
   named, and a test keeps the statement true until answers for them exist.
7. *No environment record for the development run:* the record was added while it was running. The design record (settings, prompts,
   generator file hash) covers the systems; the code version of that run is not recorded.

**Research alignment, 2026-10-08.** Objective 1 (implement and validate with predefined metrics): the pipeline, metrics, tests and
decision rules exist, and the development and held-out splits are evaluated. Objective 2 (the extent of improvement over relevant
baselines and existing works): on the held-out split R2V − R2 = +1.3 pp (95% CI −1.9 to +4.7, p = 0.51), which meets the 1-point
requirement as a point estimate and is not confirmed; no claim of a demonstrated improvement is made. Not done: the baseline
ablations, and the dementia and Alzheimer's set (208 questions), which is the next run (about 22 h, 25 h with the judge, *estimated*).
Its result will be a secondary reading under the same rule. No result, claim or planned work is described as current unless it exists
in the committed results.

**Verification.** 200 tests pass, also in a fresh virtual environment with only numpy (Python 3.11, numpy 2.4.6) and outbound
sockets blocked (one figure test skipped without matplotlib), and in a fresh clone of the pushed commit; every module imports there; `check_hermetic` reports the suite leaves the tree unchanged; `pyflakes`
reports only the intentional availability import in `encoders.py`; the design record (settings, prompts, encoders) equals the
current design, and the committed B0 and B1 answers carry the current arm-settings and prompt hashes; the dry runs of the `dev`,
`confirm` and `ad` phases print the steps that `docs/reproducibility.md` describes. Real-model code paths cannot be tested without the
models. The modules that call them changed in this phase only in where the snippet and study-type helpers are imported from (the
functions moved to `abstracts.py` are identical to the originals, checked by comparing their source), one argument default (`generate_answers --arms`
now defaults to B0 and B1, and the pipeline passes the arms explicitly), the guard of `label_audit` for the held-out split (now the design
record), strings and docstrings.

## Phase 43 — README for a general reader; the subfolder READMEs removed (Oct 8)

At the researcher's request, before the dementia/Alzheimer's run:

* *Overview.* The README overview now says in two short paragraphs what the research is about, without design detail.
* *Objectives.* The section holds the two objectives only; the table was removed.
* *Results.* The section was rewritten for a general reader: what was measured, the main finding, whether the goal was reached,
  whether the other differences are real, what the extra check does, the practice run, search quality, other models for context,
  how reliable the reference answers are, and a short conclusion. The table of the status of the experiments was removed from
  the README; which experiments are completed and which are planned stays in `protocol.md` (status table), `evaluation.md` and
  this log.
* *Subfolder READMEs.* `experiments/medchange/README.md` and `experiments/medchange/results/README.md` were removed as
  redundant: the module map is the repository structure of the README and the docstring of each module; the list of committed
  result files moved to `data.md` §4 and the note on the rounding of two analysis files to `evaluation.md` §6.5.
* *Self-contained documents.* Sentences that attributed statements or reports to a separate write-up were reworded, so that
  every document stands on its own.
* *Guards.* Tests now check that the README objectives are the two bullets only, that the results section carries no status
  wording, that there are no README files in the subfolders, and that every table, count and sentence of the README results
  matches the committed analyses; the module-map guard reads the README.

Verification: 205 tests pass, also in a fresh virtual environment with only numpy and outbound sockets blocked, and in a fresh
clone of the pushed commit. No code that runs a model was touched in this phase.

## Phase 44 — Dementia and Alzheimer's results; eight released models; constant-answer baseline (Oct 9)

The `ad` phase finished (22.7 h measured, with the judge) and its results were pushed by the researcher.

* *Data check.* Pools were complete (median 20 candidates, no empty pool); R2 admitted at least one abstract for 177 of 208
  questions; every answer parsed.
* *Result.* R2V − R2 = +3.4 pp (95% CI −1.9 to +8.7, p = 0.28): **met as a point estimate, not confirmed**, with the rule
  fixed before the run. R2C (39.4% against 38.0% for R2V) did slightly better than R2V; none of the five local systems beat the constant answer
  NOT ENOUGH INFORMATION (42.3%); two released-answer models (Llama-3.3-70B, PMC-LLaMA, 43.3%) are two questions above it. Written up in the README and `evaluation.md` §6.6.
* *Added comparisons* (`protocol.md` §8, 2026-10-09; descriptive, included whatever they score): the released answers of
  all eight models (BioMistral, PMC-LLaMA and OLMo-13B were added to the five; their files hold three lines per answer, of
  which the second is the answer), and the best constant answer. The earlier README sentence that the local model was below
  all five models stays true only for the five best; OLMo-13B (50.0%) equals R2V, and two medical models are below it.
* *Code.* `headroom.read_answers` reads the released files; `report` accepts `--split ad` (no changed-question table or
  figures for a split without changed questions; default folders `report` for the held-out split and `report_<split>`
  otherwise); `label_audit` accepts `--split ad`; `analyze_rag2` no longer prints the age/update-window case-study line for `ad`.
* *Still to do on the laptop:* `label_audit --split ad` (about 1 h), then `analyze_rag2 --split ad --label-audit ...`.
* *Guards.* Tests check the eight models and every dementia figure of the README against the committed analyses.
* *Correction (Oct 9, later).* The first version of the write-up said that no system, and in `evaluation.md` §6.6 that every
  system, scored below the constant answer. That is true of the five local systems only: Llama-3.3-70B and PMC-LLaMA (43.3%,
  90 of 208) are two questions above the constant answer (42.3%, 88 of 208). README, `evaluation.md` and this entry were corrected.

## Phase 45 — Primary evaluation re-declared as Alzheimer's-specific; README shortened (Oct 9)

After a review of the existing results and of the candidate datasets, the researcher decided:

* *Decisions.* (1) The Alzheimer's-specific question set AD-KQA is the primary evaluation, provided it is verified before it
  is relied on. (2) The supervisor has left research decisions to the researcher; the requirement (R2V at least 1 percentage
  point of verdict accuracy above R2) stays. (3) One small independent-verifier model is approved, provided it is verified
  first. (4) The optional background/history and terminology strata are decided after the trial run. (5) The research paper is
  pointed to through a placeholder; the earlier ban on mentioning it was temporary. (6) The README is short and carries no
  experiment status. (7) Small documentation faults are fixed. (8) The dementia label audit and (9) the optional extras (a
  BioASQ check, a clinician spot-check, R2V-ND on the new set) are skipped for now.
* *What was checked, and what cannot be checked from here.* Checked: the pipeline modules that can be reused as they are, and
  the nine modules that hard-code the split names (`analyze_rag2`, `freeze_candidates`, `generate_answers`, `label_audit`,
  `pubmed_asof`, `rag2_pipeline`, `rag2_run`, `report`, `runner`); the as-of query builder on template-style Alzheimer's
  questions (it keeps filler words such as "associated", so the templates must avoid them); the power figures; the licences and
  sizes of the verifier candidates as reported by their publishers; that the generator code is model-agnostic (system and user
  message through the model's own chat template, so the verifier must accept a system turn). Not checkable from the sandbox: the
  PubMed counts (Stage 0), the verifier's quality (qualification run on the development split) and BioASQ's Alzheimer's count.
* *Changes.* README rewritten (about 1,000 words: two small tables, the reading of the requirement and its limits, a placeholder
  for the paper, no experiment status); `docs/evaluation.md` §6.0 (what each result is) and §6.5 before §6.6; `docs/protocol.md`
  status row, scope note, ledger row and the amendment of 2026-10-09 (reasons, design, frozen design, requirement and power,
  verifier qualification, gates), and a blank line that had split the ledger table was removed; a scope note in `docs/data.md`
  and `docs/methodology.md`; `docs/reproducibility.md` §10; the package `experiments/adkqa` with `stage0.py` (counts only, 12
  tests, no network in the tests); the guard tests repointed to the new README, to the paper placeholder and to the new module.
* *Not changed.* No result, split, label, prompt, setting or arm; no committed result file; earlier log entries. Stale pointers in
  generated files (`docs/experimentation.md` in `RAG2_FINDINGS.md` and `rag2_analysis_confirm.md`, `docs/experiment_plan.md` in
  `earlier_stages/FINDINGS.md`) wait for a re-analysis on the laptop.
* *Next.* The researcher runs Stage 0 (`python -m experiments.adkqa.stage0`); the go/no-go criterion decides whether the
  specification amendment, the builder and the verifier qualification follow.

## Phase 46 — Stage 0 run 1 (no-go) and revision 2 of the instrument (Oct 9)

* *Result of record.* Run 1 (`experiments/adkqa/results/stage0_counts.json`, committed by the researcher): 866 eligible records;
  distinct per area treatment 314, prevention 0, diagnosis 278, causes_risk 216, progression 58, symptoms 171, care_management 20.
  Four areas reached 100 of the six required, so the pre-declared criterion gave no-go. Europe PMC reference lists: 19 of 20.
* *Instrument error found.* Prevention 0 came from the qualifier "prevention & control", which PubMed did not find (visible in the
  recorded query translation). Every query also carried PubMed's routine result-limit message as a "warning". Both are fixed in
  revision 2 (`stage0.py`): "prevention and control"; only real warnings are kept; information-only counts are added (title-only
  Alzheimer's records, the pre-cutoff window, records in no area, supply when each record serves one area only). They do not enter
  the criterion. Thresholds (700 / 100 / 6) and the window are unchanged; the protocol records that run-1 counts were seen.
* *Real, not an artefact:* progression (58) and care_management (20) are thin when the condition must be a MeSH major topic, and the
  total pool (866) is tight for 360 questions at an unknown yield.
* *Tests.* 16 tests for Stage 0 (4 new: the prevention qualifier, warning filter, exclusive supply, unchanged thresholds).
* *Next.* One corrected re-run on the laptop (`stage0_counts_r2.json`). The decision after it, if still no-go, is the researcher's.

## Phase 47 — Stage 0 run 2: no-go again (Oct 9)

Run 2 (`stage0_counts_r2.json`, committed by the researcher): 866 records; qualifier frame treatment 314, prevention 21, diagnosis 278,
causes_risk 216, progression 58, symptoms 171, care_management 20. Four areas covered, six required: no-go under the unchanged
criterion. Information only: exclusive supply 634 records in the four covered areas (withdrawn in Phase 48: that allocation gave shared records to the uncovered areas; the distinct records are 710); pre-cutoff window 525 further records; the
MeSH-heading frame would give more records in progression (135) and care (57) but it is not the declared gate. Nothing built; the
choice of how to proceed (cover four areas, a flagged wider window, a smaller test set, or stop) is the researcher's.

## Phase 48 — Amendment: four covered areas, thresholds corrected (Oct 9)

The researcher accepted covering only the areas the evidence supports. Verified before writing it: the covered areas hold 710
distinct records (union of the identifiers in `stage0_counts_r2.json`), not 634; the earlier figure, and the statement that 300 + 60
questions were unlikely to be reachable, came from an allocation that gave shared records to the uncovered areas first, and are
withdrawn (at 40% draft survival 710 records give 284 questions, at 50% 355). Power recomputed with one formula for n = 150 to 300.
Changes: `protocol.md` §8 (amendment, ledger row, status row, the replaced wording marked in the gates); `stage0.py`
(`evaluate_gate_v2`, `--recheck`, exit code by the amended criterion; the original gate stays and is printed); tests (20 for
Stage 0, including the committed run-2 file). Thresholds changed: covered areas 6 to 4; new covered supply 650; test minimum 300 to
200 (keep all that pass, up to 300). All others unchanged and listed in the amendment. Optional strata dropped (0 and 3 records).

