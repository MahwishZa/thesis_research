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

## Current status (2026-10-02)

* **Research direction:** an as-of evaluation of recency-weighted evidence admission (the Temporal Filter)
  on MedChangeQA, with the Alzheimer's study as a secondary case study. Protocol: `experiment_plan.md`.
* **Built and tested:** the MedChange benchmark (504 usable changed and 250 unchanged items, seeded
  dev/confirmatory splits), the PubMed as-of probe, frozen dev candidate pools (226 items), zero-shot
  helpfulness scores for the dev pools, the six arms, the generation harness, the analysis (accuracy,
  retrieval-level checks, McNemar with Holm, gates) and the G1 consistency check. Alzheimer's corpus
  (114,256 PMC records, 4,377,041 chunks), question pool (113 usable) and dense index (4,376,141 x 768):
  built, secondary.
* **Gates:** G0 passed (94.0% of changed dev items). G1, G2 and G3 have not been run on dev; running the
  arms B0 and B1 on all 226 dev items (about 5.3 h) is the next step.
* **Results:** none. No accuracy, hallucination or retrieval comparison exists; the only generations
  made are six timing answers (3 items x 2 arms).
* **Not built:** frozen pools for the confirmatory split; the human hallucination annotation; a second
  generator; the Alzheimer's as-of case study.
* **Abandoned and archived:** the RAG² filter reproduction (the checkpoint is not distributed; local
  retraining learned only the class prior) and the first Alzheimer's pilot runners (circular primary metric).
* **Known limitations of the design** (see `methodology.md`): the B2/P helpfulness score is an untrained
  stand-in, not RAG²; gold verdicts are model-generated; the generator is a 4-bit 8B model on CPU; the
  decisive-flip subgroup is small (114 items).
