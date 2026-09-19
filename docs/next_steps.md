# Next Actions

## Alzheimer's corpus stage: COMPLETE / FROZEN (verified 2026-09-19)

All seven pipeline stages have run for real against the live corpus, and
their outputs have been independently verified internally consistent -
see "Corpus integrity verification" below for the full evidence chain. Do
not rerun 01-07 or regenerate their outputs; the corpus is a fixed input
to everything from here on.

Two known, non-blocking gaps, neither an integrity problem: Stage 03's
guidelines/textbooks registries have 0 rows curated (manual curation, can
be added later without touching anything upstream), and the PMC
licensing gate is not enforced (see "PMC licensing gate is not enforced"
below) - this affects redistributability, not internal use for
experiments, and the corpus JSONL is gitignored and never leaves this
machine anyway, so it does not block using the corpus for research now.
It matters before any external publication or redistribution of the
corpus text itself.

**Scope note (2026-09-18):** `docs/current_objectives.md` now governs scope
and priority. The concrete next actions it implies: (1) run
`experiments/runners/run_end_to_end.py` for real once a real model and real
question/evidence data are available (see that doc's "known blockers"), (2)
extend the standard-metrics ablation in `experiments/evaluation/rag_metrics.py`
as needed, (3) do not let the plan below re-expand scope beyond those three
objectives. Everything below predates that scope note and is kept as
detailed status/history, not as the active priority order.

**Updated:** 2026-09-18 · 395 tests passing (before this realignment; see
`docs/current_objectives.md` and the new `experiments/runners/run_end_to_end.py`
+ `experiments/evaluation/rag_metrics.py` for what was added since).

**PMC retrieval, normalize, deduplicate AND chunk are now EXECUTED on the
real corpus** — see "PMC finalization" and "Normalize/deduplicate/chunk —
EXECUTED" below. Chunking's first attempt did not complete (the original
stuck run); the batched rewrite completed it for real: 111,315 documents →
4,377,041 chunks, real MedCPT tokenizer. Claim classification (Stage 07)
has been started against that real output.

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
| 1. Corpus — PMC source | **EXECUTED** (retrieval + finalization) | — |
| 1. Corpus — normalize (04) | **EXECUTED** on real corpus (114,157 docs) | — |
| 1. Corpus — deduplicate (05) | **EXECUTED** on real corpus (111,315 unique) | — |
| 1. Corpus — chunk (06) | **EXECUTED** on real corpus (4,377,041 chunks) | — |
| 1. Corpus — claim classification (07) | **EXECUTED** on real corpus (4,377,041 chunks tagged, all 43 classes) | — |
| 1. Corpus — guidelines/textbooks (03) | IMPLEMENTED, TESTED, zero rows curated | manual curation (optional, not blocking) |
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

**What this did not mean at the time:** the normalize → deduplicate → chunk
stages had not yet run over the real PMC data. They have since - see below.

---

## Normalize/deduplicate/chunk/claim classification — ALL EXECUTED (2026-09-18/19)

Independently verified from `alzheimer_corpus/logs/quality_control.log` and
`alzheimer_corpus/logs/deduplication.log` (both git-tracked) cross-checked
against the git-tracked registries/reports - not assumed:

| Stage | Evidence | Result |
|---|---|---|
| 04 normalize | `quality_control.log`: `PMC extraction \| verified=114157 \| parsed=114157 \| failed=0`, then `stage 04 \| normalized=114157 \| ad_relevant=67608 \| excluded=46549` | **EXECUTED**, matches `reports/normalization_report.csv` (114,157 data rows) |
| 05 deduplicate | `deduplication.log`: `stage 05 \| unique=111315 \| duplicates recorded=2842` | **EXECUTED**, matches `metadata/duplicates.csv` and `reports/deduplication_report.csv` (2,842 data rows, exact match) |
| 06 chunk | First attempt: `quality_control.log` shows the tokenizer-selection warning with no completion line after it anywhere - the original stuck run (running ~1 hour, no output) that motivated the Stage 06 batching rewrite. Second attempt, with the rewritten `06_chunk.py`: `tokenizer ncbi/MedCPT-Article-Encoder loaded \| fast=True`, then periodic `stage 06 progress \| documents=N \| chunks=M \| ...` lines throughout, ending `stage 06 \| documents=111315 \| chunks=4377041 \| elapsed=7892s` and `chunks_this_run=4377041 \| chunks_total=4377041 \| tokenizer=ncbi/MedCPT-Article-Encoder \| 256/32/224` | **EXECUTED for real.** 111,315 documents (matches Stage 05's exact count) → 4,377,041 chunks, real MedCPT tokenizer, correct 256/32/224 spec, ~7892s (≈2h11m), sustained ~14-20 docs/sec throughout - the batching rewrite works correctly at full real scale. |
| 07 claim classification | `quality_control.log`: after the memory/visibility fix below and a further keyword-matching speedup, progress lines every 100,000 chunks at ~975 chunks/sec sustained, ending `stage 07 \| chunks=4377041 \| tagged claim_types=26 \| tagged evidence_levels=17 \| validation sample=300 (stratified, seed=42) \| elapsed=4520s` | **EXECUTED for real.** 4,377,041 chunks (matches Stage 06's exact output count) tagged; every one of the 26 claim_types and 17 evidence_levels classes was matched at least once (real coverage, not a degenerate all-`<untagged>` run); ~75 min at ~975 chunks/sec. `reports/claim_class_distribution.csv` (27 rows: 26 classes + `<untagged>`), `reports/evidence_level_distribution.csv` (18 rows: 17 + `<untagged>`) and `reports/annotation_report.csv` (300 rows, the configured validation sample size, exactly) are all present, well-formed CSV (verified: correct column counts, zero malformed rows), and internally consistent with the log. |

### Corpus integrity verification (2026-09-19)

Full evidence chain re-verified end to end, independent of any single
stage's own claim, by cross-checking each stage's *output count* against
the *next* stage's *input count* as independently reported by that next
stage's own log line - not just trusting each stage's self-report in
isolation:

```
PMC finalize:  matched_pmcids=112260 = accounted_pmcids=112260, manifest_rows=114256   (exact)
04 normalize:  PMC verified=114157 = parsed=114157 = normalized=114157                  (exact)
05 deduplicate: input 114157, duplicates=2842, unique=111315  (114157 - 2842 = 111315)  (exact)
06 chunk:      documents=111315  (matches 05's unique count exactly)  -> chunks=4377041  (exact)
07 classify:   chunks=4377041  (matches 06's exact output count)  -> all 4377041 tagged  (exact)
```

Every arrow above is a real cross-check between two independently-written
log lines/CSV files, not an assumption. Zero count discrepancies found
anywhere in the chain. Also checked and found clean:
- Every tracked CSV (`metadata/*.csv`, `reports/*.csv`) parses as
  well-formed CSV with a consistent column count on every row - verified
  programmatically, not by eyeballing.
- `quality_control.log` and `deduplication.log` contain no `ERROR`,
  `CRITICAL`, or traceback lines anywhere.
- `retrieval.log` (Stage 01/02, 9.8 MB) contains 227 `ERROR` lines, all
  from the multi-day retrieval campaign's transient retries (network
  resets, a mid-run "108 retrieval failures" that blocked one attempt) -
  none after the log's own final, authoritative line: `Stage 02
  finalization completed successfully`, whose reconciled accounting
  (`matched_pmcids=112260, accounted_pmcids=112260`) has zero
  discrepancy. These are resolved history, not a live problem.
- A new integration test
  (`tests/integration/test_corpus_to_retrieval.py`) runs the real
  04→05→06→07 pipeline and confirms `experiments/retrieval/corpus.py`'s
  actual `read_passages()` loads the result without error, every passage
  carries the fields retrieval/admission need, chunk_ids are unique, and
  Stage 07's tags survive into what the retrieval layer hands onward -
  proof, not assumption, that the corpus schema is what the retrieval
  layer expects.
- No accidental data loss: `git status`/`git diff` against `HEAD` show
  every tracked corpus artifact (metadata, reports, logs) matches the
  student's own pushed commits exactly; nothing in this repository
  touched them this session.

**Conclusion: the Alzheimer's corpus stage is COMPLETE and FROZEN.** No
genuine integrity problem was found; nothing was corrected or rerun.

**guidelines.csv and textbooks.csv are header-only (0 rows)** — confirmed
directly (`04_normalize`'s own log line: `guidelines.csv \| rows=0 ...`,
`textbooks.csv \| rows=0 ...`). `03_guidelines.py` is implemented and tested
but no document has been curated into it yet; this does not block anything
above (04's real-data path treats both registries as optional).

**The real intermediate JSONL files exist only on the machine that produced
them** (`alzheimer_corpus/data/**` is gitignored by design - "research data
is never committed", see `alzheimer_corpus/.gitignore`) — not recoverable
from git history or a sandbox that only has this repository checked out.
Only the tracked logs/reports/registries above are visible here.

### Stage 07 had the same architectural flaw Stage 06 had - fixed before completion

`07_claim_classification.py`'s `main()` loaded every chunk into memory as
one Python list (`chunks = list(read_jsonl(src))`), then wrote nothing back
until the very end, with zero progress logging in between - the identical
shape of problem that caused Stage 06's original stuck run, just relocated
one stage later and at a larger scale (4.3M+ chunks vs. 111k documents).
Found while auditing the stage the student had just started running for
real. Rewritten the same way Stage 06 was: streams input to a temp file,
replaces the original atomically (`.part` + `.replace()`, matching this
codebase's existing atomic-write convention), logs progress every 100,000
chunks, and keeps only a lightweight per-chunk projection (8 small fields,
not the full text) in memory for the stratified validation sample instead
of every chunk's full record. Same output shape, verified field-for-field
against the original in-memory algorithm (preserved as a reference in
`tests/unit/test_corpus_claim_classification.py`) both at unit scale and
via a real subprocess run; a 30,000-synthetic-chunk smoke test completed in
~32s with ~57MB peak child-process RSS.

**Update: Stage 07 completed for real with the fix in place** - see
"Corpus integrity verification" above for the confirmed completion
evidence. A further real bottleneck was found and fixed after that first
fix (keyword matching itself, not just the memory architecture) - see the
`test_corpus_claim_classification.py`/ledger entry for
`_word_set`/`KeywordMatchEquivalenceTests` for detail. The corpus is now
ready for `experiments/runners/run_end_to_end.py` once real evaluation
data and a real generator are wired in - see `docs/current_objectives.md`.

---

## Stages 04–07 — READY FOR REAL EXECUTION, 2026-09-18 (ledger D-43, D-44)

An audit (before this fix) found stages 04–07 could only ever process the
10-record synthetic fixture: nothing connected Stage 02's real output (a
manifest plus JATS XML) to what Stage 04 read, and Stage 07 crashed
immediately (`KeyError: 'groups'`) against the current
`claim_taxonomy.yaml`, which had been restructured to `claim_types`/
`evidence_levels` since the last time 07 actually ran.

**Fixed**, all IMPLEMENTED and TESTED against synthetic fixtures (44 new
tests) — not yet run against the real 114,256-row manifest, which exists
only on the student's machine:

* `_common.py` gained a PMC-XML bridge (`parse_jats_xml`, `iter_pmc_records`,
  `read_pmc_manifest`) using stdlib `xml.etree.ElementTree` — no new
  dependency.
* `04_normalize.py` reads `metadata/pmc.csv` + XML by default now;
  `--input <fixture path>` still selects the old offline single-JSONL path
  unchanged (verified byte-identical to the fixture output already on disk
  under `alzheimer_corpus/data/` - gitignored, not git-tracked; see the
  note at the top of `tests/unit/test_corpus_normalize_pipeline.py`). A
  missing manifest fails clearly rather than silently falling back to the
  fixture.
* `05_deduplicate.py`: `metadata/duplicates.csv` now written through the
  same safe CSV writer as its report counterpart, not hand-joined strings.
* `06_chunk.py` now carries `ad_relevant`/`ad_relevance_score` from the
  normalized document onto every chunk (previously dropped there).
* `07_claim_classification.py` rewritten against `claim_types` (topical
  tagging) and `evidence_levels` (a second, independent dimension). Every
  current class uses taxonomy-derived keywords, not curated ones, and says
  so in its own output (`method: keyword-from-taxonomy`). `claim_status`,
  `temporal_status` and `disease_relevance` are deliberately **not**
  tagged here — see D-44 for why (temporal status duplicates the thesis's
  own admission-time recency treatment; disease relevance duplicates
  Stage 04's existing decision).
* **Real bug found and fixed along the way:** `redistribution_allowed()`
  only matched hyphenated licence codes (`CC-BY`); PMC's real
  `license_code` values are space-delimited (`CC BY`), so every real PMC
  record would have been marked non-redistributable regardless of its
  actual licence. Fixed; the licence families treated as distributable are
  unchanged.

**01_pubmed_download.py, 02_pmc_download.py and 03_guidelines.py were left
untouched.** Both 01 and 02 are mature, already-executed code against real
data (676 PMIDs; the independently-verified 114,256-row PMC manifest) —
"placeholder logic" does not describe them, and rewriting either would risk
the real, already-collected data disagreeing with a "corrected" script.

**Next real step (updated - 04, 05 and 06 have since run for real, see
"Normalize/deduplicate/chunk — EXECUTED" above):** confirm
`07_claim_classification.py` (now fixed for the same memory/visibility
issue Stage 06 had) completes on the real 4.3M-chunk output, then wire real
evaluation data and a real generator into
`experiments/runners/run_end_to_end.py` per `docs/current_objectives.md`.

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
| **PMC licensing gate not enforced** | corpus composition | see below |

### PMC licensing gate is not enforced (relocated from the now-removed alzheimer_corpus/README.md)

`04_normalize.py` stamps `redistribution_allowed` on every PMC-sourced
record (`_common.redistribution_allowed`) but does not act on it - unlike
the guidelines/textbooks path, which correctly excludes a restricted row's
text (`_common.iter_official_documents`). On the real, finalized manifest
(`metadata/pmc.csv`, 114,256 rows) roughly **19,669 rows (~17%)** carry a
licence outside `_common.DISTRIBUTABLE` (`CC BY-NC-ND`, `TDM`, blank, or
missing) and would be affected if this gate were enforced. Stage 04 has
been run for real since this was first found (see
"Normalize/deduplicate/chunk — EXECUTED" above), so this now DOES affect
already-generated real output, not just a future run - the gate should be
decided and, if enforced, Stage 04 re-run, before that output is treated as
final. This is a dataset-composition decision, not a bug fix silently
applied - it changes which real documents' text is in the corpus.

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
