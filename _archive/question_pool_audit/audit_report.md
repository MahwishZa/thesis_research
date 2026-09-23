# Audit of the 123-question Alzheimer's evaluation set

**Performed 2026-09-22**, in response to reviewer feedback that the answers
are too diplomatic, that there are many repeated questions, and that most
items are poorly designed for evaluation.

This audit inspects all 123 `status: validated` records in
`experiments/questions/candidates.jsonl` individually. It does not assume
the existing count, wording, or duplication level is acceptable merely
because the records already exist.

**The original dataset is untouched.** `candidates.jsonl` and `review.csv`
are byte-identical to before this audit. The repaired set is a separate
file: `experiments/questions/repaired/candidates.jsonl`.

## What "useful" means here

The thesis question (`docs/current_objectives.md`, `README.md`) is whether
adding a Temporal Filter to RAG² improves Alzheimer's QA. The existing
evaluation pipeline (`experiments/evaluation/rag_metrics.py`,
`run_end_to_end.py`'s `temporal_subgroup` breakdown) already defines how a
question contributes evidence:

| Dimension | Measured by | What a useful question needs |
|---|---|---|
| Hallucination | `groundedness` (token overlap of the generated answer vs. admitted evidence) | A reference answer with a real, checkable claim to ground against — not necessarily a *confident* claim, a *checkable* one |
| Outdated knowledge | `temporal_candidate` flag + `temporal_subgroup` breakdown | A citation the pipeline can verify has actually been revised (`.pub2`+), so the corpus plausibly holds both an older and a current version of the evidence |
| Out-of-context | `context_precision` / `context_recall` | A well-scoped question with one identifiable evidence target, so "admitted the wrong evidence" is distinguishable from "admitted the right evidence but answered badly" |
| Factual correctness | `exact_match` / `token_f1` / `rouge_l_f1` | A specific enough reference answer that a match or mismatch is meaningful |

**A hedge is not automatically a defect.** Many Cochrane conclusions
genuinely are "insufficient evidence" — that is the real scientific state,
not vagueness introduced by this dataset. An answer like *"there is
insufficient evidence to recommend X"* is in fact a **good** hallucination
probe: a generator that answers with unwarranted confidence against that
evidence is a detectable, real failure. The audit therefore does not treat
every hedge as a problem — only answers that contain **no checkable claim
at all**, or that **do not address their own question**, or that are
**redundant with a better-formed item**.

## Part 1 — counts requested before any dataset change

| # | Question | Count |
|---|---|---|
| 1 | Duplicates / near-duplicates | **41** (31 merged away + the 10 non-boilerplate-removed items also had at least one near-duplicate sibling counted once) — precisely: **31 items** were disposed of as MERGE because a stronger duplicate exists |
| 2 | Poor evaluation items (no checkable claim, or answer doesn't address the question) | **10** — see REMOVE below |
| 3 | Too vague/diplomatic to serve *any* purpose (distinct from legitimate hedges — see above) | **4** (`ADQ-75a8b9996b8e`, `ADQ-b5d49cbb5553`, `ADQ-57b72f57a492` — boilerplate methodology text with no verdict; `ADQ-b848dd75752e` — "There are two included studies.") |
| 4 | Meaningfully test hallucination | **82** of the surviving set (every surviving item has a checkable claim to ground against) |
| 5 | Meaningfully test outdated knowledge | **48** of 82 survivors carry a verified `temporal_candidate=True` (real `.pub2`+ Cochrane citation) |
| 6 | Meaningfully test out-of-context behaviour | **82** — every survivor has one identifiable evidence target; this is a property of the *pipeline* (frozen candidate sets, `context_precision/recall`), not of individual question wording, so it applies uniformly once a question is well-formed |
| 7 | Test factual correctness/quality | **82** — same reasoning as row 4 |
| 8 | No useful current evaluation purpose | **10** — the REMOVE set |

(Rows 4, 6, 7 overlap heavily by design — most surviving questions serve
multiple dimensions at once, which is expected and desirable, not double
counting error. Row 5 is the only dimension a minority of questions can
satisfy, because it requires a specific, verifiable property — a revised
citation — that most items do not have.)

## Part 2 — duplicate analysis

Duplicate detection used the Cochrane review ID (`CDxxxxxx`) embedded in
`reference_locator`, not string matching on question text, because the
reviewer's complaint is about **repeated evaluation content**, and two
different `.pub` versions of the same review can be worded completely
differently while testing the identical underlying fact.

**Finding: 74 distinct Cochrane reviews are cited across 106 Cochrane-sourced
questions. 21 of those reviews are cited by more than one question —
53 questions in total, 43% of the whole set — because the question pool
included every historical `.pub` revision of a review as a separate item
instead of one current item per review.**

For each of the 21 clusters, the most recent version was kept (the current,
non-outdated verdict — chosen without inventing any text) and earlier
versions were merged into it, **except** one cluster (`CD001747`,
galantamine, 3 versions) where **no version** states an actual efficacy
verdict — every version repeats a methodological sentence about trial
participant characteristics — so all three were removed rather than one
kept, per the "do not fabricate" rule: none contains a checkable claim to
promote.

One additional near-duplicate pair was found outside Cochrane clustering:
two MedQuAD items (`ADQ-4c5ced141d11`, `ADQ-75f93f8a9cb7`) both define "what
is Alzheimer's disease" from different NIH source pages. The more complete
of the two was kept.

**No cluster was retained at full size merely because its members are
worded differently** — per the task's explicit instruction. Where a
cluster's verdict genuinely evolved across versions (4 clusters, see below),
that evolution is exactly what makes the surviving, most-recent item useful
for outdated-knowledge evaluation — it is not a reason to keep more than one
item, because this pipeline's design (`temporal_candidate`,
`temporal_subgroup`) tests against the corpus's dated evidence at admission
time, not against a second frozen "old" question.

Clusters with a real verdict change across versions (verified by reading the
full answer text of every version, not assumed from dates):

| Review | Change |
|---|---|
| CD003154 (memantine) | Blanket "small beneficial effect" (2003–2006) → severity-dependent efficacy (2019) |
| CD003160 (statins/AD risk) | "No good evidence to recommend" (2001) → "good evidence... has no effect" (2009–2016) |
| CD007514 (statins/dementia treatment) | "Insufficient evidence" (2010) → "no benefit on primary outcomes" (2014) |
| CD000147 (nimodipine) | "Results not poolable" (2001) → "can be of some benefit" (2002) |

The remaining 17 multi-version clusters repeat the same verdict across
versions with no informational change — pure duplication, correctly merged
with no temporal significance claimed.

## Part 3 — main problems discovered

**1. Diplomatic/vague answers.** The majority of Cochrane-sourced answers
are the review's own hedge-first conclusion sentence. Most of these are
*legitimate* (see "What useful means" above) and were kept. A small number
(4, listed above) contain **no claim at all**, only methodological
boilerplate or a bare study count — these cannot be repaired without
inventing content, so they were removed rather than rewritten. **No answer
was rewritten to sound more confident than its source.** The dataset's own
provenance protocol (`research_experimental_specification.md` §13) already
requires verbatim, never paraphrased, reference answers — rewriting hedge
language into an assertive claim would itself be a form of fabrication, so
this audit did not do it. This is a stated limitation, not an oversight —
see Part H.

**2. Duplication.** 53/123 questions (43%) were multiple citation-versions
of 21 underlying reviews; systematically resolved as above.

**3. Poor question construction — a defect distinct from diplomacy.** 6
items (all MedQuAD-sourced) have a reference answer that **does not answer
their own question** — an artifact of automatic question-templating over a
source page title (e.g. *"How to diagnose Alzheimer's Caregiving?"* paired
with post-diagnosis guidance text; *"What causes Alzheimer's Disease?"*
paired with a sentence about early-vs-late-onset classification). These
were removed, not revised, because a correct question for that content
requires knowing what the source page actually says elsewhere, which this
audit does not have access to.

**4. Weak hallucination evaluability.** Beyond the boilerplate/no-verdict
items already listed, no additional systematic weakness was found: every
surviving Cochrane-sourced answer states a specific claim (a direction of
effect, a certainty level, or an explicit "insufficient evidence" verdict)
that a generated answer can be checked against.

**5. Weak out-of-context evaluability.** Not found as a separate defect
class. This dimension depends on the frozen-candidate-set mechanism and
`context_precision`/`context_recall`, both already implemented and applying
uniformly to every well-formed surviving question; it is not something an
individual question's wording can be deficient in, once the Q/A mismatch
and no-claim defects above are removed.

**6. Missing temporal/supersession structure.** Confirmed, and reported
rather than manufactured: `temporal_candidate=True` is a real, verified flag
(a `.pub2`+ Cochrane citation), never invented here. Only 48/82 survivors
carry it — the set does not, and should not be made to, claim temporal
relevance for topics whose only Cochrane review has never been revised.

## Part D — evaluation coverage of the final 82

| Dimension | Count | Basis |
|---|---|---|
| Hallucination-evaluable (has a checkable claim) | 82 / 82 | every survivor, by construction of the KEEP criterion |
| Outdated-knowledge-evaluable (`temporal_candidate=True`, verified) | 48 / 82 | real `.pub2`+ citations |
| Out-of-context-evaluable (single identifiable evidence target) | 82 / 82 | pipeline property, applies once Q/A mismatch is removed |
| Factual-correctness-evaluable | 82 / 82 | same as hallucination row |
| Cochrane evidence-synthesis sourced | 72 / 82 | treatment/diagnosis/prevention claims |
| MedQuAD definitional/factual sourced | 10 / 82 | disease characteristics, genetics, epidemiology |

No single composite score is reported, per instruction — a question can and
usually does serve more than one dimension at once.

**Discriminating power (baseline vs. proposed system).** The 48
temporal-flagged survivors are the set most likely to separate RAG² from
RAG² + Temporal Filter, because they are exactly the cases where an
admission rule blind to publication date could retrieve genuinely superseded
context. The remaining 34 non-temporal survivors are still useful — they are
the honest floor: if the Temporal Filter does not regress on questions where
recency cannot matter, that is itself evidence the mechanism isn't
overfitting to the wrong signal.

## Part E — final dataset size and justification

**82 of 123 original questions survive**, none invented. This is not a
predetermined target — it is the number of questions in the original pool
that (a) state a checkable claim, (b) answer their own question, and (c) are
not a duplicate of a stronger item covering the same underlying fact. Every
removed or merged item's disposition is recorded in `audit_table.csv` with a
stated reason; nothing was discarded silently.

82 is smaller than the ~100 the original pool status document targeted
(`docs/status_and_decisions.md` §4), which is a real, reportable consequence
of the duplication found, not a shortfall introduced by this audit. See
Part H for the recommended path to close the gap **without fabricating
content**.

## Files changed

| File | Change |
|---|---|
| `experiments/questions/candidates.jsonl` | **Untouched** |
| `experiments/questions/review.csv` | **Untouched** |
| `experiments/questions/audit_table.csv` | **New.** 123 rows: `original_id, status, final_id, problem, evaluation_purpose, action_taken, reason` |
| `experiments/questions/audit_report.md` | **New.** This file |
| `experiments/questions/repaired/candidates.jsonl` | **New.** The 82 surviving records, byte-identical to their originals (no text rewritten) |
| `experiments/questions/repaired/review.csv` | **New, generated.** Produced by the existing, unmodified `experiments/questions/export_review.py --pool-dir experiments/questions/repaired` — proves the repaired set is schema-compatible with the real pipeline without writing any new tooling |

No other repository file was modified. No folder was reorganized. No
implementation code changed.

## Validation performed

1. `python -m experiments.questions.export_review --pool-dir experiments/questions/repaired` — the existing tool's own internal checks (`assert_neutral`, `assert_complete`, `missing_provenance`) ran against all 82 repaired records: **0 missing provenance, 82/82 reviewable, exported cleanly.**
2. Every repaired record constructed successfully as a real `EvaluationQuestion` (`experiments/evaluation/questions.py`) and passed `validation_failures()` with **zero failures** — not merely written in the right shape, but validated against the actual schema class the pipeline uses.
3. Full test suite: `python -m unittest discover -s tests -t .` — **472 tests, same 2 pre-existing, unrelated failures** (fixture-vs-real-artifact comparisons, present before this audit). No new failures.
4. `git status` confirms `candidates.jsonl` and `review.csv` are unmodified.

## Remaining limitations (cannot be resolved from this repository alone)

1. **Diplomatic-but-vacuous answers could not be repaired by rewriting**, only removed. Repairing them properly means selecting a *different, already-existing* verbatim sentence from the same cited Cochrane review (e.g. its actual efficacy conclusion instead of a methodology sentence) — this requires fetching the full review text, which this audit does not have access to and will not fabricate. If a human reviewer with access to the cited DOIs re-extracts a real conclusion sentence for the 4 boilerplate items and the 6 Q/A-mismatched MedQuAD items, up to 10 more questions could potentially be restored honestly.
2. **Whether the corpus actually contains both an older and a newer passage for each of the 48 temporal-flagged survivors is unverified.** This audit checked the *questions'* citation history, not the *corpus's* retrievable content — that would require querying the real corpus (only on the student's machine, never available here) for each topic.
3. **82 is below the ~100-question target.** The honest options, in order of preference: (a) accept 82 as scientifically sufficient — it already exceeds typical thesis-scale evaluation sets and every item is individually defensible; (b) source additional questions through the *existing* pipeline (`experiments/questions/build_pool.py`, `docs/research_experimental_specification.md` §13) from category-C guideline sources not yet used, which is a data-sourcing action, not something this audit can do; (c) re-extract the 10 removable items per limitation 1. This audit does not choose among these — that is a research decision, not an implementation one.
4. **No new human review has occurred.** The repaired pool still needs the same `ACCEPT`/`REVISE`/`REJECT`/`HOLD` process as the original (`docs/question_review.md`) before any question is `approved`. This audit narrows and cleans the *candidate* pool; it does not substitute for review.
