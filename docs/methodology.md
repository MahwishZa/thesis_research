# Methodology Reference (Step 12)

**Written before any experimental results exist.** This page assembles, in
one place, the definitions and procedures the thesis methods chapter can be
written from now. It contains **no results, no conclusions, and no claim
about whether the proposed solution reduces hallucinations** — none exist
yet, because Steps 1–2 are still in progress and no final run has been
performed.

Where a topic already has its own document, this page summarises it and
points there rather than duplicating it.

---

## Research question and pipeline

> Does the proposed solution/system reduce the rate of hallucinated answers
> in Alzheimer's disease question answering, relative to the baseline system,
> under identical question and evidence conditions, while maintaining
> comparable QA accuracy?

Primary outcome: hallucinated answer rate (HAR). Secondary outcome: QA
accuracy. **There is no third outcome and no diagnostic objective.** The
twelve-step pipeline (`README.md`) runs corpus → questions → freeze →
baseline → proposed → collect → annotate → HAR → accuracy → statistics →
error analysis → write-up.

`docs/frozen_scope.md` is the canonical scope statement and now states this
same question. An earlier design made admission-rate asymmetry (Δ) primary;
it is superseded, recorded in `frozen_scope.md` §7 for provenance, and
produces no thesis outcome. Its infrastructure survives unrun under
`experiments/test_pairs/`.

## Corpus methodology

`alzheimer_corpus/` builds and documents itself; out of scope here and
untouched by this work. See its own scripts and config for the retrieval and
normalisation pipeline.

## Evaluation-question methodology

Full protocol: `docs/question_sources.md`. Summary: candidate questions come
from inspected external sources (Cochrane systematic reviews, NIH pages) with
verbatim reference answers and traceable locators — never invented by an LLM
end to end. Automatic validation and deduplication narrow the pool; a human
reviewer decides ACCEPT / REVISE / REJECT / HOLD
(`docs/question_review.md`). Current status: `docs/question_pool_status.md`.

## Evidence-freezing methodology

`experiments/evaluation/freezing.py`. A `FrozenItem` pairs one approved
question with its retrieved candidate evidence and a `corpus_snapshot`
identifier naming the corpus build the evidence came from. The candidate set
is hashed order-sensitively (`candidate_set_hash`) — order matters because
context position affects what a generator attends to. Both arms must receive
identical hashes (`assert_same_candidate_sets`) or the run refuses to
proceed. The provenance firewall (`assert_firewall`) refuses a manifest where
a question's reference evidence also appears among its candidates, which
would make the comparison circular. `from_question()` builds a `FrozenItem`
directly from an approved `EvaluationQuestion`, so the shared fields
(question text, reference answer, source, date) are copied once rather than
by hand at each call site.

## Baseline system methodology

`systems/baseline/`. Two arms: a no-filter control, and a RAG²-style arm
(`RAG2System`) whose admission filter labels each candidate `[HELPFUL]` /
`[NOT_HELPFUL]`. Fidelity to the published RAG² filter, and what could and
could not be reproduced, is documented in
`docs/rag2_classifier_feasibility.md` and `docs/experimental_parity_audit.md`.
Every arm consumes a frozen candidate set, a shared prompt template
(`assert_prompt_parity`), a shared context budget (`assert_budget_parity`) and
one shared generator instance (`assert_generator_parity`), each checked before
the first answer is generated, so admission rule is the only thing that differs
between arms by construction.

## Proposed-system methodology

`systems/proposed/`. Admits by threshold over a recency-aware score;
described fully in `README.md` and `docs/frozen_scope.md`'s formula section
(the scoring mechanism itself is unchanged by the outcome pivot above — only
which measurement is primary changed). The abstention policy — what happens
when nothing clears threshold — is fixed to `ANSWER_ALWAYS` by default so the
proposed system cannot lower its hallucination rate merely by declining to
answer; full reasoning in `docs/experimental_parity_audit.md`.

## Hallucination definition (primary outcome)

An answer is hallucinated if it contains **at least one claim unsupported by,
or contradicted by, the evidence supplied to the system that produced it** —
a faithfulness judgement against the frozen candidate set, not a clinical
correctness judgement. Schema and validation: `experiments/evaluation/
annotation.py`. Diagnostic subtypes, recorded but not separately weighted:
faithfulness, factuality, temporal, misinterpretation, ambiguity, other.

**Annotation workflow:** generated answers are exported as a blinded packet
(`build_blinded_packet`) with system identity replaced by "System A" /
"System B" and order randomised; the unblinding key is kept separate from
what the annotator sees. Completed annotations are re-validated on import
(`read_annotations`) and recombined with system identity only after review
(`unblind_annotations`).

**HAR calculation:** `experiments/evaluation/stats.py`. `har()` reports the
rate two ways — conditional on answering, and over every item with
abstentions counted as non-hallucinated — because a bare rate over zero
answers is not a rate. `coverage()` and `compare_systems()` refuse to report
a headline difference when either system answered nothing.

## QA accuracy (secondary outcome)

`experiments/evaluation/accuracy.py`. A `QAJudgment` records one system's
generated answer against the reference answer, a binary `correct` outcome,
and **who or what decided** (`judge`) — a human annotator id, or a named rule
for closed-form questions. This module deliberately does not decide
correctness itself: automatic string-overlap scoring (ROUGE, BLEU,
BERTScore) is excluded as a correctness signal by the same constraint that
keeps it out of the primary outcome. `qa_accuracy()` aggregates to total /
correct / incorrect / accuracy.

## Statistical methodology

`experiments/evaluation/stats.py`, standard library only. For the paired
binary HAR and accuracy outcomes: exact McNemar (`mcnemar`, a two-sided exact
binomial test on the discordant pairs — no normal approximation, no SciPy
dependency), and a paired bootstrap 95% CI on the absolute difference
(`paired_bootstrap_ci`, resampling questions as units so each question's
paired outcome stays together). `holm()` adjusts p-values across the
pre-declared primary comparisons. No new test was written for QA accuracy: it
is the same paired-binary shape as HAR, so the existing functions apply
directly (verified in `tests/unit/test_evaluation.py::QAAccuracyStatsTests`).

## Experimental protocol

Hardware and execution venue: `docs/hardware_and_resources.md`. Baseline vs.
proposed parity — same question, same frozen candidate set, same prompt
template, same generator, same generation config — and the one resolved
asymmetry (abstention): `docs/experimental_parity_audit.md`. Both real
systems have been run together end to end over synthetic fixtures
(`tests/integration/test_baseline_and_proposed.py`); no final or
result-generating run has been performed.

## Error analysis

`outcome_crosstab()` splits paired outcomes into baseline-only / proposed-only
/ both / neither, keeping question ids so representative examples — including
cases where the proposed system does worse — can be pulled directly.
`error_analysis()` breaks each cell down by diagnostic subtype. Both only
count what annotation already recorded; neither infers a pattern.

## Limitations to state regardless of outcome

The baseline is an adaptation of RAG², not a reproduction — see
`docs/experimental_parity_audit.md` for exactly where and why. QA accuracy
correctness is judged, not automatically scored, so its reliability depends on
the same annotation protocol as hallucination. The evaluation set size is
whatever review of the ~123-question pool converges on (`docs/
question_pool_status.md`), not a power-calculated target. Execution hardware
(`docs/hardware_and_resources.md`) constrains which generator can run where.
