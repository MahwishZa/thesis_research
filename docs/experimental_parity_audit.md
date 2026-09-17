# Experimental Parity Audit, and the Abstention Decision

**Updated:** 2026-09-17 · Verified by reading the implementations, not the docs.

---

## 1. Parity status

The only intended difference between arms is the admission rule.

| Condition | Status | How it is enforced |
|---|---|---|
| Same question | **Parity** | `System.run(question=...)`, one value passed to every arm |
| Same question ID | **Parity** | `sample_id`, set once per item by the runner |
| Same frozen candidate evidence | **Parity, enforced** | The runner builds candidates once per item; `assert_same_candidate_sets` raises on any divergence |
| Same evidence ordering | **Parity** | All three arms select by `(rerank_rank, evidence_id)` then **restore candidate-list order** before building context |
| Same generator | **Parity, now enforced** | One `Generator` instance injected into every arm; `assert_generator_parity` raises before a run if the arms hold different instances |
| Same context budget | **Parity, now enforced** | `assert_budget_parity` raises before a run if `max_admitted_passages` differs; `context_budget()` finds it wherever each arm stores it |
| Same prompt template | **Parity, now enforced** | Default templates are byte-identical; `assert_prompt_parity` raises before a run if overrides differ |
| Same output schema | **Parity** | All arms return `ExperimentResult` |
| Same generation parameters | **BLOCKER** | No concrete generator exists yet; parameters are unset |
| Same random seed | **BLOCKER** | Same reason. Greedy decoding (temperature 0) is recommended so a single run suffices |
| **Abstention behaviour** | **Was broken — now resolved** | See §2 |

**Two blockers remain, and both are the same blocker:** there is no concrete
generator. `systems/interfaces/generator.py` provides an ABC and a
`CallableGenerator` adapter; no model-backed implementation exists. Generation
parameters and seeds cannot be fixed until one does.

Retrieval and reranking are also absent from the repository (`metadata` carries
`retrieval_external: True`). They do not threaten parity — both arms receive the
same frozen set regardless of how it was produced — but the frozen set has to
come from somewhere before Step 4.

---

## 2. The abstention asymmetry

### 2.1 Where it occurred, and why

In `RecencyAwareSystem.run()`: when **no passage cleared θ**, the arm returned
`prediction=None`, `output_state=ABSTAIN`, `reason="no_admitted_evidence"`.

The baseline does **not** do this. Verified by inspection: `RAG2System.run()`
has no empty-evidence guard — when its filter admits nothing it builds a prompt
with an empty evidence block and generates anyway.

### 1.1 Context budget and generator: documented, but previously unchecked

Both controls were declared in the arms' own docstrings and assumed by the
runner, and neither was verified.

**Context budget.** Each arm caps admitted evidence at `max_admitted_passages`,
but stores it in a different place: on the system itself (`NoFilterSystem`), on
`config` (`RAG2System`), and on `admission_policy.config`
(`RecencyAwareSystem`). Nothing compared them. An arm allowed more passages than
another answers from more context, so a difference in hallucination rate would
be attributable to context volume rather than to the admission rule — the exact
confound the shared budget exists to remove. `context_budget()` resolves the
value through all three shapes and `assert_budget_parity` raises before any
answer is written. An arm with no cap reads as `None` rather than as some
number, so an uncapped arm cannot be mistaken for a capped one.

This was not hypothetical. The repository's own smoke fixture produced its two
arms by giving them *different budgets* — it obtained "arms that admit different
subsets" precisely by encoding the confound. The fixture now differs by
admission rule instead, which is the shape the real comparison has to take.

**Generator.** `RunConfig` records one model, one model version and one
generation config, and stamps them onto every result record. If the arms held
different generators, that metadata would silently mislabel which model produced
which answer — worse than an unchecked difference, because the output would look
correct. `assert_generator_parity` compares object identity, which is the only
thing the `Generator` interface exposes (it carries no model name or decoding
parameters) and is also what a real run does: load once, share. The generator is
not part of the intervention, so there is no case in this design where the arms
should differ.

Both gates run in `run_experiment` before the first item, alongside
`assert_prompt_parity`. Tests: `tests/unit/test_runner_parity.py`.

### 2.2 Is abstention an intended component?

**No.** It is an implementation convenience. It fires only in the degenerate
case where the threshold admits nothing; it is not a designed safety behaviour,
it is not part of the scoring rule `A(s) = (1−λ)·ρ(s) + λ·R(s,q,t_q)`, and
nothing in the frozen design describes abstention as an intervention. The
original RAG² has no abstention mechanism at all.

### 2.3 Why it had to be fixed

An abstention makes no claims, so it can never be labelled hallucinated. A
θ set high enough drives HAR to zero while answering nothing. That is metric
gaming by construction, and it would have been invisible in a single HAR number.

### 2.4 Options considered

**Option A — force an answer.** Generate from admitted evidence even when empty.
Restores parity by construction: both arms do the same thing in the same
situation. No new metric, no new denominator. Cost: an answer generated with no
evidence is likely to be judged hallucinated — which is the honest outcome, not
a flaw.

**Option B — allow abstention, report coverage separately.** Scientifically
sound and more informative, but it makes the primary outcome a pair (rate,
coverage) rather than a single number, and it requires the reader to weigh two
quantities with no pre-declared exchange rate.

**Option C — count abstention as failure.** Simple and ungameable, but it
conflates "declined to answer" with "answered wrongly", which are different
behaviours and should not share a denominator silently.

### 2.5 Decision

**Option A is the default; Option B's accounting is reported alongside,
always.**

Concretely:

* `AbstentionPolicy.ANSWER_ALWAYS` is the **default** — the proposed solution
  generates from whatever it admitted, including nothing, exactly as the
  baseline does. This makes the primary comparison clean with no denominator
  negotiation.
* `AbstentionPolicy.ABSTAIN_WHEN_EMPTY` remains available as a **declared
  secondary condition**, never the primary comparison.
* Under `ANSWER_ALWAYS`, an answer produced from an empty evidence block is
  recorded as `output_state=UNGROUNDED` — not `GROUNDED`, which would have been
  false, and not `ABSTAIN`, because an answer was produced and must be
  annotated like any other.
* `experiments/evaluation/stats.py` always reports `answered`, `abstained`, `hallucinated`
  and `non_hallucinated` separately, with `answer_coverage` beside every rate.
* `compare_systems()` sets `interpretable: false` and refuses to emit a headline
  difference when either system answered nothing.

**Why this one:** hallucination is the primary outcome, resources are limited,
and the baseline cannot abstain. Option A needs no extra experiment, no new
primary metric and no pre-registered trade-off, while the coverage accounting
from Option B still catches any coverage loss. It is the smallest change that
makes the comparison fair.

### 2.6 θ — provenance

`AdmissionConfig.admit_threshold` is `Optional[float]` and defaults to `None`;
`validate()` **raises** with *"Fit it on the validation split before any test
run"* if it is unresolved. There is no default value that could be silently
inherited, and no code path reads test outcomes. **θ is not tuned on the test
set, and cannot be by accident.** The same holds for λ and H.

### 2.7 Tests

`AbstentionAccountingTests` and `AbstentionPolicyConfigTests` assert that
abstention is counted separately, that an abstention cannot also be
hallucinated, that the two HAR denominators are explicit and differ, that
coverage loss is visible beside the rate, that the default policy is
`ANSWER_ALWAYS`, and — the key one — that **a system abstaining on every
question yields `interpretable: false` and no headline difference.**
