# RAG² + Temporal Filter — Alzheimer's QA thesis

MS thesis implementation. **Research code, not a clinical system** — nothing
here is validated for, or usable in, patient care.

## 1. Research question

> **Does adding a Temporal Filter to RAG² improve question answering for
> Alzheimer's disease, compared to RAG² alone?**

Do not assume the answer. The experiment exists to find out, and to report
honestly if the Temporal Filter does **not** help.

## 2. Base method — RAG²

RAG² ([Sohn et al., NAACL 2025](https://github.com/dmis-lab/RAG2)) retrieves
passages for a question, then uses a trained filter to decide which ones an
LLM actually gets to see. The filter judges each passage on its text alone —
it never looks at *when* the passage was published. Code: `systems/baseline/`.

## 3. Proposed contribution — Temporal Filter

The same idea as RAG², plus one signal: how old each passage is, relative to
the question. Code: `systems/proposed/`.

```
A(s) = (1 − λ)·ρ(s)  +  λ·T(s, q, t_q)          admit if A(s) ≥ θ
```

| Symbol | Meaning |
|---|---|
| `ρ(s)` | Relevance — the same reranker score RAG² already uses |
| `T(s, q, t_q)` | Temporal score — `2^(−age_days / H)`, 1.0 for a passage published on the question date, halving every `H` days |
| `λ` | How much weight goes to the temporal score vs. relevance (0 to 1) |
| `θ` | Admission threshold — a passage is kept only if `A(s) ≥ θ` |
| `H` | Half-life in days — how fast the temporal score decays |

`λ`, `θ`, `H` are fitted on a validation split, never guessed. `λ = 0` turns
the temporal part off entirely — that is the ablation study's "component
removed" condition.

## 4. Control — No Filter

A third arm that admits every retrieved passage up to the context budget,
with no filtering at all. It shows whether filtering helps at all, so a
result for the Temporal Filter can be read against an honest floor.

| Arm | What it does | Code |
|---|---|---|
| RAG² (baseline) | Flan-T5 filter, text only | `systems/baseline/rag2.py` |
| RAG² + Temporal Filter (proposed) | Same idea + a temporal score | `systems/proposed/` |
| No Filter (control) | Admits everything, up to the budget | `systems/baseline/no_filter.py` |

Retrieval runs **once per question**, is frozen, and every arm sees the
exact same retrieved passages. Only the admission rule differs between arms
— everything else (prompt, context budget, generator, decoding settings) is
identical and checked in code before a run starts
(`experiments/evaluation/runner.py`).

## 5. Experimental workflow

```
RAG² baseline  →  Temporal Filter  →  No-Filter control
        ↓                ↓                    ↓
                  same questions
                  same retrieved evidence
                  same generator
                        ↓
                    Evaluation
                        ↓
                     Ablation
                        ↓
              Statistical analysis
```

One command runs all three arms, evaluation, and the ablation sweep:

```bash
python -m experiments.runners.run_end_to_end
```

It prints two comparisons:
- `main_evaluation` — RAG² vs. the full Temporal Filter.
- `ablation_study` — full Temporal Filter vs. the same system with `λ=0`
  (temporal part switched off).

## 6. Evaluation

Every arm's answers are scored the same way, by
`experiments/evaluation/rag_metrics.py`:

| Metric | What it measures |
|---|---|
| Exact match, token F1, ROUGE-L | Does the generated answer match the reference answer? |
| Context precision / recall | Did the arm admit the evidence that actually supports the answer? |
| Groundedness | Does the answer's wording actually appear in the evidence it was given? |

## 7. Ablation

`λ = 0` removes the temporal score and keeps everything else the same —
this isolates whether the temporal signal specifically is what helps, not
just "filtering in general." `run_end_to_end.py` always includes `λ=0` in
its sweep for this reason.

## 8. Where the active code is

```
alzheimer_corpus/    the evidence corpus (PMC-based), COMPLETE / FROZEN
systems/             RAG² baseline, Temporal Filter, No-Filter control
experiments/         retrieval, question pool, evaluation, and
                     runners/run_end_to_end.py — the one entry point
tests/               unit + integration tests for all of the above
docs/                four documents — see below
```

| Document | Read it for |
|---|---|
| [`docs/current_objectives.md`](docs/current_objectives.md) | **Canonical scope** — the research question, the three objectives, what's blocking a real run. Start here. |
| [`docs/research_experimental_specification.md`](docs/research_experimental_specification.md) | **The method** — every arm's exact behaviour, the parameters, the generator, metrics, statistics, and how to reproduce a run. |
| [`docs/status_and_decisions.md`](docs/status_and_decisions.md) | **The record** — what has actually run, what was decided and why, and the change log. |
| [`docs/question_review.md`](docs/question_review.md) | Instructions for reviewing the candidate question pool. |

## 9. Where archived/temporary work is

`_archive/` holds an earlier, more complex research direction (matched
old-vs-new evidence pairs, contested-evidence detection, answer
verification) that does not answer the current research question. It is
not deleted, still works, and is fully explained in `_archive/README.md` —
but the active pipeline has **zero dependencies on it**.

## 10. How to run the current experiments

```bash
# Run every test (unit + integration)
python -m unittest discover -s tests -t .

# Run the fixture demo: all three arms, evaluation, and the ablation sweep
python -m experiments.runners.run_end_to_end

# Same, with a real generator once one is available (see the specification)
python -m experiments.runners.run_end_to_end --real-model \
    --model-revision <pinned-commit-sha>
```

## Status

**Alzheimer's corpus: COMPLETE / FROZEN** (verified 2026-09-19, see
`docs/status_and_decisions.md` §2). All seven pipeline stages have run for
real against the live corpus: 676 PubMed records; 114,256-row PMC manifest,
114,157 verified and normalized; 111,315 unique after dedup; 4,377,041
chunks via the real MedCPT tokenizer; every chunk tagged by claim
classification. Do not rerun stages 01-07.

The 123-question evaluation pool is under human review. The pipeline
(retrieval → Temporal Filter → generation → evaluation → ablation) runs
correctly end-to-end on a fixture and has been confirmed working with a
real (non-mock) generator. **No experimental result exists yet** — see
`docs/status_and_decisions.md` for full readiness and blockers.
