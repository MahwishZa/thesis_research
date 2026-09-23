# Archive — not part of the active thesis

This folder holds code and outputs from an earlier research direction. That
direction was replaced (see the root `README.md`) by a simpler one:
**RAG² + a Temporal Filter, compared against RAG² alone, for Alzheimer's
disease question answering.**

## What's here, and why it was moved

| Item | What it was for |
|---|---|
| `test_pairs/` | Matched temporal-counterfactual pairs (older evidence vs. newer evidence saying something different), used to measure a bias in RAG²'s filter directly. The pair-construction, eligibility, splitting, and power-analysis code, plus its tests. |
| `contested.py` | Detects when two passages disagree ("contested evidence"). |
| `verifier.py` | Post-hoc checking of claims in a generated answer. |
| `stage2_pilot.yaml`, `stage2_pilot_outputs/` | Config and output from a pilot run of the `test_pairs/` machinery. Already marked "not research data" before the move — a 10-document fixture run, not a result. |
| `question_pool_audit/` | A 2026-09-21 investigation into the 123-question pool's quality (duplication, unusable answers) that produced an alternative, smaller 82-question pool. **Investigated, not adopted**: the validation/test split actually used for every reported run (`experiments/shared/questions/splits.json`) was built from the original 123-question pool, not this one. Kept for provenance — it documents a real quality check that was done, and its full reasoning is in its own `audit_report.md`. |
| `superseded_outputs/real_evaluation_pilot/` | A 2026-09-22 real-data run (`run_real_evaluation.py`) that used **unfit placeholder** `theta`/`half_life` (not fitted on a validation split) as a deadline-driven stopgap. Superseded the next day by the properly validation-fitted, held-out-test run committed at `results/fit_and_evaluate/`. Kept for the methodological record — it is part of what showed the fitting step was necessary — not as a citable result. |
| `docs_legacy/` | The four documents `docs/` consisted of before the 2026-09-24 reorganization into topic docs (`methodology.md`, `data.md`, `research-glossary.md`, `evaluation.md`, `reproducibility.md`). Kept for the fuller internal engineering history (decision log, environment notes, dated change log) that the new docs summarize rather than reproduce in full. |

All of it is real, working, tested code and real output. None of it is
broken. It was moved because it answers a different question than the one
this thesis now asks, or because it was methodologically superseded by later,
more rigorous work — not because it was wrong — and keeping it in the active
folders made the project confusing to read.

## Rules for this folder

1. **It is not part of the current methodology.** The active pipeline is
   RAG² → Temporal Filter → No-Filter control → evaluation → ablation →
   statistics, and none of that reads from here.
2. **The active code must not import from here.** Verified: nothing under
   `src/`, `experiments/shared/`, or `experiments/baseline/` imports
   anything in `_archive/`.
3. **Nothing here is installed** (`pyproject.toml` does not list it) and
   its tests do not run under `python -m unittest discover -s tests`.
4. **Kept for history, not for use.** It shows what was tried and why the
   scope narrowed. If the Temporal Filter result turns out to be weak or
   inconclusive, this is a reasonable place to look for a next idea — the
   asymmetry-measurement approach in `test_pairs/` is a different, still
   valid way to ask a related question.
5. **This is temporary.** Once the thesis is finished, this folder can be
   deleted entirely. Nothing outside it depends on that happening, and
   nothing outside it will break if it does.

## If you want to run anything in here

It still works as standalone code, but it is not wired into
`experiments/runners/run_end_to_end.py` or any current test target. Treat
it as a separate, older project you're borrowing from, not as part of this
one.
