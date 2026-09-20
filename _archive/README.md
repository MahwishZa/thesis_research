# Archive — not part of the active thesis

This folder holds code and outputs from an earlier research direction. That
direction was replaced (see `docs/current_objectives.md`) by a simpler one:
**RAG² + a Temporal Filter, compared against RAG² alone, for Alzheimer's
disease question answering.**

## What's here, and why it was moved

| Item | What it was for |
|---|---|
| `test_pairs/` | Matched temporal-counterfactual pairs (older evidence vs. newer evidence saying something different), used to measure a bias in RAG²'s filter directly. The pair-construction, eligibility, splitting, and power-analysis code, plus its tests. |
| `contested.py` | Detects when two passages disagree ("contested evidence"). |
| `verifier.py` | Post-hoc checking of claims in a generated answer. |
| `stage2_pilot.yaml`, `stage2_pilot_outputs/` | Config and output from a pilot run of the `test_pairs/` machinery. Already marked "not research data" before the move — a 10-document fixture run, not a result. |

All of it is real, working, tested code. None of it is broken. It was moved
because it answers a different, harder research question than the one this
thesis now asks, and keeping it in the active folders made the project
confusing to read.

## Rules for this folder

1. **It is not part of the current methodology.** The active pipeline is
   RAG² → Temporal Filter → No-Filter control → evaluation → ablation →
   statistics, and none of that reads from here.
2. **The active code must not import from here.** Verified: nothing under
   `systems/`, `experiments/evaluation/`, `experiments/runners/`,
   `experiments/retrieval/`, or `experiments/questions/` imports anything
   in `_archive/`.
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
