# Archive — not part of the current research

Work that was tried and then abandoned or superseded, kept so the research history stays traceable.
**Nothing here is part of the active pipeline.** The current research is described in the root
`README.md` and `docs/experiment_plan.md`.

## Contents

| Folder / file | What it was | Why it is archived |
|---|---|---|
| `rag2_filter_reproduction/` | A local attempt to reproduce RAG²'s trained filter: label generation with a 4-bit Llama-3-8B on CPU (500 MedQA questions), Flan-T5 filter training, and diagnostics. Package `filter_training/` plus its 9 test files in `tests/`. | The authors do not distribute the checkpoint and only 5 label examples. The retrained filter learned only the class prior (accuracy 0.700 = majority baseline); diagnostics showed the labels carry almost no passage-specific signal at that scale (correctness flips were as frequent with an irrelevant passage). Results: `docs/log.md` Phases 13–18. |
| `alzheimers_pilot_v1/` | The first Alzheimer's evaluation: `fit_and_evaluate.py` and `run_real_evaluation.py`, their tests, and the committed pilot results (`results/`). | Primary metric and fitting objective were the same quantity (currency), so the proposed system would win by construction; the generator was an extractive stand-in; the baseline checkpoint was unvalidated; every question was asked as of the run date although 55 of 99 verdict-labelled questions cite pre-2010 reviews. Replaced by `experiments/medchange/` (`docs/experiment_plan.md` §13). |
| `docs_legacy/` | The four documents that preceded the topic docs (objectives, experimental specification, decision log, question-review guide). | Superseded by `docs/`. The specification remains the fullest record of earlier design decisions; the question-review guide still applies if the question pool is ever extended. |
| `question_pool_audit/` | A 2026-09-21 investigation of the 123-question pool's quality that produced an alternative 82-question pool. | Investigated, not adopted: every reported split uses the original pool. Kept for provenance. |
| `medchange_human_checks/` | `consistency.py`, the G1 human check of stated verdicts (a researcher marked 50 answers Y/N), with its tests. | Retired 2026-10-04: the researcher is not a medical expert and the outcome is the parsed verdict line; replaced by `experiments/medchange/consistency_auto.py` (`docs/experiment_plan.md` §13, item 9). Never run. |
| `test_pairs/` | Matched temporal-counterfactual pairs to measure a bias in RAG²'s filter directly, with pair construction, eligibility, splitting and power analysis. | An earlier research direction; its tests do not run in the active suite. |
| `retriever_interfaces.py` | The `Retriever` / `Reranker` abstract interfaces and pass-through stubs formerly in `src.common.retriever`. | Never used by any system or by the runner (the frozen candidate set is built by `experiments/shared/retrieval/`); an audit on 2026-10-02 found no reference in code, tests or documentation. |
| `contested.py`, `verifier.py` | Detecting contested evidence; post-hoc claim checking. | Earlier exploratory work outside the current scope. |
| `stage2_pilot.yaml`, `stage2_pilot_outputs/` | Config and output of a pilot of the `test_pairs/` machinery. | A ten-document fixture run, never research data. |

## Rules

1. **Not part of the current methodology.** No active code reads from here.
2. **Active code must not import from here** (`evaluation/tests/unit/test_scope_invariants.py` checks
   this). Archived code may import active code (e.g. the shared retrieval encoders).
3. **Nothing here is installed** (`pyproject.toml` does not list it).
4. **Archived tests run separately** and are kept passing so the archived evidence stays reproducible:
   `python -m unittest discover -s _archive -t .` (127 tests on 2026-10-04). The older `test_pairs/` tests are not part of that.
5. **Machine-local files** of the filter attempt (`filter_training/labels/`, `.textbook_index_cache/`,
   `checkpoints/`) are gitignored; the 500-question labels file represents 16.6 hours of compute.
6. **Temporary.** Everything here is also in the Git history; the folder can be deleted once the thesis
   is finished without affecting anything outside it.
