# `experiments/results/`

Machine-local, gitignored retrieval indexes for the Alzheimer's case study:

- `index/` — the full dense index over the Alzheimer's corpus (4,376,141 × 768, ~12.5 GB), built by
  `python -m experiments.shared.retrieval.build_index`.
- `index_pilot_reduced_scope/` — an earlier ~1% pilot index.

Tracked run results live next to the code that produced them: `experiments/medchange/results/` for the
primary experiment. The earlier Alzheimer's pilot outputs (`fit_and_evaluate`, `smoke_test_NOT_FINAL`,
`real_evaluation_pilot`) are archived in `_archive/alzheimers_pilot_v1/results/`; they showed the pipeline
ran, not that anything improved.
