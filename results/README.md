# Results

Committed run output, for reproducibility.

- `fit_and_evaluate/` — the pilot-scale validation-fit + held-out-test run.
  An engineering checkpoint confirming the pipeline and statistical
  procedure work correctly on real data; **not** the thesis's reported
  comparison — see `docs/evaluation.md` §7 and `docs/reproducibility.md` §5
  for what remains before a result is reported.
- `smoke_test_NOT_FINAL/` — a small real-generator smoke test confirming the
  generation code path works end to end with an actual (non-mock) model.
  Not a thesis result.

`tables/`, `figures/`, and `error_analysis/` will be added here once the
full-scale evaluation (see `docs/reproducibility.md` §5) has run.
