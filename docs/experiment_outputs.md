# Experiment outputs

## `stage2_pilot/` — fixture dry run, **not research data**

These files were produced by running the Stage-2 pilot against the 10-document
fixture corpus that currently stands in for the Alzheimer's corpus. They exist
to show that the pipeline runs end to end and to quantify the readiness gap
(ledger findings G-S2-6 and G-S2-7). They are **not** evaluation material:

* every candidate is excluded, and 0 pairs are eligible;
* the questions are placeholders, not clinical questions;
* `check_a_interpretable: false` — the change points are corpus publication
  dates, not evidence-change dates, so the strata are not a Check A result.

No number in this directory may be cited as a research finding. Regenerate
them against the real corpus once Stage 1 completes, and against the external
evaluation dataset once it is obtained.
