# Dev report (read this before saying "go")

Everything below uses the dev split only. Nothing from the confirmatory split has been looked at.

## Gate 1 (stance pilot, machine checks): **PASS**

* PASS: wording_agreement>=0.80
* PASS: control_neither_share>=0.70
* PASS: invalid_rate_real_papers<=0.02
* PASS: seconds_per_paper<=10

## Gate 2 (does stance predict the gold verdict on dev?): **FAIL**

Selected hybrid: **H0**. Stance-direction AUC 0.624 (0.5 = coin flip, needs >= 0.60). Constant-guess accuracy 46.5%.

| Variant | cross-validated accuracy |
|---|---|
| B1R | 52.2% |
| H0 | 51.8% |
| H1 | 50.5% |
| H1C | 51.8% |
| H2 | 51.2% |
| H3 | 51.0% |
| H3C | 51.3% |
| S0 | 43.6% |
| S1 | 43.9% |
| S2 | 42.9% |
| S3 | 44.3% |

* FAIL: S0_cv_accuracy>constant_prior
* FAIL: selected_hybrid_minus_B1R>=+1.0pp
* PASS: stance_auc>=0.60

## Label audit (is the gold label reproducible by another model?)

Agreement 83.2%, kappa 0.7422, 188 label-stable items of 226.

## Consistency of stated verdicts

Parse rate 100.0% (PASS); independent judge agrees on 99.0% of the sample (diagnostic).

## What happens next

Gate 2 failed. The confirmatory run will test RQ1 only (retrieval versus no evidence); RQ2 is reported as not supported on dev and not tested on the confirmatory split.
