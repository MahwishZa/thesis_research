# Synthesis layer: dev cross-validation (226 items, 151 changed)

5-fold cross-validation, 50 repeats, seed 20261003; fixed ridge 5.0. Constant-prior accuracy 46.5%. Stance-direction AUC (SUPPORTED vs REFUTED, S0 signed score): 0.624.

| Variant | accuracy | accuracy, changed items | recall SUPPORTED | recall REFUTED | recall NOT ENOUGH INFORMATION |
|---|---|---|---|---|---|
| B1R | 52.2% | 48.6% | 74% | 7% | 52% |
| S0 | 43.6% | 40.8% | 88% | 6% | 5% |
| S1 | 43.9% | 41.6% | 90% | 2% | 6% |
| S2 | 42.9% | 39.4% | 85% | 5% | 7% |
| S3 | 44.3% | 42.4% | 90% | 3% | 6% |
| H0 | 51.8% | 49.0% | 75% | 11% | 47% |
| H1 | 50.5% | 47.1% | 74% | 9% | 45% |
| H2 | 51.2% | 48.0% | 75% | 11% | 44% |
| H3 | 51.0% | 47.6% | 74% | 10% | 45% |
| H1C | 51.8% | 49.1% | 75% | 10% | 47% |
| H3C | 51.3% | 48.3% | 75% | 10% | 45% |

**Selected hybrid: H0.** Gate 2: **FAIL**

* PASS: stance_auc>=0.60
* FAIL: selected_hybrid_minus_B1R>=+1.0pp
* FAIL: S0_cv_accuracy>constant_prior
