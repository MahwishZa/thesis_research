# Per-verdict behaviour (exploratory)

exploratory; metric chosen after the accuracy results were known; see protocol §10. Intervals: 10000 bootstrap resamples of questions.

## confirm split (n = 528)

Gold shares: SUPPORTED 44.3%, REFUTED 23.9%, NOT ENOUGH INFORMATION 31.8%

| System | Macro-F1 | Predicted SUPPORTED | REFUTED recall | NOT ENOUGH INFORMATION recall |
|---|---|---|---|---|
| B0 | 0.329 | 76.3% | 3.2% | 25.6% |
| B1 | 0.395 | 58.3% | 8.7% | 47.0% |
| R2 | 0.395 | 68.0% | 11.9% | 33.3% |
| R2C | 0.405 | 69.1% | 19.1% | 26.8% |
| R2V | 0.440 | 59.7% | 19.1% | 42.9% |
| R2V-ND | 0.420 | 62.7% | 16.7% | 38.1% |

| Paired difference | Macro-F1 (95% interval) | REFUTED recall, points (95% interval) |
|---|---|---|
| R2V vs R2 | +0.045 (+0.010 to +0.080) | +7.1 (+3.0 to +12.0) |
| R2C vs R2 | +0.010 (-0.033 to +0.053) | +7.1 (+0.8 to +13.9) |
| R2V vs R2C | +0.035 (-0.002 to +0.072) | +0.0 (-5.8 to +5.7) |

## ad split (n = 208)

Gold shares: SUPPORTED 25.0%, REFUTED 32.7%, NOT ENOUGH INFORMATION 42.3%

| System | Macro-F1 | Predicted SUPPORTED | REFUTED recall | NOT ENOUGH INFORMATION recall |
|---|---|---|---|---|
| B0 | 0.283 | 63.0% | 4.4% | 30.7% |
| B1 | 0.252 | 69.7% | 2.9% | 23.9% |
| R2 | 0.292 | 69.2% | 5.9% | 23.9% |
| R2C | 0.380 | 67.3% | 23.5% | 25.0% |
| R2V | 0.372 | 61.5% | 22.1% | 30.7% |

| Paired difference | Macro-F1 (95% interval) | REFUTED recall, points (95% interval) |
|---|---|---|
| R2V vs R2 | +0.080 (+0.024 to +0.135) | +16.2 (+7.9 to +25.4) |
| R2C vs R2 | +0.088 (+0.030 to +0.147) | +17.6 (+8.1 to +27.9) |
| R2V vs R2C | -0.008 (-0.055 to +0.037) | -1.5 (-8.2 to +4.9) |
