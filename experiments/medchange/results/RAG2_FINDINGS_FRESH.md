# Pre-registered test on fresh questions (run once)

850 fresh questions; arms R2, R2C and R2V; decision rule fixed in `docs/protocol.md` §10 before any question was answered.

## Primary outcome

Macro-F1, R2V − R2 = **-0.002** (95% interval -0.027 to +0.023; 10000 bootstrap resamples).
Reading: **no evidence**.

## Secondary outcomes (no confirmatory claim)

* Macro-F1, R2V − R2C: +0.034 (+0.004 to +0.065).
* REFUTED recall, R2V − R2: +1.5 points (-2.1 to +5.1).
* Verdict accuracy, R2V − R2: -1.8 points (95% interval -4.2 to +0.8; exact McNemar p = 0.1953). Reading by the rule of `docs/evaluation.md` §5: no evidence.
* Verdict accuracy, R2V − R2C: +2.5 points (95% interval +0.0 to +4.9; exact McNemar p = 0.0572).

| System | Accuracy | Macro-F1 | Predicted SUPPORTED | REFUTED recall |
|---|---|---|---|---|
| R2 | 47.2% | 0.395 | 70.0% | 14.6% |
| R2C | 42.9% | 0.359 | 70.5% | 15.3% |
| R2V | 45.4% | 0.393 | 61.6% | 16.1% |

R2V changed 19.5% of the answers it verified: 51 fixed, 66 broken.

## Alzheimer's disease and related dementias (descriptive)

The 208 questions of the Alzheimer's disease set: macro-F1, R2V − R2 = +0.080 (+0.024 to +0.135). The sign does not agree with the fresh result. Seen before the hypothesis was formed: description, not confirmation.

## Label audit (sample)

Agreement with an independent model 83.0%, kappa 0.7332 (300 questions).
