# Findings of the realigned study (Alzheimer's disease set, secondary test set, run once)

Each conclusion follows the rules of `docs/evaluation.md` §5: a difference is confirmed only when its p-value is below .05 and its 95% interval excludes 0.

## Verdict accuracy: R2V against the adapted RAG² baseline (R2)

R2V − R2 = +3.4 pp (95% CI -1.9 to +8.7; exact McNemar p = 0.2810).

Reading: **positive but not confirmed**.

## Secondary comparisons (Holm)

* R2 vs B1: +3.8 pp (95% CI -1.4 to +9.1; Holm p = 0.4590) — not confirmed
* R2V vs B1: +7.2 pp (95% CI +0.0 to +14.4; Holm p = 0.2508) — not confirmed
* R2C vs R2: +4.8 pp (95% CI +0.0 to +9.6; Holm p = 0.2616) — not confirmed
* R2V vs R2C: -1.4 pp (95% CI -5.8 to +2.9; Holm p = 0.6776) — not confirmed

Full tables: `rag2_analysis_ad.md`.
