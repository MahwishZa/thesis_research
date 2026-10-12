# Findings of the realigned study (confirmatory split, run once)

Each conclusion follows the rules of `docs/evaluation.md` §5: a difference is confirmed only when its p-value is below .05 and its 95% interval excludes 0.

## Verdict accuracy: R2V against the adapted RAG² baseline (R2)

R2V − R2 = +1.3 pp (95% CI -1.9 to +4.7; exact McNemar p = 0.5052).

Reading: **positive but not confirmed**.

## Secondary comparisons (Holm)

* R2 vs B1: +0.6 pp (95% CI -3.0 to +4.2; Holm p = 1.0000) — not confirmed
* R2V vs B1: +1.9 pp (95% CI -2.5 to +6.1; Holm p = 1.0000) — not confirmed
* R2C vs R2: -0.2 pp (95% CI -3.6 to +3.4; Holm p = 1.0000) — not confirmed
* R2V vs R2C: +1.5 pp (95% CI -1.7 to +4.7; Holm p = 1.0000) — not confirmed
* R2V vs R2V-ND: +0.9 pp (95% CI -1.1 to +3.0; Holm p = 1.0000) — not confirmed
* R2V vs R2 (changed): +3.4 pp (95% CI -0.8 to +7.4; Holm p = 0.8880) — not confirmed

Full tables: `rag2_analysis_confirm.md`.
