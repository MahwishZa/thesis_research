# Adapted RAG² and evidence-criteria verification, ad split (secondary held-out test, Alzheimer's/dementia)

Items: 208. Arms: B0, B1, R2, R2C, R2V. Protocol: `docs/protocol.md`.

## Generation: verdict accuracy

| Arm | all (95% CI) | changed | unchanged | recall S / R / NEI | macro-F1 | answers NEI | outdated rate |
|---|---|---|---|---|---|---|---|
| B0 | 33.2% (27.1%–39.8%) | — | 33.2% | 75.0% / 4.4% / 30.7% | 28.3% | 33.7% | — |
| B1 | 30.8% (24.9%–37.3%) | — | 30.8% | 78.8% / 2.9% / 23.9% | 25.2% | 28.8% | — |
| R2 | 34.6% (28.5%–41.3%) | — | 34.6% | 90.4% / 5.9% / 23.9% | 29.2% | 27.4% | — |
| R2C | 39.4% (33.0%–46.2%) | — | 39.4% | 84.6% / 23.5% / 25.0% | 38.0% | 21.6% | — |
| R2V | 38.0% (31.7%–44.7%) | — | 38.0% | 71.2% / 22.1% / 30.7% | 37.2% | 28.4% | — |

## The requirement (+1.0 pp over the adapted RAG² baseline)

R2V − R2 = **+3.4 pp** (95% CI -1.9 to +8.7; 19 questions right only with R2V, 12 only with R2; exact McNemar p = 0.2810).

Reading (pre-declared): **met as a point estimate, not confirmed**.

## Secondary comparisons (Holm among themselves)

| Comparison | n | difference (pp) | 95% CI (pp) | a-only / b-only | p | Holm p | confirmed |
|---|---|---|---|---|---|---|---|
| R2 vs B1 | 208 | +3.8 | -1.4 to +9.1 | 21 / 13 | 0.2295 | 0.4590 | no |
| R2V vs B1 | 208 | +7.2 | +0.0 to +14.4 | 36 / 21 | 0.0627 | 0.2508 | no |
| R2C vs R2 | 208 | +4.8 | +0.0 to +9.6 | 19 / 9 | 0.0872 | 0.2616 | no |
| R2V vs R2C | 208 | -1.4 | -5.8 to +2.9 | 10 / 13 | 0.6776 | 0.6776 | no |

## Unsupported answers (automatic indicators)

| Arm | anachronism rate | unsupported decisive verdicts (answers with evidence) |
|---|---|---|
| B0 | 13.9% | — (0) |
| B1 | 0.0% | 16.3% (208) |
| R2 | 1.0% | 16.9% (177) |
| R2C | 1.0% | 0.0% (177) |
| R2V | 1.0% | 0.0% (177) |

## What the criteria arms changed (questions with evidence)

| Arm | valid output | verdict differs from R2 | changes that fixed / broke an answer |
|---|---|---|---|
| R2C | 100.0% | 18.6% | 19 / 9 |
| R2V | 100.0% | 20.9% | 19 / 12 |

## Retrieval (descriptive)

| Arm | admitted | none admitted | SR/MA / RCT / other | update-window share | mean age (y) | overlap with B1 | directness@k |
|---|---|---|---|---|---|---|---|
| B1 | 5.0 | 0.0% | 9.3% / 20.2% / 70.5% | 1.6% | 7.0 | — | 30.9% |
| R2 | 3.07 | 14.9% | 18.6% / 38.5% / 42.9% | 1.8% | 6.98 | 26.4% | 58.3% |
| R2C | 3.07 | 14.9% | 18.6% / 38.5% / 42.9% | 1.8% | 6.98 | 26.4% | 58.3% |
| R2V | 3.07 | 14.9% | 18.6% / 38.5% / 42.9% | 1.8% | 6.98 | 26.4% | 58.3% |

Alzheimer's case study (208 items, descriptive): correct answers per arm: B0 69, B1 64, R2 72, R2C 82, R2V 79.

A result is *confirmed* only if the p-value (Holm-adjusted in a family) is below .05, the 95% interval excludes 0 and the difference is positive. With 208 questions only differences of roughly 6–10 pp can be confirmed (docs/evaluation.md §4).
