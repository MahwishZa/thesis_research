# Adapted RAG² and evidence-criteria verification, confirm split (confirmatory)

Items: 528. Arms: B0, B1, R2, R2C, R2V, R2V-ND. Protocol: `docs/experimentation.md`.

## Generation: verdict accuracy

| Arm | all (95% CI) | changed | unchanged | recall S / R / NEI | macro-F1 | answers NEI | outdated rate |
|---|---|---|---|---|---|---|---|
| B0 | 46.6% (42.4%–50.9%) | 41.9% | 56.0% | 85.0% / 3.2% / 25.6% | 32.9% | 22.0% | 32.6% |
| B1 | 48.1% (43.9%–52.4%) | 45.0% | 54.3% | 70.1% / 8.7% / 47.0% | 39.5% | 37.3% | 32.9% |
| R2 | 48.7% (44.4%–52.9%) | 43.6% | 58.9% | 79.5% / 11.9% / 33.3% | 39.5% | 25.6% | 32.9% |
| R2C | 48.5% (44.2%–52.7%) | 45.0% | 55.4% | 79.9% / 19.1% / 26.8% | 40.5% | 19.7% | 31.2% |
| R2V | 50.0% (45.8%–54.2%) | 47.0% | 56.0% | 71.8% / 19.1% / 42.9% | 44.0% | 30.1% | 32.3% |
| R2V-ND | 49.0% (44.8%–53.3%) | 45.3% | 56.6% | 74.4% / 16.7% / 38.1% | 42.0% | 27.8% | 33.1% |

## The requirement (+1.0 pp over the adapted RAG² baseline)

R2V − R2 = **+1.3 pp** (95% CI -1.9 to +4.7; 44 questions right only with R2V, 37 only with R2; exact McNemar p = 0.5052).

Reading (pre-declared): **met as a point estimate, not confirmed**.

## Secondary comparisons (Holm among themselves)

| Comparison | n | difference (pp) | 95% CI (pp) | a-only / b-only | p | Holm p | confirmed |
|---|---|---|---|---|---|---|---|
| R2 vs B1 | 528 | +0.6 | -3.0 to +4.2 | 48 / 45 | 0.8358 | 1.0000 | no |
| R2V vs B1 | 528 | +1.9 | -2.5 to +6.1 | 73 / 63 | 0.4404 | 1.0000 | no |
| R2C vs R2 | 528 | -0.2 | -3.6 to +3.4 | 45 / 46 | 1.0000 | 1.0000 | no |
| R2V vs R2C | 528 | +1.5 | -1.7 to +4.7 | 40 / 32 | 0.4096 | 1.0000 | no |
| R2V vs R2V-ND | 528 | +0.9 | -1.1 to +3.0 | 19 / 14 | 0.4869 | 1.0000 | no |
| R2V vs R2 (changed) | 353 | +3.4 | -0.8 to +7.4 | 35 / 23 | 0.1480 | 0.8880 | no |

## Unsupported answers (automatic indicators)

| Arm | anachronism rate | unsupported decisive verdicts (answers with evidence) |
|---|---|---|
| B0 | 13.8% | — (0) |
| B1 | 0.0% | 14.2% (528) |
| R2 | 1.9% | 15.4% (447) |
| R2C | 1.9% | 0.0% (447) |
| R2V | 1.9% | 0.0% (447) |
| R2V-ND | 1.9% | 0.0% (447) |

## What the criteria arms changed (questions with evidence)

| Arm | valid output | verdict differs from R2 | changes that fixed / broke an answer |
|---|---|---|---|
| R2C | 100.0% | 26.6% | 45 / 46 |
| R2V | 100.0% | 22.4% | 44 / 37 |
| R2V-ND | 100.0% | 18.6% | 33 / 31 |

## Retrieval (descriptive)

| Arm | admitted | none admitted | SR/MA / RCT / other | update-window share | mean age (y) | overlap with B1 | directness@k |
|---|---|---|---|---|---|---|---|
| B1 | 5.0 | 0.0% | 10.9% / 17.1% / 72.0% | 48.0% | 9.76 | — | 28.6% |
| R2 | 3.3 | 15.3% | 24.6% / 32.9% / 42.5% | 48.5% | 9.21 | 22.5% | 57.3% |
| R2C | 3.3 | 15.3% | 24.6% / 32.9% / 42.5% | 48.5% | 9.21 | 22.5% | 57.3% |
| R2V | 3.3 | 15.3% | 24.6% / 32.9% / 42.5% | 48.5% | 9.21 | 22.5% | 57.3% |
| R2V-ND | 3.3 | 15.3% | 24.6% / 32.9% / 42.5% | 48.5% | 9.21 | 22.5% | 57.3% |

Label-stable subset (430 items, descriptive): R2V − R2 = +1.9 pp; R2 − B1 = +1.4 pp.

Alzheimer's case study (10 items, descriptive): correct answers per arm: B0 3, B1 5, R2 5, R2C 5, R2V 5, R2V-ND 4.

A result is *confirmed* only if the p-value (Holm-adjusted in a family) is below .05, the 95% interval excludes 0 and the difference is positive. With about 500 questions only differences of roughly 4–6 pp can be confirmed (docs/experimentation.md §7).
