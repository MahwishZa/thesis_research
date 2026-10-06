# Adapted RAG² and evidence-criteria verification, dev split (exploratory)

Items: 226. Arms: B0, B1, R2, R2C, R2V, R2V-ND. Protocol: `docs/experiment_plan.md`.

## Generation: verdict accuracy

| Arm | all (95% CI) | changed | unchanged | recall S / R / NEI | macro-F1 | answers NEI | outdated rate |
|---|---|---|---|---|---|---|---|
| B0 | 45.1% (38.8%–51.6%) | 41.1% | 53.3% | 81.0% / 4.1% / 20.8% | 31.2% | 22.1% | 36.4% |
| B1 | 53.1% (46.6%–59.5%) | 49.7% | 60.0% | 73.3% / 10.2% / 52.8% | 43.5% | 38.5% | 31.8% |
| R2 | 50.0% (43.5%–56.5%) | 48.3% | 53.3% | 71.4% / 16.3% / 41.7% | 42.6% | 31.4% | 33.1% |
| R2C | 49.1% (42.7%–55.6%) | 43.0% | 61.3% | 70.5% / 32.6% / 29.2% | 44.2% | 23.4% | 31.8% |
| R2V | 49.6% (43.1%–56.0%) | 42.4% | 64.0% | 65.7% / 20.4% / 45.8% | 43.8% | 35.0% | 37.8% |
| R2V-ND | 52.2% (45.7%–58.6%) | 46.4% | 64.0% | 68.6% / 24.5% / 47.2% | 46.9% | 32.3% | 34.4% |

## The requirement (+1.0 pp over the adapted RAG² baseline)

R2V − R2 = **-0.4 pp** (95% CI -5.8 to +4.9; 18 questions right only with R2V, 19 only with R2; exact McNemar p = 1.0000).

Reading (pre-declared): **dev estimate only (exploratory; the requirement is read on the confirmatory split)**.

## Secondary comparisons (Holm among themselves)

| Comparison | n | difference (pp) | 95% CI (pp) | a-only / b-only | p | Holm p | confirmed |
|---|---|---|---|---|---|---|---|
| R2 vs B1 | 226 | -3.1 | -8.8 to +2.7 | 20 / 27 | 0.3817 | 1.0000 | no |
| R2V vs B1 | 226 | -3.5 | -10.6 to +4.0 | 29 / 37 | 0.3891 | 1.0000 | no |
| R2C vs R2 | 226 | -0.9 | -7.1 to +5.3 | 25 / 27 | 0.8899 | 1.0000 | no |
| R2V vs R2C | 226 | +0.4 | -4.9 to +5.8 | 18 / 17 | 1.0000 | 1.0000 | no |
| R2V vs R2V-ND | 226 | -2.7 | -5.8 to +0.4 | 4 / 10 | 0.1796 | 0.8980 | no |
| R2V vs R2 (changed) | 151 | -6.0 | -11.9 to +0.0 | 7 / 16 | 0.0931 | 0.5586 | no |

## Unsupported answers (automatic indicators)

| Arm | anachronism rate | unsupported decisive verdicts (answers with evidence) |
|---|---|---|
| B0 | 15.5% | — (0) |
| B1 | 0.0% | 9.3% (226) |
| R2 | 3.5% | 17.5% (189) |
| R2C | 3.5% | 0.0% (189) |
| R2V | 3.5% | 0.0% (189) |
| R2V-ND | 3.5% | 0.0% (189) |

## What the criteria arms changed (questions with evidence)

| Arm | valid output | verdict differs from R2 | changes that fixed / broke an answer |
|---|---|---|---|
| R2C | 100.0% | 33.9% | 25 / 27 |
| R2V | 100.0% | 23.3% | 18 / 19 |
| R2V-ND | 100.0% | 21.7% | 19 / 14 |

## Retrieval (descriptive)

| Arm | admitted | none admitted | SR/MA / RCT / other | update-window share | mean age (y) | overlap with B1 | directness@k |
|---|---|---|---|---|---|---|---|
| B1 | 5.0 | 0.0% | 9.2% / 20.4% / 70.3% | 46.6% | 10.9 | — | 28.1% |
| R2 | 3.07 | 16.4% | 25.2% / 37.5% / 37.2% | 47.9% | 9.57 | 22.7% | 62.5% |
| R2C | 3.07 | 16.4% | 25.2% / 37.5% / 37.2% | 47.9% | 9.57 | 22.7% | 62.5% |
| R2V | 3.07 | 16.4% | 25.2% / 37.5% / 37.2% | 47.9% | 9.57 | 22.7% | 62.5% |
| R2V-ND | 3.07 | 16.4% | 25.2% / 37.5% / 37.2% | 47.9% | 9.57 | 22.7% | 62.5% |

Label-stable subset (188 items, descriptive): R2V − R2 = -1.1 pp; R2 − B1 = -3.2 pp.

Alzheimer's case study (4 items, descriptive): correct answers per arm: B0 0, B1 1, R2 2, R2C 3, R2V 3, R2V-ND 3.

A result is *confirmed* only if the p-value (Holm-adjusted in a family) is below .05, the 95% interval excludes 0 and the difference is positive. With about 500 questions only differences of roughly 4–6 pp can be confirmed (docs/experiment_plan.md §7).
