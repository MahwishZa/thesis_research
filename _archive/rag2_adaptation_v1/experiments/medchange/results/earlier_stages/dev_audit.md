# Dev audit (descriptive; recomputed from the committed answers)

## Constant answers

| Always answer | accuracy, changed items | accuracy, all dev items |
|---|---|---|
| SUPPORTED | 44.4% | 46.5% |
| REFUTED | 24.5% | 21.7% |
| NOT ENOUGH INFORMATION | 31.1% | 31.9% |

Changed items: n = 151; all dev items: n = 226.

## Per-class recall and abstention, changed items

| Arm | accuracy | recall SUPPORTED | recall REFUTED | recall NOT ENOUGH INFORMATION | share answering NOT ENOUGH INFORMATION |
|---|---|---|---|---|---|
| B0 | 41.1% | 80.6% | 2.7% | 14.9% | 21.2% |
| B1 | 49.7% | 68.7% | 10.8% | 53.2% | 39.7% |
| B2 | 37.1% | 73.1% | 0.0% | 14.9% | 22.5% |
| B3 | 45.7% | 58.2% | 8.1% | 57.4% | 47.7% |
| P | 35.8% | 62.7% | 0.0% | 25.5% | 32.5% |
| C1 | 37.7% | 67.2% | 0.0% | 25.5% | 31.8% |

## Order sensitivity (pairs of evidence arms, same question, all dev items)

| Admitted lists | pairs | same verdict |
|---|---|---|
| identical list | 17 | 100.0% |
| overlap 0.25 to 0.66 | 1515 | 80.1% |
| overlap below 0.25 | 656 | 63.9% |
| same papers, different order | 72 | 86.1% |

Identical lists in identical order give identical generated text in 8 of 17 pairs (greedy decoding is not bitwise reproducible).

## Mean age of B1's evidence by gold class (years)

| Gold | items | mean age |
|---|---|---|
| SUPPORTED | 105 | 10.6 |
| REFUTED | 49 | 11.4 |
| NOT ENOUGH INFORMATION | 72 | 10.7 |

## Does refitting help? (5-fold cross-validation, 50 repeats, seed 20261003, the stage-2 fitting)

B1 raw accuracy: 53.1% on all 226 dev items, 49.7% on 151 changed items.

| Fitted on | accuracy, all | accuracy, changed |
|---|---|---|
| constant prior | 46.5% | 44.4% |
| B1 verdict | 52.2% | 48.6% |
| B1 verdict + evidence age | 49.6% | 47.1% |
| evidence age only | 44.9% | 42.0% |

Majority vote of B1, B2, B3: 47.0% on changed items, 62.7% on unchanged items.

