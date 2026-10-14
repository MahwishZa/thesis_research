# Error analysis, dev split, questions whose verdict changed (n = 151)

Each wrong answer is assigned one cause (definitions: top of `experiments/medchange/error_analysis.py`).

The candidate pool contained at least one update-window passage for 100.0% of these questions, so for the rest no admission rule could have helped.

## Where each arm's answers end up (counts)

| Arm | correct | parse failure | retrieval miss | admission miss | evidence admitted, still wrong |
|---|---|---|---|---|---|
| B0 | 62 | 0 | 0 | 89 | 0 |
| B1 | 75 | 0 | 0 | 13 | 63 |
| B2 | 56 | 0 | 0 | 12 | 83 |
| B3 | 69 | 0 | 0 | 1 | 81 |
| P | 54 | 0 | 0 | 2 | 95 |
| C1 | 57 | 0 | 0 | 11 | 83 |

## Does admitted update-window evidence help?

| Arm | accuracy when it admitted such a passage (n) | accuracy when it did not (n) |
|---|---|---|
| B0 | n/a (0) | 41.1% (151) |
| B1 | 51.9% (131) | 35.0% (20) |
| B2 | 38.1% (134) | 29.4% (17) |
| B3 | 45.6% (149) | 50.0% (2) |
| P | 36.2% (149) | 0.0% (2) |
| C1 | 38.5% (135) | 31.2% (16) |

## Accuracy by type of change

| Arm | involves NOT ENOUGH INFORMATION | decisive flip |
|---|---|---|
| B0 | 38.8% (116) | 48.6% (35) |
| B1 | 50.9% (116) | 45.7% (35) |
| B2 | 36.2% (116) | 40.0% (35) |
| B3 | 49.1% (116) | 34.3% (35) |
| P | 35.3% (116) | 37.1% (35) |
| C1 | 37.9% (116) | 37.1% (35) |

## Which verdicts each arm gives (counts) and how often it repeats the outdated verdict

| Arm | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | none | outdated-verdict rate |
|---|---|---|---|---|---|
| B0 | 117 | 2 | 32 | 0 | 36.4% |
| B1 | 84 | 7 | 60 | 0 | 31.8% |
| B2 | 116 | 1 | 34 | 0 | 38.4% |
| B3 | 73 | 6 | 72 | 0 | 33.1% |
| P | 100 | 2 | 49 | 0 | 40.4% |
| C1 | 103 | 0 | 48 | 0 | 39.1% |
