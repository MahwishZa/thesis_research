# P0 diagnostics, dev pools

## 1. Stage-1 helpfulness inputs (question after the abstract, cut at 512 tokens)

* 837 of 4520 inputs (18.5%) exceed 512 tokens; median 379.0.
* Among B2's admitted papers: 352 of 1130 (31.1%) were over the limit.
* Any input over the limit lost the question and the "Answer yes or no" line, so its helpfulness score is not a judgement of that question. Stage-1 conclusions about recency are unaffected (they compare arms sharing the same scores); the claim that the helpfulness score is a weak selector is confounded by this.

## 2. What the stance step would read

* 36.4% of 4520 candidates have labelled RESULTS or CONCLUSIONS sections (the rest contribute their last three sentences); median 89.0 words, maximum 202.

## 3. Study types

* Pool: {'SR/MA': 0.0571, 'RCT': 0.2254, 'other': 0.7175}; top 8: {'SR/MA': 0.0774, 'RCT': 0.2163, 'other': 0.7063}.
* Items with a systematic review or meta-analysis in the top 8: 41.6%; in B1's admitted five: 33.6%.
* Study-type weight has material to act on (>= 30% of items): **yes**.

| B1 evidence | items | B1 verdicts | gold verdicts |
|---|---|---|---|
| with sr ma in b1 | 76 | {'SUPPORTED': 46, 'NOT ENOUGH INFORMATION': 25, 'REFUTED': 5} | {'SUPPORTED': 36, 'NOT ENOUGH INFORMATION': 24, 'REFUTED': 16} |
| without sr ma in b1 | 150 | {'REFUTED': 4, 'SUPPORTED': 84, 'NOT ENOUGH INFORMATION': 62} | {'REFUTED': 33, 'SUPPORTED': 69, 'NOT ENOUGH INFORMATION': 48} |
