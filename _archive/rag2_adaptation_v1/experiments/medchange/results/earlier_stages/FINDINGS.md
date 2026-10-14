# Findings (confirmatory split, run once)

Each conclusion is chosen by the rules fixed in `docs/experiment_plan.md` before the data were opened.

## RQ1: does as-of retrieval beat no evidence?  **not confirmed**

B1 minus B0 = +1.5 pp (95% CI -2.8 to +6.1; Holm p = 0.5543).

## RQ2: does the synthesis layer beat ordinary retrieval-augmented answering?  **not run**


## What this means

The synthesis layer was not tested on the confirmatory split because gate 2 failed on dev; the thesis reports RQ1 and the negative dev evidence for RQ2.

Label reproducibility: another model agrees with 81.4% of the gold labels (kappa 0.7161); this limits what any accuracy number can mean.
Stated-verdict consistency (independent judge): 97.7%; parse rate 100.0%.

Full tables: `stage2_analysis_confirm.md`.
