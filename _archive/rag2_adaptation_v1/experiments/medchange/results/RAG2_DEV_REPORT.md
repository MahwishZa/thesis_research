# Realigned study: dev report (exploratory)

Dev split only (226 items); nothing here is a confirmatory result. Full tables: `rag2_analysis_dev.md`.

* Accuracy: B0 45.1%, B1 53.1%, R2 50.0%, R2C 49.1%, R2V 49.6%, R2V-ND 52.2%.
* R2V − R2: -0.4 pp (95% CI -5.8 to +4.9).

## Pre-declared dev check

* (a) every arm parses ≥ 95% and the verifier output is valid ≥ 95%: **PASS**
* (b) the adapted RAG² baseline works (R2 ≥ B1 − 5 pp): **PASS**
* (c) direction (R2V − R2 ≥ 0): **FAIL**

Status: **REVISE ONCE**. 
The plan allows one recorded revision of the verification prompt on dev; then the design is frozen whatever dev shows.
