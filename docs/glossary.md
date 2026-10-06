# Research Glossary

Terms as used consistently throughout this repository, code and documentation.

## The comparison

| Term | Meaning |
|---|---|
| **RAG²** | Sohn et al., NAACL 2025: a retrieval-augmented medical QA method whose core is a Flan-T5 filter, trained on perplexity-derived labels, that decides which retrieved passages the LLM sees from text alone. Its checkpoint is not distributed. |
| **Temporal Filter** | The stage-1 system (a result of record; since the realignment no longer the proposed system, which is R2V): admission by A(s) = (1 − λ)·ρ(s) + λ·T(s, q, t_q), i.e. relevance plus how recent a passage is relative to the question date. Short name for "recency-weighted evidence admission"; not a claim of a new method. Negative result on the dev split. |
| **Admission rule** | The function that decides which retrieved passages reach the generator. The one thing that differs between arms. |
| **Arm** | One system in the comparison: B0, B1, B2, B3, P (stage-1 proposed) or C1; in stage 2 also B1R, S0–S3, H0–H3, H1C and H3C. See `methodology.md` §4. |
| **Helpfulness signal** | P(yes) from the unmodified Flan-T5-large asked RAG²'s prompt plus "Answer yes or no." An untrained stand-in for RAG²'s filter. |
| **Falsification control (C1)** | The proposed arm with publication dates shuffled within each pool; a gain that survives it is not temporal. |
| **Budget** | The maximum number of admitted passages (5), identical across arms. |
| **Ablation** | Removing the temporal term (λ = 0) to isolate whether the temporal signal is what helps; in stage 2, comparing a hybrid with and without recency or study-type weights. |
| **Stance** | What one study alone says about the question read as a claim: supports, contradicts, or says nothing clear ("neither"). Judged by the local model from the title and the RESULTS and CONCLUSIONS of one abstract, as one letter; the three probabilities come from the first output token. |
| **Evidence-synthesis layer** | The stage-2 proposed system: the stances of the first eight pool papers, condensed into four numbers and combined with B1's verdict by a small logistic regression fitted on dev and then frozen. It never changes what the generator reads. |
| **B1R** | B1's verdict passed through the same logistic fitting as the hybrids: the fairness control that separates "stance helps" from "any fitting on dev helps". |
| **S0–S3, H0–H3** | Stage-2 arms. S: stance features only. H (hybrid): B1's verdict plus the stance features. Index: 0 no weights, 1 recency weights, 2 study-type weights, 3 both. |
| **H1C, H3C** | H1 and H3 with publication dates shuffled within each pool; falsification controls for the recency weights. |
| **Adapted RAG² (R2)** | The realigned baseline: RAG²'s three components adapted to this setting — the model's rationale as the dense query, retrieval balanced across evidence types (systematic review or meta-analysis / trial / other), and a zero-shot filter by the generator itself — answering with the standard verdict prompt. Not a RAG² reproduction. |
| **Evidence-criteria verification (R2V)** | The realigned proposed system: the same model checks R2's answer against the admitted evidence, labelled with study design and year, using fixed evidence criteria, and gives a final verdict. |
| **Evidence criteria** | Directness, design weight, the meaning of each verdict (the benchmark's labelling rubric restated) and, in the dated version, currency (newer evidence takes precedence over older evidence it may have superseded); verbatim in `rag2.CRITERIA`. |
| **R2C, R2V-ND, R2-RQ, R2-BR, R2-NF** | Realigned controls and ablations: criteria without a draft; R2V without dates and the currency clause; R2 with the question as the dense query; without balancing; without the filter. |
| **Rationale query** | A short closed-book rationale written by the generator (population, intervention, comparison, outcome, then what is known), used instead of the question as the MedCPT dense query. |
| **Selected hybrid** | H0 unless a weighted variant's dev cross-validated accuracy is at least 1.0 pp higher; recorded in the frozen model file before any confirmatory stance run. |

## Scoring

| Term | Meaning |
|---|---|
| **ρ(s)** | Relevance: the within-pool rank of the arm's relevance signal, normalised so the best passage is 1 and the worst 0. |
| **T(s, q, t_q)** | Temporal score 2^(−age_days / H): 1.0 for a passage published on the question date, halving every H days; age is clamped at zero. |
| **λ** | Weight on the temporal term, 0 to 1 (0.5 in the experiments; fixed, not fitted). |
| **H** | Half-life in days (1,095 in the experiments). |
| **θ** | Admission threshold of the removed original framework's threshold-and-budget policy; **not used** by the MedChange arms. |
| **A(s)** | The admission score (1 − λ)·ρ(s) + λ·T(s, q, t_q), computed by `src/temporal_filter/`. |
| **Paper weight w** | Stage 2: the weight of one paper in the four features: 1 (none), 2^(−age/H) (recency, age to t_q), 3 / 2 / 1 for a systematic review or meta-analysis / a controlled trial / anything else (study type), or the product. |
| **Signed stance, no-stance share, conflict, informative mass** | The four stage-2 features: weighted mean of p(supports) − p(contradicts); weighted mean of p(neither); twice the smaller of the weighted supports and contradicts totals over the total weight; ln(1 + weighted supports + contradicts). |

## Data and protocol

| Term | Meaning |
|---|---|
| **MedChangeQA** | 512 Cochrane questions whose verdict changed between review versions (Vladika et al., EMNLP 2025 Findings). |
| **Changed / unchanged item** | An item whose newest verdict differs from / equals an earlier version's. Unchanged items are the controls. |
| **Verdict** | SUPPORTED, REFUTED or NOT ENOUGH INFORMATION: the answer to a question as stated in a review's conclusions. |
| **As-of protocol** | Each question is asked at t_q, the newest review's publication date, with only evidence first public strictly before it; the review and its versions are excluded. |
| **t_q** | The question date. |
| **Update window** | The interval after the previous review version and up to the newest: when the evidence that could have changed the verdict appeared. |
| **Frozen pool** | The 20 candidates retained for one item, with date bounds and an order-sensitive hash, replayed identically to every arm. |
| **Dev / confirmatory split** | Seeded split of the MedChange items. Dev is used for gates; the confirmatory split is run once. |
| **`ad` split (Alzheimer's/dementia test set)** | 208 MedRevQA questions on dementia, Alzheimer's disease and cognitive impairment from reviews in neither dev nor confirm (by study group and by Cochrane ID); all unchanged-verdict; run once after the freeze as a secondary evaluation (`experimentation.md` §11). |
| **Gate (G0–G3)** | A pre-stated pass/fail check on the dev split that decided whether stage 1 continued (stage-1 protocol, in the repository history). |
| **Dev check (realigned)** | The pre-declared check of the realigned dev run: every arm parses ≥ 95% and the verifier is valid ≥ 95%; R2 ≥ B1 − 5 pp; R2V − R2 ≥ 0 (`experimentation.md` §8). |
| **Requirement reading** | The pre-declared reading of the supervisor's +1 pp requirement on the held-out split: met and confirmed / met as a point estimate, not confirmed / not met (`experimentation.md` §7). |
| **Design record** | `results/rag2_design.json`: every realigned setting, prompt hash and the generator file's hash, committed before the held-out run, which refuses to start if the current design differs. |
| **RQ1, RQ2** | The two pre-specified stage-2 questions. RQ1: does as-of retrieved evidence (B1) make verdicts more accurate than none (B0)? RQ2: does the selected hybrid beat B1R? Tested once, on the confirmatory split. |
| **P0, P1 (gate 1), P2 (gate 2)** | The stage-2 dev checks, in order: P0 diagnostics (no model), the 40-item stance pilot with machine checks (gate 1), the full dev stance run with the fitted layer (gate 2). Thresholds were in §9 of the stage-2 protocol (the file `experimentation.md` at commit 92e3aaf). |
| **Label audit / label-stable item** | An independent second-family model re-labels every item's conclusions with the authors' rubric; items on which it agrees with the gold label are label-stable. Measures reproducibility, not medical truth. |
| **Genuine positive** | The pre-declared strict reading of a confirmed RQ2: also better than raw B1, macro-F1 not lower, label-stable direction, recency earned (stage-2 protocol, in the repository history). |
| **Irrelevant-paper control** | Each pilot question judged against papers belonging to another item; a stance step that reads the paper should say "neither". |
| **Frozen model** | `results/synthesis_model.json`: the coefficients, selected hybrid and stance setting fitted on dev, committed before any confirmatory stance run. |
| **Forking-path ledger** | The dated list in `experimentation.md` §9 of every design decision taken after seeing data. |
| **Question pool (Alzheimer's)** | *Archived framework.* The 113 usable Alzheimer's questions reviewed earlier by the researcher (not extended), split into validation (23) and test (90). Not the `ad` split. |
| **`temporal_candidate`** | *Archived framework.* Alzheimer's-pool flag: the cited Cochrane review has been revised at least once. Does not assert that the verdict changed. |
| **Provenance firewall** | *Archived framework.* The rule (and automated check) that a question's reference evidence never appears among its retrieval candidates. |
| **Corpus snapshot** | *Archived framework.* An id identifying exactly which corpus/index build a frozen item was produced against. |
| **Frozen candidate set / `FrozenItem`** | *Removed framework.* The original framework's cached retrieval record for one question. The current pipeline's analogue is the frozen pool of `freeze_candidates.py`. |

## Outcomes

| Term | Meaning |
|---|---|
| **Verdict accuracy** | Share of items whose parsed `VERDICT:` (or, for the criteria arms, `FINAL VERDICT:`) equals the gold newest verdict. The primary outcome. |
| **Anachronism rate** | Share of answers that mention a year later than the question date's year: a claim no admitted evidence can support. |
| **Unsupported decisive verdict** | SUPPORTED or REFUTED while citing none of the admitted studies; counted only for answers that had evidence. |
| **Directness@k** | Share of an arm's admitted abstracts that an independent second-family model judges to test the question's intervention and outcome (optional). |
| **Outdated-verdict rate** | Share of changed items whose verdict equals the previous version's. |
| **Per-class recall, macro-F1, NOT ENOUGH INFORMATION share** | Recall of each gold class; the mean F1 of the three classes; how often an arm answers NOT ENOUGH INFORMATION. They show whether an arm gains accuracy by reading evidence or only by abstaining more. |
| **Manipulation check** | A measure of what the mechanism does (update-window share, passage age, overlap with B1); never an outcome. |
| **Currency** | Historical: the mean temporal score of admitted evidence, formerly the primary metric. Circular for this system, so retired; it survived only in the first pilot, now removed (Git history). |
| **Groundedness** | *Removed framework* (`rag_metrics.py`, Git history). Fraction of an answer's content tokens present in its admitted evidence; a diagnostic, not an entailment judgement. The current pipeline uses the two indicators under "Anachronism rate" and "Unsupported decisive verdict". |
| **Context precision / recall** | *Archived framework.* Admitted evidence against known gold-relevant evidence where annotated; `None`, never 0.0, when unannotated. |
| **Paired McNemar / bootstrap** | The exact test on discordant paired outcomes and the item-resampled confidence interval used to judge differences. |
| **Extractive stand-in generator** | Historical placeholder that returned the top passage verbatim; used only by the superseded pilot and the fixture demo, both removed. |
