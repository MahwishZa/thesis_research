# Research Glossary

Terms as used consistently throughout this repository, code and documentation.

## The comparison

| Term | Meaning |
|---|---|
| **RAG²** | Sohn et al., NAACL 2025: a retrieval-augmented medical QA method whose core is a Flan-T5 filter, trained on perplexity-derived labels, that decides which retrieved passages the LLM sees from text alone. Its checkpoint is not distributed. |
| **Temporal Filter** | The stage-1 proposed system: admission by A(s) = (1 − λ)·ρ(s) + λ·T(s, q, t_q), i.e. relevance plus how recent a passage is relative to the question date. Short name for "recency-weighted evidence admission"; not a claim of a new method. Negative result on the dev split. |
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
| **Selected hybrid** | H0 unless a weighted variant's dev cross-validated accuracy is at least 1.0 pp higher; recorded in the frozen model file before any confirmatory stance run. |

## Scoring

| Term | Meaning |
|---|---|
| **ρ(s)** | Relevance: the within-pool rank of the arm's relevance signal, normalised so the best passage is 1 and the worst 0. |
| **T(s, q, t_q)** | Temporal score 2^(−age_days / H): 1.0 for a passage published on the question date, halving every H days; age is clamped at zero. |
| **λ** | Weight on the temporal term, 0 to 1 (0.5 in the experiments; fixed, not fitted). |
| **H** | Half-life in days (1,095 in the experiments). |
| **θ** | Admission threshold used by the framework's threshold-and-budget policy; **not used** by the MedChange arms. |
| **A(s)** | The admission score (1 − λ)·ρ(s) + λ·T(s, q, t_q), computed by `src/proposed/`. |
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
| **Gate (G0–G3)** | A pre-stated pass/fail check on the dev split that decides whether to continue (`experiment_plan.md` §9). |
| **RQ1, RQ2** | The two pre-specified stage-2 questions. RQ1: does as-of retrieved evidence (B1) make verdicts more accurate than none (B0)? RQ2: does the selected hybrid beat B1R? Tested once, on the confirmatory split. |
| **P0, P1 (gate 1), P2 (gate 2)** | The stage-2 dev checks, in order: P0 diagnostics (no model), the 40-item stance pilot with a hand check (gate 1), the full dev stance run with the fitted layer (gate 2). Thresholds are in `experiment_plan.md` §9. |
| **Hand check** | The researcher's own S / C / N labels of 40 pilot papers, compared with the model's stance (gate 1); it also picks the wording. A sanity check, about ±14 pp wide, not a gold standard. |
| **Irrelevant-paper control** | Each pilot question judged against papers belonging to another item; a stance step that reads the paper should say "neither". |
| **Frozen model** | `results/synthesis_model.json`: the coefficients, selected hybrid and stance wording fitted on dev, committed before any confirmatory stance run. |
| **Forking-path ledger** | The dated list in `experiment_plan.md` §13 of every design decision taken after seeing dev data. |
| **Question pool (Alzheimer's)** | The 113 usable human-reviewed Alzheimer's questions, split into validation (23) and test (90). |
| **`temporal_candidate`** | Alzheimer's-pool flag: the cited Cochrane review has been revised at least once. Does not assert that the verdict changed. |
| **Provenance firewall** | The rule (and automated check) that a question's reference evidence never appears among its retrieval candidates. |
| **Corpus snapshot** | An id identifying exactly which corpus/index build a frozen item was produced against. |
| **Frozen candidate set / `FrozenItem`** | The framework's cached retrieval record for one question (`evaluation/freezing.py`). |

## Outcomes

| Term | Meaning |
|---|---|
| **Verdict accuracy** | Share of items whose parsed `VERDICT:` equals the gold newest verdict. The primary outcome. |
| **Outdated-verdict rate** | Share of changed items whose verdict equals the previous version's. |
| **Per-class recall, macro-F1, NOT ENOUGH INFORMATION share** | Recall of each gold class; the mean F1 of the three classes; how often an arm answers NOT ENOUGH INFORMATION. They show whether an arm gains accuracy by reading evidence or only by abstaining more. |
| **Manipulation check** | A measure of what the mechanism does (update-window share, passage age, overlap with B1); never an outcome. |
| **Currency** | Historical: the mean temporal score of admitted evidence, formerly the primary metric. Circular for this system, so retired; survives only in `_archive/alzheimers_pilot_v1/`. |
| **Groundedness** | Fraction of an answer's content tokens present in its admitted evidence; a diagnostic, not an entailment judgement. |
| **Context precision / recall** | Admitted evidence against known gold-relevant evidence where annotated; `None`, never 0.0, when unannotated. |
| **Paired McNemar / bootstrap** | The exact test on discordant paired outcomes and the item-resampled confidence interval used to judge differences. |
| **Extractive stand-in generator** | Historical placeholder that returned the top passage verbatim; used only by the superseded pilot and the fixture demo. |
