# Repository audit (2026-11-24)

| | |
|---|---|
| Document status | **A dated snapshot** (2026-11-24), written before the LLM-only test runs. Findings that were fixed are marked; the current state is in `results/STATUS.md`. |
| Update the same day | Proposals P2 (`evaluate.py collect`), P3 (context precision dropped) and P4 (dormant flag code removed) were approved and built. P1 (one launcher): `run.py` takes the system as its first word; today only `llm-only` exists. Standard RAG (P6, k = 3) is approved and not built; the proposed system and RAG² wait for the researcher's design (D2, D3). |

Scope: every file under `src/`, `tests/`, `data/` (tables, views, corpus card), `requirements.txt`, `results/`. `archive/` was not used.
Marks: ✔ = checked by running something in this session (the check is named); ◻ = assumed or not checked.

## 1. What was checked and found correct

| Area | Check | Result |
|---|---|---|
| Splits | Read `data/views/*/{train,val,test}.jsonl` | ✔ train 1,797, val 160, test 240; val 20 and test 30 per class |
| Leakage | Ids and lesion groups compared across all pairs of splits | ✔ 0 shared ids, 0 shared lesion groups (train/val, train/test, val/test) |
| Three views | Same cases, options, answer letters, gold labels and groups in image_only, text_only, multimodal | ✔ identical for every split |
| Label in text | Test questions whose patient text names a diagnosis | ✔ 0 of 240 |
| Metric formulas | Recomputed from the definition on 400 random answers: macro-F1, specificity, McNemar exact (against exact fractions), Holm (textbook example), nDCG@3, MRR, Precision@3, Hit@3, Recall@3, context precision | ✔ all equal |
| Tests | `python -m unittest discover -s tests` | ✔ 206 pass, 7 skipped (the sandbox has no pandas or scikit-learn) |
| Lint | `ruff check --select F,E9 src tests` | ✔ clean |
| Dead code | Every top-level function and class is used by `src/` or by a test | ✔ none unused |
| Corpus size for nDCG | Passages tagged per class: MEL 1123, NV 631, BCC 576, AK 476, BKL 368, DF 257, VASC 279, SCC 583 (`corpus_card.json`) | ✔ at least 10 per class, so the nDCG@10 "ideal ranking" assumption holds |
| Prompt | Three views get the right wording about where findings may come from | ✔ tests plus the real gpt-4o and gemini trials |
| Live API | gpt-4o and gemini-3.8-flash, text_only, 8 and 10 questions | ✔ answers parsed, no invalid answers |

## 2. What is missing (the main gap)

| # | Gap | Effect |
|---|---|---|
| G1 | **Only the LLM-only stage exists.** `run.py` has one command (`llm-only`). There is no retriever, no index, no RAG prompt, no RAG runner. | Standard RAG, RAG² and the proposed system cannot be run yet |
| G2 | The "proposed" system is not defined anywhere in the repository (`README.md` is a template: "Existing methods do X…") | I cannot build it without inventing it. It needs the researcher's design and the supervisor's agreement |
| G3 | RAG² (Sohn et al., NAACL 2025) is not implemented | See section 5: what the paper does and what I can and cannot reproduce |
| G4 | Claude (`claude-sonnet-5-5`) and llama-4-maverick are not yet tried live | ◻ unverified until the validation trials are done |
| G5 | No image view has been tried live | ◻ unverified (the code path is tested with fake clients only) |
| G6 | No single place to collect results across systems, models and views | See proposal P2 |

## 3. Issues found, and what was done

| # | Issue | Action |
|---|---|---|
| I1 | `requirements.txt` still said scikit-learn gives "the stop-word list of the Jaccard relevance score" (Jaccard was removed) | Fixed (comment only) |
| I2 | `results/STATUS.md` header said "Updated 2026-11-22", and its "Done" section quotes the old split sizes | Header refreshed; the dated sections stay as the history |
| I3 | `diagnosis_metrics` had a dormant block for a yes/no "evidence_sufficient" flag. No prompt produces the flag any more | **Removed** (P4, approved). No result changes |
| I4 | Macro-F1 on a **trial** with few answered questions is small, because classes that were never asked count as 0 (same as scikit-learn with all 8 labels) | No change; trial runs are not results. Full runs are balanced, so it does not matter there |
| I5 | Semantic similarity is measured against the reference texts of the **true** class, so it is partly tied to accuracy (a right answer tends to talk about the true class) | Documented as a limit; no change |
| I6 | The corpus is not balanced across classes (MEL 1,123 passages, DF 257). A retriever may lean to melanoma | Documented as a limit. The test set is balanced, so accuracy is not affected by prevalence |
| I7 | Claude is called with no temperature (the API rejects it), so Claude answers are not exactly repeatable. The other three use temperature 0 and a seed | Documented as a limit; each run records its settings |
| I8 | Llama is pinned to the host `parasail/fp8` (8-bit weights). Results are for that host, not for the model "in general" | Documented as a limit; recorded in each line (`served_by`) |
| I9 | The README is a template | Not touched (the researcher's text; a test keeps it to "Overview" and "Objectives"). Proposal P5 |

No data-leakage defect, no wrong metric formula and no dead code was found.

## 4. Metric audit (retrieval, generation, diagnosis, hallucination kept apart)

| Layer | Metric in the code | Established? | Source | Verdict |
|---|---|---|---|---|
| Diagnosis | Accuracy (top-1) with a lesion-bootstrap 95% interval | Yes | standard; bootstrap: Efron & Tibshirani (1993) ◻ | Keep |
| Diagnosis | Macro-F1, per-class precision, recall, specificity, F1, confusion matrix | Yes | standard | Keep |
| Diagnosis | McNemar exact test with Holm correction | Yes | McNemar (1947); Holm (1979) | Keep |
| Diagnosis | Invalid answers and refusals, counted separately | Reporting rule | - | Keep |
| Retrieval | Hit@k, Precision@k, Recall@k, MRR, nDCG@k | Yes | BEIR (Thakur et al., 2021); nDCG: Järvelin & Kekäläinen (2002) | Keep. Weak labels: a passage is relevant when its tags contain the true class |
| Retrieval | Context precision | Yes (RAGAS name) | Es et al. (EACL 2024) | Keep, but it is a rank-weighted precision, so it repeats Precision@k and MRR in part. Candidate to drop (P3) |
| Retrieval | MedCPT context relevance (`--relevance-models`) | Yes | Jin et al. (2023) | Keep (approved) |
| Generation | Semantic similarity of the reasoning to class reference texts | Established method (sentence embeddings), project-specific use | Reimers & Gurevych (2019); model `pritamdeka/S-PubMedBert-MS-MARCO` | Keep as a supporting measure only |
| Faithfulness | Quote accuracy (the quote is word for word in the cited passage) | Exact check; the idea follows ALCE and KILT | Gao et al. (2023); Petroni et al. (2021) | Keep |
| Faithfulness | Supported accuracy (right and quote verified) | Follows KILT-AC | Petroni et al. (2021) | Keep |
| Hallucination | Coverage and selective risk with a verified quote | Selective classification | Geifman & El-Yaniv (2017) | Keep |
| Hallucination | Negative rejection ("Evidence: none" when nothing relevant was retrieved) | Yes | RGB, Chen et al. (AAAI 2024) | Keep |
| Hallucination | Image claims in the text-only view (word list) | **Project-specific**, not a published metric | - | Keep as an approximate, labelled check (approved) |
| Cost | Tokens, seconds | Common | - | Keep |

Removed so far (all on the researcher's decisions): top-k accuracy, balanced accuracy, Jaccard, RAGAS, the NLI support check (failed its own pass mark twice: 0.83, then 0.50), Exact Match (never built).

**Hallucination honesty note:** no published metric measures "this reasoning about a skin photograph is false" without an expert. Every hallucination number in this project is a proxy (a quote that is real, a retrieved passage that is relevant, a described image that does not exist). The thesis should say so.

## 5. RAG² in one paragraph (what I can state, what I cannot) ◻

Sohn et al. (NAACL 2025) propose RAG² for medical multiple-choice QA: it uses rationales (LLM-generated explanations) to build the retrieval query, a retriever trained with perplexity-based labels (a passage is useful if it makes the LLM more certain of the right answer), and balanced retrieval from several corpora. ✔ the abstract was seen in the literature search; ◻ the training recipe and the exact settings were not read in the paper. To reproduce it fairly I must read the paper itself, and training the retriever needs a labelled training set and GPU time. Whether to reproduce it fully or use its published components without training is the researcher's decision with the supervisor (D3 below).

## 6. Proposals (status in the update at the top)

| ID | Proposal | Why | Cost |
|---|---|---|---|
| P1 | **One launcher.** `python src/run.py <system> --model <m> --view <v> --split <s>` with `system` in `llm-only`, `standard-rag`, `rag2`, `proposed`; the same folder layout and the same `run.json` for all four | One way to run everything, fair comparison | Small once the RAG runners exist |
| P2 | **Automatic results.** After each run, write `report.json` next to `predictions.jsonl`; a command `python src/evaluate.py collect` builds `results/summary.csv` with one row per system, model, view and split (accuracy, macro-F1, retrieval, evidence, cost, status) | A thesis table without copying numbers by hand | Small |
| P3 | Drop context precision (it repeats Precision@k and MRR in part) | Fewer, clearer metrics | Tiny |
| P4 | Remove the dormant yes/no flag code and its tests | No prompt produces the flag | Small |
| P5 | Write `README.md` from the verified facts after the first full runs (I will not write results before they exist) | The README is a template | Small |
| P6 | Standard RAG: dense retriever (the MedCPT query and article encoders, already downloaded for relevance scoring), top-k passages from `passages.jsonl`, one RAG prompt, the model gives "Evidence: [n] "quote"" or "Evidence: none" | The simplest defensible baseline; no training; the corpus and encoders already exist | Medium |

## 7. Decisions needed from the researcher (D1 answered; D2 and D3 open)

- **D1.** Approve P1 to P4 (yes or no for each).
- **D2.** The proposed system: what does it do differently from standard RAG and from RAG²? It cannot be designed without the researcher. If it is not decided, the thesis can still report LLM-only and standard RAG first.
- **D3.** RAG²: reproduce the full method, or only its published retrieval ideas (rationale as query, balanced corpus mix), with no retriever training? I recommend the second given 5 to 7 days and a small budget, stated clearly as "RAG²-style" in the thesis. The supervisor should confirm this wording.
- **D4.** Standard RAG: approve P6 as the design?
- **D5.** Number of retrieved passages k (I suggest k = 3, because each passage is up to 200 words and the prompt stays small and cheap).

## 8. Five-to-seven-day plan (cheapest order)

| Day | Work | Paid calls |
|---|---|---|
| 1 | Finish the 10-question validation of the 4 models (text_only); then 1 image trial per model | about 40 + 20 |
| 1-2 | Build standard RAG (P6) and its tests; dry-run free | 0 |
| 2 | Validate standard RAG on 10 questions per model | about 40 |
| 2-3 | LLM-only test runs, text_only first (4 models x 240) | about 960 |
| 3-5 | RAG² style and the proposed system, if D2 and D3 are settled | per D2, D3 |
| 5-6 | Test runs of RAG systems, evaluation, `collect`, thesis tables | per budget |
| 6-7 | README results, remaining documentation, final checks | 0 |

I cannot price any of this ◻ (no per-token prices were looked up). Check the budget against: LLM-only text_only about 216 tokens in and 82 to 284 out per question (measured in the trials); a RAG prompt adds about 3 x 200 words of passages (about 800 tokens) per question (my estimate ◻). Images add image tokens (not measured).

## 9. What not to claim in the thesis

- That any hallucination number is "the hallucination rate" (they are proxies).
- That results hold for the "model" in general (they hold for the pinned host and snapshot).
- That the quote check proves support (it proves the quote exists in the passage).
- Anything from the 8- and 10-question trials.
