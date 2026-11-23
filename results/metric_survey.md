# Evaluation metrics of RAG research: what the field measures and what fits this thesis

| | |
|---|---|
| Document status | **Literature survey, a snapshot of 2026-11-23.** It informed the choice of metrics; it does not list the metrics in use. |
| The metrics in use | `results/evaluation_guide.md` |
| Decisions taken after this survey | Removed: Exact Match, Jaccard, RAGAS, context precision, the NLI support check (failed its own pass mark twice), balanced accuracy and top-k accuracy. Added: Recall@k, an approximate text-only image-claim check, a semantic-similarity validity check. See `results/CHANGELOG.md`. |

Written 2026-11-23, before the evaluation protocol is finalised. Method: web searches (the pages themselves could not be opened from
this environment, so what is written here comes from search result summaries). ✔ = stated in a result I retrieved; ◻ = from memory or
not seen: check it in the paper before citing. Nothing here has been run.

## 1. What the field measures

| Layer | What it asks | Typical metrics | Where |
|---|---|---|---|
| Task result | Is the final answer right? | Accuracy (multiple choice), EM, F1 | MIRAGE ✔, RAG² ✔, KILT ✔, RGB ✔ |
| Retrieval | Did the retriever find the right passages? | Recall@k, Precision@k, nDCG@10, MRR, R-precision | BEIR ✔ (nDCG@10), KILT ✔ (R-precision), surveys ✔ |
| Context relevance | Is each passage useful for the question? | RAGAS / ARES context relevance (LLM judge), RAGChecker context precision (entailment) | RAGAS ✔, ARES ✔, RAGChecker ✔ |
| Faithfulness / attribution | Are the claims supported by the retrieved passages? | RAGAS faithfulness, ALCE citation recall and precision, FActScore, AIS (human), AttrScore, AlignScore | ✔ all |
| Hallucination | Does the answer contain unsupported or false content? | Hallucination rate (RAGTruth, human word-level labels), CRAG score (+1 / +0.5 / 0 / -1) | RAGTruth ✔, CRAG ✔ |
| Abstention and robustness | Does it say "don't know" when it should; does noise or false context break it? | Rejection rate, error detection, noise robustness (RGB), unanswerable ratio (UAEval4RAG), sufficiency / integration / robustness (MedRGB) | ✔ |
| Efficiency | Cost and time | Tokens, latency | common |

## 2. Paper by paper

| Paper (venue) | Domain | What it reports | Note |
|---|---|---|---|
| Yu et al., *Evaluation of RAG: a survey* (2024) | survey | Compares metrics of the retrieval and generation parts (relevance, accuracy, faithfulness) | States that plain recall and precision "cannot fully capture" RAG retrieval ✔ |
| *Ask in Any Modality*, multimodal RAG survey (2025) | multimodal | About 60 different metrics in use; Recall@K preferred; EM, BLEU, ROUGE, METEOR for text; clinical relevance in medical work | No single standard ✔ |
| Xiong et al., MIRAGE / MedRAG (ACL Findings 2024) | medicine, multiple choice | **Accuracy** on 7,663 questions; gains of 1 to 18% from RAG; "lost in the middle" effect | Closest to this thesis in format ✔ |
| Sohn et al., RAG² (NAACL 2025) | medicine, multiple choice | Accuracy on medical QA benchmarks (gain up to 6.1%) | The base paper ✔ (abstract only) |
| Ngo et al., MedRGB (2024) | medicine | Four scenarios probing sufficiency, integration and robustness to noise and misinformation, 3,480 instances | Shows models are weak against noise and misinformation ✔ |
| MMed-RAG (ICLR 2025) | medical images, VQA and reports | "Factual accuracy" (average gain 43.8%) on five datasets | The metric definition was not seen ◻ |
| Chen et al., RGB (AAAI 2024) | general | Accuracy for noise robustness and information integration; rejection rate for negative rejection; error detection and correction for counterfactuals | ✔ |
| Yang et al., CRAG (NeurIPS 2024) | general | Each answer is perfect (+1), acceptable (+0.5), missing (0) or incorrect (-1); the mean is "truthfulness" | Penalises hallucination and rewards abstaining ✔ |
| Petroni et al., KILT (NAACL 2021) | knowledge tasks | R-precision for the provenance; KILT-AC, -EM, -F1 count an answer **only if the provenance is right** | "Answer counts only with correct evidence" ✔ |
| Es et al., RAGAS (EACL 2024 demos) | general | Faithfulness (claims inferable from the context), answer relevance, context relevance; the library adds context precision and recall | LLM judge ✔ |
| Saad-Falcon et al., ARES (NAACL 2024) | general | Context relevance, answer faithfulness, answer relevance with trained judges | Reported to beat RAGAS by 59.3 and 14.4 points on two tasks ✔ |
| Ru et al., RAGChecker (NeurIPS 2024) | general | Claim-level entailment: claim recall and context precision (retriever); faithfulness, noise sensitivity and others (generator) | Exact generator formulas not seen ◻ |
| Friel et al., RAGBench / TRACe (2024) | five domains incl. biomedical | Utilisation, relevance, adherence, completeness | A fine-tuned 400M DeBERTa judge beat few-shot LLM judges ✔ |
| Niu et al., RAGTruth (ACL 2024) | general | About 18,000 responses with human word-level hallucination labels; hallucination rate per model | A corpus to test detectors ✔ |
| Gao et al., ALCE (EMNLP 2023) | general | Citation recall and precision, judged by an NLI model (TRUE, T5-11B) | Agreement with humans 85.1% (recall), 77.6% (precision); cannot see partial support ✔ |
| Rashkin et al., AIS (Comput. Linguistics 2023) | general | Human protocol: is the statement supported by the source? | The reference standard for attribution ✔ |
| Yue et al., AttrScore (Findings EMNLP 2023) | general | Attributable / extrapolatory / contradictory; prompted or fine-tuned judges | ✔ |
| Zha et al., AlignScore (ACL 2023) | general | A 355M alignment model for factual consistency; matches or beats ChatGPT/GPT-4 based metrics | Free and local ✔ |
| Min et al., FActScore (EMNLP 2023) | long text | Share of atomic facts supported by a knowledge source | ✔ |
| Asai et al., Self-RAG (ICLR 2024) | general | Citation precision and recall, FactScore, correctness | ✔ |
| Chen et al., NLI to verify QA (Findings EMNLP 2021) | QA | Turns question + answer into a statement and tests it for entailment against the passage | The method behind a "passage supports the answer" check ✔ |
| Pal et al., Med-HALT (CoNLL 2023) | medicine | Reasoning tests (false confidence, none of the above, fake questions) and memory tests | Tests a model's own knowledge, not retrieval ✔ |
| Moon and Hong (2025, arXiv) | dermoscopy | Melanoma classification with retrieved similar cases and clinical metadata, no fine-tuning | Binary; no multi-class benchmark was found ✔ |
| RefusalBench, UAEval4RAG (2024-2025) | general | Refusal accuracy; unanswerable and acceptable ratios | Abstention is scored by rates, not coverage-risk curves ✔ |

## 3. What stands out

1. **Accuracy is the one metric every paper has.** For a fixed-option question, the free-text metrics (EM, BLEU, ROUGE, METEOR) add nothing.
2. **Hallucination in RAG is measured as support, not as truth.** The shared idea is attribution: is each claim supported by the cited or retrieved text (RAGAS, ALCE, RAGChecker, AIS, AlignScore). Whether the world agrees is not tested.
3. **LLM judges are common but not the best tool.** RAGBench found a small fine-tuned judge better than few-shot LLM judges ✔. A 2026 biomedical study (secondary source ◻) says faithfulness alone is an unreliable quality signal and must be read with answer relevance.
4. **NLI attribution has a known limit:** it misses partial support (ALCE ✔). A "supported" flag is therefore an upper bound.
5. **Gating an answer on its evidence is established** (KILT-AC ✔), and **penalising confident wrong answers** is too (CRAG ✔). Reporting coverage and risk for RAG is unusual: the search found no RAG benchmark that does it.
6. **Robustness to noise and misleading passages** is a standard test (RGB, MedRGB ✔) and bears directly on hallucination.
7. **Images:** no standard metric checks whether a sentence about a picture is true. Medical multimodal RAG papers report accuracy-type scores. For this thesis the image side is judged only by the answer, not by the reasoning ◻.

## 4. Fit for this thesis

The thesis compares LLM-only, standard RAG, RAG² and a proposed RAG on 8-class dermoscopy questions in three views, with no clinician, so every metric must be automatic, and preferably free and fixed.

| Metric | Source | Fit | Why |
|---|---|---|---|
| Accuracy, macro-F1, per-class scores | standard | **Core** | The result; works in all three views |
| Paired McNemar, Holm; cases fixed or broken by retrieval | McNemar, Holm, RAGChecker idea | **Core** | The comparison between systems |
| Hit@k, Precision@k, nDCG@k | BEIR, KILT, surveys | **Core (RAG)** | Retrieval quality with weak labels from the corpus tags |
| Quote accuracy (the quote is in the cited passage) | extractive check | **Core (RAG)** | Cheap, exact; turns a made-up quote into a countable event |
| ~~Support of the answer by the cited passage (NLI)~~ | Chen 2021, ALCE | **Removed 2026-11-24** | Failed its pre-set pass mark twice (0.83 pairwise, 0.50 top-1); only the exact quote check remains |
| Supported accuracy (correct and supported) | KILT-AC | **Core (RAG)** | A correct answer without evidence does not count |
| Coverage, selective risk with verified evidence | selective classification | **Core (RAG)** | The hallucination-at-answer-level view |
| Negative rejection | RGB | **Core (RAG)** | The model should say "none" when nothing relevant is retrieved |
| Semantic similarity to the reference texts | embeddings | Supporting | Says whether the reasoning talks about the right disease, not whether it is true |
| Context relevance (MedCPT cross-encoder) | MedCPT | Supporting | Measures the retriever with a trained biomedical model |
| RAGAS faithfulness on a small sample | RAGAS | **Removed 2026-11-24** | LLM judge, costs credit, not exactly reproducible |
| Noise and misleading-passage test | RGB, MedRGB | **Add as an experiment** | Shows whether RAG reduces or adds hallucination |
| CRAG score (+1 / 0 / -1) | CRAG | Optional | One number combining coverage and risk |
| Refusals and other invalid answers | - | Core | Already reported |
| Context recall (RAGAS) | RAGAS | Optional | Needs a reference answer; the reference texts are class-level, not case-level |
| Claim-level RAGChecker metrics | RAGChecker | Skip | Needs an LLM to extract claims; the reasoning is only 3 sentences |
| TRACe, RAGTruth, Med-HALT, AIS | - | Skip | They are datasets or human protocols, not scores we can compute |
| BLEU, ROUGE, METEOR, EM, BERTScore | - | Skip | The answer is a letter |
| Jaccard of content words | - | Skip | Not used by any paper above ✔; weaker than MedCPT |

## 5. Limits to state in the thesis

- All "support" scores are an automatic upper bound (partial support is missed); no clinician reads a case.
- The semantic similarity measures resemblance to class-level reference texts, not correctness of a single case.
- Claims about what is visible in the image are not verified in any view.
- Prevalence is balanced in validation and test, so precision is not the precision of a clinic.
