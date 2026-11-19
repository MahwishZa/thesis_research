# Audit of the repository before further experiments (2026-11-19)

Temporary document for supervisor discussions; removed with `docs/` when the research is final. **Scope:** a read-through of every file of `src/`,
the configs, the six prompts, the committed tables and reports of the three data folders and `docs/method.md` and `docs/v2_proposal.md`;
checks run without any model, API call or download; corrections only where the fault was clear and no approved decision was touched. The
datasets were not frozen or changed. Nothing here is a result.

Marks: ✔ verified in this audit by running code or reading the file named; ◻ an assumption or a judgement, stated as one.

## 1. Problems found and what was done

| # | Problem found | Evidence | Action |
|---|---|---|---|
| 1 | **Only one of the four approaches exists.** `APPROACHES = ("closed_book",)`. Standard RAG, RAG² and the proposed method have no code: no retriever, no reranker or filter, no evidence step, no adapter for a local open-weight model. | `src/run.py`; no retrieval code in `src/` ✔ | Not changed: each needs a design decision (section 6, D1). |
| 2 | **No evidence-grounding metric exists**, and none is in the approved method. The method scores the diagnosis, the self-reported flag and invalid answers. The grounding metrics (citation precision and recall, support of claims, patient-fact check, retrieval metrics) appear only in the proposal. | `docs/method.md` section 3; `docs/v2_proposal.md` section 5.2 ✔ | Not changed (D2, D3). |
| 3 | **No check that the three modalities of a question are the same question.** Nothing compared the views before a run. | `src/` had no such check | Added `python -m src.validate`. On the real tables all 20 checks pass ✔ (section 2). |
| 4 | **A damaged answer file was scored without complaint.** A question answered twice counted twice; a missing field or an impossible answer was scored or crashed late; a run was called finished without checking that every question was answered once. | read `evaluate.score_run` and `run._run_once` ✔ | Added `evaluate.verify_answers`; used before scoring, before comparing and before a run is marked finished. Tested by 10 tests and deliberate breaks. |
| 5 | `compare` did not require the current answer format, so a run of the first phase could be compared. | `evaluate._finished` ✔ | Now refused with a pointer to the archive. |
| 6 | **A run with a broken prompt or output limit kept spending.** Invalid answers were counted, but nothing stopped the run. | `run._run_once` ✔ | If the first 10 answers are all invalid the run stops with the reason (answers kept; run the same command to go on). |
| 7 | Unused or duplicated code: `TITLES` (no use anywhere), `evaluate._brief` (an alias), `evaluate._question` (a private helper the validator also needs), `zip` without `strict` in one metric, stale comments in `models.py`, a lambda in a test (the only lint finding). | `ruff`, `grep` ✔ | Removed or fixed. `question_of` now sits in `datasets.py`. |
| 8 | `docs/method.md` still used the Version 1 dataset names (`text`, `cases`, `case_photos`). | read ✔ | Names updated; nothing else in the method changed. |
| 9 | README held structure, usage and results sections. | read | Trimmed to Overview and Research Update as requested; structure and usage moved to `docs/code_guide.md`; the tests that read them now read that file. |
| 10 | The metrics had hand-worked tests only. | `tests/test_metrics.py` ✔ | Added a test that recomputes every score the long way, question by question, on 60 random runs; all agree. |
| 11 | The 108 passages about how lesions look (`data/knowledge/appearance.csv`) are **not in the retrieval index** (`index.csv` holds the 429 training cases and the 5,553 library passages). | `index.py` indexes cases and library only; none of the 108 ids is in `index.csv` ✔ | Decided to leave them out (D6). |

State after the audit: 468 tests pass (8 skipped: they need the laptop's files), lint clean, 34 deliberate breaks of the new code all caught.

### Second pass (same day, after the first corrections)

A second read of the whole repository, with a wider lint (bugs, simplifications, unused names) and a search for names left over from earlier versions.

| Found | Action |
|---|---|
| 14 warnings "loop variable not bound" in the builders | each checked by hand: every function is defined and called inside the same loop turn, so none is a fault; left as it is |
| Two unused loop variables, one nested `if`, one `map` with a lambda, four redundant `encode` arguments, one `startswith` twice | simplified; behaviour unchanged, all tests pass |
| `src/__init__.py` did not name `validate.py` | named |
| The builders and data sheets call the published-case datasets "the text dataset" and "the case dataset", the code `published_text` and so on | a short list of the words and what they mean added to `docs/code_guide.md`; no mass rename, since the data sheets are part of the dataset and are not to be changed now |
| `docs/v2_proposal.md` still read as a plan to approve | a status note at its top says what was approved since and what is not built |
| `holm` (src) and `COMMANDS` (enrich) are used only by tests | `holm` is part of the approved method and stays (problem table above); `COMMANDS` is harmless documentation |
| Nothing found: unused imports, dead functions, duplicate files, leftover Version 1 code, inconsistent counts between the README, the progress page, this page and the tables | – |

## 2. Does the data support the research objective?

**Objective:** evidence-based diagnostic question answering on skin diseases, with a multimodal, rationale-guided retrieval that reduces
confident wrong diagnoses and unsupported claims; closed-book against open-book; image, text and multimodal settings.

### The three settings, in both datasets

| Setting | Clinic patients (PAD-UFES-20) | Published cases |
|---|---|---|
| Image only | `clinic_photo`: 2,269 questions (test 678) | `published_photos`: 324 (test 100) |
| Text only | `clinic_record`: the same 2,269, as a six-line record | `published_text`: 716 (test 209) |
| Multimodal | `clinic_photo_record`: photo with the same record | `published_text_photos`: 324, text with the photos |

✔ Verified by `python -m src.validate`: in each dataset the views ask the same questions with the same group (patient or article), the same
split and class, and the same six or eight answer options; the photo of `clinic_photo` and `clinic_photo_record` is the same file, the record
text in the combined view equals the record file of the record view to the byte; the two published views with photos send the same
pictures in the same order, and each names the same case file as the text view. No patient or article is in two splits; no validation or
test case is in the knowledge index. A run compares views question by question (on the shared 324 cases for the published views).

### What the data supports

- **Image-only, text-only and multimodal diagnosis** on 6 classes (clinic) and 8 classes (published cases), with paired comparisons across the
  settings and with patient- or article-level intervals ✔.
- **Closed-book against open-book on the published cases' text**: the knowledge index holds 5,982 items (429 training cases, 5,553 passages
  from PubMed reviews, PMC full texts, guidelines and StatPearls) with licences, and no validation or test item ✔.
- **Confident wrong diagnosis** as the one measurable form of hallucination (section 4).

### What it does not support yet, or only weakly

1. **A "text-only" setting for the clinic patients is a six-line structured record, not text.** It is real clinical information (a nearest-neighbour
   vote on those six fields reaches 38.2% balanced accuracy, chance 16.7%, in `dataset_report.md`), so the record view measures how much a
   model gets from age, site and four symptoms, not reading skill. The narrative text-only setting is the published cases. ◻ Whether the
   supervisor accepts the record as the clinic's text setting is a question for the meeting (D8).
2. **Retrieval cannot yet be multimodal.** The index links an item to a question by disease only (`link = disease`). There is no image
   encoder, no index of precedent photos and no query written from the photo, so the research gap named in the README (retrieval that
   ignores the image) cannot be tested until they are built. The 108 appearance passages, the library's most image-relevant text, are not in
   the index (problem 11).
3. **No gold evidence.** Nothing says which passage supports which diagnosis. Retrieval relevance can only be silver: a passage is relevant
   when its topics include the true disease (topical, not proof of support).
4. **Small test sets for the photo settings.** `published_photos` and `published_text_photos` have 100 test questions over 8 classes; actinic
   keratosis has 1 photo case overall in test or validation and 1 in training. Per-class results will have wide intervals.
5. **Class sets differ** (6 against 8) and the sources differ (clinic photos against article figures), so the two datasets are analysed
   separately, never pooled.

### Threats to validity, by dataset

| Threat | Clinic patients | Published cases | Handling today |
|---|---|---|---|
| Label noise | authors' labels; all cancers biopsy-proven; benign ones by consensus ✔ (paper) | labels from title, text and a test result; **no expert review**: 2 of 74 read questions wrong (2.7%), wrong or mixed 6.8% (2.2–15.1%) ✔ (data sheet) | stated as a limitation; the closed against open comparison is not biased when both runs share the labels ◻ |
| Hints that are not the lesion | gaps in the record and file properties: 21.4% and 19.7% balanced accuracy, slightly above chance 16.7% ✔ | text may name the answer; dermoscopy words; one question reports a biopsy | measured and reported; photo standardised; the "complete record" score for the combined view |
| Model memory of the source | PAD-UFES-20 is public ◻ | many articles old; 186 cases from 2024 on (52 in test) ✔ | an analysis by year is possible, not yet written (D9) |
| Selection of photos | clinic photos, one protocol ✔ | article figures chosen to illustrate the diagnosis, mixed clinical and dermoscopic ◻ | picture checks drop charts, histology and scans |
| Overlap between datasets | n/a | 468 published pictures compared with 2,298 clinic photos: 0 are the same ✔ | recorded in the data sheet |
| Test set reused | one test run per model, dataset and approach | same | enforced by `run.py` |

## 3. The four approaches

| Approach | Status | What is missing |
|---|---|---|
| LLM-only (closed-book) | ✔ implemented and tested; run on `published_text` with four models | nothing |
| Standard RAG | ✗ no code | the retriever, the prompt with passages, evidence ids checked against what was given (the answer reader already checks ids) |
| RAG² | ✗ no code | balanced retrieval over four corpora, the reranker, the filter (a Flan-T5 trained with a language model's perplexities, which needs a GPU) |
| Proposed | ✗ no code | findings-first querying, a query per diagnosis, precedent cases (needs an image encoder and an index of precedent photos), typed citations, the check of citations |

The prompts hold only the closed-book form. `src/run.py` takes `--approach` and refuses anything but `closed_book`. All approaches must use the
same question, the same answer reader and the same scores (the method says so); the code already scores any run the same way.

## 4. Metric by metric

"Implemented" means present in `src/metrics.py` and `src/evaluate.py`, tested against hand-worked values, and agreeing with the naive
recomputation of section 1 problem 10.

| Metric | Definition | Implemented | Correct? | Relevance and remarks |
|---|---|---|---|---|
| Balanced accuracy (main) | mean recall over the classes present: the macro-averaged recall | ✔ | ✔ | Brodersen et al. 2010 define it for two classes; with more it is the macro-averaged recall (Sokolova & Lapalme 2009). Chance 1/6 or 1/8 |
| Macro-F1 | mean of 2TP/(2TP+FP+FN) over the classes present | ✔ | ✔ | a class never answered counts 0; a class with no true question is left out even if answered (stated) |
| Per-class recall, precision, F1; confusion matrix | as usual, plus an invalid column | ✔ | ✔ | |
| Macro-averaged top-3 accuracy | the true class is in the differential, averaged over the classes | ✔ | ✔ | Liu et al. 2020 report plain top-3 accuracy; the averaging is ours and is named so. Chance 3/6 = 50% (clinic) and 3/8 = 37.5% (published) |
| Invalid-answer rate | share not of the required form | ✔ | ✔ | an invalid answer counts as wrong everywhere |
| ~~Misdiagnosis detection: sensitivity, specificity~~ | **removed 2026-11-19** | – | – | the cited source (Hendrycks & Gimpel 2017) uses AUROC and AUPR on a continuous score, not a binary flag, and the two numbers overlap with coverage and selective risk |
| Coverage, selective risk | selective classification (Geifman & El-Yaniv 2017): the flag "sufficient" answers, "not sufficient" abstains | ✔ | ✔ | one operating point; a risk-coverage curve needs a confidence number, which the approved format does not ask for. The flag is the model's own word: closed-book it is self-confidence, and it will mean something else once evidence is given (D4) |
| Confident misdiagnosis | = selective risk (the standard term is selective risk) | ✔ | ✔ as defined | not hallucination in the literature's sense: Huang et al. (preprint) separate factuality hallucination (a wrong diagnosis: the balanced accuracy) from faithfulness hallucination (a claim its cited passage does not support: not measured yet) (D3) |
| **Evidence grounding** (citation precision and recall, support of claims, patient-fact check, faithfulness) | proposed only | ✗ | – | not approved, needs a format that makes claims checkable (D2) |
| **Retrieval** (Recall@k, nDCG, MRR) | proposed only | ✗ | – | needs a retriever and silver relevance (disease topics) |
| Evidence-induced change | not a standard name; would be a paired right/wrong transition analysis with McNemar | ✗ | – | `compare` already pairs closed-book and open-book runs |
| 95% intervals | stratified cluster bootstrap (Field & Welsh 2007; the stratification by class is a design choice), 2,000 resamples, seed 0; Wilson when every question is its own group | ✔ | ✔ | a patient with lesions of two classes is split into two units, one per class (small effect, not stated before) |
| Paired difference and McNemar | difference in balanced accuracy and macro-F1 with a paired interval; exact McNemar as support | ✔ | ✔ | refuses different splits, answer options, answer formats and unequal question sets |
| Holm correction | smallest p times the number of tests, and so on | function and tests only | ✔ | not used by any report: it is for the pre-registered main comparisons once there are several (method page) |
| Scores without actinic keratosis | beside the main score | ✔ | ✔ | published views only |
| Complete-record score | questions with no `unknown` in the record | ✔ for `clinic_photo_record` only | ✔ | **not for `clinic_record`**, whose predictions do not carry the record; the same hint (gaps point to the class) applies there (D5) |
| Accuracy, ROC, AUC | – | not computed | – | rightly absent (the answers give a choice, not scores), as the method says |

**Metric review (2026-11-19, against publisher and arXiv pages).** Kept because they are established: balanced accuracy and macro-F1 (macro averaging), top-3 accuracy, coverage and selective risk, Wilson, McNemar (Dietterich 1998 recommends it for a model run once), Holm, cluster bootstrap. Removed: the flag's sensitivity and specificity. Renamed: macro-averaged top-3 accuracy; "selective risk" is the term in tables, "confident misdiagnosis" only a gloss. Not adopted, as invented or non-standard: citation coverage, fabricated-citation rate (a validity rule, already enforced by the answer reader), claims contradicted by the record, class precision@k. Still proposals, none built: retrieval metrics (Recall@k, MRR, nDCG@k, Precision@k; BEIR uses human relevance, ours would be topical), citation recall and precision (ALCE), faithfulness (RAGAS; AIS is a human protocol), a paired transition analysis of the retrieval effect. Sources not re-checked in this session: the RAGAS venue (EACL 2024, from a listing), ALCE's exact precision definition, the journal version of Huang et al.

**Diagnostic error, unsupported claim and hallucination** are three different things. Today only the first (balanced accuracy and its
complement) and an overconfident form of it (selective risk) are measured. An unsupported claim (a statement the evidence given does not support) is
not measured at all, so a "hallucination" result in this thesis would, as defined, speak of confident misdiagnosis only.

**Fairness across approaches.** Every approach would be read by the same reader, scored by the same code, compared on the same questions and
splits with the same intervals; the run card stores the prompt, settings, code commit, package versions and dataset hash. The one
asymmetry in the design is that open-book runs give the model more to read; the reader then allows evidence ids, but only those the run gave.

## 5. Validation checks, and how to verify the pipeline without spending anything

Run these in order. None sends anything to a model.

| Command | What it proves |
|---|---|
| `python -m src.validate` | tables consistent; no patient or article in two splits; every view asks the same questions; same photo, same record, same pictures across views; the prompts end with one identical answer block whose example the reader accepts; every model entry complete on every dataset and approach; no validation or test case in the knowledge index |
| `python -m src.validate --files` (laptop) | the same, and every photo and text on the computer is byte for byte the one its table lists |
| `python -m unittest discover tests` | 468 tests: the builders, the reader, the metrics (also against a naive computation), run, resume, refusal and report logic, repository consistency |
| `ruff check --select F,E4,E7,E9 src tests` | no unused import or name, no syntax problem |
| `python -m src.run check --model <name> --dataset <dataset>` | one tiny request (a fraction of a cent) that shows a wrong key, an unavailable model or a cut-off answer before a run |

Checkpoints now built into a run: dataset files hashed against the table before anything is sent; the prompt's placeholders checked; the
model pinned and the Git state clean for a test run; the first 10 answers all invalid stop the run; each answer saved as it arrives; the answer
file verified before the run is called finished; the report refuses an inconsistent file; a comparison refuses unequal splits, answer
options, formats and questions.

## 6. Decisions and issues for the discussion

| # | Question | Why it matters | My recommendation |
|---|---|---|---|
| D1 | Which retrieval approaches to build, and on which models? Open-book runs use local open-weight models only (your decision), and I do not know the hardware. | three of the four approaches are missing; RAG² needs a GPU for its filter | answer: GPU or Colab? which local models? Then build standard RAG first (the baseline), with one script and the same reader |
| D2 | Does the thesis measure evidence grounding, and with which metrics? **Update 2026-11-19: retrieval Precision@k and nDCG@k are defined and tested; citation recall and precision wait for a judge and the ALCE definitions; faithfulness is the same measure and is not reported twice; AUROC and AUPR do not apply to the approved format (method page, section 6).** **The answer format is now decided (json-v2, 2026-11-19: the current format plus 1 to 3 `claims`, each with the passages it relies on); the metrics are not.** | "evidence-based" and "reduces unsupported claims" need it; the approved method has none | decide the answer format first (`json-v2` in the proposal asks for findings and claims); then the metrics follow. No metric should be added before that |
| D3 | The word "hallucination" for a confident wrong diagnosis. | an examiner may object that it is overconfidence, not hallucination | **Done (2026-11-19, your decision):** the score is called "confident misdiagnosis" in the method and the reports; "unsupported claim" is kept for the grounding metric |
| D4 | What "evidence_sufficient" means closed-book against open-book. | the same field is self-confidence in one and a judgement of the retrieved evidence in the other | state both in the method; compare detection scores only inside one approach, and use selective risk across approaches |
| D5 | The "complete record" score for the record view. | the combined view had it, the record view did not | **Done (2026-11-19):** the record view saves the record text with each answer and gets the same score |
| D6 | Put the 108 appearance passages into the index? | they describe how lesions look, which is what an image question retrieves | **Decided 2026-11-19: no.** They were not verified, and a manual reading would break the requirement that the work is automated. They stay in `data/knowledge/appearance.csv` (nothing deleted) and out of the index. A later, automatic check could revisit this |
| D7 | Should the split be redrawn? | validation holds 98% single-photo patients | no: class, age and site are balanced and a redraw would break the rule against reshuffling (section 2) |
| D8 | Is the clinic record an acceptable "text only" setting? | it is six structured fields, not narrative text | keep both text settings (record and published cases) and say plainly what each tests |
| D9 | The analysis of cases from before and after a model's training date. | memory of the source can raise closed-book scores and hide the gain from retrieval | write it when results exist ("if necessary" in your words) |
| D10 | `src/build/published_cases/text.py` has 1,766 lines. | a long file is hard to explain | leave it: it builds a dataset that must not change now; split it after the dataset is final, if at all |
| D11 | When to freeze Version 2. | the tables, reports and data sheets are consistent and checked, but you said the freezing is with your supervisor | after the meeting; name each frozen version by its commit, no tags until the final version |
