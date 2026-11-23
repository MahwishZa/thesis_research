# LLM-only baseline, text-only view, test split: results and how to read them

**Scores are shown as percentages** (25.4% = 61 of 240 questions right). Exceptions: p-values are probabilities and the semantic similarity is a cosine score; both stay as decimals.

Written 2026-11-24 from the committed run files (commit `c125030`). Every number below is read from `results/llm_only/<model>/text_only_test/report.json`,
`results/llm_only/compare_text_only_test.json` and `results/summary.csv`; nothing is typed by hand. The metric definitions and sources are in
`results/evaluation_guide.md`. Marks: ✔ = read from those files or recomputed from them; ◻ = my reading, not tested.

## 1. What was run

| Item | Value |
|---|---|
| System | LLM-only: no retrieval, one question, one answer (prompt `llm_only`) |
| View | Text only: age, sex and body site of the patient (no image) |
| Split | Test: 240 questions, 30 per class, 8 classes; chance accuracy is 1 in 8 = **12.5%** |
| Question file | `data/views/text_only/test.jsonl`, SHA-256 `ed08a96b...` (the same file for all four models) |
| Answer format | "Reasoning and evidence: ..." then "Answer: C (Melanoma)"; the letter decides |
| Options | 8, in a fixed shuffled order per case (the same order in every view) |
| Models (as used, no training) | Claude Sonnet 5.5, GPT-4o (snapshot 2024-08-06), Llama 4 Maverick (host Parasail, 8-bit), Gemini 3.8 Flash (Google AI Studio) |
| Settings | temperature 0 and seed 0 for GPT-4o, Llama and Gemini; Claude takes no temperature (its default sampling is used, so a rerun may differ); Gemini `max_tokens` 8192 |
| Status | All four runs complete: 240 of 240 answered; no missing, invalid or refused answer |

## 2. The headline table

| Model (host that answered) | Right answers | Accuracy | 95% interval | Macro-F1 | 95% interval | Invalid | Refusals |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 (`claude-sonnet-5-5`) | 61 of 240 | **25.4%** | 19.3% to 31.9% | 19.4% | 14.8% to 24.2% | 0 | 0 |
| GPT-4o (`openai/gpt-4o-2024-08-06@OpenAI`) | 48 of 240 | **20.0%** | 14.5% to 25.7% | 19.3% | 13.9% to 24.0% | 0 | 0 |
| Llama 4 Maverick (`meta-llama/llama-4-maverick@Parasail`) | 52 of 240 | **21.7%** | 15.5% to 27.8% | 17.2% | 13.1% to 21.0% | 0 | 0 |
| Gemini 3.8 Flash (`google/gemini-3.8-flash@Google AI Studio`) | 50 of 240 | **20.8%** | 15.1% to 26.5% | 15.1% | 11.0% to 19.3% | 0 | 0 |

Chance is 12.5%. **All four models are above chance**: the lower end of every accuracy interval is above 12.5% ✔.
The four intervals overlap a lot, so the table alone cannot rank the models; section 4 does that with a test.

## 3. The metrics, one by one

### 3.1 Accuracy with a 95% interval

- **What it measures:** the share of the 240 questions answered with the right class (25.4% = 61 of 240). A missing or invalid answer would count as wrong (there were none).
- **How the interval is made:** 1,000 resamples of whole lesions (photos of one lesion stay together); the 2.5th and 97.5th percentiles.
- **Why it is used:** it is the one metric every medical multiple-choice RAG paper reports (MIRAGE, RAG²), so results can be compared with the literature.
- **What it shows here:** between 20.0% (GPT-4o) and 25.4% (Claude), about twice chance but low in absolute terms. In this view the model sees only age, sex and body site, which cannot identify a skin disease. Low scores are expected, not a fault.

### 3.2 Macro-F1

- **What it measures:** the average of the eight per-class F1 scores, so every class counts the same. F1 of a class = 2·TP / (2·TP + FP + FN).
- **Why it is used:** accuracy can look fair while a model ignores some classes; macro-F1 exposes that.
- **What it shows here:** Claude 19.4% and GPT-4o 19.3% are level although GPT-4o's accuracy is lower (20.0% against 25.4%): GPT-4o spreads its answers over more classes, Claude puts many on nevus and gets that class right. Llama 17.2% and Gemini 15.1% are lower than their accuracy suggests because each ignores several classes (below).

### 3.3 Per-class precision, recall, specificity and F1

- **Recall of a class** = of the 30 cases of that disease, how many were found. **Precision** = of the answers naming that disease, how many were right. **Specificity** = of the cases that are not that disease, how many were correctly not called it.
- **Why they are used:** they show which diseases are found or missed, which accuracy hides. Melanoma is the most important to miss.

Recall:

| Class | Claude Sonnet 5.5 | GPT-4o | Llama 4 Maverick | Gemini 3.8 Flash |
|---|---|---|---|---|
| MEL | 0.0% | 0.0% | 0.0% | 0.0% |
| NV | 90.0% | 23.3% | 60.0% | 90.0% |
| BCC | 16.7% | 16.7% | 3.3% | 16.7% |
| AK | 6.7% | 33.3% | 33.3% | 6.7% |
| BKL | 53.3% | 36.7% | 70.0% | 26.7% |
| DF | 33.3% | 36.7% | 6.7% | 26.7% |
| VASC | 0.0% | 6.7% | 0.0% | 0.0% |
| SCC | 3.3% | 6.7% | 0.0% | 0.0% |

Precision:

| Class | Claude Sonnet 5.5 | GPT-4o | Llama 4 Maverick | Gemini 3.8 Flash |
|---|---|---|---|---|
| MEL | 0.0% | 0.0% | 0.0% | 0.0% |
| NV | 42.2% | 63.6% | 66.7% | 30.0% |
| BCC | 12.5% | 10.6% | 5.9% | 9.8% |
| AK | 25.0% | 20.0% | 38.5% | 28.6% |
| BKL | 15.7% | 12.2% | 13.2% | 11.8% |
| DF | 58.8% | 47.8% | 100.0% | 57.1% |
| VASC | 0.0% | 100.0% | 0.0% | 0.0% |
| SCC | 33.3% | 20.0% | 0.0% | 0.0% |

F1:

| Class | Claude Sonnet 5.5 | GPT-4o | Llama 4 Maverick | Gemini 3.8 Flash |
|---|---|---|---|---|
| MEL | 0.0% | 0.0% | 0.0% | 0.0% |
| NV | 57.4% | 34.1% | 63.2% | 45.0% |
| BCC | 14.3% | 13.0% | 4.3% | 12.3% |
| AK | 10.5% | 25.0% | 35.7% | 10.8% |
| BKL | 24.2% | 18.3% | 22.2% | 16.3% |
| DF | 42.6% | 41.5% | 12.5% | 36.4% |
| VASC | 0.0% | 12.5% | 0.0% | 0.0% |
| SCC | 6.1% | 10.0% | 0.0% | 0.0% |

Specificity:

| Class | Claude Sonnet 5.5 | GPT-4o | Llama 4 Maverick | Gemini 3.8 Flash |
|---|---|---|---|---|
| MEL | 97.1% | 96.7% | 99.0% | 95.2% |
| NV | 82.4% | 98.1% | 95.7% | 70.0% |
| BCC | 83.3% | 80.0% | 92.4% | 78.1% |
| AK | 97.1% | 81.0% | 92.4% | 97.6% |
| BKL | 59.0% | 62.4% | 34.3% | 71.4% |
| DF | 96.7% | 94.3% | 100.0% | 97.1% |
| VASC | 100.0% | 100.0% | 97.6% | 100.0% |
| SCC | 99.0% | 96.2% | 99.0% | 100.0% |

**What they show here (✔):**
- **Melanoma was never identified: recall 0% for all four models (0 of 30 each).** The models named melanoma only 6 (Claude), 7 (GPT-4o), 2 (Llama) and 10 (Gemini) times in 240 answers, and never correctly. Age, sex and site do not point to melanoma for these models.
- **Claude and Gemini lean on nevus:** recall 90% for NV, but they name NV 64 and 90 times (precision 42.2% and 30.0%). A common-diagnosis guess, not discrimination.
- **Llama leans on benign keratosis-like (BKL):** 159 of its 240 answers (66%) are BKL (recall 70.0%, precision 13.2%, specificity 34.3%).
- **Vascular lesion and squamous cell carcinoma are almost never found** (recall 0% to 6.7%).
- **Dermatofibroma is found better by Claude, GPT-4o and Gemini** (recall 33.3%, 36.7% and 26.7%) **than by Llama** (6.7%; it named DF only twice, both right, so precision 100%).

### 3.4 Confusion matrices

- **What they show:** rows are the true class, columns the class answered; the diagonal is correct. They tell which diseases are mistaken for which.
- **Why they are used:** they explain a low recall (where did the cases go?) and expose a model's habit (a heavy column).

**Claude Sonnet 5.5**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 0 | 8 | 7 | 2 | 12 | 1 | 0 | 0 | 0 |
| **NV** | 0 | 27 | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| **BCC** | 1 | 7 | 5 | 0 | 16 | 1 | 0 | 0 | 0 |
| **AK** | 0 | 2 | 14 | 2 | 12 | 0 | 0 | 0 | 0 |
| **BKL** | 0 | 5 | 7 | 0 | 16 | 2 | 0 | 0 | 0 |
| **DF** | 0 | 3 | 0 | 2 | 14 | 10 | 0 | 1 | 0 |
| **VASC** | 2 | 12 | 2 | 1 | 11 | 1 | 0 | 1 | 0 |
| **SCC** | 3 | 0 | 5 | 1 | 19 | 1 | 0 | 1 | 0 |

**GPT-4o**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 0 | 1 | 6 | 7 | 11 | 2 | 0 | 3 | 0 |
| **NV** | 0 | 7 | 0 | 1 | 18 | 4 | 0 | 0 | 0 |
| **BCC** | 0 | 0 | 5 | 2 | 20 | 2 | 0 | 1 | 0 |
| **AK** | 0 | 0 | 18 | 10 | 1 | 0 | 0 | 1 | 0 |
| **BKL** | 0 | 0 | 13 | 4 | 11 | 2 | 0 | 0 | 0 |
| **DF** | 2 | 0 | 0 | 11 | 5 | 11 | 0 | 1 | 0 |
| **VASC** | 2 | 3 | 1 | 5 | 14 | 1 | 2 | 2 | 0 |
| **SCC** | 3 | 0 | 4 | 10 | 10 | 1 | 0 | 2 | 0 |

**Llama 4 Maverick**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 0 | 2 | 2 | 5 | 20 | 0 | 1 | 0 | 0 |
| **NV** | 0 | 18 | 0 | 1 | 10 | 0 | 1 | 0 | 0 |
| **BCC** | 0 | 1 | 1 | 0 | 28 | 0 | 0 | 0 | 0 |
| **AK** | 0 | 0 | 5 | 10 | 14 | 0 | 0 | 1 | 0 |
| **BKL** | 0 | 0 | 6 | 3 | 21 | 0 | 0 | 0 | 0 |
| **DF** | 0 | 3 | 0 | 2 | 20 | 2 | 2 | 1 | 0 |
| **VASC** | 1 | 3 | 1 | 1 | 24 | 0 | 0 | 0 | 0 |
| **SCC** | 1 | 0 | 2 | 4 | 22 | 0 | 1 | 0 | 0 |

**Gemini 3.8 Flash**

| true \ answered | MEL | NV | BCC | AK | BKL | DF | VASC | SCC | invalid |
|---|---|---|---|---|---|---|---|---|---|
| **MEL** | 0 | 12 | 9 | 1 | 8 | 0 | 0 | 0 | 0 |
| **NV** | 0 | 27 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
| **BCC** | 2 | 9 | 5 | 0 | 14 | 0 | 0 | 0 | 0 |
| **AK** | 1 | 1 | 19 | 2 | 7 | 0 | 0 | 0 | 0 |
| **BKL** | 0 | 9 | 11 | 0 | 8 | 2 | 0 | 0 | 0 |
| **DF** | 1 | 10 | 0 | 2 | 9 | 8 | 0 | 0 | 0 |
| **VASC** | 4 | 12 | 2 | 1 | 10 | 1 | 0 | 0 | 0 |
| **SCC** | 2 | 10 | 5 | 0 | 11 | 2 | 0 | 0 | 0 |

The heavy columns are the "most common diagnosis" habit named above ✔: NV for Claude (64 answers) and Gemini (90), BKL for Llama (159), Claude (102) and GPT-4o (90). Only 11 of GPT-4o's 240 answers are NV.

### 3.5 Invalid answers and refusals

- **What it measures:** answers with no usable letter, and, apart, those where the model declined ("I'm sorry, I can't help with that").
- **Why it is used:** a refusal or a cut-off answer counts as wrong, so it must be reported next to accuracy to keep the comparison fair.
- **What it shows here:** **0 invalid and 0 refusals for all four models** ✔. In the earlier image trials GPT-4o refused 2 of 10 image questions; the text-only view does not trigger that.

### 3.6 Which letter was chosen

- **What it measures:** how often each letter A to H was picked. Options are shuffled, so a letter carries no meaning; a model that favours one letter has a position habit.
- **Why it is used:** a position habit adds noise or bias to accuracy.

| Letter | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| Right answer (the gold letter) | 29 | 36 | 31 | 26 | 34 | 25 | 31 | 28 |
| Claude Sonnet 5.5 | 38 | 54 | 28 | 22 | 23 | 28 | 26 | 21 |
| GPT-4o | 41 | 46 | 33 | 19 | 30 | 26 | 24 | 21 |
| Llama 4 Maverick | 25 | 34 | 40 | 26 | 26 | 25 | 26 | 38 |
| Gemini 3.8 Flash | 28 | 52 | 25 | 25 | 27 | 35 | 29 | 19 |

**What it shows (✔ counts; the p-values are my rough ◻ one-sided binomial tests against 1 in 8; p-values stay as decimals, they are probabilities):** Claude (54), Gemini (52) and GPT-4o (46) chose **B** more often than chance (30 of 240, 12.5%) or than B's share of right answers (36). The p-values are about 1e-5, 5e-5 and 0.002; Llama's 34 is unremarkable (p about 0.24). Llama chose C and H somewhat more (40 and 38). This is a mild position habit in three models; it does not explain the low accuracy but should be mentioned.

### 3.7 McNemar's exact test with Holm correction

- **What it measures:** whether two systems differ on the same questions. Only the questions where exactly one of the two is right matter. The exact test is a two-sided binomial test with p = 0.5; Holm's correction (6 pairs) controls false positives.
- **Why it is used:** comparing two accuracies by eye ignores that the same questions were used. It is the standard paired test for classifiers (McNemar 1947; Holm 1979).

| Pair | Only first right | Only second right | p | Holm-adjusted p |
|---|---|---|---|---|
| claude vs gpt-4o | 27 | 14 | 0.059584 | 0.29792 |
| claude vs llama | 27 | 18 | 0.232693 | 0.930772 |
| claude vs gemini | 13 | 2 | 0.007385 | 0.04431 |
| gpt-4o vs llama | 25 | 29 | 0.683489 | 1.0 |
| gpt-4o vs gemini | 20 | 22 | 0.877614 | 1.0 |
| llama vs gemini | 24 | 22 | 0.882996 | 1.0 |

**What it shows (✔):** only **Claude against Gemini** differs after correction (Holm p = 0.044; Claude alone was right on 13 questions, Gemini on 2). It is borderline (just under 0.05) and one pair of six. Claude against GPT-4o (Holm p = 0.30) and against Llama (0.93) are **not** established. GPT-4o, Llama and Gemini cannot be told apart.

### 3.8 Semantic similarity of the reasoning

- **What it measures:** the cosine similarity between a sentence embedding of the model's reasoning and the embedding of each reference text of the true disease (40 texts quoted from Dermoscopedia, CC BY-NC-SA 4.0); the best match is kept. Model: `pritamdeka/S-PubMedBert-MS-MARCO`.
- **Why it is used:** the supervisor asked for a measure of how close the reasoning is to the medical text. It checks that the reasoning talks about the right disease, not that the answer is right.

| Model | Mean | 95% interval | MEL | NV | BCC | AK | BKL | DF | VASC | SCC |
|---|---|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 0.9307 | 0.9292 to 0.9322 | 0.931 | 0.948 | 0.923 | 0.930 | 0.928 | 0.929 | 0.926 | 0.930 |
| GPT-4o | 0.9268 | 0.9254 to 0.9281 | 0.923 | 0.937 | 0.922 | 0.930 | 0.926 | 0.925 | 0.924 | 0.928 |
| Llama 4 Maverick | 0.9211 | 0.9199 to 0.9224 | 0.922 | 0.933 | 0.917 | 0.920 | 0.923 | 0.916 | 0.918 | 0.918 |
| Gemini 3.8 Flash | 0.9320 | 0.9305 to 0.9335 | 0.932 | 0.951 | 0.924 | 0.929 | 0.927 | 0.930 | 0.929 | 0.935 |

**What it shows (✔) and a warning (◻):** every score is between 0.916 and 0.951 (a cosine score from 0 to 1, **not a percentage**), and the 95% intervals are about ±0.002. The four models differ by about 0.01, and the per-class means barely move. A metric that gives almost the same number for every model, class and answer **may not discriminate**: general similarity to medical prose can be high for any reasoning. I have **not tested that yet**, so do not interpret differences between models from this table. The check is built (`python src/evaluate.py similarity-check`, approved 2026-11-24). It asks whether a reasoning resembles the reference texts of the disease **the model answered** more than those of the other seven (a reasoning argues for the answered disease, right or wrong). Chance is 12.5%. Pass mark, set before the first run: identified in at least 50% of the answers overall and in at least 25% for every answered disease with 10 or more answers (my proposal, not a published standard). It was changed from my first idea (compare with a wrong disease, 70%), which would have punished wrong answers that are argued consistently. Result: not run yet.

### 3.9 Image claims in the text-only view

- **What it measures:** the share of reasonings that describe an image or a dermoscopy although the model got no image (a word list; sentences that deny or hypothesise are skipped).
- **Why it is used:** it is an automatic, free proxy for one kind of hallucination, described findings that cannot exist.
- **What it shows:** **0 of 240 (0%) for every model** ✔ (interval 0% to 0%). With the first wording one model had invented dermoscopic findings in a trial; after the view-specific wording ("findings in the patient facts") none was flagged. It is a lower bound: a paraphrase would be missed ◻.

### 3.10 Cost and speed

| Model | Questions | Tokens in (total) | Tokens out (total) | Tokens out per question | Seconds (total) | Seconds per question |
|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 240 | 75,044 | 44,032 | 183.47 | 673.19 | 2.8 |
| GPT-4o | 240 | 50,534 | 18,355 | 76.48 | 379.13 | 1.58 |
| Llama 4 Maverick | 240 | 50,449 | 24,474 | 101.97 | 496.75 | 2.07 |
| Gemini 3.8 Flash | 240 | 48,431 | 93,578 | 389.91 | 1421.9 | 5.92 |

**What it shows (✔):** the four runs used about 224,000 tokens in and 180,000 tokens out. Gemini writes the longest answers (390 tokens per question, 5.9 s) and Claude counts the most input tokens for the same text (313 per question, against about 202 to 211 for the others; tokenizers differ). Prices are not in the table.

## 4. A check that changes how to read the headline (patient facts recorded or not)

Age, sex or site is missing for 28 of the 240 test cases (NV 15, MEL 7, DF 2, VASC 2, BCC 1, SCC 1) ✔. A missing fact hints at nevus, so a model that answers "nevus" when facts are missing gains accuracy. Accuracy by group (computed by me from the committed predictions with `evaluate.py`'s `subgroup_accuracy`; not stored in `report.json`; no intervals):

| Model | All three facts recorded (n = 212) | Some missing (n = 28) |
|---|---|---|
| Claude Sonnet 5.5 | 21.7% | 53.6% |
| GPT-4o | 22.2% | 3.6% |
| Llama 4 Maverick | 20.3% | 32.1% |
| Gemini 3.8 Flash | 16.5% | 53.6% |

With all three facts recorded (212 cases) the four models lie between 16.5% and 22.2%, closer together than the headline. The "some missing" group has only 28 cases, so its numbers move a lot (GPT-4o 3.6%, Claude and Gemini 53.6%). The thesis should report text-only results on all cases **and** on the 212 cases with all facts (known issue 1 in `results/STATUS.md`).

## 5. What the results say, and what they do not

**Say (✔):**
1. In the text-only view the four models are above chance (20.0% to 25.4% against 12.5%) but weak.
2. Only Claude against Gemini differs, borderline (Holm p = 0.044).
3. No model identifies melanoma (0 of 30 each), and VASC and SCC are almost never found.
4. The models lean on a common diagnosis (NV for Claude and Gemini, BKL for Llama, Claude and GPT-4o) and three of them favour the letter B.
5. No refusals, no invalid answers, no described image.

**Do not say:**
- That the ranking of the models is established (it is not, except one pair).
- That semantic similarity separates the models (not tested; it looks flat).
- That 0 image claims proves no hallucination (the other kinds are not measured, and the check is a word list).
- Anything about retrieval: this run has none. These numbers are the baseline for the RAG systems, nothing more.

## 6. Open points

- The semantic-similarity validity check (section 3.8): built, to run on the laptop; its result goes here.
- Image-only and multimodal test runs: not run yet; GPT-4o refused 2 of 10 image questions in the trial, so refusals will matter there.
- Standard RAG (k = 3) is being built; the proposed system and RAG² are not decided.
- Hallucination is only approximated here (image claims); the faithfulness metrics start with the RAG runs.

## 7. Files and how to reproduce

| What | Where |
|---|---|
| Predictions, one line per question | `results/llm_only/<model>/text_only_test/predictions.jsonl` |
| Settings, prompt, status, question-file hash | `results/llm_only/<model>/text_only_test/run.json` |
| Report with every metric above | `results/llm_only/<model>/text_only_test/report.json` |
| One row per run | `results/summary.csv` |
| Pairwise tests | `results/llm_only/compare_text_only_test.json` |

```powershell
python src/run.py llm-only --model gpt-4o --view text_only --split test      # the run (paid); resumable
python src/evaluate.py collect                                               # report.json for each run and results/summary.csv
python src/evaluate.py compare --out results/llm_only/compare_text_only_test.json --run claude=data/views/text_only/test.jsonl,results/llm_only/claude-sonnet-5-5/text_only_test/predictions.jsonl --run gpt-4o=...   # McNemar + Holm
```

## 8. Sources (✔ = seen in this project's literature search; ◻ = from memory, check before citing)

- McNemar (1947) Psychometrika 12:153-157 ◻; Holm (1979) Scandinavian Journal of Statistics 6:65-70 ◻; bootstrap: Efron and Tibshirani (1993) ◻.
- MIRAGE / MedRAG: Xiong et al. (ACL Findings 2024) ✔; RAG²: Sohn et al. (NAACL 2025) ✔ (abstract).
- Sentence embeddings: Reimers and Gurevych (2019) EMNLP ◻; model `pritamdeka/S-PubMedBert-MS-MARCO` (Hugging Face).
- Dermoscopedia reference texts: CC BY-NC-SA 4.0.
