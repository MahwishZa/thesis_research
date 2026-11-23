# Code guide: how the pipeline is built and run

| | |
|---|---|
| Document status | Current as of 2026-11-24 (the LLM-only baseline is built; the RAG systems are not). |
| Audience | The researcher, the supervisor, and anyone who wants to reproduce the experiments. |
| Related documents | `data/datasheet.md` (the data), `results/evaluation_guide.md` (the metrics), `results/README.md` (the results folder), `results/STATUS.md` (where the work stands). |

Run every command from the repository folder (PowerShell on Windows; the commands are the same on Linux). Each script explains itself at the top and with `--help`. Only the Python standard library is needed unless the table says otherwise (`requirements.txt` pins the packages).

## 1. Design in one page

The research compares four systems on the same questions: **LLM-only**, **standard RAG**, the **RAG² paper's method** (Sohn et al., NAACL 2025) and the **proposed RAG**. Each is run in three **views** (image only, text only, image + text) with four pretrained LLMs used as they are (claude-sonnet-5-5, gpt-4o, gemini-3.8-flash, llama-4-maverick). Only LLM-only is built so far.

Rules the code follows (so that runs are comparable and repeatable):

| Rule | Where it is enforced |
|---|---|
| One seed (42) for every random choice | `common.SEED` |
| The same questions, options and option order for every system and view | `build.py views` writes the question files once; `run.py` only reads them |
| Questions are asked in a fixed shuffled order (sha256 of `42:<case id>`), so a partial run covers all classes | `run.py` `asking_order` |
| Settings of each model are fixed in one table and saved in every run | `run.py` `MODELS`, `run.json` |
| A run can stop and continue without repeating answered questions | `run.py` (checkpoint after every answer, folder `*.partial` until complete) |
| Files are replaced in one step, so an interrupted run never leaves a half-written file | `common.atomic_path` |
| No secret is ever written to a file or a log | `common.load_env` reports only line numbers; keys are read from `.env` (never committed) |
| Validation and test are separate: trial and validation runs never become results | `.gitignore` (`*_val/`, `*.partial/`, `trial_*.json`); final runs use the test split once |

## 2. Data flow

```
data/raw/ + data/official_labels/
   │ preprocess.py            clean, correct labels, split by lesion, balance val/test
   ▼
data/processed_cases/, data/processed_images/
   │ build.py views           three views, one 8-option question per case
   ▼
data/views/<view>/{train,val,test}.jsonl ──┐
data/knowledge_corpus/ (build.py corpus)   │
data/knowledge_corpus/reference_texts.json │
   │                                        ▼
   │                         run.py llm-only  ──►  results/llm_only/<model>/<view>_<split>/
   ▼                                                    predictions.jsonl, run.json, run.log
evaluate.py collect  ──►  report.json next to each finished run  +  results/summary.csv
evaluate.py compare  ──►  McNemar + Holm for every pair of runs
evaluate.py similarity-check  ──►  is the semantic similarity informative?
check.py             ──►  where each step stands (reads only)
```

## 3. The files and their commands

| File | Command | Job | Reads | Writes | Needs |
|---|---|---|---|---|---|
| `preprocess.py` | – | **Step 1.** Cleans the Kaggle download: removes duplicates, corrects labels, splits by lesion, balances the test (30 per class) and validation (20 per class) splits | `data/raw/`, `data/official_labels/` | `data/processed_images/`, `data/processed_cases/` | pandas, numpy, pillow, opencv, scikit-image, scikit-learn; the images (laptop) |
| `build.py` | `views` | **Step 2.** The three views and their question files | `data/processed_cases/all_cases.csv` | `data/views/<view>/cases.csv`, `{train,val,test}.jsonl` | Python only |
| `build.py` | `corpus` | **Step 3.** The knowledge corpus for RAG (four libraries) | PubMed, PMC, guidelines, Dermoscopedia (internet) | `data/knowledge_corpus/` | internet; `datasets` for the Meditron guidelines |
| `build.py` | `references` | **Step 3b.** Quoted reference texts per disease, for the semantic-similarity metric | Dermoscopedia (internet) | `data/knowledge_corpus/reference_texts.json` | internet; `--token-check` needs sentence-transformers |
| `run.py` | `llm-only` | **Step 4.** The LLM-only baseline: one model, one view, one split (the other systems are added here) | `data/views/<view>/<split>.jsonl` (+ images) | `results/llm_only/<model>/<view>_<split>/` | keys in `.env`; **costs credit** |
| `evaluate.py` | – | **Step 5.** One report for one run: all metrics | a question file and a prediction file | one `report.json` | Python only (the optional metrics need models) |
| `evaluate.py` | `collect` | **Step 6.** A `report.json` next to every finished run and one table of all runs | `results/<system>/<model>/<view>_<split>/` | `results/summary.csv` | Python (sentence-transformers for the semantic similarity; `--skip-text` leaves it out) |
| `evaluate.py` | `compare` | **Step 7.** Several runs on the same cases: McNemar for every pair, Holm-corrected | several runs | one `compare.json` | Python only |
| `evaluate.py` | `similarity-check` | Does the semantic similarity carry information? (pass mark set in advance) | finished runs, reference texts | `results/similarity_check.json` | sentence-transformers |
| `check.py` | – | Where the pipeline stands: each step done, not run, or a problem, and why (reads only) | everything above | a table on the screen | Python only |
| `check.py` | `features` | The feature and shortcut analysis of the cleaned data (an analysis, not an experiment) | `data/processed_cases/all_cases.csv`, `data/processed_images/` | `results/feature_check/` | the image packages |

`common.py` holds what every script shares: the 8 classes and their names, the option texts, the seed, the layout of `data/`, the balanced-split rule (`balanced_splits`), safe writing, reading and logging. It needs only Python.

## 4. Commands to run

```powershell
python src/preprocess.py                                   # step 1 (minutes; needs the images)
python src/build.py views                                  # step 2 (seconds)
python src/build.py corpus                                 # step 3 (internet; downloads are cached)
python src/build.py references --token-check               # step 3b (internet)
python src/check.py                                        # where everything stands

# LLM-only (paid). Models: claude-sonnet-5-5, gpt-4o, gemini-3.8-flash, llama-4-maverick
python src/run.py llm-only --model gpt-4o --view text_only --split val --dry-run --limit 2   # see the prompt (free)
python src/run.py llm-only --model gpt-4o --view text_only --split val --limit 10            # a small paid trial
python src/run.py llm-only --model gpt-4o --view text_only --split test                      # the final run (240 questions; resumable)

python src/evaluate.py collect                             # report.json for each finished run and results/summary.csv
python src/evaluate.py compare --out results/llm_only/compare_text_only_test.json `
    --run claude=data/views/text_only/test.jsonl,results/llm_only/claude-sonnet-5-5/text_only_test/predictions.jsonl `
    --run gpt-4o=data/views/text_only/test.jsonl,results/llm_only/gpt-4o/text_only_test/predictions.jsonl
python src/evaluate.py similarity-check                    # free; needs finished runs
python src/evaluate.py --questions data/views/text_only/val.jsonl --predictions <predictions.jsonl> --answered-only --out <report.json>   # a trial

python -m unittest discover -s tests                       # the checks
ruff check --select F,E9 src tests                         # lint
```

**Validation before test.** A trial (`--limit 10`) checks the format and the cost; its accuracy is never reported. Trial folders end in `.partial` and are ignored by Git. A test run (`--split test`) is made once per model, view and system, after the settings are fixed.

## 5. The run folder

`results/llm_only/<model>/<view>_<split>/` (a trial or unfinished run ends in `.partial`):

| File | What it holds |
|---|---|
| `predictions.jsonl` | one line per question (below) |
| `run.json` | the model's settings (id, host, temperature, seed, `max_tokens`), the exact prompt, the SHA-256 of the question file, counts and the status (`complete`, `stopped`, `unfinished`) |
| `run.log` | progress, retries and errors (not committed) |
| `report.json` | written by `evaluate.py collect`: every metric of this run |

A prediction line (`run.py` writes it; `evaluate.py` reads it): `{"id": "ISIC_0024948", "pred": "C", "valid": true, "rationale": "...", "raw_output": "...", "finish_reason": "stop", "served_by": "openai/gpt-4o-2024-08-06@OpenAI", "view": "text_only", "split": "test", "prompt": "llm_only", "tokens_in": 215, "tokens_out": 80, "seconds": 1.87, "time": "..."}`. Fields used by the metrics: `pred` (the letter), `rationale` (the reasoning), `view` (the text-only image-claim check), `finish_reason` and `raw_output` (refusals), tokens and seconds (cost). For the RAG systems a line also carries `query`, `contexts` (the retrieved passages, best first, with their tags) and `evidence` (`{"chunk_id", "quote"}` or null for "Evidence: none").

**The prompt** (`llm_only`): the system message says the model is an expert dermatologist answering a research benchmark. The user message holds the question, the 8 lettered options and the instruction: *"Reply in exactly this form. First line: "Reasoning and evidence: <at most three sentences that name the findings of this case, {where}, that support your choice>". Second line: "Answer: <letter> (<name of that option>)"."* `{where}` is "in the patient facts" (text only), "in the image" (image only) or "in the image and/or in the patient facts" (both): a model is never told to cite an image it did not get. The letter decides; the first letter on the last "Answer:" line is read.

## 6. The model settings

| Model (as used) | Host | Settings |
|---|---|---|
| claude-sonnet-5-5 | Anthropic API | no temperature (the API takes none: default sampling), `max_tokens` 1024, effort low |
| gpt-4o (snapshot 2024-08-06) | OpenRouter, host OpenAI only | temperature 0, seed 0, `max_tokens` 1024, image detail high |
| gemini-3.8-flash | OpenRouter, host Google AI Studio only | temperature 0, seed 0, minimal reasoning, `max_tokens` 8192 (2048 cut off an answer in a trial) |
| llama-4-maverick | OpenRouter, host Parasail (8-bit) only | temperature 0, seed 0, `max_tokens` 1024 |

Each model is pinned to one host (`allow_fallbacks` off), so the answers come from the same service every time. Claude's sampling is not fixed, so a rerun may differ.

## 7. Tests (`tests/`)

Run with `python -m unittest discover -s tests`: 222 tests, 7 skipped where pandas or scikit-learn are missing (last run 2026-11-24).

| File | Level | What it checks | Needs |
|---|---|---|---|
| `test_unit.py` | unit | one function or rule at a time, on made-up input: common, the balanced split, cleaning, options, the metrics (accuracy, F1, McNemar, Holm, retrieval, evidence, image claims, similarity), the prompt, the answer rule, the model requests | Python (pandas for the cleaning tests) |
| `test_integration.py` | integration | parts working together on fake services: the corpus builder, the reference-text builder, a whole run with a fake model read by `evaluate.py`, `collect`, `check.py` on small folders | Python only |
| `test_system.py` | system | the committed folder as it is: tables and views, the views rebuilt byte for byte, the balanced split, every command answering `--help`, the archives, the fixed top-level structure, the README rule | Python only |
| `test_acceptance.py` | acceptance | the researcher's own requirements, one test each: Kaggle only, 8 classes, balanced test and validation, three views, four models, approved metrics only, sources named, no secret, no tag | Python only |

**New code is tested by breaking it on purpose:** a deliberate fault is put in, and a test must fail. A fault that no test catches means a missing test.

## 8. Known limits of the code

- The RAG systems (standard RAG, RAG², proposed) are not implemented: `run.py` has one command.
- `evaluate.py` scores trial runs only with `--answered-only` (otherwise an unasked question counts as wrong), and marks such a report as a trial.
- The semantic similarity and the MedCPT relevance need models that are downloaded on first use (`sentence-transformers`, `transformers`, `torch`).
- Image claims in the text-only view are found by a word list: an approximate lower bound.

## 9. Old names (renamed on 2026-11-22)

| Old | New |
|---|---|
| `build_views.py` | `build.py views` |
| `build_knowledge_corpus.py` | `build.py corpus` |
| `build_reference_texts.py` | `build.py references` |
| `check_pipeline.py` | `check.py` |
| `feature_check.py` | `check.py features` |
| `run_llm_only.py` | `run.py llm-only` |
| `compare.py` | `evaluate.py compare` |
| `balance_test.py` | `common.py` (`balanced_splits`) |
