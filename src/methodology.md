# Code guide

Run every command from the repository folder. Each script explains itself at the top and with `--help`.
Six files, each with one job. A script with several jobs has commands; the command is the first word after the file name.

| File | Command | Job | Reads | Writes | Needs |
|---|---|---|---|---|---|
| `preprocess.py` | – | **Step 1.** Cleans the Kaggle download: removes duplicates, corrects labels, splits (lesion-grouped, then the test split balanced to 30 per class) | `data/raw/`, `data/official_labels/` | `data/processed_images/`, `data/processed_cases/` | the Kaggle download; pandas, OpenCV, scikit-learn |
| `build.py` | `views` | **Step 2.** The three views and their question files | `data/processed_cases/all_cases.csv` | `data/views/<view>/cases.csv`, `{train,val,test}.jsonl` | Python only |
| `build.py` | `corpus` | **Step 3.** The knowledge corpus for RAG (four libraries) | PubMed, PMC, guidelines, Dermoscopedia (internet) | `data/knowledge_corpus/` | internet; `datasets` for the Meditron guidelines |
| `build.py` | `references` | **Step 3b.** Quoted reference texts per disease, for semantic similarity | Dermoscopedia (internet) | `data/knowledge_corpus/reference_texts.json` | internet; `--token-check` needs sentence-transformers |
| `run.py` | `llm-only` | **Step 4.** The LLM-only baseline: one model, one view, one split (the other baselines are added here later) | `data/views/<view>/<split>.jsonl` (+ images) | `results/llm_only/<model>/<view>_<split>/` | keys in `.env`; costs credit |
| `evaluate.py` | – | **Step 5.** One report for one run: all metrics | a question file and a prediction file | one `report.json` | Python only (optional metrics need models) |
| `evaluate.py` | `collect` | **Step 6.** A report for every finished run (`report.json` next to its predictions) and one table of all runs | `results/<system>/<model>/<view>_<split>/` | `results/summary.csv` | Python only (semantic similarity needs the sentence model; `--skip-text` leaves it out) |
| `evaluate.py` | `compare` | **Step 7.** Several runs on the same cases: McNemar for every pair, Holm-corrected | several runs | one `compare.json` | Python only |
| `check.py` | – | Where the pipeline stands: each step done / not run / problem, and why (reads only) | everything above | a table on the screen | Python only |
| `check.py` | `features` | The feature and shortcut check of the cleaned data (an analysis, not an experiment) | `data/processed_cases/all_cases.csv`, `data/processed_images/` | `results/feature_check/` | the image packages |

`common.py` holds what every script shares: the 8 classes and their names, the option texts, the seed, the layout of `data/`,
the balanced test rule (`balanced_splits`), safe writing (a file is replaced in one step, so an interrupted run never leaves half a
file), reading and logging. It needs only Python.

```powershell
python src/preprocess.py                                                  # step 1 (minutes; needs the images)
python src/build.py views                                                 # step 2 (seconds)
python src/build.py corpus                                                # step 3 (internet; downloads are cached)
python src/build.py references --token-check                              # step 3b (internet)
python src/check.py                                                       # where everything stands
python src/run.py llm-only --model gpt-4o --view text_only --split val --dry-run --limit 2   # see the prompt (free)
python src/run.py llm-only --model gpt-4o --view text_only --split val --limit 5             # a small paid trial
# models: claude-sonnet-5-5, gpt-4o, gemini-3.8-flash, llama-4-maverick (settings fixed in MODELS in run.py)
python src/evaluate.py --questions data/views/text_only/test.jsonl --predictions <predictions.jsonl> --out <report.json> `
    --cases data/processed_cases/all_cases.csv --by source facts_recorded
python src/evaluate.py --questions data/views/text_only/val.jsonl --predictions results/llm_only/<model>/text_only_val.partial/predictions.jsonl --answered-only --out <report.json>   # a trial: only the asked questions are scored
python src/evaluate.py collect                                            # all finished runs: report.json for each and results/summary.csv
python -m unittest discover -s tests                                      # the checks
```

A prediction line (written by `run.py llm-only`, read by `evaluate.py`): `{"id": "ISIC_0024948", "pred": "C"}`; optional
`"rationale"`, `"view"` (the text-only image-claim check needs it), `"tokens_in"`, `"tokens_out"`, `"seconds"`, and for RAG systems `"query"` and `"contexts"`.

## Tests (`tests/`)

| File | Level | What it checks | Needs |
|---|---|---|---|
| `test_unit.py` | unit | one function or rule at a time, on made-up input: common, the balanced test rule, cleaning, options, the metrics, compare, the prompt, the answer rule, the requests, the commands | Python (pandas for the cleaning tests) |
| `test_integration.py` | integration | parts working together on fake services: the corpus builder, the reference-text builder, a whole run with a fake model read by `evaluate.py`, `check.py` on small folders | Python only |
| `test_system.py` | system | the committed folder as it is: tables and views, the views rebuilt byte for byte, the balanced split, every command answering `--help`, the archives, the fixed top-level structure | Python only |
| `test_acceptance.py` | user acceptance | the researcher's own requirements, one test each: Kaggle only, 8 classes, balanced test, three views, four models, approved metrics only, sources named, quoted reference texts, no secret, no tag | Python only |

## Old names (renamed on 2026-11-22)

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
