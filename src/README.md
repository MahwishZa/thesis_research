# Code guide

Run every command from the repository folder. Each script explains itself at the top and with `--help`.

| Order | Script | Reads | Writes | Needs |
|---|---|---|---|---|
| 1 | `preprocess.py` | `data/raw/`, `data/official_labels/` | `data/processed_images/`, `data/processed_cases/` | the Kaggle download; pandas, OpenCV, scikit-learn |
| 2 | `build_views.py` | `data/processed_cases/all_cases.csv` | `data/views/<view>/cases.csv`, `{train,val,test}.jsonl` | Python only |
| 3 | `build_knowledge_corpus.py` | PubMed, PMC, guidelines, Dermoscopedia (internet) | `data/knowledge_corpus/` | internet; `datasets` for the Meditron guidelines |
| 3b | `build_reference_texts.py` | Dermoscopedia (internet) | `data/knowledge_corpus/reference_texts.json`: one quoted reference text per disease, for semantic similarity | internet |
| – | `feature_check.py` | `data/processed_cases/all_cases.csv`, `data/processed_images/` | `results/feature_check/` | an analysis of the data, not an experiment |
| 4 | `run_llm_only.py` | `data/views/<view>/<split>.jsonl` (+ images) | `results/llm_only/<model>/<view>_<split>/` | keys in `.env`; costs credit |
| 5 | `evaluate.py` | a question file and a prediction file | one `report.json` | Python only (optional metrics need models) |
| any | `check_pipeline.py` | everything above | a table: each step done / not run / problem, and why | Python only |
| 6 | `compare.py` | several runs on the same cases | McNemar for every pair, Holm-corrected | Python only |

`common.py` holds what every script shares: the 8 classes and their names, the option texts, the seed, the layout of `data/`,
safe writing (a file is replaced in one step, so an interrupted run never leaves half a file), reading and logging.

```powershell
python src/preprocess.py                                                  # step 1 (minutes; needs the images)
python src/build_views.py --data data                                     # step 2 (seconds)
python src/build_knowledge_corpus.py                                      # corpus (internet; downloads are cached)
python src/run_llm_only.py --model gpt-4o --view text_only --split val --dry-run --limit 2   # see the prompt (free)
python src/run_llm_only.py --model gpt-4o --view text_only --split val --limit 5             # a small paid trial
# models: claude-sonnet-5-5, gpt-4o, gemini-3.8-flash, llama-4-maverick (settings fixed in MODELS in the script)
python src/evaluate.py --questions data/views/text_only/test.jsonl --predictions <predictions.jsonl> --out <report.json> `
    --cases data/processed_cases/all_cases.csv --by source facts_recorded
python -m unittest discover -s tests                                      # the checks
```

A prediction line (written by `run_llm_only.py`, read by `evaluate.py`): `{"id": "ISIC_0024948", "pred": "C"}`; optional
`"rationale"`, `"tokens_in"`, `"tokens_out"`, `"seconds"`, and for RAG systems `"query"` and `"contexts"`.
