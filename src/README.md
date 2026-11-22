# Code guide

Run every command from the repository folder. Each script explains itself at the top and with `--help`.

| Order | Script | Reads | Writes | Needs |
|---|---|---|---|---|
| 1 | `preprocess.py` | `data/raw/`, `data/meta/` | `data/images/`, `data/processed/` | the Kaggle download; pandas, OpenCV, scikit-learn |
| 2 | `build_views.py` | `data/processed/master_clean.csv` | `data/views/<view>/cases.csv`, `qa_{train,val,test}.jsonl` | Python only |
| 3 | `build_knowledge_corpus.py` | PubMed, PMC, guidelines, Dermoscopedia (internet) | `data/corpus/` | internet; `datasets` for the Meditron guidelines |
| – | `feature_check.py` | `data/processed/master_clean.csv`, `data/images/` | `results/feature_check/` | an analysis of the data, not an experiment |
| 4 | *(next)* a run script per system | `data/views/<view>/qa_test.jsonl` | predictions, one JSON line per question | the chosen LLM |
| 5 | `evaluate.py` | a question file and a prediction file | one `report.json` | Python only (optional metrics need models) |

`common.py` holds what every script shares: the 8 classes and their names, the option texts, the seed, the layout of `data/`,
safe writing (a file is replaced in one step, so an interrupted run never leaves half a file), reading and logging.

```powershell
python src/preprocess.py --raw data/raw --meta data/meta --out data       # step 1 (minutes; needs the images)
python src/build_views.py --data data                                     # step 2 (seconds)
python src/build_knowledge_corpus.py --out data/corpus                    # corpus (internet; downloads are cached)
python src/evaluate.py --qa data/views/text_only/qa_test.jsonl --pred <predictions.jsonl> --out <report.json> `
    --cases data/processed/master_clean.csv --by source facts_recorded
python -m unittest discover -s tests                                      # the checks
```

A prediction line for `evaluate.py`: `{"id": "ISIC_0024948", "pred": "C"}`; optional `"rationale"`, `"query"`, `"contexts"` (RAG),
`"tokens_in"`, `"tokens_out"`, `"seconds"`.
