# Archived: the original Alzheimer's-specific framework

Archived on 2026-10-05. **Nothing here is used by the current research** (`README.md`, `docs/experimentation.md`).
It is kept, tested and runnable, because it is the record of the work the thesis started from.

## What it was

The first design of the thesis: a RAG² baseline and a Temporal Filter compared on Alzheimer's questions, over a local
Alzheimer's corpus, with a hallucination rate scored by a human annotator.

| Part | What it does |
|---|---|
| `corpus/` | Seven stages over PubMed/PMC full text (`scripts/01_…` to `07_…`): 114,256 PMC records, 4,377,041 chunks, claim-type tags. Tracked: queries and configs (`config/`), registries and metadata (`metadata/`), reports, logs. Not tracked: the text (`data/`). |
| `experiments/shared/questions/` | The reviewed Alzheimer's question pool (123 reviewed, 113 usable; split by seed 20260921) with its builder, review export and splitter. |
| `experiments/shared/retrieval/` | Dense retrieval over the local index: corpus reader, index, retrieval pipeline, memory-bounded index build (4,376,141 × 768, ≈ 12.5 GB). |
| `experiments/shared/runners/run_end_to_end.py` | The fixture demo: all arms, evaluation and ablation on synthetic data. |
| `src/baseline/`, `src/temporal_filter/admission.py`, `src/common/{generator,hf_generator,system}.py` | The three-arm systems: RAG² baseline slot (needs a trained filter checkpoint that is not distributed), No-Filter control, threshold-and-budget Temporal Filter, generator interfaces. |
| `evaluation/` | Question schema, freezing contract (candidate-set hashes, provenance firewall), runner with parity assertions, annotation workflow (blinding, agreement), QA judgements, RAG metrics (token F1, ROUGE-L, context precision/recall, groundedness) and hallucination-rate accounting (`har_stats.py`). |
| `tests/` | 465 tests of all of the above, run with the other archived work (592 tests in all: `python -m unittest discover -s _archive -t .`). |

## Why it is archived

* The current pipeline imports none of it (checked by an import analysis on 2026-10-05; only
  `src.common.evidence`, `src.temporal_filter.{scorer,temporal}`, `evaluation.stats` and the MedCPT encoders, which stayed
  active, are shared).
* The local corpus cannot support an as-of test: 71% of it is from 2020 or later, while the Alzheimer's questions mostly
  cite older reviews (`docs/data.md`). The current study uses per-question as-of PubMed evidence instead.
* The Alzheimer's evaluation is now the 212 dementia/Alzheimer's questions of MedRevQA (`experiments/medchange/ad_benchmark.py`),
  not the reviewed pool.
* The human-annotated hallucination rate was dropped: nobody involved can judge medical evidence
  (`docs/experimentation.md` §1, `docs/log.md` Phase 29).

## Where things were (before 2026-10-05) and are now

| Before | Now |
|---|---|
| `corpus/` | `_archive/alzheimers_framework/corpus/` |
| `experiments/shared/questions/` | `_archive/alzheimers_framework/experiments/shared/questions/` |
| `experiments/shared/retrieval/{corpus,index,pipeline,build_index,streaming_index_build}.py` | `_archive/alzheimers_framework/experiments/shared/retrieval/` |
| `experiments/shared/retrieval/encoders.py` | **`experiments/medchange/encoders.py`** (active: MedCPT, used by `freeze_candidates.py` and `rag2_run.py`) |
| `experiments/shared/runners/` | `_archive/alzheimers_framework/experiments/shared/runners/` |
| `experiments/results/README.md` | `_archive/alzheimers_framework/experiments/results/README.md` |
| `src/baseline/` | `_archive/alzheimers_framework/src/baseline/` |
| `src/common/{generator,hf_generator,system}.py` | `_archive/alzheimers_framework/src/common/` (`evidence.py` stays active in `src/common/`) |
| `src/proposed/{temporal,scorer}.py` | **`src/temporal_filter/`** (active, renamed: the Temporal Filter is a stage-1 result of record, no longer "the proposed system") |
| `src/proposed/admission.py` | `_archive/alzheimers_framework/src/temporal_filter/admission.py` |
| `evaluation/{runner,freezing,questions,accuracy,rag_metrics,annotation}.py` | `_archive/alzheimers_framework/evaluation/` |
| the hallucination functions of `evaluation/stats.py` | `_archive/alzheimers_framework/evaluation/har_stats.py` (`evaluation/stats.py` keeps `mcnemar`, `paired_bootstrap_ci`, `holm`) |
| `evaluation/tests/{unit,integration}/` tests of the above, `corpus_scaffold.py` | `_archive/alzheimers_framework/tests/` |

## Running it

From the repository root (the paths are the new ones; the commands are otherwise unchanged):

```bash
python -m unittest discover -s _archive -t .                          # its tests, with the other archived work
python -m _archive.alzheimers_framework.experiments.shared.runners.run_end_to_end        # the fixture demo
python _archive/alzheimers_framework/corpus/scripts/04_normalize.py --input data/raw/pubmed/records.example.jsonl   # offline fixture
python -m _archive.alzheimers_framework.experiments.shared.retrieval.build_index --device cpu   # dense index (needs the corpus data)
```

The corpus scripts need `pip install -e ".[archive]"` (PyYAML, requests, pypdf). The tests of the corpus scripts
copy a lean scaffold of `corpus/` and never touch `corpus/data/`.

## If your computer still holds the corpus data or the index

They are not tracked and were not moved. After a `git pull` the old folders `corpus/data/` and
`experiments/results/index/` stay where they are, and the root `.gitignore` keeps them out of Git (so `git add -A` is
safe). The current study does not use them. Options:

* **Leave them.** Nothing breaks.
* **Free the disk space** (≈ 12.5 GB index plus the corpus text) once you are sure you will not rerun the archived
  pipeline. Rebuilding them took days of compute, so do not delete them casually.
* **Reuse them from the archive** (Windows PowerShell, from the repository root; `/MOVE` moves and merges):
  `robocopy corpus\data _archive\alzheimers_framework\corpus\data /E /MOVE` and
  `robocopy experiments\results _archive\alzheimers_framework\experiments\results /E /MOVE`.

## Restoring it to the active tree

Everything is in Git history. To undo the move completely, revert the commit named "Archive the superseded
Alzheimer's framework" (`git revert <hash>`), or check out the recovery point `fa4cb61` (the last commit before the
move) in a separate folder: `git worktree add ..\thesis_before_move fa4cb61`.
