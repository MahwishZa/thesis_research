# Repository Structure

**Updated:** 2026-09-17, after consolidating the evaluation code.

```
thesis_research/
├── docs/                          audit and protocol documents
├── alzheimer_corpus/              Stage 1: corpus pipeline, config, metadata
│   ├── config/                    MeSH vocabulary, claim taxonomy, queries
│   └── scripts/                   01_pubmed → 07_claim_classification
├── systems/                       the three experimental arms
│   ├── interfaces/                Evidence, Candidate, ExperimentResult,
│   │                              Generator, Retriever, System
│   ├── baseline/                  no-filter control · RAG²-style Flan-T5 filter
│   └── proposed/                  the proposed solution/system
├── experiments/                   ← ONE top-level evaluation/experiment folder
│   ├── configs/                   run configuration
│   ├── outputs/                   run artefacts (never overwritten)
│   ├── runners/                   reserved for launch scripts
│   ├── evaluation/                evaluation infrastructure
│   │   ├── questions.py           question schema, validation, dedup, export
│   │   ├── freezing.py            frozen manifest, hashing, parity gate,
│   │   │                          provenance firewall
│   │   ├── annotation.py          annotation schema, blinding, agreement
│   │   ├── stats.py               HAR, coverage, McNemar, bootstrap, Holm
│   │   └── runner.py              replays a frozen set through every arm
│   ├── question_sources/          Step 2: sourcing adapters and the pool
│   │   ├── cochrane.py            Cochrane reviews via MedRevQA
│   │   ├── nih_medquad.py         NIH pages via MedQuAD (CC BY 4.0)
│   │   ├── build_pool.py          builds, screens and exports the pool
│   │   └── pool/                  candidates.jsonl · review.csv
│   └── test_pairs/                Stage 2 matched-pair infrastructure
└── tests/                         unit and integration tests
```

## The consolidation

`evaluation/` was a second top-level package overlapping with `experiments/`.
It is now `experiments/evaluation/`.

| Before | After |
|---|---|
| `evaluation/questions.py` | `experiments/evaluation/questions.py` |
| `evaluation/freezing.py` | `experiments/evaluation/freezing.py` |
| `evaluation/annotation.py` | `experiments/evaluation/annotation.py` |
| `evaluation/stats.py` | `experiments/evaluation/stats.py` |
| `evaluation/runner.py` | `experiments/evaluation/runner.py` |

`experiments/` was kept rather than `evaluations/`: it already held `configs/`,
`outputs/`, `runners/` and `test_pairs/`, so the move touched fewer paths and
left fewer chances for a stale reference. Moved with `git mv`, so history
follows the files.

Updated with it: imports in `tests/unit/test_evaluation.py` and
`tests/integration/test_smoke_pipeline.py`; the package list in
`pyproject.toml`; path references in `docs/NEXT_ACTIONS.md` and
`docs/EXPERIMENTAL_PARITY_AUDIT.md`. A repository-wide search for
`from evaluation` / `import evaluation` returns nothing, and 229 tests pass.

`experiments/test_pairs/` was **kept**. It holds the matched-pair, attrition,
split, power and provenance machinery from the earlier temporal design. That
design is now a mechanism diagnostic rather than the primary claim, but the
infrastructure is sound and deleting it would destroy working, tested code for
a cosmetic gain.

## What is not in the repository, and why

| Not vendored | Reason |
|---|---|
| The Alzheimer's corpus data | size; built locally; ignored by `alzheimer_corpus/.gitignore` |
| MedRevQA / MedChangeQA CSVs | no upstream licence file; Cochrane abstract text |
| MedQuAD checkout | CC BY 4.0 but large; cloned on demand |
| Model weights | disk; see `HARDWARE_AND_RESOURCE_STATUS.md` |

Each source adapter takes a path and fails with a message naming what to fetch,
rather than silently falling back to thesis-authored material.
