# Repository Structure

**Updated:** 2026-09-17, after simplifying folder and file names.

```
thesis_research/
├── README.md                      repository root readme (stays at root)
├── docs/                          every other Markdown document
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
│   ├── questions/                 Step 2: candidate question pool
│   │   ├── build_pool.py          builds, screens and exports the pool
│   │   ├── export_review.py       regenerates review.csv from candidates.jsonl
│   │   ├── candidates.jsonl       full pool, incl. internal flags (150 rows)
│   │   ├── review.csv             neutral reviewer export (123 rows)
│   │   └── sources/                adapters over inspected external sources
│   │       ├── cochrane.py         Cochrane reviews via MedRevQA
│   │       └── nih_medquad.py      NIH pages via MedQuAD (CC BY 4.0)
│   └── test_pairs/                Stage 2 matched-pair infrastructure
└── tests/                         unit and integration tests
```

## docs/

Every Markdown document except the root `README.md`, named
`lowercase_with_underscores.md`:

| File | Contents |
|---|---|
| `research_understanding.md` | the original research-understanding report |
| `research_ledger.md` | living decision log |
| `research_experimental_specification.md` | the frozen experimental spec |
| `frozen_scope.md` | canonical statement of what is primary vs. secondary |
| `proposal_scope_amendment.md` | edits required in the thesis proposal |
| `feasibility_and_alignment_audit.md` | read-only audit: RAG² alignment, dataset feasibility |
| `experimental_parity_audit.md` | baseline vs. proposed parity, the abstention decision |
| `rag2_classifier_feasibility.md` | the verified Flan-T5 filter training recipe |
| `hardware_and_resources.md` | verified student hardware, execution strategy |
| `question_sources.md` | the provenance protocol, written before sourcing |
| `question_pool_status.md` | candidate pool counts and known limitations |
| `question_review.md` | reviewer instructions for `experiments/questions/review.csv` |
| `experiment_outputs.md` | what `experiments/outputs/` contains and why it isn't research data |
| `external_evaluation_data.md` | what belongs in the (empty) external test-pair data directory |
| `repository_structure.md` | this file |
| `next_steps.md` | ordered action list |

## This round's renaming

**`experiments/question_sources/` → `experiments/questions/`.** Removed one
level of jargon and one level of nesting.

| Before | After |
|---|---|
| `experiments/question_sources/cochrane.py` | `experiments/questions/sources/cochrane.py` |
| `experiments/question_sources/nih_medquad.py` | `experiments/questions/sources/nih_medquad.py` |
| `experiments/question_sources/build_pool.py` | `experiments/questions/build_pool.py` |
| `experiments/question_sources/export_review.py` | `experiments/questions/export_review.py` |
| `experiments/question_sources/pool/candidates.jsonl` | `experiments/questions/candidates.jsonl` |
| `experiments/question_sources/pool/review.csv` | `experiments/questions/review.csv` |
| `experiments/question_sources/pool/HUMAN_REVIEW_INFORMATION.md` | `docs/question_review.md` |

The two source adapters (`cochrane.py`, `nih_medquad.py`) moved into a
`sources/` subpackage because they are a distinct concern from `build_pool.py`
and `export_review.py`: adapters read one external format each, while the
other two files orchestrate and export. `build_pool.py` kept its name — it
still builds a *candidate pool*, a term used throughout the docs, and the
directory rename already removes the awkward nesting without touching a name
that still describes what the file does.

**Every `.md` file except the root `README.md` moved to `docs/`, renamed from
`SCREAMING_SNAKE_CASE.md` to `lowercase_snake_case.md`.** Includes two
directory-local READMEs that were previously read in place:
`experiments/outputs/README.md` → `docs/experiment_outputs.md`, and
`experiments/test_pairs/data/external/README.md` →
`docs/external_evaluation_data.md`. The one hardcoded reference to the latter,
in `validate_external.py`'s error message, was updated to point at the new
location — **see the judgment-call note in the migration report** for why this
one is worth a second look.

Updated with the renames: every internal doc-to-doc cross-reference, the
`pyproject.toml` package list (`experiments.questions`,
`experiments.questions.sources` added — `experiments.question_sources` had
never been registered), imports in `tests/unit/test_question_sources.py` and
`tests/unit/test_review_export.py`, and the root `README.md`'s two doc
pointers. A repository-wide search for the old uppercase filenames and for
`question_sources` returns nothing outside historical prose in
`research_ledger.md`'s change log, which is intentionally left as a record of
what was true at the time.

## Earlier consolidation (previous round)

`evaluation/` was a second top-level package overlapping with `experiments/`.
It became `experiments/evaluation/`, chosen over `evaluations/` because
`experiments/` already held `configs/`, `outputs/`, `runners/` and
`test_pairs/`.

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
| Model weights | disk; see `hardware_and_resources.md` |

Each source adapter takes a path and fails with a message naming what to fetch,
rather than silently falling back to thesis-authored material.
