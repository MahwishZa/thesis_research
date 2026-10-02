# Recency-Weighted Evidence Admission in Retrieval-Augmented Medical Question Answering

An as-of evaluation on Cochrane verdict changes (MedChangeQA), with an Alzheimer's disease case study.

## 1. Problem statement

Medical evidence is revised: a systematic review's conclusion can change as new trials appear. A
retrieval-augmented question-answering (RAG) system whose evidence-admission step judges passages on
textual relevance alone cannot prefer current evidence over superseded evidence; it treats an outdated
and a current passage as equally admissible if both are topically relevant. RAG² (Sohn et al., NAACL
2025) exemplifies this: a trained filter decides which retrieved passages an LLM sees, but never looks at
*when* a passage was published. The risk is an answer that reflects a conclusion the literature has since
revised.

## 2. Research question and objectives

> On medical questions whose Cochrane verdict changed between review versions, asked as of the newest
> review's publication date with only evidence published before that date, does recency-weighted evidence
> admission make a local LLM give the current verdict more often than standard RAG, a helpfulness-ranked
> admission and a published-style recency reranking, without lowering accuracy on questions whose verdict
> did not change?

**Objectives**

1. To implement and validate the proposed system using predefined evaluation metrics for retrieval and
   generation performance.
2. To determine the extent to which the proposed system improves retrieval and generation performance
   compared with relevant baseline models and existing works.

Both are addressed by direct comparison, controls, an ablation and paired significance tests, not by
assuming an improvement. Either outcome directly answers Objective 2; this repository does not commit in
advance to which one it will report.

## 3. Approach

The **proposed system (the Temporal Filter)** adds one signal to evidence admission: how recent a passage
is relative to the question date. A(s) = (1 − λ)·ρ(s) + λ·T(s, q, t_q), where ρ is a relevance rank and
T = 2^(−age/H) is plain age decay. Recency-aware retrieval is an established idea (e.g. TempRALM); **no
novelty is claimed for the formula.** The contribution, if the results hold, is a controlled as-of
evaluation on real verdict changes with a falsification control, and an honest measurement of whether the
signal helps.

```
question → as-of candidates (PubMed, before the newest review) → MedCPT rank + rerank → frozen pool of 20
        → arm-specific admission (≤ 5 passages) → one local LLM, one prompt, greedy → VERDICT line
        → verdict accuracy against the gold verdict, paired tests, error analysis
```

| Arm | Admission | Role |
|---|---|---|
| B0 | none | the model's own knowledge |
| B1 | cross-encoder top-5 | standard RAG |
| B2 | zero-shot helpfulness top-5 | RAG²-inspired, untrained |
| B3 | cross-encoder + recency | published-style recency reranking |
| **P** | helpfulness + recency | proposed |
| C1 | as P with dates shuffled | falsification control |

**Primary outcome: verdict accuracy** (SUPPORTED / REFUTED / NOT ENOUGH INFORMATION against the newest
review's verdict, parsed from a fixed `VERDICT:` line; no judge model). Retrieval-level measures such as the
share of admitted passages from the update window are manipulation checks, never outcomes.

**Honest scope.** RAG²'s trained filter is not distributed and a local retraining attempt failed (archived),
so B2 and P use an **untrained** Flan-T5 helpfulness score and are **not** a RAG² reproduction. Gold labels
are model-generated. The generator is a 4-bit 8B model on CPU. See `docs/methodology.md` (deviation
register) and `docs/experiment_plan.md` §12.

## 4. Status (2026-10-02)

| Component | Status |
|---|---|
| Alzheimer's corpus (114,256 PMC records, 4,377,041 chunks), question pool (113 usable), dense index (4,376,141 × 768) | built (secondary) |
| MedChange benchmark: 504 usable changed + 250 unchanged items, seeded dev/confirmatory splits | built, verified against the release |
| Gate G0 (new evidence exists before each review date) | passed: 94.0% of changed dev items |
| Dev candidate pools (226 items) and zero-shot helpfulness scores | built |
| Arms, generation harness, analysis (accuracy, retrieval metrics, McNemar + Holm, gates) | built and unit-tested |
| Gates G1–G3; **any accuracy result** | pending: none exists yet |
| Confirmatory split; human hallucination annotation; second generator; Alzheimer's as-of case study | planned |
| RAG² filter reproduction; v1 Alzheimer's pilot runners | abandoned / superseded, archived |

## 5. Repository structure

```
README.md
pyproject.toml
docs/                    methodology, data, evaluation, reproducibility, glossary, experiment plan, log
experiments/
├── medchange/           PRIMARY pipeline: benchmark, as-of pools, arms, generation, analysis (+ results/)
├── shared/              question pool, retrieval pipeline (MedCPT, dense index), framework demo runner
└── results/             gitignored retrieval indexes
src/                     reference implementation of the admission mechanisms (common, baseline, proposed)
evaluation/              metrics, freezing, statistics, annotation, framework runner, tests/
corpus/                  Alzheimer's evidence corpus pipeline (stages 01-07) and its provenance records
_archive/                abandoned or superseded work, kept for history (see _archive/README.md)
```

`_archive/` is not part of the active pipeline and nothing active imports it (checked by
`evaluation/tests/unit/test_scope_invariants.py`).

| Document | Read it for |
|---|---|
| [`docs/experiment_plan.md`](docs/experiment_plan.md) | The protocol: benchmark, arms, settings, outcomes, statistics, gates, decision rules, amendments |
| [`docs/methodology.md`](docs/methodology.md) | What is compared and how; what is held constant; deviation register |
| [`docs/data.md`](docs/data.md) | Datasets, provenance, limitations; what is committed and what is rebuilt |
| [`docs/evaluation.md`](docs/evaluation.md) | Metrics, statistics, controls, evaluation status |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Install, test, run; measured costs; results policy |
| [`docs/glossary.md`](docs/glossary.md) | Term definitions |
| [`docs/log.md`](docs/log.md) | Chronological record of work, decisions and results |

## 6. Quick start

```bash
pip install -e ".[models,medchange]"
python -m unittest discover -s evaluation -t .                 # active suite: no network, no models
python -m experiments.medchange.build_benchmark --medchange-dir ../MedChange
```

The full MedChange workflow (PubMed probe, candidate freezing, helpfulness scores, generation, analysis)
and its measured costs are in `docs/reproducibility.md` §3. Nothing there has produced an accuracy result yet.
