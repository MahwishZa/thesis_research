# Data

Covers the evidence corpus and the evaluation question pool: what they are,
where they came from, and their current state. For how retrieval consumes
the corpus, see `methodology.md`. For how to (re)build any of this, see
`reproducibility.md`.

## 1. The Alzheimer's evidence corpus

**Status: complete and frozen.** Built by a seven-stage pipeline
(`corpus/scripts/01_pubmed_download.py` through `07_claim_classification.py`),
sourced from PubMed/PMC full text plus government public-health pages, and
verified end to end by cross-checking each stage's output count against the
next stage's independently-reported input count — not by trusting any
stage's self-report in isolation.

| Stage | What it does |
|---|---|
| 01 PubMed | Retrieves a PMID list (no text or dates — see stage 02) |
| 02 PMC retrieval + finalisation | Downloads full text from PMC, resolves publication dates from JATS XML |
| 03 Guidelines / textbooks | Optional; not populated in the current corpus (no document could be independently source-verified in this build environment) |
| 04 Normalise | Text cleanup, AD-relevance tagging |
| 05 Deduplicate | Near-duplicate removal (content id + token-Jaccard) |
| 06 Chunk | Real `ncbi/MedCPT-Article-Encoder` tokenizer, fixed window/stride |
| 07 Claim classification | Tags every chunk by claim type and evidence level |

Every tracked provenance file (`corpus/metadata/*.csv`, `corpus/reports/*.csv`,
`corpus/logs/*.log`) is well-formed and free of unresolved errors. The full
corpus text itself (`corpus/data/**`) is **not committed** — it is large,
built locally, and gitignored by design; only the logs, reports, and
registries above are tracked provenance.

**Known, non-blocking gaps**: the guideline/textbook registry (stage 03) is
implemented but empty (no document was added without independent source
verification); the PMC redistribution-licence gate is recorded per record
but not enforced, which affects redistribution of corpus text, not research
use.

## 2. Retrieval index

A retrieval index is built over the corpus for a given run
(`experiments/shared/retrieval/build_index.py`). **The corpus above and the
index built over it are not the same thing** — the corpus is complete, but
an index build can (and, for early runs, does) use a reduced slice of it,
which is stated wherever it affects a run. Every frozen item records a
`corpus_snapshot` id so a manifest is traceable to exactly which build
produced it.

## 3. Evaluation question pool

**Provenance rule**: every reference answer must be traceable to a real,
external, published record — never authored by this project or by a
language model. Forbidden: an invented question, answer, or citation; a
paraphrase presented as a quotation; a search snippet as evidence.

```
external source record → factual proposition → candidate question
→ verbatim reference answer → citation + locator + date
→ automatic validation and deduplication
→ human review (the only source of approval)
→ final evaluation question
```

**Sources**: peer-reviewed systematic reviews (Cochrane) for treatment,
diagnosis, prevention and prognosis questions; government public-health
pages (NIH, via MedQuAD) for disease-characteristics, genetics, symptoms
and epidemiology questions. Where sources conflict, a dated peer-reviewed
synthesis outranks an undated public-health page.

**Human review** (`docs/reproducibility.md` links the reviewer worksheet):
every candidate is judged ACCEPT / REVISE / REJECT / HOLD against source
verification, relevance, clarity, specificity, determinacy and whether the
reference answer overreaches its source. No code path can produce an
approved question.

**Validation/test split**: usable (ACCEPT + REVISE) questions are split into
a validation set (used only to fit `λ`/`θ`/`H`) and a held-out test set
(used only to report), stratified by topic and by whether the question's
evidence base is known to have changed over time. Two splits, not three —
nothing in this thesis trains on this pool; it only feeds parameter fitting,
which needs a set to fit on and a separate, untouched set to report on.

`temporal_candidate` marks a question whose cited Cochrane review has been
revised at least once (`.pub2` or higher) — diagnostic only, it does not
assert that the verdict itself changed. This is the subgroup where a
temporal signal should matter most if it matters at all.

## 4. Known limitations of the data

1. Identifiers (PMIDs, DOIs) were transcribed from source datasets, not
   independently resolved in every build environment — flagged per record.
2. Sources are predominantly Cochrane (~85%); a guideline source would
   strengthen topic balance.
3. Some public-health source pages carry no publication date; the retrieval
   date is recorded and flagged as such rather than treated as a real date.
4. Reference answers are single verbatim sentences, short by design for
   copyright and judgeability — not full-paragraph summaries.
