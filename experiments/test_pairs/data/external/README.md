# External evaluation material (primary pool)

This directory holds the **externally-authored** evaluation items that the
primary test-pair pool is built from. It is empty on purpose: the dataset is
fetched at build time and is not redistributed from here.

Ledger decision D-2 and specification section 33.2 put the thesis's primary
claim on material this thesis did not author. Nothing in this repository
generates that material, and `build_pairs.py --pool primary_external` fails
with a message naming the dependency rather than falling back to
thesis-written questions.

## Dependency — structure verified 2026-09-16 (D-34)

**MedChangeQA** (Vladika, Dhaini & Matthes, *Facts Fade Fast: Evaluating
Memorization of Outdated Medical Knowledge in Large Language Models*,
Findings of EMNLP 2025). Source of record:
<https://github.com/jvladika/MedChange>.

The released files have now been **inspected directly**. Everything below is
verified against them; nothing here is inferred from the paper's prose.

Reachability: `raw.githubusercontent.com` is permitted by the session egress
policy. `huggingface.co` and `eutils.ncbi.nlm.nih.gov` remain blocked, so any
step needing PubMed E-utilities must run elsewhere. No E-utilities call is
required for the primary pool — see the join below.

**Licensing: no `LICENSE` file is published.** Absent an explicit licence the
default is all-rights-reserved, and the underlying text is Cochrane Library
abstract content (Wiley copyright). Therefore **do not commit the dataset**.
Fetch it at build time, record the commit SHA and the file SHA-256, and cite
the paper.

## What the three released files actually contain

| File | Rows | Columns |
|---|---|---|
| `Datasets/MedChangeQA.csv` | 512 | `Question`, `Newest Label`, `Outdated Label` |
| `Datasets/MedRevQA.csv` | 16,501 | `background`, `objectives`, `conclusions`, `Question`, `Label`, `DOI_Date`, `Author`, `PMID` |
| `Datasets/AllStudyGroups.csv` | 4,379 | `Group_ID`, `Study_ID`, `Label` |

Three facts govern the build, and each one was checked:

1. **`MedChangeQA.csv` alone is not sufficient.** It carries a question and two
   verdict labels. It has **no PMIDs, no dates and no evidence text**. The
   older/newer passages this thesis needs are not in that file.
2. **`AllStudyGroups.csv` is the version linkage.** `Group_ID` is *sparse*
   (written once per group, blank on continuation rows) and must be
   forward-filled. It yields 1,535 groups of sizes 2–9. `Study_ID` is a
   **0-based row index into `MedRevQA.csv`** — verified by label agreement on
   4,379/4,379 rows (100%).
3. **The 512 are reconstructible exactly.** Groups holding more than one
   distinct `Label` number **512**, matching the published count, and in file
   order they align 1:1 with `MedChangeQA.csv` — verified by `Newest Label`
   agreement on 512/512 rows (100%).

`MedRevQA.csv` is a complete Cochrane census: 16,501/16,501 rows cite
*Cochrane Database of Systematic Reviews*, spanning 2000–2024.

## Acquisition and build procedure

```bash
# 1. Fetch the three files (record the commit SHA you fetched at).
BASE=https://raw.githubusercontent.com/jvladika/MedChange/main/Datasets
for f in MedChangeQA.csv MedRevQA.csv AllStudyGroups.csv; do
  curl -fsS -o "experiments/test_pairs/data/external/$f" "$BASE/$f"
done

# 2. Convert to the JSONL contract below via the documented join:
#      forward-fill Group_ID -> group rows -> Study_ID indexes MedRevQA
#      -> keep groups with >1 distinct Label -> order-align to MedChangeQA
#    Per side: evidence text = `conclusions`, document id = `PMID`,
#    publication date = year parsed from `DOI_Date`.
#    Oldest version = older side; newest version = newer side.

# 3. Check the result BEFORE building anything.
python -m experiments.test_pairs.scripts.validate_external \
    --input experiments/test_pairs/data/external/pairs_input.jsonl \
    --dataset "MedChangeQA @ <commit-sha>" \
    --output experiments/outputs/stage2_pilot/acquisition.json

# 4. Record the dataset name and commit in experiments/configs/stage2_pilot.yaml
#    (`external_dataset`), and set `extraction_date` when freezing.

# 5. Build the pairs. The full pool is used; nothing is sampled.
python -m experiments.test_pairs.scripts.build_pairs \
    --pool primary_external \
    --input experiments/test_pairs/data/external/pairs_input.jsonl \
    --output-dir experiments/outputs/stage2_pilot
```

**When fields are missing**, two different things happen, and the difference
is deliberate:

* a missing **required** field means the record does not identify an
  evaluation item. The validator marks the file unusable and exits non-zero;
  the builder refuses it. Neither fills the gap.
* a missing **optional** field costs that item, visibly. The validator says
  so in advance (`anticipated_exclusions`) and the builder records the
  exclusion so it appears in the attrition table.

The join above is deterministic: an integer row index and a published group
table. It involves **no lexical matching, no embeddings and no semantic
retrieval**, so it stays inside the tier-1 identifier-mapping rule.

Do **not** attempt to recover the two sides by matching `MedChangeQA.Question`
against `MedRevQA.Question`. That was tried and it fails: 320 of the 512
questions match exactly one review row, eleven match 366 rows each, and only 6
of 512 recover both labels. The `AllStudyGroups` index is the only correct
route.

## Measured properties of the 512 (recorded so they are not re-derived)

* **Temporal separation**, oldest to newest version: min 1 y, Q1 8 y,
  median **12 y**, Q3 16 y, max 23 y. This is the distribution
  specification 10.3 defers `min_separation_days` to; it can now be set from
  evidence rather than assumed.
* **Change points** (year of the newest version): range 2004–2024, peaking
  2013–2015. Only **1 of 512** falls after the Llama-3-8B pre-training cutoff
  (Dec 2023); 16 fall in 2023 or later. See ledger A1, which this retires as a
  blocking assumption for the reduced design.
* **Alzheimer's-domain items: 9 of 512** under a deliberately generous text
  criterion (`alzheimer|dementia|cognitive impairment|cognitive decline|mild
  cognitive`), of which only 2 name Alzheimer's disease. See ledger D-35 for
  why this rules out an AD-restricted primary pool.

## Why no corpus mapping step exists

An earlier reading of the specification suggested each external item's
evidence would have to be matched back onto Alzheimer's-corpus chunks. It does
not, and that step has been deliberately removed (ledger D-20).

MedChangeQA items are derived from **successive versions of the same Cochrane
review** — an earlier version reaching one verdict and a later version
reaching a different one. Both passages are therefore already determined by
the item, each with its own PMID and date. Nothing needs to be matched.

Specification 15.2 defines the admission effect as what happens *"when the
same candidate set is available"*. So Stage 3 places both passages into one
cached candidate set alongside corpus-retrieved passages, and replays that set
across every arm (specification 16). Whether frozen retrieval would have
surfaced them is a **retrieval** effect (15.1) — a separate, secondary
measurement, not a precondition for the bias probe.

This also means the primary probe is general-medical, not
Alzheimer's-specific. That is consistent with the provenance firewall, which
already places the primary claim on external material and confines
thesis-curated Alzheimer's evidence to replication and case study (D-2), and
with the understanding report's own observation that "the Stage-1 corpus is
not on the critical path for FRB-PAIRS if its passages come from external
MedChangeQA provenance".

**The caveat that used to sit here is discharged.** D-20 rested on an
unverified premise — that each item ships both review records (ledger E15).
It does, through `AllStudyGroups.csv`, and the route is a deterministic index
join. The E-utilities fallback described previously is no longer needed.

## Input contract

One JSON object per line. The loader validates these and supplies no
defaults, because a silently defaulted reference answer is indistinguishable
from a real one.

### Required

| Field | Meaning | Source in MedChange |
|---|---|---|
| `question_id` | Stable id; the partition unit, so all pairs for one question stay together | group index |
| `question_text` | The clinical question, as authored externally | `MedChangeQA.Question` |
| `reference_answer` | The correct answer as of the question date | `MedChangeQA.Newest Label` |
| `older_document_id` | Source record id (PMID) for the pre-change review | `MedRevQA.PMID` of oldest version |
| `newer_document_id` | Source record id (PMID) for the post-change review | `MedRevQA.PMID` of newest version |

Only these five are required: without them the record does not identify an
evaluation item. **Everything else is optional at load time** and its absence
becomes a counted exclusion rather than a load failure.

### Optional — absence is recorded and counted, never filled in

`older_text`, `newer_text` (from each version's `conclusions`; absent ⇒
`missing_evidence_text`), `older_publication_date`, `newer_publication_date`
(from `DOI_Date`; absent ⇒ `missing_publication_date`), `question_date`,
`change_point_date`, `older_evidence_id`, `newer_evidence_id`, `claim_class`,
`pair_category`, `contradiction_status`, and per-side `*_source_tier`,
`*_persistent_id`, `*_length_tokens`.

`evidence_id` is minted as `EXT:<dataset>:<question_id>:<side>` when the
dataset supplies none, so Stage 3 can address the passage without rebuilding
the pair definition.

## Dates: four different things

Keep these apart; the schema does.

| Field | Meaning |
|---|---|
| `question_date` (t_q) | The information state the admission decision is evaluated against |
| `*_publication_date` | When each passage appeared |
| `change_point_date` | When the clinical verdict moved — Check A counts these |
| `label_model_cutoffs` (config) | Pre-training cutoffs of the label-generating models |

`question_date` is taken from the dataset when it supplies one, otherwise from
the single configured `evaluation_as_of_date`. MedChangeQA supplies none, so
the configured fallback applies to every item. It is **never** derived from a
passage in the pair: setting t_q to the newer passage's publication date gives
that passage R = 1 by construction and inflates the recency contrast the
proposed method is being measured on (ledger D-21). The schema rejects such a
value outright.

A `change_point_date` the dataset does not supply is left absent, not
synthesised from a publication date. Check A reports
`check_a_interpretable: false` rather than a stratum count built on
substituted values. For MedChangeQA the newest version's publication year is
the closest available proxy and is **not** substituted silently.
