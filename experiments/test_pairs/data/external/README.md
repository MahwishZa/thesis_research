# External evaluation material (primary pool)

This directory holds the **externally-authored** evaluation items that the
primary test-pair pool is built from. It is empty on purpose.

Ledger decision D-2 and specification section 33.2 put the thesis's primary
claim on material this thesis did not author. Nothing in this repository
generates that material, and `build_pairs.py --pool primary_external` fails
with a message naming the dependency rather than falling back to
thesis-written questions.

## Outstanding dependency

**MedChangeQA** (Vladika et al., Findings of EMNLP 2025) — 512 changed-verdict
items, identified in the understanding report (section 20, Check A) as the
source of the temporal-counterfactual pairs.

It could not be retrieved from the session environment: outbound requests to
`huggingface.co` and `eutils.ncbi.nlm.nih.gov` are blocked by the
organisational egress policy (both return no response; `api.github.com`
returns 200, so the block is host-specific rather than a general network
failure).

Acquiring it is therefore a manual step, from a network that permits it:

1. Download the dataset from the source named in the paper.
2. Record its exact version or commit, and the download date.
3. Convert it to the JSONL contract below and save it here.
4. Set `external_dataset` in `experiments/configs/stage2_pilot.yaml` to the
   dataset name and version — it is copied into every pair's provenance.

Do not commit the dataset itself unless its licence permits redistribution.
Unknown licence means not redistributable.

## Why no corpus mapping step exists

An earlier reading of the specification suggested each external item's
evidence would have to be matched back onto Alzheimer's-corpus chunks. It does
not, and that step has been deliberately removed (ledger D-20).

MedChangeQA-style items are derived from **pairs of systematic-review
records** — an earlier review reaching one verdict and a later review reaching
the opposite one. The two passages are therefore already determined by the
item, each with its own identifier and date. Nothing needs to be matched.

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

**If the distributed files contain questions and verdicts but not the review
abstracts**, recover them by PMID with a single deterministic E-utilities
fetch per identifier. That stays in the deterministic-identifier tier: no
lexical matching, no embeddings, no semantic retrieval.

## Input contract

One JSON object per line. The loader validates these and supplies no
defaults, because a silently defaulted reference answer is indistinguishable
from a real one.

### Required

| Field | Meaning |
|---|---|
| `question_id` | Stable id; the partition unit, so all pairs for one question stay together |
| `question_text` | The clinical question, as authored externally |
| `reference_answer` | The correct answer as of the question date |
| `older_document_id` | Source record id (PMID) for the pre-change review |
| `newer_document_id` | Source record id (PMID) for the post-change review |

Only these five are required: without them the record does not identify an
evaluation item. **Everything else is optional at load time** and its absence
becomes a counted exclusion rather than a load failure — "the dataset has no
date for this item" is exactly the loss the pilot exists to measure.

### Optional — absence is recorded and counted, never filled in

`older_text`, `newer_text` (the passage text; absent ⇒ `missing_evidence_text`),
`older_publication_date`, `newer_publication_date` (absent ⇒
`missing_publication_date`), `question_date`, `change_point_date`,
`older_evidence_id`, `newer_evidence_id`, `claim_class`, `pair_category`,
`contradiction_status`, and per-side `*_source_tier`, `*_persistent_id`,
`*_length_tokens`.

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
the single configured `evaluation_as_of_date`. It is **never** derived from a
passage in the pair: setting t_q to the newer passage's publication date gives
that passage γ = 1 by construction and inflates the currency contrast SCAF is
being measured on (ledger D-21). The schema rejects such a value outright.

A `change_point_date` the dataset does not supply is left absent, not
synthesised from a publication date. Check A reports
`check_a_interpretable: false` rather than a stratum count built on
substituted values.
