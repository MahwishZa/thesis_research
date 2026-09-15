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
| `older_evidence_id` | Corpus passage id for the pre-change side |
| `older_document_id` | Its document id |
| `newer_evidence_id` | Corpus passage id for the post-change side |
| `newer_document_id` | Its document id |

### Optional — absence is recorded, never filled in

`question_date`, `change_point_date`, `claim_class`, `pair_category`,
`contradiction_status`, and per-side `*_publication_date`, `*_source_tier`,
`*_persistent_id`, `*_length_tokens`.

When `question_date` is absent it falls back to the **newer side's**
publication date, per item, and `question_date_source` records that it was
derived. There is no global cutoff date.

## Note on the evidence ids

`older_evidence_id` and `newer_evidence_id` must be ids of passages in the
frozen Alzheimer's corpus, not free text copied from the dataset. Stage 2
emits references so that Stage 3 can build one cached candidate set and
replay it identically across arms (specification section 16). Linking the
dataset's evidence to corpus passages is a separate step and is not yet
implemented — the corpus it would link into does not exist yet.
