# Question Pool Status

**Built:** 2026-09-17 · Reproduce with:

```bash
python -m experiments.question_sources.build_pool \
    --medrevqa <path>/MedRevQA.csv \
    --medquad  <path>/MedQuAD \
    --retrieved-on 2026-09
```

Source files are not vendored (see the protocol). Output lives in
`experiments/question_sources/pool/`.

> **Nothing in this pool is final.** Every record is `candidate` or `rejected`.
> No question has been human-reviewed, and no question has been checked against
> the Alzheimer's corpus, which is still downloading.

## Counts

| | |
|---|---|
| Sourced from adapters | 170 |
| After dropping identical-id repeats | 150 |
| **Auto-validated, awaiting human review** | **123** |
| Auto-rejected | 27 |
| Near-duplicate pairs detected | 35 |
| Identical-id repeats dropped | 20 |
| Distinct source records | 133 |
| AD-anchored | 150 |
| Determinate | 150 |
| Temporal candidates | 74 |
| Ambiguity candidates | 47 |
| Missing provenance | 0 — the schema refuses a record without source, locator and date |
| Approved / final | **0** |

Target is roughly 100 after review. 123 candidates gives room
to reject weak ones without dropping below it.

## Rejections

| Reason | Count |
|---|---|
| near_duplicate_of_earlier_candidate | 27 |

All automatic rejections were near-duplicates. Rejected records are kept in
`candidates.jsonl` with their reason, so counts reconcile and a reviewer can
see what collided.

## Topic distribution

| Topic | Count |
|---|---|
| diagnosis | 17 |
| disease_characteristics | 6 |
| disease_course | 5 |
| epidemiology | 4 |
| general | 7 |
| management | 17 |
| mechanism | 6 |
| prevention | 9 |
| treatment | 79 |

Skewed to treatment because Cochrane is intervention-heavy. **Not corrected** —
forcing balance would mean dropping sound items or inventing weak ones. A
reviewer wanting more diagnosis or epidemiology coverage should add a
guideline source rather than rebalance this one.

## Source types

| Type | Count |
|---|---|
| government_public_health | 22 |
| peer_reviewed_evidence_synthesis | 128 |

## Human review

`pool/review.csv` carries the 123 validated candidates with
blank `reviewer_decision` and `reviewer_note` columns. The reviewer may accept,
reject, revise or flag for source verification. A revision must keep its
provenance; a revision the citation no longer supports is a rejection.

## Limitations

1. **Identifiers were transcribed, not resolved.** PubMed E-utilities and
   doi.org are blocked in the build environment, so no PMID or DOI was
   confirmed to resolve. Every record carries `verification_required: true`.
   Spot-checking these is the first review task.
2. **Two sources, ~85% Cochrane.** A guideline source would strengthen it.
3. **MedQuAD items carry no publication date**; the retrieval date is recorded
   and flagged as such. Two of its AD source sites have been retired into
   MedlinePlus and their URLs may redirect.
4. **`corpus_support_expected` is an expectation, not a check.** The real check
   runs when the corpus completes.
5. **Reference answers are one-sentence verbatim extracts.** Short by design,
   for copyright and for judgeability — but a reviewer should confirm each one
   answers its question rather than merely relating to it.
