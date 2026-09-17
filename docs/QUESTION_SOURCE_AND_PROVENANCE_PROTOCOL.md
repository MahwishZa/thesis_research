# Question Source and Provenance Protocol

**Established:** 2026-09-17, before any question was sourced.

This defines how the Alzheimer's hallucination evaluation set is built. Its
purpose is to keep a single property true: **every reference answer is
traceable to a real record that this thesis did not author.**

---

## The chain

```
inspected external source record
        ↓  (the source states the fact)
factual proposition
        ↓  (the source states the question too, or it is read off the record)
candidate question
        ↓  (verbatim extract, never a paraphrase)
reference answer
        ↓
citation + stable locator + date
        ↓
automatic validation and deduplication
        ↓
human review                ← authority to approve lives here
        ↓
approved candidate
        ↓  (after the corpus completes)
corpus-support check
        ↓
final evaluation question
```

**What is forbidden.** A language model inventing a question, an answer and a
citation, any of which is then treated as ground truth. Search snippets as
reference evidence. An LLM cited as a source. A paraphrase presented as a
quotation, because the citation would no longer support the words.

**What automation does here.** Parsing, normalising, classifying by keyword,
deduplicating, checking completeness and formatting a review file. It does not
decide what is true. No LLM was used to write any question or answer in the
current pool: both are extracted verbatim from source records.

---

## The twelve questions

**1. Where do questions come from?** From the source record itself. A Cochrane
review states its own review question; an NIH page states its own section
question. Neither is composed by the thesis.

**2. Where does each reference answer come from?** One verbatim sentence of the
source's own conclusion or section text — the authors' bottom line. Methodological
preamble ("We included 24 trials…") and restated headings are skipped, because
they cite correctly but answer nothing.

**3. How is it independently verified?** It is not, yet. Every record carries
`verification_required: true` and `status: candidate`. A human confirms it
against the cited record before approval. **This audit could not resolve PMIDs
or DOIs: PubMed E-utilities and doi.org are blocked by the network policy here.
Identifiers are transcribed verbatim from the source dataset, not checked to
resolve.** That check is part of human review.

**4. What date is recorded?** The source's publication date where it has one
(Cochrane: year and month, parsed from the citation). Where the source has no
date — MedQuAD carries none — the retrieval date is recorded and flagged
`reference_date_is_retrieval_date: true`, so the two are never confused.

**5. How is Alzheimer's relevance established?** Two mechanisms, neither a bare
keyword match on the question:
* **Cochrane:** the record must name Alzheimer's *somewhere*, and the question
  or objectives must be about Alzheimer's, dementia or cognition. This is the
  corpus's own rule — a dementia record qualifies when an Alzheimer's anchor is
  present, never on the dementia term alone.
* **NIH:** UMLS CUI **C0002395** (Alzheimer's Disease), assigned by NLM
  indexers. An ontology code does not match a passing mention.

**6. How is determinacy established?** Cochrane items carry an explicit verdict
label, which is what makes the question answerable; `NOT ENOUGH INFORMATION` is
still a determinate finding *about the evidence base* and is flagged ambiguous
so a reviewer sees it. A keyword screen flags opinion-shaped wording. Final
determinacy is the reviewer's call.

**7. How is corpus support checked?** Not yet. `corpus_support_expected` means
*"expected to be answerable from the corpus once built"*, nothing more. The real
check runs after the corpus completes and is a separate step.

**8. Temporal questions.** `temporal_candidate` is set when the Cochrane
citation carries `.pub2` or higher — the review was republished, so its
conclusion has been revisited. **This does not assert the verdict changed.** It
is a diagnostic attribute for error analysis, not a primary criterion.

**9. Ambiguous questions.** `ambiguity_candidate` is recorded, not acted on.
Ambiguity is a *cause* of hallucination, so these are the cases most likely to
expose the behaviour under study; discarding them would remove the signal. A
question with genuinely no determinate answer is rejected at review.

**10. Duplicates.** Content-derived ids catch exact repeats. Jaccard over
normalised tokens at 0.85 catches near-duplicates; the first occurrence is kept
and later ones are marked `rejected` with a reason rather than deleted, so the
counts still reconcile. Deliberately crude and deterministic — an embedding
model would make the set depend on an unreproducible judgement.

**11. How is provenance preserved?** Every record stores `reference_source`,
`reference_source_type`, `reference_locator`, `reference_date`, the source
dataset and citation, and how the answer was extracted. Locators are concrete —
`PMID:34918337 | CD013304.pub2 | doi:10.1002/14651858.CD013304.pub2`, or
`MedQuAD:6_NINDS_QA/0000001 | qid:… | CUI:C0002395 | <url>` — never a bare title.

**12. Who approves?** A human reviewer, and only a human. The builder emits
`candidate` or `rejected`; no code path produces `approved` or `final`. A
reviewer may accept, reject, revise or flag for source verification. A revised
reference answer must keep its provenance, and a revision that the citation no
longer supports is a rejection, not an edit.

---

## Source categories, and what each may be used for

| Category | Status | Acceptable for | Not acceptable for |
|---|---|---|---|
| **A. Peer-reviewed evidence synthesis** (Cochrane) | **In use** | treatment, diagnosis, prevention, prognosis — anything resting on aggregate evidence | facts the review does not state |
| **D. Government / public health** (NIH via MedQuAD) | **In use** | disease characteristics, genetics, symptoms, epidemiology | fine-grained treatment efficacy, where synthesis is stronger |
| **C. Guidelines / consensus documents** | Not yet used | diagnostic criteria, recommendations, temporal facts | — |
| **B. Research organisations** | Not yet used | epidemiology, general characteristics | efficacy claims |
| **E. Existing ADRD QA datasets** | Not used | — | licensing and provenance unverified |
| **F. The thesis corpus itself** | **Excluded by design** | — | **anything** — a question written from a passage later shown to the model is circular |

Preference order where sources conflict: a dated peer-reviewed synthesis beats
an undated public-health page. Reviewers should prefer Cochrane items when both
cover the same fact.

---

## Known limitations of the current pool

1. **Two sources, unevenly weighted** — about 85% Cochrane. Part of the reason
   is the network policy in the build environment, which blocked every source
   except GitHub-hosted datasets. Adding a guideline source (category C) would
   strengthen the pool and is a manual step.
2. **Identifiers are transcribed, not resolved** (see question 3).
3. **Topic distribution is skewed to treatment**, because Cochrane is
   intervention-heavy. Not corrected: forcing balance would mean dropping good
   items or inventing weak ones. Reported instead.
4. **MedQuAD items are undated**, and two of its AD source sites
   (NIHSeniorHealth, Genetics Home Reference) have since been retired into
   MedlinePlus, so their URLs may redirect.

---

## Test-set independence

The pool and the final evaluation set are not used to select θ, λ, H,
generation temperature, prompt wording, retrieval parameters or any model
setting. Those come from a separate development set. `AdmissionConfig.validate()`
already raises if θ is unresolved, so a test-set-fitted threshold cannot be
inherited silently.
