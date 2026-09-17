# Human Review — Alzheimer's Evaluation Questions

**What you have:** `review.csv`, 123 candidate questions.
**What we need:** your judgement on whether each one is defensible enough to
put in front of a research system and later defend in a thesis.

---

## What this review is for

These questions will be asked of two question-answering systems. Each system's
answer will then be judged against the evidence it was shown, to see whether it
stated things the evidence did not support.

That downstream judgement is only as good as the questions. If a question has
no determinate answer, two reasonable answers will disagree and neither is
wrong. If a reference answer says more than its source does, a system that
matches the source will be marked wrong. Either way the measurement stops
meaning anything, and the fault is invisible once the experiment has run.

**This is a review of the questions and their sources. It is not an evaluation
of any system, and no system output is involved.**

## Where the candidates came from

Each candidate was extracted from a published source that this project did not
write — peer-reviewed systematic reviews, and public-health pages from NIH
bodies. The question is the source's own question. The reference answer is one
sentence taken **verbatim** from the source's own conclusion or section text.
Nothing was paraphrased, and no language model wrote any question, answer or
citation.

That construction is mechanical, and mechanical extraction fails in ways only a
reader notices. It can take a sentence that cites correctly but answers a
different question than the one asked. It can take a sentence that is true but
too vague to judge anything against. It can pull a fragment whose meaning
depended on the sentence before it. Those are the failures this review exists
to catch.

**One thing has not been done and you should assume it has not:** the
identifiers — PMIDs, DOIs, Cochrane numbers, URLs — were copied from the source
dataset but never resolved. Nobody has confirmed that they point where they
claim to. Treat every locator as unverified.

## Judging each candidate from the evidence, not from us

`review.csv` deliberately gives you the question, the answer, and what you need
to trace that answer — and nothing else.

This pipeline did form opinions about these candidates along the way. Those
opinions are kept out of the review file on purpose. If a column said a
candidate looked weak, the natural thing is to go looking for the weakness and
find it; if a column said it looked strong, the natural thing is to skim. Both
would turn an independent assessment into agreement with an automated one, and
the automated one is exactly what needs checking.

So the 123 arrive undifferentiated. Some are stronger than others. Nothing in
the file tells you which, because working that out from the source is the task.

## Verifying the source exists and the locator points to it

Start from the locator. Cochrane items give a PMID, a review number such as
`CD013304.pub2`, and a DOI; NIH items give a MedQuAD document id, a UMLS
concept id and a page URL.

Look the identifier up and confirm you reach a real record — and that it is the
record named in `reference_source`. A PMID that resolves to a different paper is
a worse problem than one that resolves to nothing, because it looks correct.
Check that the title and author group match, and that the date is the one
recorded. For Cochrane, check the version suffix too: `.pub2` and `.pub3` are
different documents with potentially different conclusions.

Some NIH pages have moved. Two of the source sites were retired into
MedlinePlus, so a URL may redirect or fail. A redirect that lands on the
equivalent current page is usually fine; note it. A dead link with no successor
means the source cannot be checked, which is a real problem for that candidate.

## Verifying the source supports the reference answer

Once you have the record, find the sentence. It should be there, in those
words, in the conclusion or the section the locator names.

Then ask the harder question: does that sentence, read in its place in the
source, still mean what it appears to mean standing alone? A conclusion that
begins "In this subgroup, the effect was larger" says something quite different
without its preceding sentence. An extract carrying a qualifier whose referent
has been left behind is not supported by its source, even though every word
appears there.

Watch for a sentence that is verbatim and correctly cited but does not answer
the question asked — describing what the studies measured, or how confident the
authors were, rather than what was found. That is a common outcome of automatic
extraction and it is a rejection unless a better sentence in the same source
can replace it.

## Checking the answer does not overreach

Compare the reference answer to the question and to the source at the same time.

The answer should say what the source says, no more. Hedges matter: "may
reduce", "probably reduces" and "reduces" are three different claims, and a
reference answer that firms up a hedge has stopped representing its source.
Scope matters too — a finding about moderate disease stated as a finding about
the disease is an overreach, and so is a finding about one drug stated about a
class.

The answer should also be specific enough to judge against. "The evidence is
mixed" is faithful to many sources and useless as a reference: almost any
generated answer could be called consistent with it.

## Alzheimer's relevance

The question should be about Alzheimer's disease or the dementias in a way that
matters clinically for Alzheimer's — its diagnosis, treatment, course,
mechanisms, prevention or management.

Some candidates come from reviews about dementia in general where Alzheimer's is
the dominant cause, and those are usually in scope. Others mention Alzheimer's
only in passing, in background or as one example among many, while the question
itself is about something else. Those are not in scope, whatever the anchoring
suggests. The test is what the question is *about*, not whether the word appears.

## Clarity, specificity, determinacy, answerability

Read the question as a system would, with no context beyond the words.

It should be clear enough that two careful readers would understand it the same
way; specific enough that the population, intervention and outcome are evident
or obviously implied; and determinate enough that there is a fact of the matter.
"Is this treatment worthwhile?" has no determinate answer. "Does this treatment
improve cognition in mild Alzheimer's disease?" does.

Answerability is about whether the question can be settled from evidence at all
— not whether it is easy. A question whose honest answer is "the evidence does
not settle this" can still be a good evaluation item, provided the reference
answer says exactly that and a system claiming otherwise would be visibly wrong.

## Wording or factual basis that is not defensible

Some problems are not about accuracy. A question carrying an assumption into
its phrasing — asking which of two treatments is better when the source
compared neither — is not defensible regardless of its answer. Nor is one whose
wording only makes sense to someone who has read the source, since the system
will not have. Nor is a question whose answer has plainly been overtaken, where
the cited source is old and the field has moved; note that rather than fixing
it, because deciding what is current is a separate judgement.

When something feels wrong but you cannot name it, that is usually worth a
HOLD and a note. A vague note on a real problem is more useful than a confident
decision on the wrong question.

## The four outcomes

Record one in `review_decision`, with your reasoning in `reviewer_note`, plus
`reviewer_id` and `review_date`.

**ACCEPT** — provenance and question quality are both sufficiently defensible.
The source exists, the locator finds it, the sentence is there and supports the
answer, and the question is clear, relevant and determinate.

**REVISE** — the underlying factual proposition is defensible, but the wording
or the reference answer needs correcting. Use this when the source genuinely
supports the point and the problem is in how it has been expressed: an extract
that needs replacing with a better sentence from the same source, a question
that needs its scope stated, a hedge that needs restoring. **The proposition
must not change.** If the correction would make it a different claim, that is a
REJECT. Put the proposed wording in the note.

**REJECT** — provenance, factual support, relevance, clarity or answerability is
inadequate, and no revision within the same proposition fixes it. Say which.

**HOLD** — you need to verify the source before deciding. Use it freely; it is
the right answer when a link is dead, an identifier does not resolve, or you
need the full text to judge context. Say what would settle it.

## On the target

The final set is intended to be around 100 questions. That is a planning figure,
not a quota.

There is no requirement to reach it and no requirement to stay under it. If 108
are defensible, that is the set. If only 85 are, that is the set, and the thesis
reports the number honestly. Accepting a weak question to reach a round number
puts a flaw in the evaluation that no later analysis can remove — and the count
is the easiest thing in this project to defend, while a bad question is one of
the hardest.

Reject freely. The pool was built with room for it.

## Practical notes

Work in the order the file gives you. Partial reviews are fine — a row with no
decision is simply not yet reviewed.

Do not edit any column other than `review_decision`, `reviewer_note`,
`reviewer_id` and `review_date`. If a reference answer needs changing, describe
the change in the note and mark it REVISE; the change is applied afterwards, so
the original wording and its provenance stay recoverable.

If you notice two candidates that are the same question in different words, note
both ids. Near-duplicates were removed automatically, but only where the wording
was close enough for a machine to see it.
