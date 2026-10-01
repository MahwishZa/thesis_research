# Draft email to the RAG² authors

Do **not** invent addresses: take the first author's and the corresponding/senior
author's emails from the first page of the paper (ACL Anthology / arXiv 2411.00300)
and cc your supervisor. Fill the bracketed fields. Read GitHub issues #1 and #2 of
`dmis-lab/RAG2` first - they may already answer part of this.

---

**Subject:** Research-use request: RAG² filter checkpoint (or filter decisions) for an extension to temporal evidence in Alzheimer's QA

Dear Dr. [Sohn] and Dr. [Kang],

I am an MS student at [university], supervised by [supervisor, cc'd], working on an
extension of RAG² (NAACL 2025) that adds the publication age of a passage to the
admission decision, evaluated on Alzheimer's-disease questions. Thank you for
releasing the retriever and classifier code.

We would like RAG² as a faithful baseline. We understand from the README that the
trained filter checkpoint is not available for distribution, and that the shipped
`5%-train.json` is a 5-example sample (its ids reach `llama3_5%_23600`). We tried to
retrain: we generated 500 MedQA labels with Llama-3-8B-Instruct (4-bit, CPU) using
the perplexity decision tree, but the resulting filter learned only the class prior.
Our control tests suggest the labels were largely noise at that scale and
quantisation (log: [link to docs/log.md on your repository]).

Would any of the following be possible, in order of preference?

1. The trained Flan-T5 filter checkpoint for non-commercial research use. We will
   accept any terms, cite RAG² and acknowledge you.
2. If you cannot distribute it: running your filter on a fixed list of about 2,300
   (question, passage) pairs that we would send, returning [HELPFUL]/[NOT_HELPFUL]
   and the two logits. This is all our baseline needs, since retrieval is frozen.
3. At minimum: the exact CoT/rationale prompt, how perplexity was computed and the
   threshold used, and the true size and class balance of the `5%` split.

We are happy to share our code and results, and to credit your help in any
resulting work. Thank you for your time.

Kind regards,
[Name]
[Programme, university]
[Repository link]

---

Follow-up: if there is no reply, send one polite reminder after 2-3 weeks. Do not
let the project wait on this; plan as if no reply comes.
