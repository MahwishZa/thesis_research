"""Component 1 of the RAG² baseline: rationale-based query formulation (docs/methodology.md, section 2).

A language model writes a step-by-step rationale for a medical multiple-choice question, and the rationale alone becomes
the retrieval query for the next component.

    prompt   the paper's rationale prompt, with the question (and its options) in place of [initial_query]

Importing the package loads nothing heavy.
"""
