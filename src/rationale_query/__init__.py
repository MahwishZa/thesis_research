"""Component 1 of the RAG² baseline: rationale-based query formulation (docs/methodology.md, section 2).

A language model (Meerkat-7B, run with llama.cpp) writes a step-by-step rationale for a medical multiple-choice question,
and the rationale alone becomes the retrieval query for the next component.

    prompt      the paper's rationale prompt, with the question (and its options) in place of [initial_query]
    generator   the model: Meerkat's chat format, greedy decoding, no length limit except the memory window
    store       the questions file and the saved rationales, so that a long run can be stopped and started again
    run         the loop that writes a rationale for every question that has none yet
    query       the rationale as the retrieval query, and the JSON list that the retriever reads
    cli         python -m rationale_query generate | show-prompt | export-queries

Importing the package loads nothing heavy: llama_cpp is imported only when a model is loaded.
"""
