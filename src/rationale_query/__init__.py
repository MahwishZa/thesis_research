"""Part 1 of RAG²: rationale-based query formulation (Sohn et al., NAACL 2025, section 3.3).

The language model writes a step-by-step rationale for the question. The rationale alone, without the question,
is the query that the MedCPT query encoder turns into a vector for the search in Part 2.

    prompt       the paper's prompt text
    search_text  which text is searched with and which text the re-ranker reads
    generator    the language model behind one interface (llama.cpp), with every setting recorded
    store        questions in, one resumable record per question out
    batch        the loop that turns questions into rationale records
    encoder      the MedCPT query encoder (first-token vector, 512 tokens)
    diagnostics  what the rationales look like: length, truncation, final option
    cli          python -m rationale_query ...

Importing the package loads nothing heavy: llama_cpp, torch, transformers and numpy are imported only where used.
"""
