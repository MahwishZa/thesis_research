"""Part 3 of RAG²: rationale-guided filtering (Sohn et al., NAACL 2025, section 3.2).

A small sequence-to-sequence model judges each retrieved snippet on its own, helpful or not helpful, and only the helpful
snippets reach the answering model. Its training labels come from comparing the answer, and the perplexity of the
rationale, without and with the snippet (Figure 2 of the paper).

    labels      the label rule of Figure 2, the training tokens, and the threshold of the top 25% of perplexity differences
    inference   the filter input, the decision between the two label tokens, and keeping the helpful snippets

Importing the package loads nothing heavy; these modules need no third-party package.
"""
