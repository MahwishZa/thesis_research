"""Part 4 of the baseline: answering the questions and judging the answers (docs/methodology.md, sections 7.7 and 7.9).

The same model that wrote the rationale answers the question, with the helpful snippets in front of it or with none. The
answer is read from the text it writes, compared with the answer key, and the systems are compared on the same questions
with paired statistics.

    prompt  the answer prompt: the paper's prompt text, with an "Evidence" section when snippets are kept (D14)
    reader  the strict reading of the final answer, and the two ways an unreadable answer counts (D16)

Importing the package loads nothing heavy; these modules need no third-party package.
"""
