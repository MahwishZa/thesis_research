"""Part 4 of the baseline: answering the questions and judging the answers (docs/methodology.md, sections 7.7 and 7.9).

The same model that wrote the rationale answers the question, with the helpful snippets in front of it or with none. The
answer is read from the text it writes, compared with the answer key, and the systems are compared on the same questions
with paired statistics.

    prompt        the answer prompt: the paper's prompt text, with an "Evidence" section when snippets are kept (D14)
    reader        the strict reading of the final answer, and the two ways an unreadable answer counts (D16)
    paired_tests  exact McNemar, the paired bootstrap interval and Holm's adjustment (D24)
    ledger        a record of the test sets that were scored, refusing a second scoring of the same system
    records       one resumable record per answer: question, system and k
    runner        every question answered by every system, resumable; a test run needs the ledger
    results       accuracy per system, the paired comparison of two systems, and the choice of k on validation data

Importing the package loads nothing heavy; these modules need no third-party package.
"""
