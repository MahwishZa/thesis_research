"""Which question belongs to which role, and how the questions of the three benchmarks are read (docs/methodology.md, 7.7).

The roles are training (labels are made from these questions), validation (k and the checkpoint are chosen on these) and
test (scored once at the end). A role is a function of the question id and a recorded seed only, so adding or removing
questions never moves another question.

    splits     one role per question id, from a hash of the id and the seed; a check that no id has two roles
    questions  the questions of MedQA, MedMCQA and MMLU-Med as one kind of record, and how a question is written out
    overlap    questions of one split that repeat questions of another
    cli        python -m data_roles ...

Importing the package loads nothing heavy; it needs no third-party package at all.
"""
