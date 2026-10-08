"""Statistics shared by the analysis scripts (``evaluation/stats.py``) and the repository's tests.

The evaluation of the study itself (metrics, families of tests, the reading of the +1 point requirement) is
specified in ``docs/evaluation.md`` and implemented in ``experiments/medchange/analyze_rag2.py``. The
original corpus-independent evaluation framework (question schema, freezing contract, annotation workflow,
hallucination-rate accounting) was removed on 2026-10-06 (Git history, commit 5e03540).
"""
