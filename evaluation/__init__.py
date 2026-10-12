"""Statistics shared by the analysis scripts (``evaluation/stats.py``): the exact McNemar test, the paired bootstrap interval and the Holm correction.

The evaluation of the study itself (metrics, families of tests, the reading of the differences) is
specified in ``docs/evaluation.md`` and implemented in ``experiments/medchange/analyze_rag2.py``. The tests of the
repository are in ``tests/``. The original corpus-independent evaluation framework (question schema, freezing contract,
annotation workflow, hallucination-rate accounting) was removed on 2026-10-06 (Git history, commit 5e03540).
"""
