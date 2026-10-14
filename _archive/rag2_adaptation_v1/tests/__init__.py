"""Tests of the repository (``python -m unittest discover -s tests -t .``).

One file per module under test, named ``test_<module>.py``: ``test_rag2.py`` covers the adapted RAG² logic, its step runner, its phase
driver and the analysis, ``test_audits.py`` the two audits, ``test_fresh_pipeline.py`` the pre-registered test, and
``test_repository_guards.py`` the agreement of the documents, the committed results and the layout. ``check_hermetic.py`` runs the
suite and fails if it left any file of the repository changed.

The tests of the original framework and of the Alzheimer's corpus scripts were removed with it on 2026-10-06 (Git history, commit 5e03540).
"""
