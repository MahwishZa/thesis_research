"""Automated reproduction audit for the RAG2 baseline.

Answers one question: does the code in rag2/ actually implement the method the
original RAG2 paper describes? Run it with::

    python -m rag2_audit.run

The audit never imports torch/transformers, so it runs anywhere. Anything that
genuinely requires loading weights is reported as MANUAL rather than silently
skipped.
"""

__version__ = "1.0.0"

from .registry import CHECKS, Result, Status, check  # noqa: F401
