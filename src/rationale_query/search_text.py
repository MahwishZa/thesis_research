"""Which text is searched with, and which text the re-ranker reads (paper p.4-5)."""

from __future__ import annotations


def _clean(text: object, what: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"the {what} must be a string, got {type(text).__name__}")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"the {what} is empty")
    return cleaned


def dense_query(rationale: str) -> str:
    """The text that is encoded for the search: the rationale alone, with surrounding whitespace removed.

    The question is not added: both together exceed the retriever's length limit and gave worse results in the
    authors' tests. An empty rationale is an error; it is never replaced by the question, because that would
    change the method.
    """
    return _clean(rationale, "rationale")


def rerank_query(*, question: str, rationale: str, against: str = "question") -> str:
    """The text the cross-encoder reads together with each snippet.

    The paper (p.5) says the original question; the authors' released code passes the rationale. The default
    follows the paper, and ``against="rationale"`` is the code's variant, run as a labelled sensitivity check.
    """
    if against == "question":
        return _clean(question, "question")
    if against == "rationale":
        return _clean(rationale, "rationale")
    raise ValueError(f"against must be 'question' or 'rationale', got {against!r}")
