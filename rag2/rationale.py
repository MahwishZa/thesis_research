"""Stage 1: rationale-based query formulation (paper section 3.3).

The backbone LLM is prompted with the paper's chain-of-thought prompt and the
resulting rationale becomes the retrieval query. The paper is explicit that the
initial query is *not* concatenated: "We search for document snippets solely
using the rationale, excluding the initial query."
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from .llm.base import LLM
from .prompts import DEFAULT_PROMPTS, PromptSet
from .schema import Question


def generate_rationales(
    llm: LLM,
    questions: Sequence[Question],
    prompts: Optional[PromptSet] = None,
    batch_size: int = 8,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    progress: Optional[callable] = None,
) -> Dict[str, str]:
    """Return ``{qid: rationale}`` for every question.

    ``max_new_tokens``/``temperature`` are passed explicitly rather than left to
    the backend's own defaults, so the rationale's decoding settings are visible
    at the call site instead of being implied by whichever config the LLM object
    happened to be built with.
    """
    prompts = prompts or DEFAULT_PROMPTS
    rendered = [prompts.render_rationale_prompt(q) for q in questions]
    generate_kwargs: Dict[str, Any] = {}
    if max_new_tokens is not None:
        generate_kwargs["max_new_tokens"] = max_new_tokens
    if temperature is not None:
        generate_kwargs["temperature"] = temperature
    out: Dict[str, str] = {}
    for start in range(0, len(rendered), max(batch_size, 1)):
        chunk = rendered[start : start + max(batch_size, 1)]
        completions = llm.generate(chunk, **generate_kwargs)
        for question, completion in zip(questions[start : start + len(chunk)], completions):
            out[question.qid] = completion.strip()
        if progress:
            progress(min(start + len(chunk), len(rendered)), len(rendered))
    return out


def retrieval_query(rationale: str, question: Question, use_rationale: bool = True) -> str:
    """The string handed to the retriever.

    ``use_rationale=False`` reproduces the paper's ``MedCPT`` baseline row, which
    retrieves with the initial query instead.
    """
    if use_rationale and rationale.strip():
        return rationale.strip()
    return question.question.strip()
