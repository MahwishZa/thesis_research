"""The rationale prompt of RAG² (Sohn et al., NAACL 2025, section 3.3).

The paper gives the prompt in a box and says that ``[initial_query]`` is replaced by the initial query x. The question
passed in already carries its answer options, so the options reach the language model as part of the question.

Two details are not stated by the paper and are kept as one named setting each, so that they can be changed without touching
anything else:

* the separator before "Here is the question:". The printed box has one line break there: the line above has room for
  "Here is", the line spacing is even (no blank line) and "Here" is not indented, so it is neither a wrap, a blank line nor a new
  paragraph. The setting is therefore one line break. Whether the authors' own string has exactly this break is not stated;
* the chat format and the system message (the paper shows only this text, so it is sent as one user message and no
  system message is added; that belongs to the generator, not to this file).
"""

from __future__ import annotations

INSTRUCTION = (
    "The following are multiple choice questions about medical knowledge. Solve them in a step-by-step fashion, "
    "starting by summarizing the available information. Output your explanation and single option from the given "
    "options as the final answer."
)
LEAD_IN = "Here is the question:"

#: What stands between the instruction and "Here is the question:": one line break, as the printed box shows (see the module docstring).
SEPARATOR = "\n"


def build_rationale_prompt(question: str) -> str:
    """The prompt for one question: the paper's box with ``[initial_query]`` replaced by ``question``, unchanged.

    The question is not stripped, shortened or reformatted. An empty or blank question is an error: it is never
    replaced by anything.
    """
    if not isinstance(question, str):
        raise TypeError(f"the question must be a string, got {type(question).__name__}")
    if not question.strip():
        raise ValueError("the question is empty")
    return f"{INSTRUCTION}{SEPARATOR}{LEAD_IN} {question}"
