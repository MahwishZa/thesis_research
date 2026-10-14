"""The closed-book answers of other language models that the MedChange release contains (``Code/GeneratedAnswers``), read as verdicts.

Used by ``report`` (the closed-book rows of the tables, the existing models of the comparison). The release holds one answer per
MedRevQA row for each of eight models.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

#: The eight models whose closed-book answers the MedChange release contains, by the key used here.
FILES = {"qwen25-7b": "qwen25-7b_answers.txt", "llama33-70b": "llama33-70b_answers.txt",
         "mistral-24b": "mistral-24b_answers.txt", "gpt4o-mini": "gpt4o-mini_answers.txt",
         "deepsek-v3": "deepsek-v3_answers.txt", "biomistral": "biomistral_answers.txt",
         "pmcllama": "pmcllama_answers.txt", "olmo-13b": "olmo_gguf_answers_13b.txt"}
#: These three files hold three lines per answer ("Question: ...", "Answer: ...", "---"); the authors' own notebook
#: (Code/experiments_results.ipynb) reads the answers as ``lines[1::3]``. The others hold one line per answer.
THREE_LINE = ("biomistral", "pmcllama", "olmo-13b")
MODELS = tuple(FILES)
_LABEL = re.compile(r"NOT ENOUGH INFORMATION|SUPPORTED|REFUTED")


def find_label(text: str):
    """The first verdict word in an answer (any position), or None."""
    m = _LABEL.search(text.upper())
    return m.group(0) if m else None


def read_answers(folder: Path, model: str) -> Optional[list]:
    """One verdict (or None) per MedRevQA row from the release's answers of ``model``; None if the file is absent."""
    path = Path(folder) / FILES[model]
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    if model in THREE_LINE:
        if lines and lines[-1] == "":
            lines = lines[:-1]
        lines = lines[1::3]
    return [find_label(x) for x in lines]
