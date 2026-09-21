"""MedQA questions and a general-medical passage corpus, for filter training.

Two things the paper's label function (labeling.py) needs that nothing else
in this repository provides:

1. **MedQA questions**, multiple-choice so correctness is checkable without
   a human (specification SS10.2). Loaded from the public
   ``GBaker/MedQA-USMLE-4-options`` dataset on the Hugging Face Hub - the
   standard, widely-used release of the USMLE-style MedQA benchmark
   (Jin et al., 2020), 4 options per question, matching the option format
   ``admission.py`` already documents ("... A) ... B) ... C) ... D) ...").
   This is a recorded choice, not a pinned fact from the RAG² release: the
   official repository's own data file
   (``classifier/data/medqa/llama3_cot/5%-train.json``) is a 5-example
   illustration, not the real MedQA split, and does not name its source
   dataset id (see specification SS10.1). If your MedQA source should be a
   specific different release, change ``MEDQA_DATASET_ID`` and say so.

2. **Evidence passages to retrieve against.** RAG²'s own filter is trained
   on retrieval from its 116.7M-passage general-medical corpus (PubMed +
   PMC + CPG + 18 textbooks, specification E4) - not available here (it is
   564 GB and was never built for this thesis, which deliberately indexes
   only an Alzheimer's slice, specification E4/SS10.6: "The filter is
   trained on general-medical MedQA, never on the thesis's Alzheimer's
   evaluation questions"). Using the *Alzheimer's* corpus for this would be
   worse than using nothing - it would make the "general-medical" filter
   secretly AD-specific, undermining the whole reason it's trained outside
   the thesis's own data. Instead this loads ``MedRAG/textbooks`` from the
   Hugging Face Hub: pre-chunked passages from medical textbooks - the same
   *category* of source RAG²'s own corpus includes (its Table A1 lists "18
   textbooks" as one of its four sources). This is NOT verified to be the
   identical 18 textbooks RAG² indexed - that isn't knowable from the
   released paper - so it is recorded here as a reasoned substitute, not
   presented as a fidelity guarantee.

Both loaders fail loudly and specifically if the actual dataset schema does
not match what this module expects, rather than silently reading garbage
columns - a schema mismatch here would poison every downstream label.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Optional, Sequence

from experiments.retrieval.corpus import CorpusPassage

#: Recorded choice - see the module docstring. Change and note in
#: docs/status_and_decisions.md if a different MedQA release is used.
MEDQA_DATASET_ID = "GBaker/MedQA-USMLE-4-options"

#: Recorded choice - see the module docstring.
TEXTBOOK_DATASET_ID = "MedRAG/textbooks"

OPTION_LETTERS = ("A", "B", "C", "D")


class MedQADataError(RuntimeError):
    """Raised when the actual dataset does not match what this module expects."""


@dataclass(frozen=True)
class MedQAItem:
    """One MedQA question, normalised to a fixed 4-option shape."""

    item_id: str
    question: str
    options: dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    answer_letter: str

    def __post_init__(self) -> None:
        if set(self.options) != set(OPTION_LETTERS):
            raise MedQADataError(
                f"{self.item_id}: options must have exactly keys "
                f"{OPTION_LETTERS}, got {sorted(self.options)}"
            )
        if self.answer_letter not in OPTION_LETTERS:
            raise MedQADataError(
                f"{self.item_id}: answer_letter must be one of "
                f"{OPTION_LETTERS}, got {self.answer_letter!r}"
            )

    def rendered_question(self) -> str:
        """The question with its options appended, exactly the form
        ``admission.py`` documents the released filter as trained on."""
        options_text = " ".join(
            f"{letter}) {self.options[letter]}" for letter in OPTION_LETTERS
        )
        return f"{self.question} {options_text}"


def load_medqa(
    n: Optional[int] = None,
    *,
    split: str = "train",
    seed: int = 42,
    dataset_id: str = MEDQA_DATASET_ID,
) -> tuple[MedQAItem, ...]:
    """Load (optionally a seeded random subsample of) MedQA questions.

    ``n=None`` loads the whole split - expect that to be tens of thousands
    of questions; pass an explicit ``n`` (the labelled-set size chosen from
    the timing measurement, specification SS10.4) for anything but a
    from-scratch full reproduction.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise MedQADataError(
            "load_medqa requires the 'datasets' package: pip install datasets"
        ) from exc

    raw = load_dataset(dataset_id, split=split)

    required_columns = {"question", "options", "answer_idx"}
    missing = required_columns - set(raw.column_names)
    if missing:
        raise MedQADataError(
            f"{dataset_id!r} split {split!r} is missing column(s) {missing} "
            f"- this module expects {sorted(required_columns)}, got "
            f"{raw.column_names}. The dataset schema has likely changed; "
            "update this loader rather than guessing which column is which."
        )

    rng = random.Random(seed)
    indices = list(range(len(raw)))
    if n is not None:
        if n > len(indices):
            raise MedQADataError(
                f"asked for n={n} but {dataset_id} split {split!r} only has "
                f"{len(indices)} examples"
            )
        rng.shuffle(indices)
        indices = sorted(indices[:n])  # sorted: stable, readable ids

    items = []
    for i in indices:
        row = raw[i]
        options = row["options"]
        if not isinstance(options, dict) or set(options) != set(OPTION_LETTERS):
            raise MedQADataError(
                f"row {i}: expected options to be a dict with keys "
                f"{OPTION_LETTERS}, got {options!r}"
            )
        answer_idx = row["answer_idx"]
        if answer_idx not in OPTION_LETTERS:
            raise MedQADataError(
                f"row {i}: expected answer_idx in {OPTION_LETTERS}, "
                f"got {answer_idx!r}"
            )
        items.append(MedQAItem(
            item_id=f"medqa_{split}_{i}",
            question=str(row["question"]).strip(),
            options={k: str(v).strip() for k, v in options.items()},
            answer_letter=answer_idx,
        ))
    return tuple(items)


def load_textbook_passages(
    n: Optional[int] = None,
    *,
    split: str = "train",
    seed: int = 42,
    dataset_id: str = TEXTBOOK_DATASET_ID,
) -> tuple[CorpusPassage, ...]:
    """Load (optionally a seeded random subsample of) textbook passages,
    as ``CorpusPassage`` - the same shape the Alzheimer's corpus reader
    produces, so the existing MedCPT/``DenseIndex`` retrieval code
    (experiments/retrieval/) works unmodified over this corpus too.

    ``publication_date`` is left ``None`` - a general-medical textbook
    passage has no meaningful publication date for this purpose, and
    filter-label generation has no temporal component (the Temporal
    Filter is the thesis's *proposed* system, evaluated only on the
    Alzheimer's question set - specification SS10.6).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise MedQADataError(
            "load_textbook_passages requires the 'datasets' package: "
            "pip install datasets"
        ) from exc

    raw = load_dataset(dataset_id, split=split)

    required_columns = {"id", "title", "content"}
    missing = required_columns - set(raw.column_names)
    if missing:
        raise MedQADataError(
            f"{dataset_id!r} split {split!r} is missing column(s) {missing} "
            f"- this module expects {sorted(required_columns)}, got "
            f"{raw.column_names}. The dataset schema has likely changed; "
            "update this loader rather than guessing which column is which."
        )

    rng = random.Random(seed)
    indices = list(range(len(raw)))
    if n is not None:
        if n > len(indices):
            raise MedQADataError(
                f"asked for n={n} but {dataset_id} split {split!r} only has "
                f"{len(indices)} passages"
            )
        rng.shuffle(indices)
        indices = sorted(indices[:n])

    passages = []
    for i in indices:
        row = raw[i]
        title = str(row["title"]).strip()
        content = str(row["content"]).strip()
        chunk_id = f"textbook_{split}_{row['id']}"
        passages.append(CorpusPassage(
            chunk_id=chunk_id,
            document_id=title or chunk_id,
            text=content,
            retrieval_text=f"[TITLE] {title}\n[TEXT] {content}" if title else content,
            publication_date=None,
            source_tier="textbook",
            retracted=False,
        ))
    return tuple(passages)


def extract_answer_letter(generated_text: str) -> Optional[str]:
    """Best-effort extraction of a predicted option letter from free-form
    generated text.

    Looks for the LAST standalone occurrence of a known option letter,
    reading generated CoT text the way a person skimming to the bottom for
    "so the answer is..." would. Not perfect - some generations will not
    contain an unambiguous letter, and this is a real, recorded limitation
    of automatic answer-extraction (specification SS10.2 notes MedQA
    correctness-checking is automatic, not that extraction is lossless).
    Returns ``None`` rather than guessing when nothing is found; callers
    must treat that as "could not be scored", not as any particular answer.
    """
    candidates = []
    for token in generated_text.replace(")", " ").replace(".", " ").replace(
            ":", " ").split():
        cleaned = token.strip(string.punctuation)
        if cleaned in OPTION_LETTERS:
            candidates.append(cleaned)
    return candidates[-1] if candidates else None
