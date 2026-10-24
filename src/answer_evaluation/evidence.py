"""The snippets that each system reads: the re-ranked list as it is, or the helpful snippets of it (docs/methodology.md, D13).

In the authors' retriever one ``top_k`` sets both the candidates taken from each corpus and the number of snippets kept after
re-ranking, so the list for k = 4 is not the first four of the list for k = 32: another pool is re-ranked. Every k therefore has
its own retrieval folder, and ``EvidenceProvider`` takes one list per k. Two systems read these lists:

    rag2_no_filter  the first k snippets of the list for k, as they are
    rag2            the same snippets, each judged by the filter on its own, and only the helpful ones kept in order

The filter judges a snippet from the question and the snippet alone, so a decision is kept and reused when the same snippet comes
up again for the same question, whichever k asked for it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from balanced_retrieval.records import RetrievalStore
from rationale_filter.inference import decisions_for, keep_helpful
from rationale_query.generator import text_sha256
from rationale_query.search_text import dense_query
from rationale_query.store import Question, StoreError, question_hash

from .runner import Arm, EvalQuestion

NO_FILTER = "rag2_no_filter"
WITH_FILTER = "rag2"
FILTER_ARMS = (NO_FILTER, WITH_FILTER)


def read_evidence(retrieved_dir: str | Path, questions: Sequence[Question], rationales: Mapping[str, str]) -> dict[str, list[str]]:
    """The snippet texts of every question, in re-ranked order, from a ``balanced_retrieval retrieve`` folder.

    Every question needs a list that was found for its own text and its own rationale (the hashes must agree), or the call
    raises; nothing is left out quietly.
    """
    records = RetrievalStore(retrieved_dir).load()
    lists: dict[str, list[str]] = {}
    for question in questions:
        record = records.get(question.id)
        if record is None:
            raise StoreError(f"{question.id} has no evidence in {retrieved_dir}")
        if record.question_sha256 != question_hash(question.text) or record.rationale_sha256 != text_sha256(dense_query(rationales[question.id])):
            raise StoreError(f"{question.id}: the evidence in {retrieved_dir} was found for another question or rationale")
        texts = []
        for rank, passage in enumerate(record.evidence, start=1):
            text = passage.get("text") if isinstance(passage, Mapping) else None
            if not isinstance(text, str) or not text.strip():
                raise StoreError(f"{question.id}: snippet {rank} has no text")
            texts.append(text)
        lists[question.id] = texts
    return lists


def retrieval_top_k(retrieved_dir: str | Path) -> int:
    """The k a retrieval folder was made with (it is in its ``config.json``)."""
    config = RetrievalStore(retrieved_dir).read_config()
    top_k = config.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise StoreError(f"{retrieved_dir} does not record the top_k it was made with")
    return top_k


class EvidenceProvider:
    """``provider(question, arm)`` gives the snippets that the arm reads for the question. Use it as the ``evidence`` of ``run_arms``."""

    def __init__(self, lists: Mapping[int, Mapping[str, Sequence[str]]], judge_many: Callable[[list[str]], Sequence[bool]] | None = None, *, batch_size: int = 8) -> None:
        self.lists = lists
        self.judge_many = judge_many
        self.batch_size = batch_size
        self._decisions: dict[str, bool] = {}
        self.considered: dict[int, int] = {}
        self.kept: dict[int, int] = {}

    def __call__(self, question: EvalQuestion, arm: Arm) -> list[str]:
        if arm.name not in FILTER_ARMS:
            raise ValueError(f"the arm {arm.name!r} has no evidence; the arms with evidence are {', '.join(FILTER_ARMS)}")
        per_question = self.lists.get(arm.k)
        if per_question is None:
            raise ValueError(f"no evidence lists were given for k = {arm.k}")
        if question.id not in per_question:
            raise ValueError(f"{question.id} has no evidence list for k = {arm.k}")
        snippets = list(per_question[question.id])[: arm.k]
        if arm.name == NO_FILTER:
            return snippets
        if self.judge_many is None:
            raise ValueError("the arm rag2 needs a filter")
        decisions = decisions_for(question=question.text, snippets=snippets, judge_many=self._judge)
        kept = keep_helpful(snippets, decisions)
        self.considered[arm.k] = self.considered.get(arm.k, 0) + len(snippets)
        self.kept[arm.k] = self.kept.get(arm.k, 0) + len(kept)
        return kept

    def _judge(self, inputs: list[str]) -> list[bool]:
        keys = [text_sha256(text) for text in inputs]
        new = [(key, text) for key, text in dict(zip(keys, inputs, strict=True)).items() if key not in self._decisions]
        if new:
            assert self.judge_many is not None
            answers = list(self.judge_many([text for _, text in new]))
            if len(answers) != len(new):
                raise ValueError(f"{len(new)} filter inputs but {len(answers)} decisions")
            for (key, _), answer in zip(new, answers, strict=True):
                self._decisions[key] = bool(answer)
        return [self._decisions[key] for key in keys]

    def summary(self) -> dict[str, Any]:
        """How many snippets the filter looked at and how many it kept, for each k: the share kept is what the filter changes."""
        return {
            k: {"considered": self.considered[k], "kept": self.kept.get(k, 0), "kept_share": round(self.kept.get(k, 0) / self.considered[k], 4)}
            for k in sorted(self.considered)
            if self.considered[k]
        }
