"""Standard, automatic RAG evaluation metrics for the ablation study.

These are the metrics the current objectives' main evaluation (proposed vs
RAG²) and ablation study are scored with - see _archive/docs_legacy/current_objectives.md.
``accuracy.py`` remains the separate human-judged correctness track and
deliberately computes no automatic score of its own; this module owns
automatic scoring so the two cannot drift into two competing judges of the
same thing.

Every metric here is stdlib-only (matching stats.py's no-scipy convention)
and operates on plain strings/ids already present in a runner result record
plus one extra piece of information the runner does not itself carry: which
evidence ids are actually relevant to a question ("gold_evidence_ids"). That
has to come from the question/evidence pool, not from this module.

Metrics implemented:
  - exact_match: normalized string equality against the reference answer.
  - token_f1: SQuAD-style unigram precision/recall/F1 between the generated
    and reference answers.
  - rouge_l_f1: longest-common-subsequence-based F1, standard ROUGE-L.
  - context_precision / context_recall / context_f1: how well the admitted
    evidence ids match the question's gold-relevant evidence ids.
  - groundedness: the fraction of the generated answer's content tokens that
    also appear in the admitted evidence text. This is an automatic PROXY
    for faithfulness, not a substitute for the human hallucination-rate
    protocol in annotation.py/stats.py - it has no access to entailment or
    meaning, only token overlap, and is reported as such.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

_ARTICLES = {"a", "an", "the"}


def normalize_text(text: Optional[str]) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace.

    The standard SQuAD/RAG normalization: it is intentionally lossy so that
    trivial surface differences (capitalization, "a" vs "the", a trailing
    period) do not count as disagreement.
    """
    if not text:
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _tokens(text: Optional[str]) -> list[str]:
    return normalize_text(text).split()


def exact_match(prediction: Optional[str], reference: Optional[str]) -> float:
    """1.0 if the normalized strings match exactly, else 0.0."""
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def token_f1(prediction: Optional[str], reference: Optional[str]) -> float:
    """SQuAD-style unigram F1 between two answers.

    Returns 1.0 when both are empty (nothing to disagree on), 0.0 when only
    one is empty, and precision/recall F1 over multiset token overlap
    otherwise.
    """
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    ref_counts: dict[str, int] = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    overlap = sum(min(pred_counts[t], ref_counts.get(t, 0)) for t in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def rouge_l_f1(prediction: Optional[str], reference: Optional[str]) -> float:
    """Standard ROUGE-L: F1 over the longest common (in-order) subsequence."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class ContextScore:
    precision: float
    recall: float
    f1: float


def context_scores(
    admitted_evidence_ids: Iterable[str],
    gold_evidence_ids: Iterable[str],
) -> Optional[ContextScore]:
    """How well the admitted evidence matches the question's gold evidence,
    or ``None`` when the question carries no gold-evidence annotation.

    ``gold_evidence_ids`` must come from the question/evidence pool (a
    curated relevance judgement), never be guessed from the run itself.

    **Why "no gold" returns None rather than 0.0.** Context precision/recall
    are undefined without an annotation to score against, and for this
    thesis that is the EXPECTED case on real data, not an edge case: the
    provenance firewall means an externally-authored question's reference
    answer is deliberately independent of the retrieved candidate set (see
    ``FrozenItem.provenance_leak`` - an overlap there is a defect, not the
    goal). Returning 0.0 would then report "context precision 0.000" for
    every arm, which reads as a real, uniformly terrible result instead of
    "this metric does not apply to these questions" - and it would do so
    identically for the baseline and the proposed system, quietly diluting
    the comparison. ``aggregate()`` skips None and reports how many rows
    were scorable, so an unannotated run says so rather than inventing a
    number.
    """
    admitted = set(admitted_evidence_ids)
    gold = set(gold_evidence_ids)

    if not gold:
        return None
    if not admitted:
        return ContextScore(precision=0.0, recall=0.0, f1=0.0)

    overlap = admitted & gold
    precision = len(overlap) / len(admitted)
    recall = len(overlap) / len(gold)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return ContextScore(precision=precision, recall=recall, f1=f1)


_STOPWORDS = _ARTICLES | {
    "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was",
    "were", "be", "been", "it", "that", "this", "as", "at", "by", "from",
}


def groundedness(prediction: Optional[str], admitted_context: Optional[str]) -> float:
    """Fraction of the answer's content tokens that appear in the admitted
    evidence text.

    An automatic PROXY for faithfulness/groundedness, not the thesis's
    primary hallucination signal (that is human-annotated - see
    annotation.py). Token overlap has no access to entailment, negation or
    paraphrase, so a high score here does not certify a claim is actually
    supported, and a low score does not certify it is not. It is a cheap,
    deterministic diagnostic for comparing ablation cells against each
    other, not a substitute for the human protocol.

    Returns 1.0 for an empty prediction (nothing unsupported was claimed)
    and 0.0 for a non-empty prediction against empty/no admitted context
    (nothing could have grounded any claim).
    """
    pred_tokens = [t for t in _tokens(prediction) if t not in _STOPWORDS]
    if not pred_tokens:
        return 1.0

    context_tokens = set(t for t in _tokens(admitted_context) if t not in _STOPWORDS)
    if not context_tokens:
        return 0.0

    grounded = sum(1 for t in pred_tokens if t in context_tokens)
    return grounded / len(pred_tokens)


@dataclass(frozen=True)
class MetricRow:
    """One system's answer to one question, scored on every metric."""

    question_id: str
    system: str
    exact_match: float
    token_f1: float
    rouge_l_f1: float
    #: None when the question carries no gold-evidence annotation - see
    #: context_scores(). Not the same as 0.0, which means "annotated, and
    #: nothing the arm admitted was in the gold set".
    context_precision: Optional[float]
    context_recall: Optional[float]
    context_f1: Optional[float]
    groundedness: float


def score_record(
    record: Mapping[str, object],
    gold_evidence_ids: Iterable[str],
) -> MetricRow:
    """Score one runner result record (see runner.run_experiment's schema).

    ``record`` is one line of the JSONL a run_experiment call writes:
    it must carry ``question_id``, ``system``, ``generated_answer``,
    ``reference_answer``, ``admitted_evidence_ids`` and
    ``admitted_evidence_text``.
    """
    prediction = record.get("generated_answer") or ""
    reference = record.get("reference_answer") or ""
    admitted_ids = record.get("admitted_evidence_ids") or []
    admitted_text = " ".join(record.get("admitted_evidence_text") or [])

    ctx = context_scores(admitted_ids, gold_evidence_ids)

    return MetricRow(
        question_id=str(record["question_id"]),
        system=str(record["system"]),
        exact_match=exact_match(prediction, reference),
        token_f1=token_f1(prediction, reference),
        rouge_l_f1=rouge_l_f1(prediction, reference),
        context_precision=ctx.precision if ctx else None,
        context_recall=ctx.recall if ctx else None,
        context_f1=ctx.f1 if ctx else None,
        groundedness=groundedness(prediction, admitted_text),
    )


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    """Mean of the scorable values, or None when none are scorable."""
    scorable = [v for v in values if v is not None]
    if not scorable:
        return None
    return sum(scorable) / len(scorable)


def aggregate(rows: Sequence[MetricRow]) -> dict[str, Any]:
    """Mean of every metric across a set of rows (one system, typically).

    ``context_*`` are None when no row in the group carried a gold-evidence
    annotation, and are averaged over only the annotated rows otherwise;
    ``context_scored_n`` reports how many rows that was, so a reader can
    tell "0.0 across 10 annotated questions" from "not measurable here".
    """
    if not rows:
        return {
            "n": 0, "exact_match": 0.0, "token_f1": 0.0, "rouge_l_f1": 0.0,
            "context_precision": None, "context_recall": None,
            "context_f1": None, "context_scored_n": 0, "groundedness": 0.0,
        }
    n = len(rows)
    return {
        "n": n,
        "exact_match": sum(r.exact_match for r in rows) / n,
        "token_f1": sum(r.token_f1 for r in rows) / n,
        "rouge_l_f1": sum(r.rouge_l_f1 for r in rows) / n,
        "context_precision": _mean_or_none([r.context_precision for r in rows]),
        "context_recall": _mean_or_none([r.context_recall for r in rows]),
        "context_f1": _mean_or_none([r.context_f1 for r in rows]),
        "context_scored_n": sum(1 for r in rows if r.context_precision is not None),
        "groundedness": sum(r.groundedness for r in rows) / n,
    }


def aggregate_by_system(
    rows: Iterable[MetricRow],
) -> dict[str, dict[str, Any]]:
    """Group rows by system and aggregate each group."""
    by_system: dict[str, list[MetricRow]] = {}
    for row in rows:
        by_system.setdefault(row.system, []).append(row)
    return {system: aggregate(group) for system, group in by_system.items()}
