"""Standard, automatic RAG evaluation metrics for the ablation study.

``accuracy.py`` deliberately excludes automatic string-overlap scoring
(ROUGE/BLEU/EM/F1) from the thesis's PRIMARY correctness signal, and that
decision stands: hallucination rate (human-annotated) is still primary, QA
accuracy (human- or exact-match-judged) is still secondary. This module does
not reopen that decision.

What it adds is a separate, net-new track: the ablation study asked for in
the current objectives, which is required to use STANDARD, automatic RAG
metrics so different admission configurations (lambda, theta, no-filter,
RAG², proposed) can be compared cheaply, deterministically, and without a
human annotator in the loop for every ablation cell. These numbers are a
diagnostic signal for the ablation, not a replacement for the primary
human-judged outcomes recorded elsewhere.

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
from typing import Iterable, Mapping, Optional, Sequence

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
) -> ContextScore:
    """How well the admitted evidence matches the question's gold evidence.

    ``gold_evidence_ids`` must come from the question/evidence pool (e.g. a
    curated relevance judgement), not be guessed from the run itself. An
    empty gold set makes precision/recall undefined; this returns 0.0 for
    both rather than raising, so a question with no annotated gold evidence
    does not crash an ablation sweep - callers that need to distinguish
    "no gold evidence" from "zero overlap" should check the gold set length
    themselves.
    """
    admitted = set(admitted_evidence_ids)
    gold = set(gold_evidence_ids)

    if not admitted and not gold:
        return ContextScore(precision=1.0, recall=1.0, f1=1.0)
    if not admitted or not gold:
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
    context_precision: float
    context_recall: float
    context_f1: float
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
        context_precision=ctx.precision,
        context_recall=ctx.recall,
        context_f1=ctx.f1,
        groundedness=groundedness(prediction, admitted_text),
    )


def aggregate(rows: Sequence[MetricRow]) -> dict[str, float]:
    """Mean of every metric across a set of rows (one system, typically)."""
    if not rows:
        return {
            "n": 0, "exact_match": 0.0, "token_f1": 0.0, "rouge_l_f1": 0.0,
            "context_precision": 0.0, "context_recall": 0.0, "context_f1": 0.0,
            "groundedness": 0.0,
        }
    n = len(rows)
    return {
        "n": n,
        "exact_match": sum(r.exact_match for r in rows) / n,
        "token_f1": sum(r.token_f1 for r in rows) / n,
        "rouge_l_f1": sum(r.rouge_l_f1 for r in rows) / n,
        "context_precision": sum(r.context_precision for r in rows) / n,
        "context_recall": sum(r.context_recall for r in rows) / n,
        "context_f1": sum(r.context_f1 for r in rows) / n,
        "groundedness": sum(r.groundedness for r in rows) / n,
    }


def aggregate_by_system(
    rows: Iterable[MetricRow],
) -> dict[str, dict[str, float]]:
    """Group rows by system and aggregate each group."""
    by_system: dict[str, list[MetricRow]] = {}
    for row in rows:
        by_system.setdefault(row.system, []).append(row)
    return {system: aggregate(group) for system, group in by_system.items()}
