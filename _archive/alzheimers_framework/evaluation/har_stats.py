"""Hallucinated-answer-rate accounting of the original (human-annotation) design. Archived.

Split from ``evaluation/stats.py`` on 2026-10-05: the realigned study measures verdict accuracy with the paired
statistics that stay in ``evaluation/stats.py`` (``mcnemar``, ``paired_bootstrap_ci``, ``holm``). The functions
here need per-answer hallucination and abstention labels, which a human annotator would have supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from evaluation.stats import StatsError, _paired, mcnemar, paired_bootstrap_ci

from .annotation import Annotation


def har(
    hallucinated: Mapping[str, int],
    abstained: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """Hallucinated Answer Rate, reported both ways.

    An abstention contains no claims and so can never be labelled
    hallucinated. Reporting only the conditional rate therefore rewards a
    system for declining to answer. Both denominators are returned, always
    with the abstention rate, so a rate can never be read without it.
    """
    n_all = len(hallucinated)
    if n_all == 0:
        raise StatsError("no answers to score")
    abstained = abstained or {k: 0 for k in hallucinated}

    answered = [k for k in hallucinated if not abstained.get(k, 0)]
    n_answered = len(answered)
    total_h = sum(hallucinated.values())

    return {
        "n_answers": n_all,
        "n_answered": n_answered,
        "n_abstained": n_all - n_answered,
        "abstention_rate": round((n_all - n_answered) / n_all, 6),
        "n_hallucinated": total_h,
        # Denominator = answers actually produced.
        "har_conditional": (
            None if n_answered == 0
            else round(sum(hallucinated[k] for k in answered) / n_answered, 6)
        ),
        # Denominator = every item; abstentions count as non-hallucinated.
        "har_all_items": round(total_h / n_all, 6),
    }


def hallucination_outcomes(
    by_question: Mapping[str, Annotation],
) -> tuple[dict[str, int], dict[str, int]]:
    """Extract ``(hallucinated, abstained)`` mappings from one system's
    annotations, ready for ``har``, ``coverage``, ``mcnemar`` or
    ``compare_systems``.

    Takes the ``{question_id: Annotation}`` shape ``annotation.
    unblind_annotations`` produces for one system. This is the glue between
    Step 7 (annotation) and Step 8 (rate calculation): without it, a caller
    would hand-write the same dict comprehension at every call site.
    """
    hallucinated = {qid: a.hallucinated for qid, a in by_question.items()}
    abstained = {qid: a.abstained for qid, a in by_question.items()}
    return hallucinated, abstained


def qa_accuracy(correct: Mapping[str, int]) -> dict[str, Any]:
    """QA accuracy: correct answers over total evaluated.

    ``correct`` is a per-question 0/1 judgement (see
    ``accuracy.QAJudgment.correct``) supplied by the thesis's QA protocol,
    not computed here - this function only aggregates. Structurally this is
    the same paired-binary shape as ``har``, so ``mcnemar`` and
    ``paired_bootstrap_ci`` already support the baseline-vs-proposed
    comparison without a separate accuracy-specific test.
    """
    n = len(correct)
    if n == 0:
        raise StatsError("no answers to score")
    n_correct = sum(correct.values())
    return {
        "n_evaluated": n,
        "n_correct": n_correct,
        "n_incorrect": n - n_correct,
        "accuracy": round(n_correct / n, 6),
    }


def subtype_distribution(
    subtypes: Sequence[Optional[str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in subtypes:
        if s:
            counts[s] = counts.get(s, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def outcome_crosstab(
    baseline: Mapping[str, int],
    proposed: Mapping[str, int],
) -> dict[str, list[str]]:
    """Which questions fall in each cell, for error analysis.

    Returns question ids rather than counts so the write-up can pull
    representative examples - including the cases where the proposed system
    did worse, which are the ones most easily left out.
    """
    keys = _paired(baseline, proposed)
    return {
        "baseline_only": [k for k in keys
                          if baseline[k] == 1 and proposed[k] == 0],
        "proposed_only": [k for k in keys
                          if baseline[k] == 0 and proposed[k] == 1],
        "both": [k for k in keys if baseline[k] == 1 and proposed[k] == 1],
        "neither": [k for k in keys if baseline[k] == 0 and proposed[k] == 0],
    }


def error_analysis(
    crosstab: Mapping[str, Sequence[str]],
    subtypes: Mapping[str, Optional[str]],
) -> dict[str, dict[str, int]]:
    """Subtype breakdown within each ``outcome_crosstab`` cell.

    ``subtypes`` maps question_id to the diagnostic subtype recorded for that
    answer (``annotation.SUBTYPES``: faithfulness, factuality, temporal,
    misinterpretation, ambiguity, other), typically taken from whichever
    system's hallucinations are being explained in that cell. A question
    absent from ``subtypes`` - a non-hallucinated answer has no subtype - is
    simply not counted, which is correct: only hallucinated answers carry a
    subtype at all.

    This only counts what annotation already recorded; it draws no
    conclusions and manufactures no patterns.
    """
    return {
        cell: subtype_distribution([subtypes.get(qid) for qid in question_ids])
        for cell, question_ids in crosstab.items()
    }


@dataclass(frozen=True)
class Coverage:
    """Answer accounting for one system, kept separate from any rate.

    HAR alone is not interpretable: a system that answers nothing has no
    hallucinated answers. Every HAR must be read next to these counts, which
    is why they travel together rather than being computed on demand.
    """

    total_questions: int
    answered: int
    abstained: int
    hallucinated: int
    non_hallucinated: int

    @property
    def answer_coverage(self) -> float:
        return self.answered / self.total_questions if self.total_questions else 0.0

    @property
    def is_degenerate(self) -> bool:
        """True when the system answered nothing, so no rate is meaningful."""
        return self.answered == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "answered": self.answered,
            "abstained": self.abstained,
            "hallucinated": self.hallucinated,
            "non_hallucinated": self.non_hallucinated,
            "answer_coverage": round(self.answer_coverage, 6),
            "is_degenerate": self.is_degenerate,
        }


def coverage(
    hallucinated: Mapping[str, int],
    abstained: Mapping[str, int],
) -> Coverage:
    """Count answered, abstained and hallucinated outcomes for one system."""
    if set(hallucinated) != set(abstained):
        raise StatsError(
            "hallucination and abstention records cover different questions"
        )
    total = len(hallucinated)
    answered_keys = [k for k in hallucinated if not abstained[k]]
    for k in hallucinated:
        if abstained[k] and hallucinated[k]:
            raise StatsError(
                f"{k}: an abstention cannot also be hallucinated"
            )
    h = sum(hallucinated[k] for k in answered_keys)
    return Coverage(
        total_questions=total,
        answered=len(answered_keys),
        abstained=total - len(answered_keys),
        hallucinated=h,
        non_hallucinated=len(answered_keys) - h,
    )


def compare_systems(
    baseline_hallucinated: Mapping[str, int],
    baseline_abstained: Mapping[str, int],
    proposed_hallucinated: Mapping[str, int],
    proposed_abstained: Mapping[str, int],
    *,
    seed: str = "har",
    iterations: int = 10000,
) -> dict[str, Any]:
    """The primary comparison, with the accounting that makes it readable.

    Refuses to emit a headline difference when either system answered
    nothing, because a rate over zero answers is not a rate. The refusal is a
    recorded field rather than an exception: a degenerate run is a result the
    thesis should report, not a crash.
    """
    base_cov = coverage(baseline_hallucinated, baseline_abstained)
    prop_cov = coverage(proposed_hallucinated, proposed_abstained)

    base_har = har(baseline_hallucinated, baseline_abstained)
    prop_har = har(proposed_hallucinated, proposed_abstained)

    degenerate = base_cov.is_degenerate or prop_cov.is_degenerate

    report: dict[str, Any] = {
        "baseline": {"coverage": base_cov.to_dict(), "har": base_har},
        "proposed": {"coverage": prop_cov.to_dict(), "har": prop_har},
        "coverage_difference": round(
            prop_cov.answer_coverage - base_cov.answer_coverage, 6
        ),
        "interpretable": not degenerate,
    }

    if degenerate:
        report["warning"] = (
            "at least one system answered no questions; a hallucination rate "
            "over zero answers is undefined and this comparison must not be "
            "reported as hallucination reduction"
        )
        return report

    # Abstentions count as non-hallucinated here, which is the denominator a
    # coverage-losing system cannot game: it is reported beside the coverage.
    report["mcnemar_all_items"] = mcnemar(
        baseline_hallucinated, proposed_hallucinated
    ).to_dict()
    report["delta_har_all_items"] = paired_bootstrap_ci(
        {k: float(v) for k, v in baseline_hallucinated.items()},
        {k: float(v) for k, v in proposed_hallucinated.items()},
        seed=seed, iterations=iterations,
    )
    return report
