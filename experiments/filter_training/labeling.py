"""RAG²'s label function: does this passage help answer this question?

The filter checkpoint is not distributed, so the baseline has to be trained,
and training needs labels. This module implements the paper's labelling
procedure (ledger E1–E3, D2 §3.2, Fig. 2, Eq. 3) rather than inventing one,
because the label function *is* what makes the baseline RAG²'s baseline.

The decision tree, in the paper's order:

    1. Answer the question WITHOUT the passage      -> correct_without
    2. Answer the question WITH the passage         -> correct_with
    3. If correctness flipped, that decides it:
         wrong -> right   =>  [HELPFUL]
         right -> wrong   =>  [NOT_HELPFUL]
    4. If correctness is unchanged, fall back to the perplexity differential
       of the generated rationale: the passage is [HELPFUL] if it reduced
       rationale perplexity enough to land in the top tau fraction of
       differentials, else [NOT_HELPFUL].

Step 4 exists because step 3 is silent on most pairs - the paper introduces
perplexity explicitly "to address" that case (E2). Perplexity is computed over
the generated **rationale**, not the query (E3): the paper's Eq. 4 notation is
ambiguous and its prose is not.

**tau is a property of the label function, not a thesis parameter.** It is
fixed at the paper's value and is not fitted, tuned, or varied. Fitting it
would make the baseline something the thesis chose rather than something the
paper specifies.

Nothing here loads a model. Rationale generation and perplexity are supplied
through small interfaces, so the decision tree is testable exactly, and the
expensive part runs where the GPU is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

HELPFUL = "[HELPFUL]"
NOT_HELPFUL = "[NOT_HELPFUL]"

#: The paper's tie-break threshold: the top 25% of perplexity differentials
#: are labelled helpful (E1). Fixed by the reference method.
TAU = 0.25

#: RAG²'s training-example prompt, from the released data format.
FILTER_TRAINING_PROMPT = (
    "Given the following evidence, determine whether it helps answer the "
    "provided question.\n\nEvidence: {evidence}\n\nQuestion: {question}"
)


class LabelingError(RuntimeError):
    """Raised when labels cannot be produced faithfully."""


@dataclass(frozen=True)
class RationaleOutcome:
    """One rationale generation: was the answer right, and how perplexed."""

    correct: bool
    perplexity: float

    def __post_init__(self) -> None:
        if self.perplexity <= 0:
            raise LabelingError(
                f"perplexity must be positive, got {self.perplexity}")


class RationaleScorer(Protocol):
    """Generates a rationale and reports correctness plus its perplexity.

    Implemented on the GPU machine against the same generator the experiment
    uses. Kept as a protocol so the decision tree below is exercised exactly,
    with no model anywhere near the test suite.
    """

    def score(self, question: str, choices: Sequence[str], answer: str,
              evidence: Optional[str]) -> RationaleOutcome:
        ...


@dataclass(frozen=True)
class PairOutcome:
    """Both halves of one (question, passage) pair, before labelling."""

    pair_id: str
    question: str
    evidence: str
    without: RationaleOutcome
    with_evidence: RationaleOutcome

    @property
    def flipped_to_correct(self) -> bool:
        return self.with_evidence.correct and not self.without.correct

    @property
    def flipped_to_wrong(self) -> bool:
        return self.without.correct and not self.with_evidence.correct

    @property
    def correctness_unchanged(self) -> bool:
        return self.without.correct == self.with_evidence.correct

    @property
    def perplexity_reduction(self) -> float:
        """How much the passage reduced rationale perplexity.

        Positive means the passage made the rationale less perplexing, which
        is the direction the paper treats as evidence of helpfulness.
        """
        return self.without.perplexity - self.with_evidence.perplexity


@dataclass(frozen=True)
class LabelledExample:
    """One training example in the released RAG² format."""

    id: str
    answer: str
    dataset_name: str
    question: str
    #: Why this label, kept for auditability. Not part of the training file.
    rule: str = ""

    def to_training_record(self) -> dict[str, Any]:
        """Exactly the four fields the released training data carries."""
        return {
            "id": self.id,
            "answer": self.answer,
            "dataset_name": self.dataset_name,
            "question": self.question,
        }


def perplexity_threshold(
    outcomes: Sequence[PairOutcome],
    tau: float = TAU,
) -> Optional[float]:
    """The cutoff for the top ``tau`` fraction of perplexity reductions.

    Computed over the pairs the tie-break actually applies to - those whose
    correctness did not change - because a threshold taken over all pairs
    would be set partly by pairs it never judges.

    Returns ``None`` when no pair needs the tie-break.
    """
    if not 0 < tau < 1:
        raise LabelingError(f"tau must be in (0, 1), got {tau}")
    reductions = sorted(
        (o.perplexity_reduction for o in outcomes if o.correctness_unchanged),
        reverse=True,
    )
    if not reductions:
        return None
    index = max(1, round(tau * len(reductions))) - 1
    return reductions[index]


def label_pair(
    outcome: PairOutcome,
    threshold: Optional[float],
) -> tuple[str, str]:
    """Apply the decision tree to one pair. Returns ``(label, rule)``."""
    if outcome.flipped_to_correct:
        return HELPFUL, "correctness_flip_to_correct"
    if outcome.flipped_to_wrong:
        return NOT_HELPFUL, "correctness_flip_to_wrong"
    if threshold is None:
        raise LabelingError(
            f"{outcome.pair_id}: correctness did not change, so the "
            "perplexity tie-break decides, but no threshold was computed"
        )
    if outcome.perplexity_reduction >= threshold:
        return HELPFUL, "perplexity_differential_top_tau"
    return NOT_HELPFUL, "perplexity_differential_below_tau"


def label_dataset(
    outcomes: Sequence[PairOutcome],
    *,
    dataset_name: str,
    tau: float = TAU,
) -> tuple[LabelledExample, ...]:
    """Label a whole set of pairs.

    The threshold is computed over the set, so labelling is a batch operation:
    the paper's tie-break is a quantile, and a quantile of one pair is not the
    same rule.
    """
    if not outcomes:
        raise LabelingError("cannot label an empty set of pairs")
    threshold = perplexity_threshold(outcomes, tau)
    examples = []
    for outcome in outcomes:
        label, rule = label_pair(outcome, threshold)
        examples.append(LabelledExample(
            id=f"{dataset_name}_{outcome.pair_id}",
            answer=label,
            dataset_name=dataset_name,
            question=FILTER_TRAINING_PROMPT.format(
                evidence=outcome.evidence, question=outcome.question),
            rule=rule,
        ))
    return tuple(examples)


def write_training_file(
    examples: Sequence[LabelledExample],
    path: str | Path,
) -> Path:
    """Write the JSON list the RAG² classifier reads.

    Refuses to overwrite: labels cost GPU hours to produce and a rerun that
    silently replaced them would destroy the provenance of a trained
    checkpoint.
    """
    path = Path(path)
    if path.exists():
        raise LabelingError(f"refusing to overwrite existing labels: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([e.to_training_record() for e in examples], indent=2),
        encoding="utf-8",
    )
    return path


def label_distribution(
    examples: Sequence[LabelledExample],
) -> dict[str, int]:
    """Counts per label and per rule, for the training report.

    A filter trained on a set that is 95% one label learns the prior rather
    than the task, so this is checked before training rather than diagnosed
    afterwards from a bad validation number.
    """
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.answer] = counts.get(example.answer, 0) + 1
        if example.rule:
            counts[f"rule:{example.rule}"] = counts.get(
                f"rule:{example.rule}", 0) + 1
    return counts
