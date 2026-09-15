"""Check A - temporal reach of the pairs against a label model's cutoff.

Understanding report section 20 and ledger risk K2: the thesis's mechanism
claim is that the baseline filter's confidence signal favours the evidence
state the label-generating model saw during pre-training. If every change
point in the evaluation set predates that model's cutoff, the model saw both
states, the mechanism cannot fire as stated, and a null result is
uninterpretable rather than informative.

This module answers one question over pairs that already exist: how many
change points fall after a given cutoff date. It deliberately does not read
any external dataset, because the shape of that dataset is not known here and
guessing its fields would produce a number with no evidential value.

There is no pass/fail threshold in the specification, so none is invented.
The decision rule stays where the understanding report put it: if the
post-cutoff stratum is too small to power H1, choose explicitly between an
earlier-cutoff label model, reframing the mechanism as frequency dominance,
or (last resort, requiring sign-off) breaching the provenance firewall.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

from .schema import TestPair, _parse_partial_date


@dataclass(frozen=True)
class CutoffStratum:
    model: str
    cutoff: str
    pairs_total: int
    pairs_dated: int
    post_cutoff: int
    pre_cutoff: int
    undated_change_point: int

    @property
    def post_cutoff_share(self) -> Optional[float]:
        if self.pairs_dated == 0:
            return None
        return self.post_cutoff / self.pairs_dated

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "cutoff": self.cutoff,
            "pairs_total": self.pairs_total,
            "pairs_dated": self.pairs_dated,
            "post_cutoff": self.post_cutoff,
            "pre_cutoff": self.pre_cutoff,
            "undated_change_point": self.undated_change_point,
            "post_cutoff_share": self.post_cutoff_share,
        }


def _change_point(pair: TestPair) -> Optional[date]:
    """The pair's change point, or None when the dataset supplied none.

    There is deliberately no fallback to the newer publication date. A
    publication date says when a paper appeared; a change point says when the
    clinical verdict moved. Substituting one for the other would turn a
    missing value into a stratum count that looks like a Check A result.
    Pairs without a change point are counted separately instead.
    """

    if pair.change_point_date:
        return _parse_partial_date(pair.change_point_date)[0]

    return None


def stratify(
    pairs: Sequence[TestPair],
    *,
    model: str,
    cutoff: str,
) -> CutoffStratum:
    """Count pairs whose change point falls after ``cutoff``."""

    boundary = _parse_partial_date(cutoff)[0]

    if boundary is None:
        raise ValueError(
            f"cutoff {cutoff!r} is not a parseable date (YYYY, YYYY-MM or "
            "YYYY-MM-DD)."
        )

    post = pre = undated = 0

    for pair in pairs:
        point = _change_point(pair)
        if point is None:
            undated += 1
        elif point > boundary:
            post += 1
        else:
            pre += 1

    return CutoffStratum(
        model=model,
        cutoff=cutoff,
        pairs_total=len(pairs),
        pairs_dated=post + pre,
        post_cutoff=post,
        pre_cutoff=pre,
        undated_change_point=undated,
    )


def evaluate(
    pairs: Sequence[TestPair],
    cutoffs: dict[str, str],
) -> list[dict[str, object]]:
    """Run :func:`stratify` for each configured label model."""

    return [
        stratify(pairs, model=model, cutoff=cutoff).to_dict()
        for model, cutoff in sorted(cutoffs.items())
    ]
