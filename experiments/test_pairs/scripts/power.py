"""Sample size for the paired bias probe (specification 43).

The primary outcome is paired and categorical: for each pair, was the older
passage admitted and the newer one not, or the reverse? Concordant pairs
(both admitted, or neither) carry no information about asymmetry, so the
analysis reduces to McNemar's test over the discordant pairs, and the sample
size follows from two quantities the pilot must supply:

* ``discordant_rate`` - the share of pairs on which the two passages receive
  different admission decisions;
* ``older_share`` - among those, the share favouring the older passage
  (0.5 is exactly no bias).

Both are estimates, and nothing here invents them: :func:`required_pairs`
raises unless they are measured. A normal approximation is used rather than
an exact binomial design; at the accuracies a pilot of this size can supply,
the exact method's extra precision is not meaningful, and the approximation
is transparent enough to check by hand.

The result is a *floor*, not a target: it counts usable pairs after
attrition, so the number of candidates to construct is this figure divided by
the pilot's yield rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


#: Two-sided normal quantiles. Only the conventional levels are provided;
#: an unusual alpha or power should be justified, not silently supported.
_Z_ALPHA = {0.05: 1.959963985, 0.01: 2.575829304}
_Z_BETA = {0.80: 0.841621234, 0.90: 1.281551566, 0.95: 1.644853627}


@dataclass(frozen=True)
class PowerResult:
    discordant_rate: float
    older_share: float
    alpha: float
    power: float
    discordant_pairs_needed: int
    total_pairs_needed: int
    candidates_needed: Optional[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "discordant_rate": self.discordant_rate,
            "older_share": self.older_share,
            "alpha": self.alpha,
            "power": self.power,
            "discordant_pairs_needed": self.discordant_pairs_needed,
            "total_pairs_needed": self.total_pairs_needed,
            "candidates_needed": self.candidates_needed,
        }


def required_pairs(
    *,
    discordant_rate: Optional[float],
    older_share: Optional[float],
    alpha: float = 0.05,
    power: float = 0.80,
    yield_rate: Optional[float] = None,
) -> PowerResult:
    """Usable pairs needed to detect the given admission asymmetry.

    Args:
        discordant_rate: measured share of pairs with differing admission
            decisions. Required - it cannot be assumed.
        older_share: measured share of discordant pairs favouring the older
            passage. Required. 0.5 means no asymmetry and has no finite
            sample size.
        yield_rate: pilot yield (eligible / candidates). When given, the
            result also reports how many candidates must be constructed.

    Raises:
        ValueError: if either estimate is missing or degenerate.
    """

    if discordant_rate is None or older_share is None:
        raise ValueError(
            "discordant_rate and older_share must be measured on the pilot. "
            "They are not given defaults: a sample size computed from an "
            "assumed effect size is not a power analysis."
        )

    if not 0.0 < discordant_rate <= 1.0:
        raise ValueError("discordant_rate must be in (0, 1].")

    if not 0.0 <= older_share <= 1.0:
        raise ValueError("older_share must be in [0, 1].")

    if abs(older_share - 0.5) < 1e-9:
        raise ValueError(
            "older_share of 0.5 is exactly the null hypothesis; no finite "
            "sample size can detect it. Report the pilot estimate and its "
            "confidence interval instead."
        )

    if alpha not in _Z_ALPHA:
        raise ValueError(f"alpha must be one of {sorted(_Z_ALPHA)}.")

    if power not in _Z_BETA:
        raise ValueError(f"power must be one of {sorted(_Z_BETA)}.")

    z_alpha = _Z_ALPHA[alpha]
    z_beta = _Z_BETA[power]

    # Normal approximation to the binomial test that the discordant pairs
    # split evenly: n_discordant = ((z_a/2 + z_b*sqrt(p(1-p))) / (p - 0.5))^2
    p = older_share
    numerator = z_alpha * 0.5 + z_beta * math.sqrt(p * (1.0 - p))
    discordant_needed = math.ceil((numerator / (p - 0.5)) ** 2)

    total_needed = math.ceil(discordant_needed / discordant_rate)

    candidates = None
    if yield_rate is not None:
        if not 0.0 < yield_rate <= 1.0:
            raise ValueError("yield_rate must be in (0, 1].")
        candidates = math.ceil(total_needed / yield_rate)

    return PowerResult(
        discordant_rate=discordant_rate,
        older_share=older_share,
        alpha=alpha,
        power=power,
        discordant_pairs_needed=discordant_needed,
        total_pairs_needed=total_needed,
        candidates_needed=candidates,
    )


def detectable_effect(
    *,
    total_pairs: int,
    discordant_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Optional[float]:
    """Smallest older-favouring share detectable with ``total_pairs``.

    The inverse of :func:`required_pairs`, for the situation this thesis is
    actually in: the pool is fixed by the external dataset rather than chosen,
    so the honest question is not "how many pairs do we need?" but "what could
    this many pairs detect?". Returns None when no share in (0.5, 1] suffices.

    Solved by bisection on the monotone relationship between the share and
    the discordant pairs required; a closed form exists but is less legible
    and no more accurate here.
    """

    if total_pairs <= 0:
        raise ValueError("total_pairs must be positive.")
    if not 0.0 < discordant_rate <= 1.0:
        raise ValueError("discordant_rate must be in (0, 1].")

    available = total_pairs * discordant_rate

    z_alpha = _Z_ALPHA[alpha]
    z_beta = _Z_BETA[power]

    def needed(share: float) -> float:
        numerator = z_alpha * 0.5 + z_beta * math.sqrt(share * (1.0 - share))
        return (numerator / (share - 0.5)) ** 2

    if needed(1.0 - 1e-9) > available:
        return None

    low, high = 0.5 + 1e-9, 1.0 - 1e-9
    for _ in range(200):
        mid = (low + high) / 2.0
        if needed(mid) > available:
            low = mid
        else:
            high = mid

    # Rounded up, so the value returned is genuinely detectable with the
    # given pool rather than a shade under it.
    return math.ceil(high * 10000.0) / 10000.0


def sensitivity_curve(
    *,
    total_pairs: int,
    discordant_rates: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    alpha: float = 0.05,
    power: float = 0.80,
) -> list[dict[str, object]]:
    """What a fixed pool could detect across plausible discordant rates.

    Reported as a curve rather than a point because the discordant rate is a
    Stage-3 measurement. Assuming one value and quoting a single sample size
    would be the fabricated effect size this module exists to refuse.
    """

    return [
        {
            "discordant_rate": rate,
            "expected_discordant_pairs": round(total_pairs * rate, 1),
            "min_detectable_older_share": detectable_effect(
                total_pairs=total_pairs,
                discordant_rate=rate,
                alpha=alpha,
                power=power,
            ),
        }
        for rate in discordant_rates
    ]
