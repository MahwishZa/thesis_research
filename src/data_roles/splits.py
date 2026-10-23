"""One role per question, from the id and a seed, and a check that the roles share no question.

The role of a question is a function of its id and the seed only (docs/methodology.md, 7.7, rule 1). It does not depend
on the order of the rows, on the Python process (the built-in ``hash`` of a string changes from one run to the next, so
it is not used: SHA-256 is), or on which other questions are present, so a question never moves when others are added.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping

_TOLERANCE = 1e-9


def _unit_interval(question_id: str, seed: int) -> float:
    """A number in [0, 1) that depends on the id and the seed only (the first 8 bytes of a SHA-256, read as a fraction)."""
    digest = hashlib.sha256(f"{seed}\x00{question_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _check_fractions(fractions: Mapping[str, float]) -> list[tuple[str, float]]:
    if not isinstance(fractions, Mapping) or not fractions:
        raise ValueError("fractions must be a non-empty mapping from a role name to its share")
    for role, share in fractions.items():
        if not isinstance(role, str) or not role:
            raise ValueError(f"a role name must be a non-empty string, got {role!r}")
        if isinstance(share, bool) or not isinstance(share, int | float) or not math.isfinite(share) or share < 0:
            raise ValueError(f"the share of {role!r} must be a number from 0 up, got {share!r}")
    total = math.fsum(fractions.values())
    if abs(total - 1.0) > _TOLERANCE:
        raise ValueError(f"the shares must add up to 1, they add up to {total!r}")
    # Names are sorted so that the role of a question does not depend on the order in which the caller listed the roles.
    return sorted(fractions.items())


def split_by_question(*, question_ids: Iterable[str], fractions: Mapping[str, float], seed: int) -> dict[str, str]:
    """The role of every question id. Repeated ids are one question; no ids give an empty split.

    ``fractions`` maps a role name to its share; the shares add up to 1. The shares are met on average: with n questions
    the count of a role differs from ``share * n`` by about the square root of n, as with any hash split.
    """
    ordered = _check_fractions(fractions)
    edges: list[tuple[float, str]] = []
    running = 0.0
    for role, share in ordered:
        running += share
        edges.append((running, role))
    roles: dict[str, str] = {}
    for question_id in question_ids:
        if not isinstance(question_id, str):
            raise TypeError(f"a question id must be a string, got {type(question_id).__name__}")
        if question_id in roles:
            continue
        point = _unit_interval(question_id, seed)
        for edge, role in edges:
            if point < edge:
                roles[question_id] = role
                break
        else:  # the last edge is 1 up to rounding; a point beyond it belongs to the last role
            roles[question_id] = edges[-1][1]
    return roles


def assert_disjoint(roles: Mapping[str, Iterable[str]]) -> None:
    """Raise ``ValueError`` naming the ids that two roles share. A repeated id inside one role is fine."""
    first_role: dict[str, str] = {}
    shared: dict[str, set[str]] = {}
    for role, ids in roles.items():
        for question_id in set(ids):
            if question_id in first_role and first_role[question_id] != role:
                shared.setdefault(question_id, {first_role[question_id]}).add(role)
            else:
                first_role.setdefault(question_id, role)
    if shared:
        shown = sorted(shared)[:10]
        more = f" ({len(shared)} in all)" if len(shared) > len(shown) else ""
        detail = "; ".join(f"{question_id} in {', '.join(sorted(shared[question_id]))}" for question_id in shown)
        raise ValueError(f"these questions are in more than one role: {detail}{more}")
