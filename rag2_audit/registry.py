"""Check registry and result types."""

from __future__ import annotations

import enum
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class Status(str, enum.Enum):
    PASS = "PASS"                    # matches the original work
    FAIL = "FAIL"                    # contradicts the original work
    PARTIAL = "PARTIAL"              # broadly matches, differs in an important detail
    UNKNOWN = "UNKNOWN"              # cannot be verified from available information
    APPROXIMATION = "APPROXIMATION"  # original resource unavailable, alternative used
    MANUAL = "MANUAL"                # needs a human or a GPU run to settle

    @property
    def is_blocking(self) -> bool:
        return self is Status.FAIL


@dataclass
class Result:
    check_id: str
    component: str
    status: Status
    summary: str
    paper_says: str = ""
    code_does: str = ""
    why_it_matters: str = ""
    how_to_fix: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "component": self.component,
            "status": self.status.value,
            "summary": self.summary,
            "paper_says": self.paper_says,
            "code_does": self.code_does,
            "why_it_matters": self.why_it_matters,
            "how_to_fix": self.how_to_fix,
            "evidence": self.evidence,
        }


@dataclass
class Check:
    check_id: str
    component: str
    title: str
    fn: Callable[[], Result]


CHECKS: List[Check] = []


def check(check_id: str, component: str, title: str):
    """Register an audit check. The function returns a :class:`Result`."""

    def decorator(fn: Callable[[], Result]):
        if any(c.check_id == check_id for c in CHECKS):
            raise ValueError(f"duplicate check id {check_id!r}")
        CHECKS.append(Check(check_id=check_id, component=component, title=title, fn=fn))
        return fn

    return decorator


def run_all(only: Optional[str] = None) -> List[Result]:
    """Execute every registered check. A raising check becomes an UNKNOWN."""
    results: List[Result] = []
    for entry in CHECKS:
        if only and not (entry.check_id.startswith(only) or only.lower() in entry.component.lower()):
            continue
        try:
            results.append(entry.fn())
        except Exception as error:  # a broken check must not hide the rest
            results.append(
                Result(
                    check_id=entry.check_id,
                    component=entry.component,
                    status=Status.UNKNOWN,
                    summary=f"check raised {type(error).__name__}: {error}",
                    evidence={"traceback": traceback.format_exc(limit=5)},
                )
            )
    return results
