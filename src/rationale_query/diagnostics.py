"""What the rationales look like: length, truncation, final option, and agreement between writers.

These numbers describe a set of rationales. They are not tests of correctness, and the final-option reading is a
diagnostic with known limits (it looks for phrases such as "the answer is B").
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence

from .store import RationaleRecord

# "answer is B)", "Answer: C", "final answer: **D**", "option A." The letter must be capital and be followed by a
# closing bracket, a full stop, a colon, a comma, an asterisk or the end of the line, so "answer is A cholinesterase
# inhibitor" does not count.
_OPTION = re.compile(
    r"(?i:\b(?:answer|option|choice)\b)(?i:\s+(?:is|would\s+be|should\s+be))?[^A-Za-z0-9\n]{0,8}\(?([A-E])\)?(?=[\).:,;*]|[ \t]*$)",
    re.MULTILINE,
)


def final_option(text: str) -> str | None:
    """The option letter named last as the answer, or None if no such phrase is found."""
    found = None
    for match in _OPTION.finditer(text):
        found = match.group(1)
    return found


def _quantile(values: Sequence[float], share: float) -> float:
    """The smallest value that at least ``share`` of the values do not exceed (the nearest-rank method)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


def _spread(values: Sequence[float]) -> dict | None:
    if not values:
        return None
    return {
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(_quantile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def summarise(
    records: Sequence[RationaleRecord],
    token_counts: Mapping[str, int] | None = None,
    encoder_limit: int = 512,
) -> dict:
    """Counts and spreads for one output folder. ``token_counts`` (id to encoder tokens) adds the truncation share."""
    ok = [r for r in records if r.status == "ok"]
    errors = [r for r in records if r.status == "error"]
    parsed = [final_option(r.rationale or "") for r in ok]
    summary = {
        "records": len(records),
        "ok": len(ok),
        "errors": len(errors),
        "error_kinds": dict(Counter((r.error or "").split(":")[0] for r in errors)),
        "completion_tokens": _spread([r.completion_tokens for r in ok if r.completion_tokens is not None]),
        "hit_generation_limit": sum(1 for r in ok if r.finish_reason == "length"),
        "seconds_per_question": _spread([r.seconds for r in ok if r.seconds is not None]),
        "final_option_found": sum(1 for p in parsed if p is not None),
        "final_option_counts": dict(sorted(Counter(p for p in parsed if p is not None).items())),
    }
    if token_counts is not None:
        counts = [token_counts[r.id] for r in ok if r.id in token_counts]
        summary["encoder_tokens"] = _spread(counts)
        summary["encoder_truncated"] = sum(1 for c in counts if c > encoder_limit)
    return summary


def agreement(first: Sequence[RationaleRecord], second: Sequence[RationaleRecord]) -> dict:
    """How often two writers name the same final option on the questions both answered with a readable option."""
    left = {r.id: final_option(r.rationale or "") for r in first if r.status == "ok"}
    right = {r.id: final_option(r.rationale or "") for r in second if r.status == "ok"}
    shared = [i for i in left if i in right and left[i] is not None and right[i] is not None]
    same = sum(1 for i in shared if left[i] == right[i])
    return {"compared": len(shared), "same_option": same, "share": round(same / len(shared), 3) if shared else None}


def provenance(label: str, config: Mapping[str, object]) -> str:
    """One line that says how the rationales of a folder were made, read from its config.json."""
    sha = str(config.get("model_sha256") or "")[:12]
    return (
        f"{label}: {config.get('model_file')} (SHA-256 {sha}...), {config.get('chat_format')} format "
        f"[template: {config.get('template_check')}], at most {config.get('max_new_tokens')} new tokens, "
        f"context {config.get('n_ctx')}, llama-cpp-python {config.get('llama_cpp_python')}"
    )


def format_table(named: Mapping[str, dict]) -> str:
    """A plain-text comparison, one column per output folder."""
    names = list(named)
    rows = [
        ("records", lambda s: s["records"]),
        ("with a rationale", lambda s: s["ok"]),
        ("errors", lambda s: s["errors"]),
        ("new tokens, median", lambda s: (s["completion_tokens"] or {}).get("median", "-")),
        ("new tokens, 95th percentile", lambda s: (s["completion_tokens"] or {}).get("p95", "-")),
        ("cut off at the limit", lambda s: s["hit_generation_limit"]),
        ("seconds per question, median", lambda s: (s["seconds_per_question"] or {}).get("median", "-")),
        ("final option readable", lambda s: s["final_option_found"]),
        ("longer than the encoder limit", lambda s: s.get("encoder_truncated", "-")),
    ]
    width = max(len(label) for label, _ in rows)
    column = max([len(n) for n in names] + [8]) + 2
    lines = [" " * width + "".join(n.rjust(column) for n in names)]
    for label, pick in rows:
        lines.append(label.ljust(width) + "".join(str(pick(named[n])).rjust(column) for n in names))
    return "\n".join(lines)
