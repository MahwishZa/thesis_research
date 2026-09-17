"""Runs the arms over frozen items and records what happened.

The runner's job is not to be clever. It replays one frozen candidate set
through every arm, refuses to continue if the arms did not receive the same
set, and writes every raw answer to disk before anything is scored.

Raw model output is never overwritten: the output path must not already exist.
A re-run goes to a new file, so an earlier result can always be recovered.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from systems.interfaces.evidence import Candidate, Evidence

from .freezing import FreezeError, FrozenItem, assert_same_candidate_sets


class RunnerError(RuntimeError):
    """Raised when a run cannot proceed safely."""


def _parse_date(value: Optional[str]):
    if not value:
        return None
    parts = [int(p) for p in value.split("-")]
    while len(parts) < 3:
        parts.append(1)
    return _dt.date(*parts[:3])


def to_candidates(item: FrozenItem) -> tuple[Candidate, ...]:
    """Convert frozen records into the Candidate objects the arms consume.

    Candidate-list order is preserved exactly: it is part of the frozen set's
    identity and all three arms emit context in this order.
    """
    out = []
    for c in item.candidates:
        out.append(Candidate(
            evidence=Evidence(
                evidence_id=c.evidence_id,
                text=c.text,
                source_tier=str(c.source_metadata.get("source_tier", "unknown")),
                persistent_id=c.source_metadata.get("persistent_id"),
                publication_date=_parse_date(c.publication_date),
            ),
            rerank_score=float(c.rerank_score if c.rerank_score is not None else 0.0),
            rerank_rank=int(c.rerank_rank if c.rerank_rank is not None
                            else c.retrieval_rank),
            retrieval_score=c.retrieval_score,
            retrieval_rank=c.retrieval_rank,
        ))
    return tuple(out)


def assert_prompt_parity(systems: Mapping[str, Any]) -> None:
    """Every arm must be configured with the same context prompt template.

    Each arm accepts a ``context_prompt`` override and nothing else checks
    them against one another, so a parity break would otherwise be silent and
    would confound the comparison with a prompt difference.
    """
    templates = {}
    for name, system in systems.items():
        template = getattr(system, "context_prompt", None)
        if template is None:
            config = getattr(system, "config", None)
            template = getattr(config, "context_prompt", None)
        templates[name] = template
    distinct = {json.dumps(t, sort_keys=True) for t in templates.values()}
    if len(distinct) > 1:
        raise RunnerError(
            f"prompt templates differ between arms: {templates}. "
            "The comparison would confound the intervention with the prompt."
        )


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    model: str
    model_version: str
    generation_config: Mapping[str, Any]
    system_config_hash: str
    notes: Optional[str] = None


def run_experiment(
    items: Sequence[FrozenItem],
    systems: Mapping[str, Any],
    config: RunConfig,
    output_path: str,
) -> dict[str, Any]:
    """Run every arm over every item and write JSONL results.

    Returns a summary. Raises before writing if the arms are misconfigured,
    and raises after running if they did not see identical candidate sets.
    """
    if len(systems) < 2:
        raise RunnerError("at least two arms are needed for a comparison")
    if os.path.exists(output_path):
        raise RunnerError(
            f"refusing to overwrite existing model output: {output_path}"
        )

    assert_prompt_parity(systems)

    seen_hashes: dict[str, dict[str, str]] = {name: {} for name in systems}
    records: list[dict[str, Any]] = []

    for item in items:
        candidates = to_candidates(item)
        for name, system in systems.items():
            seen_hashes[name][item.question_id] = item.candidate_set_hash
            admitted_text: list[str] = []
            error = None
            try:
                result = system.run(
                    sample_id=item.question_id,
                    experiment_id=config.run_id,
                    question=item.question,
                    candidates=candidates,
                )
                by_id = {c.evidence_id: c.text for c in item.candidates}
                admitted_text = [by_id[e] for e in result.admitted_evidence_ids
                                 if e in by_id]
                status = "ok"
            except Exception as exc:  # recorded, never silently dropped
                result = None
                status = "error"
                error = f"{type(exc).__name__}: {exc}"

            records.append({
                "run_id": config.run_id,
                "question_id": item.question_id,
                "system": name,
                "question": item.question,
                "reference_answer": item.reference_answer,
                "candidate_evidence_ids": list(item.candidate_evidence_ids),
                "candidate_set_hash": item.candidate_set_hash,
                "admitted_evidence_ids": (
                    list(result.admitted_evidence_ids) if result else []
                ),
                "admitted_evidence_text": admitted_text,
                "generated_answer": result.prediction if result else None,
                "output_state": result.output_state if result else None,
                "model": config.model,
                "model_version": config.model_version,
                "generation_config": dict(config.generation_config),
                "system_config_hash": config.system_config_hash,
                "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "status": status,
                "error": error,
            })

    names = list(systems)
    for other in names[1:]:
        assert_same_candidate_sets(seen_hashes[names[0]], seen_hashes[other])

    with open(output_path, "x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True,
                                    ensure_ascii=False) + "\n")

    return {
        "run_id": config.run_id,
        "output_path": output_path,
        "n_items": len(items),
        "n_systems": len(systems),
        "n_records": len(records),
        "n_errors": sum(1 for r in records if r["status"] == "error"),
    }
