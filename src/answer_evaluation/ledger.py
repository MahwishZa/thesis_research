"""A record of which test sets were scored, and a refusal to score one twice (docs/methodology.md, 7.7, rule 6; decision D24).

A test set is used once per system, after every setting is frozen. The ledger is an append-only file of events. ``claim`` is
called before the first answer of a test run is written: it refuses when the same system has already been scored on the same
dataset and split, unless a reason is given, and the reason is kept in the file. A run that was stopped may continue with the
same settings; the same claim then goes through without a new entry. ``finish`` marks the run as complete. Nothing is deleted.

This does not stop a person who deletes the file; it makes a second look at a test set a deliberate act with a written reason.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AlreadyScored(RuntimeError):
    """The test set was already scored by this system, or a started run used other settings."""


class LedgerError(RuntimeError):
    """The ledger file cannot be read."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: str | Path, clock: Callable[[], str] = _now) -> None:
        self.path = Path(path)
        self.clock = clock

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").split("\n"), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"{self.path}, line {number} is damaged: {exc}") from exc
            events.append(event)
        return events

    def _append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "ab") as handle:
            handle.write((json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())

    def _runs(self, dataset: str, split: str, system: str) -> list[dict[str, Any]]:
        """The claims of this system on this test set, each with a ``finished`` flag, oldest first."""
        runs: list[dict[str, Any]] = []
        for event in self.events():
            if (event.get("dataset"), event.get("split"), event.get("system")) != (dataset, split, system):
                continue
            if event["event"] == "claim":
                runs.append({**event, "finished": False})
            elif event["event"] == "finish" and runs:
                runs[-1]["finished"] = True
        return runs

    def claim(self, *, dataset: str, split: str, system: str, settings_sha256: str, rerun_reason: str | None = None) -> dict[str, Any]:
        """Permission to score ``system`` on ``dataset`` ``split``; raises ``AlreadyScored`` when that would be a second scoring."""
        for name, value in (("dataset", dataset), ("split", split), ("system", system), ("settings_sha256", settings_sha256)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty text")
        runs = self._runs(dataset, split, system)
        if runs and rerun_reason is None:
            last = runs[-1]
            if not last["finished"] and last["settings_sha256"] == settings_sha256:
                return last  # the same run, continued after a stop
            what = "already scored" if last["finished"] else f"started with other settings (claimed {last['claimed_at']})"
            raise AlreadyScored(
                f"{system} was {what} on {dataset} {split}. A test set is scored once per system. "
                "Give a rerun_reason to score it again on purpose; both results are then reported."
            )
        if rerun_reason is not None and not rerun_reason.strip():
            raise ValueError("a rerun_reason must say why")
        event = {
            "event": "claim", "dataset": dataset, "split": split, "system": system, "settings_sha256": settings_sha256,
            "claimed_at": self.clock(), "rerun_reason": rerun_reason.strip() if rerun_reason else None, "run": len(runs) + 1,
        }
        self._append(event)
        return {**event, "finished": False}

    def finish(self, *, dataset: str, split: str, system: str) -> None:
        runs = self._runs(dataset, split, system)
        if not runs:
            raise LedgerError(f"{system} has no claim on {dataset} {split} to finish")
        if runs[-1]["finished"]:
            return
        self._append({"event": "finish", "dataset": dataset, "split": split, "system": system, "finished_at": self.clock()})
