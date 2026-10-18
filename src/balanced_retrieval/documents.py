"""A resumable download of documents into one JSON-lines file, in batches.

Beside ``out`` the run keeps ``out.ids.txt`` (every identifier to fetch, found once) and ``out.state.json`` (the settings, how many
identifiers are done and how many bytes of ``out`` they filled). A run that stops anywhere continues from the last finished batch;
bytes written after it are cut off first. Run again with other settings, the files refuse to continue rather than mix two downloads.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rationale_query.store import ConfigMismatch, StoreError, write_atomic


def _state_path(out: Path) -> Path:
    return out.with_name(out.name + ".state.json")


def _ids_path(out: Path) -> Path:
    return out.with_name(out.name + ".ids.txt")


def read_state(out: Path) -> dict[str, Any] | None:
    try:
        return json.loads(_state_path(out).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise StoreError(f"cannot read {_state_path(out)}: {exc}") from exc


def download_in_batches(
    out: Path,
    settings: dict[str, Any],
    find_ids: Callable[[], list[str]],
    fetch_batch: Callable[[list[str]], list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, int]]],
    *,
    batch_size: int,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Fetch the documents of every identifier (or of the first ``limit``) and append them to ``out``; returns the state.

    ``find_ids`` is called once per output file; ``fetch_batch`` turns a list of identifiers into the documents to keep (it may
    return fewer), optionally with a dict of counts of why others were left out, which the state adds up under ``notes``. Each
    document is a dict with at least ``doc_key`` and ``text``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = read_state(out)
    if state is not None and state["settings"] != settings:
        raise ConfigMismatch(f"{out} was started with other settings ({state['settings']}); choose a new --out or delete it")
    ids_path = _ids_path(out)
    if state is None:
        if out.exists():
            raise StoreError(f"{out} exists but has no state file; refusing to append to it")
        ids = find_ids()
        write_atomic(ids_path, "".join(i + "\n" for i in ids))
        state = {"settings": settings, "ids": len(ids), "done": 0, "bytes": 0, "documents": 0, "notes": {}}
        write_atomic(_state_path(out), json.dumps(state, indent=2) + "\n")
        out.write_bytes(b"")
    else:
        ids = ids_path.read_text(encoding="utf-8").split()
        if len(ids) != state["ids"]:
            raise StoreError(f"{ids_path} holds {len(ids)} identifiers but the state says {state['ids']}")
        if not out.exists() or out.stat().st_size < state["bytes"]:
            raise StoreError(f"{out} is shorter than the state file says; the download cannot continue")

    target = len(ids) if limit is None else min(limit, len(ids))
    with open(out, "r+b") as handle:
        handle.truncate(state["bytes"])
        handle.seek(state["bytes"])
        while state["done"] < target:
            batch = ids[state["done"] : min(state["done"] + batch_size, target)]
            fetched = fetch_batch(batch)
            documents, notes = fetched if isinstance(fetched, tuple) else (fetched, {})
            totals = dict(state.get("notes", {}))
            for reason, number in notes.items():
                totals[reason] = totals.get(reason, 0) + number
            handle.writelines((json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for document in documents)
            handle.flush()
            os.fsync(handle.fileno())
            state = {**state, "done": state["done"] + len(batch), "bytes": handle.tell(), "documents": state["documents"] + len(documents), "notes": totals}
            write_atomic(_state_path(out), json.dumps(state, indent=2) + "\n")
            if progress:
                progress(state["done"], len(ids))
    return state
