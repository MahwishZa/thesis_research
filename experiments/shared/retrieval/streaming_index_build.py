"""Memory-bounded, checkpointed index build for the real, full-scale corpus.

``build_index()``/``DenseIndex.save()`` in ``index.py`` hold the whole
corpus (every ``CorpusPassage``, i.e. every passage's text) and the whole
float32 vector matrix in process memory at once. That is correct and
simple for a small fixture, but on the real corpus (4.3M+ chunks, a 13 GB
chunk file, a ~12.8 GB vector matrix) it does not fit: a 2026-09-24
diagnostic run showed that materializing just the passage *text* - before
any model or vectors were involved - pushed a 16 GB-RAM laptop to ~0 GB
free and caused sustained OS paging (recorded in ``docs/research_log.md``).

This module builds the identical index - byte-for-byte the same
``vectors.npy``/``manifest.json`` schema, loadable by the unmodified
``DenseIndex.load()`` - in two bounded-memory passes over
``StreamingCorpusReader`` (``corpus.py``), which yields one passage at a
time rather than a materialized list:

  **Pass 1 (plan).** Stream the corpus once to determine the final,
  ordered set of passage ids (after the ``on_duplicate``/``dated_only``
  policy) and the corpus snapshot id. Written to ``<out>/_build/`` - a
  list of ids and a small metadata file, on the order of tens to low
  hundreds of MB even at full scale, not the corpus text itself. Cached:
  a second call with the same corpus file and the same policy reuses the
  existing plan instead of re-reading 4.3M lines.

  **Pass 2 (encode).** Stream the corpus again, batch by batch, writing
  each batch's vectors directly into a disk-backed ``numpy`` memmap
  instead of an in-memory array (the OS pages it in and out of the
  destination file itself, not the pagefile - the actual mechanism that
  was thrashing before). Peak memory is one batch of text and vectors
  plus the model, never the whole corpus.

**Checkpointed**, because this is an unattended, multi-hour job on
consumer hardware with no other recovery path if it is interrupted.
Every ``checkpoint_every_batches`` batches, progress is flushed to disk
and recorded atomically (write-temp-then-replace, so a kill mid-write
cannot corrupt the checkpoint) in ``<out>/_build/build_state.json``. A
resumed run validates that the corpus file, the encoder, and the
duplicate/dated-only policy all still match what the checkpoint was
built against before continuing - a mismatch refuses loudly rather than
silently writing a misaligned index.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .corpus import CorpusError, StreamingCorpusReader
from .index import l2_normalize

#: Subdirectory holding in-progress build artifacts (plan + checkpoint).
#: Renamed to ``_build_completed`` on success, kept for audit trail rather
#: than deleted - the same reasoning as keeping ``duplicate_chunks.json``.
BUILD_DIR_NAME = "_build"
COMPLETED_DIR_NAME = "_build_completed"

#: How many of a sampled row's values are checked for finiteness after the
#: build completes - cheap defense against a batch that was silently
#: skipped or wrote garbage, without reading the whole 12.8 GB array back.
_FINAL_SAMPLE_ROWS = 2000


class StreamingBuildError(RuntimeError):
    """Raised when a streaming index build cannot proceed safely."""


@dataclass(frozen=True)
class BuildResult:
    out_dir: Path
    snapshot_id: str
    n_passages: int
    dim: int
    encoder_name: str
    duplicates_count: int


def _write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON so a process killed mid-write cannot corrupt ``path``.

    ``os.replace`` is atomic on both POSIX and Windows when source and
    destination are on the same filesystem, which they always are here
    (same output directory) - so the checkpoint file is either the old
    version or the new one, never a half-written one.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _corpus_file_stat(corpus_root: str | Path) -> dict[str, Any]:
    path = Path(corpus_root) / "data" / "chunks" / "chunks.jsonl"
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime": st.st_mtime}


@dataclass(frozen=True)
class PlanResult:
    total: int
    snapshot_id: str
    dated_only: bool
    on_duplicate: str
    duplicates: tuple[dict[str, Any], ...]
    corpus_stat: dict[str, Any]


def _plan_matches_request(
    meta: dict[str, Any],
    *,
    corpus_root: str | Path,
    dated_only: bool,
    on_duplicate: str,
) -> bool:
    try:
        current = _corpus_file_stat(corpus_root)
    except OSError:
        return False
    return (
        meta.get("corpus_stat", {}).get("path") == current["path"]
        and meta.get("corpus_stat", {}).get("size") == current["size"]
        and meta.get("corpus_stat", {}).get("mtime") == current["mtime"]
        and meta.get("dated_only") == dated_only
        and meta.get("on_duplicate") == on_duplicate
    )


def plan_index_build(
    corpus_root: str | Path,
    out_dir: str | Path,
    *,
    dated_only: bool = True,
    on_duplicate: str = "raise",
    on_progress: Optional[Callable[[int], None]] = None,
) -> PlanResult:
    """Pass 1: stream the corpus once to fix the final row order.

    Writes ``<out>/_build/passage_ids.txt`` (one chunk_id per line, in the
    exact order Pass 2 must encode them) and ``<out>/_build/plan_meta.json``.
    Reused on a later call with an unchanged corpus file and policy -
    checked by file size + mtime, not re-hashed, so re-planning a resumed
    build is a stat call, not another 4.3M-line read.
    """
    build_dir = Path(out_dir) / BUILD_DIR_NAME
    build_dir.mkdir(parents=True, exist_ok=True)
    meta_path = build_dir / "plan_meta.json"
    ids_path = build_dir / "passage_ids.txt"

    if meta_path.exists() and ids_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if _plan_matches_request(
            meta, corpus_root=corpus_root, dated_only=dated_only,
            on_duplicate=on_duplicate,
        ):
            return PlanResult(
                total=meta["total"],
                snapshot_id=meta["snapshot_id"],
                dated_only=meta["dated_only"],
                on_duplicate=meta["on_duplicate"],
                duplicates=tuple(meta["duplicates"]),
                corpus_stat=meta["corpus_stat"],
            )

    reader = StreamingCorpusReader(
        corpus_root, on_progress=on_progress, on_duplicate=on_duplicate,
        dated_only=dated_only,
    )
    total = 0
    with open(ids_path, "w", encoding="utf-8") as out:
        for passage in reader:
            out.write(passage.chunk_id)
            out.write("\n")
            total += 1

    if total == 0:
        raise CorpusError(
            f"{reader.path} produced zero passages under dated_only="
            f"{dated_only!r}, on_duplicate={on_duplicate!r}"
        )

    result = PlanResult(
        total=total,
        snapshot_id=reader.snapshot_id,
        dated_only=dated_only,
        on_duplicate=on_duplicate,
        duplicates=reader.duplicates,
        corpus_stat=_corpus_file_stat(corpus_root),
    )
    _write_json_atomic(meta_path, {
        "total": result.total,
        "snapshot_id": result.snapshot_id,
        "dated_only": result.dated_only,
        "on_duplicate": result.on_duplicate,
        "duplicates": list(result.duplicates),
        "corpus_stat": result.corpus_stat,
    })
    return result


def encode_index_streaming(
    corpus_root: str | Path,
    out_dir: str | Path,
    encoder: Any,
    *,
    dated_only: bool = True,
    on_duplicate: str = "raise",
    batch_size: int = 32,
    checkpoint_every_batches: int = 200,
    resume: bool = False,
    on_progress: Optional[Callable[[int], None]] = None,
    on_batch_progress: Optional[Callable[[int, int, float], None]] = None,
) -> BuildResult:
    """Pass 2: encode the planned passages, writing vectors straight to a
    disk-backed memmap, checkpointing every ``checkpoint_every_batches``
    batches so an interrupted run can be resumed instead of restarted.

    Output is written to ``<out>/vectors.npy`` + ``<out>/manifest.json`` -
    the exact schema ``DenseIndex.save()`` writes and ``DenseIndex.load()``
    reads, unchanged. ``<out>/_build/`` holds only working state and is
    renamed to ``<out>/_build_completed/`` on success.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        raise StreamingBuildError(
            f"refusing to build into {out_dir}: a complete index (manifest."
            "json) already exists there. Choose a new --out to build a "
            "fresh one, or remove the existing index first."
        )

    build_dir = out_dir / BUILD_DIR_NAME
    state_path = build_dir / "build_state.json"
    vectors_path = out_dir / "vectors.npy"

    corpus_stat_now = _corpus_file_stat(corpus_root)

    if state_path.exists():
        if not resume:
            raise StreamingBuildError(
                f"an incomplete build is already in progress at {out_dir} "
                f"({state_path} exists but {manifest_path} does not). Pass "
                "resume=True (CLI: --resume) to continue it, or choose a "
                "different --out to start over."
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        mismatches = []
        if state["corpus_stat"] != corpus_stat_now:
            mismatches.append(
                "the corpus file has changed size/modification time since "
                "the checkpoint was written"
            )
        if state["dated_only"] != dated_only:
            mismatches.append(
                f"dated_only was {state['dated_only']!r}, now {dated_only!r}"
            )
        if state["on_duplicate"] != on_duplicate:
            mismatches.append(
                f"on_duplicate was {state['on_duplicate']!r}, now "
                f"{on_duplicate!r}"
            )
        encoder_name = getattr(encoder, "name", type(encoder).__name__)
        if state["encoder_name"] != encoder_name:
            mismatches.append(
                f"encoder was {state['encoder_name']!r}, now {encoder_name!r}"
            )
        if mismatches:
            raise StreamingBuildError(
                f"cannot resume the build at {out_dir}: " + "; ".join(mismatches)
                + ". Delete " + str(build_dir) + " and start a fresh build "
                "if this is intentional."
            )
        plan = PlanResult(
            total=state["total"], snapshot_id=state["snapshot_id"],
            dated_only=state["dated_only"], on_duplicate=state["on_duplicate"],
            duplicates=tuple(state["duplicates"]), corpus_stat=state["corpus_stat"],
        )
        dim = state["dim"]
        rows_done = state["rows_done"]
        vectors = np.lib.format.open_memmap(vectors_path, mode="r+")
        if vectors.shape != (plan.total, dim):
            raise StreamingBuildError(
                f"checkpoint says shape ({plan.total}, {dim}) but "
                f"{vectors_path} is {vectors.shape} - the output file does "
                "not match the checkpoint. Delete both and start over."
            )
    else:
        plan = plan_index_build(
            corpus_root, out_dir, dated_only=dated_only,
            on_duplicate=on_duplicate, on_progress=on_progress,
        )
        dim = None  # learned from the first encoded batch, below
        rows_done = 0
        vectors = None  # created once dim is known

    encoder_name = getattr(encoder, "name", type(encoder).__name__)
    ids_path = build_dir / "passage_ids.txt"

    def _checkpoint() -> None:
        vectors.flush()
        _write_json_atomic(state_path, {
            "total": plan.total, "snapshot_id": plan.snapshot_id,
            "dated_only": plan.dated_only, "on_duplicate": plan.on_duplicate,
            "duplicates": list(plan.duplicates), "corpus_stat": plan.corpus_stat,
            "encoder_name": encoder_name, "dim": dim, "rows_done": rows_done,
            "updated_at": time.time(),
        })

    reader = StreamingCorpusReader(
        corpus_root, on_progress=on_progress, on_duplicate=on_duplicate,
        dated_only=dated_only,
    )
    encode_t0 = time.monotonic()
    batches_done = 0
    with open(ids_path, "r", encoding="utf-8") as id_stream:
        # Advance both the corpus stream and the plan's id list past
        # whatever was already encoded, so a resume never re-encodes a row
        # or drifts out of alignment with it.
        for _ in range(rows_done):
            next(id_stream)

        passage_iter = iter(reader)
        for _ in range(rows_done):
            next(passage_iter)

        batch_passages: list = []
        batch_ids: list[str] = []

        def _flush_batch() -> None:
            nonlocal vectors, dim, rows_done, batches_done
            if not batch_passages:
                return
            texts = [p.retrieval_text for p in batch_passages]
            batch_vectors = l2_normalize(np.asarray(
                encoder.encode(texts), dtype=np.float32
            ))
            if batch_vectors.shape[0] != len(batch_passages):
                raise StreamingBuildError(
                    f"encoder returned {batch_vectors.shape[0]} vectors for "
                    f"{len(batch_passages)} passages"
                )
            if vectors is None:
                dim = int(batch_vectors.shape[1])
                vectors = np.lib.format.open_memmap(
                    vectors_path, mode="w+", dtype=np.float32,
                    shape=(plan.total, dim),
                )
            elif batch_vectors.shape[1] != dim:
                raise StreamingBuildError(
                    f"encoder produced dimension {batch_vectors.shape[1]} "
                    f"but this build started at dimension {dim}"
                )
            start = rows_done
            end = start + len(batch_passages)
            vectors[start:end] = batch_vectors
            rows_done = end
            batches_done += 1
            batch_passages.clear()
            batch_ids.clear()
            if batches_done % checkpoint_every_batches == 0:
                _checkpoint()
            if on_batch_progress is not None:
                on_batch_progress(rows_done, plan.total, time.monotonic() - encode_t0)

        for passage, expected_id in zip(passage_iter, id_stream):
            expected_id = expected_id.rstrip("\n")
            if passage.chunk_id != expected_id:
                raise StreamingBuildError(
                    f"corpus/plan misalignment at row {rows_done + len(batch_passages)}: "
                    f"expected {expected_id!r} from the plan, got "
                    f"{passage.chunk_id!r} from the corpus. The corpus file "
                    "likely changed since planning; delete "
                    f"{build_dir} and start a fresh build."
                )
            batch_passages.append(passage)
            batch_ids.append(passage.chunk_id)
            if len(batch_passages) >= batch_size:
                _flush_batch()
        _flush_batch()  # final partial batch, if any

    if rows_done != plan.total:
        raise StreamingBuildError(
            f"encoded {rows_done} rows but the plan expected {plan.total} "
            "- the corpus stream ended early. Delete "
            f"{build_dir} and start a fresh build."
        )

    _checkpoint()  # final checkpoint before validation, belt-and-braces

    # Cheap corruption check: a sample of rows must be finite and non-zero.
    # A batch that was silently skipped would leave its rows at the
    # memmap's zero-fill default, which a real MedCPT embedding never is.
    rng = np.random.default_rng(0)
    sample_rows = rng.choice(
        plan.total, size=min(_FINAL_SAMPLE_ROWS, plan.total), replace=False
    )
    sample = np.asarray(vectors[sample_rows])
    if not np.isfinite(sample).all():
        raise StreamingBuildError(
            "the completed vector matrix contains non-finite values in a "
            "sampled row - the build did not complete cleanly. Delete "
            f"{build_dir} and {vectors_path}, and start a fresh build."
        )
    if (sample == 0.0).all(axis=1).any():
        raise StreamingBuildError(
            "the completed vector matrix contains an all-zero row in a "
            "sampled row, consistent with an unwritten batch. Delete "
            f"{build_dir} and {vectors_path}, and start a fresh build."
        )
    vectors.flush()
    del vectors  # release the memmap before manifest write / directory rename

    passage_ids = ids_path.read_text(encoding="utf-8").splitlines()
    if len(passage_ids) != plan.total:
        raise StreamingBuildError(
            f"{ids_path} has {len(passage_ids)} lines but the plan expected "
            f"{plan.total}"
        )
    manifest = {
        "index_format": 1,
        "encoder_name": encoder_name,
        "corpus_snapshot": plan.snapshot_id,
        "n_passages": plan.total,
        "dim": dim,
        "passage_ids": passage_ids,
        "metadata": {"dated_only": dated_only},
    }
    _write_json_atomic(manifest_path, manifest)

    completed_dir = out_dir / COMPLETED_DIR_NAME
    if completed_dir.exists():
        # A previous completed-build archive from an earlier attempt at
        # this same --out; keep the latest, don't silently merge.
        import shutil
        shutil.rmtree(completed_dir)
    build_dir.rename(completed_dir)

    return BuildResult(
        out_dir=out_dir, snapshot_id=plan.snapshot_id, n_passages=plan.total,
        dim=dim, encoder_name=encoder_name, duplicates_count=len(plan.duplicates),
    )
