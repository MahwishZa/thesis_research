"""A lean copy of ``corpus/`` for tests that run the stage scripts.

Several tests need a throwaway ``corpus/`` tree to run stages 03-07 against, writing
into it freely. ``corpus/data/`` is gitignored and, on a machine that has built the
real corpus, holds many GB (raw PMC XML/JSON, normalised documents, chunks). Copying
the whole tree per test would be slow, fill the disk, and make results depend on
local state. This copy takes everything tracked (scripts, config, metadata, reports,
logs) but, under ``data/``, only the directory skeleton and the committed offline
fixture, so a test sees the same tree on a fresh clone and on the researcher's laptop.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus"

#: Directories under corpus/data/ recreated (empty) in the scaffold.
_DATA_DIRS = {
    "data", "data/raw", "data/raw/pubmed", "data/raw/guidelines", "data/raw/textbooks",
    "data/raw/currency_pack", "data/normalized", "data/deduplicated", "data/chunks",
}
#: The only data files copied: the committed synthetic fixture and .gitkeep markers.
_DATA_FILES = {"records.example.jsonl", ".gitkeep"}


def _ignore(directory: str, names: list[str]) -> set[str]:
    rel = Path(directory).resolve().relative_to(CORPUS).as_posix()
    skipped = {n for n in names if n == "__pycache__" or n.endswith((".pyc", ".pyo"))}
    inside_data = rel == "data" or rel.startswith("data/")
    if not inside_data:
        return skipped
    for n in names:
        full = os.path.join(directory, n)
        child = f"{rel}/{n}" if rel != "." else n
        if os.path.isdir(full):
            if child not in _DATA_DIRS:
                skipped.add(n)
        elif n not in _DATA_FILES:
            skipped.add(n)
    return skipped


def copy_corpus_scaffold(dest: Path) -> Path:
    """Copy the lean corpus tree to ``dest`` (which must not exist) and return it."""
    shutil.copytree(CORPUS, dest, ignore=_ignore)
    return Path(dest)
