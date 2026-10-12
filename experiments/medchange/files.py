"""Reading and hashing files, shared by the commands: JSON-lines files that survive a run killed mid-write, and the SHA-256
of model files and of the text files of the MedChange release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """The records of a JSON-lines file; a missing file gives ``[]`` and a truncated last line (a run killed mid-write) is ignored."""
    rows = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    break
                raise
    return rows


def file_sha256(path) -> str:
    """SHA-256 of the bytes of a file, read in 1 MiB blocks (a model file is several gigabytes)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_file_sha256(path: Path) -> str:
    """SHA-256 of a text file with CRLF normalised to LF, so a Windows checkout with git's autocrlf conversion and a Linux
    checkout of the same release hash alike."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
