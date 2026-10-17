"""The guidelines corpus: the openly licensed clinical practice guidelines of the Meditron collection (``epfl-llm/guidelines``).

The paper's guideline corpus comes from the same collection ("8 of 16 sources public"). The collection is a JSON-lines file on
the Hugging Face hub with one guideline per line; the text is in ``clean_text`` (or ``raw_text``). ``read_guidelines`` turns such
lines into documents; ``download_guidelines`` fetches the file with ``huggingface_hub`` and records which file and revision it
used. The licence of each guideline is the one of its source and was not checked; the source name is kept in each document.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from rationale_query.store import StoreError, write_atomic

REPO = "epfl-llm/guidelines"
TEXT_FIELDS = ("clean_text", "raw_text", "text")


def read_guidelines(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """One document per guideline that has text; a guideline is keyed by its URL, else by its id."""
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StoreError(f"guidelines line {number} is not valid JSON: {exc}") from exc
        text = next((item[f] for f in TEXT_FIELDS if isinstance(item.get(f), str) and item[f].strip()), "")
        if not text:
            continue
        url, ident = str(item.get("url") or ""), str(item.get("id") or "")
        title = str(item.get("title") or "").strip()
        yield {
            "doc_key": f"URL:{url}" if url else (f"GUIDELINE:{ident}" if ident else ""),
            "text": f"{title}\n{text}" if title and not text.startswith(title) else text,
            "source": str(item.get("source") or ""),
            "title": title,
        }


def choose_file(names: list[str]) -> str:
    """The open guideline file among the files of the repository: a ``.jsonl`` with ``open`` in its name, the newest version."""
    open_files = sorted(n for n in names if n.endswith(".jsonl") and "open" in n.lower())
    if not open_files:
        raise StoreError(f"{REPO} holds no open .jsonl file (files: {', '.join(names)}); name one with --file")
    return open_files[-1]


def download_guidelines(out: Path, *, file: str | None = None, revision: str | None = None, hub: Any = None) -> dict[str, Any]:
    """Write the guidelines as documents into ``out``; returns what was used. ``hub`` is the ``huggingface_hub`` module."""
    if hub is None:
        import huggingface_hub as hub  # imported here: a plain install of the package does not need it
    name = file or choose_file(hub.HfApi().list_repo_files(REPO, repo_type="dataset", revision=revision))
    path = hub.hf_hub_download(REPO, name, repo_type="dataset", revision=revision)
    sha = hub.HfApi().dataset_info(REPO, revision=revision).sha
    count = 0
    lines = []
    with open(path, encoding="utf-8") as handle:
        for document in read_guidelines(handle):
            lines.append(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    write_atomic(out, "".join(lines))
    info = {"source": f"{REPO}/{name}", "revision": sha, "documents": count}
    write_atomic(out.with_name(out.name + ".source.json"), json.dumps(info, indent=2) + "\n")
    return info
