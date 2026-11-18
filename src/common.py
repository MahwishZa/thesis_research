"""Helpers the code shares: for the dataset builders (src/build/), a polite download that tries again when a service is busy, a
report with one section per step that each step rewrites on its own, reading a table, and a nested record as flat names; for the
runs (src/run.py, src/evaluate.py), crash-safe answer files."""

from __future__ import annotations

import csv
import json
import os
import http.client
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "thesis-research-dataset-preparation/1.0 (academic research)"


def fetch(url: str, accept: str, check=None, delay: float = 0.3, attempts: int = 4, form: dict | None = None):
    """The answer at ``url`` (bytes, or what ``check`` makes of them), after waiting ``delay`` seconds (a polite pace); with
    ``form``, its fields are sent in the request's body (a POST, for a list too long for an address). A busy service, a network
    failure or a garbled answer is tried again with growing waits (4, 8, 16 ... seconds); any other refusal raises urllib's
    HTTPError at once (retrying cannot help)."""
    body = urllib.parse.urlencode(form).encode("ascii") if form is not None else None
    request = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT, "Accept": accept})
    problem = ""
    for attempt in range(1, attempts + 1):
        time.sleep(delay if attempt == 1 else 2 ** attempt)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
            return check(content) if check else content
        except urllib.error.HTTPError as exc:  # the service answered, but not with what was asked for
            if exc.code != 429 and exc.code < 500:
                raise
            problem = f"HTTP {exc.code}"
        except (OSError, http.client.HTTPException, ValueError) as exc:  # network, cut-off or garbled answer (JSON, picture)
            problem = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
    raise ValueError(f"cannot get {url} after {attempts} attempts ({problem}). What was downloaded so far is kept: run the "
                     "same command again later to continue")


def write_section(path: Path, head: list[str], steps: tuple[str, ...], step: str, lines: list[str]) -> None:
    """Rewrites one step's section of the report at ``path`` (its headings one level down, under the report's ``head``) and keeps
    the other steps' sections, in the order the steps run (``steps``)."""
    sections, current = {}, None
    if path.is_file():
        for line in path.read_text(encoding="utf-8").split("\n"):
            marker = re.fullmatch(r"<!-- step: (\w+) -->", line)
            if marker:
                current = marker.group(1)
                sections[current] = []
            elif current:
                sections[current].append(line)
    sections[step] = [f"#{line}" if line.startswith("#") else line for line in lines]
    text = list(head)
    for name in (name for name in steps if name in sections):
        body = sections[name]
        while body and not body[-1]:
            body = body[:-1]
        text += [f"<!-- step: {name} -->", *body, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text).rstrip("\n") + "\n", encoding="utf-8", newline="\n")


def read_rows(path: Path, missing: str) -> list[dict]:
    """The rows of a CSV table; ``missing`` says what to do when the table does not exist."""
    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} not found: {missing}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def flatten(value, prefix: str = "") -> dict:
    """A nested record as dotted names: {"metadata": {"clinical": {"sex": "male"}}} -> {"metadata.clinical.sex": "male"}."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


# ---------------------------------------------------------------------------------------------------------------------------------
# Crash-safe files: a run can be stopped at any moment (power cut, Ctrl+C) and continued without losing or damaging answers


def _split_last_line(content: bytes) -> tuple[str, bytes]:
    """The complete lines of a JSON-lines file (decoded strictly) and the bytes after its last end of line."""
    cut = content.rfind(b"\n") + 1
    return content[:cut].decode("utf-8"), content[cut:]


def _complete_row(tail: bytes) -> dict | None:
    """The row in bytes that lack their end of line, or None when they are a cut-off line (even one cut inside a character)."""
    try:
        row = json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return row if isinstance(row, dict) else None


def read_jsonl(path: Path) -> list[dict]:
    """The rows of a JSON-lines file. An incomplete last line (the program was stopped while writing it) is ignored; a damaged
    line anywhere else is an error, never silently skipped."""
    if not path.exists():
        return []
    head, tail = _split_last_line(path.read_bytes())
    rows = []
    for number, line in enumerate(head.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {number} is not valid JSON ({exc.msg}); the file is damaged") from exc
    if tail.strip() and (last := _complete_row(tail)) is not None:
        rows.append(last)
    return rows


def repair_jsonl(path: Path) -> None:
    """Before answers are appended: remove an incomplete last line, or add the missing end of line after a complete one."""
    if not path.exists():
        return
    content = path.read_bytes()
    if not content or content.endswith(b"\n"):
        return
    head, tail = _split_last_line(content)
    if _complete_row(tail) is not None:
        fixed = content + b"\n"
    else:
        fixed = head.encode("utf-8")
        print(f"{path}: removed an incomplete last line (the program was stopped while writing it); that file is asked again")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(fixed)
    os.replace(tmp, path)


def write_json(path: Path, content: dict) -> None:
    """Written to a temporary file first, so a stop in the middle never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
