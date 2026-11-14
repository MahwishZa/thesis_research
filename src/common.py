"""Helpers the dataset folders share (src/image/, src/text/, src/multimodal/): a polite download that tries again when a service is busy, and a
report with one section per step that each step rewrites on its own."""

from __future__ import annotations

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
