"""Safe reading and writing of the project's files (JSON, JSON Lines, CSV): a half-written file never replaces a good one."""
import contextlib
import csv
import json
import os
import tempfile
from pathlib import Path


@contextlib.contextmanager
def atomic_path(path):
    """Yields a temporary path next to `path`; when the block ends without an error it replaces `path` in one step.
    An interrupted run therefore never leaves a half-written file behind (the old file stays as it was)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_text(path, text):
    with atomic_path(path) as tmp, open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_json(path, obj):
    """The same format as json.dump(obj, f, indent=2), which made the committed .json files (no newline at the end)."""
    write_text(path, json.dumps(obj, indent=2))


def write_jsonl(path, records):
    write_text(path, "".join(json.dumps(r) + "\n" for r in records))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    """Writes rows (dicts) with the given columns, LF line endings and minimal quoting (the same format pandas writes)."""
    with atomic_path(path) as tmp, open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        w.writerows([r[c] for c in columns] for r in rows)
