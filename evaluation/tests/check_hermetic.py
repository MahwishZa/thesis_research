"""Run the test suites and fail if they created, modified or deleted any file in the repository.

Tests must be hermetic: they may read the tree but write only to temporary directories. A test
that writes into the real tree - for example a stage script's resume marker under ``corpus/data`` -
can damage a genuine run's state on a machine that holds the real corpus, and it passes silently.
This check found exactly that once (2026-10-02); it compares the size and modification time of every
file, ignored files included, before and after both suites.

    python -m evaluation.tests.check_hermetic
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: Directories whose contents are not part of the check (git's own state and Python bytecode).
IGNORED_PARTS = {".git", "__pycache__"}
#: The suites that must leave the tree untouched.
SUITES = ("evaluation",)


def snapshot(root=ROOT):
    """``{relative posix path: (size, mtime_ns)}`` for every file under ``root``."""
    state = {}
    for path in Path(root).rglob("*"):
        parts = path.relative_to(root).parts
        if IGNORED_PARTS.intersection(parts) or not path.is_file():
            continue
        stat = path.stat()
        state[Path(*parts).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return state


def changes(before, after):
    """Human-readable differences between two snapshots; empty when nothing changed."""
    found = [f"created:  {name}" for name in sorted(set(after) - set(before))]
    found += [f"deleted:  {name}" for name in sorted(set(before) - set(after))]
    found += [f"modified: {name}" for name in sorted(set(before) & set(after))
              if before[name] != after[name]]
    return found


def main():
    before = snapshot()
    failed = []
    for suite in SUITES:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", suite, "-t", "."],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"suite {suite}: {'passed' if result.returncode == 0 else 'FAILED'}")
        if result.returncode != 0:
            failed.append(suite)
    found = changes(before, snapshot())
    for line in found:
        print(line)
    print("hermetic: the suites left the repository tree unchanged" if not found
          else f"NOT hermetic: {len(found)} file(s) changed")
    return 1 if (found or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
