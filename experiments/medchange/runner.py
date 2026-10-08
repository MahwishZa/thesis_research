"""Helpers shared by the phase drivers (``rag2_pipeline.py``): the expected item counts, the command builder,
the step executor, the "is it on origin/main" guard, and a commit that never pushes.

A phase is a list of steps; a step is either a command (argv list) or a function returning ``(ok, message)``.
``execute`` runs them in order and stops at the first failure, so a rerun of the same command resumes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED_ITEMS = {"dev": 226, "confirm": 528, "ad": 208}   # equal to manifest.json / manifest_ad.json (a test checks)

Step = tuple[str, object]          # (name, argv list | callable returning (ok, message))


def py(module: str, *args) -> list[str]:
    return [sys.executable, "-m", f"experiments.medchange.{module}", *map(str, args)]


def frozen_is_pushed(rel: str, repo: Path = ROOT) -> tuple[bool, str]:
    """True when the working copy of ``rel`` equals the one on origin/main (last known state)."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)
    local = git("hash-object", rel)
    remote = git("rev-parse", f"origin/main:{rel}")
    if local.returncode != 0:
        return False, f"{rel} does not exist"
    if remote.returncode != 0:
        return False, f"{rel} is not on origin/main: commit it and push it yourself (git push origin main)"
    if local.stdout.strip() != remote.stdout.strip():
        return False, f"{rel} differs from origin/main: commit it and push the final version yourself first"
    return True, "file is on origin/main"


def commit_results(message: str, repo: Path = ROOT) -> tuple[bool, str]:
    """Commit the results; never pushes (the researcher pushes by hand: ``git push origin main``)."""
    subprocess.run(["git", "add", "experiments/medchange/results"], cwd=repo)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode == 0:
        return True, "nothing new to commit"
    if subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo).returncode != 0:
        return False, "git commit failed; fix it and rerun the same command"
    return True, "committed; now push it yourself: git push origin main"


def print_checks(rows: Sequence[tuple[str, bool, str]]) -> tuple[bool, str]:
    for name, ok, detail in rows:
        print(f"    {'ok  ' if ok else 'FAIL'} {name} {detail}")
    return all(ok for _, ok, _ in rows), "integrity checks"


def execute(steps: Sequence[Step], dry_run: bool) -> int:
    for k, (name, action) in enumerate(steps, 1):
        print(f"[{k}/{len(steps)}] {name}", flush=True)
        if dry_run:
            if isinstance(action, list):
                print("    " + " ".join(map(str, action)))
            continue
        if callable(action):
            ok, message = action()
            print(f"    {message}")
        else:
            ok = subprocess.run([str(x) for x in action], cwd=ROOT).returncode == 0
        if not ok:
            print(f"STOPPED at step {k} ({name}). Nothing later was run; fix the cause and rerun the same command.",
                  file=sys.stderr)
            return 2
    return 0
