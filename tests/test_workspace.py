"""The clean workspace and the archive of Version 2: the README holds only the overview and the objectives, and archive/v2/ is the
research of commit 95a9506 exactly (every file as listed in its manifest, and the same as at that commit)."""

import hashlib
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "v2"
COMMIT = "95a9506"
ADDED = ("ARCHIVE.md", "MANIFEST.sha256")  # in the archive, not in the commit
NOT_ARCHIVED = (".claude/", ".gitignore", ".gitattributes", "archive/")  # stay where they are


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def history_available() -> bool:
    return bool(shutil.which("git")) and (ROOT / ".git").exists() and git("cat-file", "-e", f"{COMMIT}^{{commit}}").returncode == 0


def tracked(folder: str) -> list[str]:
    return [p for p in git("ls-files", "-z", folder).stdout.decode("utf-8").split("\0") if p]


class WorkspaceTests(unittest.TestCase):
    def test_the_readme_holds_only_the_overview_and_the_objectives(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(re.findall(r"^#{2,3} (.+)$", readme, re.M), ["Overview", "Objectives"])
        self.assertEqual(len(re.findall(r"^\d+\. ", readme, re.M)), 2)  # the two objectives

    def test_the_archived_readme_and_requirements_are_the_ones_of_the_commit_and_the_active_ones_are_unchanged(self):
        self.assertEqual((ARCHIVE / "README.md").read_bytes(), (ROOT / "README.md").read_bytes())
        self.assertEqual((ARCHIVE / "requirements.txt").read_bytes(), (ROOT / "requirements.txt").read_bytes())


class ArchiveTests(unittest.TestCase):
    def test_every_file_of_the_archive_is_in_its_manifest_with_the_same_hash_and_nothing_else_is_there(self):
        listed = {}
        for line in (ARCHIVE / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
            digest, path = line.split("  ", 1)
            listed[path] = digest
        for path, digest in listed.items():
            with self.subTest(path):
                self.assertEqual(hashlib.sha256((ARCHIVE / path).read_bytes()).hexdigest(), digest)
        if shutil.which("git") and (ROOT / ".git").exists():
            held = {p[len("archive/v2/"):] for p in tracked("archive/v2")} - set(ADDED)
            self.assertEqual(sorted(held - set(listed)), [], "files in the archive that the manifest does not list")
            self.assertEqual(sorted(set(listed) - held), [], "files the manifest lists that are not in the archive")

    @unittest.skipUnless(history_available(), "the Git history of the archived commit is not available here")
    def test_the_archive_is_the_research_of_the_commit_file_for_file(self):
        names = [n for n in git("ls-tree", "-r", "-z", "--name-only", COMMIT).stdout.decode("utf-8").split("\0") if n]
        moved = [n for n in names if not n.startswith(NOT_ARCHIVED)]
        self.assertGreater(len(moved), 200)
        for name in moved:
            content = git("show", f"{COMMIT}:{name}").stdout
            self.assertTrue((ARCHIVE / name).is_file(), f"missing from the archive: {name}")
            self.assertEqual((ARCHIVE / name).read_bytes(), content, f"differs from the commit: {name}")


if __name__ == "__main__":
    unittest.main()
