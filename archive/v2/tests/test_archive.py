"""The Version 1 archive (archive/v1/): every file is the same as at the Version 1 commit, apart from the label of the
hand-written documents, and nothing is missing or added."""

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "v1"
COMMIT = "a1c9209"
FOLDERS = ("data", "results", "docs", "configs", "prompts")
BANNER = ("> **Version 1** (commit `a1c9209`), archived on 2026-11-19: a frozen copy kept for reference. "
          "It does not bind Version 2 (see `docs/v2_proposal.md` in the repository).")
LABELLED = {"data/image/datasheet.md", "data/image_metadata/datasheet.md", "data/text/datasheet.md", "data/cases/datasheet.md",
            "data/knowledge/datasheet.md", "docs/method.md", "docs/results.md"}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def history_available() -> bool:
    return bool(shutil.which("git")) and (ROOT / ".git").exists() and git("cat-file", "-e", f"{COMMIT}^{{commit}}").returncode == 0


def unlabelled(path: str, content: bytes) -> bytes:
    """A labelled document without its label: the banner line and the blank line after it."""
    head = (BANNER + "\n\n").encode("utf-8")
    if path in LABELLED:
        if not content.startswith(head):
            raise AssertionError(f"archive/v1/{path} does not start with the Version 1 label")
        return content[len(head):]
    return content


class ArchiveTests(unittest.TestCase):
    def test_the_archive_has_its_readme_and_the_labelled_documents(self):
        self.assertTrue((ARCHIVE / "README.md").is_file())
        readme = (ARCHIVE / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"commit `{COMMIT}`", readme)
        for path in LABELLED:
            self.assertIn(f"`{path}`", readme)
            self.assertTrue((ARCHIVE / path).read_text(encoding="utf-8").startswith(BANNER + "\n\n"), path)

    @unittest.skipUnless(history_available(), f"needs a Git checkout that holds commit {COMMIT}")
    def test_every_archived_file_is_the_file_of_the_version_1_commit(self):
        listed = git("ls-tree", "-r", "--name-only", COMMIT, "--", *FOLDERS)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        expected = sorted(p for p in listed.stdout.decode("utf-8").splitlines() if p)
        archived = sorted(p.relative_to(ARCHIVE).as_posix() for p in ARCHIVE.rglob("*") if p.is_file() and p != ARCHIVE / "README.md")
        self.assertEqual(archived, expected, "files missing from the archive, or added to it")
        for path in expected:
            with self.subTest(path=path):
                original = git("show", f"{COMMIT}:{path}")
                self.assertEqual(original.returncode, 0, original.stderr)
                self.assertEqual(unlabelled(path, (ARCHIVE / path).read_bytes()), original.stdout,
                                 f"archive/v1/{path} differs from {path} at commit {COMMIT}")

    def test_a_label_without_its_blank_line_or_an_unlabelled_document_is_caught(self):
        with self.assertRaises(AssertionError):
            unlabelled("docs/results.md", b"# Closed-book results\n")
        with self.assertRaises(AssertionError):
            unlabelled("docs/results.md", (BANNER + "\n# x\n").encode("utf-8"))
        self.assertEqual(unlabelled("docs/results.md", (BANNER + "\n\n# x\n").encode("utf-8")), b"# x\n")
        self.assertEqual(unlabelled("configs/models.toml", b"a = 1\n"), b"a = 1\n")


if __name__ == "__main__":
    unittest.main()
