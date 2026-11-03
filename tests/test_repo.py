"""The repository against itself: README, command line, configs and committed results must agree (a drift fails here)."""

import argparse
import ast
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src import dataset, evaluate, isic

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "results" / "dataset_profile"


class RepoTests(unittest.TestCase):
    def setUp(self):
        cwd = os.getcwd()
        os.chdir(ROOT)  # configs name their files relative to the repository folder
        self.addCleanup(os.chdir, cwd)
        self.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_every_command_in_the_readme_is_accepted_by_the_command_line(self):
        commands = re.findall(r"python -m src\.evaluate ([^`\n]+)", self.readme)
        self.assertGreaterEqual(len(commands), 6)
        for arguments in commands:
            with self.subTest(arguments):
                evaluate.build_parser().parse_args(shlex.split(arguments))

    def test_commands_and_modules_are_documented(self):
        parser = evaluate.build_parser()
        names = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices
        for name in names:
            self.assertIn(f"python -m src.evaluate {name}", evaluate.__doc__, f"the module docstring omits '{name}'")
            self.assertIn(f"python -m src.evaluate {name}", self.readme, f"the README omits '{name}'")
        for module in sorted((ROOT / "src").glob("*.py")) + [ROOT / "src" / "models"]:
            if module.name != "__init__.py":
                self.assertIn(module.name if module.is_file() else "models/", self.readme, f"the README tree omits {module.name}")

    def test_configs_are_valid_and_point_to_files_that_exist(self):
        configs = sorted((ROOT / "configs").glob("*.toml"))
        self.assertTrue(configs)
        for path in configs:
            with self.subTest(path.name):
                config = evaluate.load_config(path)
                self.assertEqual(config["name"], path.stem, "the config name is the run folder name and must match the file")
                for key in (config["prompt"]["file"], config["data"]["corrections"], config["data"]["manifest"]):
                    self.assertTrue((ROOT / key).is_file(), f"{path.name} names {key}, which does not exist")

    def test_committed_results_agree_with_each_other(self):
        with open(PROFILE / "manifest.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        records = [dataset.Record(r["split"], r["label"], r["path"], int(r["bytes"]), int(r["width"]), int(r["height"]),
                                  r["sha256"], r["dhash"], r["isic_id"]) for r in rows]
        profile = (PROFILE / "profile.md").read_text(encoding="utf-8")
        self.assertIn(f"Images: {len(records)} ", profile)
        self.assertIn(dataset.items_hash(records), profile, "profile.md was not written from this manifest")
        train = {r.label for r in records if r.split == "train"}
        self.assertFalse({r.label for r in records if r.split == "test"} - train, "a test class has no training images")
        with open(PROFILE / "isic_check.csv", encoding="utf-8") as f:
            checked = {r["isic_id"] for r in csv.DictReader(f)}
        untested = [r.path for r in records if r.split == "test" and r.isic_id not in checked]
        self.assertEqual(untested, [], "test images without an ISIC check")
        with tempfile.TemporaryDirectory() as d:  # the corrections file must be exactly what the isic command derives
            isic.write_corrections(PROFILE / "manifest.csv", PROFILE / "isic_check.csv", Path(d) / "c.csv")
            self.assertEqual((Path(d) / "c.csv").read_bytes(), (PROFILE / "label_corrections.csv").read_bytes())
        folder_labels = {r["path"]: r["label"] for r in rows}
        with open(PROFILE / "label_corrections.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.assertEqual(folder_labels[r["path"]], r["folder_label"])
                self.assertIn(r["corrected_label"], train)

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a Git checkout")
    def test_git_ignores_keys_unfinished_runs_and_the_dataset_but_not_results(self):
        def ignored(path):
            return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0

        for path in (".env", ".env.txt", ".env.local", "data/Train/x.jpg", "results/m/dev_1.partial/predictions.jsonl",
                     "src/__pycache__/x.pyc", ".venv/x"):
            self.assertTrue(ignored(path), f"{path} must be ignored by Git")
        for path in ("README.md", "results/m/dev_1/report.md", "results/EXPERIMENT_LOG.md", "configs/gemini-3.8-flash.toml"):
            self.assertFalse(ignored(path), f"{path} must not be ignored by Git")

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a Git checkout")
    def test_no_key_or_unfinished_run_is_tracked(self):
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
        self.assertEqual([p for p in tracked if p.startswith(".env") or ".partial/" in p], [])
        for path in tracked:  # a Google key looks like "AIza" followed by 35 characters
            if path and (ROOT / path).suffix in {".py", ".md", ".toml", ".txt", ".csv", ".json", ".jsonl"}:
                self.assertIsNone(re.search(r"AIza[0-9A-Za-z_-]{35}", (ROOT / path).read_text(encoding="utf-8")), f"key-like text in {path}")

    def test_every_imported_package_is_pinned_in_requirements(self):
        pins = {line.split("==")[0].strip().lower() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                if "==" in line}
        distribution = {"PIL": "pillow", "google": "google-genai", "httpx": "httpx"}  # import name -> package name
        imported = set()
        for path in (ROOT / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imported |= {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])
        outside = sorted(imported - set(sys.stdlib_module_names) - {"src"})
        self.assertEqual([name for name in outside if name not in distribution], [], "add this package to the map in this test and to requirements.txt")
        for name in outside:
            self.assertIn(distribution[name], pins, f"{name} is imported but {distribution[name]} is not pinned in requirements.txt")

    def test_text_files_use_unix_line_endings(self):
        for path in [ROOT / "README.md", ROOT / "configs" / "gemini-3.8-flash.toml", ROOT / "prompts" / "diagnosis_v1.txt"]:
            self.assertNotIn(b"\r", path.read_bytes(), f"{path.name} has Windows line endings (.gitattributes should prevent this)")


if __name__ == "__main__":
    unittest.main()
