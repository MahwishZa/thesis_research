"""The repository against itself: README, command line, configs and committed results must agree (a drift fails here)."""

import argparse
import ast
import csv
import json
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

    def test_every_command_in_the_usage_documentation_is_accepted_by_the_command_line(self):
        usage = re.findall(r"^\s+python -m src\.evaluate (.+?)(?:\s{2,}\(.*)?$", evaluate.__doc__, re.M)
        self.assertGreaterEqual(len(usage), 5)
        for line in usage:
            for form in (re.sub(r"\[[^\]]*\]", "", line), re.sub(r"\[([^\]]*)\]", r"\1", line)):  # without / with the optional [parts]
                with self.subTest(form):
                    evaluate.build_parser().parse_args(shlex.split(form))

    def test_the_commands_written_in_the_readme_are_accepted_by_the_command_line(self):
        commands = re.findall(r"`python -m src\.evaluate ([^`]+)`", self.readme)
        self.assertGreaterEqual(len(commands), 2)
        for command in commands:
            with self.subTest(command):
                args = evaluate.build_parser().parse_args(shlex.split(command))
                if hasattr(args, "config"):
                    self.assertTrue((ROOT / args.config).is_file(), f"the README names a config that does not exist: {args.config}")

    def test_the_results_written_in_the_documents_match_the_saved_results(self):
        def pct(x):
            return f"{100 * x:.1f}"

        rows = [line for line in self.readme.splitlines() if line.startswith("| ") and "](results/" in line]
        self.assertTrue(rows, "the README results table links to no run")
        for row in rows:
            link = re.search(r"\]\((results/[^)]+/report\.md)\)", row).group(1)
            self.assertTrue((ROOT / link).is_file(), f"the README links to a report that does not exist: {link}")
            m = json.loads((ROOT / link).with_name("metrics.json").read_text(encoding="utf-8"))
            main = m["corrected_labels"] or m["folder_labels"]
            low, high = main["accuracy_ci95"]
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            self.assertEqual(cells[1:4], [pct(main["accuracy"]), f"{pct(low)} to {pct(high)}", pct(m["folder_labels"]["accuracy"])])
        brief = (ROOT / "docs" / "meeting_brief.md").read_text(encoding="utf-8").replace("**", "")
        link = re.search(r"results/[^`\s]+/report\.md", brief).group(0)
        m = json.loads((ROOT / link).with_name("metrics.json").read_text(encoding="utf-8"))
        main = m["corrected_labels"] or m["folder_labels"]
        low, high = main["accuracy_ci95"]
        for scoring in (m["corrected_labels"], m["folder_labels"]):
            if scoring:
                self.assertIn(f"{pct(scoring['accuracy'])}% ({scoring['correct']}/{scoring['n']})", brief)
        self.assertIn(f"{pct(low)}% to {pct(high)}%", brief)

    def test_commands_and_modules_are_documented(self):
        parser = evaluate.build_parser()
        names = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices
        for name in names:
            self.assertIn(f"python -m src.evaluate {name}", evaluate.__doc__, f"the module docstring omits '{name}'")
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
        for path in ("README.md", "results/m/dev_1/report.md", "docs/EXPERIMENT_LOG.md", "configs/example.toml"):
            self.assertFalse(ignored(path), f"{path} must not be ignored by Git")

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a Git checkout")
    def test_no_key_or_unfinished_run_is_tracked(self):
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
        self.assertEqual([p for p in tracked if p.startswith(".env") or ".partial/" in p], [])
        key_formats = re.compile(r"AIza[0-9A-Za-z_-]{35}|\bsk-[A-Za-z0-9_-]{20,}|\bgsk_[A-Za-z0-9]{20,}|\bhf_[A-Za-z0-9]{30,}"
                                 r"|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}")  # Google, OpenAI-style, Groq, Hugging Face, GitHub
        for path in tracked:
            if path and (ROOT / path).suffix in {".py", ".md", ".toml", ".txt", ".csv", ".json", ".jsonl"}:
                self.assertIsNone(key_formats.search((ROOT / path).read_text(encoding="utf-8")), f"key-like text in {path}")

    def test_every_imported_package_is_in_the_one_requirements_file_and_pinned(self):
        listed = {}  # package -> how it is constrained ("==" exact pin, ">=" lowest version)
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            if line := line.split("#")[0].strip():
                name, operator, _ = re.fullmatch(r"([A-Za-z0-9._-]+)(==|>=)(\S+)", line).groups()
                listed[name.lower()] = operator
        self.assertFalse(list((ROOT).glob("requirements*.txt"))[1:], "there must be one requirements file")
        distribution = {"PIL": "pillow", "google": "google-genai", "httpx": "httpx", "torch": "torch",
                        "transformers": "transformers", "huggingface_hub": "huggingface-hub"}  # import name -> package name
        installed_first = {"torch"}  # from pytorch.org (the plain pip download for Windows has no GPU support); the run card records its version
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
            package = distribution[name]
            self.assertIn(package, listed, f"{name} is imported but {package} is not in requirements.txt")
            self.assertEqual(listed[package], ">=" if package in installed_first else "==", f"{package} is not constrained as expected")
        for package, operator in listed.items():
            self.assertTrue(operator == "==" or package in installed_first, f"{package} must be pinned with ==")

    def test_text_files_use_unix_line_endings(self):
        for path in [ROOT / "README.md", ROOT / "prompts" / "diagnosis_v1.txt"]:
            self.assertNotIn(b"\r", path.read_bytes(), f"{path.name} has Windows line endings (.gitattributes should prevent this)")


if __name__ == "__main__":
    unittest.main()
