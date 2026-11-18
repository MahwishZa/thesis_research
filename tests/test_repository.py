"""The repository against itself: README, command line, configs and committed tables must agree; a drift fails here."""

import argparse
import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from src import datasets, evaluate, prompts, run


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------------------------------------------
# The repository against itself


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        previous = os.getcwd()
        os.chdir(ROOT)  # configs name their files relative to the repository folder
        self.addCleanup(os.chdir, previous)
        self.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_every_command_in_the_usage_documentation_is_accepted_by_the_command_line(self):
        for module, name in ((run, "run"), (evaluate, "evaluate")):
            usage = re.findall(rf"^\s+python -m src\.{name} (.+?)$", module.__doc__, re.M)
            self.assertGreaterEqual(len(usage), 2)
            for line in usage:
                for form in (re.sub(r"\[[^\]]*\]", "", line), re.sub(r"\[([^\]]*)\]", r"\1", line)):  # without / with the optional [parts]
                    with self.subTest(form):
                        module.build_parser().parse_args(shlex.split(form))
            commands = next(a for a in module.build_parser()._actions if isinstance(a, argparse._SubParsersAction)).choices
            for command in commands:
                self.assertIn(f"python -m src.{name} {command}", module.__doc__, f"the usage documentation omits '{command}'")

    def test_the_commands_written_in_the_readme_are_accepted_by_the_command_line(self):
        found = re.findall(r"`python -m src\.(run|evaluate) ([^`]+)`", self.readme)
        self.assertTrue(found)
        for name, command in found:
            with self.subTest(command):
                args = {"run": run, "evaluate": evaluate}[name].build_parser().parse_args(shlex.split(command))
                if name == "run":
                    run.load_config(args.model, args.dataset, args.approach)  # a model of the list, a dataset and an approach
                for folder in (getattr(args, key, None) for key in ("run_dir", "run_a", "run_b")):
                    if folder:
                        self.assertTrue((ROOT / folder / "run_card.json").is_file(), f"the README names a run that does not exist: {folder}")

    def test_the_readme_has_its_sections_in_order_and_its_structure_lists_real_paths_and_every_top_level_folder(self):
        self.assertEqual(re.findall(r"^## (.+)$", self.readme, re.M),
                         ["Overview", "Objectives", "Results", "Repository Structure", "How to Run"])
        block = re.search(r"## Repository Structure\n\n```\n(.*?)\n```", self.readme, re.S).group(1).splitlines()
        self.assertEqual(block[0], "thesis_research/")
        parents, listed = [], set()
        for line in block[1:]:
            entry = re.fullmatch(r"((?:│   |    )*)(?:├── |└── )(\S+)(?:\s{2,}.+)?", line)
            self.assertIsNotNone(entry, f"not a line of a folder tree: {line}")
            level, name = len(entry.group(1)) // 4, entry.group(2).rstrip("/")
            parents = parents[:level] + [name]
            self.assertTrue(ROOT.joinpath(*parents).exists(), f"the README structure lists a path that does not exist: {'/'.join(parents)}")
            self.assertEqual(entry.group(2).endswith("/"), ROOT.joinpath(*parents).is_dir(), f"folders end with a slash, files do not: {line}")
            if level == 0:
                listed.add(name)
        if shutil.which("git") and (ROOT / ".git").exists():
            tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
            folders = {p.split("/")[0] for p in tracked if "/" in p and not p.startswith(".")}
            self.assertEqual(sorted(folders - listed), [], "the README structure omits a top-level folder")

    def test_the_results_written_in_the_readme_match_the_saved_results(self):
        """The README gives balanced accuracy with its 95% interval for every finished test run of phase 1, once each."""
        section = re.search(r"## Results\n(.*?)\n## ", self.readme, re.S).group(1)
        lines = [line for line in section.splitlines() if line.startswith("|")]
        header = [cell.strip() for cell in lines[0].strip().strip("|").split("|")]
        self.assertEqual(header[0], "Model")
        titles = {title: dataset for dataset, title in datasets.TITLES.items()}
        columns = [titles[title] for title in header[1:]]
        names = {name: model for model, name in evaluate.MODEL_NAMES.items()}
        shown = set()
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            model = names[cells[0]]
            for dataset, cell in zip(columns, cells[1:]):
                with self.subTest((model, dataset)):
                    (folder,) = [p for p in (ROOT / "results" / "phase1" / dataset / model).glob("test_*") if not p.name.endswith(".partial")]
                    main = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))["main"]
                    low, high = main["balanced_accuracy_ci95"]
                    self.assertEqual(cell, f"{100 * main['balanced_accuracy']:.1f} ({100 * low:.1f} to {100 * high:.1f})")
                    shown.add(folder)
        finished = {p for p in (ROOT / "results" / "phase1").glob("*/*/test_*") if p.is_dir() and not p.name.endswith(".partial")}
        self.assertEqual(shown, finished, "every finished test run of phase 1 is in the README once")

    def test_every_model_of_the_list_can_run_every_dataset_and_approach_with_its_prompt(self):
        with (ROOT / run.MODELS).open("rb") as f:
            names = [entry["name"] for entry in tomllib.load(f)["models"]]
        self.assertEqual(names, list(evaluate.MODEL_NAMES), "the model list and the names in the tables must agree")
        for model in names:
            for dataset in datasets.DATASETS:
                for approach in run.APPROACHES:
                    with self.subTest((model, dataset, approach)):
                        config = run.load_config(model, dataset, approach)
                        prompts.load_template(ROOT / config["prompt"]["file"], dataset)

    def test_the_committed_dataset_tables_are_consistent(self):
        for dataset, kind in datasets.DATASETS.items():
            table = datasets.dataset_paths(dataset)[0]
            if table.exists():  # committed once the dataset is prepared; its files stay out of Git
                with self.subTest(dataset):
                    labels = datasets.classes(datasets.read_table(table, dataset))
                    self.assertEqual([c for c in kind.set_apart if c not in labels], [], "a class set apart must be a class")
                    self.assertTrue(all(item.group for item in datasets.read_table(table, dataset)), "every file needs its group")

    def test_every_prompt_file_is_a_template_with_one_label_placeholder_and_a_final_line_break(self):
        files = sorted((ROOT / "prompts").rglob("*"))
        self.assertEqual(sorted(p for p in files if p.is_file()), sorted(ROOT / run.prompt_file(d, a) for d in datasets.DATASETS for a in run.APPROACHES),
                         "one prompt per dataset and approach, and nothing else")
        for path in (p for p in files if p.is_file()):
            with self.subTest(path.name):
                prompts.load_template(path, path.parent.name)  # refuses a file without exactly one {labels}
                self.assertTrue(path.read_bytes().endswith(b"\n"))
                self.assertNotIn(b"\r", path.read_bytes(), "Windows line endings (.gitattributes should prevent this)")
                self.assertNotIn("\n\n", path.read_text(encoding="utf-8"), "a blank line would change what the model is shown")

    def test_every_imported_package_is_in_the_one_requirements_file_and_pinned(self):
        listed = {}  # package -> how it is constrained (every package must be an exact "==" pin)
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            if line := line.split("#")[0].strip():
                name, operator, _ = re.fullmatch(r"([A-Za-z0-9._-]+)(==|>=)(\S+)", line).groups()
                listed[name.lower()] = operator
        self.assertFalse(list(ROOT.glob("requirements*.txt"))[1:], "there must be one requirements file")
        distribution = {"PIL": "pillow", "anthropic": "anthropic", "openai": "openai", "cv2": "opencv-python-headless", "numpy": "numpy",
                        "torch": "torch", "open_clip": "open_clip_torch", "easyocr": "easyocr"}  # import name -> package name
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
            self.assertIn(distribution[name], listed, f"{name} is imported but {distribution[name]} is not in requirements.txt")
        for package, operator in listed.items():
            self.assertEqual(operator, "==", f"{package} must be pinned with ==")

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a Git checkout")
    def test_git_ignores_keys_images_unfinished_and_validation_runs_but_not_tables_and_test_results(self):
        def ignored(path):
            return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0

        for path in (".env", ".env.txt", ".env.local", "data/image/raw/Train/melanoma/x.jpg", "data/image/downloads/isic/ISIC_0000001.json",
                     "data/image/downloads/isic_pictures/ISIC_0000001.jpg", "data/image/dataset/test/nevus/x.jpg",
                     "data/image/dataset/x.csv", "data/text/sources/pmc_patients/PMC-Patients-V2.json",
                     "data/text/sources/pmc_patients/download.json", "data/text/sources/x.csv", "data/text/dataset/test/melanoma/1-1.txt",
                     "data/text/sources/pubmed/abstracts.jsonl", "data/text/sources/audit_sample.txt",
                     "data/knowledge/sources/statpearls/NBK1.html", "data/knowledge/texts/passages.jsonl", "data/anything_else.txt",
                     "data/cases/figures/images/PMC1/f1.jpg", "data/cases/figures/captions.jsonl", "data/text/sources/removed/test/x/1-1.txt",
                     "data/image/build/x.txt", "data/text/build/nested/x.csv",
                     "data/x.csv",
                     "results/image/closed_book/m/val_1/report.md", "results/image/closed_book/m/test_1.partial/predictions.jsonl",
                     "src/__pycache__/x.pyc", "results/text/closed_book/m/val_1/report.md",
                     "results/text/closed_book/m/test_1.partial/predictions.jsonl",
                     "src/build/image/__pycache__/x.pyc", ".venv/x", ".coverage"):
            self.assertTrue(ignored(path), f"{path} must be ignored by Git")
        for path in ("README.md", "data/image/dataset.csv", "data/image/build/raw_photos.csv", "data/image/dataset_report.md",
                     "data/image/datasheet.md", "data/text/dataset.csv", "data/text/build/case_candidates.csv", "data/text/build/article_mesh.csv",
                     "data/text/build/abstract_candidates.csv", "data/text/build/review_flags.csv", "data/image_metadata/dataset.csv",
                     "data/cases/dataset.csv", "data/cases/build/figure_checks.csv", "data/cases/datasheet.md", "data/knowledge/library.csv",
                     "data/knowledge/index.csv", "data/knowledge/dataset_report.md", "results/image/closed_book/m/test_1/report.md",
                     "results/image/closed_book/m/test_1/predictions.jsonl", "configs/models.toml", "prompts/image/closed_book.txt",
                     "results/text/closed_book/m/test_1/report.md", "results/phase1/text/m/test_1/report.md"):
            self.assertFalse(ignored(path), f"{path} must not be ignored by Git")

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a Git checkout")
    def test_no_key_image_or_unfinished_run_is_tracked(self):
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
        self.assertEqual([p for p in tracked if p.startswith(".env") or ".partial/" in p], [])
        self.assertEqual([p for p in tracked if Path(p).suffix.lower() in datasets.MEDIA_TYPES], [], "no image belongs in Git")
        key_formats = re.compile(r"AIza[0-9A-Za-z_-]{35}|\bsk-[A-Za-z0-9_-]{20,}|\bgsk_[A-Za-z0-9]{20,}|\bhf_[A-Za-z0-9]{30,}"
                                 r"|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}")  # Google, OpenAI-style, Groq, Hugging Face, GitHub
        for path in tracked:
            if path and (ROOT / path).suffix in {".py", ".md", ".toml", ".txt", ".csv", ".json", ".jsonl"} and (ROOT / path).is_file():
                self.assertIsNone(key_formats.search((ROOT / path).read_text(encoding="utf-8")), f"key-like text in {path}")

    @unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a Git checkout")
    def test_every_run_records_a_commit_that_is_part_of_the_history(self):
        """A run card names the Git commit it ran on. A commit that only exists as a leftover object on one computer (after a history
        rewrite) is not in a fresh clone, so the commit must be an ancestor of the checked-out one, not merely a known object."""
        if subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, capture_output=True, text=True).stdout.strip() == "true":
            self.skipTest("a shallow clone does not hold the whole history")
        cards = sorted(p for p in (ROOT / "results").rglob("run_card.json") if p.parent.name.startswith("test_"))
        self.assertTrue(cards)
        for path in cards:
            card = json.loads(path.read_text(encoding="utf-8"))
            commits = {s["commit"] for s in card["sessions"]} | {card["git"]["commit"]}
            for commit in commits:
                with self.subTest(f"{path.parent.relative_to(ROOT).as_posix()} {commit[:10]}"):
                    self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT).returncode, 0,
                                     "the commit named in this run card is not in the history")

    def test_the_results_document_holds_the_tables_the_summary_command_writes_and_all_its_links_work(self):
        text = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")
        for target in re.findall(r"\]\((?!https?:)([^)#]+)", text):
            self.assertTrue((ROOT / "docs" / target).exists(), f"docs/results.md links to {target}, which does not exist")
        start, end = text.index(evaluate.SUMMARY_START) + len(evaluate.SUMMARY_START), text.index(evaluate.SUMMARY_END)
        self.assertEqual(text[start:end].strip("\n").split("\n"), evaluate.summary_tables(ROOT / "results" / "phase1", "../"),
                         "docs/results.md is out of date: run python -m src.evaluate summary")
        runs = [p for p in (ROOT / "results" / "phase1").glob("*/*/test_*") if p.is_dir() and not p.name.endswith(".partial")]
        for finished in runs:  # every finished run is in the tables, once, with its report linked
            self.assertEqual(text.count(f"](../{finished.relative_to(ROOT).as_posix()}/report.md)"), 1, f"{finished} must be linked exactly once")
        for word in ("ROC", "Holm", "Specificity", "Malignancy sensitivity"):
            self.assertIn(word, text)

    def test_the_holm_correction_by_hand(self):
        # four tests: 0.01 x 4 = 0.04; 0.03 x 3 = 0.09; 0.04 x 2 = 0.08, raised to the 0.09 before it; 0.5 x 1
        self.assertEqual(evaluate.holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])
        self.assertEqual(evaluate.holm([0.6, 0.9]), [1.0, 1.0])  # never above 1
        self.assertEqual((evaluate.holm([]), evaluate.holm([0.2])), ([], [0.2]))

    def test_the_summary_command_rewrites_only_the_tables_between_the_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            page = Path(folder) / "docs" / "page.md"
            page.parent.mkdir()
            page.write_text(f"before\n{evaluate.SUMMARY_START}\nold\n{evaluate.SUMMARY_END}\nafter\n", encoding="utf-8")
            with mock.patch.object(evaluate, "summary_tables", return_value=["| a |", "| b |"]):
                evaluate.write_summary(page, ROOT / "results")
            self.assertEqual(page.read_text(encoding="utf-8"),
                             f"before\n{evaluate.SUMMARY_START}\n\n| a |\n| b |\n\n{evaluate.SUMMARY_END}\nafter\n")
            page.write_text("no markers\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must hold the lines"):
                evaluate.write_summary(page, ROOT / "results")

    def test_the_readme_uses_unix_line_endings(self):
        self.assertNotIn(b"\r", (ROOT / "README.md").read_bytes(), "Windows line endings (.gitattributes should prevent this)")


if __name__ == "__main__":
    unittest.main()
