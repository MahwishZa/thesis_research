"""System tests: the whole committed research folder as it is, with no mock. The cleaned tables and the three views (every case once,
every question its case, no label in an input, the views rebuilt byte for byte), the balanced test split, the archives, the fixed
top-level structure, every command-line script answering --help, and a free dry run of run.py on the committed questions.
They skip when the tables are not in data/. Set THESIS_DATA to check another copy of the data folder.
Run: python -m unittest discover -s tests"""

import csv
import filecmp
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
import build as bd  # noqa: E402
import check as cp  # noqa: E402
import common  # noqa: E402
import evaluate as ev  # noqa: E402
from common import ALL_CASES, PROCESSED_CASES, SPLITS, VIEWS, question_file, read_csv, view_table  # noqa: E402

DATA = Path(os.environ.get("THESIS_DATA", ROOT / "data"))


# ----------------------------------------------------------------------------------------------------------------------
# the committed tables and views (data/processed_cases, data/views)
# ----------------------------------------------------------------------------------------------------------------------


LEAKY = ("confirm_type", "diagnosis_detail", "source", "label_source", "folder_labels", "orig_width", "orig_height", "orig_kb")


CLASS_WORDS = ("melanoma", "nevus", "carcinoma", "keratosis", "dermatofibroma", "vascular")


def rows(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


@unittest.skipUnless((DATA / ALL_CASES).is_file(), "the dataset tables are not in data/ yet")
class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = {r["isic_id"]: r for r in rows(ALL_CASES)}
        cls.labels = json.loads((DATA / PROCESSED_CASES / "classes.json").read_text(encoding="utf-8"))

    def test_every_case_is_once_in_one_split_with_a_known_label(self):
        ids = [r["isic_id"] for r in rows(ALL_CASES)]
        self.assertEqual(len(ids), len(set(ids)), "a case is listed twice")
        for r in self.master.values():
            self.assertIn(r["split"], SPLITS)
            self.assertIn(r["label"], self.labels)
            self.assertEqual(r["cv_fold"] == "-1", r["split"] == "test", f"{r['isic_id']}: the test split must be outside the folds")

    def test_no_lesion_group_is_in_two_splits(self):
        splits = {}
        for r in self.master.values():
            splits.setdefault(r["group_id"], set()).add(r["split"])
        self.assertEqual([g for g, s in splits.items() if len(s) > 1], [])

    def test_the_three_views_hold_the_same_cases_splits_and_labels(self):
        for view in VIEWS:
            with self.subTest(view):
                table = {r["isic_id"]: r for r in rows(view_table(view))}
                self.assertEqual(set(table), set(self.master))
                for i, r in table.items():
                    self.assertEqual((r["split"], r["label"]), (self.master[i]["split"], self.master[i]["label"]), i)

    def test_every_question_is_a_case_of_its_split_and_its_answer_is_the_label(self):
        for view in VIEWS:
            for split in SPLITS:
                with self.subTest(f"{view} {split}"):
                    items = jsonl(question_file(view, split))
                    ids = [q["id"] for q in items]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(set(ids), {i for i, r in self.master.items() if r["split"] == split})
                    for q in items:
                        self.assertEqual(q["options"][q["answer"]], q["answer_text"], q["id"])
                        self.assertEqual(q["gold_label"], self.master[q["id"]]["label"], q["id"])
                        self.assertFalse(any(w in q["question"].lower() for w in CLASS_WORDS), f"{q['id']}: the question names a class")
                        self.assertEqual("image_path" in q, view != "text_only", f"{q['id']}: image path and view disagree")
                        if view == "text_only":
                            self.assertNotIn("image", q["question"].lower(), f"{q['id']}: a text-only question mentions an image")

    def test_the_balanced_lists_use_training_cases_only_with_equal_classes(self):
        train = {i for i, r in self.master.items() if r["split"] == "train"}
        for name in ("train_oversampled.csv", "train_undersampled.csv"):
            with self.subTest(name):
                table = rows(f"{PROCESSED_CASES}/{name}")
                self.assertTrue({r["isic_id"] for r in table} <= train)
                self.assertTrue(all(r["label"] == self.master[r["isic_id"]]["label"] for r in table))
                self.assertEqual(len(set(Counter(r["label"] for r in table).values())), 1, "classes are not equal")

    def test_the_class_weights_are_from_the_training_counts(self):
        counts = Counter(r["label"] for r in self.master.values() if r["split"] == "train")
        balance = json.loads((DATA / PROCESSED_CASES / "class_weights.json").read_text(encoding="utf-8"))
        self.assertEqual(balance["class_counts_train"], dict(counts))

    def test_every_original_file_in_the_audit_is_a_kept_case_or_a_marked_copy_of_one(self):
        for r in rows(f"{PROCESSED_CASES}/raw_files_audit.csv"):
            if r["isic_id"] not in self.master:
                self.assertTrue(r["split"].startswith("removed: same picture as "), r["isic_id"])
                self.assertIn(r["split"].removeprefix("removed: same picture as "), self.master, r["isic_id"])

    def test_the_audit_lists_all_2357_files_of_the_kaggle_download(self):
        self.assertEqual(len(rows(f"{PROCESSED_CASES}/raw_files_audit.csv")), 2357)

    def test_the_answer_options_are_not_in_one_fixed_order(self):
        orders = {json.dumps(q["options"]) for q in jsonl(question_file("image_only", "test"))}
        self.assertGreater(len(orders), 1)

    def test_every_image_path_is_processed_images_slash_the_case_id(self):
        expected = {i: f"processed_images/{i}.jpg" for i in self.master}
        for name in (ALL_CASES, view_table("image_only"), view_table("multimodal")):
            with self.subTest(name):
                wrong = [r["isic_id"] for r in rows(name) if r["image_path"] != expected[r["isic_id"]]]
                self.assertEqual(wrong[:5], [], f"{len(wrong)} image paths are not processed_images/<id>.jpg")
        for view in ("image_only", "multimodal"):
            for split in SPLITS:
                with self.subTest(f"{view} {split}"):
                    wrong = [q["id"] for q in jsonl(question_file(view, split)) if q["image_path"] != expected[q["id"]]]
                    self.assertEqual(wrong[:5], [], f"{len(wrong)} image paths are not processed_images/<id>.jpg")

    def test_the_text_only_table_does_not_mention_an_image(self):
        self.assertFalse(any("image" in r["clinical_text"].lower() for r in rows(view_table("text_only"))))

    def test_each_view_holds_only_its_own_inputs_and_no_column_that_reveals_the_label(self):
        inputs = {"image_only": {"image_path"}, "text_only": {"clinical_text", "age", "sex", "site"},
                  "multimodal": {"image_path", "clinical_text", "age", "sex", "site"}}
        core = {"isic_id", "split", "cv_fold", "group_id", "label", "label_idx", "label_name"}
        for view in VIEWS:
            with self.subTest(view):
                columns = set(rows(view_table(view))[0])
                self.assertEqual(columns - core, inputs[view])
                self.assertFalse(columns & set(LEAKY))

    def test_no_table_lies_loose_in_the_data_folder(self):
        self.assertFalse(list(DATA.glob("*.csv")), "a table lies loose in data/: tables belong in processed_cases/ or views/")


@unittest.skipUnless((DATA / common.ALL_CASES).is_file(), "data/processed_cases/all_cases.csv is not there")
class ViewsRebuildTests(unittest.TestCase):
    def test_building_the_views_again_gives_the_committed_files_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, common.PROCESSED_CASES).mkdir()
            shutil.copy(DATA / common.ALL_CASES, Path(d, common.ALL_CASES))
            bd.build_views(d)
            names = [common.view_table(v) for v in common.VIEWS]
            names += [common.question_file(v, s) for v in common.VIEWS for s in common.SPLITS]
            different = [n for n in names if not filecmp.cmp(Path(d, n), DATA / n, shallow=False)]
            built = sorted(p.relative_to(d).as_posix() for p in Path(d, common.VIEWS_DIR).rglob("*") if p.is_file())
        self.assertEqual(different, [])
        self.assertEqual(built, sorted(names))                           # nothing else is written

    def test_an_unknown_class_stops_the_build(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, common.PROCESSED_CASES).mkdir()
            text = (DATA / common.ALL_CASES).read_text(encoding="utf-8").replace(",MEL,", ",XYZ,", 1)
            Path(d, common.ALL_CASES).write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                bd.build_views(d)


# ----------------------------------------------------------------------------------------------------------------------
# the balanced test split on the committed cases
# ----------------------------------------------------------------------------------------------------------------------


TABLE = DATA / ALL_CASES


def spec_counts(cs, new):
    return Counter((c["label"], new[c["isic_id"]]) for c in cs)


@unittest.skipUnless(TABLE.is_file(), "data/processed_cases/all_cases.csv is not there")
class BalancedTestSplitTests(unittest.TestCase):
    def test_the_committed_cases_give_30_test_images_of_every_class(self):
        cs = read_csv(TABLE)
        new = common.balanced_splits(cs)
        c = spec_counts(cs, new)
        labels = sorted({x["label"] for x in cs})
        self.assertEqual({lab: c[(lab, "test")] for lab in labels}, {lab: 30 for lab in labels})
        self.assertTrue(all(new[x["isic_id"]] == "val" for x in cs if x["split"] == "val"))
        self.assertEqual(sum(v == "val" for v in new.values()), sum(x["split"] == "val" for x in cs))
        groups = {}
        for x in cs:
            groups.setdefault(x["group_id"], set()).add(new[x["isic_id"]])
        self.assertTrue(all(len(s) == 1 for s in groups.values()))
        self.assertEqual(common.balanced_splits(cs), new)                                 # the same again


# ----------------------------------------------------------------------------------------------------------------------
# the committed questions with the metrics of evaluate.py
# ----------------------------------------------------------------------------------------------------------------------


class CommittedQuestionMetricTests(unittest.TestCase):
    def test_on_the_committed_test_questions_one_fixed_letter_is_about_chance(self):
        path = Path(os.environ.get("THESIS_DATA", ROOT / "data")) / "views" / "multimodal" / "test.jsonl"
        if not path.is_file():
            self.skipTest("data/views/ is not there")
        qa = ev.load(path)
        right = ev.diagnosis_metrics(qa, {i: {"pred": q["answer"]} for i, q in qa.items()})
        self.assertEqual(right["accuracy"], 1.0)
        self.assertEqual(sum(r["support"] for r in right["per_class"].values()), len(qa))
        always_a = ev.diagnosis_metrics(qa, {i: {"pred": "A"} for i in qa})
        self.assertLess(always_a["accuracy"], 0.25)  # "A" is a different class in each question
        self.assertGreater(sum(r["recall"] > 0 for r in always_a["per_class"].values()), 1)  # "A" picks several classes


# ----------------------------------------------------------------------------------------------------------------------
# the scripts as a user runs them, and the pipeline status of the committed data
# ----------------------------------------------------------------------------------------------------------------------
class CommandLineTests(unittest.TestCase):
    """Every script and command answers --help in a fresh Python process (no import problem, no missing package at the start)."""

    def run_src(self, *args):
        return subprocess.run([sys.executable, str(SRC / args[0]), *args[1:]], capture_output=True, text=True, cwd=ROOT, timeout=120)

    def test_every_command_answers_help_and_an_unknown_command_is_refused(self):
        commands = [("build.py",), ("build.py", "views"), ("build.py", "corpus"), ("build.py", "references"), ("check.py",),
                    ("check.py", "features"), ("run.py",), ("run.py", "llm-only"), ("evaluate.py",), ("evaluate.py", "compare")]
        for command in commands:
            with self.subTest(" ".join(command)):
                if command == ("check.py", "features") and not importlib.util.find_spec("pandas"):
                    self.skipTest("the image packages are not installed")
                done = self.run_src(*command, "--help")
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertTrue((done.stdout + done.stderr).strip(), "no help text")
        for script in ("build.py", "run.py"):
            self.assertEqual(self.run_src(script, "nonsense").returncode, 2)

    def test_the_six_scripts_are_the_only_ones_and_the_old_names_are_gone(self):
        self.assertEqual(sorted(p.stem for p in SRC.glob("*.py")), ["build", "check", "common", "evaluate", "preprocess", "run"])


@unittest.skipUnless((DATA / ALL_CASES).is_file(), "the dataset tables are not in data/ yet")
class PipelineStatusTests(unittest.TestCase):
    def test_no_data_step_of_the_committed_folder_has_a_problem(self):
        for name, status, note in cp.check(str(DATA), str(ROOT / "results")):
            if not name.startswith("llm_only"):
                with self.subTest(name):
                    self.assertNotEqual(status, cp.PROBLEM, note)

    def test_a_dry_run_shows_the_prompt_without_any_key(self):
        done = subprocess.run([sys.executable, str(SRC / "run.py"), "llm-only", "--model", "gpt-4o", "--view", "text_only",
                               "--split", "val", "--dry-run", "--limit", "1", "--data", str(DATA), "--results", tempfile.gettempdir()],
                              capture_output=True, text=True, cwd=ROOT, timeout=120, env={**os.environ, "OPENROUTER_API_KEY": ""})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[user] Patient:", done.stdout)
        self.assertIn("Answer: <the letters of the three most likely options", done.stdout)


# ----------------------------------------------------------------------------------------------------------------------
# the archives and the fixed top-level structure
# ----------------------------------------------------------------------------------------------------------------------


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


V1_ARCHIVE = ROOT / "archive" / "v1"


V1_COMMIT = "a1c9209"


FOLDERS = ("data", "results", "docs", "configs", "prompts")


BANNER = ("> **Version 1** (commit `a1c9209`), archived on 2026-11-19: a frozen copy kept for reference. "
          "It does not bind Version 2 (see `docs/v2_proposal.md` in the repository).")


LABELLED = {"data/image/datasheet.md", "data/image_metadata/datasheet.md", "data/text/datasheet.md", "data/cases/datasheet.md",
            "data/knowledge/datasheet.md", "docs/method.md", "docs/results.md"}


def v1_history_available() -> bool:
    return bool(shutil.which("git")) and (ROOT / ".git").exists() and git("cat-file", "-e", f"{V1_COMMIT}^{{commit}}").returncode == 0


def unlabelled(path: str, content: bytes) -> bytes:
    """A labelled document without its label: the banner line and the blank line after it."""
    head = (BANNER + "\n\n").encode("utf-8")
    if path in LABELLED:
        if not content.startswith(head):
            raise AssertionError(f"archive/v1/{path} does not start with the Version 1 label")
        return content[len(head):]
    return content


class ArchiveV1Tests(unittest.TestCase):
    def test_the_archive_has_its_readme_and_the_labelled_documents(self):
        self.assertTrue((V1_ARCHIVE / "README.md").is_file())
        readme = (V1_ARCHIVE / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"commit `{V1_COMMIT}`", readme)
        for path in LABELLED:
            self.assertIn(f"`{path}`", readme)
            self.assertTrue((V1_ARCHIVE / path).read_text(encoding="utf-8").startswith(BANNER + "\n\n"), path)

    @unittest.skipUnless(v1_history_available(), f"needs a Git checkout that holds commit {V1_COMMIT}")
    def test_every_archived_file_is_the_file_of_the_version_1_commit(self):
        listed = git("ls-tree", "-r", "--name-only", V1_COMMIT, "--", *FOLDERS)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        expected = sorted(p for p in listed.stdout.decode("utf-8").splitlines() if p)
        archived = sorted(p.relative_to(V1_ARCHIVE).as_posix() for p in V1_ARCHIVE.rglob("*") if p.is_file() and p != V1_ARCHIVE / "README.md")
        self.assertEqual(archived, expected, "files missing from the archive, or added to it")
        for path in expected:
            with self.subTest(path=path):
                original = git("show", f"{V1_COMMIT}:{path}")
                self.assertEqual(original.returncode, 0, original.stderr)
                self.assertEqual(unlabelled(path, (V1_ARCHIVE / path).read_bytes()), original.stdout,
                                 f"archive/v1/{path} differs from {path} at commit {V1_COMMIT}")

    def test_a_label_without_its_blank_line_or_an_unlabelled_document_is_caught(self):
        with self.assertRaises(AssertionError):
            unlabelled("docs/results.md", b"# Closed-book results\n")
        with self.assertRaises(AssertionError):
            unlabelled("docs/results.md", (BANNER + "\n# x\n").encode("utf-8"))
        self.assertEqual(unlabelled("docs/results.md", (BANNER + "\n\n# x\n").encode("utf-8")), b"# x\n")
        self.assertEqual(unlabelled("configs/models.toml", b"a = 1\n"), b"a = 1\n")


V2_ARCHIVE = ROOT / "archive" / "v2"


V2_COMMIT = "95a9506"


ADDED = ("ARCHIVE.md", "MANIFEST.sha256", "claude_rules_2026-11-20.md")  # in the archive, not in the commit


NOT_ARCHIVED = (".claude/", ".gitignore", ".gitattributes", "archive/")  # stay where they are


def v2_history_available() -> bool:
    return bool(shutil.which("git")) and (ROOT / ".git").exists() and git("cat-file", "-e", f"{V2_COMMIT}^{{commit}}").returncode == 0


def tracked(folder: str) -> list[str]:
    return [p for p in git("ls-files", "-z", folder).stdout.decode("utf-8").split("\0") if p]


TOP_LEVEL = {".claude", ".gitattributes", ".gitignore", "README.md", "archive", "data", "requirements.txt", "results", "src", "tests"}


class WorkspaceTests(unittest.TestCase):
    def test_the_top_level_structure_is_the_one_the_researcher_fixed_until_the_research_is_final(self):
        """Decided 2026-11-20 (.claude/CLAUDE.md): no top-level file or folder is added, renamed or removed; new material goes inside."""
        if shutil.which("git") and (ROOT / ".git").exists():
            held = {p.split("/")[0] for p in tracked(".")}
            self.assertEqual(sorted(held ^ TOP_LEVEL), [], "the tracked top level differs from the fixed structure")
        else:
            present = {p.name for p in ROOT.iterdir() if p.name not in {".git", ".ruff_cache", "__pycache__", ".venv", "venv", ".env"}}
            self.assertEqual(sorted(present - TOP_LEVEL), [])

    def test_the_readme_holds_only_the_overview_and_the_objectives(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(re.findall(r"^#{2,3} (.+)$", readme, re.M), ["Overview", "Objectives"])
        self.assertEqual(len(re.findall(r"^\d+\. ", readme, re.M)), 2)  # the two objectives


class ArchiveV2Tests(unittest.TestCase):
    def test_every_file_of_the_archive_is_in_its_manifest_with_the_same_hash_and_nothing_else_is_there(self):
        listed = {}
        for line in (V2_ARCHIVE / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
            digest, path = line.split("  ", 1)
            listed[path] = digest
        for path, digest in listed.items():
            with self.subTest(path):
                self.assertEqual(hashlib.sha256((V2_ARCHIVE / path).read_bytes()).hexdigest(), digest)
        if shutil.which("git") and (ROOT / ".git").exists():
            held = {p[len("archive/v2/"):] for p in tracked("archive/v2")} - set(ADDED)
            self.assertEqual(sorted(held - set(listed)), [], "files in the archive that the manifest does not list")
            self.assertEqual(sorted(set(listed) - held), [], "files the manifest lists that are not in the archive")

    @unittest.skipUnless(v2_history_available(), "the Git history of the archived commit is not available here")
    def test_the_archive_is_the_research_of_the_commit_file_for_file(self):
        names = [n for n in git("ls-tree", "-r", "-z", "--name-only", V2_COMMIT).stdout.decode("utf-8").split("\0") if n]
        moved = [n for n in names if not n.startswith(NOT_ARCHIVED)]
        self.assertGreater(len(moved), 200)
        for name in moved:
            content = git("show", f"{V2_COMMIT}:{name}").stdout
            self.assertTrue((V2_ARCHIVE / name).is_file(), f"missing from the archive: {name}")
            self.assertEqual((V2_ARCHIVE / name).read_bytes(), content, f"differs from the commit: {name}")
