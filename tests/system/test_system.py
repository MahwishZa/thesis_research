"""System tests: the whole committed research folder as it is, with no mock. The cleaned tables and the three views (every case once,
every question its case, no label in an input, the views rebuilt byte for byte), the balanced test split, the fixed
top-level structure, every command answering --help, and a free dry run of the LLM-only system on the committed questions.
They skip when the tables are not in data/. Set THESIS_DATA to check another copy of the data folder.
Run: python -m unittest discover -s tests -t ."""

import csv
import filecmp
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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
# ruff: noqa: E402  (the imports below need the line above)
from src import main as src_main
from src.data import dataset as data_dataset
from src.data import loaders as data_loaders
from src.data import views as data_views
from src.data.loaders import question_file, view_table
from src.evaluation import diagnostic as evaluation_diagnostic
from src.evaluation import evaluator as evaluation_evaluator
from src.utils import common as utils_common
from src.utils import status as utils_status
from src.utils.common import ALL_CASES, PROCESSED_CASES, SPLITS, VIEWS
from src.utils.io import read_csv

DATA = Path(os.environ.get("THESIS_DATA", ROOT / "data"))


# ----------------------------------------------------------------------------------------------------------------------
# the committed tables and views (data/processed/cases, data/views)
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

    def test_every_image_path_is_processed_images_folder_slash_the_case_id(self):
        expected = {i: f"processed/images/{i}.jpg" for i in self.master}
        for name in (ALL_CASES, view_table("image_only"), view_table("multimodal")):
            with self.subTest(name):
                wrong = [r["isic_id"] for r in rows(name) if r["image_path"] != expected[r["isic_id"]]]
                self.assertEqual(wrong[:5], [], f"{len(wrong)} image paths are not processed/images/<id>.jpg")
        for view in ("image_only", "multimodal"):
            for split in SPLITS:
                with self.subTest(f"{view} {split}"):
                    wrong = [q["id"] for q in jsonl(question_file(view, split)) if q["image_path"] != expected[q["id"]]]
                    self.assertEqual(wrong[:5], [], f"{len(wrong)} image paths are not processed/images/<id>.jpg")

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
        self.assertFalse(list(DATA.glob("*.csv")), "a table lies loose in data/: tables belong in processed/cases/ or views/")


@unittest.skipUnless((DATA / utils_common.ALL_CASES).is_file(), "data/processed/cases/all_cases.csv is not there")
class ViewsRebuildTests(unittest.TestCase):
    def test_building_the_views_again_gives_the_committed_files_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, utils_common.PROCESSED_CASES).mkdir(parents=True)
            shutil.copy(DATA / utils_common.ALL_CASES, Path(d, utils_common.ALL_CASES))
            data_views.build_views(d)
            names = [data_loaders.view_table(v) for v in utils_common.VIEWS]
            names += [data_loaders.question_file(v, s) for v in utils_common.VIEWS for s in utils_common.SPLITS]
            different = [n for n in names if not filecmp.cmp(Path(d, n), DATA / n, shallow=False)]
            built = sorted(p.relative_to(d).as_posix() for p in Path(d, utils_common.VIEWS_DIR).rglob("*") if p.is_file())
        self.assertEqual(different, [])
        self.assertEqual(built, sorted(names))                           # nothing else is written

    def test_an_unknown_class_stops_the_build(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, utils_common.PROCESSED_CASES).mkdir(parents=True)
            text = (DATA / utils_common.ALL_CASES).read_text(encoding="utf-8").replace(",MEL,", ",XYZ,", 1)
            Path(d, utils_common.ALL_CASES).write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                data_views.build_views(d)


# ----------------------------------------------------------------------------------------------------------------------
# the balanced test split on the committed cases
# ----------------------------------------------------------------------------------------------------------------------


TABLE = DATA / ALL_CASES


def spec_counts(cs, new):
    return Counter((c["label"], new[c["isic_id"]]) for c in cs)


@unittest.skipUnless(TABLE.is_file(), "data/processed/cases/all_cases.csv is not there")
class BalancedValidationRuleTests(unittest.TestCase):
    """The rule for the validation split (20 per class), applied to the committed cases after the test split is balanced: it can
    reach exactly 20 for every class by whole lesions, and it leaves the test split alone."""

    def test_20_validation_images_of_every_class_are_reachable_and_test_is_untouched(self):
        cs = read_csv(TABLE)
        new = data_dataset.balanced_splits(cs, data_dataset.VAL_PER_CLASS, split="val")
        count = spec_counts(cs, new)
        labels = sorted({x["label"] for x in cs})
        self.assertEqual({lab: count[(lab, "val")] for lab in labels}, {lab: 20 for lab in labels})
        self.assertEqual({x["isic_id"] for x in cs if x["split"] == "test"}, {i for i, v in new.items() if v == "test"})
        groups = {}
        for x in cs:
            groups.setdefault(x["group_id"], set()).add(new[x["isic_id"]])
        self.assertTrue(all(len(v) == 1 for v in groups.values()))
        self.assertEqual(Counter(new.values()), {"train": len(cs) - 240 - 160, "val": 160, "test": 240})
        self.assertEqual({i: v for i, v in new.items()}, {x["isic_id"]: x["split"] for x in cs})     # the committed tables already are this result


@unittest.skipUnless(TABLE.is_file(), "data/processed/cases/all_cases.csv is not there")
class BalancedTestSplitTests(unittest.TestCase):
    def test_the_committed_cases_give_30_test_images_of_every_class(self):
        cs = read_csv(TABLE)
        new = data_dataset.balanced_splits(cs)
        c = spec_counts(cs, new)
        labels = sorted({x["label"] for x in cs})
        self.assertEqual({lab: c[(lab, "test")] for lab in labels}, {lab: 30 for lab in labels})
        self.assertTrue(all(new[x["isic_id"]] == "val" for x in cs if x["split"] == "val"))
        self.assertEqual(sum(v == "val" for v in new.values()), sum(x["split"] == "val" for x in cs))
        groups = {}
        for x in cs:
            groups.setdefault(x["group_id"], set()).add(new[x["isic_id"]])
        self.assertTrue(all(len(s) == 1 for s in groups.values()))
        self.assertEqual(data_dataset.balanced_splits(cs), new)                                 # the same again


# ----------------------------------------------------------------------------------------------------------------------
# the committed questions with the evaluation metrics
# ----------------------------------------------------------------------------------------------------------------------


class CommittedQuestionMetricTests(unittest.TestCase):
    def test_on_the_committed_test_questions_one_fixed_letter_is_about_chance(self):
        path = Path(os.environ.get("THESIS_DATA", ROOT / "data")) / "views" / "multimodal" / "test.jsonl"
        if not path.is_file():
            self.skipTest("data/views/ is not there")
        qa = evaluation_evaluator.load(path)
        right = evaluation_diagnostic.diagnosis_metrics(qa, {i: {"pred": q["answer"]} for i, q in qa.items()})
        self.assertEqual(right["accuracy"], 1.0)
        self.assertEqual(sum(r["support"] for r in right["per_class"].values()), len(qa))
        always_a = evaluation_diagnostic.diagnosis_metrics(qa, {i: {"pred": "A"} for i in qa})
        self.assertLess(always_a["accuracy"], 0.25)  # "A" is a different class in each question
        self.assertGreater(sum(r["recall"] > 0 for r in always_a["per_class"].values()), 1)  # "A" picks several classes


# ----------------------------------------------------------------------------------------------------------------------
# the scripts as a user runs them, and the pipeline status of the committed data
# ----------------------------------------------------------------------------------------------------------------------
class CommandLineTests(unittest.TestCase):
    """Every script and command answers --help in a fresh Python process (no import problem, no missing package at the start)."""

    def run_main(self, *args):
        return subprocess.run([sys.executable, "-m", "src.main", *args], capture_output=True, text=True, cwd=ROOT, timeout=120)

    def test_every_command_answers_help_and_an_unknown_command_is_refused(self):
        needs_images = {"preprocess", "features"}                      # cv2, pandas, scikit-learn
        needs_more = {"visual-calibrate", "visual-index"}               # numpy only, but checked like the others
        for name in src_main.COMMANDS:
            with self.subTest(name):
                if name in needs_images and not importlib.util.find_spec("cv2"):
                    continue
                if name in needs_more and not importlib.util.find_spec("numpy"):
                    continue
                done = self.run_main(name, "--help")
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertTrue((done.stdout + done.stderr).strip(), "no help text")
        self.assertEqual(self.run_main("nonsense").returncode, 2)
        self.assertEqual(self.run_main("--help").returncode, 0)

    def test_the_one_entry_point_and_the_packages_are_the_only_code_and_the_old_scripts_are_gone(self):
        """The 2026-11-25 restructuring: src/main.py is the only script; the code is in the seven packages (researcher's approval)."""
        self.assertEqual(sorted(p.name for p in SRC.glob("*.py")), ["__init__.py", "main.py"])
        self.assertEqual(sorted(p.name for p in SRC.iterdir() if p.is_dir() and not p.name.startswith((".", "__"))),
                         ["data", "evaluation", "knowledge", "models", "pipelines", "retrieval", "utils"])


@unittest.skipUnless((DATA / ALL_CASES).is_file(), "the dataset tables are not in data/ yet")
class PipelineStatusTests(unittest.TestCase):
    def test_no_data_step_of_the_committed_folder_has_a_problem(self):
        for name, status, note in utils_status.check(str(DATA), str(ROOT / "results")):
            if not name.startswith("llm_only"):
                with self.subTest(name):
                    self.assertNotEqual(status, utils_status.PROBLEM, note)

    def test_a_dry_run_shows_the_prompt_without_any_key(self):
        done = subprocess.run([sys.executable, "-m", "src.main", "llm-only", "--model", "gpt-4o", "--view", "text_only",
                               "--split", "val", "--dry-run", "--limit", "1", "--data", str(DATA), "--results", tempfile.gettempdir()],
                              capture_output=True, text=True, cwd=ROOT, timeout=120, env={**os.environ, "OPENROUTER_API_KEY": ""})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[user] Patient:", done.stdout)
        self.assertIn("Reasoning and evidence: <at most three sentences", done.stdout)
        self.assertIn("Answer: <the letter of the most likely option> (<the name of that option>)", done.stdout)


# ----------------------------------------------------------------------------------------------------------------------
# the fixed top-level structure
# ----------------------------------------------------------------------------------------------------------------------


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


def tracked(folder: str) -> list[str]:
    return [p for p in git("ls-files", "-z", folder).stdout.decode("utf-8").split("\0") if p]


TOP_LEVEL = {".claude", ".gitattributes", ".gitignore", "README.md", "configs", "data", "requirements.txt", "results", "src", "tests"}
NOT_YET = {"configs"}        # approved on 2026-11-25; Git keeps no empty folder, so it appears with its first settings file


class LineEndingTests(unittest.TestCase):
    def test_every_text_file_the_code_writes_has_lf_line_endings_on_windows_too(self):
        """Approved 2026-11-25: on Windows, open(..., "w") writes CRLF, so a file's fingerprint differed from the same file in Git.
        Every open() that writes text must say newline= (the helpers in src/utils/io.py write LF)."""
        import ast
        missing = []
        for path in sorted(SRC.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open" and len(node.args) > 1:
                    mode = node.args[1]
                    if (isinstance(mode, ast.Constant) and isinstance(mode.value, str) and set(mode.value) & set("wa")
                            and "b" not in mode.value and not any(k.arg == "newline" for k in node.keywords)):
                        missing.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
                if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "to_csv"
                        and not any(k.arg == "lineterminator" for k in node.keywords)):
                    missing.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno} (pandas to_csv)")
        self.assertEqual(missing, [])


class WorkspaceTests(unittest.TestCase):
    def test_the_top_level_structure_is_the_one_the_researcher_fixed_until_the_research_is_final(self):
        """.claude/CLAUDE.md: no top-level file or folder is added, renamed or removed; new material goes inside. The folder archive/ was
        removed on 2026-11-24 at the researcher's request; configs/ was added to the structure on 2026-11-25 (researcher's approval)."""
        if shutil.which("git") and (ROOT / ".git").exists():
            held = {p.split("/")[0] for p in tracked(".")}
            self.assertEqual(sorted(held - TOP_LEVEL), [], "a tracked top-level file or folder is not in the fixed structure")
            self.assertEqual(sorted(TOP_LEVEL - NOT_YET - held), [], "a top-level file or folder of the fixed structure is missing")
        else:
            present = {p.name for p in ROOT.iterdir() if p.name not in {".git", ".ruff_cache", "__pycache__", ".venv", "venv", ".env"}}
            self.assertEqual(sorted(present - TOP_LEVEL), [])

    def test_the_tests_are_in_the_four_folders_the_researcher_chose(self):
        """Researcher's correction (2026-11-25): tests/unit, tests/system, tests/integration and tests/acceptance."""
        folders = sorted(p.name for p in (ROOT / "tests").iterdir() if p.is_dir() and not p.name.startswith((".", "__")))
        self.assertEqual(folders, ["acceptance", "integration", "system", "unit"])
        self.assertEqual(sorted(p.name for p in (ROOT / "tests").glob("*.py")), ["__init__.py"])
        for folder in folders:
            self.assertTrue((ROOT / "tests" / folder / "__init__.py").is_file(), folder)
            self.assertTrue(list((ROOT / "tests" / folder).glob("test_*.py")), folder)

    def test_the_data_folder_holds_the_approved_folders(self):
        """Approved 2026-11-25, with data/corpus/ simplified from data/knowledge/corpus/ the same day (researcher's request)."""
        if not (shutil.which("git") and (ROOT / ".git").exists()):
            self.skipTest("git is not available")
        self.assertEqual(sorted({p.split("/")[1] for p in tracked("data")}), ["annotations", "corpus", "datasheet.md", "processed", "views"])
        for constant in ("CORPUS_DIR", "PROCESSED_CASES", "OFFICIAL_LABELS", "VIEWS_DIR"):          # the code points to these folders
            self.assertTrue((ROOT / "data" / getattr(utils_common, constant)).is_dir(), constant)
        self.assertTrue((ROOT / "data" / utils_common.REFERENCE_TEXTS).is_file())

    def test_the_readme_holds_only_the_overview_and_the_objectives(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual(re.findall(r"^#{2,3} (.+)$", readme, re.M), ["Overview", "Objectives"])
        self.assertEqual(len(re.findall(r"^\d+\. ", readme, re.M)), 2)  # the two objectives
