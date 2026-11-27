"""User acceptance tests: the researcher's own requirements, one test for each, checked on what is committed. Each docstring says
where the requirement comes from (.claude/CLAUDE.md or a decision of the researcher). If one of these fails, the research no longer
does what the researcher asked, whatever the other tests say. They skip when the part they check is not in the folder.
Run: python -m unittest"""

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
# ruff: noqa: E402  (the imports below need the line above)
import src.evaluation.evaluator
import src.evaluation.metrics
import src.models.llm
import src.pipelines.base
from src.data.views import TEST_PER_CLASS, VAL_PER_CLASS, question_file
from src.utils import ALL_CASES, CLASSES, PROCESSED_CASES, REFERENCE_TEXTS, SPLITS, VIEWS

EVALUATION_MODULES = (src.evaluation.metrics, src.evaluation.evaluator)   # what evaluate.py held before the 2026-11-25 move

DATA = Path(os.environ.get("THESIS_DATA", ROOT / "data"))
HAVE_GIT = bool(shutil.which("git")) and (ROOT / ".git").exists()
TABLES = (DATA / ALL_CASES).is_file()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)


@unittest.skipUnless(TABLES, "the dataset tables are not in data/ yet")
class DatasetRequirements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import csv
        with (DATA / ALL_CASES).open(encoding="utf-8", newline="") as f:
            cls.cases = list(csv.DictReader(f))

    def test_only_the_kaggle_dataset_is_used_and_every_case_comes_from_its_2357_files(self):
        """The researcher's decision: only the Kaggle "Skin Cancer ISIC 9 classes" dataset (no HAM10000 or Derm7pt images)."""
        import csv
        with (DATA / PROCESSED_CASES / "raw_files_audit.csv").open(encoding="utf-8", newline="") as f:
            audit = list(csv.DictReader(f))
        self.assertEqual(len(audit), 2357)
        listed = {r["isic_id"] for r in audit}
        self.assertTrue({c["isic_id"] for c in self.cases} <= listed)
        self.assertEqual(len(self.cases), 2197)

    def test_there_are_exactly_the_eight_classes(self):
        """The 8 ISIC 2019 classes; the Kaggle seborrheic keratosis folder held no genuine images, so it is not a class."""
        self.assertEqual(sorted({c["label"] for c in self.cases}), sorted(CLASSES))
        self.assertEqual(len(CLASSES), 8)

    def test_the_test_and_validation_splits_are_balanced_and_no_lesion_is_in_two_splits(self):
        """Decided 2026-11-22: 30 test images and 20 validation images of every class; train keeps the natural mix; lesion groups
        never cross splits (no leakage)."""
        for split, size in (("test", TEST_PER_CLASS), ("val", VAL_PER_CLASS)):
            per_class = Counter(c["label"] for c in self.cases if c["split"] == split)
            self.assertEqual(dict(per_class), {c: size for c in CLASSES}, split)
        train = Counter(c["label"] for c in self.cases if c["split"] == "train")
        self.assertGreater(max(train.values()) / min(train.values()), 3)                 # train is not balanced by removal
        splits = defaultdict(set)
        for c in self.cases:
            splits[c["group_id"]].add(c["split"])
        self.assertEqual([g for g, s in splits.items() if len(s) > 1], [])

    def test_the_three_views_hold_the_same_cases_and_each_only_its_own_inputs(self):
        """Three views of the same cases: image only, text only, image + text (the text is age, sex and body site, nothing else)."""
        for split in SPLITS:
            ids = {v: [q["id"] for q in read_jsonl(DATA / question_file(v, split))] for v in VIEWS}
            self.assertEqual(sorted(ids["image_only"]), sorted(ids["text_only"]))
            self.assertEqual(sorted(ids["image_only"]), sorted(ids["multimodal"]))
        for view in VIEWS:
            q = read_jsonl(DATA / question_file(view, "test"))[0]
            self.assertEqual("image_path" in q, view != "text_only", view)
            self.assertEqual(re.search(r"\bPatient:", q["question"]) is not None, view != "image_only", view)
            self.assertEqual(len(q["options"]), 8)

    def test_the_question_files_are_named_train_validation_and_test(self):
        """Decided 2026-11-22: views/<view>/train.jsonl, val.jsonl, test.jsonl (no questions_ prefix); val.jsonl renamed validation.jsonl
        on 2026-11-25 (researcher's approval of the new structure)."""
        for view in VIEWS:
            self.assertEqual(sorted(p.name for p in (DATA / "views" / view).glob("*.jsonl")), ["test.jsonl", "train.jsonl", "validation.jsonl"])

    def test_the_2026_11_25_move_changed_only_the_image_paths_and_the_validation_file_names(self):
        """Researcher's approval of the new structure (2026-11-25): the move must not change any question, answer or option. Every
        file in data/views/provenance.json has its recorded SHA-256 now, and writing its image paths the old way
        (processed_images/) gives back exactly the bytes of the file before the move (its previous SHA-256)."""
        path = DATA / "views" / "provenance.json"
        if not path.is_file():
            self.skipTest("data/views/provenance.json is not there")
        files = read_json(path)["files"]
        self.assertEqual(len(files), 10)
        for name, record in files.items():
            with self.subTest(name):
                now = (DATA / name).read_bytes()
                self.assertEqual(hashlib.sha256(now).hexdigest(), record["sha256"])
                before = now.replace(b"processed/images/", b"processed_images/")
                self.assertEqual(hashlib.sha256(before).hexdigest(), record["previous_sha256"])
                self.assertEqual(record["previous_path"], name.replace("validation.jsonl", "val.jsonl").replace("processed/cases", "processed_cases"))

    def test_the_folders_have_the_names_the_researcher_chose(self):
        """raw, processed/images, processed/cases, raw_files_audit.csv and data/datasheet.md (decided 2026-11-22; moved under processed/
        on 2026-11-25)."""
        self.assertTrue((DATA / "datasheet.md").is_file())
        self.assertTrue((DATA / PROCESSED_CASES / "raw_files_audit.csv").is_file())
        self.assertTrue(re.search(r"/data/raw/", (ROOT / ".gitignore").read_text(encoding="utf-8")))
        self.assertTrue(re.search(r"/data/processed/images/", (ROOT / ".gitignore").read_text(encoding="utf-8")))


class LibraryIsNotSyntheticRequirements(unittest.TestCase):
    def test_no_module_that_builds_the_libraries_can_call_a_model(self):
        """The knowledge libraries hold only text and pictures that others published (researcher's rule, 2026-11-24: nothing
        synthetic). The builders (text library, image library, indexes, retrieval) import no language-model client."""
        import ast
        forbidden = ("anthropic", "openai", "src.models.llm", "src.pipelines")
        for name in ("corpus.py", "images.py", "indexer.py", "retriever.py"):
            tree = ast.parse(Path(SRC / "retrieval" / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = ([a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                for module in modules:
                    with self.subTest(f"{name}: {module}"):
                        self.assertFalse(module.startswith(forbidden), f"{name} imports {module}")


class ModelAndPromptRequirements(unittest.TestCase):
    def test_the_four_research_models_are_used_as_they_are_through_pinned_providers(self):
        """Claude Sonnet 5.5, Llama 4 Maverick, Gemini 3.8 Flash and GPT-4o; pre-trained, no fine-tuning; the provider is pinned and
        never replaced by another one (a changed provider would be a different model)."""
        self.assertEqual(sorted(src.models.llm.MODELS), ["claude-sonnet-5-5", "gemini-3.8-flash", "gpt-4o", "llama-4-maverick", "medgemma-1.5-4b-it", "qwen3.5-2b"])
        for name, cfg in src.models.llm.MODELS.items():
            with self.subTest(name):
                self.assertNotIn("fine_tuned", json.dumps(cfg).lower())
                if cfg["api"] == "local":
                    self.assertRegex(cfg["revision"], r"^[0-9a-f]{40}$", "the local model is pinned to one revision")
                if cfg["api"] == "openrouter":
                    self.assertEqual(len(cfg["route"]), 1, "the provider is pinned")
        self.assertIn("allow_fallbacks", Path(SRC / "models" / "llm.py").read_text(encoding="utf-8"))

    def test_every_model_gets_the_same_question_and_must_give_its_reasoning_and_evidence_and_one_letter(self):
        """The answer format (researcher's decision, 2026-11-23): "Reasoning and evidence: ..." naming the findings that support the
        choice, then "Answer: <letter> (<name of that option>)", e.g. "C (Melanoma)" (no top-3 list, no yes/no evidence flag); the prompt is fixed and versioned."""
        text = src.pipelines.base.INSTRUCTIONS["rationale"]
        self.assertIn("Reasoning and evidence: <at most three sentences", text)
        self.assertIn("Answer: <the letter of the most likely option> (<the name of that option>)", text)
        self.assertNotIn("Evidence sufficient", text)
        self.assertEqual(src.pipelines.base.PROMPT_NAME, "llm_only")


class MetricRequirements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = DATA / question_file("multimodal", "test")
        if not path.is_file():
            raise unittest.SkipTest("data/views/ is not there")
        cls.qa = src.evaluation.evaluator.load(path)
        cls.pred = {i: {"id": i, "pred": q["answer"], "raw_output": "", "finish_reason": "stop"} for i, q in cls.qa.items()}

    def test_the_approved_diagnosis_metrics_are_reported(self):
        """Approved: accuracy (top-1) with an interval, macro-F1, per-class precision / recall / specificity / F1, the confusion
        matrix; refusals apart from other invalid answers."""
        d = src.evaluation.metrics.diagnosis_metrics(self.qa, self.pred)
        for key in ("accuracy", "accuracy_95ci", "macro_f1", "macro_f1_95ci", "confusion_matrix",
                    "invalid_answers", "invalid_answers_by_cause", "missing_predictions"):
            self.assertIn(key, d)
        for row in d["per_class"].values():
            self.assertTrue({"precision", "recall", "specificity", "f1", "support"} <= set(row))
        self.assertEqual(set(d["per_class"]), set(CLASSES))

    def test_the_metrics_the_researcher_did_not_approve_are_not_reported(self):
        """Not approved: top-k accuracy (removed 2026-11-22: standard top-1 accuracy only), balanced accuracy and macro-recall (the
        same number on a balanced test split), BERTScore, a separate melanoma sensitivity, a score on seen questions."""
        text = json.dumps(src.evaluation.metrics.diagnosis_metrics(self.qa, self.pred)).lower()
        for banned in ("top3", "top_k", "topk", "balanced_accuracy", "macro_recall", "bertscore", "melanoma_sensitivity", "melanoma_recall"):
            self.assertNotIn(banned, text)

    def test_the_feature_check_reports_accuracy_not_balanced_accuracy(self):
        """Renamed 2026-11-24: the splits are balanced, so the plain accuracy is the one number."""
        source = (SRC / "data" / "preprocessing.py").read_text(encoding="utf-8")
        self.assertIn("accuracy_score", source)
        self.assertNotIn("balanced_accuracy", source)
        self.assertNotIn("bal_acc", source)
        self.assertNotIn("balanced accuracy", source)

    def test_an_answer_is_covered_only_when_its_evidence_quote_is_verified(self):
        """Option B (researcher's decision, 2026-11-23): for RAG answers the evidence replaces the yes/no flag. A correct answer with a
        quote that is not in the passage is not covered (an NLI support step and supported accuracy were removed on 2026-11-24)."""
        passage = "Melanomas show asymmetry and irregular borders here. Nevi are usually symmetric."
        good = {i: {"pred": q["answer"], "contexts": [{"chunk_id": "c", "text": passage, "tags": [q["gold_label"]]}],
                    "evidence": {"chunk_id": "c", "quote": "Melanomas show asymmetry and irregular borders here"}}
                for i, q in self.qa.items()}
        invented = {i: {**p, "evidence": {"chunk_id": "c", "quote": "this sentence is not in the passage at all"}} for i, p in good.items()}
        self.assertEqual(src.evaluation.metrics.evidence_metrics(self.qa, good)["coverage_verified"], 1.0)
        self.assertEqual(src.evaluation.metrics.evidence_metrics(self.qa, invented)["coverage_verified"], 0.0)
        self.assertIsNone(src.evaluation.metrics.evidence_metrics(self.qa, self.pred))                 # the LLM-only answers have no evidence to check

    def test_no_llm_judge_metric_is_offered(self):
        """Removed on the researcher's decision (2026-11-24): no RAGAS and no other LLM judge that costs credit."""
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            src.evaluation.evaluator.report_main(["--questions", "q", "--predictions", "p", "--ragas"])
        self.assertFalse(any("ragas" in n.lower() for m in EVALUATION_MODULES for n in dir(m)))

    def test_every_metric_family_names_its_source_in_evaluate_py(self):
        """Only established, correctly named metrics, each with its source (.claude/CLAUDE.md)."""
        doc = src.evaluation.evaluator.__doc__
        for source in ("McNemar (1947)", "Holm (1979)", "Geifman", "Jarvelin", "MedCPT"):
            self.assertIn(source, doc)


class IntegrityRequirements(unittest.TestCase):
    def test_the_reference_texts_are_quotes_from_a_cited_source_never_written_by_us(self):
        """Never invent data or sources: every reference text has its page, link and licence, and the link is Dermoscopedia."""
        path = DATA / REFERENCE_TEXTS
        if not path.is_file():
            self.skipTest("no reference texts yet")
        refs = read_json(path)
        self.assertEqual(sorted(refs), sorted(CLASSES))
        for cls, r in refs.items():
            with self.subTest(cls):
                self.assertTrue(r["texts"])
                self.assertTrue(r["url"].startswith("https://dermoscopedia.org/"))
                self.assertEqual(r["licence"], "CC BY-NC-SA 4.0")
                self.assertTrue(r["source"].startswith("Dermoscopedia: "))

    @unittest.skipUnless(HAVE_GIT, "needs a Git checkout")
    def test_no_key_is_in_a_tracked_file_and_the_env_file_is_ignored(self):
        """Never paste a secret: the keys live in .env only, which Git ignores."""
        self.assertEqual(git("check-ignore", "-q", ".env").returncode, 0, ".env is not ignored")
        self.assertNotIn(".env", git("ls-files").stdout.decode("utf-8").split("\n"))
        shapes = re.compile(r"sk-or-v1-[0-9a-f]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{32,}|AIza[0-9A-Za-z_-]{35}")
        found = []
        for name in git("ls-files", "-z").stdout.decode("utf-8").split("\0"):
            path = ROOT / name
            if name and path.is_file() and path.stat().st_size < 3_000_000 and path.suffix.lower() not in {".jpg", ".png", ".pdf"}:
                if shapes.search(path.read_text(encoding="utf-8", errors="ignore")):
                    found.append(name)
        self.assertEqual(found, [])

    def test_no_e_mail_address_is_in_the_code_that_calls_outside_services(self):
        """The researcher's e-mail address is never sent to an outside service: no address is written in the scripts."""
        address = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
        for path in SRC.glob("*.py"):
            self.assertEqual(address.findall(path.read_text(encoding="utf-8")), [], path.name)

    @unittest.skipUnless(HAVE_GIT, "needs a Git checkout")
    def test_nothing_is_tagged_before_the_researcher_says_the_final_version_is_ready(self):
        """.claude/CLAUDE.md: no Git tags until the final version."""
        self.assertEqual(git("tag", "--list").stdout.decode("utf-8").strip(), "")


if __name__ == "__main__":
    unittest.main()
