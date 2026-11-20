"""Shared test fixtures: tiny synthetic projects laid out like the repository (coloured squares for the clinic photo dataset, short
invented cases for the text dataset, photos with a record for the clinic photo + record dataset) and stand-in models. No tests here."""

import contextlib
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from src import models, run


ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================================================================
# The pipeline and the repository
# ==============================================================================================================================


COLOURS = {"melanoma": (120, 40, 30), "nevus": (200, 150, 120)}
SIZES = {"train": 3, "val": 2, "test": 2}  # photos per class and split in the synthetic project
MODEL_LIST = """[[models]]
name = "fake"
requests_per_minute = 600000
order_seed = 0
[models.settings]
provider = "fake"
id = "fake-1"
"""  # configs/models.toml of the synthetic projects: one stand-in model


def write_project_files(base: Path, dataset: str, prompt: str) -> tuple[Path, Path]:
    """configs/models.toml and prompts/<dataset>/closed_book.txt, as in the repository."""
    config, prompt_path = base / "configs" / "models.toml", base / "prompts" / dataset / "closed_book.txt"
    for path, text in ((config, MODEL_LIST), (prompt_path, prompt)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return config, prompt_path
LABELS = ["nevus", "melanoma", "vascular lesion"]


def write_table(base: Path) -> Path:
    """data/clinic_patients/dataset.csv for the photos now in data/clinic_patients/files/photos (as the dataset preparation writes it)."""
    image_dir, table = base / "data" / "clinic_patients" / "files" / "photos", base / "data" / "clinic_patients" / "dataset.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["split", "label", "path", "sha256", "patient_id", "source"])  # and a further column, as the preparation adds
        for file in sorted(image_dir.rglob("*.jpg")):
            split, label, _ = file.relative_to(image_dir).parts
            group = f"patient-{label}" if split == "test" else file.stem  # each class's two test photos are of one patient
            writer.writerow([split, label, file.relative_to(image_dir).as_posix(), hashlib.sha256(file.read_bytes()).hexdigest(), group,
                             "PAD-UFES-20"])
    return table


class Project(unittest.TestCase):
    """A prepared dataset (3+3 training, 2+2 validation, 2+2 test photos: PAT_0000001 to PAT_0000014, in that order), its
    table, a prompt and the model list, in a folder laid out like the repository; each test runs inside it."""

    dataset = "clinic_photo"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.image_dir = self.base / "data" / "clinic_patients" / "files" / "photos"
        number = 0
        for split, n in SIZES.items():
            for label, colour in COLOURS.items():
                folder = self.image_dir / split / label
                folder.mkdir(parents=True, exist_ok=True)
                for _ in range(n):
                    number += 1
                    image = Image.new("RGB", (60, 45), colour)
                    image.paste((255, 255, 255), (0, 0, 2 + number, 2 + number))  # every image differs from every other
                    image.save(folder / f"PAT_{number:07d}.jpg", quality=95)
        self.table = write_table(self.base)
        self.config, self.prompt = write_project_files(self.base, self.dataset, "Pick one:\n{labels}\nAnswer with the name only.")
        self.results = self.base / "results"
        previous = os.getcwd()
        os.chdir(self.base)  # the paths data/, configs/ and prompts/ point into the project
        self.addCleanup(os.chdir, previous)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))  # the progress lines of the runs inside the tests


# ---------------------------------------------------------------------------------------------------------------------------------
# Runs


def listed(prompt: str) -> list[str]:
    """The answer options a prompt lists, one per line after "- "."""
    return re.findall(r"^- (.+)$", prompt, re.M)


def json_answer(first: str, labels: list[str], flag: str = "yes") -> str:
    """An answer of the required form: ``first`` as the diagnosis, then the other listed classes, as many as the format asks."""
    others = [label for label in labels if label != first]
    return json.dumps({"differential": [first, *others][:min(3, len(labels))], "evidence_sufficient": flag,
                       "reason": "A stand-in answer.", "claims": [{"statement": "A stand-in claim.", "evidence": []}], "evidence": []})


class FakeModel:
    """Says "melanoma" for every image, as an answer of the required form. Calls are counted, including retries.

    ``fail_on``: call numbers that fail once with a transient error; ``fail_from``: every call from this number on fails
    transiently; ``fatal_from``: every call from this number on fails with an error that retrying cannot fix;
    ``quota_from``: every call from this number on finds the daily quota used up.
    """

    def __init__(self, fail_on=(), fail_from=None, fatal_from=None, quota_from=None):
        self.calls, self.fail_on, self.fail_from, self.fatal_from, self.quota_from = 0, set(fail_on), fail_from, fatal_from, quota_from

    def describe(self):
        return {"model_id": "fake-1", "generation_config": {}}

    def predict(self, images, prompt):
        self.calls += 1
        if self.quota_from and self.calls >= self.quota_from:
            raise models.QuotaExhausted("Daily quota used up for fake-1 (limit 20 requests per day on the free tier)")
        if self.fatal_from and self.calls >= self.fatal_from:
            raise models.FatalError("Provider refused the request (HTTP 404 NOT_FOUND): gone", 404)
        if self.calls in self.fail_on or (self.fail_from and self.calls >= self.fail_from):
            raise models.TransientError("rate limited")
        return {"text": json_answer("melanoma", listed(prompt)), "finish_reason": "STOP", "model_version": "fake-1-001",
                "response_id": str(self.calls), "usage": {}}


class UnpinnedModel(FakeModel):
    """A model whose name may move to a newer version (the adapter says ``pinned = False``)."""

    pinned = False


class RunProject(Project):
    def setUp(self):
        super().setUp()
        self.model_dir = self.results / self.dataset / "closed_book" / "fake"
        patcher = mock.patch("src.run.time.sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)
        self.clean = mock.patch("src.run._git", return_value={"commit": "c" * 40, "uncommitted_changes": False})
        self.clean.start()
        self.addCleanup(self.clean.stop)

    def _run(self, model, *args, **kwargs):
        with mock.patch("src.run.load", return_value=model):
            return run.run("fake", self.dataset, *args, results=self.results, **kwargs)

    def _preds(self, folder):
        return [json.loads(x) for x in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------------------------------------------------------------
# The setup check


class ProbeModel:
    """Answers the blank-image question as told: ``answer`` (a dict), or raises ``error``."""

    def __init__(self, answer=None, error=None):
        self.answer = answer or {"text": "ok", "finish_reason": "STOP", "model_version": "fake-1-001", "response_id": "r",
                                 "usage": {"output_tokens": 1}}
        self.error, self.calls, self.images = error, 0, []

    def predict(self, images, prompt):
        self.calls += 1
        self.images.append((*(images[0] if images else (None, None)), prompt))  # the first image, as a single-image request sends it
        self.sent = images
        if self.error:
            raise self.error
        return self.answer


# ---------------------------------------------------------------------------------------------------------------------------------
# The text dataset: each case goes into the prompt, word for word, with no image

TEXT_CASES = {  # (split, class): the article of each case
    ("train", "actinic keratosis"): ["101"], ("train", "melanoma"): ["102", "103"], ("train", "nevus"): ["104", "105"],
    ("val", "actinic keratosis"): ["106"], ("val", "melanoma"): ["107"], ("val", "nevus"): ["108"],
    ("test", "actinic keratosis"): ["201", "202"], ("test", "melanoma"): ["203", "203", "204"],  # one article, two patients
    ("test", "nevus"): ["205", "206", "207"],
}


class TextProject(unittest.TestCase):
    """A prepared text dataset (16 cases: 5 training, 3 validation, 8 test from 7 articles), its table, a prompt with the case
    placeholder and the model list, in a folder laid out like the repository; each test runs inside it."""

    dataset = "published_text"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.case_dir, self.table = self.base / "data" / "published_cases" / "files" / "text", self.base / "data" / "published_cases" / "dataset.csv"
        self.texts, rows, number = {}, [], 0
        for (split, label), articles in TEXT_CASES.items():
            (self.case_dir / split / label).mkdir(parents=True)
            for pmid in articles:
                number += 1
                path = f"{split}/{label}/{pmid}-{number}.txt"
                self.texts[path] = f"A {40 + number}-year-old woman had a “café au lait” lesion – case {number}.\nIt grew."
                (self.case_dir / path).write_bytes(self.texts[path].encode("utf-8"))
                rows.append(f"{split},{label},{path},{hashlib.sha256(self.texts[path].encode('utf-8')).hexdigest()},{pmid}-{number},"
                            f"{pmid},PMC-Patients\n")
        self.table.write_text("split,label,path,sha256,patient_uid,pmid,source\n" + "".join(rows), encoding="utf-8")
        self.config, self.prompt = write_project_files(self.base, self.dataset,
                                                       "A case.\nCase: {case}\nPick one:\n{labels}\nAnswer with the name only.\n")
        self.results = self.base / "results"
        previous = os.getcwd()
        os.chdir(self.base)
        self.addCleanup(os.chdir, previous)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))  # the progress lines of the runs inside the tests
        for name, value in (("time.sleep", None), ("_git", {"commit": "c" * 40, "uncommitted_changes": False})):
            patcher = mock.patch(f"src.run.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)


class RecordingModel(FakeModel):
    """Says "melanoma" for every case, and keeps what each request carried."""

    def __init__(self):
        super().__init__()
        self.requests = []

    def predict(self, images, prompt):
        self.requests.append((*(images[0] if images else (None, None)), prompt))
        return super().predict(images, prompt)


# ---------------------------------------------------------------------------------------------------------------------------------
# The clinic photo + record dataset: each photo with its patient's record in the prompt

class MultimodalProject(Project):
    """The photo project, with the table's facts column giving each photo its patient's record (one test photo has an unknown) and a
    prompt with the facts placeholder."""

    dataset = "clinic_photo_record"

    MISSING = "test/nevus/PAT_0000013.jpg"

    def setUp(self):
        super().setUp()
        with self.table.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.questions = self.table  # one table serves the photo view and the photo + record view
        with self.questions.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "label", "path", "sha256", "patient_id", "facts"], lineterminator="\n")
            writer.writeheader()
            for i, row in enumerate(rows):
                itch = "unknown" if row["path"] == self.MISSING else "no"
                writer.writerow({k: row[k] for k in ("split", "label", "path", "sha256", "patient_id")}
                                | {"facts": f"1. Age in years: {30 + 5 * i}\n2. Body site: trunk\n3. The lesion itches: {itch}"})
        self.prompt.write_text("A photo.\nRecord:\n{facts}\nPick one:\n{labels}\nAnswer with the name only.\n", encoding="utf-8")
        for name, value in (("time.sleep", None), ("_git", {"commit": "c" * 40, "uncommitted_changes": False})):
            patcher = mock.patch(f"src.run.{name}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
