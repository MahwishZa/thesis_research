"""Where the case images, the case tables and the question files of each view are."""
import json
import os

from src.utils.common import PROCESSED_IMAGES, SPLIT_FILES, VIEWS_DIR


def image_path(isic_id):
    """Where a cleaned image is, relative to the data folder, with "/" on every system (never "\\" from Windows)."""
    return f"{PROCESSED_IMAGES}/{isic_id}.jpg"


def images_not_in_table(data, isic_ids):
    """The .jpg files in data/processed/images/ whose name is not a case in the table (left over from an earlier run), sorted."""
    folder = os.path.join(data, PROCESSED_IMAGES)
    if not os.path.isdir(folder):
        return []
    keep = {f"{i}.jpg" for i in isic_ids}
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg") and f not in keep)


def view_table(view):
    return f"{VIEWS_DIR}/{view}/cases.csv"


def question_file(view, split):
    return f"{VIEWS_DIR}/{view}/{SPLIT_FILES[split]}.jsonl"


def moved_files(data):
    """data/views/provenance.json: for each file that the 2026-11-25 move changed (new image paths, a new file name), its earlier
    path and its SHA-256 before and after, keyed by its path now (relative to the data folder). Empty when there is no record."""
    path = os.path.join(data, VIEWS_DIR, "provenance.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)["files"]


def recorded_question_path(root, recorded):
    """The question file that a run recorded (a path relative to the repository folder, e.g. data/views/text_only/val.jsonl), at
    its place now: a file renamed by the 2026-11-25 move is found through data/views/provenance.json."""
    path = os.path.join(root, recorded)
    if os.path.isfile(path):
        return path
    data, _, rest = recorded.replace("\\", "/").partition("/")
    for now, record in moved_files(os.path.join(root, data)).items():
        if record["previous_path"] == rest:
            return os.path.join(root, data, now)
    return path
