"""The prepared datasets: what differs between them (DATASETS), reading a dataset's committed table and checking every file
on disk against it, byte for byte, before anything is sent."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

DATA = Path("data")  # one folder per dataset (clinic_patients, published_cases), and the knowledge base (data/knowledge/)
TABLE_NAME = "dataset.csv"  # in a dataset's folder: the table of the prepared dataset (a dataset with other views has more tables)
SPLITS = ("train", "val", "test")  # the splits of the prepared dataset
COLUMNS = ("split", "label", "path", "sha256")  # required in the table; other columns are kept there but not needed here
MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}  # the image files a dataset can hold
TEXT_TYPES = {".txt": "text/plain"}  # the text files: UTF-8, each one case
SYSTEM_FILES = {"thumbs.db", "desktop.ini"}  # written by Windows Explorer into folders it shows; never part of the dataset
SHA256 = re.compile(r"[0-9a-f]{64}")


# ---------------------------------------------------------------------------------------------------------------------------------
# The prepared dataset


@dataclass(frozen=True)
class Dataset:
    """What differs between the datasets; everything else (answer reader, metrics, runs, reports) is shared."""

    media_types: dict[str, str]  # the files it holds: file ending -> media type
    file_kind: str  # the same in words, for messages
    in_prompt: bool  # True: each file's text goes into the prompt, at {case}; False: each file is an image sent next to the prompt
    group_column: str  # optional column of its table: the group of each file (the files of one group are resampled together)
    item: str  # what one file is, in messages and reports
    group: str  # what one group is
    why_grouped: str  # why the files of one group are not independent
    set_apart: tuple[str, ...] = ()  # classes with so few files that the report also gives the scores without them
    facts_column: str = ""  # the table's column with each file's patient facts, put into the prompt at {facts}
    table: str = TABLE_NAME  # its table, in data/<folder>/
    files: str = ""  # its folder of files, inside data/ (for example clinic_patients/files/photos)
    folder: str = ""  # its folder in data/, which holds its table (for example clinic_patients)
    pictures_column: str = ""  # the table's column with each question's pictures ("file|sha256;file|sha256"), sent next to the prompt
    pictures_root: str = ""  # the folder in data/ the pictures' paths start from
    subset: bool = False  # its questions are some of another dataset's files: other files in its folder of files are expected
    unique_content: bool = True  # False: two questions may have the same content (a patient record is the same words for two patients)
    missing_fact: str = ""  # how a missing fact reads in what the model is shown (the questions without one are scored again beside the main score); required with a facts column; a text dataset that is a record sets it too

    def __post_init__(self):
        if not self.folder or not self.files:
            raise ValueError("a dataset must say its folder and its folder of files")
        if self.facts_column and not self.missing_fact:
            raise ValueError("a dataset with a facts column must say how a missing fact reads (missing_fact)")


DATASETS = {  # the datasets a config can name
    # The published cases (data/published_cases/): the text of a patient described in a case report (dataset.csv), the same patient's photos
    # from the report's own article (photos.csv, the cases whose article shows the lesion), and the three views of them: the text alone, the
    # text with the photos and the photos alone. The photos' questions are some of the text's: a text run is compared with them on those only.
    "published_text": Dataset(TEXT_TYPES, "a .txt file", True, "pmid", "case", "article",
                              "Patients described in one article are not independent", set_apart=("actinic keratosis",),
                              folder="published_cases", files="published_cases/files/text"),  # 2 test cases: limitation 3 of its data sheet
    "published_text_photos": Dataset(TEXT_TYPES, "a .txt file", True, "pmid", "case", "article",
                                     "Patients described in one article are not independent", set_apart=("actinic keratosis",),
                                     table="photos.csv", folder="published_cases", files="published_cases/files/text",
                                     pictures_column="pictures", pictures_root="published_cases/files", subset=True),
    "published_photos": Dataset(TEXT_TYPES, "a .txt file", False, "pmid", "case", "article",
                                "Patients described in one article are not independent", set_apart=("actinic keratosis",),
                                table="photos.csv", folder="published_cases", files="published_cases/files/text",
                                pictures_column="pictures", pictures_root="published_cases/files", subset=True),
    # The clinic patients cohort (PAD-UFES-20; data/clinic_patients/): the same questions in three views, so that the answers pair.
    # One question is one photo of one patient's lesion with the record of the same visit; its tables are written by step views.
    "clinic_photo": Dataset(MEDIA_TYPES, "a JPEG or PNG file", False, "patient_id", "photo", "patient",
                            "Photos of one patient are not independent", folder="clinic_patients", files="clinic_patients/files/photos"),
    "clinic_record": Dataset(TEXT_TYPES, "a .txt file", True, "patient_id", "record", "patient",
                             "Photos and records of one patient are not independent", table="record.csv", folder="clinic_patients",
                             files="clinic_patients/files/records", unique_content=False, missing_fact="unknown"),
    "clinic_photo_record": Dataset(MEDIA_TYPES, "a JPEG or PNG file", False, "patient_id", "photo", "patient",
                                   "Photos of one patient are not independent", facts_column="facts", folder="clinic_patients",
                                   files="clinic_patients/files/photos", missing_fact="unknown"),
}
SAME_QUESTIONS = ({"published_text", "published_text_photos", "published_photos"},
                  {"clinic_photo", "clinic_record", "clinic_photo_record"})  # datasets whose questions can be paired: same photos or cases


def question_of(path: str) -> str:
    """What identifies a question across datasets: its path without the file ending (a clinic photo, its record and the two together
    are <split>/<class>/<img_id>.jpg or .txt; a published case and its photos share the case's .txt path)."""
    return path.rsplit(".", 1)[0]


def dataset_of(card: dict) -> str:
    """The dataset a run card names."""
    return card["config"]["dataset"]


@dataclass(frozen=True)
class Item:
    split: str
    label: str
    path: str  # inside the dataset's folder of files, with forward slashes: <split>/<label>/<file>
    sha256: str
    group: str = ""  # the patient a photo or record is of, or the article a case comes from; empty when the table does not say
    facts: str = ""  # a photo + record question's record text, as the model is shown it
    pictures: tuple = ()  # a case question's pictures: (path inside the pictures' folder, sha256), in the order they are sent


def dataset_paths(dataset: str) -> tuple[Path, Path]:
    """The table and the folder of files of one prepared dataset, as DATASETS names them (the clinic photo dataset:
    data/clinic_patients/dataset.csv and data/clinic_patients/files/photos/)."""
    kind = DATASETS[dataset]
    return DATA / kind.folder / kind.table, DATA / kind.files


TABLE, IMAGES = dataset_paths("clinic_photo")  # the defaults below


def read_table(table: str | Path = TABLE, dataset: str = "clinic_photo") -> list[Item]:
    """The rows of the table, after checking that they describe one consistent dataset (no file is opened)."""
    kind = DATASETS[dataset]
    table = Path(table)
    if not table.is_file():
        raise FileNotFoundError(f"{table.as_posix()} not found: the prepared dataset does not exist yet (run the dataset "
                                "preparation first)")
    with table.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        needed = (*COLUMNS, *([kind.facts_column] if kind.facts_column else []), *([kind.pictures_column] if kind.pictures_column else []))
        missing = [c for c in needed if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{table.as_posix()} lacks the column(s): {', '.join(missing)}")
        items = [Item(*(row[c] for c in COLUMNS), group=(row.get(kind.group_column) or "").strip(),
                      facts=(row[kind.facts_column] or "").strip() if kind.facts_column else "",
                      pictures=_pictures(row[kind.pictures_column]) if kind.pictures_column else ()) for row in reader]
    if not items:
        raise ValueError(f"{table.as_posix()} lists no file")
    first_line = {}
    for line, item in enumerate(items, start=2):  # line 1 is the header
        where = f"{table.as_posix()} line {line}"
        if item.split not in SPLITS:
            raise ValueError(f"{where}: split {item.split!r} is not one of {list(SPLITS)}")
        parts = item.path.split("/")
        if len(parts) != 3 or parts[:2] != [item.split, item.label] or any(p in ("", ".", "..") for p in parts):
            raise ValueError(f"{where}: path {item.path!r} must be <split>/<label>/<file>, here {item.split}/{item.label}/<file>")
        if Path(item.path).suffix.lower() not in kind.media_types:
            raise ValueError(f"{where}: {item.path} is not {kind.file_kind}")
        if not SHA256.fullmatch(item.sha256):
            raise ValueError(f"{where}: sha256 must be 64 lowercase hexadecimal characters")
        if kind.facts_column and not item.facts:
            raise ValueError(f"{where}: the {kind.facts_column} column is empty (a missing fact is written \"{kind.missing_fact}\")")
        if kind.pictures_column and not item.pictures:
            raise ValueError(f"{where}: the {kind.pictures_column} column lists no picture")
        for picture, sha in item.pictures:
            if Path(picture).suffix.lower() not in MEDIA_TYPES or any(p in ("", ".", "..") for p in picture.split("/")):
                raise ValueError(f"{where}: picture {picture!r} must be a JPEG or PNG file inside the pictures' folder")
            if not SHA256.fullmatch(sha):
                raise ValueError(f"{where}: the sha256 of picture {picture} must be 64 lowercase hexadecimal characters")
        for key, what in ((item.path, "path"), *([(item.sha256, kind.item)] if kind.unique_content else []),
                          *((p, "picture") for p, _ in item.pictures), *((s, "picture") for _, s in item.pictures)):
            if key in first_line:  # a second copy of one file could sit in two splits: a leak between them
                raise ValueError(f"{where}: the same {what} is also on line {first_line[key]}")
            first_line[key] = line
    return items


def _pictures(value: str) -> tuple:
    """A table's list of pictures, "file|sha256;file|sha256", as ((file, sha256), ...)."""
    pairs = [part.split("|") for part in (value or "").split(";") if part.strip()]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError(f"a list of pictures must read file|sha256;file|sha256 (got {value!r})")
    return tuple((path.strip(), sha.strip()) for path, sha in pairs)


def pictures_root(dataset: str) -> Path:
    """The folder a dataset's pictures' paths start from (data/<pictures_root>/)."""
    return DATA / DATASETS[dataset].pictures_root


def load_dataset(table: str | Path = TABLE, root: str | Path = IMAGES, dataset: str = "clinic_photo") -> list[Item]:
    """The table's rows, after checking that the files on disk are exactly the ones it lists, byte for byte."""
    items = read_table(table, dataset)
    files = f"{DATASETS[dataset].item}s"
    root = Path(root)
    changed, missing = [], []
    for item in items:
        file = root / item.path
        if not file.is_file():
            missing.append(item.path)
        elif hashlib.sha256(file.read_bytes()).hexdigest() != item.sha256:
            changed.append(item.path)
    pictures = pictures_root(dataset) if DATASETS[dataset].pictures_column else None
    for item in items:
        for picture, sha in item.pictures:
            file = pictures / picture
            if not file.is_file():
                missing.append(picture)
            elif hashlib.sha256(file.read_bytes()).hexdigest() != sha:
                changed.append(picture)
    listed = {item.path for item in items}
    extra = [] if DATASETS[dataset].subset else sorted(
        p.relative_to(root).as_posix() for split in SPLITS if (root / split).is_dir() for p in (root / split).rglob("*")
        if p.is_file() and p.relative_to(root).as_posix() not in listed and not p.name.startswith(".")
        and p.name.lower() not in SYSTEM_FILES)
    if changed or missing or extra:
        examples = "; ".join(f"{why}: {paths[0]}" for why, paths in (("changed", changed), ("missing", missing),
                                                                       ("not in the table", extra)) if paths)
        raise ValueError(f"the {files} in {root.as_posix()} differ from {Path(table).as_posix()}: {len(changed)} changed, "
                         f"{len(missing)} missing, {len(extra)} not in the table ({examples}). Run the dataset preparation "
                         f"again, or restore the {files}")
    return items


def classes(items: list[Item]) -> list[str]:
    """The answer options: the classes of the training split. A validation or test class without training files is refused."""
    train = sorted({item.label for item in items if item.split == "train"})
    if not train:
        raise ValueError("the training split is empty, so the classes are unknown")
    unknown = sorted({item.label for item in items if item.split != "train"} - set(train))
    if unknown:
        raise ValueError(f"classes without training files: {unknown}")
    return train


def items_hash(items: list[Item]) -> str:
    """One hash for a list of files: it changes if any file, label, path, split, group or fact changes."""
    lines = sorted(f"{i.split}\t{i.label}\t{i.path}\t{i.sha256}" + (f"\t{i.group}" if i.group else "")
                   + (f"\t{i.facts}" if i.facts else "") + "".join(f"\t{p}|{s}" for p, s in i.pictures) for i in items)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def read_image(item: Item, root: str | Path = IMAGES) -> tuple[bytes, str, dict]:
    """The bytes every model receives for one image (the file exactly as stored), its media type, and a record of what was sent."""
    return read_picture(item.path, root)


def read_picture(path: str, root: str | Path) -> tuple[bytes, str, dict]:
    """One picture as every model receives it (the file exactly as stored), its media type, and a record of what was sent."""
    content = (Path(root) / path).read_bytes()
    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
    except Exception as exc:  # any decoding failure stops the run with a message, never a silent skip
        raise ValueError(f"{path} is not a readable image ({exc})") from exc
    return content, MEDIA_TYPES[Path(path).suffix.lower()], {
        "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "width": width, "height": height}


def read_case(item: Item, root: str | Path) -> tuple[str, dict]:
    """The text every model receives for one case (the file exactly as stored, read as UTF-8) and a record of what was sent."""
    content = (Path(root) / item.path).read_bytes()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:  # stops the run with a message, never a silent skip
        raise ValueError(f"{item.path} is not UTF-8 text ({exc})") from exc
    if not text.strip():
        raise ValueError(f"{item.path} is empty")
    return text, {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "words": len(text.split())}
