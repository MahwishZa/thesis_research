"""Preparation of the image dataset, one step per command, run in this order. Nothing in data/image/raw is ever changed.

    python -m src.image.prepare inventory
    python -m src.image.prepare isic [--limit 5]
    python -m src.image.prepare identity [--limit 5]
    python -m src.image.prepare ham10000
    python -m src.image.prepare labels
    python -m src.image.prepare duplicates
    python -m src.image.prepare lesions [--limit 20]
    python -m src.image.prepare splits
    python -m src.image.prepare images
    python -m src.image.prepare shortcuts

Every step writes into the same few committed files in data/image/ and rewrites only its own part of them, so running a step again
gives byte-identical files:
    raw_photos.csv           one row per original photo (identical files count once): everything the steps find out about it
    similar_photo_pairs.csv  pairs of photos found alike: by comparing whole pictures (duplicates) or by skin pattern (lesions)
    dataset_report.md        one section per step, in the order above: its counts, its tables and what it does not do
    dataset.csv              the prepared images, written by the images step: the table the experiments read
Two committed files are written by people, not by code: isic_to_class.csv (which ISIC wording belongs to which class) and
eye_check_verdicts.csv (the verdicts of the eye check on flagged pairs). Downloads stay in data/image/downloads/ (not committed).

``inventory`` lists every file in data/image/raw (the Kaggle dataset as extracted: Train/<class>/ and Test/<class>/), opens every
image completely to find damaged files, records its format, colour mode, size, orientation and colour profile, fingerprints it
(SHA-256) and groups identical files into one photo. Once raw_photos.csv exists, the files in data/image/raw must keep matching it:
a changed, missing or new file stops the step instead of being written over, so the original dataset provably stays as downloaded.

``isic`` downloads the public ISIC Archive record of every ISIC number (no key, no cost) into data/image/downloads/isic/ (a record
already there is never downloaded again, so a stopped download continues where it stopped). ``--limit N`` downloads only the next N
records and shows what they hold, to check the format first. Once every record is there, it adds to each photo the diagnosis, how it
was confirmed, image type, lesion and patient numbers, licence, credit line, download date and a fingerprint of each record; later
steps read the downloaded records and check them against these fingerprints. No label is changed.

``identity`` checks that each photo really is the picture its ISIC number names: it downloads the ISIC Archive's preview picture of
every number (into data/image/downloads/isic_pictures/; a link that has expired is renewed by asking for the record again) and
compares it with the photo in data/image/raw (shape, a difference hash and the mean colour difference of small versions). Any pair
not judged the same picture is listed in the report for an eye check. ``--limit N`` downloads only the next N pictures first.
Nothing is changed.

``ham10000`` cross-checks the photos that come from the HAM10000 collection against HAM10000's own metadata (Harvard Dataverse
doi:10.7910/DVN/DBW86T; the file is downloaded once into data/image/downloads/ham10000/ and checked against the checksum Dataverse
publishes): the diagnosis group, how the diagnosis was confirmed, and which photos show the same lesion. No label is changed.

``labels`` gives every photo its label from its ISIC record, by the naming rule in isic_to_class.csv (or says why it gets none). If
any ISIC wording has no rule, or a photo's ISIC numbers would get two classes, it stops instead of guessing. No image is changed and
no split is made.

``duplicates`` compares every photo with every other one (shape, outline hash and mean colour difference of small versions) and
flags pairs that are one picture saved differently ("near-identical") or look alike enough to be a second photo of one lesion
("similar"). It measures how many of the same-lesion pairs that ISIC's lesion numbers name it finds, and how often it flags
different lesions. A flagged pair decides nothing by itself: only the verdicts in eye_check_verdicts.csv (per pair: same photo, and
which copy is kept; same lesion, new photo; or different lesions) and ISIC's lesion numbers decide what is left out or kept together
in the splits. No image is changed.

``lesions`` looks for second photos of one lesion (another angle, zoom or day), which a comparison of whole pictures misses. Within
each class (one lesion has one diagnosis), it compares every photo with every other one by their skin pattern: OpenCV's SIFT finds
distinctive points in each photo, the points are paired by Lowe's ratio test (each point with one other at most), and RANSAC keeps
only the pairs that fit one shift, turn and zoom; a zoom beyond 4 times closer or farther is no match. The number of points that fit
("inliers") is the evidence. It measures the method on the same-lesion pairs that ISIC's lesion numbers name and on the pairs they
name as different lesions, at several thresholds, so the threshold is chosen from data. Pairs without lesion numbers above the
lowest threshold are listed for an eye check. ``--limit N`` compares only the first N photos of each class and shows how long the
whole search will take. Copies left out by the eye check are not compared. Nothing is changed.

``splits`` puts every labelled photo into train, validation (val) or test, with every lesion in one split: photos that ISIC's or
HAM10000's lesion numbers, or the eye check, name as one lesion form a group that moves as a whole. The Kaggle dataset's own Train
and Test folders play no part: the splits are made anew. Each class gets exactly 40 test and 15 validation photos (stratified); the
rest is train, the database that open-book runs retrieve from. The split starts from a random fill with a seed fixed in advance
(SEED) and then swaps lesion groups of equal size between the parts whenever that brings the test and validation parts closer to
their class's own mix of subtype, confirmation, source and original image size; nothing else is used, and never a model's result.
The report compares the balance reached with simple random splits under the same rules. Copies left out by the eye check and photos
without a class go into no split, with the reason.

``images`` makes the image of every photo in a split from its original file, all in the same way: upright, in sRGB colours, turned
from portrait to landscape, fitted whole within 600 x 450 pixels keeping its shape (shrunk, never enlarged; nothing cut, nothing
added) and saved as JPEG with one setting and no metadata, as data/image/dataset/<split>/<class>/<photo_id>.jpg (not committed). It
writes dataset.csv (split, class, file, SHA-256, lesion group, original-test mark, size and what was done) and checks the result the
way the evaluation does before every run. Nothing in data/image/raw is changed.

``shortcuts`` measures how much simple properties of the prepared images, which say nothing about the lesion (size, file size, dark
corners, the colour of the outer frame, sharpness), reveal each image's class and source: a nearest-neighbour vote from the training
images, scored on the test split like a model. It also tabulates where each class's photos come from. Nothing is changed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import PIL
from PIL import Image, ImageChops, ImageCms, ImageOps, ImageStat, features

from ..common import fetch, write_section
from ..evaluate import FILES, MEDIA_TYPES, SPLITS, SYSTEM_FILES, TABLE_NAME, items_hash, load_dataset, summarize
from ..evaluate import classes as training_classes

DATA = Path("data/image")  # the image dataset's folder: its committed tables and reports, and raw/, downloads/ and dataset/ (on this computer)
PHOTOS = "raw_photos.csv"  # one row per original photo: everything the steps find out about it
PAIRS = "similar_photo_pairs.csv"  # pairs of photos found alike: by comparing whole pictures, or by skin pattern
REPORT = "dataset_report.md"  # one section per step, in the order the steps run
NAMING_RULE = "isic_to_class.csv"  # written by people, not by code: which ISIC wording belongs to which class
EYE_CHECK = "eye_check_verdicts.csv"  # written by people, not by code: the verdicts of the eye check on flagged pairs
STEPS = ("inventory", "isic", "identity", "ham10000", "labels", "duplicates", "lesions", "splits", "images", "shortcuts")
STEP_COLUMNS = {  # the columns of raw_photos.csv that each step writes (and rewrites when run again); every other column is kept
    "inventory": ["photo_id", "isic_ids", "files", "kaggle_labels", "kaggle_test", "sha256", "bytes", "format", "mode", "width",
                  "height", "exif_orientation", "icc_profile", "problem"],
    "isic": ["isic_record", "isic_diagnosis", "diagnosis_confirm_type", "concomitant_biopsy", "benign_malignant",
             "other_diagnosis_fields", "image_type", "lesion_id", "patient_id", "licence", "attribution", "isic_retrieved_utc",
             "isic_record_sha256"],
    "identity": ["identity", "identity_shape_difference_pct", "identity_hash_distance", "identity_mean_difference",
                 "isic_picture_width", "isic_picture_height"],
    "ham10000": ["second_source", "ham_dx", "ham_dx_type", "ham_confirmation", "ham_lesion_id", "ham_lesion_group"],
    "labels": ["label", "status", "label_vs_folder"],
    "splits": ["split", "split_note", "lesion_group"]}
PHOTO_COLUMNS = [  # in reading order: which photo, its label, the evidence for it, then the file itself
    "photo_id", "isic_ids", "label", "status", "split", "split_note", "lesion_group", "label_vs_folder", "kaggle_labels", "kaggle_test", "isic_diagnosis",
    "diagnosis_confirm_type", "concomitant_biopsy", "benign_malignant", "second_source", "ham_dx", "ham_dx_type", "ham_confirmation",
    "lesion_id", "ham_lesion_id", "ham_lesion_group", "patient_id", "image_type", "other_diagnosis_fields", "identity",
    "identity_shape_difference_pct", "identity_hash_distance", "identity_mean_difference", "isic_picture_width",
    "isic_picture_height", "files", "sha256", "bytes", "format", "mode", "width", "height", "exif_orientation", "icc_profile",
    "problem", "licence", "attribution", "isic_record", "isic_retrieved_utc", "isic_record_sha256"]
DONE = {"inventory": "sha256", "isic": "isic_record", "identity": "identity", "ham10000": "second_source",
        "labels": "status", "splits": "split"}  # the column every photo has filled once the step has run
PAIR_COLUMNS = ["photo_a", "photo_b", "label_a", "label_b", "found_by", "isic_lesion", "tier", "shape_difference_pct",
                "hash_distance", "mean_difference", "good_matches", "inliers", "zoom", "turn_degrees"]
FOUND_BY = {"duplicates": "picture comparison", "lesions": "skin pattern"}  # each step rewrites only the pairs it found
REPORT_HEAD = ["# Preparation of the dataset", "",
               "Written by `python -m src.image.prepare`, one section per step in the order the steps run; each step rewrites only its own "
               "section. What the steps find out about each photo is in `raw_photos.csv` (one row per original photo), the pairs of photos "
               "found alike are in `similar_photo_pairs.csv`, and `isic_to_class.csv` and `eye_check_verdicts.csv` are written by people. Nothing in `data/image/raw` "
               "is ever changed.", ""]
ISIC_API = "https://api.isic-archive.com/api/v2/images/{}/"
ISIC_DELAY = 0.3  # seconds between requests: a polite pace for a free public service (about 15 minutes for 2,200 records)
ISIC_ATTEMPTS = 4  # tries per record when the service is busy or the network fails: waits of 4, 8 and 16 seconds
LEVELS = [f"diagnosis_{n}" for n in range(1, 6)]  # ISIC's diagnosis from the broadest (1) to the most specific (5)
# The most each may be for "same picture". The mean colour difference decides: on realistic synthetic lesion photos, a photo and its
# own preview (another size, another JPEG quality, a 3% brightness shift) differed by at most 5.4, while 89,400 pairs of different
# lesions never differed by less than 8.6. The hash only guards against a gross mismatch: similar-looking lesions can have close hashes.
SAME = {"shape_difference_pct": 2.0, "hash_distance": 16, "mean_difference": 5.0}
SIGNATURE_SIZE = 32  # the comparison looks at 32 x 32 versions of both pictures
DATAVERSE = "https://dataverse.harvard.edu"
HAM_DOI = "doi:10.7910/DVN/DBW86T"  # HAM10000 (Tschandl, Rosendahl and Kittler, Scientific Data 5:180161, 2018)
HAM_GROUPS = {  # HAM10000's seven diagnosis groups, as its paper defines them
    "akiec": "actinic keratoses and intraepithelial carcinoma (Bowen's disease)", "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesions (solar lentigines, seborrheic keratoses, lichen-planus like keratoses)",
    "df": "dermatofibroma", "mel": "melanoma", "nv": "melanocytic nevi",
    "vasc": "vascular lesions (cherry angiomas, angiokeratomas, pyogenic granulomas, haemorrhage)"}
HAM_EXPECTED = {  # the HAM10000 group each ISIC wording belongs to; HAM10000 has no group of its own for squamous cell carcinoma
    "Solar or actinic keratosis": "akiec", "Squamous cell carcinoma, NOS": "akiec", "Squamous cell carcinoma in situ": "akiec",
    "Basal cell carcinoma": "bcc", "Pigmented benign keratosis": "bkl", "Dermatofibroma": "df", "Nevus": "nv",
    "Malignant melanocytic proliferations (Melanoma)": "mel", "Benign soft tissue proliferations - Vascular": "vasc"}
HAM_CONFIRMATION = {"histo": "histopathology", "follow_up": "serial imaging showing no change",
                    "consensus": "single image expert consensus", "confocal": "confocal microscopy with consensus dermoscopy"}
RULE_COLUMNS = ["level", "isic_wording", "class", "reason"]
EYE_CHECK_COLUMNS = ["photo_a", "photo_b", "verdict", "keep", "note"]
VERDICTS = ("same photo", "same lesion, new photo", "different lesions")
FEATURES = 1000  # SIFT points per photo (the strongest)
WORK_SIDE = 600  # photos are compared at this length of their longer side (HAM10000's photos are already 600 x 450)
RATIO = 0.75  # Lowe's ratio test: a point pairs only if its best partner is clearly better than its second best
RANSAC_PIXELS = 5.0  # how far (at WORK_SIDE) a paired point may lie from where the shift, turn and zoom put it
ZOOM_RANGE = (0.25, 4.0)  # a second photo of one lesion is at most 4 times closer or farther; a fit outside this is not a real one
THRESHOLDS = (10, 15, 20, 30, 50, 100)  # inliers at which the calibration is shown; pairs from the lowest one on are listed
# Pairs of different files flagged as one picture or one lesion. "near-identical" is the identity check's "same picture" (the same
# photo saved again: resized, re-compressed); "similar" is looser and may also catch a second photo of the same lesion (or, now and
# then, a different lesion that looks alike). Both are measured against the pairs ISIC's lesion numbers already name.
NEAR_IDENTICAL = SAME
SIMILAR = {"shape_difference_pct": 10.0, "hash_distance": 20, "mean_difference": 10.0}
# (True would keep the original test photos, and the other photos of their lesions, in test)
SEED = 2026  # chosen before the first split was drawn and never changed: the split depends on nothing else
TEST_PER_CLASS = 40
VAL_PER_CLASS = 15
MIN_TRAIN = 50  # the fewest training photos a class may keep (the database that open-book runs retrieve from)
BALANCED = ("isic_diagnosis", "diagnosis_confirm_type", "attribution", "size")  # each part should mirror its class in these
RANDOM_DRAWS = 1000  # simple random splits under the same rules: the yardstick for the balance reached
PATIENCE = 3000  # swaps tried in a row without improvement before the search for a better balance stops (per class)
IMAGE_COLUMNS = ["split", "label", "path", "sha256", "lesion_group", "photo_id", "kaggle_test", "width", "height", "bytes", "changes"]
SIZE = (600, 450)  # width x height every photo is fitted within: HAM10000's own size, which most photos already have
SNAP_PIXELS = 1  # a fitted size this close to SIZE becomes SIZE exactly (a photo whose shape is 4:3 up to rounding)
JPEG = {"quality": 95, "subsampling": 0}  # one setting for every image: high quality and full colour resolution (4:4:4)
EDGE = 0.1  # the shortcut check's corner squares and outer frame: this share of the shorter side (45 pixels at 600 x 450)
SHORTCUT_PROPERTIES = {"image size": ["width", "height"], "file size": ["bytes"], "dark corners": ["corners"],
                       "edge colour": ["edge_r", "edge_g", "edge_b"], "sharpness": ["sharpness"]}
SHORTCUT_NEIGHBOURS = (1, 5, 15, 45)  # the number of neighbours that vote, chosen on the validation split
PROFILE_NAME = re.compile(r"colour profile \((.*)\) to sRGB")  # how an image's changes name the colour profile it was converted from
ISIC_NUMBER = re.compile(r"ISIC_\d{7}")
FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}  # the format each file extension promises
ORIENTATION = 0x0112  # the EXIF tag that tells a viewer to rotate or mirror the picture
FILE_FIELDS = ("sha256", "bytes", "format", "mode", "width", "height", "exif_orientation", "icc_profile", "problem")  # the same for copies


# ---------------------------------------------------------------------------------------------------------------------------------
# The committed files: raw_photos.csv, similar_photo_pairs.csv and dataset_report.md, each step rewriting only its own part of them


def _read_photos(data: Path, *after: str) -> dict[str, dict]:
    """raw_photos.csv by photo_id, after checking that the steps named in ``after`` have run."""
    table = data / PHOTOS
    if not table.is_file():
        raise FileNotFoundError(f"{table.as_posix()} not found: run python -m src.image.prepare inventory first")
    with table.open(encoding="utf-8", newline="") as f:
        photos = {row["photo_id"]: {c: row.get(c) or "" for c in PHOTO_COLUMNS} for row in csv.DictReader(f)}
    for step in after:
        if not photos or any(not row[DONE[step]] for row in photos.values()):
            raise FileNotFoundError(f"{table.as_posix()} lacks what the {step} step adds: run python -m src.image.prepare {step} first")
    return photos


def _write_photos(data: Path, photos: dict[str, dict]) -> None:
    data.mkdir(parents=True, exist_ok=True)
    with (data / PHOTOS).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PHOTO_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(photos[i] for i in sorted(photos))


def _update_photos(data: Path, step: str, values: dict[str, dict]) -> dict[str, dict]:
    """Rewrites the step's columns of raw_photos.csv (a photo without values gets empty ones) and keeps every other column."""
    photos = _read_photos(data)
    for photo_id, row in photos.items():
        row.update({c: values.get(photo_id, {}).get(c, "") for c in STEP_COLUMNS[step]})
    _write_photos(data, photos)
    return photos


def _files(photos: dict[str, dict]) -> list[dict]:
    """One row per original file, sorted by path, rebuilt from raw_photos.csv (identical files share one photo row; a file's folders
    and ISIC number are in its path)."""
    rows = []
    for p in photos.values():
        paths = p["files"].split(" | ")
        for path in paths:
            kaggle_split, kaggle_label, name = path.split("/")
            number = ISIC_NUMBER.search(name)
            rows.append({"path": path, "kaggle_split": kaggle_split, "kaggle_label": kaggle_label, "isic_id": number.group(0) if number else "",
                         "copies": str(len(paths)), **{field: p[field] for field in FILE_FIELDS}})
    return sorted(rows, key=lambda row: row["path"])


def _update_pairs(data: Path, step: str, rows: list[dict]) -> None:
    """Rewrites the pairs this step found in similar_photo_pairs.csv and keeps the other step's (the duplicates' pairs first, then the lesions')."""
    table = data / PAIRS
    kept = []
    if table.is_file():
        with table.open(encoding="utf-8", newline="") as f:
            kept = [row for row in csv.DictReader(f) if row["found_by"] != FOUND_BY[step]]
    new = [{c: "" for c in PAIR_COLUMNS} | row | {"found_by": FOUND_BY[step]} for row in rows]
    order = list(FOUND_BY.values())
    data.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(kept + new, key=lambda row: order.index(row["found_by"])))  # stable: each step keeps its own order


def _write_section(data: Path, step: str, lines: list[str]) -> None:
    """Rewrites the step's section of dataset_report.md and keeps the other steps' sections, in the order the steps run."""
    write_section(data / REPORT, REPORT_HEAD, STEPS, step, lines)


# ---------------------------------------------------------------------------------------------------------------------------------
# The inventory: every original file, opened and fingerprinted


def _describe(file: Path, rel: str) -> dict:
    """One original file: what it is, its fingerprint, and any problem found when opening it completely."""
    content = file.read_bytes()
    kaggle_split, kaggle_label, name = rel.split("/")
    number = ISIC_NUMBER.search(name)
    row = {"path": rel, "kaggle_split": kaggle_split, "kaggle_label": kaggle_label, "isic_id": number.group(0) if number else "",
           "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(), "copies": 0, "format": "", "mode": "",
           "width": "", "height": "", "exif_orientation": "", "icc_profile": "", "problem": ""}
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()  # decodes every pixel: a damaged or cut-off file fails here, not later during a run
            row.update(format=image.format, mode=image.mode, width=image.width, height=image.height,
                       exif_orientation=image.getexif().get(ORIENTATION, ""),
                       icc_profile="yes" if image.info.get("icc_profile") else "no")
    except Exception as exc:  # any decoding failure is recorded, never silently skipped
        reason = re.sub(r"\s*<[^<>]* at 0x[0-9a-fA-F]+>", "", str(exc)).strip()  # no memory address: the same file, the same text
        row["problem"] = f"unreadable: {reason or type(exc).__name__}"
        return row
    expected = FORMATS[Path(name).suffix.lower()]
    if row["format"] != expected:
        row["problem"] = f"the file is {row['format']} but its name ends in {Path(name).suffix}"
    return row


def scan_raw(raw: str | Path = DATA / "raw") -> tuple[list[dict], list[str]]:
    """The rows for every image file in data/image/raw/<split>/<class>/, sorted by path, and every other file (not used)."""
    raw = Path(raw)
    if not raw.is_dir():
        raise FileNotFoundError(f"{raw.as_posix()} not found: put the original dataset there (the Train and Test folders from the "
                                "Kaggle ZIP)")
    rows, others = [], []
    # sorted by the path text, so that the order is the same on every computer (Windows compares paths ignoring upper and lower case)
    for file in sorted((p for p in raw.rglob("*") if p.is_file()), key=lambda p: p.relative_to(raw).as_posix()):
        rel = file.relative_to(raw).as_posix()
        if file.name.startswith(".") or file.name.lower() in SYSTEM_FILES:
            continue
        if len(rel.split("/")) == 3 and file.suffix.lower() in MEDIA_TYPES:
            rows.append(_describe(file, rel))
        else:
            others.append(rel)
    if not rows:
        raise ValueError(f"no image found in {raw.as_posix()}/<split>/<class>/: put the Train and Test folders from the Kaggle ZIP there")
    copies = Counter(row["sha256"] for row in rows)
    for row in rows:
        row["copies"] = copies[row["sha256"]]
    return rows, others


def _fingerprint(rows: list[dict]) -> str:
    """One hash for the whole original dataset: it changes if any file is changed, added, removed, moved or renamed."""
    return hashlib.sha256("\n".join(f"{r['path']}\t{r['sha256']}" for r in rows).encode("utf-8")).hexdigest()


def _refuse_changed_raw(rows: list[dict], files: list[dict], table: Path) -> None:
    """Stops when data/image/raw no longer matches the committed raw_photos.csv (the original dataset must stay as downloaded)."""
    before = {r["path"]: r["sha256"] for r in files}
    now = {r["path"]: r["sha256"] for r in rows}
    changed = sorted(p for p in now.keys() & before.keys() if now[p] != before[p])
    missing, new = sorted(before.keys() - now.keys()), sorted(now.keys() - before.keys())
    if changed or missing or new:
        examples = "; ".join(f"{why}: {paths[0]}" for why, paths in (("changed", changed), ("missing", missing), ("new", new)) if paths)
        raise ValueError(f"data/image/raw differs from the committed {table.as_posix()}: {len(changed)} changed, {len(missing)} missing, "
                         f"{len(new)} new ({examples}). The original dataset must stay as downloaded: extract the Kaggle ZIP again. "
                         f"Only if the change is intended, delete {table.as_posix()} and run every step again")


def _area(row: dict) -> tuple[int, str]:
    return row["width"] * row["height"], row["path"]  # the path breaks ties, so the same data always names the same file


def _size(row: dict) -> str:
    return f"{row['width']}x{row['height']} ({row['path']})"


def _summary(rows: list[dict], others: list[str]) -> list[str]:
    """The inventory section: the counts a reader needs, every group of identical files, and what this step does not check."""
    readable = [r for r in rows if not r["problem"].startswith("unreadable")]
    groups = defaultdict(list)
    for r in rows:
        groups[r["sha256"]].append(r)
    duplicates = [g for g in groups.values() if len(g) > 1]
    two_labels = [g for g in duplicates if len({r["kaggle_label"] for r in g}) > 1]
    two_splits = [g for g in duplicates if len({r["kaggle_split"] for r in g}) > 1]
    two_numbers = [g for g in duplicates if len({r["isic_id"] for r in g}) > 1]
    contents_per_number = defaultdict(set)
    for r in rows:
        if r["isic_id"]:
            contents_per_number[r["isic_id"]].add(r["sha256"])
    splits = sorted({r["kaggle_split"] for r in rows})
    labels = sorted({r["kaggle_label"] for r in rows})
    per_folder = Counter((r["kaggle_split"], r["kaggle_label"]) for r in rows)
    unique_per_label = {label: len({r["sha256"] for r in rows if r["kaggle_label"] == label}) for label in labels}

    def count(values) -> str:
        return ", ".join(f"{k if k != '' else 'none'}: {v}" for k, v in sorted(Counter(values).items(), key=lambda kv: (-kv[1], str(kv[0]))))

    lines = ["# Inventory of the original dataset", "",
             "Written by `python -m src.image.prepare inventory` from the files in `data/image/raw`, which it does not change. One row per photo "
             "(identical files count once, with every file named) is in `raw_photos.csv`.", "",
             f"Fingerprint of `data/image/raw` (the path and SHA-256 of every image file): `{_fingerprint(rows)}`", "",
             "## Files", "", "| | Files |", "|---|---|",
             f"| Image files in `<split>/<class>/` folders | {len(rows)} |",
             f"| Opened completely without error | {len(readable)} |",
             f"| Unreadable (damaged or not an image) | {len(rows) - len(readable)} |",
             f"| Readable, but the file type does not match its name | {sum(1 for r in readable if r['problem'])} |",
             f"| Other files in `data/image/raw` (not used) | {len(others)} |",
             f"| Files whose name has no ISIC number | {sum(1 for r in rows if not r['isic_id'])} |", "",
             "## Files per folder and class (the dataset's own labels, not yet verified)", "",
             "| Class folder | " + " | ".join(splits) + " | All files | Unique photos |", "|---|" + "---|" * (len(splits) + 2)]
    for label in labels:
        lines.append(f"| {label} | " + " | ".join(str(per_folder[(s, label)]) for s in splits)
                     + f" | {sum(per_folder[(s, label)] for s in splits)} | {unique_per_label[label]} |")
    lines.append("| **Total** | " + " | ".join(str(sum(per_folder[(s, x)] for x in labels)) for s in splits)
                 + f" | {len(rows)} | {len(groups)} |")
    lines += ["", "A photo filed under two class folders counts once in each folder's unique photos, so that column adds up to more "
              "than the total.", "",
              "## How the images are stored", "",
              f"- File format: {count(r['format'] or 'unreadable' for r in rows)}",
              f"- Colour mode: {count(r['mode'] or 'unreadable' for r in readable)}",
              f"- EXIF orientation tag (1 = upright as stored; other values tell a viewer to rotate): "
              f"{count(r['exif_orientation'] for r in readable)}",
              f"- Embedded colour profile (ICC): {count(r['icc_profile'] for r in readable)}",
              f"- Smallest image (width x height): {_size(min(readable, key=_area)) if readable else '–'}; "
              f"largest: {_size(max(readable, key=_area)) if readable else '–'}",
              "", "## Image sizes per class folder", "",
              "The size of a photo should say nothing about its diagnosis. Where one class has its own typical size, a model could use "
              "size instead of the lesion.", "",
              "| Class folder | Different sizes | Most common size (files) | Share of the folder |", "|---|---|---|---|"]
    for label in labels:
        sizes = Counter(f"{r['width']}x{r['height']}" for r in readable if r["kaggle_label"] == label)
        if sizes:
            size, n = sizes.most_common(1)[0]
            lines.append(f"| {label} | {len(sizes)} | {size} ({n}) | {100 * n / sum(sizes.values()):.0f}% |")
    lines += ["", "## Identical files (the same SHA-256)", "",
              f"- Unique photos: {len(groups)} (from {len(rows)} files)",
              f"- Groups of identical files: {len(duplicates)}, holding {sum(len(g) for g in duplicates) - len(duplicates)} extra copies",
              f"- Groups filed under two or more class folders: {len(two_labels)}",
              f"- Groups with copies in two or more split folders: {len(two_splits)}",
              f"- Groups whose copies carry different ISIC numbers: {len(two_numbers)}",
              f"- ISIC numbers that name two or more different photos: {sum(1 for v in contents_per_number.values() if len(v) > 1)}", ""]
    if duplicates:
        lines += ["| Group | Files | Class folders | Split folders |", "|---|---|---|---|"]
        ordered = sorted(duplicates, key=lambda g: g[0]["path"])
        for n, g in enumerate(ordered, start=1):
            lines.append(f"| {n} | " + "<br>".join(r["path"] for r in g) + f" | {', '.join(sorted({r['kaggle_label'] for r in g}))} | "
                         + ", ".join(sorted({r["kaggle_split"] for r in g})) + " |")
        lines.append("")
    problems = [r for r in rows if r["problem"]]
    if problems or others:
        lines += ["## Files with a problem", ""] + [f"- `{r['path']}`: {r['problem']}" for r in problems]
        lines += [f"- `{p}`: not an image in a `<split>/<class>/` folder, so it is not used" for p in others] + [""]
    lines += ["## Not checked in this step", "",
              "- Whether the class folders are right: the labels are checked against the ISIC Archive in a later step.",
              "- Photos of the same lesion, and near-identical photos with different bytes: found in later steps, from the ISIC "
              "records and by comparing the photos.", ""]
    return lines


def inventory(data: str | Path = DATA) -> dict:
    """Groups the files in data/image/raw into photos (identical files are one photo), writes their file columns of raw_photos.csv and the
    inventory section of dataset_report.md, and returns the main counts. Nothing in data/image/raw is changed."""
    data = Path(data)
    rows, others = scan_raw(data / "raw")
    table = data / PHOTOS
    before = _read_photos(data) if table.is_file() else {}
    if before:
        _refuse_changed_raw(rows, _files(before), table)
    groups = defaultdict(list)
    for r in rows:
        groups[r["sha256"]].append(r)
    photos = {}
    for sha, files in sorted(groups.items(), key=lambda kv: kv[1][0]["path"]):
        numbers = sorted({f["isic_id"] for f in files if f["isic_id"]})
        photo_id = numbers[0] if numbers else f"SHA256_{sha[:12]}"
        if photo_id in photos:  # one ISIC number on two different photos: both keep it, told apart by their fingerprints
            photo_id = f"{photo_id}_SHA256_{sha[:12]}"
        photos[photo_id] = before.get(photo_id, {c: "" for c in PHOTO_COLUMNS}) | {
            "photo_id": photo_id, "isic_ids": " ".join(numbers), "files": " | ".join(f["path"] for f in files),
            "kaggle_labels": " | ".join(sorted({f["kaggle_label"] for f in files})),
            "kaggle_test": "yes" if any(f["kaggle_split"] == "Test" for f in files) else "no",
            **{field: files[0][field] for field in FILE_FIELDS}}
    _write_photos(data, photos)
    _write_section(data, "inventory", _summary(rows, others))
    return {"files": len(rows), "unreadable": sum(1 for r in rows if r["problem"].startswith("unreadable")),
            "other_problems": sum(1 for r in rows if r["problem"] and not r["problem"].startswith("unreadable")),
            "other_files": len(others), "unique": len(groups), "duplicate_groups": sum(1 for g in groups.values() if len(g) > 1),
            "table": table.as_posix(), "summary": (data / REPORT).as_posix()}


# ---------------------------------------------------------------------------------------------------------------------------------
# The ISIC Archive records


def _fetch(url: str, accept: str, check=None):
    """A download from the ISIC Archive or Harvard Dataverse at a polite pace, tried again when busy (see src/common.py)."""
    return fetch(url, accept, check, delay=ISIC_DELAY, attempts=ISIC_ATTEMPTS)


def _download(isic_id: str) -> dict:
    """The ISIC record of one number, wrapped with where and when it was downloaded; "not found" when ISIC has no such record."""
    url = ISIC_API.format(isic_id)
    try:
        record = _fetch(url, "application/json", lambda content: json.loads(content.decode("utf-8")))
        status = "found"
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise ValueError(f"the ISIC Archive refused {url} (HTTP {exc.code}). What was downloaded so far is kept") from exc
        record, status = None, "not found"
    return {"isic_id": isic_id, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": status, "record": record}


def _flatten(value, prefix: str = "") -> dict:
    """A nested record as dotted names: {"metadata": {"clinical": {"sex": "male"}}} -> {"metadata.clinical.sex": "male"}."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value}


def _isic_row(entry: dict) -> dict:
    """What one ISIC record says, as one row. Fields are found by their own name wherever ISIC nests them; nothing is guessed."""
    flat = {k: v for k, v in _flatten(entry["record"] or {}).items() if v not in (None, "")}

    def field(name: str) -> str:
        return " | ".join(dict.fromkeys(str(v) for k, v in sorted(flat.items()) if k.split(".")[-1] == name))

    named = set(LEVELS) | {"diagnosis_confirm_type"}
    other = "; ".join(f"{k}={v}" for k, v in sorted(flat.items()) if "diagnosis" in k.split(".")[-1] and k.split(".")[-1] not in named)
    record_hash = hashlib.sha256(json.dumps(entry["record"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"isic_id": entry["isic_id"], "status": entry["status"], **{level: field(level) for level in LEVELS},
            "diagnosis_confirm_type": field("diagnosis_confirm_type"),
            "concomitant_biopsy": field("concomitant_biopsy"),  # whether a biopsy was taken when the photo was made
            "benign_malignant": field("benign_malignant"),
            "other_diagnosis_fields": other, "image_type": field("image_type"), "lesion_id": field("lesion_id"),
            "patient_id": field("patient_id"), "copyright_license": field("copyright_license"), "attribution": field("attribution"),
            "retrieved_utc": entry["retrieved_utc"], "record_sha256": record_hash if entry["record"] is not None else ""}


def _chain(row: dict, depth: int = 3) -> str:
    if row["status"] != "found":
        return "(not in the ISIC Archive)"
    return " > ".join(row[level] for level in LEVELS[:depth] if row[level]) or "(no diagnosis recorded)"


def _isic_summary(raw_rows: list[dict], records: dict[str, dict]) -> list[str]:
    """The isic section: what ISIC records for the photos of each Kaggle class folder (the evidence for the naming rule)."""
    folders = sorted({r["kaggle_label"] for r in raw_rows})
    numbers = {label: sorted({r["isic_id"] for r in raw_rows if r["kaggle_label"] == label and r["isic_id"]}) for label in folders}
    found = [row for row in records.values() if row["status"] == "found"]
    missing = sorted(i for i, row in records.items() if row["status"] != "found")

    def crosstab(title: str, value) -> list[str]:
        columns = sorted({value(records[i]) for label in folders for i in numbers[label]}, key=lambda v: (v.startswith("("), v))
        lines = [f"## {title}", "", "| Kaggle folder | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
        for label in folders:
            counts = Counter(value(records[i]) for i in numbers[label])
            lines.append(f"| {label} | " + " | ".join(str(counts[c]) if counts[c] else "" for c in columns) + " |")
        return lines + [""]

    lines = ["# ISIC Archive records of the original dataset", "",
             "Written by `python -m src.image.prepare isic` from the public ISIC Archive record of every ISIC number of the photos (the "
             "records themselves are kept unchanged in `data/image/downloads/isic/`; what they say about each photo is in `raw_photos.csv`, with "
             "a fingerprint of each record). "
             "No label has been changed: this is the evidence for the naming rule that maps ISIC's wording to the class names.", "",
             f"- ISIC numbers: {len(records)}; found in the ISIC Archive: {len(found)}; not found: {len(missing)}"
             + (f" ({', '.join(missing)})" if missing else ""),
             f"- With a diagnosis recorded: {sum(1 for row in found if any(row[level] for level in LEVELS))}",
             f"- Downloaded between {min(row['retrieved_utc'] for row in records.values())} and "
             f"{max(row['retrieved_utc'] for row in records.values())} (UTC)", "",
             "A photo filed under two Kaggle folders is counted in both, and a photo with two ISIC numbers is counted twice.", "",
             "## ISIC diagnosis in each Kaggle folder (levels 1 > 2 > 3)", "", "| Kaggle folder | ISIC diagnosis | ISIC numbers |",
             "|---|---|---|"]
    for label in folders:
        for chain, n in sorted(Counter(_chain(records[i]) for i in numbers[label]).items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| {label} | {chain} | {n} |")
    lines.append("")
    lines += crosstab("How the diagnosis was confirmed", lambda row: row["diagnosis_confirm_type"] or "(not recorded)")
    lines += crosstab("Biopsy taken when the photo was made (concomitant_biopsy)", lambda row: row["concomitant_biopsy"] or "(not recorded)")
    lines += crosstab("Image type", lambda row: row["image_type"] or "(not recorded)")
    lines += crosstab("Licence", lambda row: row["copyright_license"] or "(not recorded)")

    by_lesion = defaultdict(set)
    for row in found:
        if row["lesion_id"]:
            by_lesion[row["lesion_id"]].add(row["isic_id"])
    splits_of = defaultdict(set)
    labels_of = defaultdict(set)
    for r in raw_rows:
        splits_of[r["isic_id"]].add(r["kaggle_split"])
        labels_of[r["isic_id"]].add(r["kaggle_label"])
    shared = {lesion: ids for lesion, ids in by_lesion.items() if len(ids) > 1}
    across_splits = {lesion: ids for lesion, ids in shared.items() if len(set().union(*(splits_of[i] for i in ids))) > 1}
    lines += ["## Photos of the same lesion (lesion numbers)", "",
              f"- ISIC numbers with a lesion number: {sum(1 for row in found if row['lesion_id'])} of {len(found)}",
              f"- With a patient number: {sum(1 for row in found if row['patient_id'])} of {len(found)}",
              f"- Lesions with two or more ISIC numbers: {len(shared)}",
              f"- Of these, lesions whose photos sit in both Kaggle splits (Train and Test): {len(across_splits)}", ""]
    if across_splits:
        lines += ["| Lesion | ISIC numbers (Kaggle split/folder) |", "|---|---|"]
        for lesion, ids in sorted(across_splits.items()):
            lines.append(f"| {lesion} | " + ", ".join(f"{i} ({' and '.join(sorted(splits_of[i]))}; {', '.join(sorted(labels_of[i]))})"
                                                       for i in sorted(ids)) + " |")
        lines.append("")
    lines += ["## Not done in this step", "",
              "- No label is changed. The naming rule (which ISIC wording belongs to which class) is written next, with the supervisor.",
              "- Whether each file really is the image its ISIC number names is checked in a later step, against ISIC's own picture.", ""]
    return lines


def isic(limit: int | None = None, data: str | Path = DATA) -> dict:
    """Downloads the missing ISIC records (at most ``limit``); once every record is there, writes the record columns of raw_photos.csv
    and the isic section of dataset_report.md."""
    data = Path(data)
    cache = data / "downloads" / "isic"
    photos = _read_photos(data, "inventory")
    raw_rows = _files(photos)
    ids = sorted({r["isic_id"] for r in raw_rows if r["isic_id"]})
    cache.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (cache / f"{i}.json").exists()]
    new, fields = [], []
    for isic_id in todo[:limit]:
        entry = _download(isic_id)
        tmp = cache / f"{isic_id}.json.tmp"  # written whole, then renamed: a stop never leaves half a record
        tmp.write_text(json.dumps(entry, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, cache / f"{isic_id}.json")
        new.append(_isic_row(entry))
        fields = fields or sorted(_flatten(entry["record"] or {}))  # the field names of the first record found, to check the format
    have = sum(1 for i in ids if (cache / f"{i}.json").exists())
    done = {"total": len(ids), "have": have, "new": new, "fields": fields, "written": False}
    if have < len(ids):
        return done
    records = {i: _isic_row(json.loads((cache / f"{i}.json").read_text(encoding="utf-8"))) for i in ids}
    values = {}
    for photo_id, p in photos.items():
        numbers = p["isic_ids"].split()
        if not numbers:
            values[photo_id] = {"isic_record": "no ISIC number"}
            continue
        found = [records[n] for n in numbers if records[n]["status"] == "found"]
        first = found[0] if found else records[numbers[0]]  # a photo with two ISIC numbers is described by its first record
        values[photo_id] = {"isic_record": first["status"], "isic_diagnosis": _chain(first, 5) if found else "",
                            **{field: first[field] for field in ("diagnosis_confirm_type", "concomitant_biopsy", "benign_malignant",
                                                                  "other_diagnosis_fields", "image_type", "lesion_id", "patient_id",
                                                                  "attribution")},
                            "licence": first["copyright_license"], "isic_retrieved_utc": first["retrieved_utc"],
                            "isic_record_sha256": " ".join(records[n]["record_sha256"] or "-" for n in numbers)}
    _update_photos(data, "isic", values)
    _write_section(data, "isic", _isic_summary(raw_rows, records))
    return done | {"written": True, "found": sum(1 for row in records.values() if row["status"] == "found"),
                   "table": (data / PHOTOS).as_posix(), "summary": (data / REPORT).as_posix()}


def _records(data: Path, photos: dict[str, dict]) -> dict[str, dict]:
    """The ISIC record of every ISIC number, read from data/image/downloads/isic/ and checked against the fingerprints in raw_photos.csv."""
    cache, records = data / "downloads" / "isic", {}
    for p in photos.values():
        numbers, fingerprints = p["isic_ids"].split(), p["isic_record_sha256"].split()
        if len(numbers) != len(fingerprints):
            raise FileNotFoundError(f"{(data / PHOTOS).as_posix()} lacks the record fingerprints of {p['photo_id']}: run python -m "
                                    "src.image.prepare isic first")
        for number, fingerprint in zip(numbers, fingerprints):
            path = cache / f"{number}.json"
            if not path.is_file():
                raise FileNotFoundError(f"{path.as_posix()} not found: run python -m src.image.prepare isic first")
            row = _isic_row(json.loads(path.read_text(encoding="utf-8")))
            if (row["record_sha256"] or "-") != fingerprint:
                raise ValueError(f"{path.as_posix()} is not the record {(data / PHOTOS).as_posix()} was made from: run python -m "
                                 "src.image.prepare isic again")
            records[number] = row
    return records


# ---------------------------------------------------------------------------------------------------------------------------------
# The identity check: is each photo the picture its ISIC number names?


def _signature(content: bytes) -> dict:
    """What the comparison needs from a picture: its shape, a 64-bit difference hash and a small colour version."""
    with Image.open(io.BytesIO(content)) as image:
        width, height = image.size
        image.draft("RGB", (4 * SIGNATURE_SIZE, 4 * SIGNATURE_SIZE))  # a large JPEG is decoded at a reduced scale: fast, same result
        image = ImageOps.exif_transpose(image).convert("RGB")
        grey = image.convert("L").resize((9, 8), Image.Resampling.BOX).tobytes()
        bits = sum(1 << (row * 8 + col) for row in range(8) for col in range(8) if grey[row * 9 + col] > grey[row * 9 + col + 1])
        small = image.resize((SIGNATURE_SIZE, SIGNATURE_SIZE), Image.Resampling.BOX).tobytes()
    return {"width": width, "height": height, "hash": bits, "pixels": small}


def _compare(a: dict, b: dict) -> dict:
    """How far apart two pictures are: shape (% difference of width/height), hash distance (0-64) and mean colour difference (0-255)."""
    ratio_a, ratio_b = a["width"] / a["height"], b["width"] / b["height"]
    return {"shape_difference_pct": round(100 * abs(ratio_a - ratio_b) / ratio_a, 1),
            "hash_distance": (a["hash"] ^ b["hash"]).bit_count(),
            "mean_difference": round(sum(abs(x - y) for x, y in zip(a["pixels"], b["pixels"])) / len(a["pixels"]), 1)}


def _isic_picture(isic_id: str, records: Path, thumbnails: Path) -> bytes | None:
    """The ISIC Archive's preview picture of one number (saved in ``thumbnails``), or None when ISIC has no record of it."""
    saved = thumbnails / f"{isic_id}.jpg"
    if saved.exists():
        return saved.read_bytes()
    entry = json.loads((records / f"{isic_id}.json").read_text(encoding="utf-8"))
    if entry["status"] != "found":
        return None

    def link(record: dict) -> str:
        url = ((record.get("files") or {}).get("thumbnail_256") or {}).get("url")
        if not url:
            raise ValueError(f"the ISIC record of {isic_id} names no preview picture (files.thumbnail_256.url)")
        return url

    def picture(content: bytes) -> bytes:
        try:
            Image.open(io.BytesIO(content)).verify()
        except Exception as exc:  # a cut-off or garbled picture is fetched again
            raise ValueError(f"not a readable picture ({type(exc).__name__})") from exc
        return content

    try:
        content = _fetch(link(entry["record"]), "image/*", picture)
    except urllib.error.HTTPError as exc:  # a link with an expiry date that has passed: ask ISIC for the record again
        if exc.code not in (401, 403, 404, 410):
            raise ValueError(f"the ISIC Archive refused the preview picture of {isic_id} (HTTP {exc.code})") from exc
        fresh = _download(isic_id)  # the saved record stays as it was downloaded; only its fresh link is used
        try:
            content = _fetch(link(fresh["record"] or {}), "image/*", picture)
        except urllib.error.HTTPError as again:
            raise ValueError(f"the ISIC Archive refused the preview picture of {isic_id} (HTTP {again.code})") from again
    thumbnails.mkdir(parents=True, exist_ok=True)
    tmp = saved.with_name(saved.name + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, saved)
    return content


def _identity_summary(rows: list[dict]) -> list[str]:
    verdicts = Counter(row["verdict"] for row in rows)
    compared = [row for row in rows if row["hash_distance"] != ""]
    bins = [("0 to 1", 0, 1), ("over 1 to 2", 1, 2), ("over 2 to 5", 2, 5), ("over 5 to 10", 5, 10), ("over 10", 10, 255)]
    lines = ["# Identity check: each photo against the ISIC Archive's own picture", "",
             "Written by `python -m src.image.prepare identity`. For every ISIC number, the photo in `data/image/raw` is compared with the preview "
             "picture the ISIC Archive serves for that number: the shape (width / height), the mean colour difference of 32 x 32 "
             "versions (0 to 255; 0 = the same) and a 64-bit difference hash of an 8 x 9 grey version (0 = the same outline). Nothing "
             "is changed; the result for each photo is in `raw_photos.csv`.", "",
             f"- ISIC numbers: {len(rows)}; " + "; ".join(f"{verdict}: {n}" for verdict, n in sorted(verdicts.items())),
             f"- \"same picture\" means: shapes differ by at most {SAME['shape_difference_pct']}%, mean difference at most "
             f"{SAME['mean_difference']} and hash distance at most {SAME['hash_distance']} of 64. Anything else is listed below for an "
             "eye check.", "",
             "## Mean colour differences (0 = the same picture; different lesions were never below 8.6 in the calibration)", "",
             "| Mean difference | ISIC numbers |", "|---|---|"]
    lines += [f"| {name} | {sum(1 for row in compared if (low < row['mean_difference'] or low == 0) and row['mean_difference'] <= high)} |"
              for name, low, high in bins]
    lines += ["", "## Largest differences among the pictures judged the same", "",
              f"- Shape: {max((row['shape_difference_pct'] for row in compared if row['verdict'] == 'same picture'), default='–')}%; "
              f"hash distance: {max((row['hash_distance'] for row in compared if row['verdict'] == 'same picture'), default='–')}; "
              f"mean difference: {max((row['mean_difference'] for row in compared if row['verdict'] == 'same picture'), default='–')}", ""]
    review = [row for row in rows if row["verdict"] != "same picture"]
    if review:
        lines += ["## For an eye check", "", "| ISIC number | Photo in data/image/raw | Verdict | Shape difference | Hash distance | "
                  "Mean difference |", "|---|---|---|---|---|---|"]
        lines += [f"| {row['isic_id']} | {row['path']} | {row['verdict']} | {row['shape_difference_pct']} | {row['hash_distance']} | "
                  f"{row['mean_difference']} |" for row in review]
        lines.append("")
    lines += ["## Not done in this step", "", "- No label, file or record is changed.", ""]
    return lines


def identity(limit: int | None = None, data: str | Path = DATA) -> dict:
    """Compares every photo with ISIC's picture of its number (at most ``limit`` new downloads); once all are compared, writes the
    identity columns of raw_photos.csv and the identity section of dataset_report.md."""
    data = Path(data)
    raw, downloads = data / "raw", data / "downloads"
    photos = _read_photos(data, "isic")
    raw_rows = _files(photos)
    records, thumbnails = downloads / "isic", downloads / "isic_pictures"
    ids = sorted({r["isic_id"] for r in raw_rows if r["isic_id"]})
    if not all((records / f"{i}.json").exists() for i in ids):
        raise FileNotFoundError("not every ISIC record is downloaded: run python -m src.image.prepare isic first")
    first_file = {}
    for r in raw_rows:
        first_file.setdefault(r["isic_id"], r)
    rows, fetched = [], 0
    for isic_id in ids:
        saved = (thumbnails / f"{isic_id}.jpg").exists()
        if limit is not None and fetched >= limit and not saved:
            continue
        picture = _isic_picture(isic_id, records, thumbnails)
        fetched += not saved and picture is not None  # only pictures really downloaded now count
        file = first_file[isic_id]
        local = (raw / file["path"]).read_bytes()
        if hashlib.sha256(local).hexdigest() != file["sha256"]:
            raise ValueError(f"{file['path']} differs from {(data / PHOTOS).as_posix()}: the original dataset must stay as downloaded")
        row = {"isic_id": isic_id, "path": file["path"], "width": file["width"], "height": file["height"], "isic_width": "",
               "isic_height": "", "shape_difference_pct": "", "hash_distance": "", "mean_difference": ""}
        if picture is None:
            row["verdict"] = "no ISIC picture"
        elif file["problem"].startswith("unreadable"):
            row["verdict"] = "photo unreadable"
        else:
            mine, theirs = _signature(local), _signature(picture)
            row.update(isic_width=theirs["width"], isic_height=theirs["height"], **_compare(mine, theirs))
            same = all(row[key] <= most for key, most in SAME.items())
            row["verdict"] = "same picture" if same else "check by eye"
        rows.append(row)
    done = {"total": len(ids), "compared": len(rows), "new": fetched, "rows": rows, "written": False}
    if len(rows) < len(ids):
        return done
    by_number, values = {row["isic_id"]: row for row in rows}, {}
    for photo_id, p in photos.items():
        numbers = p["isic_ids"].split()
        if not numbers:
            values[photo_id] = {"identity": "no ISIC number"}
            continue
        verdicts, first = {by_number[n]["verdict"] for n in numbers}, by_number[numbers[0]]
        values[photo_id] = {"identity": first["verdict"] if len(verdicts) == 1 else "; ".join(f"{n}: {by_number[n]['verdict']}" for n in numbers),
                            "identity_shape_difference_pct": first["shape_difference_pct"], "identity_hash_distance": first["hash_distance"],
                            "identity_mean_difference": first["mean_difference"], "isic_picture_width": first["isic_width"],
                            "isic_picture_height": first["isic_height"]}
    _update_photos(data, "identity", values)
    _write_section(data, "identity", _identity_summary(rows))
    return done | {"written": True, "table": (data / PHOTOS).as_posix(), "summary": (data / REPORT).as_posix()}


# ---------------------------------------------------------------------------------------------------------------------------------
# The HAM10000 cross-check: a second, independent source for the photos that come from the HAM10000 collection


def _ham_metadata(folder: Path) -> tuple[dict, Path]:
    """The Dataverse description of HAM10000 and its metadata file (downloaded once into ``folder``, then reused unchanged)."""
    described = folder / "dataset.json"
    if not described.exists():
        listing = _fetch(f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={HAM_DOI}", "application/json",
                         lambda content: json.loads(content.decode("utf-8")))
        folder.mkdir(parents=True, exist_ok=True)
        described.write_text(json.dumps({"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "listing": listing},
                                        indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    info = json.loads(described.read_text(encoding="utf-8"))
    files = [f for f in info["listing"]["data"]["latestVersion"]["files"] if f.get("label", "").lower().startswith("ham10000_metadata")]
    if len(files) != 1:
        names = [f.get("label") for f in info["listing"]["data"]["latestVersion"]["files"]]
        raise ValueError(f"expected one HAM10000_metadata file in {HAM_DOI}, found {len(files)} among {names}")
    data_file = files[0]["dataFile"]
    original = data_file.get("originalFileName")  # Dataverse turns an uploaded CSV into a .tab table; the original is asked for
    name = original or data_file.get("filename") or files[0]["label"]
    target = folder / Path(name).name
    if not target.exists():
        url = f"{DATAVERSE}/api/access/datafile/{data_file['id']}" + ("?format=original" if original else "")
        content = _fetch(url, "text/csv, text/tab-separated-values, */*")
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, target)
    return info | {"file": data_file, "name": name}, target


def _read_ham(path: Path) -> dict[str, dict]:
    """HAM10000's rows by image number (ISIC_xxxxxxx); comma- or tab-separated, with or without quotes."""
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t" if "\t" in text.split("\n", 1)[0] else ",")
    rows = [{(k or "").strip().strip('"').lower(): (v or "").strip() for k, v in row.items()} for row in reader]
    missing = [c for c in ("image_id", "lesion_id", "dx", "dx_type") if rows and c not in rows[0]]
    if not rows or missing:
        raise ValueError(f"{path.as_posix()} is not the HAM10000 metadata file (missing columns: {missing or 'all'})")
    return {row["image_id"]: row for row in rows}


def _isic_wording(row: dict) -> str:
    """The ISIC wording that decides the HAM10000 group: level 2 for melanoma and vascular lesions, level 3 otherwise."""
    if row["diagnosis_2"] in ("Malignant melanocytic proliferations (Melanoma)", "Benign soft tissue proliferations - Vascular"):
        return row["diagnosis_2"]
    return row["diagnosis_3"] or " > ".join(row[level] for level in LEVELS if row[level]) or "(no diagnosis recorded)"


def _tally(values) -> str:
    """Counts as text, agreement first and what cannot be compared last: "agree: 4; differ: 1; cannot be compared: 1"."""
    counts = Counter(values)
    return "; ".join(f"{value or 'cannot be compared'}: {n}" for value, n in sorted(counts.items(), key=lambda kv: (kv[0] == "", kv[0])))


def _ham_summary(rows: list[dict], info: dict, path: Path, raw_rows: list[dict], total: int) -> list[str]:
    published = (info["file"].get("checksum") or {}).get("value") or info["file"].get("md5") or ""
    local = hashlib.md5(path.read_bytes()).hexdigest()
    in_ham = {row["isic_id"] for row in rows}
    folders = sorted({r["kaggle_label"] for r in raw_rows})
    per_folder = {label: {r["isic_id"] for r in raw_rows if r["kaggle_label"] == label and r["isic_id"]} for label in folders}

    def tally(key: str) -> str:
        return _tally(row[key] for row in rows)

    def crosstab(row_key: str, column_key: str, title: str, heading: str) -> list[str]:
        columns = sorted({row[column_key] for row in rows})
        lines = [f"## {title}", "", f"| {heading} | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
        for value in sorted({row[row_key] for row in rows}):
            counts = Counter(row[column_key] for row in rows if row[row_key] == value)
            lines.append(f"| {value or '(not recorded)'} | " + " | ".join(str(counts[c]) if counts[c] else "" for c in columns) + " |")
        return lines + [""]

    lines = ["# HAM10000 cross-check", "",
             "Written by `python -m src.image.prepare ham10000`. Every photo of this dataset that is also in HAM10000 is compared with "
             "HAM10000's own metadata: its diagnosis group, how the diagnosis was confirmed, and which photos show the same lesion. "
             "No label is changed; the result for each photo is in `raw_photos.csv`.", "",
             f"Source: HAM10000 (Tschandl, Rosendahl and Kittler, *Scientific Data* 5:180161, 2018), Harvard Dataverse {HAM_DOI}, "
             f"version {info['listing']['data']['latestVersion'].get('versionNumber', '?')}."
             f"{info['listing']['data']['latestVersion'].get('versionMinorNumber', '?')}, file `{info['name']}`, downloaded "
             f"{info['retrieved_utc']} (UTC). MD5 published by Dataverse: `{published or 'none'}`; of the downloaded file: `{local}`"
             + (" (they match)." if published == local else " (they do not match: see whether Dataverse publishes it for the .tab version)."), "",
             f"- ISIC numbers in this dataset: {total}; also in HAM10000: {len(in_ham)}",
             f"- Diagnosis group: {tally('diagnosis')}",
             f"- How the diagnosis was confirmed: {tally('confirmation')}",
             f"- Lesion groups: {tally('lesion_group')}", "",
             "## Photos in HAM10000 per Kaggle folder", "", "| Kaggle folder | ISIC numbers | In HAM10000 |", "|---|---|---|"]
    lines += [f"| {label} | {len(per_folder[label])} | {len(per_folder[label] & in_ham)} |" for label in folders]
    lines.append("")
    lines += crosstab("isic_diagnosis", "ham_dx", "ISIC's wording against HAM10000's diagnosis group", "ISIC wording (rows), HAM10000 group (columns)")
    lines += crosstab("isic_confirmed", "ham_dx_type", "How the diagnosis was confirmed: ISIC against HAM10000",
                      "ISIC (rows), HAM10000 (columns)")
    different = [row for row in rows if row["diagnosis"] == "differ" or row["confirmation"] == "differ" or row["lesion_group"] == "differ"]
    if different:
        lines += ["## Where the two sources differ", "", "| ISIC number | ISIC diagnosis | HAM10000 | ISIC confirmed | HAM10000 | ISIC lesion "
                  "| HAM10000 lesion | Differs in |", "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {row['isic_id']} | {row['isic_diagnosis']} | {row['ham_dx']} | {row['isic_confirmed'] or '-'} | {row['ham_dx_type']} "
                  f"| {row['isic_lesion_id'] or '-'} | {row['ham_lesion_id']} | "
                  + ", ".join(k for k in ("diagnosis", "confirmation", "lesion_group") if row[k] == "differ") + " |" for row in different]
        lines.append("")
    extra = [row for row in rows if row["ham_lesion_id"] and not row["isic_lesion_id"]]
    lines += ["## Lesion numbers that only HAM10000 records", "",
              f"- Photos with a HAM10000 lesion number but no ISIC lesion number: {len(extra)}"
              + (f" ({', '.join(row['isic_id'] for row in extra)})" if extra else ""),
              "- When the splits are made, photos that either source puts in one lesion stay together.", "",
              "## HAM10000's diagnosis groups (from its paper)", ""]
    lines += [f"- `{dx}`: {meaning}" for dx, meaning in HAM_GROUPS.items()]
    lines += ["", "## Not done in this step", "", "- No label is changed. The comparison is evidence for the naming rule.", ""]
    return lines


def ham10000(data: str | Path = DATA) -> dict:
    """Compares every photo that is also in HAM10000 with HAM10000's metadata, and writes the HAM10000 columns of raw_photos.csv and
    the ham10000 section of dataset_report.md."""
    data = Path(data)
    photos = _read_photos(data, "isic")
    raw_rows = _files(photos)
    isic_rows = _records(data, photos)
    info, path = _ham_metadata(data / "downloads" / "ham10000")
    ham = _read_ham(path)
    rows = []
    for isic_id in sorted(isic_rows):
        if isic_id not in ham:
            continue
        mine, theirs = isic_rows[isic_id], ham[isic_id]
        wording = _isic_wording(mine)
        expected = HAM_EXPECTED.get(wording, "")
        confirmed = HAM_CONFIRMATION.get(theirs["dx_type"], "")
        rows.append({"isic_id": isic_id, "isic_diagnosis": wording, "ham_dx": theirs["dx"],
                     "diagnosis": ("agree" if expected == theirs["dx"] else "differ") if expected else "",
                     "isic_confirmed": mine["diagnosis_confirm_type"], "ham_dx_type": theirs["dx_type"],
                     "confirmation": ("agree" if confirmed == mine["diagnosis_confirm_type"] else "differ")
                     if confirmed and mine["diagnosis_confirm_type"] else "",
                     "isic_lesion_id": mine["lesion_id"], "ham_lesion_id": theirs["lesion_id"], "lesion_group": ""})
    by_isic, by_ham = defaultdict(set), defaultdict(set)  # lesion groups, among the photos that both sources number
    both = [row for row in rows if row["isic_lesion_id"] and row["ham_lesion_id"]]
    for row in both:
        by_isic[row["isic_lesion_id"]].add(row["isic_id"])
        by_ham[row["ham_lesion_id"]].add(row["isic_id"])
    for row in both:
        row["lesion_group"] = "agree" if by_isic[row["isic_lesion_id"]] == by_ham[row["ham_lesion_id"]] else "differ"
    by_number, values = {row["isic_id"]: row for row in rows}, {}
    for photo_id, p in photos.items():
        numbers = [n for n in p["isic_ids"].split() if n in by_number]
        first = by_number[numbers[0]] if numbers else {}
        values[photo_id] = {"second_source": _second_source([by_number[n]["diagnosis"] for n in numbers]),
                            "ham_dx": first.get("ham_dx", ""), "ham_dx_type": first.get("ham_dx_type", ""),
                            "ham_confirmation": first.get("confirmation", ""), "ham_lesion_id": first.get("ham_lesion_id", ""),
                            "ham_lesion_group": first.get("lesion_group", "")}
    _update_photos(data, "ham10000", values)
    _write_section(data, "ham10000", _ham_summary(rows, info, path, raw_rows, len(isic_rows)))
    return {"total": len(isic_rows), "in_ham": len(rows), "rows": rows, "table": (data / PHOTOS).as_posix(),
            "summary": (data / REPORT).as_posix()}


# ---------------------------------------------------------------------------------------------------------------------------------
# The labels: one row per unique photo, labelled by the naming rule from its ISIC record


def _read_rule(path: Path, allowed: set[str]) -> list[dict]:
    """The naming rule, after checking that it can be applied without guessing."""
    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} not found: the naming rule is part of the repository (git pull origin main)")
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != RULE_COLUMNS:
            raise ValueError(f"{path.as_posix()} must have the columns {RULE_COLUMNS} (it has {reader.fieldnames})")
        rules = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    seen = set()
    for n, rule in enumerate(rules, start=2):
        where = f"{path.as_posix()} line {n}"
        if rule["level"] not in LEVELS or not rule["isic_wording"]:
            raise ValueError(f"{where}: level must be one of {LEVELS} and isic_wording must not be empty")
        if rule["class"] and rule["class"] not in allowed:
            raise ValueError(f"{where}: class {rule['class']!r} is not one of the dataset's class names {sorted(allowed)}")
        if not rule["class"] and not rule["reason"]:
            raise ValueError(f"{where}: a wording that gets no class needs a reason")
        if (rule["level"], rule["isic_wording"]) in seen:
            raise ValueError(f"{where}: {rule['level']} {rule['isic_wording']!r} has a rule already")
        seen.add((rule["level"], rule["isic_wording"]))
    return rules


def _apply_rule(record: dict, rules: list[dict]) -> dict | None:
    """The rule for an ISIC record: of the rules whose wording the record has, the one at the most specific level (a record has
    one wording per level, and the rule file has one rule per wording, so it is unambiguous). None when no rule matches."""
    matches = [rule for rule in rules if record[rule["level"]] == rule["isic_wording"]]
    return max(matches, key=lambda rule: LEVELS.index(rule["level"])) if matches else None


def _second_source(verdicts: list[str]) -> str:
    """What HAM10000 says about a photo's diagnosis: agrees, differs, cannot compare, or the photo is not in HAM10000."""
    if not verdicts:
        return "not in HAM10000"
    if "differ" in verdicts:
        return "HAM10000 differs"
    return "HAM10000 agrees" if all(v == "agree" for v in verdicts) else "HAM10000 cannot compare"


def labels(data: str | Path = DATA) -> dict:
    """Labels every photo by the naming rule from its ISIC record, and writes the label columns of raw_photos.csv and the labels
    section of dataset_report.md. No image is changed."""
    data = Path(data)
    photos = _read_photos(data, "isic", "ham10000")
    records = _records(data, photos)
    rules = _read_rule(data / NAMING_RULE, {r["kaggle_label"] for r in _files(photos)})
    unmatched = sorted({f"{record['isic_id']} ({_chain(record, 5)})" for record in records.values()
                        if record["status"] == "found" and _apply_rule(record, rules) is None})
    if unmatched:
        raise ValueError(f"the naming rule in {(data / NAMING_RULE).as_posix()} covers no rule for {len(unmatched)} ISIC records, so "
                         f"their class would be a guess: {unmatched[:5]}. Add a rule for their wording (or a reason why they get no class)")
    values = {}
    for photo_id, p in photos.items():
        numbers = p["isic_ids"].split()
        found = [records[i] for i in numbers if records[i]["status"] == "found"]
        matched = {(m["class"], m["reason"]) for m in (_apply_rule(record, rules) for record in found)}
        if len(matched) > 1:
            raise ValueError(f"the ISIC numbers {numbers} of one photo ({p['files'].split(' | ')[0]}) name different classes: "
                             f"{sorted(matched)}")
        folders = p["kaggle_labels"].split(" | ")
        label, reason = matched.pop() if matched else ("", "")
        if p["problem"].startswith("unreadable"):
            label, status = "", "unreadable file"
        elif not numbers:
            label, status = "", "no ISIC number"
        elif not found:
            label, status = "", "no ISIC record"
        else:
            status = "labelled" if label else reason
        values[photo_id] = {"label": label, "status": status,
                            "label_vs_folder": ("no class" if not label else "same as folder" if folders == [label]
                                                else "one of two folders" if label in folders else "differs from folder")}
    rows = list(_update_photos(data, "labels", values).values())
    _write_section(data, "labels", _labels_summary(rows, rules, records))
    return {"photos": len(rows), "labelled": sum(1 for p in rows if p["label"]), "rows": rows,
            "table": (data / PHOTOS).as_posix(), "summary": (data / REPORT).as_posix()}


def _labels_summary(photos: list[dict], rules: list[dict], records: dict[str, dict]) -> list[str]:
    labelled = [p for p in photos if p["label"]]
    classes = sorted({p["label"] for p in labelled})
    per_rule = Counter()
    for p in photos:
        for number in p["isic_ids"].split():
            rule = _apply_rule(records[number], rules) if records.get(number, {}).get("status") == "found" else None
            if rule:
                per_rule[(rule["level"], rule["isic_wording"])] += 1
    lines = ["# Labels of the original dataset", "",
             "Written by `python -m src.image.prepare labels`. Each unique photo (identical files count once) is labelled from its ISIC "
             "Archive record by the naming rule in `isic_to_class.csv`, never from its Kaggle folder. One row per photo is in "
             "`raw_photos.csv`. No image is changed and no split is made in this step.", "",
             f"- Photos: {len(photos)}; labelled: {len(labelled)}; without a class: {len(photos) - len(labelled)}",
             f"- Classes: {len(classes)} ({', '.join(classes)})", "",
             "## The naming rule", "", "| ISIC level | ISIC wording | Class | Reason | ISIC numbers |", "|---|---|---|---|---|"]
    lines += [f"| {r['level']} | {r['isic_wording']} | {r['class'] or '(no class)'} | {r['reason']} | "
              f"{per_rule[(r['level'], r['isic_wording'])]} |" for r in rules]
    lines += ["", "## Photos per class", "",
              "| Class | Photos | Of these, original test photos | Confirmed by biopsy | Also confirmed by HAM10000 |", "|---|---|---|---|---|"]
    for name in classes:
        group = [p for p in labelled if p["label"] == name]
        lines.append(f"| {name} | {len(group)} | {sum(p['kaggle_test'] == 'yes' for p in group)} | "
                     f"{sum(p['diagnosis_confirm_type'] == 'histopathology' for p in group)} | "
                     f"{sum(p['second_source'] == 'HAM10000 agrees' for p in group)} |")
    lines.append(f"| **Total** | {len(labelled)} | {sum(p['kaggle_test'] == 'yes' for p in labelled)} | "
                 f"{sum(p['diagnosis_confirm_type'] == 'histopathology' for p in labelled)} | "
                 f"{sum(p['second_source'] == 'HAM10000 agrees' for p in labelled)} |")
    folders = sorted({folder for p in photos for folder in p["kaggle_labels"].split(" | ")})
    columns = classes + ["(no class)"]
    lines += ["", "## Kaggle folder against the new label (photos; a photo in two folders counts in both)", "",
              "| Kaggle folder | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for folder in folders:
        counts = Counter(p["label"] or "(no class)" for p in photos if folder in p["kaggle_labels"].split(" | "))
        lines.append(f"| {folder} | " + " | ".join(str(counts[c]) if counts[c] else "" for c in columns) + " |")
    lines += ["", "## Changes from the Kaggle folders", "",
              "- " + "; ".join(f"{k}: {v}" for k, v in sorted(Counter(p["label_vs_folder"] for p in photos).items()))]
    double = Counter((p["kaggle_labels"], p["label"]) for p in photos if p["label_vs_folder"] == "one of two folders")
    lines += [f"- Filed under {folders_} and labelled **{label}**: {n}" for (folders_, label), n in sorted(double.items())]
    moved = [p for p in photos if p["label_vs_folder"] in ("differs from folder", "no class")]
    if moved:
        lines += ["", "| Photo | Kaggle folder | ISIC diagnosis | New label | Status |", "|---|---|---|---|---|"]
        lines += [f"| {p['photo_id']} | {p['kaggle_labels']} | {p['isic_diagnosis'] or '-'} | {p['label'] or '(no class)'} | {p['status']} |"
                  for p in moved]
    lines += ["", "## Not done in this step", "",
              "- No image is changed and no split is made. Photos without a class stay in `raw_photos.csv`, marked, and go into no split.",
              "- Near-identical photos (different files of the same picture) and photos of the same lesion are handled in the next steps.",
              ""]
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------
# Near-duplicates: different files of one picture, and possibly second photos of one lesion


def _tier(measures: dict) -> str:
    for name, limits in (("near-identical", NEAR_IDENTICAL), ("similar", SIMILAR)):
        if all(measures[key] <= most for key, most in limits.items()):
            return name
    return ""


def _components(pairs: list[tuple[str, str]]) -> list[set[str]]:
    """The groups that pairs link together (a with b and b with c make one group of three)."""
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[root(a)] = root(b)
    groups = defaultdict(set)
    for x in parent:
        groups[root(x)].add(x)
    return sorted(groups.values(), key=lambda g: (-len(g), min(g)))


def duplicates(data: str | Path = DATA) -> dict:
    """Compares every photo with every other one, and writes the flagged pairs to similar_photo_pairs.csv and how well known same-lesion pairs
    are found to the duplicates section of dataset_report.md."""
    data = Path(data)
    raw, table = data / "raw", data / PHOTOS
    photos = [p for _, p in sorted(_read_photos(data, "labels").items())]
    signatures, paths = {}, {}
    for photo in photos:
        if photo["status"] == "unreadable file":
            continue
        path = raw / photo["files"].split(" | ")[0]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != photo["sha256"]:
            raise ValueError(f"{path.as_posix()} differs from {table.as_posix()}: the original dataset must stay as downloaded")
        signature = _signature(content)
        signature["image"] = Image.frombytes("RGB", (SIGNATURE_SIZE, SIGNATURE_SIZE), signature["pixels"])
        signatures[photo["photo_id"]] = signature
        paths[photo["photo_id"]] = path
    by_id = {p["photo_id"]: p for p in photos}
    ids = sorted(signatures)

    def measure(a: str, b: str) -> dict:
        sa, sb = signatures[a], signatures[b]
        ratio_a, ratio_b = sa["width"] / sa["height"], sb["width"] / sb["height"]
        return {"shape_difference_pct": round(100 * abs(ratio_a - ratio_b) / ratio_a, 1), "hash_distance": (sa["hash"] ^ sb["hash"]).bit_count(),
                "mean_difference": round(sum(ImageStat.Stat(ImageChops.difference(sa["image"], sb["image"])).mean) / 3, 1)}

    lesion = {i: by_id[i]["lesion_id"] for i in ids}
    members = defaultdict(list)
    for i in ids:
        if lesion[i]:
            members[lesion[i]].append(i)
    known_same = sorted((a, b) for group in members.values() for n, a in enumerate(group) for b in group[n + 1:])
    flagged, measured = [], {}
    gate = max(NEAR_IDENTICAL["hash_distance"], SIMILAR["hash_distance"])  # only these pairs can be flagged: the rest is not measured
    hashes = [signatures[i]["hash"] for i in ids]
    for a_index, a in enumerate(ids):
        hash_a = hashes[a_index]
        for b_index in range(a_index + 1, len(ids)):
            if (hash_a ^ hashes[b_index]).bit_count() > gate:
                continue
            b = ids[b_index]
            measured[(a, b)] = m = measure(a, b)
            if tier := _tier(m):
                same_lesion = ("same lesion" if lesion[a] == lesion[b] else "different lesions") if lesion[a] and lesion[b] else "unknown"
                flagged.append({"photo_a": a, "photo_b": b, "label_a": by_id[a]["label"], "label_b": by_id[b]["label"], **m, "tier": tier,
                                "isic_lesion": same_lesion})
    for pair in known_same:  # the known same-lesion pairs are measured whatever their hashes, for the calibration
        measured.setdefault(pair, measure(*pair))
    flagged.sort(key=lambda row: (row["mean_difference"], row["photo_a"], row["photo_b"]))
    _update_pairs(data, "duplicates", flagged)
    summary = _duplicates_summary(ids, flagged, known_same, measured, lesion)
    _write_section(data, "duplicates", summary)
    return {"photos": len(ids), "pairs": len(ids) * (len(ids) - 1) // 2, "flagged": flagged, "known_same": len(known_same),
            "found_same": sum(1 for pair in known_same if _tier(measured[pair])), "table": (data / PAIRS).as_posix(),
            "summary": (data / REPORT).as_posix()}


def _duplicates_summary(ids, flagged, known_same, measured, lesion) -> list[str]:
    tiers = Counter(row["tier"] for row in flagged)
    numbered = [i for i in ids if lesion[i]]
    known_different = len(numbered) * (len(numbered) - 1) // 2 - len(known_same)
    same_tiers = Counter(_tier(measured[pair]) or "not flagged" for pair in known_same)
    false_alarms = Counter(row["tier"] for row in flagged if row["isic_lesion"] == "different lesions")
    means = sorted(measured[pair]["mean_difference"] for pair in known_same)

    def quartiles(values):
        return ", ".join(f"{values[int(q * (len(values) - 1))]}" for q in (0, 0.25, 0.5, 0.75, 1)) if values else "–"

    lines = ["# Near-duplicates: different files of one picture, and second photos of one lesion", "",
             "Written by `python -m src.image.prepare duplicates`. Every photo in `raw_photos.csv` is compared with every other one (identical "
             "files were already merged by the inventory). Flagged pairs are in `similar_photo_pairs.csv` (found by \"picture comparison\"). No image is "
             "changed.", "",
             f"- Photos compared: {len(ids)}; pairs: {len(ids) * (len(ids) - 1) // 2}",
             f"- Flagged pairs: {len(flagged)} (near-identical: {tiers['near-identical']}; similar: {tiers['similar']})",
             f"- Flagged pairs whose two photos have different labels: {sum(1 for row in flagged if row['label_a'] != row['label_b'])}",
             f"- \"near-identical\": shapes differ by at most {NEAR_IDENTICAL['shape_difference_pct']}%, hash distance at most "
             f"{NEAR_IDENTICAL['hash_distance']}, mean colour difference at most {NEAR_IDENTICAL['mean_difference']} (the identity "
             f"check's \"same picture\"). \"similar\": at most {SIMILAR['shape_difference_pct']}%, {SIMILAR['hash_distance']} and "
             f"{SIMILAR['mean_difference']}.", "",
             "## How well it finds what ISIC's lesion numbers already tell (calibration)", "",
             f"Photos with an ISIC lesion number: {len(numbered)}. Pairs of them that show the same lesion: {len(known_same)}; "
             f"pairs that show different lesions: {known_different}.", "",
             "| Pairs | Near-identical | Similar | Not flagged |", "|---|---|---|---|",
             f"| Same lesion (should be flagged) | {same_tiers['near-identical']} | {same_tiers['similar']} | {same_tiers['not flagged']} |",
             f"| Different lesions (should not be flagged) | {false_alarms['near-identical']} | {false_alarms['similar']} | "
             f"{known_different - sum(false_alarms.values())} |", "",
             f"- Share of same-lesion pairs found: {100 * (len(known_same) - same_tiers['not flagged']) / len(known_same):.1f}%"
             if known_same else "- No same-lesion pairs to measure against.",
             f"- Mean colour difference of the same-lesion pairs (smallest, 25%, median, 75%, largest): {quartiles(means)}", "",
             "Second photos of one lesion (another angle, zoom or day) are different pictures: a comparison of whole pictures finds "
             "only those that look alike. The share above is how many it finds where the answer is known; for melanoma and nevus, "
             "which have no lesion numbers, the same share is the best estimate. This is a limitation to state.", "",
             "## Groups that the flagged pairs would make", "",
             "Pairs link into groups (a with b and b with c make one group). If the \"similar\" tier makes very large groups, it is "
             "linking unrelated lesions that merely look alike. A flagged pair decides nothing by itself: the eye check's verdicts in "
             f"`{EYE_CHECK}` and ISIC's lesion numbers decide what is left out or kept together when the splits are made.", ""]
    for name, rows in (("near-identical", [r for r in flagged if r["tier"] == "near-identical"]), ("near-identical or similar", flagged)):
        groups = _components([(r["photo_a"], r["photo_b"]) for r in rows])
        sizes = Counter(len(g) for g in groups)
        lines.append(f"- {name}: {len(groups)} groups holding {sum(len(g) for g in groups)} photos; sizes: "
                     + (", ".join(f"{n} of {size}" for size, n in sorted(sizes.items())) or "none"))
    mixed = [r for r in flagged if r["label_a"] != r["label_b"]]
    if mixed:
        lines += ["", "## Flagged pairs with different labels (worth an eye check: the same picture cannot have two diagnoses)", "",
                  "| Photo | Label | Photo | Label | Tier | Mean difference |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['photo_a']} | {r['label_a'] or '(no class)'} | {r['photo_b']} | {r['label_b'] or '(no class)'} | {r['tier']} | "
                  f"{r['mean_difference']} |" for r in mixed]
    lines += ["", "## Not done in this step", "", f"- No image or label is changed. The eye check's verdicts go into `{EYE_CHECK}`.", ""]
    return lines


def _read_eye_check(path: Path, photos: dict[str, dict]) -> dict:
    """The eye check's verdicts, after checking that they can be applied without guessing. Returns which photos are left out (as
    copies of the photo kept) and which pairs are second photos of one lesion (kept together)."""
    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} not found: the eye check is part of the repository (git pull origin main)")
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EYE_CHECK_COLUMNS:
            raise ValueError(f"{path.as_posix()} must have the columns {EYE_CHECK_COLUMNS} (it has {reader.fieldnames})")
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    seen, copies, same_lesion = set(), [], []
    for n, row in enumerate(rows, start=2):
        where, a, b = f"{path.as_posix()} line {n}", row["photo_a"], row["photo_b"]
        unknown = [i for i in (a, b) if i not in photos]
        if unknown or a == b:
            raise ValueError(f"{where}: photo_a and photo_b must be two different photo_id values of raw_photos.csv ({unknown or a})")
        if row["verdict"] not in VERDICTS:
            raise ValueError(f"{where}: verdict must be one of {list(VERDICTS)} (it is {row['verdict']!r})")
        if frozenset((a, b)) in seen:
            raise ValueError(f"{where}: {a} and {b} have a verdict already")
        seen.add(frozenset((a, b)))
        if row["verdict"] == "same photo":
            if row["keep"] not in (a, b):
                raise ValueError(f"{where}: a same photo needs keep: {a} or {b} (the copy that stays)")
            if photos[a]["label"] != photos[b]["label"]:
                raise ValueError(f"{where}: one photo cannot have two labels ({a}: {photos[a]['label'] or 'no class'}, {b}: "
                                 f"{photos[b]['label'] or 'no class'}); settle the label first")
            copies.append((a, b, row["keep"]))
        elif row["keep"]:
            raise ValueError(f"{where}: keep is only for a same photo (both photos of a {row['verdict']!r} pair stay)")
        elif row["verdict"] == "same lesion, new photo":
            same_lesion.append((a, b))
    left_out = {}
    for group in _components([(a, b) for a, b, _ in copies]):  # one picture filed three times is one group: one copy stays
        kept = {keep for a, b, keep in copies if a in group}
        if len(kept) != 1:
            raise ValueError(f"{path.as_posix()}: {', '.join(sorted(group))} are one photo, so every row about them must keep the same "
                             f"copy (they keep {', '.join(sorted(kept))})")
        (keep,) = kept
        left_out |= {i: keep for i in group - kept}
    return {"left_out": dict(sorted(left_out.items())), "same_lesion": sorted(same_lesion)}


def _opencv():
    """OpenCV and NumPy, imported only by the step that needs them."""
    try:
        import cv2
        import numpy
    except ImportError as exc:
        raise FileNotFoundError(f"OpenCV is not installed ({exc}): run pip install -r requirements.txt") from exc
    return cv2, numpy


def _features(content: bytes, sift) -> tuple:
    """The SIFT points of a photo (positions and descriptors), found on a grey version whose longer side is WORK_SIDE."""
    cv2, np = _opencv()
    with Image.open(io.BytesIO(content)) as image:
        image.draft("L", (2 * WORK_SIDE, 2 * WORK_SIDE))
        image = ImageOps.exif_transpose(image).convert("L")
        scale = WORK_SIDE / max(image.size)
        if scale < 1:
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        points, descriptors = sift.detectAndCompute(np.asarray(image), None)
    return np.float32([point.pt for point in points]).reshape(-1, 2), descriptors


def _match(a: tuple, b: tuple, matcher) -> dict:
    """How well two photos' points pair up: paired points (ratio test), the ones that fit one shift, turn and zoom, and that
    zoom and turn. RANSAC starts from the same random seed for every pair, so the result is the same on every run."""
    cv2, np = _opencv()
    (points_a, descriptors_a), (points_b, descriptors_b) = a, b
    result = {"good_matches": 0, "inliers": 0, "zoom": "", "turn_degrees": ""}
    if descriptors_a is None or descriptors_b is None or len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return result
    good = [m for m, n in (pair for pair in matcher.knnMatch(descriptors_a, descriptors_b, k=2) if len(pair) == 2)
            if m.distance < RATIO * n.distance]
    closest = {}  # one point of b pairs with one point of a only (the closest): many points paired with one let RANSAC
    for m in good:  # "fit" them all by shrinking the photo to that point (a zoom near 0), which is no match at all
        if m.trainIdx not in closest or m.distance < closest[m.trainIdx].distance:
            closest[m.trainIdx] = m
    good = sorted(closest.values(), key=lambda m: m.queryIdx)
    result["good_matches"] = len(good)
    if len(good) < 3:
        return result
    cv2.setRNGSeed(0)
    transform, mask = cv2.estimateAffinePartial2D(points_a[[m.queryIdx for m in good]], points_b[[m.trainIdx for m in good]],
                                                  method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PIXELS)
    if transform is not None and mask is not None:
        zoom = float(np.hypot(transform[0, 0], transform[1, 0]))
        result["inliers"] = int(mask.sum()) if ZOOM_RANGE[0] <= zoom <= ZOOM_RANGE[1] else 0
        result["zoom"] = round(zoom, 2)
        result["turn_degrees"] = round(float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))))
    return result


def lesions(data: str | Path = DATA, limit: int | None = None, show=None) -> dict:
    """Compares every photo with every other photo of its class by skin pattern, to find second photos of one lesion; writes the
    pairs to similar_photo_pairs.csv and the calibration to the lesions section of dataset_report.md."""
    cv2, _ = _opencv()
    data = Path(data)
    raw, table = data / "raw", data / PHOTOS
    photos = _read_photos(data, "labels")
    left_out = _read_eye_check(data / EYE_CHECK, photos)["left_out"]
    classes = defaultdict(list)
    for i in sorted(photos):
        if photos[i]["label"] and i not in left_out:
            classes[photos[i]["label"]].append(i)
    all_photos, all_pairs = sum(len(ids) for ids in classes.values()), sum(len(ids) * (len(ids) - 1) // 2 for ids in classes.values())
    if limit:
        classes = {label: ids[:limit] for label, ids in classes.items()}
    sift, matcher = cv2.SIFT_create(nfeatures=FEATURES), cv2.BFMatcher(cv2.NORM_L2)
    features, paths, started = {}, {}, time.monotonic()
    for ids in classes.values():
        for i in ids:
            path = raw / photos[i]["files"].split(" | ")[0]
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != photos[i]["sha256"]:
                raise ValueError(f"{path.as_posix()} differs from {table.as_posix()}: the original dataset must stay as downloaded")
            features[i], paths[i] = _features(content, sift), path
    feature_seconds = time.monotonic() - started
    rows, same_scores, different_scores, counts = [], [], [], {}
    for label in sorted(classes):
        ids = classes[label]
        counts[label] = (len(ids), len(ids) * (len(ids) - 1) // 2)
        if show:
            show(f"{label}: {counts[label][0]} photos, {counts[label][1]} pairs")
        for n, a in enumerate(ids):
            for b in ids[n + 1:]:
                m = _match(features[a], features[b], matcher)
                lesion_a, lesion_b = photos[a]["lesion_id"], photos[b]["lesion_id"]
                known = ("same lesion" if lesion_a == lesion_b else "different lesions") if lesion_a and lesion_b else "unknown"
                if known == "same lesion":
                    same_scores.append(m["inliers"])
                elif known == "different lesions" and m["inliers"] >= THRESHOLDS[0]:
                    different_scores.append(m["inliers"])
                if m["inliers"] >= THRESHOLDS[0]:
                    rows.append({"photo_a": a, "photo_b": b, "label_a": label, "label_b": label, **m, "isic_lesion": known})
    seconds = time.monotonic() - started
    rows.sort(key=lambda r: (-r["inliers"], r["photo_a"], r["photo_b"]))
    done = {"photos": sum(n for n, _ in counts.values()), "pairs": sum(p for _, p in counts.values()), "rows": rows, "seconds": seconds,
            "known_same": len(same_scores), "found_same": sum(score >= THRESHOLDS[0] for score in same_scores), "all_pairs": all_pairs,
            # the time of the whole search: finding the points grows with the photos, comparing them with the pairs
            "estimate_seconds": feature_seconds / max(len(features), 1) * all_photos
                                + (seconds - feature_seconds) / max(sum(p for _, p in counts.values()), 1) * all_pairs}
    if limit:  # a trial run times the search and writes nothing
        return done
    _update_pairs(data, "lesions", rows)
    known_different = sum(1 for label, ids in classes.items() for n, a in enumerate(ids) for b in ids[n + 1:]
                          if photos[a]["lesion_id"] and photos[b]["lesion_id"] and photos[a]["lesion_id"] != photos[b]["lesion_id"])
    summary = _lesions_summary(counts, rows, same_scores, different_scores, known_different, cv2.__version__)
    _write_section(data, "lesions", summary)
    done |= {"table": (data / PAIRS).as_posix(), "summary": (data / REPORT).as_posix()}
    return done


def _lesions_summary(counts, rows, same_scores, different_scores, known_different, opencv_version) -> list[str]:
    def quartiles(values):
        values = sorted(values)
        return ", ".join(f"{values[int(q * (len(values) - 1))]}" for q in (0, 0.25, 0.5, 0.75, 1)) if values else "–"

    labels = sorted(counts)
    lines = ["# Second photos of one lesion (skin-pattern matching)", "",
             "Written by `python -m src.image.prepare lesions`. Within each class, every photo is compared with every other one by its skin "
             f"pattern (OpenCV {opencv_version}: the {FEATURES} strongest SIFT points of a grey version {WORK_SIDE} pixels long, paired "
             f"by Lowe's ratio test at {RATIO}, each point with one other at most, and RANSAC keeping the pairs that fit one shift, turn "
             f"and zoom within {RANSAC_PIXELS} pixels; a zoom outside {ZOOM_RANGE[0]}–{ZOOM_RANGE[1]} is no match). \"Inliers\" are the points that fit: two photos of one lesion share many, two different lesions few. Pairs with "
             f"at least {THRESHOLDS[0]} inliers are in `{PAIRS}` (found by \"skin pattern\"). Copies left out by the eye check "
             "are not compared. No image or label is changed.", "",
             f"- Photos compared: {sum(n for n, _ in counts.values())}; pairs (within each class): {sum(p for _, p in counts.values())}",
             "- Per class: " + "; ".join(f"{label}: {counts[label][0]} photos, {counts[label][1]} pairs" for label in labels), "",
             "## How well it finds what ISIC's lesion numbers already tell (calibration)", "",
             f"Pairs that ISIC names as one lesion: {len(same_scores)}; as different lesions (same class): {known_different}. The "
             "threshold is chosen where many same-lesion pairs are found and almost no different lesions are flagged.", "",
             "| Inliers at least | Same-lesion pairs found | Different-lesion pairs flagged | Pairs without lesion numbers flagged |",
             "|---|---|---|---|"]
    for t in THRESHOLDS:
        found = sum(score >= t for score in same_scores)
        share = f" ({100 * found / len(same_scores):.1f}%)" if same_scores else ""
        lines.append(f"| {t} | {found} of {len(same_scores)}{share} | {sum(score >= t for score in different_scores)} of {known_different} | "
                     f"{sum(1 for r in rows if r['isic_lesion'] == 'unknown' and r['inliers'] >= t)} |")
    lines += ["", f"- Inliers of the same-lesion pairs (smallest, 25%, median, 75%, largest): {quartiles(same_scores)}", "",
              "## Pairs without lesion numbers flagged, per class", "",
              "| Class | " + " | ".join(f"≥ {t}" for t in THRESHOLDS) + " |", "|---|" + "---|" * len(THRESHOLDS)]
    for label in labels:
        lines.append(f"| {label} | " + " | ".join(str(sum(1 for r in rows if r["label_a"] == label and r["isic_lesion"] == "unknown"
                                                             and r["inliers"] >= t)) for t in THRESHOLDS) + " |")
    lines += ["", "## Not done in this step", "",
              f"- No image or label is changed. A flagged pair decides nothing by itself: the eye check's verdicts go into `{EYE_CHECK}` "
              "(\"same lesion, new photo\" keeps both photos in one split).", ""]
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------
# The splits: train, validation and test, every lesion in one of them


def _property(photo: dict, name: str) -> str:
    if name == "size":
        return f"{photo['width']}x{photo['height']}"
    return photo[name] or "(not recorded)"


class _Balance:
    """The mix of one class's test and validation parts, kept up to date as lesion groups move, and how far it is from the
    class's own mix: the mean total variation distance over the balanced properties that vary in the class (0 = the same
    proportions, 1 = nothing in common)."""

    def __init__(self, groups: dict[str, list[str]], photos: dict[str, dict]):
        self.values = {g: [Counter(_property(photos[i], name) for i in ids) for name in BALANCED] for g, ids in groups.items()}
        self.sizes = {g: len(ids) for g, ids in groups.items()}
        self.overall = [sum((self.values[g][n] for g in groups), Counter()) for n in range(len(BALANCED))]
        self.varying = [n for n, total in enumerate(self.overall) if len(total) > 1]
        self.total = sum(self.sizes.values())

    def score(self, assign: dict[str, str]) -> float:
        counts = {part: [Counter() for _ in BALANCED] for part in ("test", "val")}
        size = Counter()
        for g, part in assign.items():
            if part in counts:
                size[part] += self.sizes[g]
                for n in self.varying:
                    counts[part][n].update(self.values[g][n])
        distances = [sum(abs(counts[part][n][value] / size[part] - k / self.total) for value, k in self.overall[n].items()) / 2
                     for part in ("test", "val") for n in self.varying if size[part]]
        return sum(distances) / len(distances) if distances else 0.0


def _fill(sizes: dict[str, int], order: list[str], targets: dict[str, int]) -> dict[str, str] | None:
    """The lesion groups in the given order: into test while it has room, then into validation, the rest
    into train (a group too large for the room left goes on). None when a target cannot be met exactly."""
    assign, room = {}, {"test": targets["test"], "val": targets["val"]}
    for g in order:
        part = next((p for p in ("test", "val") if sizes[g] <= room[p]), "train")
        assign[g] = part
        if part != "train":
            room[part] -= sizes[g]
    return assign if room == {"test": 0, "val": 0} else None


def _optimise(balance: _Balance, assign: dict[str, str], movable: list[str], rng: random.Random, patience: int) -> dict[str, str]:
    """Swaps two movable lesion groups of equal size between two parts whenever that lowers the imbalance; stops after
    ``patience`` swaps in a row that do not."""
    best, stale = balance.score(assign), 0
    while stale < patience and len(movable) > 1:
        a, b = rng.sample(movable, 2)
        if assign[a] == assign[b] or balance.sizes[a] != balance.sizes[b]:
            stale += 1
            continue
        assign[a], assign[b] = assign[b], assign[a]
        score = balance.score(assign)
        if score < best - 1e-12:
            best, stale = score, 0
        else:
            assign[a], assign[b] = assign[b], assign[a]
            stale += 1
    return assign


def splits(data: str | Path = DATA, test: int = TEST_PER_CLASS, val: int = VAL_PER_CLASS, min_train: int = MIN_TRAIN,
           seed: int = SEED, draws: int = RANDOM_DRAWS, patience: int = PATIENCE) -> dict:
    """Puts every labelled photo into train, val or test (every lesion in one), and writes the split columns of raw_photos.csv and
    the splits section of dataset_report.md."""
    data = Path(data)
    photos = _read_photos(data, "labels")
    eye = _read_eye_check(data / EYE_CHECK, photos)
    left_out = eye["left_out"]
    labelled = sorted(i for i, p in photos.items() if p["label"])
    links = list(eye["same_lesion"]) + list(left_out.items())  # a copy shares its kept photo's group, so its links count too
    for key in ("lesion_id", "ham_lesion_id"):
        members = defaultdict(list)
        for i in labelled:
            if photos[i][key]:
                members[photos[i][key]].append(i)
        links += [(ids[0], other) for ids in members.values() for other in ids[1:]]
    group_of = {i: i for i in labelled}
    for group in _components(links):
        for i in group:
            group_of[i] = min(group)
    groups = defaultdict(list)
    for i in labelled:
        if i not in left_out:
            groups[group_of[i]].append(i)
    for name, ids in groups.items():
        if len({photos[i]["label"] for i in ids}) > 1:
            raise ValueError(f"the photos {ids} show one lesion but have different labels "
                             f"({', '.join(sorted({photos[i]['label'] for i in ids}))}): settle the label first")
    classes = defaultdict(list)
    for name in sorted(groups):
        classes[photos[groups[name][0]]["label"]].append(name)
    assign, per_class, start_scores, balances = {}, {}, {}, {}
    for label in sorted(classes):
        names = classes[label]
        sizes = {g: len(groups[g]) for g in names}
        total = sum(sizes.values())
        targets = {"test": test, "val": val}
        if total - targets["test"] - targets["val"] < min_train:
            raise ValueError(f"{label}: {total} photos leave {total - targets['test'] - targets['val']} for training after "
                             f"{targets['test']} test and {targets['val']} validation photos, fewer than {min_train}")
        rng = random.Random(f"{seed}:{label}")
        order = list(names)
        rng.shuffle(order)
        start = _fill(sizes, order, targets)
        if start is None:
            raise ValueError(f"{label}: the lesion groups cannot fill exactly {targets['test']} test and {targets['val']} "
                             "validation photos")
        balances[label] = _Balance({g: groups[g] for g in names}, photos)
        start_scores[label] = balances[label].score(start)
        assign.update(_optimise(balances[label], start, names, rng, patience))
        per_class[label] = {"names": names, "sizes": sizes, "targets": targets}
    random_scores = []  # the yardstick: simple random splits under the same rules
    for draw in range(draws):
        scores = []
        for label in sorted(classes):
            c, rng = per_class[label], random.Random(f"{seed}:random:{draw}:{label}")
            order = list(c["names"])
            rng.shuffle(order)
            filled = _fill(c["sizes"], order, c["targets"])
            if filled is not None:
                scores.append(balances[label].score(filled))
        random_scores.append(sum(scores) / len(scores))
    scores = {label: balances[label].score({g: assign[g] for g in classes[label]}) for label in classes}
    values = {}
    for i in sorted(photos):
        p = photos[i]
        if not p["label"]:
            values[i] = {"split": "none", "split_note": f"no class: {p['status']}"}
        elif i in left_out:
            values[i] = {"split": "none", "split_note": f"same photo as {left_out[i]} (eye check): left out",
                         "lesion_group": group_of[left_out[i]]}
        else:
            g = group_of[i]
            values[i] = {"split": assign[g], "split_note": "", "lesion_group": g}
    updated = _update_photos(data, "splits", values)
    rows = [updated[i] for i in sorted(updated)]
    random_scores.sort()
    _write_section(data, "splits", _splits_summary(rows, groups, scores, start_scores, random_scores, seed, test, val))
    counts = Counter((p["label"], p["split"]) for p in rows if p["split"] != "none")
    return {"counts": counts, "imbalance": sum(scores.values()) / len(scores), "start": sum(start_scores.values()) / len(start_scores),
            "random": random_scores, "rows": rows, "table": (data / PHOTOS).as_posix(), "summary": (data / REPORT).as_posix()}


def _splits_summary(rows, groups, scores, start_scores, random_scores, seed, test, val) -> list[str]:
    used = [p for p in rows if p["split"] != "none"]
    classes = sorted({p["label"] for p in used})
    parts = ("test", "val", "train")
    count = Counter((p["label"], p["split"]) for p in used)
    split_of = defaultdict(set)
    for p in used:
        split_of[p["lesion_group"]].add(p["split"])
    in_two = sum(len(parts_of_group) > 1 for parts_of_group in split_of.values())
    mean = sum(scores.values()) / len(scores)

    def quantile(q):
        return random_scores[int(q * (len(random_scores) - 1))] if random_scores else float("nan")

    lines = ["# Splits: train, validation and test", "",
             "Written by `python -m src.image.prepare splits`. Every labelled photo goes into train, validation (val) or test, with every "
             "lesion in one split: photos that ISIC's or HAM10000's lesion numbers, or the eye check, name as one lesion form a "
             "group that moves as a whole (group-based). "
             "The Kaggle dataset's own Train and Test folders play no part: the splits are made anew, and the original test photos "
             "are split like every other photo (where they went is shown below). "
             f"Each class gets exactly {test} test and {val} validation photos (stratified); the rest is train, the database that open-book runs retrieve from. The split starts from a random fill "
             f"(seed {seed}, fixed in advance; Python {sys.version_info.major}.{sys.version_info.minor}'s random module) and then swaps lesion groups of equal size between the parts whenever that brings the "
             "test and validation parts closer to their class's own mix of subtype (ISIC diagnosis), how the diagnosis was "
             "confirmed, source (credit line) and original image size. Only these facts, known before any experiment, are used; "
             "never a model's result. The split column of `raw_photos.csv` holds the result; no image is changed.", "",
             "## Photos per class and split", "",
             "| Class | Test | Validation | Train | Photos | Original test photos (test / val / train) |", "|---|---|---|---|---|---|"]

    def originals(group):
        return " / ".join(str(sum(p["kaggle_test"] == "yes" and p["split"] == part for p in group)) for part in parts)

    for label in classes:
        mine = [p for p in used if p["label"] == label]
        lines.append(f"| {label} | {count[(label, 'test')]} | {count[(label, 'val')]} | {count[(label, 'train')]} | {len(mine)} | "
                     f"{originals(mine)} |")
    lines.append(f"| **Total** | {sum(count[(c, 'test')] for c in classes)} | {sum(count[(c, 'val')] for c in classes)} | "
                 f"{sum(count[(c, 'train')] for c in classes)} | {len(used)} | {originals(used)} |")
    left = [p for p in rows if p["split"] == "none"]
    lines += ["", f"- Lesion groups: {len(groups)}, of which {sum(len(ids) > 1 for ids in groups.values())} hold two or more photos; "
              f"groups found in two splits: {in_two}",
              f"- In no split: {len(left)} (" + "; ".join(f"{k}: {n}" for k, n in sorted(Counter(
                  'same photo as another (eye check)' if 'eye check' in p['split_note'] else p['split_note'] for p in left).items()))
              + ")", "",
              "## Balance (0 = the test and validation parts have their class's own mix)", "",
              f"- This split: {mean:.3f} (the random fill it started from: {sum(start_scores.values()) / len(start_scores):.3f})",
              f"- {len(random_scores)} simple random splits under the same rules: median {quantile(0.5):.3f}, 95th percentile "
              f"{quantile(0.95):.3f}, worst {quantile(1):.3f}", "",
              "| Class | Property | Value | Class | Test | Validation | Train |", "|---|---|---|---|---|---|---|"]
    names = {"isic_diagnosis": "subtype", "diagnosis_confirm_type": "confirmed by", "attribution": "source", "size": "original size"}
    for label in classes:
        mine = [p for p in used if p["label"] == label]
        for name in BALANCED:
            share = Counter(_property(p, name) for p in mine)
            if len(share) < 2:
                continue
            for value, n in sorted(share.items(), key=lambda kv: (-kv[1], kv[0])):
                if n / len(mine) < 0.05:
                    continue
                cells = [f"{100 * sum(_property(p, name) == value for p in mine if p['split'] == part) / max(count[(label, part)], 1):.0f}%"
                         for part in parts]
                lines.append(f"| {label} | {names[name]} | {value} | {100 * n / len(mine):.0f}% | " + " | ".join(cells) + " |")
    lines += ["", "Values below 5% of a class are not listed (they are balanced too).", "",
              "## Not done in this step", "", "- No image is changed: the images of each split are made in the next step.", ""]
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------
# The prepared images: every photo of a split fitted whole within one standard size and saved the same way


def _fitted(width: int, height: int) -> tuple[tuple[int, int], bool]:
    """The size a landscape photo of this shape gets: shrunk to fit within SIZE keeping its shape, never enlarged. A size within
    SNAP_PIXELS of SIZE becomes SIZE exactly (the photo's shape is 4:3 up to rounding); the second value says when that happened."""
    scale = min(SIZE[0] / width, SIZE[1] / height)
    if scale >= 1:
        return (width, height), False
    exact = (width * scale, height * scale)
    rounded = (max(1, round(exact[0])), max(1, round(exact[1])))
    if rounded != SIZE and all(abs(e - s) <= SNAP_PIXELS for e, s in zip(exact, SIZE)):
        return SIZE, True
    return rounded, False


def _prepare_image(content: bytes) -> tuple[bytes, list[str]]:
    """One original photo as every model receives it, and what was done to it besides saving it again. Nothing is cut and nothing
    is added: the only changes are turning, converting colours and shrinking the whole picture."""
    changes = []
    with Image.open(io.BytesIO(content)) as original:
        original.load()
        image = ImageOps.exif_transpose(original)
        if original.getexif().get(ORIENTATION, 1) != 1:
            changes.append("turned upright as its EXIF orientation tag says")
        profile = original.info.get("icc_profile")
        if profile:
            source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
            name = ImageCms.getProfileDescription(source).strip() or "unnamed"
            image = ImageCms.profileToProfile(image, source, ImageCms.createProfile("sRGB"), outputMode="RGB")
            changes.append(f"colours converted from its embedded colour profile ({name}) to sRGB")
    if image.mode != "RGB":
        changes.append(f"converted from {image.mode} to RGB")
        image = image.convert("RGB")
    if image.height > image.width:
        image = image.transpose(Image.Transpose.ROTATE_90)
        changes.append("turned a quarter turn to the left (portrait to landscape)")
    size, stretched = _fitted(*image.size)
    if size != image.size:
        changes.append(f"shrunk from {image.width}x{image.height}" + (" (stretched by less than one pixel to make it exactly "
                                                                       f"{size[0]}x{size[1]})" if stretched else ""))
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", **JPEG)  # only these settings: no EXIF, no colour profile, no thumbnail is written
    return buffer.getvalue(), changes or ["saved again only"]


def images(data: str | Path = DATA, show=None) -> dict:
    """Makes the image of every photo in a split from its original file, as data/image/dataset/<split>/<class>/<photo_id>.jpg (an image
    already there with the same bytes is left as it is; one no longer in a split is removed), and writes dataset.csv and the images
    section of dataset_report.md. Nothing in data/image/raw is changed."""
    data = Path(data)
    photos = _read_photos(data, "splits")
    root, table = data / FILES, data / TABLE_NAME
    used = [photos[i] for i in sorted(photos) if photos[i]["split"] != "none"]
    rows, written = [], 0
    for n, p in enumerate(used, start=1):
        source = data / "raw" / p["files"].split(" | ")[0]
        if not source.is_file():
            raise FileNotFoundError(f"{source.as_posix()} not found: put the original dataset in data/image/raw (the Train and Test folders "
                                    "from the Kaggle ZIP)")
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != p["sha256"]:
            raise ValueError(f"{source.as_posix()} differs from raw_photos.csv: the original dataset must stay as downloaded (extract "
                             "the Kaggle ZIP again)")
        prepared, changes = _prepare_image(content)
        path = f"{p['split']}/{p['label']}/{p['photo_id']}.jpg"
        file = root / path
        if not file.is_file() or file.read_bytes() != prepared:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(prepared)
            written += 1
        with Image.open(io.BytesIO(prepared)) as image:
            width, height = image.size
        rows.append({"split": p["split"], "label": p["label"], "path": path, "sha256": hashlib.sha256(prepared).hexdigest(),
                     "lesion_group": p["lesion_group"], "photo_id": p["photo_id"], "kaggle_test": p["kaggle_test"], "width": width,
                     "height": height, "bytes": len(prepared), "changes": "; ".join(changes)})
        if show and (n % 250 == 0 or n == len(used)):
            show(f"{n} of {len(used)} images made")
    listed = {row["path"] for row in rows}
    stale = sorted(f for split in SPLITS if (root / split).is_dir() for f in (root / split).rglob("*")
                   if f.is_file() and f.suffix.lower() in MEDIA_TYPES and f.relative_to(root).as_posix() not in listed)
    for file in stale:  # an image of a photo that is no longer in this split (or in no split): never left for a run to find
        file.unlink()
    for folder in sorted((d for split in SPLITS if (root / split).is_dir() for d in (root / split).rglob("*") if d.is_dir()),
                         key=lambda d: len(d.parts), reverse=True):
        if not any(folder.iterdir()):
            folder.rmdir()
    rows.sort(key=lambda row: row["path"])
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=IMAGE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    items = load_dataset(table, root)  # the evaluation's own check: the images on disk are exactly the ones the table lists
    training_classes(items)  # and every validation and test class has training images
    _write_section(data, "images", _images_summary(rows, items, photos))
    return {"rows": rows, "written": written, "removed": len(stale), "table": table.as_posix(), "folder": root.as_posix(),
            "summary": (data / REPORT).as_posix()}


def _images_summary(rows: list[dict], items: list, photos: dict[str, dict]) -> list[str]:
    parts = ("test", "val", "train")
    labels = sorted({row["label"] for row in rows})
    count = Counter((row["label"], row["split"]) for row in rows)
    standard = [row for row in rows if (row["width"], row["height"]) == SIZE]
    other = [row for row in rows if (row["width"], row["height"]) != SIZE]

    def having(text):
        return [row for row in rows if text in row["changes"]]

    lines = ["# Images: the prepared dataset", "",
             "Written by `python -m src.image.prepare images`. Every photo in a split is made from its original file in `data/image/raw` (checked "
             "against its fingerprint in `raw_photos.csv`; nothing in `data/image/raw` is changed) and saved as "
             f"`{(DATA / FILES).as_posix()}/<split>/<class>/<photo_id>.jpg`. `{TABLE_NAME}` lists every image: its split, class, file, SHA-256, "
             "lesion group, whether it is an original test photo, its size and what was done to it. The evaluation checks every "
             "image against that table before a run, so every model gets byte-identical images.", "",
             "Every photo is prepared the same way, in this order:", "",
             "1. Upright as a viewer shows it (its EXIF orientation tag, where it has one).",
             "2. Colours in sRGB: a photo with an embedded colour profile is converted from that profile; a photo without one is "
             "already taken as sRGB, which is what a missing profile means.",
             "3. Landscape: a portrait photo is turned a quarter turn to the left. A dermoscopic photo has no up or down, so nothing "
             "is lost.",
             f"4. Fitted whole within {SIZE[0]} x {SIZE[1]} pixels (HAM10000's own size, which most photos already have), keeping its "
             "shape: shrunk with Lanczos filtering, never enlarged. Nothing is cut and nothing is added (no crop, no border). Where "
             f"the fitted size is within {SNAP_PIXELS} pixel of {SIZE[0]} x {SIZE[1]} (a photo whose shape is 4:3 up to rounding, "
             f"such as 767 x 576), it is made exactly {SIZE[0]} x {SIZE[1]}: a stretch of less than one pixel.",
             f"5. Saved as JPEG at quality {JPEG['quality']} with full colour resolution (4:4:4 chroma), without metadata (no EXIF, "
             "no colour profile, no thumbnail).", "",
             f"Made with Pillow {PIL.__version__} (libjpeg-turbo {features.version('jpg')}, Little CMS {features.version('littlecms2')}).",
             "", "## Images per class and split", "",
             "| Class | Test | Validation | Train | Images |", "|---|---|---|---|---|"]
    for label in labels:
        lines.append(f"| {label} | " + " | ".join(str(count[(label, part)]) for part in parts)
                     + f" | {sum(count[(label, part)] for part in parts)} |")
    lines.append("| **Total** | " + " | ".join(str(sum(count[(label, part)] for label in labels)) for part in parts)
                 + f" | {len(rows)} |")
    lines += ["", "## Sizes", "",
              f"- Exactly {SIZE[0]} x {SIZE[1]}: {len(standard)} of {len(rows)} images ({100 * len(standard) / len(rows):.1f}%)",
              f"- Another size: {len(other)}. A photo that is not 4:3 keeps its own shape, because nothing is cut or added; so the "
              "shape of these images still differs from the others.", "",
              f"| Class | Images | Exactly {SIZE[0]} x {SIZE[1]} | Another size | The most common other sizes (images) |",
              "|---|---|---|---|---|"]
    for label in labels:
        mine = [row for row in rows if row["label"] == label]
        others = Counter(f"{row['width']}x{row['height']}" for row in mine if (row["width"], row["height"]) != SIZE)
        common = ", ".join(f"{size} ({n})" for size, n in sorted(others.items(), key=lambda kv: (-kv[1], kv[0]))[:3]) or "–"
        lines.append(f"| {label} | {len(mine)} | {len(mine) - sum(others.values())} | {sum(others.values())} | {common} |")
    turned, profiles = sorted(row["photo_id"] for row in having("quarter turn")), having("colour profile")
    profile_names = sorted(f"{row['photo_id']}: {re.search(PROFILE_NAME, row['changes']).group(1)}" for row in profiles)
    lines += ["", "## What was done besides saving again", "",
              f"- Shrunk: {len(having('shrunk from'))} (of which made exactly {SIZE[0]} x {SIZE[1]} by a stretch of less than one "
              f"pixel: {len(having('stretched'))})",
              f"- Already {SIZE[0]} x {SIZE[1]} or smaller (saved again only, never enlarged): {len(having('saved again only'))}",
              f"- Turned upright by an EXIF orientation tag: {len(having('EXIF orientation'))}",
              f"- Turned from portrait to landscape: {len(turned)}" + (f" ({', '.join(turned)})" if turned else ""),
              f"- Colours converted from an embedded colour profile: {len(profiles)}"
              + (f" ({'; '.join(profile_names)})" if profiles else ""),
              f"- Converted to RGB from another colour mode: {len(having('to RGB'))}", "",
              "## Original test photos", "",
              f"`kaggle_test` marks the {sum(row['kaggle_test'] == 'yes' for row in rows)} photos from the Kaggle dataset's Test folder: "
              + ", ".join(f"{part} {sum(row['kaggle_test'] == 'yes' and row['split'] == part for row in rows)}" for part in parts)
              + ". They can be run on their own as a closed-book check against results on the original test folder; not "
              "open-book, because most of them are in train, the database that open-book runs retrieve from.", "",
              "## Fingerprints", "",
              "One hash per split over every image's split, class, file, SHA-256 and lesion group (it changes if any of them changes). "
              "A run's run card shows the same hash as `images_sha256` when the run answered the whole split.", ""]
    for part in parts:
        lines.append(f"- {part}: `{items_hash([item for item in items if item.split == part])}`")
    lines += ["", "## Not done in this step", "",
              "- No crop, border, colour correction, hair removal or sharpening: every lesion is shown whole, as photographed.",
              f"- Photos in no split ({sum(p['split'] == 'none' for p in photos.values())}) get no image.", ""]
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------
# The shortcut check: how much simple properties of the prepared images, which say nothing about the lesion, reveal the class


def _measure(content: bytes) -> dict:
    """The simple properties of one prepared image (all on the 0-255 scale): how dark its corners are (the dermoscope's round
    field leaves dark corners in some collections), the colour of its outer frame (mostly skin, lighting and camera; rarely the
    lesion) and its sharpness (the mean brightness step between neighbouring pixels). Integer sums only, so every computer gets the
    same values."""
    with Image.open(io.BytesIO(content)) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    grey = image.convert("L")
    side = max(1, min(round(EDGE * min(width, height)), min(width, height) // 2))
    boxes = ((0, 0, side, side), (width - side, 0, width, side), (0, height - side, side, height), (width - side, height - side, width, height))
    corners = sum(ImageStat.Stat(grey.crop(box)).sum[0] for box in boxes) / (4 * side * side)
    frame = width * height - (width - 2 * side) * (height - 2 * side)
    whole, inner = ImageStat.Stat(image).sum, ImageStat.Stat(image.crop((side, side, width - side, height - side))).sum
    steps = (ImageStat.Stat(ImageChops.difference(grey.crop((1, 0, width, height)), grey.crop((0, 0, width - 1, height)))).sum[0]
             + ImageStat.Stat(ImageChops.difference(grey.crop((0, 1, width, height)), grey.crop((0, 0, width, height - 1)))).sum[0])
    return {"corners": corners, **{f"edge_{c}": (whole[n] - inner[n]) / frame for n, c in enumerate("rgb")},
            "sharpness": steps / ((width - 1) * height + width * (height - 1))}


def _ranked(train: list[list[float]], queries: list[list[float]]) -> list[list[tuple[float, int]]]:
    """For each query, every training row with its squared distance, nearest first (the row number breaks ties, so the order is
    fixed; the vote below never depends on it)."""
    return [sorted((sum((a - b) ** 2 for a, b in zip(query, row)), j) for j, row in enumerate(train)) for query in queries]


def _vote(ranked: list[tuple[float, int]], classes: list[str], weight: dict[str, float], k: int) -> str:
    """The class of the k training rows nearest to one query, each training class counting equally (as in the test split). Every
    row as near as the k-th votes too, so rows at the same distance are never chosen by their order; a tied vote goes to the class
    of the nearest row, then to the first class name."""
    limit = ranked[k - 1][0] + 1e-9
    score, nearest = Counter(), {}
    for distance, j in ranked:
        if distance > limit:
            break
        score[classes[j]] += weight[classes[j]]
        nearest.setdefault(classes[j], distance)
    best = max(score.values())
    return min((c for c in score if score[c] >= best - 1e-12), key=lambda c: (nearest[c], c))


def _balanced_accuracy(truth: list[str], predicted: list[str]) -> float:
    per_class = defaultdict(list)
    for t, p in zip(truth, predicted):
        per_class[t].append(t == p)
    return sum(sum(hits) / len(hits) for hits in per_class.values()) / len(per_class)


def shortcuts(data: str | Path = DATA) -> dict:
    """Measures how much simple properties of the prepared images (size, file size, dark corners, edge colour, sharpness) reveal
    each image's class and source, and writes the shortcut section of dataset_report.md. Nothing is changed."""
    data = Path(data)
    table, root = data / TABLE_NAME, data / FILES
    if not table.is_file():
        raise FileNotFoundError(f"{table.as_posix()} not found: run python -m src.image.prepare images first")
    items = load_dataset(table, root)  # the images on disk are exactly the prepared ones
    photos = _read_photos(data)
    with table.open(encoding="utf-8", newline="") as f:
        rows = {row["path"]: row for row in csv.DictReader(f)}
    images = []
    for item in sorted(items, key=lambda item: item.path):
        row = rows[item.path]
        images.append({"split": item.split, "label": item.label, "lesion_group": item.lesion_group or item.path,
                       "source": photos[row["photo_id"]]["attribution"] or "(not recorded)", "width": int(row["width"]),
                       "height": int(row["height"]), "bytes": int(row["bytes"]), **_measure((root / item.path).read_bytes())})
    part = {name: [image for image in images if image["split"] == name] for name in ("train", "val", "test")}
    if not part["train"] or not part["val"] or not part["test"]:
        raise ValueError("the shortcut check needs images in train, val and test")
    labels = sorted({image["label"] for image in part["train"]})
    results = {}
    for name, columns in [*SHORTCUT_PROPERTIES.items(), ("all five together", [c for cs in SHORTCUT_PROPERTIES.values() for c in cs])]:
        centre = {c: statistics.fmean(image[c] for image in part["train"]) for c in columns}
        spread = {c: statistics.pstdev([image[c] for image in part["train"]]) or 1.0 for c in columns}

        def values(image):
            return [(image[c] - centre[c]) / spread[c] for c in columns]

        train = [values(image) for image in part["train"]]
        ranked = {split: _ranked(train, [values(image) for image in part[split]]) for split in ("val", "test")}
        result = {}
        for target in ("label", "source"):
            classes = [image[target] for image in part["train"]]
            weight = {c: 1 / n for c, n in Counter(classes).items()}
            ks = [k for k in SHORTCUT_NEIGHBOURS if k <= len(train)]
            truth = {split: [image[target] for image in part[split]] for split in ("val", "test")}
            on_val = {k: _balanced_accuracy(truth["val"], [_vote(r, classes, weight, k) for r in ranked["val"]]) for k in ks}
            k = min(ks, key=lambda k: (-on_val[k], k))  # the best on the validation split; the smallest of equals
            predicted = [_vote(r, classes, weight, k) for r in ranked["test"]]
            scored = summarize(truth["test"], predicted, sorted(set(classes) | set(truth["test"])),
                               lesions=[image["lesion_group"] for image in part["test"]])
            result[target] = {"k": k, "balanced_accuracy": scored["balanced_accuracy"], "ci95": scored["balanced_accuracy_ci95"],
                              "recall": {c: v["recall"] for c, v in scored["per_class"].items()}}
        results[name] = result
    _write_section(data, "shortcuts", _shortcuts_summary(images, labels, results))
    return {"results": results, "labels": labels, "sources": sorted({image["source"] for image in images}),
            "summary": (data / REPORT).as_posix()}


def _shortcuts_summary(images: list[dict], labels: list[str], results: dict) -> list[str]:
    sources = sorted({image["source"] for image in images})
    test = [image for image in images if image["split"] == "test"]

    def median(label, value):
        return statistics.median(value(image) for image in images if image["label"] == label)

    def percent(score, interval):
        return f"{100 * score:.1f}% ({100 * interval[0]:.1f}–{100 * interval[1]:.1f}%)"

    lines = ["# Shortcut check: what the images reveal besides the lesion", "",
             "Written by `python -m src.image.prepare shortcuts` from the prepared images; it changes nothing. A model should recognise the "
             "lesion, not where or how the photo was taken. This step measures how much simple properties of the images, which are "
             "mostly about how the photo was taken rather than about the lesion (file size and sharpness also depend a little on "
             "the lesion's own texture), reveal each image's class: where they do, a model, or the image search of an open-book "
             "run, could get the class partly right without looking at the lesion. It shows how much these clues could tell, not "
             "whether a model uses them.", "",
             "## Where the photos come from (ISIC credit line)", "",
             "| Class | Images | " + " | ".join(sources) + " |", "|---|---|" + "---|" * len(sources)]
    for label in labels:
        mine = [image for image in images if image["label"] == label]
        lines.append(f"| {label} | {len(mine)} | " + " | ".join(
            f"{sum(image['source'] == s for image in mine)} ({100 * sum(image['source'] == s for image in mine) / len(mine):.0f}%)"
            for s in sources) + " |")
    lines += ["", "Where a class comes from one source only, the source itself is a clue to the class: any difference in camera, "
              "lighting or file handling between sources can stand in for the diagnosis. Between classes from the same source it "
              "tells nothing.", "",
              "## The properties per class (median over all splits)", "",
              f"| Class | Not {SIZE[0]} x {SIZE[1]} | File size (kB) | Dark corners (0–255) | Edge brightness (0–255) | Sharpness |",
              "|---|---|---|---|---|---|"]
    for label in labels:
        mine = [image for image in images if image["label"] == label]
        other = sum((image["width"], image["height"]) != SIZE for image in mine)
        lines.append(f"| {label} | {other} ({100 * other / len(mine):.0f}%) | {median(label, lambda i: i['bytes'] / 1000):.0f} | "
                     f"{median(label, lambda i: i['corners']):.0f} | "
                     f"{median(label, lambda i: (i['edge_r'] + i['edge_g'] + i['edge_b']) / 3):.0f} | "
                     f"{median(label, lambda i: i['sharpness']):.1f} |")
    lines += ["", "Dark corners: the mean brightness of the four corner squares (a tenth of the shorter side); edge brightness: the "
              "outer frame of the same width; sharpness: the mean brightness step between neighbouring pixels.", "",
              "## How much the properties reveal", "",
              "For each property, a nearest-neighbour vote predicts each test image's class, and its source, from the training "
              "images most alike in that property alone. Each training class counts equally, as in the test split; the number of "
              "neighbours (from " + ", ".join(map(str, SHORTCUT_NEIGHBOURS)) + ") is chosen on the validation split. Chance is "
              f"{100 / len(labels):.1f}% for the {len(labels)} classes and {100 / len(sources):.0f}% for the {len(sources)} "
              "sources; the 95% intervals resample whole lesions, as in the evaluation.", "",
              "| Property | Class: balanced accuracy on test (95% interval) | Neighbours | Source: balanced accuracy on test (95% interval) "
              "| Neighbours |", "|---|---|---|---|---|"]
    for name, result in results.items():
        lines.append(f"| {name} | {percent(result['label']['balanced_accuracy'], result['label']['ci95'])} | {result['label']['k']} | "
                     f"{percent(result['source']['balanced_accuracy'], result['source']['ci95'])} | {result['source']['k']} |")
    together = results["all five together"]["label"]["recall"]
    lines += ["", "Recall per class on test, all five properties together:", "",
              "| Class | Recall |", "|---|---|", *(f"| {label} | {100 * together[label]:.0f}% |" for label in labels), "",
              "## Not done in this step", "",
              f"- No image, label or split is changed: it measures only ({len(test)} test images, with the training images as the "
              "ones they are compared with).", ""]
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.image.prepare", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory", help="list, open and fingerprint every file in data/image/raw, and group identical copies")
    i = sub.add_parser("isic", help="download the ISIC Archive record of every ISIC number (no label is changed)")
    i.add_argument("--limit", type=int, help="download only the next N records and show what they hold")
    d = sub.add_parser("identity", help="compare every photo with the ISIC Archive's own picture of its ISIC number")
    d.add_argument("--limit", type=int, help="download only the next N pictures and show how they compare")
    sub.add_parser("ham10000", help="compare the photos that are also in HAM10000 with HAM10000's own metadata (no label is changed)")
    sub.add_parser("labels", help="label every unique photo from its ISIC record by the naming rule (no image is changed)")
    sub.add_parser("duplicates", help="compare every photo with every other one, to find different files of one picture or lesion")
    m = sub.add_parser("lesions", help="compare the skin pattern of every two photos of a class, to find second photos of one lesion")
    m.add_argument("--limit", type=int, help="compare only the first N photos of each class and show how long the full search takes")
    sub.add_parser("splits", help="put every labelled photo into train, val or test, every lesion in one split, balanced per class")
    sub.add_parser("images", help="make the image of every photo in a split, fitted whole within 600 x 450, and write dataset.csv")
    sub.add_parser("shortcuts", help="measure how much simple image properties (size, file size, corners, edges, sharpness) reveal the class")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inventory":
            done = inventory()
            print(f"data/image/raw: {done['files']} image files, {done['unreadable']} unreadable, {done['other_problems']} with another "
                  f"problem, {done['other_files']} other files")
            print(f"{done['unique']} unique photos; {done['duplicate_groups']} groups of identical files")
            print(f"Written: {done['table']} and {done['summary']}. Nothing in data/image/raw was changed. Commit both files.")
        elif args.command == "isic":
            if args.limit is not None and args.limit < 1:
                raise ValueError(f"--limit must be at least 1 (got {args.limit})")
            done = isic(args.limit)
            if args.limit:  # a sample run shows what the records hold, to check the format before the full download
                for row in done["new"]:
                    print(f"{row['isic_id']}: {row['status']}; {_chain(row, 5)}; confirmed by {row['diagnosis_confirm_type'] or '-'}; "
                          f"image type {row['image_type'] or '-'}; lesion {row['lesion_id'] or '-'}; patient {row['patient_id'] or '-'}; "
                          f"licence {row['copyright_license'] or '-'}; credit {row['attribution'] or '-'}")
                if done["fields"]:
                    print(f"Field names in the first record: {', '.join(done['fields'])}")
            print(f"ISIC records: {done['have']} of {done['total']} downloaded ({len(done['new'])} just now)")
            if done["written"]:
                print(f"{done['found']} found in the ISIC Archive, {done['total'] - done['found']} not found. Written: {done['table']} "
                      f"and {done['summary']}. No label was changed. Commit both files.")
            else:
                print("raw_photos.csv is written once every record is downloaded: run the same command without --limit.")
        elif args.command == "identity":
            if args.limit is not None and args.limit < 1:
                raise ValueError(f"--limit must be at least 1 (got {args.limit})")
            done = identity(args.limit)
            if args.limit:  # a sample run shows how the pictures compare, to check the method before the full download
                for row in done["rows"]:
                    print(f"{row['isic_id']}: {row['verdict']}; photo {row['width']}x{row['height']}, ISIC picture "
                          f"{row['isic_width'] or '-'}x{row['isic_height'] or '-'}; shape difference {row['shape_difference_pct']}%; "
                          f"hash distance {row['hash_distance']}; mean difference {row['mean_difference']}")
            print(f"Compared: {done['compared']} of {done['total']} ISIC numbers ({done['new']} pictures downloaded just now)")
            if done["written"]:
                verdicts = Counter(row["verdict"] for row in done["rows"])
                print("; ".join(f"{verdict}: {n}" for verdict, n in sorted(verdicts.items())) + f". Written: {done['table']} and "
                      f"{done['summary']}. Nothing was changed. Commit both files.")
                if verdicts["check by eye"]:
                    print(f"Photos to check by eye are listed in {done['summary']}.")
            else:
                print("raw_photos.csv is written once every picture is downloaded: run the same command without --limit.")
        elif args.command == "ham10000":
            done = ham10000()
            print(f"ISIC numbers: {done['total']}; also in HAM10000: {done['in_ham']}")
            for key, name in (("diagnosis", "Diagnosis group"), ("confirmation", "How confirmed"), ("lesion_group", "Lesion groups")):
                print(f"{name}: {_tally(row[key] for row in done['rows'])}")
            print(f"Written: {done['table']} and {done['summary']}. No label was changed. Commit both files.")
        elif args.command == "labels":
            done = labels()
            rows = done["rows"]
            print(f"Photos: {done['photos']}; labelled: {done['labelled']}; without a class: {done['photos'] - done['labelled']}")
            print("Per class: " + "; ".join(f"{name}: {n}" for name, n in sorted(Counter(p["label"] for p in rows if p["label"]).items())))
            print("Against the Kaggle folders: " + _tally(p["label_vs_folder"] for p in rows))
            print(f"Written: {done['table']} and {done['summary']}. No image was changed. Commit both files.")
        elif args.command == "duplicates":
            done = duplicates()
            tiers = Counter(row["tier"] for row in done["flagged"])
            print(f"Photos compared: {done['photos']} ({done['pairs']} pairs). Flagged: {len(done['flagged'])} "
                  f"(near-identical: {tiers['near-identical']}; similar: {tiers['similar']})")
            print(f"Known same-lesion pairs (ISIC lesion numbers) found: {done['found_same']} of {done['known_same']}")
            print(f"Written: {done['table']} and {done['summary']}. No image was changed. Commit both files.")
        elif args.command == "lesions":
            if args.limit is not None and args.limit < 2:
                raise ValueError(f"--limit must be at least 2 (got {args.limit})")
            done = lesions(limit=args.limit, show=print)
            print(f"Photos compared: {done['photos']} ({done['pairs']} pairs within classes) in {done['seconds']:.0f} s. Pairs with at "
                  f"least {THRESHOLDS[0]} inliers: {len(done['rows'])}")
            print(f"Known same-lesion pairs (ISIC lesion numbers) with at least {THRESHOLDS[0]} inliers: {done['found_same']} of "
                  f"{done['known_same']}")
            if args.limit:
                minutes = max(1, round(done["estimate_seconds"] / 60))
                print(f"The full search compares {done['all_pairs']} pairs: about {minutes} minute{'s' * (minutes > 1)} on this computer. "
                      "Nothing was written: run the same command without --limit.")
            else:
                print(f"Written: {done['table']} and {done['summary']}. No image was changed. Commit both files.")
        elif args.command == "splits":
            done = splits()
            counts = done["counts"]
            for label in sorted({label for label, _ in counts}):
                print(f"{label}: test {counts[(label, 'test')]}, val {counts[(label, 'val')]}, train {counts[(label, 'train')]}")
            r = done["random"]
            print(f"Balance (0 = each part mirrors its class): {done['imbalance']:.3f}; simple random splits: median "
                  f"{r[len(r) // 2]:.3f}, worst {r[-1]:.3f}")
            print(f"Written: {done['table']} and {done['summary']}. No image was changed. Commit both files.")
        elif args.command == "images":
            done = images(show=print)
            rows = done["rows"]
            parts = Counter(row["split"] for row in rows)
            standard = sum((row["width"], row["height"]) == SIZE for row in rows)
            print(f"Images: {len(rows)} (test {parts['test']}, val {parts['val']}, train {parts['train']}); exactly {SIZE[0]}x{SIZE[1]}: "
                  f"{standard}; another size: {len(rows) - standard}")
            print(f"Written: {done['folder']}/ ({done['written']} images new or changed, {done['removed']} removed), {done['table']} "
                  f"and {done['summary']}. Nothing in data/image/raw was changed. Commit {done['table']} and {done['summary']}; the "
                  "images stay on this computer.")
        elif args.command == "shortcuts":
            done = shortcuts()
            print(f"Balanced accuracy on test from each property alone (chance: classes {100 / len(done['labels']):.1f}%, "
                  f"sources {100 / len(done['sources']):.0f}%):")
            for name, result in done["results"].items():
                print(f"  {name}: classes {100 * result['label']['balanced_accuracy']:.1f}%, sources "
                      f"{100 * result['source']['balanced_accuracy']:.1f}%")
            print(f"Written: {done['summary']}. Nothing was changed. Commit it.")
    except (ValueError, FileNotFoundError) as exc:  # expected, explainable problems: a message, not a traceback
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])
