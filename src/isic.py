"""Checks the dataset's folder labels against the diagnoses recorded in the ISIC Archive.

For each image it fetches the public ISIC record of its ISIC number, keeps every field whose name mentions a diagnosis
(plus benign/malignant, lesion ID and licence), and reports which of the image's dataset labels that record supports.
The raw records are cached in data/isic_cache/ (not committed) so the check can be resumed and re-read without
fetching again. By default it checks every image with conflicting labels and every testing image.

From the check it also writes label_corrections.csv: every testing image whose folder label the ISIC record contradicts,
with the label ISIC supports. A test run is scored against the folder labels and, separately, against these corrected
labels; the dataset's folders are never changed.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

API = "https://api.isic-archive.com/api/v2/images/{}/"
USER_AGENT = "thesis-research-isic-label-check/1.0 (academic research)"
EXTRA_FIELDS = {"benign_malignant", "lesion_id", "copyright_license"}
SUPPORTS = "ISIC supports: "  # the verdict when the ISIC record supports exactly one of the image's dataset labels
CORRECTION_FIELDS = ["path", "isic_id", "folder_label", "corrected_label", "isic_fields"]

# Words in an ISIC diagnosis that support each dataset class. Shown next to the raw diagnosis text, which stays the
# evidence: a human reads the table, these words only sort it.
LABEL_TERMS = {
    "actinic keratosis": ["actinic keratosis", "actinic"],
    "basal cell carcinoma": ["basal cell carcinoma"],
    "dermatofibroma": ["dermatofibroma"],
    "melanoma": ["melanoma"],
    "nevus": ["nevus", "naevus"],
    "pigmented benign keratosis": ["pigmented benign keratosis", "benign keratosis", "solar lentigo", "lichen planus",
                                   "lichenoid"],
    "seborrheic keratosis": ["seborrheic keratosis", "seborrhoeic keratosis"],
    "squamous cell carcinoma": ["squamous cell carcinoma"],
    "vascular lesion": ["vascular", "hemangioma", "haemangioma", "angioma", "angiokeratoma", "pyogenic granuloma"],
}


def _flatten(value, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: value}


def diagnosis_fields(record: dict) -> dict[str, str]:
    """The diagnosis-related fields of one ISIC record, as dotted key -> text."""
    flat = _flatten(record)
    return {k: str(v) for k, v in sorted(flat.items())
            if v not in (None, "") and ("diagnosis" in k.split(".")[-1] or k.split(".")[-1] in EXTRA_FIELDS)}


def supported_labels(fields: dict[str, str], labels: list[str]) -> list[str]:
    """Which of ``labels`` the record's diagnosis text supports (licence and lesion ID are ignored)."""
    text = " ".join(v.lower() for k, v in fields.items() if k.split(".")[-1] not in {"lesion_id", "copyright_license"})
    return [label for label in labels if any(term in text for term in LABEL_TERMS.get(label, [label]))]


def fetch(isic_id: str, cache: Path, delay: float = 0.3, attempts: int = 3) -> dict:
    """The ISIC record (cached as data/isic_cache/<id>.json), or {"_error": ...} when it cannot be fetched."""
    path = cache / f"{isic_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    request = urllib.request.Request(API.format(isic_id), headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(1, attempts + 1):
        time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                record = json.loads(response.read().decode("utf-8"))
            cache.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            return record
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"_error": "not found in the ISIC Archive (404)"}
            error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = str(exc)
        time.sleep(2 ** attempt)
    return {"_error": error}


def check(manifest: str | Path, out_dir: str | Path, cache: str | Path = "data/isic_cache", everything: bool = False,
          delay: float = 0.3) -> Path:
    """Writes isic_check.csv (one row per checked ISIC number) and isic_check.md (summary) to ``out_dir``."""
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    entries = defaultdict(list)  # isic_id -> ["split:label", ...]
    for r in rows:
        if r["isic_id"]:
            entries[r["isic_id"]].append(f"{r['split']}:{r['label']}")
    conflicted = {i for i, e in entries.items() if len({x.split(":", 1)[1] for x in e}) > 1}
    tested = {i for i, e in entries.items() if any(x.startswith("test:") for x in e)}
    ids = sorted(entries) if everything else sorted(conflicted | tested)
    cache = Path(cache)

    table = []
    for n, isic_id in enumerate(ids, start=1):
        record = fetch(isic_id, cache, delay)
        labels = sorted({x.split(":", 1)[1] for x in entries[isic_id]})
        if "_error" in record:
            fields, support, verdict = {}, [], f"not checked: {record['_error']}"
        else:
            fields = diagnosis_fields(record)
            support = supported_labels(fields, labels)
            if not any("diagnosis" in k.split(".")[-1] for k in fields):
                verdict = "no diagnosis in the ISIC record"
            elif len(support) == 1:
                verdict = SUPPORTS + support[0]
            elif support:
                verdict = "ISIC supports several labels: " + ", ".join(support)
            else:
                verdict = "ISIC supports none of the dataset labels"
        licence = next((v for k, v in fields.items() if k.endswith("copyright_license")), "not recorded")
        table.append({"isic_id": isic_id, "dataset_labels": " | ".join(sorted(entries[isic_id])),
                      "conflict": "yes" if isic_id in conflicted else "no", "in_test": "yes" if isic_id in tested else "no",
                      "isic_fields": "; ".join(f"{k}={v}" for k, v in fields.items()), "licence": licence,
                      "verdict": verdict})
        if n % 25 == 0 or n == len(ids):
            print(f"{n}/{len(ids)} checked")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "isic_check.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]) if table else ["isic_id"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)

    summary = defaultdict(Counter)
    for row in table:
        labels = sorted({x.split(":", 1)[1] for x in row["dataset_labels"].split(" | ")})
        if row["conflict"] == "yes":
            key = "conflict: " + " vs ".join(labels) + (" (a copy is in test)" if row["in_test"] == "yes" else " (training only)")
        else:
            key = f"test image, no conflict: {labels[0]}"
        summary[key][row["verdict"]] += 1
    licences = Counter(row["licence"] for row in table)
    lines = ["# ISIC Archive label check", "",
             f"Checked {len(table)} ISIC numbers ({'all images' if everything else 'images with conflicting labels and all testing images'}).",
             "Evidence: the diagnosis fields of each public ISIC record (full rows in isic_check.csv; raw records cached in data/isic_cache/).", "",
             "| Group | Verdict | Images |", "|---|---|---|"]
    for key in sorted(summary):
        for verdict, n in summary[key].most_common():
            lines.append(f"| {key} | {verdict} | {n} |")
    lines += ["", "## Licences recorded by ISIC", "", "| Licence | Images |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in licences.most_common()]
    report = out_dir / "isic_check.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    fixes, undecided = write_corrections(manifest, out_dir / "isic_check.csv", out_dir / "label_corrections.csv")
    print(f"{len(fixes)} testing images have a folder label that ISIC contradicts (label_corrections.csv)")
    if undecided:
        print(f"{len(undecided)} testing images could not be decided from ISIC and keep their folder label: {undecided[:10]}")
    return report


def write_corrections(manifest: str | Path, check_csv: str | Path, out: str | Path) -> tuple[list[dict], list[str]]:
    """Writes ``out``: each testing image whose folder label the ISIC record contradicts, with the label ISIC supports.

    Returns the corrections and the testing images that cannot be decided (no ISIC record, or a record that supports none
    or several of the image's dataset labels); those keep their folder label.
    """
    with open(manifest, encoding="utf-8") as f:
        images = [r for r in csv.DictReader(f) if r["split"] == "test"]
    with open(check_csv, encoding="utf-8") as f:
        checked = {r["isic_id"]: r for r in csv.DictReader(f)}
    fixes, undecided = [], []
    for image in images:
        row = checked.get(image["isic_id"])
        if row is None or not row["verdict"].startswith(SUPPORTS):
            undecided.append(image["path"])
        elif row["verdict"][len(SUPPORTS):] != image["label"]:
            fixes.append({"path": image["path"], "isic_id": image["isic_id"], "folder_label": image["label"],
                          "corrected_label": row["verdict"][len(SUPPORTS):], "isic_fields": row["isic_fields"]})
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CORRECTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(fixes)
    return fixes, undecided


def read_corrections(path: str | Path, labels: list[str], images: dict[str, str]) -> dict[str, str]:
    """Corrected label by path for the images in ``images`` (path -> folder label) that label_corrections.csv lists.

    Refuses a corrected label that is not an answer option, and a listed image whose folder label differs from the one
    in the file (the file was written for another version of the dataset).
    """
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["corrected_label"] not in labels:
            raise ValueError(f"{path}: corrected label {r['corrected_label']!r} of {r['path']} is not an answer option")
        if r["path"] in images and images[r["path"]] != r["folder_label"]:
            raise ValueError(f"{path}: {r['path']} is labelled {images[r['path']]!r} in the dataset, "
                             f"not {r['folder_label']!r}; write the corrections again with the isic command")
    return {r["path"]: r["corrected_label"] for r in rows if r["path"] in images}
