"""PAD-UFES-20, the clinic cohort of Version 2: smartphone photos of skin lesions, each with the clinical record of the same visit
(Pacheco et al., Data in Brief, 2020; Mendeley Data, doi 10.17632/zr7vgbcyr2.1). One command per step:

    python -m src.build.clinic_patients.prepare inventory

The original download stays on this computer: the files as downloaded in data/clinic_patients/downloads/ (for example the Mendeley ZIP), and
their unpacked contents in data/clinic_patients/raw/ (metadata.csv and the photos, in whatever folders the download holds).

``inventory`` only reads. It changes no file of the download, asks no service and decides nothing about the dataset. It writes what
the download holds, so that the data sheet states facts and not what a web page or a memory says:
- build/raw_images.csv: every photo named in metadata.csv, with its file, size in bytes, SHA-256, format and width and height;
- build/columns.csv: every column of metadata.csv, how often it is filled, its values with their counts (or, for a column of
  numbers, the lowest, median and highest), and whether its value stays the same for all photos of one patient or of one lesion;
- build/downloads.csv: the files in downloads/, with their size and SHA-256;
- the inventory section of dataset_report.md: the counts of photos, lesions and patients, each class with its photos, lesions,
  patients and how the ``biopsed`` column is filled, the photos named in the table without a file and the files not named in it,
  identical files, lesions with more than one diagnosis or patient, and how many clinical fields are filled, by class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from ...common import write_section
from ...datasets import MEDIA_TYPES

DATA = Path("data/clinic_patients")  # this cohort's folder: committed tables and report; downloads/ and raw/ stay on this computer
RAW, DOWNLOADS = "raw", "downloads"
REPORT = "dataset_report.md"
STEPS = ("inventory",)  # the report's sections, in the order the steps run
REQUIRED = ("img_id", "patient_id", "lesion_id", "diagnostic", "biopsed")  # the columns every later step depends on
NOT_CLINICAL = set(REQUIRED)  # identifiers, the label and the biopsy column: not part of the clinical record
EMPTY = {"", "NAN", "NA", "N/A", "NONE", "NULL", "UNK", "UNKNOWN"}  # how a gap may be written (compared in capitals); reported as found
LISTED_VALUES = 20  # a column with at most this many different values has every value listed with its count
IMAGE_COLUMNS = ["img_id", "file", "bytes", "sha256", "format", "width", "height"]
COLUMN_COLUMNS = ["column", "filled", "empty", "distinct", "level", "values"]
DOWNLOAD_COLUMNS = ["file", "bytes", "sha256"]
REPORT_HEAD = ["# Preparation of the PAD-UFES-20 cohort", "",
               "Made by `python -m src.build.clinic_patients.prepare <step>`, one section per step, in the order the steps run. Each step rewrites "
               "only its own section.", ""]
EXAMPLES = 10  # how many names a report line lists before it says "and N more"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _names(items, n: int = EXAMPLES) -> str:
    items = sorted(items)
    shown = ", ".join(f"`{x}`" for x in items[:n])
    return shown + (f" and {len(items) - n} more" if len(items) > n else "")


def is_empty(value: str | None) -> bool:
    """True for a value that records nothing: blank, or one of the usual ways of writing a gap (EMPTY)."""
    return value is None or value.strip().upper() in EMPTY


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def find_metadata(raw: Path) -> Path:
    """The one metadata.csv in the unpacked download."""
    if not raw.is_dir():
        raise FileNotFoundError(f"{raw.as_posix()} not found: unpack the PAD-UFES-20 download there (metadata.csv and the photos)")
    found = sorted(p for p in raw.rglob("*") if p.is_file() and p.name.lower() == "metadata.csv")
    if len(found) != 1:
        raise ValueError(f"{raw.as_posix()} must hold exactly one metadata.csv; found {len(found)}"
                         + (f": {', '.join(p.relative_to(raw).as_posix() for p in found)}" if found else ""))
    return found[0]


def read_metadata(path: Path) -> tuple[list[str], list[dict]]:
    """The columns and rows of metadata.csv, read as text, exactly as written (a byte-order mark is ignored)."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    missing = [c for c in REQUIRED if c not in columns]
    if missing:
        raise ValueError(f"{path.as_posix()} has no column {', '.join(missing)}; its columns are: {', '.join(columns)}")
    return columns, rows


def column_level(rows: list[dict], column: str) -> str:
    """Where a column's value is fixed: 'patient' (the same for all photos of a patient), 'lesion' (the same for all photos of a
    lesion, not of a patient), or 'photo' (it differs between photos of one lesion). Gaps count as values."""
    by_patient, by_lesion = defaultdict(set), defaultdict(set)
    for row in rows:
        by_patient[row["patient_id"]].add(row[column])
        by_lesion[row["lesion_id"]].add(row[column])
    if all(len(v) == 1 for v in by_patient.values()):
        return "patient"
    if all(len(v) == 1 for v in by_lesion.values()):
        return "lesion"
    return "photo"


def summarize_column(rows: list[dict], column: str) -> dict:
    values = [row[column] for row in rows]
    filled = [v for v in values if not is_empty(v)]
    counts = Counter(values)
    numbers = [_number(v) for v in filled]
    if filled and None not in numbers and len(counts) > LISTED_VALUES:
        described = (f"numbers: lowest {min(numbers):g}, median {statistics.median(numbers):g}, highest {max(numbers):g}; gaps: "
                     + ("; ".join(f"{v!r}={n}" for v, n in sorted(Counter(v for v in values if is_empty(v)).items())) or "none"))
    elif len(counts) <= LISTED_VALUES:
        described = "; ".join(f"{v!r}={n}" for v, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    else:
        described = f"{len(counts)} different values (not listed)"
    return {"column": column, "filled": len(filled), "empty": len(values) - len(filled), "distinct": len(counts),
            "level": column_level(rows, column), "values": described}


def inventory(data: Path = DATA) -> dict:
    """Reads the unpacked download and writes build/raw_images.csv, build/columns.csv, build/downloads.csv and the inventory section
    of the report (see the module's description). Returns the main counts."""
    data = Path(data)
    raw = data / RAW
    metadata = find_metadata(raw)
    columns, rows = read_metadata(metadata)
    if not rows:
        raise ValueError(f"{metadata.as_posix()} has no rows")

    files = defaultdict(list)  # file name -> its paths in raw/
    for path in raw.rglob("*"):
        if path.is_file() and path.suffix.lower() in MEDIA_TYPES:
            files[path.name].append(path)
    if not files:
        raise ValueError(f"no photo (JPEG or PNG) found in {raw.as_posix()}: unpack the photos there too")
    named = Counter(row["img_id"] for row in rows)
    repeated_rows = sorted(name for name, n in named.items() if n > 1)
    missing = sorted(name for name in named if name not in files)
    unnamed = sorted(name for name in files if name not in named)
    twice_on_disk = sorted(name for name, paths in files.items() if len(paths) > 1)

    images, unreadable = [], []
    for name in sorted(n for n in named if n in files):
        path = sorted(files[name])[0]
        entry = {"img_id": name, "file": path.relative_to(raw).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        try:
            with Image.open(path) as picture:
                entry |= {"format": picture.format, "width": picture.width, "height": picture.height}
        except OSError:
            unreadable.append(name)
            entry |= {"format": "", "width": "", "height": ""}
        images.append(entry)
    same_content = [sorted(e["img_id"] for e in group) for group in
                    _groups(images, key=lambda e: e["sha256"]) if len(group) > 1]

    downloads = []
    if (data / DOWNLOADS).is_dir():
        for path in sorted(p for p in (data / DOWNLOADS).rglob("*") if p.is_file()):
            downloads.append({"file": path.relative_to(data / DOWNLOADS).as_posix(), "bytes": path.stat().st_size,
                              "sha256": _sha256(path)})

    summaries = [summarize_column(rows, c) for c in columns]
    clinical = [c for c in columns if c not in NOT_CLINICAL]
    lesions = defaultdict(lambda: {"diagnoses": set(), "patients": set()})
    for row in rows:
        lesions[row["lesion_id"]]["diagnoses"].add(row["diagnostic"])
        lesions[row["lesion_id"]]["patients"].add(row["patient_id"])
    two_diagnoses = sorted(k for k, v in lesions.items() if len(v["diagnoses"]) > 1)
    two_patients = sorted(k for k, v in lesions.items() if len(v["patients"]) > 1)

    _write_csv(data / "build" / "raw_images.csv", IMAGE_COLUMNS, images)
    _write_csv(data / "build" / "columns.csv", COLUMN_COLUMNS, summaries)
    _write_csv(data / "build" / "downloads.csv", DOWNLOAD_COLUMNS, downloads)
    lines = _report(metadata.relative_to(data).as_posix(), _sha256(metadata), columns, rows, clinical, images, downloads,
                    {"repeated_rows": repeated_rows, "missing": missing, "unnamed": unnamed, "twice_on_disk": twice_on_disk,
                     "unreadable": unreadable, "same_content": same_content, "two_diagnoses": two_diagnoses,
                     "two_patients": two_patients})
    write_section(data / REPORT, REPORT_HEAD, STEPS, "inventory", lines)
    return {"rows": len(rows), "photos": len(images), "lesions": len({r["lesion_id"] for r in rows}),
            "patients": len({r["patient_id"] for r in rows}), "classes": Counter(r["diagnostic"] for r in rows),
            "missing": len(missing), "unnamed": len(unnamed), "columns": len(columns),
            "written": [(data / "build" / f).as_posix() for f in ("raw_images.csv", "columns.csv", "downloads.csv")]
            + [(data / REPORT).as_posix()]}


def _groups(items, key):
    grouped = defaultdict(list)
    for item in items:
        grouped[key(item)].append(item)
    return grouped.values()


def _report(metadata: str, metadata_sha: str, columns: list[str], rows: list[dict], clinical: list[str], images: list[dict],
            downloads: list[dict], problems: dict) -> list[str]:
    patients, lesions = {r["patient_id"] for r in rows}, {r["lesion_id"] for r in rows}
    biopsy_values = sorted({r["biopsed"] for r in rows})
    lines = ["# Inventory: what the download holds", "",
             "Written by `python -m src.build.clinic_patients.prepare inventory`, which only reads the download. Nothing here is decided yet.", "",
             f"- `{metadata}`: {len(rows)} rows, {len(columns)} columns (SHA-256 `{metadata_sha[:12]}`)",
             f"- Photos named in the table and found: {len(images)}; lesions: {len(lesions)}; patients: {len(patients)}",
             f"- Downloaded files fingerprinted (`build/downloads.csv`): {len(downloads) or 'none (downloads/ is empty or absent)'}",
             "", "## Classes", "",
             "The `diagnostic` column, with the `biopsed` column as written (it is never shown to a model: it points to the label).", "",
             "| Class | Photos | Lesions | Patients | " + " | ".join(f"`biopsed` = {v!r}" for v in biopsy_values) + " |",
             "|---|---|---|---|" + "---|" * len(biopsy_values)]
    for label in sorted({r["diagnostic"] for r in rows}):
        mine = [r for r in rows if r["diagnostic"] == label]
        biopsy = Counter(r["biopsed"] for r in mine)
        lines.append(f"| {label} | {len(mine)} | {len({r['lesion_id'] for r in mine})} | {len({r['patient_id'] for r in mine})} | "
                     + " | ".join(str(biopsy[v]) for v in biopsy_values) + " |")
    per_lesion = Counter(Counter(r["lesion_id"] for r in rows).values())
    per_patient = Counter(len(v) for v in _lesions_by_patient(rows).values())
    lines += ["", "## How photos, lesions and patients relate", "",
              "- Photos per lesion: " + ", ".join(f"{k}: {v} lesions" for k, v in sorted(per_lesion.items())),
              "- Lesions per patient: " + ", ".join(f"{k}: {v} patients" for k, v in sorted(per_patient.items())), "",
              "## Clinical fields filled, by class", "",
              f"The {len(clinical)} columns other than {', '.join(f'`{c}`' for c in REQUIRED)}. A field counts as filled unless it is "
              "blank or written as a gap (" + ", ".join(sorted(f"`{v}`" for v in EMPTY if v)) + ", in any case). If the classes "
              "differ much here, the gaps alone could point to the class.", "",
              "| Class | Photos | Mean fields filled | Fewest | Most |", "|---|---|---|---|---|"]
    for label in sorted({r["diagnostic"] for r in rows}):
        counts = [sum(not is_empty(r[c]) for c in clinical) for r in rows if r["diagnostic"] == label]
        lines.append(f"| {label} | {len(counts)} | {statistics.mean(counts):.1f} | {min(counts)} | {max(counts)} |")
    lines += ["", "Each column's values and level (patient, lesion or photo) are in `build/columns.csv`.", "", "## Checks", ""]
    checks = [("Rows naming the same photo twice", problems["repeated_rows"]),
              ("Photos named in the table but not found in raw/", problems["missing"]),
              ("Photo files in raw/ not named in the table", problems["unnamed"]),
              ("File names found in more than one folder of raw/ (the first path is used)", problems["twice_on_disk"]),
              ("Photos that cannot be opened", problems["unreadable"]),
              ("Lesions with more than one diagnosis", problems["two_diagnoses"]),
              ("Lesions under more than one patient", problems["two_patients"])]
    for title, items in checks:
        lines.append(f"- {title}: {len(items)}" + (f" ({_names(items)})" if items else ""))
    groups = problems["same_content"]
    lines.append(f"- Groups of photos with identical content: {len(groups)}"
                 + (" (" + "; ".join(" = ".join(f"`{n}`" for n in g) for g in groups[:EXAMPLES])
                    + (f"; and {len(groups) - EXAMPLES} more" if len(groups) > EXAMPLES else "") + ")" if groups else ""))
    formats = Counter(e["format"] for e in images if e["format"])
    widths = [e["width"] for e in images if e["width"] != ""]
    heights = [e["height"] for e in images if e["height"] != ""]
    if widths:
        lines += ["", "## Photos", "",
                  "- Formats: " + ", ".join(f"{k} {v}" for k, v in sorted(formats.items())),
                  f"- Width: {min(widths)} to {max(widths)} pixels (median {statistics.median(widths):g}); height: {min(heights)} to "
                  f"{max(heights)} (median {statistics.median(heights):g})"]
    return lines + [""]


def _lesions_by_patient(rows: list[dict]) -> dict:
    out = defaultdict(set)
    for row in rows:
        out[row["patient_id"]].add(row["lesion_id"])
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.build.clinic_patients.prepare", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("inventory", help="report what the PAD-UFES-20 download in data/clinic_patients/raw holds (nothing is changed)")
    return parser


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    try:
        done = inventory()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Rows: {done['rows']}; photos found: {done['photos']}; lesions: {done['lesions']}; patients: {done['patients']}; "
          f"columns: {done['columns']}")
    print("Classes: " + ", ".join(f"{k} {v}" for k, v in sorted(done["classes"].items())))
    print(f"Named but not found: {done['missing']}; found but not named: {done['unnamed']}")
    print("Written: " + ", ".join(done["written"]) + ". Commit them all.")


if __name__ == "__main__":
    main(sys.argv[1:])
