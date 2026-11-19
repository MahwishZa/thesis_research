"""What every step of the clinic patients cohort (PAD-UFES-20) shares: the folders and file names, the columns the steps depend on, how a
gap is recognised, and reading and writing the tables. Nothing here reads a photo or decides anything about the data."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

DATA = Path("data/clinic_patients")  # this cohort's folder: committed tables and report; downloads/, raw/ and files/ stay on this computer
RAW, DOWNLOADS, BUILD = "raw", "downloads", "build"  # inside DATA: the unpacked download, the files as downloaded, the records of the steps
REPORT, DATASET = "dataset_report.md", "dataset.csv"
STEPS = ("inventory", "clean", "dedupe", "split", "shortcuts")  # the report's sections, in the order the steps run
REQUIRED = ("img_id", "patient_id", "lesion_id", "diagnostic", "biopsed")  # the columns every step depends on
NOT_CLINICAL = set(REQUIRED)  # identifiers, the label and the biopsy column: not part of the clinical record
EMPTY = {"", "NAN", "NA", "N/A", "NONE", "NULL", "UNK", "UNKNOWN"}  # how a gap may be written (compared in capitals); reported as found
NOT_RECORDED = "not recorded"  # how a gap is written in the cleaned table
EXAMPLES = 10  # how many names a report line lists before it says "and N more"
REPORT_HEAD = ["# Preparation of the PAD-UFES-20 cohort", "",
               "Made by `python -m src.build.clinic_patients.prepare <step>`, one section per step, in the order the steps run. Each step rewrites "
               "only its own section.", ""]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path, missing: str) -> list[dict]:
    """The rows of a table the earlier steps wrote; ``missing`` says which step writes it."""
    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} not found: {missing}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def names(items, n: int = EXAMPLES) -> str:
    items = sorted(items)
    shown = ", ".join(f"`{x}`" for x in items[:n])
    return shown + (f" and {len(items) - n} more" if len(items) > n else "")


def is_empty(value: str | None) -> bool:
    """True for a value that records nothing: blank, or one of the usual ways of writing a gap (EMPTY)."""
    return value is None or value.strip().upper() in EMPTY


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


def lesion_key(row: dict) -> tuple[str, str]:
    """What identifies a lesion: the patient and the lesion_id together. The same lesion_id is used for lesions of different patients."""
    return row["patient_id"], row["lesion_id"]


def clinical_columns(columns: list[str]) -> list[str]:
    """The columns that hold the clinical record: all but the identifiers, the label and the biopsy column."""
    return [c for c in columns if c not in NOT_CLINICAL]


def components(items: list[str], links: list[tuple[str, str]]) -> list[list[str]]:
    """The groups of ``items`` joined by ``links`` (a link joins its two items, and so on through chains). Each group is sorted, and the
    groups are in the order of their first item, so the result does not depend on the order of the links."""
    parent = {item: item for item in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for item in items:
        groups.setdefault(find(item), []).append(item)
    return [sorted(members) for _, members in sorted(groups.items())]
