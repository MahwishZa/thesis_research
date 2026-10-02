"""Build the as-of benchmark from the MedChange release.

Inputs are the authors' files (``Datasets/MedRevQA.csv``,
``Datasets/AllStudyGroups.csv``, ``Datasets/MedChangeQA.csv``), read from a
local clone - the release carries no licence statement, so the data is NOT
copied into this repository.

How MedChangeQA was built (``Code/experiments_results.ipynb``): reviews are
grouped by DOI; the member with the lowest MedRevQA row index is the newest
version (verified: true for 1,534 of 1,535 groups by publication date); the
"outdated" member is the first lower-ranked (older) member whose label differs.
``rebuild_medchangeqa`` reproduces that rule and ``verify_against_release``
checks it row by row, so every item carries BOTH versions' dates and PMIDs -
which the released CSV does not.

Verdict labels were produced by gpt-4o-mini from each abstract's conclusions
(``Code/generate_questions_labels.ipynb``): they are model labels, not human
ones. Pairs whose two conclusions are near-identical text cannot reflect a real
change in evidence and are flagged ``likely_label_noise``.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION")
_MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}
_AD = re.compile(
    r"dementia|alzheimer|cognitive impairment|cognitive decline|memantine|"
    r"donepezil|cholinesterase|amyloid", re.I)
#: Conclusions at least this similar (difflib ratio, first 2,000 chars) are
#: treated as the same conclusion: a label change between them is labelling
#: noise, not evidence change. 482 of 512 MedChangeQA pairs are < 0.6.
NOISE_SIMILARITY = 0.85


class BenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class Version:
    row: int
    pmid: Optional[str]
    cochrane_id: Optional[str]
    date: str               # ISO YYYY-MM-DD; month/day default to 01
    date_precision: str     # "day" | "month" | "year"
    label: str


@dataclass
class Item:
    item_id: str
    group_id: int
    kind: str               # "changed" | "unchanged"
    question: str
    newest: Version
    previous: Version       # changed: the outdated version; unchanged: the next-older version
    change_type: Optional[str]
    decisive_flip: bool
    conclusion_similarity: float
    likely_label_noise: bool
    ad_related: bool
    split: str = ""
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def parse_cochrane_date(citation: str) -> tuple[str, str]:
    """'Cochrane Database Syst Rev. 2024 Jan 18;1(1):...' -> ('2024-01-18', 'day').

    A missing month/day defaults to 01 and the precision says so: as-of
    cutoffs built from a year-only date are conservative (earlier than the
    true date), so they can drop evidence but never leak future evidence.
    """
    m = re.search(r"Rev\.?\s*(\d{4})(?:\s+([A-Z][a-z]{2}))?(?:\s+(\d{1,2}))?", citation)
    if not m:
        raise BenchmarkError(f"no date in citation: {citation[:80]!r}")
    year = int(m.group(1))
    month = _MONTHS.get(m.group(2) or "")
    day = int(m.group(3)) if (month and m.group(3)) else None
    precision = "day" if day else "month" if month else "year"
    return f"{year:04d}-{month or 1:02d}-{day or 1:02d}", precision


def _cochrane_id(citation: str) -> Optional[str]:
    m = re.search(r"(CD\d{6})", citation)
    return m.group(1) if m else None


def _pmid(link: str) -> Optional[str]:
    m = re.search(r"(\d+)", link or "")
    return m.group(1) if m else None


def read_csv(path: Path) -> list[dict]:
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def load_groups(rows: Iterable[dict]) -> dict[int, list[int]]:
    """AllStudyGroups.csv: a blank Group_ID continues the previous group."""
    groups: dict[int, list[int]] = {}
    current = None
    for r in rows:
        if r["Group_ID"].strip():
            current = int(r["Group_ID"])
            groups.setdefault(current, [])
        if current is None:
            raise BenchmarkError("AllStudyGroups.csv starts with a blank Group_ID")
        groups[current].append(int(r["Study_ID"]))
    return groups


def _version(medrev: dict[int, dict], row: int) -> Version:
    r = medrev[row]
    date, precision = parse_cochrane_date(r["DOI_Date"])
    return Version(row=row, pmid=_pmid(r.get("PMID", "")),
                   cochrane_id=_cochrane_id(r["DOI_Date"]),
                   date=date, date_precision=precision, label=r["Label"].strip())


def _similarity(a: str, b: str) -> float:
    return round(difflib.SequenceMatcher(None, a[:2000], b[:2000]).ratio(), 4)


def rebuild_medchangeqa(medrev: dict[int, dict],
                        groups: dict[int, list[int]]) -> list[tuple[int, int, int]]:
    """(group_id, newest_row, outdated_row) by the authors' rule, in their order."""
    out = []
    for gid in sorted(groups):
        keys = sorted(set(groups[gid]))
        if len(keys) < 2:
            continue
        newest = keys[0]
        for k in keys[1:]:
            if medrev[k]["Label"].strip() != medrev[newest]["Label"].strip():
                out.append((gid, newest, k))
                break
    return out


def verify_against_release(rebuilt: list[tuple[int, int, int]], medrev: dict[int, dict],
                           released: list[dict]) -> None:
    """Refuse to build on a reconstruction that differs from MedChangeQA.csv."""
    if len(rebuilt) != len(released):
        raise BenchmarkError(f"rebuilt {len(rebuilt)} items, release has {len(released)}")
    for i, ((_, n, o), r) in enumerate(zip(rebuilt, released)):
        if (medrev[n]["Label"].strip(), medrev[o]["Label"].strip()) != (
                r["Newest Label"].strip(), r["Outdated Label"].strip()):
            raise BenchmarkError(f"label mismatch at MedChangeQA row {i}")


def build_items(medrev: dict[int, dict], groups: dict[int, list[int]],
                rebuilt: list[tuple[int, int, int]], *, n_unchanged: int,
                seed: int) -> list[Item]:
    items: list[Item] = []
    changed_groups = set()
    for gid, n, o in rebuilt:
        changed_groups.add(gid)
        items.append(_make_item(medrev, gid, n, o, "changed"))
    # Unchanged controls: groups whose versions all share one label; compared
    # newest vs next-older version, so they too span a window of new evidence.
    pool = []
    for gid in sorted(groups):
        keys = sorted(set(groups[gid]))
        if gid in changed_groups or len(keys) < 2:
            continue
        if len({medrev[k]["Label"].strip() for k in keys}) == 1:
            pool.append((gid, keys[0], keys[1]))
    rng = random.Random(seed)
    for gid, n, p in sorted(rng.sample(pool, min(n_unchanged, len(pool)))):
        items.append(_make_item(medrev, gid, n, p, "unchanged"))
    return items


def _make_item(medrev, gid, n, o, kind) -> Item:
    newest, prev = _version(medrev, n), _version(medrev, o)
    if prev.date > newest.date:
        raise BenchmarkError(f"group {gid}: previous version dated after newest")
    sim = _similarity(medrev[n]["conclusions"], medrev[o]["conclusions"])
    changed = kind == "changed"
    text = medrev[n]["Question"] + " " + medrev[n].get("objectives", "")
    return Item(
        item_id=f"MC-{gid:05d}",
        group_id=gid, kind=kind, question=medrev[n]["Question"].strip(),
        newest=newest, previous=prev,
        change_type=f"{prev.label} -> {newest.label}" if changed else None,
        decisive_flip=changed and {prev.label, newest.label} == {"SUPPORTED", "REFUTED"},
        conclusion_similarity=sim,
        likely_label_noise=changed and sim >= NOISE_SIMILARITY,
        ad_related=bool(_AD.search(text)),
    )


def assign_splits(items: list[Item], *, dev_fraction: float, seed: int) -> None:
    """Seeded split, stratified by kind x change_type, fixed before any run."""
    if not 0 < dev_fraction < 1:
        raise BenchmarkError("dev_fraction must be in (0, 1)")
    strata: dict[tuple, list[Item]] = {}
    for it in items:
        strata.setdefault((it.kind, it.change_type), []).append(it)
    rng = random.Random(seed)
    for key in sorted(strata, key=str):
        members = sorted(strata[key], key=lambda x: x.item_id)
        rng.shuffle(members)
        n_dev = round(len(members) * dev_fraction)
        for i, it in enumerate(members):
            it.split = "dev" if i < n_dev else "confirm"


def file_sha256(path: Path) -> str:
    """SHA-256 of the file with CRLF normalised to LF, so a Windows checkout with
    git's autocrlf conversion and a Linux checkout of the same release hash alike."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
