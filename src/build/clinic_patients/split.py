"""Step ``split`` of the clinic patients cohort: train, validation and test splits **by patient**. Photos dropped by step ``dedupe`` are
left out; nothing else is.

Rules (fixed before the numbers were seen):
- **A patient is never in two splits.** A lesion number alone does not identify a lesion (the same one is used for different patients),
  so lesions are not used to group anything.
- **Patients stay together when a picture does**: two patients with photos at most ``LINK_BITS`` apart (step ``dedupe``,
  build/near_pairs.csv) are one *split group*, and a split group is never divided. A copy of a photo cannot then lead from the test
  split into training or retrieval.
- **About 60% train, 10% validation, 30% test, class by class** (photos of each diagnosis). The groups are placed one by one, the
  ones holding the rarest classes first, each into the split that brings its class shares closest to the targets; ties are broken by
  a fixed hash of the seed and the group, so the result depends on nothing else.
- Every class must occur in every split, or the step stops.

The split is drawn once; after the cohort is frozen it is never redrawn. ``SEED`` was chosen before the first draw.

Writes build/split.csv (one row per photo that stays, with its split and its clinical record; the photos themselves are made by step
``images``, which writes dataset.csv), build/excluded.csv (every photo left out and why) and the split section of dataset_report.md.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from ...common import write_section
from .table import (BUILD, DATA, REPORT, REPORT_HEAD, SPLIT, STEPS, clinical_columns, components, names, read_csv, write_csv)

SEED = 20261119  # chosen before the first split was drawn; never changed
FRACTIONS = (("train", 0.6), ("val", 0.1), ("test", 0.3))
SPLITS = tuple(name for name, _ in FRACTIONS)
FIRST_COLUMNS = ["split", "label", "img_id", "patient_id", "lesion_id", "split_group", "biopsed"]
EXCLUDED_COLUMNS = ["img_id", "patient_id", "lesion_id", "label", "reason"]


def _tie(group: str) -> str:
    return hashlib.sha256(f"{SEED}:{group}".encode()).hexdigest()


def assign(groups: dict[str, Counter], fractions=FRACTIONS) -> dict[str, str]:
    """The split of each group (group name -> its photos per class): rarest classes first, each group into the split that brings the
    class shares nearest to the targets."""
    totals = Counter()
    for counts in groups.values():
        totals.update(counts)
    placed = {name: Counter() for name, _ in fractions}
    order = sorted(groups, key=lambda g: (-max(n / totals[c] for c, n in groups[g].items()), _tie(g)))
    result = {}
    for g in order:
        def cost(item):
            name, share = item
            increase = sum(((placed[name][c] + n) / totals[c] - share) ** 2 - (placed[name][c] / totals[c] - share) ** 2
                           for c, n in groups[g].items())
            return round(increase, 12), [n for n, _ in fractions].index(name)  # rounded: a tie is a tie on every computer
        best = min(fractions, key=cost)[0]
        result[g] = best
        placed[best].update(groups[g])
    return result


def patients_in_two_splits(rows: list[dict]) -> list[str]:
    """The patients whose photos are in more than one split (there must be none)."""
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        seen[row["patient_id"]].add(row["split"])
    return sorted(patient for patient, splits in seen.items() if len(splits) > 1)


def split(data: Path = DATA) -> dict:
    """Writes build/split.csv, build/excluded.csv and the split section of the report (see the module's description)."""
    data = Path(data)
    cleaned = read_csv(data / BUILD / "cleaned.csv", "run the clean step first")
    decisions = {r["img_id"]: r for r in read_csv(data / BUILD / "duplicates.csv", "run the dedupe step first")}
    pairs = read_csv(data / BUILD / "near_pairs.csv", "run the dedupe step first")
    columns = list(cleaned[0])
    by_id = {r["img_id"]: r for r in cleaned}

    kept = [r for r in cleaned if decisions.get(r["img_id"], {}).get("decision") != "drop"]
    excluded = [{"img_id": i, "patient_id": by_id[i]["patient_id"], "lesion_id": by_id[i]["lesion_id"], "label": by_id[i]["diagnostic"],
                 "reason": d["reason"]} for i, d in sorted(decisions.items()) if d["decision"] == "drop"]
    patients = sorted({r["patient_id"] for r in kept})
    present = set(patients)
    links = [(by_id[p["img_a"]]["patient_id"], by_id[p["img_b"]]["patient_id"]) for p in pairs if p["same_patient"] == "no"]
    links = [(a, b) for a, b in links if a in present and b in present]
    group_of = {}
    for members in components(patients, links):
        for patient in members:
            group_of[patient] = f"G_{members[0]}"
    counts = defaultdict(Counter)
    for r in kept:
        counts[group_of[r["patient_id"]]][r["diagnostic"]] += 1
    where = assign(dict(counts))
    classes = sorted({r["diagnostic"] for r in kept})
    empty = [(label, name) for label in classes for name in SPLITS if not any(
        r["diagnostic"] == label and where[group_of[r["patient_id"]]] == name for r in kept)]
    if empty:
        raise ValueError("a class has no photo in a split: " + ", ".join(f"{label} in {name}" for label, name in empty))

    clinical = [c for c in clinical_columns(columns)]
    rows = []
    for r in sorted(kept, key=lambda r: (SPLITS.index(where[group_of[r["patient_id"]]]), r["diagnostic"], r["img_id"])):
        rows.append({"split": where[group_of[r["patient_id"]]], "label": r["diagnostic"], "img_id": r["img_id"],
                     "patient_id": r["patient_id"], "lesion_id": r["lesion_id"], "split_group": group_of[r["patient_id"]],
                     "biopsed": r["biopsed"], **{c: r[c] for c in clinical}})
    wrong = patients_in_two_splits(rows)
    if wrong:  # cannot happen by construction; a guard against a later change
        raise ValueError("patients in more than one split: " + names(wrong))
    write_csv(data / BUILD / SPLIT, FIRST_COLUMNS + clinical, rows)
    write_csv(data / BUILD / "excluded.csv", EXCLUDED_COLUMNS, excluded)
    write_section(data / REPORT, REPORT_HEAD, STEPS, "split", _report(rows, excluded, group_of, counts, classes, len(links)))
    sizes = Counter(row["split"] for row in rows)
    return {"summary": f"{len(rows)} photos in {len(set(group_of.values()))} split groups: "
                       + ", ".join(f"{s} {sizes[s]}" for s in SPLITS) + f"; {len(excluded)} photos left out",
            "photos": len(rows), "splits": dict(sizes), "excluded": len(excluded)}


def _report(rows, excluded, group_of, counts, classes, link_count) -> list[str]:
    total = Counter(r["label"] for r in rows)
    lines = ["# Split: train, validation and test, by patient", "",
             "Written by `python -m src.build.clinic_patients.prepare split`. The questions with their splits are `build/split.csv` "
             "(step `images` adds the photos and writes `dataset.csv`); the photos left out, with the reason, are in `build/excluded.csv`.", "",
             f"- Seed: {SEED} (chosen before the first draw). Targets: " + ", ".join(f"{n} {s:.0%}" for n, s in FRACTIONS) + ", class by class.",
             f"- Photos that stay: {len(rows)}; left out: {len(excluded)} ("
             + (", ".join(f"{k} {v}" for k, v in sorted(Counter(e['reason'] for e in excluded).items())) or "none") + ")",
             f"- Split groups (patients that must stay together): {len(set(group_of.values()))} for {len(group_of)} patients; patients "
             f"joined because of a near-identical photo of another patient: {link_count} links",
             "- No patient is in two splits (checked).", "", "## Photos by class and split", "",
             "| Class | " + " | ".join(SPLITS) + " | Total | " + " | ".join(f"{s} share" for s in SPLITS) + " |",
             "|---|" + "---|" * (2 * len(SPLITS) + 1)]
    for label in classes:
        n = {s: sum(1 for r in rows if r["label"] == label and r["split"] == s) for s in SPLITS}
        lines.append(f"| {label} | " + " | ".join(str(n[s]) for s in SPLITS) + f" | {total[label]} | "
                     + " | ".join(f"{n[s] / total[label]:.1%}" for s in SPLITS) + " |")
    lines += ["", "## Patients and lesions by split", "", "| Split | Photos | Patients | Lesions |", "|---|---|---|---|"]
    for s in SPLITS:
        mine = [r for r in rows if r["split"] == s]
        lines.append(f"| {s} | {len(mine)} | {len({r['patient_id'] for r in mine})} | "
                     f"{len({(r['patient_id'], r['lesion_id']) for r in mine})} |")
    big = sorted(((sum(c.values()), g) for g, c in counts.items()), reverse=True)[:3]
    lines += ["", "Largest split groups (photos): " + ", ".join(f"`{g}` {n}" for n, g in big), ""]
    return lines
