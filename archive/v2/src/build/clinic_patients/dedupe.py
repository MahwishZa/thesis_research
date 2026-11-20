"""Step ``dedupe`` of the clinic patients cohort: identical and near-identical photos found by a picture fingerprint, and a fixed rule
for which of them stay. It changes no photo and no original file.

Why: a picture that occurs twice may be a copy (the model then answers it twice), may carry two different labels (it cannot be a
question with one right answer), and, if its copies fall into different splits, leaks the test question into training or retrieval.

The fingerprint is a 256-bit *difference hash* of each photo: the photo in grey, shrunk to 17 x 16 pixels, and one bit per pair of
neighbouring pixels (is the left one brighter?). Copies and re-saved copies of one picture differ in a few bits; two different photos
differ in about half of the 256. The distance of two photos is the number of bits in which they differ.

The rule (fixed before the numbers were seen; the thresholds are constants of this module, written in the report):
- Photos at most ``DUP_BITS`` apart (identical files are 0 bits apart) are **one picture**. Photos linked through such pairs are one cluster.
- A cluster whose photos all have the same diagnosis keeps **one** photo, the one with the smallest ``img_id``; the others are dropped
  (``identical_copy`` if all have the same SHA-256, else ``near_copy``).
- A cluster whose photos have different diagnoses is **dropped completely** (``conflicting_labels``): one picture has no single
  right answer.
- Photos of **different patients** at most ``LINK_BITS`` apart are not dropped, but their patients are kept together in one split
  (step ``split``), so that a copy cannot leak across splits.

Writes build/photo_hashes.csv, build/duplicates.csv (every cluster of two or more photos with its decision), build/near_pairs.csv
(every pair at most ``LINK_BITS`` apart) and the dedupe section of dataset_report.md.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ...common import write_section
from .table import (BUILD, DATA, EXAMPLES, RAW, REPORT, REPORT_HEAD, STEPS, components, names, read_csv, write_csv)

HASH_SIDE = 16  # the hash has HASH_SIDE x HASH_SIDE = 256 bits
DUP_BITS = 8  # at most this many bits apart: the same picture
LINK_BITS = 16  # at most this many bits apart, under different patients: their patients stay in one split
CHUNK = 128  # photos compared with all others at a time
HASH_COLUMNS = ["img_id", "sha256", "dhash"]
CLUSTER_COLUMNS = ["cluster", "img_id", "patient_id", "lesion_id", "label", "decision", "reason", "bits_from_kept"]
PAIR_COLUMNS = ["img_a", "img_b", "bits", "same_patient", "same_lesion", "label_a", "label_b", "same_cluster"]
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
BINS = ((0, 0), (1, DUP_BITS), (DUP_BITS + 1, LINK_BITS), (LINK_BITS + 1, 32), (33, 64), (65, 256))  # nearest-neighbour distances reported


def dhash(path: Path) -> np.ndarray:
    """The 256-bit difference hash of a photo, as 32 bytes."""
    with Image.open(path) as picture:
        small = picture.convert("L").resize((HASH_SIDE + 1, HASH_SIDE), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    return np.packbits((pixels[:, 1:] > pixels[:, :-1]).flatten())


def distances(hashes: np.ndarray, first: int, last: int, others: np.ndarray | None = None) -> np.ndarray:
    """The distances in bits of photos ``first`` to ``last - 1`` to all photos of ``others`` (default: ``hashes`` itself): an array of
    shape (last - first, number of photos of ``others``)."""
    others = hashes if others is None else others
    return POPCOUNT[hashes[first:last, None, :] ^ others[None, :, :]].sum(axis=2, dtype=np.int32)


def dedupe(data: Path = DATA) -> dict:
    """Writes the tables and the report (see the module's description); returns the counts."""
    data = Path(data)
    cleaned = read_csv(data / BUILD / "cleaned.csv", "run the clean step first")
    images = {r["img_id"]: r for r in read_csv(data / BUILD / "raw_images.csv", "run the inventory step first")}
    rows = {r["img_id"]: r for r in cleaned}
    ids = sorted(rows)
    missing = [i for i in ids if i not in images]
    if missing:
        raise ValueError(f"{len(missing)} photos of the cleaned table are not in build/raw_images.csv (run inventory again): "
                         + names(missing))
    unreadable = [i for i in ids if images[i]["format"] == ""]
    readable = [i for i in ids if images[i]["format"] != ""]
    hashes = np.stack([dhash(data / RAW / images[i]["file"]) for i in readable]) if readable else np.zeros((0, 32), dtype=np.uint8)
    write_csv(data / BUILD / "photo_hashes.csv", HASH_COLUMNS,
              [{"img_id": i, "sha256": images[i]["sha256"], "dhash": h.tobytes().hex()} for i, h in zip(readable, hashes)])

    pairs, nearest = [], []  # (a, b, bits) with a < b, at most LINK_BITS apart; the distance of each photo to its nearest other photo
    for first in range(0, len(readable), CHUNK):
        block = distances(hashes, first, min(first + CHUNK, len(readable)))
        for r in range(block.shape[0]):
            i = first + r
            block[r, i] = 255  # a photo is not its own neighbour
            nearest.append(int(block[r].min()) if len(readable) > 1 else 255)
            for j in np.nonzero(block[r, i + 1:] <= LINK_BITS)[0]:
                pairs.append((readable[i], readable[i + 1 + int(j)], int(block[r, i + 1 + int(j)])))
    bits = {(a, b): d for a, b, d in pairs}

    same_content = Counter(images[i]["sha256"] for i in readable)
    links = [(a, b) for (a, b), d in bits.items() if d <= DUP_BITS]  # a file identical to another is 0 bits from it
    clusters = [c for c in components(readable, links) if len(c) > 1]
    decisions, cluster_rows = {}, []
    for number, members in enumerate(clusters, start=1):
        labels = {rows[i]["diagnostic"] for i in members}
        kept = members[0]  # the smallest img_id
        reason = "identical_copy" if len({images[i]["sha256"] for i in members}) == 1 else "near_copy"
        for i in members:
            if len(labels) > 1:
                decision, why = "drop", "conflicting_labels"
            elif i == kept:
                decision, why = "keep", "kept_one_of_cluster"
            else:
                decision, why = "drop", reason
            decisions[i] = (decision, why)
            cluster_rows.append({"cluster": f"C{number:03d}", "img_id": i, "patient_id": rows[i]["patient_id"],
                                 "lesion_id": rows[i]["lesion_id"], "label": rows[i]["diagnostic"], "decision": decision,
                                 "reason": why, "bits_from_kept": "" if i == kept else bits.get((kept, i), "")})
    for i in unreadable:
        decisions[i] = ("drop", "unreadable")
        cluster_rows.append({"cluster": "", "img_id": i, "patient_id": rows[i]["patient_id"], "lesion_id": rows[i]["lesion_id"],
                             "label": rows[i]["diagnostic"], "decision": "drop", "reason": "unreadable", "bits_from_kept": ""})
    in_cluster = {i: n for n, members in enumerate(clusters, start=1) for i in members}
    pair_rows = [{"img_a": a, "img_b": b, "bits": d, "same_patient": "yes" if rows[a]["patient_id"] == rows[b]["patient_id"] else "no",
                  "same_lesion": "yes" if (rows[a]["patient_id"], rows[a]["lesion_id"]) == (rows[b]["patient_id"], rows[b]["lesion_id"])
                  else "no", "label_a": rows[a]["diagnostic"], "label_b": rows[b]["diagnostic"],
                  "same_cluster": "yes" if a in in_cluster and in_cluster.get(a) == in_cluster.get(b) else "no"}
                 for a, b, d in sorted(pairs)]
    write_csv(data / BUILD / "duplicates.csv", CLUSTER_COLUMNS, cluster_rows)
    write_csv(data / BUILD / "near_pairs.csv", PAIR_COLUMNS, pair_rows)
    dropped = [i for i, (decision, _) in decisions.items() if decision == "drop"]
    write_section(data / REPORT, REPORT_HEAD, STEPS, "dedupe",
                  _report(rows, readable, nearest, clusters, decisions, pair_rows, unreadable, same_content))
    return {"summary": f"{len(readable)} photos hashed; {len(clusters)} clusters of the same picture; {len(dropped)} photos dropped, "
                       f"{len(ids) - len(dropped)} stay; {sum(r['same_patient'] == 'no' for r in pair_rows)} pairs of different "
                       f"patients at most {LINK_BITS} bits apart", "clusters": len(clusters), "dropped": sorted(dropped),
            "kept": len(ids) - len(dropped), "pairs": len(pair_rows)}


def _report(rows, readable, nearest, clusters, decisions, pair_rows, unreadable, same_content) -> list[str]:
    classes = sorted({r["diagnostic"] for r in rows.values()})
    lines = ["# Dedupe: identical and near-identical photos", "",
             "Written by `python -m src.build.clinic_patients.prepare dedupe`. Every photo has a 256-bit difference hash "
             "(`build/photo_hashes.csv`); the distance of two photos is the number of bits in which their hashes differ.", "",
             "## The rule (fixed before the numbers were seen)", "",
             f"- At most **{DUP_BITS}** bits apart (identical files are 0 bits apart): one picture. Pictures linked through such pairs form a cluster.",
             "- All photos of a cluster have the same diagnosis: the one with the smallest `img_id` stays, the others are dropped.",
             "- The photos of a cluster have different diagnoses: all are dropped (one picture has no single right answer).",
             f"- Photos of different patients at most **{LINK_BITS}** bits apart stay, but their patients go to one split together.", "",
             "## How far each photo is from its nearest other photo", "", "| Bits | Photos |", "|---|---|"]
    for low, high in BINS:
        label = f"{low}" if low == high else f"{low} to {high}"
        lines.append(f"| {label} | {sum(low <= d <= high for d in nearest)} |")
    lines += ["", "Two different photos are about 128 bits apart. A gap between the first rows and the last ones shows that the "
              "threshold separates copies from other photos.", "", "## Clusters and decisions", ""]
    lines.append(f"- Photos hashed: {len(readable)}; unreadable (dropped): {len(unreadable)}")
    lines.append(f"- Clusters of the same picture: {len(clusters)} ({sum(len(c) for c in clusters)} photos); identical SHA-256 groups: "
                 f"{sum(n > 1 for n in same_content.values())}")
    conflict = [c for c in clusters if len({rows[i]["diagnostic"] for i in c}) > 1]
    lines.append(f"- Clusters with conflicting diagnoses (all photos dropped): {len(conflict)}")
    lines.append(f"- Clusters with one diagnosis (one photo stays): {len(clusters) - len(conflict)}")
    dropped = [i for i, (decision, _) in decisions.items() if decision == "drop"]
    lines += ["", "| Class | Photos | Dropped | of which conflicting | of which copies |", "|---|---|---|---|---|"]
    for label in classes:
        mine = [i for i in rows if rows[i]["diagnostic"] == label]
        gone = [i for i in mine if i in dropped]
        lines.append(f"| {label} | {len(mine)} | {len(gone)} | {sum(decisions[i][1] == 'conflicting_labels' for i in gone)} | "
                     f"{sum(decisions[i][1] in ('identical_copy', 'near_copy') for i in gone)} |")
    if clusters:
        lines += ["", "Clusters (the first " + str(EXAMPLES) + "; all are in `build/duplicates.csv`):", ""]
        for number, members in enumerate(clusters[:EXAMPLES], start=1):
            lines.append(f"- C{number:03d}: " + ", ".join(f"`{i}` ({rows[i]['diagnostic']})" for i in members) + " "
                         + ("(conflicting: all dropped)" if len({rows[i]['diagnostic'] for i in members}) > 1 else f"(`{members[0]}` stays)"))
    cross = [r for r in pair_rows if r["same_patient"] == "no"]
    same = [r for r in pair_rows if r["same_patient"] == "yes" and r["same_cluster"] == "no"]
    lines += ["", "## Near pairs", "",
              f"- Pairs of photos of **different patients** at most {LINK_BITS} bits apart: {len(cross)} (their patients are kept in one "
              "split)" + (": " + ", ".join(f"`{r['img_a']}` and `{r['img_b']}` ({r['bits']} bits)" for r in cross[:EXAMPLES]) if cross else ""),
              f"- Pairs of photos of one patient between {DUP_BITS + 1} and {LINK_BITS} bits apart (kept; one patient is always in one "
              f"split): {len(same)}", "- All pairs at most " + str(LINK_BITS) + " bits apart are in `build/near_pairs.csv`.", ""]
    return lines
