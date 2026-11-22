"""
The balanced test set (decided 2026-11-22): the test split holds exactly the same number of images of every class (30 by
default), so every class weighs the same in the test accuracy. Used by src/preprocess.py (step 8) after the lesion-grouped split.

The fixed rule, for each class on its own:
  1. Whole lesion groups only: a group (the photos of one lesion) is moved as a whole, so no lesion is ever in two splits.
  2. Order: the groups are taken in a fixed order, the SHA-256 of "<seed>:<group_id>" (no hand choice, the same on every system).
  3. Keep: the class's test groups, in that order, as long as they fit within the target (a group that would go over it is
     skipped); the test groups not kept move to train.
  4. Fill: if the class still has fewer than the target, train groups of the class are added in the same order, again skipping a
     group that would go over it.
  5. The validation split is never changed (its questions, and any run already made on them, stay valid).
The rule stops with an error when a class cannot reach the target exactly. Only the Python standard library is used, so the
rule is tested on the committed table without the image packages.
"""
import hashlib
from collections import defaultdict

TEST_PER_CLASS = 30


def group_order(seed, group_id):
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def balanced_splits(cases, per_class=TEST_PER_CLASS, seed=42):
    """cases: [{"isic_id", "group_id", "label", "split"}, ...] after the lesion-grouped split.
    Returns {isic_id: new split} for every case (val unchanged; train and test changed only by whole groups)."""
    groups = defaultdict(list)
    for c in cases:
        groups[c["group_id"]].append(c)
    for g, members in groups.items():
        if len({m["label"] for m in members}) != 1 or len({m["split"] for m in members}) != 1:
            raise ValueError(f"lesion group {g} has more than one class or split")
    new = {c["isic_id"]: c["split"] for c in cases}
    for label in sorted({c["label"] for c in cases}):
        def ordered(split):
            gs = [g for g, m in groups.items() if m[0]["label"] == label and m[0]["split"] == split]
            return sorted(gs, key=lambda g: group_order(seed, g))
        n = 0
        for g in ordered("test"):
            if n + len(groups[g]) <= per_class:
                n += len(groups[g])
            else:
                for m in groups[g]:
                    new[m["isic_id"]] = "train"
        for g in ordered("train"):
            if n + len(groups[g]) <= per_class:
                n += len(groups[g])
                for m in groups[g]:
                    new[m["isic_id"]] = "test"
        if n != per_class:
            raise ValueError(f"{label}: only {n} test images can be reached by whole lesion groups, not {per_class}")
    return new
