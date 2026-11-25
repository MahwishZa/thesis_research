"""The balanced test and validation splits (used by data/preprocessing.py step 8 and checked by utils/status.py)."""
import hashlib
from collections import defaultdict

from src.utils.reproducibility import SEED

TEST_PER_CLASS = 30                   # images of each class in the balanced test split
VAL_PER_CLASS = 20                    # images of each class in the balanced validation split


# ---- the balanced test and validation splits (used by data/preprocessing.py step 8, checked by utils/status.py) -----------------------------
# The balanced splits (test decided 2026-11-22: 30 per class; validation decided 2026-11-22: 20 per class): the split holds exactly
# the same number of images of every class, so every class weighs the same in its accuracy. Used by src/preprocess.py (step 8)
# after the lesion-grouped split: first for "test", then for "val" (the test split is not touched by the second pass). Train keeps
# the natural class mix (the imbalance is handled when a model is trained: class weights, the oversampled and undersampled lists).
#
# The fixed rule, for each class on its own, for the split being balanced (call it S):
#   1. Whole lesion groups only: a group (the photos of one lesion) is moved as a whole, so no lesion is ever in two splits.
#   2. Order: the groups are taken in a fixed order, the SHA-256 of "<seed>:<group_id>" (no hand choice, the same on every system).
#   3. Keep: the class's S groups, in that order, as long as they fit within the target (a group that would go over it is
#      skipped); the S groups not kept move to train.
#   4. Fill: if the class still has fewer than the target, train groups of the class are added in the same order, again skipping a
#      group that would go over it.
#   5. Every split other than S and train is never changed.
# The rule stops with an error when a class cannot reach the target exactly. Only the Python standard library is used, so the
# rule is tested on the committed table without the image packages.
def group_order(seed, group_id):
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def balanced_splits(cases, per_class=TEST_PER_CLASS, seed=SEED, split="test"):
    """cases: [{"isic_id", "group_id", "label", "split"}, ...] after the lesion-grouped split.
    Returns {isic_id: new split} for every case (the other splits unchanged; train and `split` changed only by whole groups)."""
    groups = defaultdict(list)
    for c in cases:
        groups[c["group_id"]].append(c)
    for g, members in groups.items():
        if len({m["label"] for m in members}) != 1 or len({m["split"] for m in members}) != 1:
            raise ValueError(f"lesion group {g} has more than one class or split")
    new = {c["isic_id"]: c["split"] for c in cases}
    for label in sorted({c["label"] for c in cases}):
        def ordered(which):
            gs = [g for g, m in groups.items() if m[0]["label"] == label and m[0]["split"] == which]
            return sorted(gs, key=lambda g: group_order(seed, g))
        n = 0
        for g in ordered(split):
            if n + len(groups[g]) <= per_class:
                n += len(groups[g])
            else:
                for m in groups[g]:
                    new[m["isic_id"]] = "train"
        for g in ordered("train"):
            if n + len(groups[g]) <= per_class:
                n += len(groups[g])
                for m in groups[g]:
                    new[m["isic_id"]] = split
        if n != per_class:
            raise ValueError(f"{label}: only {n} {split} images can be reached by whole lesion groups, not {per_class}")
    return new
