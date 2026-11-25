"""The feature sufficiency and shortcut check of the cleaned dataset (python -m src.main features)."""
import argparse
import os

from src.utils.common import ALL_CASES, CLASSES
from src.utils.io import write_json
from src.utils.reproducibility import SEED


# ---------------------------------------------------------------------------------------------------------------------------
# features: the feature sufficiency and shortcut check (python -m src.main features)
# ---------------------------------------------------------------------------------------------------------------------------
# Feature sufficiency + shortcut check (CPU only, no deep learning).
#
# Answers three questions on the cleaned dataset:
#   1. Do the metadata (text) features carry diagnostic signal on their own?
#   2. Do simple image features (colour, texture, shape) carry signal?
#   3. Can a model "cheat" by recognising the image SOURCE / file properties instead of the lesion?
#
# It is an analysis of the data, not an evaluation of the LLM/RAG systems: its score is the accuracy of small classifiers
# trained here (validation and test are balanced, so the 8 classes count the same).
#
# Usage (this repository): python -m src.main features --data data --out results/data_quality
# Reads data/processed/cases/all_cases.csv and the images in data/processed/images/; writes feature_check_results.json and
# image_handcrafted_features.csv (a cache: when it exists, the image features are not computed again).
# The image packages are imported only when this command runs, so the pipeline check needs the Python standard library only.
def image_features(path):
    import cv2
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.measure import label as cc_label
    from skimage.measure import regionprops
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 225), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    f = {}
    # lesion mask: Otsu on blurred gray, keep the largest component near the centre
    _, th = cv2.threshold(cv2.GaussianBlur(gray, (7, 7), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lab_cc = cc_label(th > 0)
    props = sorted(regionprops(lab_cc), key=lambda p: -p.area)
    mask = np.zeros_like(gray, bool)
    if props:
        mask = lab_cc == props[0].label
        p = props[0]
        f.update(area_frac=p.area / mask.size, eccentricity=p.eccentricity, solidity=p.solidity,
                 extent=p.extent, perimeter_ratio=p.perimeter ** 2 / (4 * np.pi * p.area + 1e-6))
        # asymmetry: overlap with flipped mask about centroid axes
        cy, cx = map(int, p.centroid)
        m = np.roll(np.roll(mask, mask.shape[0] // 2 - cy, 0), mask.shape[1] // 2 - cx, 1)
        f["asym_lr"] = 1 - (m & m[:, ::-1]).sum() / (m.sum() + 1e-6)
        f["asym_ud"] = 1 - (m & m[::-1, :]).sum() / (m.sum() + 1e-6)
    else:
        f.update(area_frac=0, eccentricity=0, solidity=0, extent=0, perimeter_ratio=0, asym_lr=0, asym_ud=0)
    if mask.sum() < 50:
        mask = np.zeros_like(gray, bool)
        mask[56:169, 75:225] = True
    skin = ~mask
    for name, sp in (("rgb", img), ("hsv", hsv), ("lab", lab)):
        for c in range(3):
            les, bg = sp[..., c][mask].astype(float), sp[..., c][skin].astype(float)
            f[f"{name}{c}_les_mean"] = les.mean()
            f[f"{name}{c}_les_std"] = les.std()
            f[f"{name}{c}_les_p10"] = np.percentile(les, 10)
            f[f"{name}{c}_les_p90"] = np.percentile(les, 90)
            f[f"{name}{c}_contrast"] = les.mean() - (bg.mean() if bg.size else 0)
    # colour variety: number of occupied coarse HSV bins inside lesion
    hq = (hsv[..., 0][mask] // 30) * 16 + (hsv[..., 1][mask] // 64) * 4 + hsv[..., 2][mask] // 64
    f["n_colours"] = len(np.unique(hq))
    f["colour_entropy"] = float(
        -(lambda p: (p * np.log2(p + 1e-12)).sum())(np.bincount(hq, minlength=96) / max(len(hq), 1)))
    # texture: GLCM + LBP on gray
    g8 = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(g8, [1, 3], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity"):
        f[f"glcm_{prop}"] = graycoprops(glcm, prop).mean()
    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    hist = np.bincount(lbp[mask].astype(int), minlength=10) / max(mask.sum(), 1)
    for i, v in enumerate(hist):
        f[f"lbp_{i}"] = v
    return f


def score_model(name, model, Xtr, ytr, Xev, yev, classes):
    from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
    model.fit(Xtr, ytr)
    pr = model.predict(Xev)
    pp = model.predict_proba(Xev)
    out = dict(accuracy=round(accuracy_score(yev, pr), 3),
               macro_f1=round(f1_score(yev, pr, average="macro"), 3),
               macro_auc_ovr=round(roc_auc_score(yev, pp, multi_class="ovr", labels=model.classes_), 3),
               per_class_recall={c: round(r, 2) for c, r in zip(classes, recall_score(yev, pr, labels=classes, average=None))})
    print(f"{name:55s} accuracy={out['accuracy']:.3f}  macroF1={out['macro_f1']:.3f}  AUC={out['macro_auc_ovr']:.3f}")
    return out


def features_main(argv=None):
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser(prog="python -m src.main features")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=os.path.join("results", "data_quality"))
    a = ap.parse_args(argv)
    out = a.out
    os.makedirs(out, exist_ok=True)
    m = pd.read_csv(os.path.join(a.data, ALL_CASES))
    classes = CLASSES
    fpath = os.path.join(out, "image_handcrafted_features.csv")
    if os.path.exists(fpath):
        F = pd.read_csv(fpath)
    else:
        F = pd.DataFrame([dict(isic_id=i, **image_features(os.path.join(a.data, p)))
                          for i, p in zip(m.isic_id, m.image_path)])
        F.to_csv(fpath, index=False, lineterminator="\n")       # LF on Windows too
    m = m.merge(F, on="isic_id")
    img_cols = [c for c in F.columns if c != "isic_id"]
    meta_cols = ["age_z", "age_missing", "sex_male", "sex_missing", "site_missing", "site_head_neck", "site_trunk",
                 "site_upper_extremity", "site_lower_extremity", "site_palms_soles", "site_oral_genital"]
    meta_nomiss = ["age_z", "sex_male", "site_head_neck", "site_trunk", "site_upper_extremity",
                   "site_lower_extremity", "site_palms_soles", "site_oral_genital"]
    file_cols = ["orig_width", "orig_height", "orig_kb"]
    tr = m[m.split == "train"]
    ev = {"val": m[m.split == "val"], "test": m[m.split == "test"]}

    def gb():
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED)

    def lr():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))

    res = {}
    print("chance level accuracy = 0.125 (8 classes)")
    for split, E in ev.items():
        print(f"\n=== evaluated on {split} (n={len(E)}) ===")
        r = {}
        r["shortcut_file_properties_only"] = score_model("SHORTCUT: original width/height/file size only", gb(), tr[file_cols], tr.label, E[file_cols], E.label, classes)
        r["metadata_only"] = score_model("TEXT/METADATA: age, sex, site (+missing flags)", gb(), tr[meta_cols], tr.label, E[meta_cols], E.label, classes)
        r["metadata_only_no_missing_flags"] = score_model("TEXT/METADATA without missing-value flags", gb(), tr[meta_nomiss], tr.label, E[meta_nomiss], E.label, classes)
        r["image_handcrafted"] = score_model("IMAGE: colour/texture/shape features", gb(), tr[img_cols], tr.label, E[img_cols], E.label, classes)
        r["image_plus_metadata"] = score_model("MULTIMODAL: image features + metadata", gb(), tr[img_cols + meta_cols], tr.label, E[img_cols + meta_cols], E.label, classes)
        r["image_handcrafted_logreg"] = score_model("IMAGE (logistic regression)", lr(), tr[img_cols], tr.label, E[img_cols], E.label, classes)
        res[split] = r
    # residual source signal after standardisation: can image features tell HAM vs archive apart?
    src = (m.source == "HAM10000").astype(int)
    clf = gb().fit(tr[img_cols], src[tr.index])
    auc = roc_auc_score(src[ev["test"].index], clf.predict_proba(ev["test"][img_cols])[:, 1])
    print(f"\nSource detectable from STANDARDISED image features: AUC={auc:.3f} (0.5 = not detectable)")
    res["source_detectability_auc_after_standardisation"] = round(auc, 3)
    write_json(os.path.join(out, "feature_check_results.json"), res)
