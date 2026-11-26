"""Step 1 of the data pipeline: cleans the Kaggle "Skin Cancer ISIC - 9 classes" dataset (raw -> cleaned, de-duplicated, split).

What it does (in order):
  1. Inventory      - reads every JPG in the original Train/Test folders, hashes it.
  2. Official labels- attaches the official diagnosis to every ISIC ID from
                      (a) ISIC 2019 Training GroundTruth, (b) HAM10000 metadata,
                      (c) ISIC Archive API records (for images not in ISIC 2019).
  3. Label correction - compares the folder label with the official label.
  4. Deduplication  - exact (MD5) and near-duplicate (perceptual hash) detection;
                      lesion-level grouping (same lesion photographed twice).
  5. Quality flags  - blur, exposure, dark dermoscope vignette, hair amount.
  6. Standardisation- centre-crop to 4:3, resize to 600x450, Shades-of-Gray
                      colour constancy, sRGB JPEG q=95.
  7. Metadata clean - harmonised anatomical site, age, sex; missing indicators.
  8. Splits         - lesion-grouped, stratified 70/15/15; then the test split is balanced to 30 images per class by
                      whole lesion groups, then the validation split to 20 images per class the same way (balanced_splits in
                      src/data/views.py; train keeps the natural class mix); then 5 CV folds over train+val.
  9. Normalisation  - per-channel mean/std and age mean/std from TRAIN only.
 10. Outputs        - OUT_DIR/processed/images/ (the cleaned images) and OUT_DIR/processed/cases/: all_cases.csv (every kept case),
                      raw_files_audit.csv, near_duplicates.csv, classes.json, normalization.json,
                      class_weights.json, the balanced training lists, and preprocess.log.
                      The three views are made from all_cases.csv by the next step, the views command.

Usage (this repository):
  python -m src.main preprocess --raw data/raw --meta data/annotations/official_labels --out data [--no-colour-constancy]

RAW_DIR (data/raw) holds the Kaggle download as unpacked: "Train/<class folder>" and "Test/<class folder>".
META_DIR must contain ISIC_2019_Training_GroundTruth.csv, ISIC_2019_Training_Metadata.csv,
HAM10000_metadata.csv and isic_archive_api_441.csv."""
import argparse
import hashlib
import os
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.data.views import TEST_PER_CLASS, VAL_PER_CLASS, balanced_splits, image_path, images_not_in_table
from src.utils import (
    ALL_CASES,
    CLASS_NAMES,
    CLASSES,
    OFFICIAL_LABELS,
    PROCESSED_CASES,
    PROCESSED_IMAGES,
    RAW,
    SEED,
    atomic_path,
    get_logger,
    hamming,
    write_json,
)

META_FILES = ("ISIC_2019_Training_GroundTruth.csv", "ISIC_2019_Training_Metadata.csv", "HAM10000_metadata.csv",
              "isic_archive_api_441.csv")
FOLDER_TO_CODE = {
    "melanoma": "MEL", "nevus": "NV", "basal cell carcinoma": "BCC",
    "actinic keratosis": "AK", "pigmented benign keratosis": "BKL",
    "seborrheic keratosis": "SK", "dermatofibroma": "DF",
    "vascular lesion": "VASC", "squamous cell carcinoma": "SCC",
}
SOURCE_DIRS = [("Train", "Train"), ("Test", "Test")]   # (original split, folder in RAW_DIR)
OUT_W, OUT_H = 600, 450


# ----------------------------------------------------------------------------- helpers
GENERIC_SITES = {"trunk", "head and neck", "head/neck", "upper extremity", "lower extremity", "anogenital region",
                 "oral/genital", "palms/soles"}


def clinical_text(age, sex, site, site_detail):
    """The patient's recorded facts as one sentence (real metadata only, the same template for every case, never the label).
    It says nothing about an image: each question adds its own line about the image when it shows one."""
    age_text = f"{int(age)}-year-old" if pd.notna(age) else "age not recorded,"
    sex_text = sex if pd.notna(sex) else "sex not recorded"
    if pd.notna(site):
        detail = site_detail if pd.notna(site_detail) else ""
        where = f"Skin lesion on the {site}" + (f" ({detail})" if detail and detail not in GENERIC_SITES else "") + "."
    else:
        where = "Lesion site not recorded."
    return f"Patient: {age_text} {sex_text}. {where}"


def audit_table(raw, kept, same_picture_as, official_label):
    """Every original file with its final label and split. A file of a removed copy keeps its official label, and its split says
    which kept case it duplicates. Stops if any original file is left without a split."""
    audit = raw.merge(kept[["isic_id", "label", "split"]], on="isic_id", how="left")
    gone = audit.isic_id.isin(list(same_picture_as))
    if gone.any():
        audit.loc[gone, "label"] = audit.loc[gone, "isic_id"].map(official_label)
        audit.loc[gone, "split"] = [f"removed: same picture as {same_picture_as[i]}" for i in audit.loc[gone, "isic_id"]]
    assert audit.split.notna().all() and audit.label.notna().all(), "an original file has no label or split in the audit"
    return audit.drop(columns=["path"])


def image_rel_path(isic_id):
    """Where a cleaned image is, relative to the output folder, with "/" on every system (never "\\" from Windows)."""
    return image_path(isic_id)


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def check_inputs(raw, meta):
    """Stops with a clear message when an input folder or file is missing, before any work is done."""
    missing = [os.path.join(raw, sub) for _, sub in SOURCE_DIRS if not os.path.isdir(os.path.join(raw, sub))]
    missing += [os.path.join(meta, f) for f in META_FILES if not os.path.isfile(os.path.join(meta, f))]
    if missing:
        raise SystemExit("missing input(s):\n  " + "\n  ".join(missing))


def save_csv(df, path):
    """Writes a table in one step (an interrupted run leaves the previous file untouched)."""
    with atomic_path(path) as tmp:
        df.to_csv(tmp, index=False, lineterminator="\n")   # LF on every system, so a rerun on Windows gives the same bytes


def phash(gray):
    """64-bit DCT perceptual hash (no external library)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8]
    med = np.median(d[1:].ravel()) if d.size else 0
    bits = (d > med).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def centre_crop_4x3(img):
    h, w = img.shape[:2]
    target = 4 / 3
    if w / h > target:            # too wide
        nw = int(round(h * target))
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    nh = int(round(w / target))
    y0 = (h - nh) // 2
    return img[y0:y0 + nh, :]


def shades_of_gray(img_rgb, p=6):
    """Shades-of-Gray colour constancy (Finlayson & Trezzi 2004), as used in ISIC challenge winners."""
    img = img_rgb.astype(np.float32)
    norm = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1 / p)
    norm = norm / (np.sqrt(np.sum(norm ** 2)) + 1e-6)
    gain = 1 / (norm * np.sqrt(3) + 1e-6)
    return np.clip(img * gain, 0, 255).astype(np.uint8)


def quality(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (600, 450), interpolation=cv2.INTER_AREA)
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
    bright = float(small.mean())
    h, w = small.shape
    c = max(8, int(0.08 * min(h, w)))
    corners = np.concatenate([small[:c, :c].ravel(), small[:c, -c:].ravel(),
                              small[-c:, :c].ravel(), small[-c:, -c:].ravel()])
    dark_corner_frac = float((corners < 30).mean())
    # hair: black-hat morphology (DullRazor idea), fraction of strong thin dark structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    bh = cv2.morphologyEx(small, cv2.MORPH_BLACKHAT, kernel)
    hair_frac = float((bh > 25).mean())
    return blur, bright, dark_corner_frac, hair_frac


SITE_MAP = {
    # HAM10000 localization
    "scalp": "head/neck", "face": "head/neck", "ear": "head/neck", "neck": "head/neck",
    "back": "trunk", "trunk": "trunk", "chest": "trunk", "abdomen": "trunk",
    "upper extremity": "upper extremity", "hand": "upper extremity",
    "lower extremity": "lower extremity", "foot": "lower extremity",
    "acral": "palms/soles", "genital": "oral/genital", "unknown": None,
    # ISIC 2019 anatom_site_general
    "anterior torso": "trunk", "posterior torso": "trunk", "lateral torso": "trunk",
    "head/neck": "head/neck", "palms/soles": "palms/soles", "oral/genital": "oral/genital",
    # ISIC Archive API anatom_site_general (v2 vocabulary)
    "head and neck": "head/neck", "anogenital region": "oral/genital",
}


def harmonise_site(x):
    if pd.isna(x) or str(x).strip() == "":
        return None
    return SITE_MAP.get(str(x).strip().lower(), str(x).strip().lower())


def api_to_code(d1, d3):
    d3 = "" if pd.isna(d3) else str(d3).lower()
    d1 = "" if pd.isna(d1) else str(d1).lower()
    if "melanoma" in d3:
        return "MEL"
    if "nevus" in d3:
        return "NV"
    if "actinic" in d3:
        return "AK"
    if "dermatofibroma" in d3:
        return "DF"
    if "squamous" in d3:
        return "SCC"
    if "basal" in d3:
        return "BCC"
    if "seborrheic" in d3 or "lentigo" in d3 or "lichen" in d3:
        return "BKL"
    if "hemangioma" in d3 or "angioma" in d3 or "vascular" in d3:
        return "VASC"
    if d3 == "" and d1 == "malignant":  # melanocytic malignant with blank subtype
        return "MEL"
    return None


class UnionFind:
    def __init__(self): self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b): self.p[self.find(a)] = self.find(b)


# ----------------------------------------------------------------------------- main
def preprocess_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main preprocess")
    ap.add_argument("--raw", default=os.path.join("data", RAW), help="the Kaggle download")
    ap.add_argument("--meta", default=os.path.join("data", OFFICIAL_LABELS), help="the official label files")
    ap.add_argument("--out", default="data", help="the data folder; writes processed/images/ and processed/cases/ in it")
    ap.add_argument("--no-colour-constancy", action="store_true")
    # Visual audit: pHash distance <=3 = same lesion (re-photographed / re-uploaded);
    # 4-6 produced false matches between different lesions in dermoscopy, so 3 is the default.
    ap.add_argument("--near-dup-threshold", type=int, default=3)
    ap.add_argument("--test-per-class", type=int, default=TEST_PER_CLASS, help="images of each class in the balanced test split")
    ap.add_argument("--val-per-class", type=int, default=VAL_PER_CLASS, help="images of each class in the balanced validation split")
    a = ap.parse_args(argv)
    check_inputs(a.raw, a.meta)
    proc = os.path.join(a.out, PROCESSED_CASES)
    os.makedirs(os.path.join(a.out, PROCESSED_IMAGES), exist_ok=True)
    os.makedirs(proc, exist_ok=True)
    log = get_logger("preprocess", os.path.join(proc, "preprocess.log"))
    log.info(f"preprocess: raw={a.raw} meta={a.meta} out={a.out} colour_constancy={not a.no_colour_constancy}")
    rng = np.random.default_rng(SEED)

    # 1. inventory -------------------------------------------------------------
    rows = []
    for split, sub in SOURCE_DIRS:
        d = os.path.join(a.raw, sub)
        for folder in sorted(os.listdir(d)):
            for f in sorted(os.listdir(os.path.join(d, folder))):
                if not f.lower().endswith(".jpg"):
                    continue
                p = os.path.join(d, folder, f)
                rows.append(dict(isic_id=f[:-4], orig_split=split, folder_label=folder,
                                 folder_code=FOLDER_TO_CODE[folder], path=p, md5=md5(p)))
    raw = pd.DataFrame(rows)
    log.info(f"[1] files: {len(raw)}")

    # 2. official labels ---------------------------------------------------------
    gt = pd.read_csv(os.path.join(a.meta, "ISIC_2019_Training_GroundTruth.csv"))
    gt["gt2019"] = gt[["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]].idxmax(axis=1)
    md19 = pd.read_csv(os.path.join(a.meta, "ISIC_2019_Training_Metadata.csv"))
    ham = pd.read_csv(os.path.join(a.meta, "HAM10000_metadata.csv")).rename(
        columns={"image_id": "isic_id", "lesion_id": "ham_lesion_id", "dataset": "ham_dataset",
                 "age": "ham_age", "sex": "ham_sex", "localization": "ham_site"})
    api = pd.read_csv(os.path.join(a.meta, "isic_archive_api_441.csv"), dtype=str).rename(
        columns={"age_approx": "api_age", "sex": "api_sex", "anatom_site_general": "api_site",
                 "lesion_id": "api_lesion_id"})

    uniq = raw.groupby("isic_id").agg(
        md5=("md5", "first"), n_files=("md5", "size"), n_md5=("md5", "nunique"),
        folder_labels=("folder_label", lambda s: " | ".join(sorted(set(s)))),
        folder_codes=("folder_code", lambda s: sorted(set(s))),
        orig_splits=("orig_split", lambda s: " | ".join(sorted(set(s)))),
        path=("path", "first")).reset_index()
    assert (uniq.n_md5 == 1).all(), "same ISIC id with different file content"
    u = (uniq.merge(gt[["image", "gt2019"]].rename(columns={"image": "isic_id"}), on="isic_id", how="left")
             .merge(md19.rename(columns={"image": "isic_id", "lesion_id": "isic19_lesion_id",
                                         "age_approx": "isic19_age", "sex": "isic19_sex",
                                         "anatom_site_general": "isic19_site"}), on="isic_id", how="left")
             .merge(ham, on="isic_id", how="left")
             .merge(api, on="isic_id", how="left"))
    u["api_code"] = [api_to_code(x, y) for x, y in zip(u.diagnosis_1, u.diagnosis_3)]
    u["label"] = u.gt2019.where(u.gt2019.notna(), u.api_code)
    u["label_source"] = np.where(u.gt2019.notna(), "ISIC2019_GroundTruth",
                                 np.where(u.api_code.notna(), "ISIC_Archive_API", "none"))
    ham_map = {"mel": "MEL", "nv": "NV", "bcc": "BCC", "akiec": "AK/SCC", "bkl": "BKL", "df": "DF", "vasc": "VASC"}
    u["ham_code"] = u.dx.map(ham_map)
    assert u.label.notna().all(), "some images have no official label"

    # 3. label correction --------------------------------------------------------
    u["folder_label_correct"] = [lab in codes for lab, codes in zip(u.label, u.folder_codes)]
    u["was_in_two_folders"] = u.folder_codes.str.len() > 1
    u["label_corrected"] = [(len(c) > 1) or (c[0] != lab) for lab, c in zip(u.label, u.folder_codes)]
    u["folder_codes"] = u.folder_codes.str.join(" | ")
    # consistency check HAM vs ISIC2019
    chk = u[u.ham_code.notna()]
    bad = chk[~chk.apply(lambda r: r.label in r.ham_code.split("/"), axis=1)]
    log.info(f"[3] label corrected: {int(u.label_corrected.sum())}; HAM vs ISIC2019 disagreements: {len(bad)}")

    # diagnosis confirmation type (kept as analysis column, NOT a model input: it leaks the label)
    u["confirm_type"] = u.dx_type.map({"histo": "histopathology", "follow_up": "follow-up",
                                       "consensus": "expert consensus", "confocal": "confocal microscopy"})
    u["confirm_type"] = u.confirm_type.fillna(u.diagnosis_confirm_type.replace({"": np.nan}))
    u["diagnosis_detail"] = u.diagnosis_3

    # 7. metadata harmonisation ---------------------------------------------------
    def first(*cols):
        out = pd.Series([np.nan] * len(u), dtype=object)
        for c in cols:
            v = u[c].replace({"": np.nan, "unknown": np.nan})
            out = out.where(out.notna(), v)
        return out
    u["age"] = pd.to_numeric(first("ham_age", "isic19_age", "api_age"), errors="coerce")
    u["age"] = u.age.where(u.age > 0)   # 0 is a missing code in HAM
    u["sex"] = first("ham_sex", "isic19_sex", "api_sex").str.lower()
    site_raw = first("ham_site", "isic19_site", "api_site")
    u["site_detail"] = site_raw.str.lower()
    u["site"] = site_raw.map(harmonise_site)
    u["lesion_id"] = first("ham_lesion_id", "isic19_lesion_id", "api_lesion_id")
    n = u.isic_id.str[5:].astype(int)
    u["source"] = np.where(u.ham_lesion_id.notna(), "HAM10000",
                           np.where((n >= 24306) & (n <= 34320), "HAM10000", "ISIC archive (2016-2017 era)"))

    # 4/5/6. images: near-dup hash, quality, standardisation ------------------------
    recs = []
    cc = not a.no_colour_constancy
    for r in u.itertuples():
        img = np.array(Image.open(r.path).convert("RGB"))
        h0, w0 = img.shape[:2]
        blur, bright, dark_c, hair = quality(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ph = phash(gray)
        crop = centre_crop_4x3(img)
        std = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        if cc:
            std = shades_of_gray(std)
        out_rel = image_rel_path(r.isic_id)
        Image.fromarray(std).save(os.path.join(a.out, out_rel), quality=95)
        recs.append(dict(isic_id=r.isic_id, orig_width=w0, orig_height=h0,
                         orig_kb=round(os.path.getsize(r.path) / 1024, 1),
                         blur_laplacian_var=round(blur, 1), mean_brightness=round(bright, 1),
                         dark_corner_frac=round(dark_c, 3), hair_frac=round(hair, 4),
                         phash=ph, image_path=out_rel))
    q = pd.DataFrame(recs)
    u = u.merge(q, on="isic_id")
    u["flag_blurry"] = u.blur_laplacian_var < u.blur_laplacian_var.quantile(0.02)
    u["flag_exposure"] = (u.mean_brightness < 60) | (u.mean_brightness > 235)
    u["flag_dermoscope_vignette"] = u.dark_corner_frac > 0.5
    u["flag_heavy_hair"] = u.hair_frac > u.hair_frac.quantile(0.95)
    log.info(f"[5] blurry {u.flag_blurry.sum()}, exposure {u.flag_exposure.sum()}, "
          f"vignette {u.flag_dermoscope_vignette.sum()}, heavy hair {u.flag_heavy_hair.sum()}")

    # near-duplicates (different ISIC id, visually same) + lesion grouping
    uf = UnionFind()
    ids, hashes = u.isic_id.tolist(), u.phash.tolist()
    near_pairs = []
    for i in range(len(ids)):
        uf.find(ids[i])
        for j in range(i + 1, len(ids)):
            if hamming(hashes[i], hashes[j]) <= a.near_dup_threshold:
                near_pairs.append((ids[i], ids[j], hamming(hashes[i], hashes[j])))
                uf.union(ids[i], ids[j])
    for lid, grp in u[u.lesion_id.notna()].groupby("lesion_id"):
        g = grp.isic_id.tolist()
        for x in g[1:]:
            uf.union(g[0], x)
    u["group_id"] = [uf.find(x) for x in u.isic_id]
    npairs = pd.DataFrame(near_pairs, columns=["isic_id_a", "isic_id_b", "phash_distance"])
    lab = u.set_index("isic_id").label
    npairs["label_a"] = npairs.isic_id_a.map(lab)
    npairs["label_b"] = npairs.isic_id_b.map(lab)
    md5s = u.set_index("isic_id").md5
    npairs["byte_identical"] = npairs.isic_id_a.map(md5s) == npairs.isic_id_b.map(md5s)
    # same picture uploaded under two ISIC ids (distance 0) -> keep the lower id, drop the other
    drop = npairs[(npairs.phash_distance == 0) & (npairs.label_a == npairs.label_b)]
    npairs["action"] = np.where(npairs.index.isin(drop.index), "removed isic_id_b (same image, same label)",
                                "kept both; grouped into same split")
    save_csv(npairs, os.path.join(proc, "near_duplicates.csv"))
    removed = set(drop.isic_id_b)
    conflict = npairs[(npairs.phash_distance == 0) & (npairs.label_a != npairs.label_b)]
    assert conflict.empty, f"identical images with different official labels: {conflict}"
    for x in removed:
        os.remove(os.path.join(a.out, PROCESSED_IMAGES, f"{x}.jpg"))
    u = u[~u.isic_id.isin(removed)].copy()
    u["group_size"] = u.groupby("group_id").isic_id.transform("size")
    log.info(f"[4] near-duplicate pairs: {len(npairs)}; removed as duplicates: {len(removed)}; "
          f"groups with >1 image: {(u.groupby('group_id').size() > 1).sum()}")

    # 8. splits -------------------------------------------------------------------
    y = u.label.values
    g = u.group_id.values
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    tv_idx, te_idx = next(sgkf.split(u, y, g))
    u["split"] = "train"
    u.loc[u.index[te_idx], "split"] = "test"
    tv = u.loc[u.index[tv_idx]]
    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
    tr_i, va_i = next(sgkf2.split(tv, tv.label.values, tv.group_id.values))
    u.loc[tv.index[va_i], "split"] = "val"
    # balanced test split, then balanced validation split (the test split is not touched by the second pass): the same number of
    # images of every class in each, by whole lesion groups; train keeps the natural class mix
    for name, per_class in (("test", a.test_per_class), ("val", a.val_per_class)):
        before = u.split.copy()
        u["split"] = u.isic_id.map(balanced_splits(u[["isic_id", "group_id", "label", "split"]].to_dict("records"),
                                                   per_class, split=name))
        log.info(f"[8] balanced {name} split: {per_class} images per class; moved {name}->train "
                 f"{((before == name) & (u.split == 'train')).sum()}, train->{name} {((before == 'train') & (u.split == name)).sum()}")
    # 5-fold CV over train+val (test kept untouched)
    u["cv_fold"] = -1
    tv = u[u.split != "test"]
    for k, (_, vi) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                                .split(tv, tv.label.values, tv.group_id.values)):
        u.loc[tv.index[vi], "cv_fold"] = k
    for one, other in (("train", "test"), ("train", "val"), ("val", "test")):
        leak = set(u[u.split == one].group_id) & set(u[u.split == other].group_id)
        assert not leak, f"group leakage between {one} and {other}"
    log.info("[8] split counts\n" + str(pd.crosstab(u.label, u.split, margins=True)))

    # 9. normalisation (TRAIN only) ------------------------------------------------
    tr = u[u.split == "train"]
    s = np.zeros(3)
    s2 = np.zeros(3)
    npx = 0
    for p in tr.image_path:
        im = np.asarray(Image.open(os.path.join(a.out, p)), dtype=np.float64) / 255.0
        s += im.sum((0, 1))
        s2 += (im ** 2).sum((0, 1))
        npx += im.shape[0] * im.shape[1]
    mean = s / npx
    std = np.sqrt(s2 / npx - mean ** 2)
    age_mu, age_sd = float(tr.age.mean()), float(tr.age.std())
    norm = dict(image_mean_rgb=[round(x, 4) for x in mean], image_std_rgb=[round(x, 4) for x in std],
                image_size=[OUT_W, OUT_H], colour_constancy="shades_of_gray_p6" if cc else "none",
                age_mean=round(age_mu, 2), age_std=round(age_sd, 2), computed_on="train split only")
    write_json(os.path.join(proc, "normalization.json"), norm)

    # metadata features (model-ready, no label leakage)
    u["age_missing"] = u.age.isna().astype(int)
    u["age_z"] = ((u.age - age_mu) / age_sd).fillna(0.0).round(4)
    u["sex_missing"] = u.sex.isna().astype(int)
    u["sex_male"] = (u.sex == "male").astype(int)
    u["site_missing"] = u.site.isna().astype(int)
    site_cats = ["head/neck", "trunk", "upper extremity", "lower extremity", "palms/soles", "oral/genital"]
    for c in site_cats:
        u["site_" + re.sub(r"[^a-z]+", "_", c)] = (u.site == c).astype(int)

    # clinical text (real metadata only, deterministic, label-free)
    u["clinical_text"] = [clinical_text(*x) for x in zip(u.age, u.sex, u.site, u.site_detail)]
    u["label_name"] = u.label.map(CLASS_NAMES)
    u["label_idx"] = u.label.map({c: i for i, c in enumerate(CLASSES)})

    # 10. outputs --------------------------------------------------------------------
    meta_feats = ["age_z", "age_missing", "sex_male", "sex_missing", "site_missing"] + \
                 ["site_" + re.sub(r"[^a-z]+", "_", c) for c in site_cats]
    core = ["isic_id", "split", "cv_fold", "group_id", "label", "label_idx", "label_name"]
    master_cols = core + ["label_source", "folder_labels", "folder_label_correct", "label_corrected",
                          "was_in_two_folders", "n_files", "orig_splits", "diagnosis_detail", "confirm_type",
                          "age", "sex", "site", "site_detail", "lesion_id", "group_size", "source",
                          "orig_width", "orig_height", "orig_kb", "blur_laplacian_var", "mean_brightness",
                          "dark_corner_frac", "hair_frac", "flag_blurry", "flag_exposure",
                          "flag_dermoscope_vignette", "flag_heavy_hair", "image_path", "clinical_text"] + meta_feats
    u = u.sort_values(["split", "label", "isic_id"])
    save_csv(u[master_cols], os.path.join(proc, "all_cases.csv"))
    save_csv(audit_table(raw, u, dict(zip(drop.isic_id_b, drop.isic_id_a)), lab), os.path.join(proc, "raw_files_audit.csv"))

    # balancing helpers (TRAIN only)
    tr = u[u.split == "train"]
    cnt = tr.label.value_counts().reindex(CLASSES)
    w = (len(tr) / (len(CLASSES) * cnt)).round(4)
    write_json(os.path.join(proc, "class_weights.json"),
               dict(class_counts_train=cnt.astype(int).to_dict(), class_weights_balanced=w.to_dict(),
                    class_weights_effective_number_beta_0_999=
                    ((1 - 0.999) / (1 - 0.999 ** cnt) / ((1 - 0.999) / (1 - 0.999 ** cnt)).mean()).round(4).to_dict(),
                    imbalance_ratio=round(float(cnt.max() / cnt.min()), 2)))
    over, under = [], []
    for c in CLASSES:
        ids_c = tr[tr.label == c].isic_id.values
        over += list(ids_c) + list(rng.choice(ids_c, cnt.max() - len(ids_c), replace=True))
        under += list(rng.choice(ids_c, cnt.min(), replace=False))
    save_csv(pd.DataFrame({"isic_id": over}).merge(tr[["isic_id", "label"]], on="isic_id"),
             os.path.join(proc, "train_oversampled.csv"))
    save_csv(pd.DataFrame({"isic_id": under}).merge(tr[["isic_id", "label"]], on="isic_id"),
             os.path.join(proc, "train_undersampled.csv"))
    write_json(os.path.join(proc, "classes.json"), {c: dict(idx=i, name=CLASS_NAMES[c]) for i, c in enumerate(CLASSES)})
    extra = images_not_in_table(a.out, u.isic_id)
    if extra:
        log.warning(f"[10] {len(extra)} images in {os.path.join(a.out, PROCESSED_IMAGES)} are not in the case table (left from an "
                    f"earlier run; not deleted, delete them by hand): {', '.join(extra)}")
    log.info(f"[10] done: {proc}. Next: python -m src.main views --data {a.out}")


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
