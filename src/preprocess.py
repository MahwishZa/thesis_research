#!/usr/bin/env python3
"""
Preprocessing pipeline for the Kaggle "Skin Cancer ISIC - 9 classes" dataset.

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
  8. Splits         - lesion-grouped, stratified 70/15/15 + 5 CV folds.
  9. Normalisation  - per-channel mean/std and age mean/std from TRAIN only.
 10. Outputs        - image / text / multimodal dataset tables, class weights,
                      balanced sampling indices, data card.

Usage:
  python preprocess.py --raw RAW_DIR --meta META_DIR --out OUT_DIR [--no-colour-constancy]

RAW_DIR must contain "Train_1/Train 1", "Train_2/Train 2", "Test/Test" (as in the Kaggle zips).
META_DIR must contain ISIC_2019_Training_GroundTruth.csv, ISIC_2019_Training_Metadata.csv,
HAM10000_metadata.csv and isic_archive_api_441.csv.
"""
import argparse, hashlib, json, os, re
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_NAMES = {
    "MEL": "melanoma", "NV": "melanocytic nevus", "BCC": "basal cell carcinoma",
    "AK": "actinic keratosis", "BKL": "benign keratosis-like lesion (pigmented benign keratosis)",
    "DF": "dermatofibroma", "VASC": "vascular lesion", "SCC": "squamous cell carcinoma",
}
FOLDER_TO_CODE = {
    "melanoma": "MEL", "nevus": "NV", "basal cell carcinoma": "BCC",
    "actinic keratosis": "AK", "pigmented benign keratosis": "BKL",
    "seborrheic keratosis": "SK", "dermatofibroma": "DF",
    "vascular lesion": "VASC", "squamous cell carcinoma": "SCC",
}
SOURCE_DIRS = [("Train", "Train_1/Train 1"), ("Train", "Train_2/Train 2"), ("Test", "Test/Test")]
OUT_W, OUT_H = 600, 450


# ----------------------------------------------------------------------------- helpers
def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def phash(gray):
    """64-bit DCT perceptual hash (no external library)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8]
    med = np.median(d[1:].ravel()) if d.size else 0
    bits = (d > med).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def centre_crop_4x3(img):
    h, w = img.shape[:2]
    target = 4 / 3
    if w / h > target:            # too wide
        nw = int(round(h * target)); x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    nh = int(round(w / target)); y0 = (h - nh) // 2
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
    if "melanoma" in d3: return "MEL"
    if "nevus" in d3: return "NV"
    if "actinic" in d3: return "AK"
    if "dermatofibroma" in d3: return "DF"
    if "squamous" in d3: return "SCC"
    if "basal" in d3: return "BCC"
    if "seborrheic" in d3 or "lentigo" in d3 or "lichen" in d3: return "BKL"
    if "hemangioma" in d3 or "angioma" in d3 or "vascular" in d3: return "VASC"
    if d3 == "" and d1 == "malignant": return "MEL"  # melanocytic malignant with blank subtype
    return None


class UnionFind:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-colour-constancy", action="store_true")
    ap.add_argument("--ham-images", default=None,
                    help="optional folder with HAM10000 JPGs; adds real MEL and NV images from HAM10000")
    ap.add_argument("--ham-per-class", type=int, default=300)
    # Visual audit: pHash distance <=3 = same lesion (re-photographed / re-uploaded);
    # 4-6 produced false matches between different lesions in dermoscopy, so 3 is the default.
    ap.add_argument("--near-dup-threshold", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
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
    print(f"[1] files: {len(raw)}")

    # 1b. OPTIONAL: add real HAM10000 melanoma / nevus images to break the source confound
    #     (in the Kaggle copy every MEL and NV image comes from the old ISIC archive and every
    #      other class from HAM10000, so "source" alone predicts the class).
    if a.ham_images:
        hm = pd.read_csv(os.path.join(a.meta, "HAM10000_metadata.csv"))
        files = {}
        for root, _, fs in os.walk(a.ham_images):
            for f in fs:
                if f.lower().endswith(".jpg") and f.startswith("ISIC_"):
                    files[f[:-4]] = os.path.join(root, f)
        hm = hm[hm.image_id.isin(files) & hm.dx.isin(["mel", "nv"]) & ~hm.image_id.isin(raw.isic_id)]
        hm = hm.sort_values("image_id").drop_duplicates("lesion_id")       # one image per lesion
        add = []
        for dx, code in (("mel", "MEL"), ("nv", "NV")):
            pool = hm[hm.dx == dx]
            k = min(a.ham_per_class, len(pool))
            for iid in rng.choice(pool.image_id.values, k, replace=False):
                p = files[iid]
                add.append(dict(isic_id=iid, orig_split="added_HAM10000", folder_label="added from HAM10000",
                                folder_code=code, path=p, md5=md5(p)))
        raw = pd.concat([raw, pd.DataFrame(add)], ignore_index=True)
        print(f"[1b] added {len(add)} real HAM10000 images (MEL+NV, one per lesion)")

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
    print(f"[3] label corrected: {int(u.label_corrected.sum())}; HAM vs ISIC2019 disagreements: {len(bad)}")

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
        out_rel = os.path.join("images", f"{r.isic_id}.jpg")
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
    print(f"[5] blurry {u.flag_blurry.sum()}, exposure {u.flag_exposure.sum()}, "
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
    npairs["label_a"] = npairs.isic_id_a.map(lab); npairs["label_b"] = npairs.isic_id_b.map(lab)
    md5s = u.set_index("isic_id").md5
    npairs["byte_identical"] = npairs.isic_id_a.map(md5s) == npairs.isic_id_b.map(md5s)
    # same picture uploaded under two ISIC ids (distance 0) -> keep the lower id, drop the other
    drop = npairs[(npairs.phash_distance == 0) & (npairs.label_a == npairs.label_b)]
    npairs["action"] = np.where(npairs.index.isin(drop.index), "removed isic_id_b (same image, same label)",
                                "kept both; grouped into same split")
    npairs.to_csv(os.path.join(a.out, "near_duplicate_pairs.csv"), index=False)
    removed = set(drop.isic_id_b)
    conflict = npairs[(npairs.phash_distance == 0) & (npairs.label_a != npairs.label_b)]
    assert conflict.empty, f"identical images with different official labels: {conflict}"
    for x in removed:
        os.remove(os.path.join(a.out, "images", f"{x}.jpg"))
    u = u[~u.isic_id.isin(removed)].copy()
    u["group_size"] = u.groupby("group_id").isic_id.transform("size")
    print(f"[4] near-duplicate pairs: {len(npairs)}; removed as duplicates: {len(removed)}; "
          f"groups with >1 image: {(u.groupby('group_id').size() > 1).sum()}")

    # 8. splits -------------------------------------------------------------------
    y = u.label.values; g = u.group_id.values
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    tv_idx, te_idx = next(sgkf.split(u, y, g))
    u["split"] = "train"
    u.loc[u.index[te_idx], "split"] = "test"
    tv = u.loc[u.index[tv_idx]]
    sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
    tr_i, va_i = next(sgkf2.split(tv, tv.label.values, tv.group_id.values))
    u.loc[tv.index[va_i], "split"] = "val"
    # 5-fold CV over train+val (test kept untouched)
    u["cv_fold"] = -1
    tv = u[u.split != "test"]
    for k, (_, vi) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                                .split(tv, tv.label.values, tv.group_id.values)):
        u.loc[tv.index[vi], "cv_fold"] = k
    leak = set(u[u.split == "train"].group_id) & set(u[u.split == "test"].group_id)
    assert not leak, "group leakage between train and test"
    print("[8] split counts\n", pd.crosstab(u.label, u.split, margins=True))

    # 9. normalisation (TRAIN only) ------------------------------------------------
    tr = u[u.split == "train"]
    s = np.zeros(3); s2 = np.zeros(3); npx = 0
    for p in tr.image_path:
        im = np.asarray(Image.open(os.path.join(a.out, p)), dtype=np.float64) / 255.0
        s += im.sum((0, 1)); s2 += (im ** 2).sum((0, 1)); npx += im.shape[0] * im.shape[1]
    mean = s / npx; std = np.sqrt(s2 / npx - mean ** 2)
    age_mu, age_sd = float(tr.age.mean()), float(tr.age.std())
    norm = dict(image_mean_rgb=[round(x, 4) for x in mean], image_std_rgb=[round(x, 4) for x in std],
                image_size=[OUT_W, OUT_H], colour_constancy="shades_of_gray_p6" if cc else "none",
                age_mean=round(age_mu, 2), age_std=round(age_sd, 2), computed_on="train split only")
    json.dump(norm, open(os.path.join(a.out, "normalization.json"), "w"), indent=2)

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
    def text(r):
        age = f"{int(r.age)}-year-old" if pd.notna(r.age) else "age not recorded,"
        sex = r.sex if pd.notna(r.sex) else "sex not recorded"
        if pd.notna(r.site):
            detail = r.site_detail if pd.notna(r.site_detail) else ""
            generic = {"trunk", "head and neck", "head/neck", "upper extremity", "lower extremity",
                       "anogenital region", "oral/genital", "palms/soles"}
            site = r.site + (f" ({detail})" if detail and detail not in generic else "")
            where = f"Skin lesion on the {site}."
        else:
            where = "Lesion site not recorded."
        return f"Patient: {age} {sex}. {where} A dermoscopic image of the lesion is available."
    u["clinical_text"] = u.apply(text, axis=1)
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
    u[master_cols].to_csv(os.path.join(a.out, "master_clean.csv"), index=False)
    u[core + ["image_path"]].to_csv(os.path.join(a.out, "dataset_image_only.csv"), index=False)
    u[core + ["clinical_text", "age", "sex", "site"] + meta_feats].to_csv(
        os.path.join(a.out, "dataset_text_only.csv"), index=False)
    u[core + ["image_path", "clinical_text", "age", "sex", "site"] + meta_feats].to_csv(
        os.path.join(a.out, "dataset_multimodal.csv"), index=False)
    raw.merge(u[["isic_id", "label", "split"]], on="isic_id").drop(columns=["path"]).to_csv(
        os.path.join(a.out, "original_files_audit.csv"), index=False)

    # balancing helpers (TRAIN only)
    tr = u[u.split == "train"]
    cnt = tr.label.value_counts().reindex(CLASSES)
    w = (len(tr) / (len(CLASSES) * cnt)).round(4)
    json.dump(dict(class_counts_train=cnt.astype(int).to_dict(),
                   class_weights_balanced=w.to_dict(),
                   class_weights_effective_number_beta_0_999=
                   ((1 - 0.999) / (1 - 0.999 ** cnt) / ((1 - 0.999) / (1 - 0.999 ** cnt)).mean()).round(4).to_dict(),
                   imbalance_ratio=round(float(cnt.max() / cnt.min()), 2)),
              open(os.path.join(a.out, "class_balance.json"), "w"), indent=2)
    over, under = [], []
    for c in CLASSES:
        ids_c = tr[tr.label == c].isic_id.values
        over += list(ids_c) + list(rng.choice(ids_c, cnt.max() - len(ids_c), replace=True))
        under += list(rng.choice(ids_c, cnt.min(), replace=False))
    pd.DataFrame({"isic_id": over}).merge(tr[["isic_id", "label"]], on="isic_id").to_csv(
        os.path.join(a.out, "train_oversampled_index.csv"), index=False)
    pd.DataFrame({"isic_id": under}).merge(tr[["isic_id", "label"]], on="isic_id").to_csv(
        os.path.join(a.out, "train_undersampled_index.csv"), index=False)
    json.dump({c: dict(idx=i, name=CLASS_NAMES[c]) for i, c in enumerate(CLASSES)},
              open(os.path.join(a.out, "label_map.json"), "w"), indent=2)
    print("[10] done:", a.out)


if __name__ == "__main__":
    main()
