#!/usr/bin/env python3
"""
Checks of the data and of the pipeline. Two commands:

  python src/check.py                where the research pipeline stands: for every step done / not run / problem, and the next
                                     command (reads only; changes nothing; calls no API; exit code 1 when any step has a problem)
  python src/check.py features       the feature sufficiency and shortcut check of the cleaned data (needs the image packages)

  done      the output is there and agrees with its input
  not run   the output is missing (the step has not been run here)
  problem   the output is there but incomplete, out of date or inconsistent: rerun the step

"""
import argparse
import filecmp
import hashlib
import json
import os
import shutil
import sys
import tempfile

from build import build_views
from common import (ALL_CASES, CLASSES, INDEX_DIR, KNOWLEDGE_CORPUS, PROCESSED_CASES, PROCESSED_IMAGES, RAW, SEED, SPLITS, TEST_PER_CLASS, VAL_PER_CLASS,
                    VIEWS, images_not_in_table, question_file, read_csv, view_table, write_json)

DONE, NOT_RUN, PROBLEM = "done", "not run", "problem"
PROCESSED_FILES = ("all_cases.csv", "raw_files_audit.csv", "near_duplicates.csv", "classes.json", "normalization.json",
                   "class_weights.json", "train_oversampled.csv", "train_undersampled.csv")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def step_raw(data):
    audit = os.path.join(data, PROCESSED_CASES, "raw_files_audit.csv")
    folders = [os.path.join(data, RAW, d) for d in ("Train", "Test")]
    if not all(os.path.isdir(f) for f in folders):
        return NOT_RUN, "the Kaggle download is not in data/raw/ (Train/, Test/)"
    n = sum(f.lower().endswith(".jpg") for d in folders for _, _, fs in os.walk(d) for f in fs)
    expected = len(read_csv(audit)) if os.path.isfile(audit) else None
    if expected is not None and n != expected:
        return PROBLEM, f"{n} JPG files, but the audit lists {expected}"
    return DONE, f"{n} JPG files"


def step_clean(data):
    missing = [f for f in PROCESSED_FILES if not os.path.isfile(os.path.join(data, PROCESSED_CASES, f))]
    if len(missing) == len(PROCESSED_FILES):
        return NOT_RUN, "no table in data/processed_cases/"
    if missing:
        return PROBLEM, f"missing: {', '.join(missing)}"
    cases = read_csv(os.path.join(data, ALL_CASES))
    for split, per_class in (("test", TEST_PER_CLASS), ("val", VAL_PER_CLASS)):
        counts = {c: sum(r["split"] == split and r["label"] == c for r in cases) for c in CLASSES}
        if set(counts.values()) != {per_class}:
            return PROBLEM, (f"the {split} split is not balanced to {per_class} per class ("
                             + ", ".join(f"{c} {n}" for c, n in counts.items()) + "): rerun preprocess.py, then build.py views")
    images = os.path.join(data, PROCESSED_IMAGES)
    if not os.path.isdir(images):
        return NOT_RUN, f"{len(cases)} cases; the cleaned images are not in data/{PROCESSED_IMAGES}/ (laptop only)"
    absent = [c["isic_id"] for c in cases if not os.path.isfile(os.path.join(data, c["image_path"]))]
    extra = images_not_in_table(data, [c["isic_id"] for c in cases])
    notes = []
    if absent:
        notes.append(f"{len(absent)} images missing, rerun preprocess.py (e.g. {', '.join(absent[:3])})")
    if extra:
        notes.append(f"{len(extra)} images not in the case table, left from an earlier run; delete them from "
                     f"data/{PROCESSED_IMAGES}/: {', '.join(extra[:10])}" + (" ..." if len(extra) > 10 else ""))
    if notes:
        return PROBLEM, "; ".join(notes)
    return DONE, f"{len(cases)} cases, {TEST_PER_CLASS} test and {VAL_PER_CLASS} validation images per class, every image present"


def step_views(data):
    names = [view_table(v) for v in VIEWS] + [question_file(v, s) for v in VIEWS for s in SPLITS]
    if not os.path.isfile(os.path.join(data, ALL_CASES)):
        return NOT_RUN, "needs step 1"
    missing = [n for n in names if not os.path.isfile(os.path.join(data, n))]
    if len(missing) == len(names):
        return NOT_RUN, "no view built"
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, PROCESSED_CASES))
        shutil.copy(os.path.join(data, ALL_CASES), os.path.join(d, ALL_CASES))
        build_views(d)
        stale = [n for n in names if n in missing or not filecmp.cmp(os.path.join(d, n), os.path.join(data, n), shallow=False)]
    if stale:
        return PROBLEM, f"{len(stale)} view files differ from a fresh build (e.g. {stale[0]})"
    return DONE, f"{len(names)} files, the same as a fresh build"


def step_feature_check(results):
    path = os.path.join(results, "feature_check", "feature_check_results.json")
    return (DONE, "results present") if os.path.isfile(path) else (NOT_RUN, "no results")


def step_corpus(data):
    folder = os.path.join(data, KNOWLEDGE_CORPUS)
    files = {f: os.path.join(folder, f) for f in ("passages.jsonl", "manifest.csv", "corpus_card.json")}
    present = [f for f, p in files.items() if os.path.isfile(p)]
    if not present:
        return NOT_RUN, "the corpus has not been built"
    if "manifest.csv" in present and "passages.jsonl" not in present:
        return NOT_RUN, f"manifest only ({len(read_csv(files['manifest.csv']))} passages); passages.jsonl is on the laptop"
    if len(present) < 3:
        return PROBLEM, f"only {', '.join(present)}"
    n_passages, n_manifest = count_lines(files["passages.jsonl"]), len(read_csv(files["manifest.csv"]))
    if n_passages != n_manifest:
        return PROBLEM, f"{n_passages} passages but {n_manifest} manifest rows: rebuild"
    with open(files["corpus_card.json"], encoding="utf-8") as f:
        empty = json.load(f).get("empty_topic_boxes", [])
    return DONE, f"{n_passages} passages" + (f"; topics with no passage: {', '.join(empty)}" if empty else "")


def step_references(data):
    path = os.path.join(data, KNOWLEDGE_CORPUS, "reference_texts.json")
    if not os.path.isfile(path):
        return NOT_RUN, "no reference texts"
    with open(path, encoding="utf-8") as f:
        refs = json.load(f)
    old = [c for c, r in refs.items() if "texts" not in r]
    if old:
        return PROBLEM, "the file has the old form (one text per disease): rerun build.py references"
    missing = [c for c in CLASSES if c not in refs]
    empty = [c for c, r in refs.items() if not r["texts"]]
    if missing or empty:
        return PROBLEM, "; ".join(([f"missing {missing}"] if missing else []) + ([f"no text: {empty}"] if empty else []))
    counts = ", ".join(f"{c} {len(refs[c]['texts'])}" for c in CLASSES)
    return DONE, f"{sum(len(r['texts']) for r in refs.values())} reference texts ({counts})"


def step_index(data):
    folder = os.path.join(data, INDEX_DIR)
    meta_path = os.path.join(folder, "index.json")
    if not os.path.isfile(meta_path):
        return NOT_RUN, "no index: run build.py index on the laptop (needs passages.jsonl)"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if not os.path.isfile(os.path.join(folder, "embeddings.f32")):
        return PROBLEM, "index.json without embeddings.f32: build the index again"
    if os.path.getsize(os.path.join(folder, "embeddings.f32")) != 4 * meta["n_passages"] * meta["dim"]:
        return PROBLEM, "embeddings.f32 has the wrong size: build the index again"
    passages = os.path.join(data, KNOWLEDGE_CORPUS, "passages.jsonl")
    if os.path.isfile(passages) and sha256(passages) != meta.get("passages_sha256"):
        return PROBLEM, "passages.jsonl changed after the index was built: run build.py index again"
    note = f"{meta['n_passages']} passages, {meta['dim']} numbers each ({meta['encoder']})"
    return DONE, note + ("; reference pages left out" if meta.get("exclude_reference_pages") else "; all passages")


def step_runs(data, results, models, systems=("llm_only", "standard_rag")):
    """One line per system, model, view and split that has a run folder; the test split is listed even when missing."""
    rows = []
    for system, model in [(sy, m) for sy in systems for m in models]:
        for view in VIEWS:
            for split in ("val", "test"):
                final = os.path.join(results, system, model, f"{view}_{split}")
                name = f"{system} {model} {view} {split}"
                if os.path.isdir(final + ".partial"):
                    with open(os.path.join(final + ".partial", "run.json"), encoding="utf-8") as f:
                        info = json.load(f)
                    rows.append((name, PROBLEM, f"unfinished: {info['answered']}/{info['of']} ({info['status']}); run the same "
                                                "command again to continue"))
                elif os.path.isdir(final):
                    rows.append((name, *check_run(data, final, view, split)))
                elif split == "test":
                    rows.append((name, NOT_RUN, ""))
    return rows


def check_run(data, folder, view, split):
    with open(os.path.join(folder, "run.json"), encoding="utf-8") as f:
        info = json.load(f)
    qfile = os.path.join(data, question_file(view, split))
    if info["settings"].get("questions_sha256") != sha256(qfile):
        return PROBLEM, "the question file changed after this run: the run does not match the current questions"
    report = os.path.join(folder, "report.json")
    if not os.path.isfile(report):
        return PROBLEM, "complete, but not evaluated yet (no report.json)"
    return DONE, "complete and evaluated"


def check(data="data", results="results"):
    from run import MODELS
    steps = [("1 raw Kaggle download", step_raw(data)), ("1 cleaning (preprocess.py)", step_clean(data)),
             ("2 views (build.py views)", step_views(data)), ("- feature check (check.py features)", step_feature_check(results)),
             ("3 knowledge corpus (build.py corpus)", step_corpus(data)),
             ("3b reference texts (build.py references)", step_references(data)),
             ("4 retrieval index (build.py index)", step_index(data))]
    return [(name, *r) for name, r in steps] + step_runs(data, results, sorted(MODELS))


def status_main(argv=None):
    ap = argparse.ArgumentParser(prog="check.py", description="Where the research pipeline stands (reads only).")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    rows = check(a.data, a.results)
    width = max(len(r[0]) for r in rows)
    mark = {DONE: "[x]", NOT_RUN: "[ ]", PROBLEM: "[!]"}
    for name, status, note in rows:
        print(f"{mark[status]} {name.ljust(width)}  {status:8s} {note}")
    problems = sum(r[1] == PROBLEM for r in rows)
    print(f"\n{sum(r[1] == DONE for r in rows)} done, {sum(r[1] == NOT_RUN for r in rows)} not run, {problems} with a problem")
    return 1 if problems else 0


# ---------------------------------------------------------------------------------------------------------------------------
# features: the feature sufficiency and shortcut check (python src/check.py features)
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
# Usage (this repository): python src/check.py features --data data --out results/feature_check
# Reads data/processed_cases/all_cases.csv and the images in data/processed_images/; writes feature_check_results.json and
# image_handcrafted_features.csv (a cache: when it exists, the image features are not computed again).
# The image packages are imported only when this command runs, so the pipeline check needs the Python standard library only.
def image_features(path):
    import cv2
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.measure import label as cc_label, regionprops
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
    ap = argparse.ArgumentParser(prog="check.py features")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=os.path.join("results", "feature_check"))
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
        F.to_csv(fpath, index=False)
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

def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "features":
        return features_main(argv[1:])
    return status_main(argv)


if __name__ == "__main__":
    sys.exit(main())
