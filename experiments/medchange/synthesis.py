"""Evidence-synthesis layer (stage 2): from per-paper stance to a verdict.

Per question, the first 8 candidates of the frozen pool (cross-encoder order) each have stance
probabilities from ``stance.py``. They are summarised into four numbers (optionally weighting newer
and stronger study types more) and combined, with or without the RAG answer's verdict (B1), by a
small multinomial logistic regression fitted on the dev split and then frozen.

    python -m experiments.medchange.synthesis fit                     # dev: cross-validation, gate 2, freeze
    python -m experiments.medchange.synthesis predict --split confirm # confirmatory: apply the frozen model

Arms written (``data/synthesis_<split>.jsonl``): B1R (B1's verdict through the same fitting; fairness
control), S0-S3 (stance features only; weights none / recency / study type / both), H0-H3 (B1's
verdict plus the stance features), H1C and H3C (H1/H3 with dates shuffled; falsification controls).
Nothing is tuned: the penalty, top-k, half-life and study-type weights are fixed in the plan
(§4 of the stage-2 protocol, the file ``experiment_plan.md`` at commit 92e3aaf); the only data-dependent choice is which of H0-H3 is *selected*,
by the pre-stated rule (H0 unless another variant's dev cross-validated accuracy is >= 1.0 pp higher).
No feature uses an item's kind, change type, previous-version date or label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import arms as A
from .generate_answers import check_config, config_path, file_sha256, load_jsonl
from .stance import TOP_K, top_candidates

HERE = Path(__file__).resolve().parent
LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION")
L2 = 5.0                         # ridge on the summed log-loss, standardised features, fixed
INTERCEPT_RIDGE = 1e-6           # numerical only
CV_FOLDS, CV_REPEATS, CV_SEED = 5, 50, 20261003
SELECTION_MARGIN = 0.01          # a weighted variant must beat H0 by this much (dev CV accuracy)
MIN_AUC, MIN_GAIN = 0.60, 0.01   # gate 2
SR_MA_TYPES = ("Meta-Analysis", "Systematic Review")
RCT_TYPES = ("Randomized Controlled Trial", "Controlled Clinical Trial", "Clinical Trial")
DESIGN_WEIGHTS = {"SR/MA": 3.0, "RCT": 2.0, "other": 1.0}
FEATURES = ("signed", "neither_share", "conflict", "mass")
VARIANTS = ("B1R", "S0", "S1", "S2", "S3", "H0", "H1", "H2", "H3", "H1C", "H3C")
HYBRIDS = ("H0", "H1", "H2", "H3")


# --------------------------------------------------------------------------------------
# Weights and features
# --------------------------------------------------------------------------------------

def study_type(pubtypes: Optional[Sequence[str]]) -> str:
    """"SR/MA", "RCT" or "other", from PubMed publication types."""
    kinds = set(pubtypes or ())
    if kinds & set(SR_MA_TYPES):
        return "SR/MA"
    if kinds & set(RCT_TYPES):
        return "RCT"
    return "other"


def design_weight(pubtypes: Optional[Sequence[str]]) -> float:
    return DESIGN_WEIGHTS[study_type(pubtypes)]


def variant_spec(name: str) -> dict:
    """What a variant uses: B1's verdict, stance features, recency weights, study-type weights, shuffled dates."""
    if name == "B1R":
        return {"verdict": True, "stance": False, "recency": False, "design": False, "shuffled": False}
    m = re.fullmatch(r"([SH])([0-3])(C?)", name)
    if not m:
        raise ValueError(f"unknown variant {name!r}")
    k = int(m.group(2))
    return {"verdict": m.group(1) == "H", "stance": True, "recency": k in (1, 3), "design": k in (2, 3),
            "shuffled": bool(m.group(3))}


def paper_weights(cands: Sequence[dict], dated: dict, cutoff: str, recency: bool, design: bool) -> list[float]:
    """w_i = (2^(-age/H) if recency) * (study-type weight if design); ``dated`` maps pmid to the
    candidate whose dates are used (the shuffled pool for the falsification controls)."""
    out = []
    for c in cands:
        w = 1.0
        if recency:
            w *= A.recency(dated[c["pmid"]], cutoff)
        if design:
            w *= design_weight(c.get("pubtypes"))
        out.append(w)
    return out


def stance_features(probs: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    """[signed, neither_share, conflict, mass] from (p_sup, p_con, p_nei) per paper and paper weights."""
    total = float(sum(weights))
    if total <= 0 or not probs:
        return [0.0, 1.0, 0.0, 0.0]
    sup = sum(w * p[0] for w, p in zip(weights, probs))
    con = sum(w * p[1] for w, p in zip(weights, probs))
    nei = sum(w * p[2] for w, p in zip(weights, probs))
    return [(sup - con) / total, nei / total, 2.0 * min(sup, con) / total, float(np.log1p(sup + con))]


WORDINGS_USED = {"A": ("A",), "B": ("B",), "both": ("A", "B")}


def load_stance_probs(records: Sequence[dict], wording: str) -> dict:
    """{(item_id, pmid): (p_sup, p_con, p_nei)}; an invalid output counts as "neither".

    ``wording`` is "A", "B" or "both". With "both" the two wordings' probabilities are averaged
    (the pre-declared stance measure: it removes the choice of a wording and reduces the order
    sensitivity of a single prompt); a paper missing under either wording is left out, so the
    caller's missing-stance check fails loudly instead of silently using half an ensemble."""
    wanted = WORDINGS_USED[wording]
    seen: dict = {}
    for r in records:
        if r.get("control") or r["wording"] not in wanted:
            continue
        seen.setdefault((r["item_id"], r["pmid"]), {})[r["wording"]] = (
            tuple(r["probs"]) if r["probs"] else (0.0, 0.0, 1.0))
    return {key: tuple(sum(v[w][k] for w in wanted) / len(wanted) for k in range(3))
            for key, v in seen.items() if all(w in v for w in wanted)}


def item_features(item_id: str, pool: dict, cutoff: str, probs: dict, *, recency: bool, design: bool,
                  shuffled: bool, top_k: int = TOP_K) -> list[float]:
    cands = top_candidates(pool, top_k)
    ranked = sorted(pool["candidates"], key=lambda c: (c["rank"], c["pmid"]))
    dated = {c["pmid"]: c for c in (A.shuffle_dates(ranked, seed=item_id) if shuffled else ranked)}
    chosen = []
    for c in cands:
        if (item_id, c["pmid"]) not in probs:
            raise KeyError(f"missing stance for item {item_id}, paper {c['pmid']}")
        chosen.append(probs[(item_id, c["pmid"])])
    return stance_features(chosen, paper_weights(cands, dated, cutoff, recency, design))


def verdict_onehot(verdict: Optional[str]) -> list[float]:
    return [float(verdict == label) for label in LABELS]


def design_matrix(name: str, items: Sequence[dict], pools: dict, probs: dict, b1_verdicts: dict) -> np.ndarray:
    """Rows in the order of ``items``; columns: B1 verdict one-hot (if used) then the four stance features."""
    spec = variant_spec(name)
    rows = []
    for it in items:
        row: list[float] = []
        if spec["verdict"]:
            row += verdict_onehot(b1_verdicts.get(it["item_id"]))
        if spec["stance"]:
            row += item_features(it["item_id"], pools[it["item_id"]], it["newest"]["date"], probs,
                                 recency=spec["recency"], design=spec["design"], shuffled=spec["shuffled"])
        rows.append(row)
    return np.array(rows, dtype=float).reshape(len(items), -1)


def gold_indices(items: Sequence[dict]) -> np.ndarray:
    return np.array([LABELS.index(it["newest"]["label"]) for it in items], dtype=int)


# --------------------------------------------------------------------------------------
# The fitted layer: multinomial logistic regression, fixed ridge, Newton's method
# --------------------------------------------------------------------------------------

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_softmax(X, y, l2: float = L2, n_classes: int = 3, max_iter: int = 100, tol: float = 1e-9) -> dict:
    """Standardise the features, then minimise the summed log-loss + (l2/2)·||weights||² (intercepts
    not penalised) by damped Newton steps; class 0 is the reference. Returns a plain-data model."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=int)
    n, d = X.shape
    mean = X.mean(axis=0) if d else np.zeros(0)
    std = X.std(axis=0) if d else np.zeros(0)
    std = np.where(std < 1e-12, 1.0, std)
    Xb = np.c_[np.ones(n), (X - mean) / std]
    K = n_classes
    Y = np.eye(K)[y]
    pen = np.r_[INTERCEPT_RIDGE, np.full(d, l2)]
    theta = np.zeros((d + 1, K - 1))

    def objective(th):
        P = _softmax(Xb @ np.c_[np.zeros(d + 1), th])
        return -np.log(P[np.arange(n), y] + 1e-300).sum() + 0.5 * (pen[:, None] * th ** 2).sum()

    for _ in range(max_iter):
        W = np.c_[np.zeros(d + 1), theta]
        P = _softmax(Xb @ W)
        grad = Xb.T @ (P - Y)[:, 1:] + pen[:, None] * theta
        g = grad.T.reshape(-1)
        if np.abs(g).max() < tol:
            break
        size = d + 1
        H = np.zeros((size * (K - 1), size * (K - 1)))
        for a in range(1, K):
            for b in range(1, K):
                w = P[:, a] * ((a == b) - P[:, b])
                block = (Xb * w[:, None]).T @ Xb
                if a == b:
                    block = block + np.diag(pen)
                H[(a - 1) * size:a * size, (b - 1) * size:b * size] = block
        step = np.linalg.solve(H + 1e-10 * np.eye(len(H)), g).reshape(K - 1, size).T
        current, t = objective(theta), 1.0
        while objective(theta - t * step) > current + 1e-12 and t > 1e-8:
            t *= 0.5
        theta = theta - t * step
    return {"mean": mean.tolist(), "std": std.tolist(), "W": np.c_[np.zeros(d + 1), theta].tolist(),
            "labels": list(LABELS), "l2": l2}


def predict_proba(model: dict, X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    mean, std = np.asarray(model["mean"], dtype=float), np.asarray(model["std"], dtype=float)
    Xb = np.c_[np.ones(len(X)), (X - mean) / std if X.shape[1] else X]
    return _softmax(Xb @ np.asarray(model["W"], dtype=float))


def repeated_cv(X, y, folds: int = CV_FOLDS, repeats: int = CV_REPEATS, seed: int = CV_SEED,
                l2: float = L2) -> np.ndarray:
    """Out-of-fold predicted class indices, shape (repeats, n). The fold assignment depends only on
    the seed and n, so different variants are compared on identical splits."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(y)
    pred = np.zeros((repeats, n), dtype=int)
    for r in range(repeats):
        order = list(range(n))
        random.Random(seed * 1000 + r).shuffle(order)
        for f in range(folds):
            test = np.array(sorted(order[f::folds]), dtype=int)
            train = np.setdiff1d(np.arange(n), test)
            pred[r, test] = predict_proba(fit_softmax(X[train], y[train], l2), X[test]).argmax(axis=1)
    return pred


def cv_accuracy(pred: np.ndarray, y: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    ok = pred == y[None, :]
    if mask is not None:
        ok = ok[:, mask]
    return float(ok.mean())


def cv_recall(pred: np.ndarray, y: np.ndarray) -> dict:
    return {LABELS[k]: float(((pred == k) & (y[None, :] == k)).sum(axis=1).mean() / max((y == k).sum(), 1))
            for k in range(len(LABELS))}


def bagged_verdicts(pred: np.ndarray) -> list[int]:
    """The most frequent out-of-fold prediction per item across repeats (ties: lowest class index)."""
    return [int(np.bincount(pred[:, i], minlength=len(LABELS)).argmax()) for i in range(pred.shape[1])]


def auc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    s = np.asarray(scores, dtype=float)
    pos = np.asarray(positive, dtype=bool)
    if pos.all() or not pos.any():
        raise ValueError("AUC needs both classes")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def select_variant(cv_acc: dict, margin: float = SELECTION_MARGIN) -> str:
    """H0 unless a weighted hybrid beats it by ``margin`` in dev cross-validated accuracy; then the best."""
    base = cv_acc["H0"]
    better = {v: cv_acc[v] for v in HYBRIDS[1:] if cv_acc[v] >= base + margin - 1e-12}
    return max(better, key=lambda v: (better[v], -HYBRIDS.index(v))) if better else "H0"


def gate2(cv_acc: dict, prior_acc: float, auc_value: float, selected: str) -> dict:
    """The pre-stated gate 2 (dev): stance direction AUC, gain over B1R, stance alone above the prior."""
    checks = {
        "stance_auc>=0.60": auc_value >= MIN_AUC,
        "selected_hybrid_minus_B1R>=+1.0pp": cv_acc[selected] - cv_acc["B1R"] >= MIN_GAIN - 1e-12,
        "S0_cv_accuracy>constant_prior": cv_acc["S0"] > prior_acc,
    }
    return {"checks": checks, "gate2": "PASS" if all(checks.values()) else "FAIL"}


# --------------------------------------------------------------------------------------
# Inputs, fitting, freezing, prediction
# --------------------------------------------------------------------------------------

def load_inputs(data: Path, split: str, wording: str, answers_name: Optional[str] = None):
    items = sorted((r for r in load_jsonl(data / "benchmark.jsonl")
                    if r["split"] == split and not r["likely_label_noise"]), key=lambda r: r["item_id"])
    pools = {r["item_id"]: r for r in load_jsonl(data / f"frozen_{split}.jsonl")}
    probs = load_stance_probs(load_jsonl(data / f"stance_{split}.jsonl"), wording)
    answers = load_jsonl(Path(answers_name) if answers_name else data / f"answers_{split}.jsonl")
    b1 = {r["item_id"]: r["verdict"] for r in answers if r["arm"] == "B1"}
    missing = [it["item_id"] for it in items if it["item_id"] not in pools or it["item_id"] not in b1]
    if missing:
        raise ValueError(f"{len(missing)} items lack a frozen pool or a B1 answer, e.g. {missing[:3]}")
    return items, pools, probs, b1


def fit_report(items, pools, probs, b1, folds: int = CV_FOLDS, repeats: int = CV_REPEATS) -> dict:
    """Cross-validate every variant on dev, apply gate 2, select, and fit the final models."""
    y = gold_indices(items)
    changed = np.array([it["kind"] == "changed" for it in items])
    X = {name: design_matrix(name, items, pools, probs, b1) for name in VARIANTS}
    prior_pred = repeated_cv(np.zeros((len(items), 0)), y, folds, repeats)
    preds = {name: repeated_cv(X[name], y, folds, repeats) for name in VARIANTS}
    acc = {name: cv_accuracy(preds[name], y) for name in VARIANTS}
    selected = select_variant(acc)
    gold_sr = y != LABELS.index("NOT ENOUGH INFORMATION")
    signed = X["S0"][:, FEATURES.index("signed")]
    direction = auc(signed[gold_sr], y[gold_sr] == LABELS.index("SUPPORTED"))
    prior_acc = cv_accuracy(prior_pred, y)
    models = {name: fit_softmax(X[name], y) for name in VARIANTS}
    return {
        "n_items": len(items), "n_changed": int(changed.sum()),
        "cv": {"folds": folds, "repeats": repeats, "seed": CV_SEED, "l2": L2,
               "prior_accuracy": round(prior_acc, 4),
               "variants": {name: {"accuracy": round(acc[name], 4),
                                   "accuracy_changed": round(cv_accuracy(preds[name], y, changed), 4),
                                   "recall": {k: round(v, 4) for k, v in cv_recall(preds[name], y).items()}}
                            for name in VARIANTS}},
        "stance_direction_auc": round(direction, 4), "selected": selected,
        "gate2": gate2(acc, prior_acc, direction, selected),
        "models": models, "bagged": {name: bagged_verdicts(preds[name]) for name in VARIANTS},
    }


def arm_records(name: str, items: Sequence[dict], pools: dict, verdict_idx: Sequence[int],
                probs_matrix: Optional[np.ndarray] = None) -> list[dict]:
    out = []
    for i, it in enumerate(items):
        rec = {"item_id": it["item_id"], "arm": name, "verdict": LABELS[verdict_idx[i]],
               "admitted": [c["pmid"] for c in top_candidates(pools[it["item_id"]])], "seconds": 0.0}
        if probs_matrix is not None:
            rec["probs"] = [round(float(p), 4) for p in probs_matrix[i]]
        out.append(rec)
    return out


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")


def to_markdown(report: dict) -> str:
    cv = report["cv"]
    lines = [f"# Synthesis layer: dev cross-validation ({report['n_items']} items, {report['n_changed']} changed)", "",
             f"{cv['folds']}-fold cross-validation, {cv['repeats']} repeats, seed {cv['seed']}; fixed ridge {cv['l2']}. "
             f"Constant-prior accuracy {100 * cv['prior_accuracy']:.1f}%. Stance-direction AUC "
             f"(SUPPORTED vs REFUTED, S0 signed score): {report['stance_direction_auc']:.3f}.", "",
             "| Variant | accuracy | accuracy, changed items | recall SUPPORTED | recall REFUTED | recall NOT ENOUGH INFORMATION |",
             "|---|---|---|---|---|---|"]
    for name, v in cv["variants"].items():
        r = v["recall"]
        lines.append(f"| {name} | {100 * v['accuracy']:.1f}% | {100 * v['accuracy_changed']:.1f}% | "
                     f"{100 * r['SUPPORTED']:.0f}% | {100 * r['REFUTED']:.0f}% | {100 * r['NOT ENOUGH INFORMATION']:.0f}% |")
    g = report["gate2"]
    lines += ["", f"**Selected hybrid: {report['selected']}.** Gate 2: **{g['gate2']}**", ""]
    lines += [f"* {'PASS' if ok else 'FAIL'}: {name}" for name, ok in g["checks"].items()]
    return "\n".join(lines) + "\n"


def cmd_fit(args) -> int:
    data, out_dir = Path(args.data_dir), Path(args.out_dir)
    wording = args.wording
    try:
        items, pools, probs, b1 = load_inputs(data, "dev", wording)
        report = fit_report(items, pools, probs, b1, repeats=args.repeats)
    except (ValueError, KeyError) as exc:
        print(f"cannot fit: {exc}", file=sys.stderr)
        return 2
    stance_records = load_jsonl(data / "stance_dev.jsonl")
    backend = next((r["backend"] for r in stance_records if not r.get("control")), None)
    frozen = {"format": 1, "fit_split": "dev", "stance": {"wording": wording, "backend": backend, "top_k": TOP_K},
              "settings": {"l2": L2, "half_life_days": A.HALF_LIFE_DAYS, "design_weights": DESIGN_WEIGHTS,
                           "features": list(FEATURES), "variants": list(VARIANTS)},
              "selected": report["selected"], "gate2": report["gate2"], "cv": report["cv"],
              "stance_direction_auc": report["stance_direction_auc"], "models": report["models"],
              "dev_items_sha256": hashlib.sha256(" ".join(it["item_id"] for it in items).encode()).hexdigest()}
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "synthesis_model.json"
    model_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "synthesis_cv_dev.md").write_text(to_markdown(report), encoding="utf-8", newline="\n")
    records = []
    for name in VARIANTS:
        records += arm_records(name, items, pools, report["bagged"][name])
    write_jsonl(data / "synthesis_dev.jsonl", records)
    print(to_markdown(report))
    print(f"frozen model -> {model_path} (sha256 {file_sha256(model_path)[:16]}...); commit it before "
          "running anything on the confirmatory split")
    return 0


def cmd_predict(args) -> int:
    data = Path(args.data_dir)
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"frozen model not found: {model_path} (run: synthesis fit)", file=sys.stderr)
        return 2
    frozen = json.loads(model_path.read_text(encoding="utf-8"))
    if frozen.get("fit_split") != "dev":
        print("the model must have been fitted on the dev split", file=sys.stderr)
        return 2
    try:
        items, pools, probs, b1 = load_inputs(data, args.split, frozen["stance"]["wording"])
        records = []
        for name in VARIANTS:
            P = predict_proba(frozen["models"][name], design_matrix(name, items, pools, probs, b1))
            records += arm_records(name, items, pools, P.argmax(axis=1).tolist(), P)
    except (ValueError, KeyError) as exc:
        print(f"cannot predict: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out or data / f"synthesis_{args.split}.jsonl")
    differs = check_config(out, {"model_sha256": file_sha256(model_path), "stance_wording": frozen["stance"]["wording"],
                                 "stance_backend": frozen["stance"]["backend"]},
                           ("model_sha256", "stance_wording", "stance_backend"))
    if differs:
        print(f"refusing to overwrite {out.name}: it was written with a different frozen model "
              f"({', '.join(differs)}; see {config_path(out).name})", file=sys.stderr)
        return 2
    write_jsonl(out, records)
    print(f"wrote {len(records)} arm records ({len(items)} items x {len(VARIANTS)} arms) -> {out}; "
          f"selected hybrid per the frozen model: {frozen['selected']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=("fit", "predict"))
    ap.add_argument("--split", default="confirm", choices=("confirm",),
                    help="predict only; dev predictions come from the out-of-fold pass inside fit")
    ap.add_argument("--wording", default="both", choices=("both", "A", "B"),
                    help="fit only; both = the average of the two wordings (pre-declared)")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "results"), help="fit: model and report")
    ap.add_argument("--model", default=str(HERE / "results" / "synthesis_model.json"), help="predict only")
    ap.add_argument("--out", default=None, help="predict: output file")
    ap.add_argument("--repeats", type=int, default=CV_REPEATS, help="cross-validation repeats (pre-stated: 50)")
    args = ap.parse_args(argv)
    return cmd_fit(args) if args.mode == "fit" else cmd_predict(args)


if __name__ == "__main__":
    sys.exit(main())
