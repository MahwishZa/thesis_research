"""Stage-2 analysis: RQ1 (does retrieval help?), RQ2 (does the synthesis layer help?) and the ablations.

    python -m experiments.medchange.analyze_stage2 --split dev        # exploratory (out-of-fold arms)
    python -m experiments.medchange.analyze_stage2 --split confirm    # confirmatory: run once

Arms come from ``answers_<split>.jsonl`` (B0, B1) and ``synthesis_<split>.jsonl`` (B1R, S0-S3, H0-H3,
H1C, H3C). Primary family (Holm, family-wise alpha .05), all items of the split: RQ1 = B1 vs B0 and
RQ2 = the selected hybrid vs B1R. Secondary family (Holm among themselves): the selected hybrid vs
B1; vs H0; vs its date-shuffled control; S0 vs B1; and the changed-items-only versions of RQ1 and RQ2.
A result is *confirmed* only if the Holm-adjusted p < .05 and the 95% interval excludes 0 (and the
difference is positive); everything else is reported as not confirmed. On the dev split the same
tables are exploratory: the hybrids there are out-of-fold predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

from evaluation.stats import holm, mcnemar, paired_bootstrap_ci

from .generate_answers import load_jsonl
from .synthesis import LABELS

HERE = Path(__file__).resolve().parent
ARM_ORDER = ("B0", "B1", "B1R", "S0", "S1", "S2", "S3", "H0", "H1", "H2", "H3", "H1C", "H3C")
ALPHA = 0.05


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson interval for a proportion."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def load_arms(data: Path, split: str) -> dict:
    """{(item_id, arm): record} from the answers and the synthesis files of ``split``."""
    out = {}
    for name in (f"answers_{split}.jsonl", f"synthesis_{split}.jsonl"):
        for r in load_jsonl(data / name):
            out[(r["item_id"], r["arm"])] = r
    return out


def correct_map(items: dict, answers: dict, arm: str, kinds: Sequence[str]) -> dict:
    return {i: int(answers[(i, arm)]["verdict"] == it["newest"]["label"])
            for i, it in items.items() if it["kind"] in kinds and (i, arm) in answers}


def class_stats(items: dict, answers: dict, arm: str, kinds: Sequence[str]) -> dict:
    ids = [i for i, it in items.items() if it["kind"] in kinds and (i, arm) in answers]
    recall, f1s = {}, []
    for label in LABELS:
        gold = [i for i in ids if items[i]["newest"]["label"] == label]
        predicted = [i for i in ids if answers[(i, arm)]["verdict"] == label]
        hit = sum(answers[(i, arm)]["verdict"] == label for i in gold)
        recall[label] = round(hit / len(gold), 4) if gold else None
        precision = hit / len(predicted) if predicted else 0.0
        r = hit / len(gold) if gold else 0.0
        f1s.append(2 * precision * r / (precision + r) if precision + r else 0.0)
    return {"recall": recall, "macro_f1": round(sum(f1s) / len(f1s), 4),
            "predicted_share": {label: round(sum(answers[(i, arm)]["verdict"] == label for i in ids) / len(ids), 4)
                                for label in LABELS}}


def summarize(items: dict, answers: dict, arms: Sequence[str]) -> dict:
    out = {}
    for arm in arms:
        row = {}
        for name, kinds in (("all", ("changed", "unchanged")), ("changed", ("changed",)),
                            ("unchanged", ("unchanged",))):
            c = correct_map(items, answers, arm, kinds)
            if c:
                row[name] = {"n": len(c), "accuracy": round(sum(c.values()) / len(c), 4),
                             "wilson95": wilson(sum(c.values()), len(c))}
        if row:
            row.update(class_stats(items, answers, arm, ("changed", "unchanged")))
            ids = [i for i, it in items.items() if it["kind"] == "changed" and (i, arm) in answers]
            row["outdated_verdict_rate_changed"] = (
                round(sum(answers[(i, arm)]["verdict"] == items[i]["previous"]["label"] for i in ids) / len(ids), 4)
                if ids else None)
            row["unparsed"] = sum(answers[(i, arm)]["verdict"] is None
                                  for i in items if (i, arm) in answers)
            out[arm] = row
    return out


def paired(items: dict, answers: dict, a: str, b: str, kinds: Sequence[str]) -> Optional[dict]:
    ca, cb = correct_map(items, answers, a, kinds), correct_map(items, answers, b, kinds)
    keys = sorted(set(ca) & set(cb))
    if len(keys) < 5:
        return None
    A_ = {k: ca[k] for k in keys}
    B_ = {k: cb[k] for k in keys}
    m = mcnemar(B_, A_)                       # baseline = b, proposed = a
    ci = paired_bootstrap_ci({k: float(v) for k, v in B_.items()}, {k: float(v) for k, v in A_.items()},
                             seed=f"{a}-{b}-{'+'.join(kinds)}", iterations=4000)
    return {"a": a, "b": b, "items": "+".join(kinds), "n": len(keys), "diff_a_minus_b": ci["difference"],
            "ci95": [ci["ci_low"], ci["ci_high"]], "a_only_correct": m.proposed_only,
            "b_only_correct": m.baseline_only, "mcnemar_p": round(m.p_value, 4)}


def _family(results: dict) -> dict:
    results = {k: v for k, v in results.items() if v}
    if not results:
        return {}
    adjusted = holm({k: v["mcnemar_p"] for k, v in results.items()})
    out = {}
    for k, v in results.items():
        confirmed = adjusted[k] < ALPHA and v["ci95"][0] > 0 and v["diff_a_minus_b"] > 0
        out[k] = dict(v, holm_p=round(adjusted[k], 4), confirmed=bool(confirmed))
    return out


def stage2_report(items: dict, answers: dict, primary: str, split: str) -> dict:
    both = ("changed", "unchanged")
    arms = [a for a in ARM_ORDER if any((i, a) in answers for i in items)]
    control = {"H1": "H1C", "H3": "H3C"}.get(primary)
    primary_family = _family({"RQ1 B1 vs B0": paired(items, answers, "B1", "B0", both),
                              f"RQ2 {primary} vs B1R": paired(items, answers, primary, "B1R", both)})
    secondary = {f"{primary} vs B1": paired(items, answers, primary, "B1", both),
                 "S0 vs B1": paired(items, answers, "S0", "B1", both),
                 "RQ1 B1 vs B0, changed items": paired(items, answers, "B1", "B0", ("changed",)),
                 f"RQ2 {primary} vs B1R, changed items": paired(items, answers, primary, "B1R", ("changed",))}
    if primary != "H0":
        secondary[f"{primary} vs H0"] = paired(items, answers, primary, "H0", both)
    if control:
        secondary[f"{primary} vs {control}"] = paired(items, answers, primary, control, both)
    return {"split": split, "status": "confirmatory" if split == "confirm" else "exploratory (dev; hybrids are out-of-fold)",
            "primary_arm": primary, "n_items": len(items), "per_arm": summarize(items, answers, arms),
            "primary_family": primary_family, "secondary_family": _family(secondary)}


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def to_markdown(rep: dict) -> str:
    lines = [f"# Stage-2 analysis, {rep['split']} split ({rep['status']})", "",
             f"Selected hybrid: **{rep['primary_arm']}**. Items: {rep['n_items']}.", "",
             "## Accuracy per arm", "",
             "| Arm | all items (95% CI) | changed | unchanged | recall S / R / NEI | macro-F1 | outdated rate |",
             "|---|---|---|---|---|---|---|"]
    for arm, r in rep["per_arm"].items():
        allr, ch, un = r.get("all"), r.get("changed"), r.get("unchanged")
        rc = r["recall"]
        lines.append(f"| {arm} | {_pct(allr['accuracy'])} ({_pct(allr['wilson95'][0])}–{_pct(allr['wilson95'][1])}) | "
                     f"{_pct(ch['accuracy']) if ch else 'n/a'} | {_pct(un['accuracy']) if un else 'n/a'} | "
                     f"{_pct(rc['SUPPORTED'])} / {_pct(rc['REFUTED'])} / {_pct(rc['NOT ENOUGH INFORMATION'])} | "
                     f"{_pct(r['macro_f1'])} | {_pct(r['outdated_verdict_rate_changed'])} |")
    for title, key in (("Primary family (Holm)", "primary_family"), ("Secondary family (Holm)", "secondary_family")):
        lines += ["", f"## {title}", "",
                  "| Comparison | difference (pp) | 95% CI (pp) | a-only / b-only | p | Holm p | confirmed |",
                  "|---|---|---|---|---|---|---|"]
        for name, v in rep[key].items():
            lines.append(f"| {name} | {100 * v['diff_a_minus_b']:+.1f} | {100 * v['ci95'][0]:+.1f} to {100 * v['ci95'][1]:+.1f} | "
                         f"{v['a_only_correct']} / {v['b_only_correct']} | {v['mcnemar_p']:.4f} | {v['holm_p']:.4f} | "
                         f"{'yes' if v['confirmed'] else 'no'} |")
    lines += ["", "A result is *confirmed* only if the Holm-adjusted p is below .05, the 95% interval excludes 0 and "
              "the difference is positive. Everything else is an estimate with an interval, not a finding."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm"))
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--model", default=str(HERE / "results" / "synthesis_model.json"))
    ap.add_argument("--primary", default=None, help="selected hybrid (default: from the frozen model file)")
    ap.add_argument("--out-dir", default=None, help="write stage2_analysis_<split>.json/.md here")
    args = ap.parse_args(argv)
    data = Path(args.data_dir)
    primary = args.primary
    if primary is None:
        model = Path(args.model)
        if not model.is_file():
            print(f"frozen model not found: {model}; pass --primary or run synthesis fit", file=sys.stderr)
            return 2
        primary = json.loads(model.read_text(encoding="utf-8"))["selected"]
    items = {r["item_id"]: r for r in load_jsonl(data / "benchmark.jsonl")
             if r["split"] == args.split and not r["likely_label_noise"]}
    answers = load_arms(data, args.split)
    needed = ("B0", "B1", "B1R", primary)
    missing = [a for a in needed if not any((i, a) in answers for i in items)]
    if missing:
        print(f"missing arms for {args.split}: {', '.join(missing)}", file=sys.stderr)
        return 2
    incomplete = [a for a in needed if sum((i, a) in answers for i in items) < len(items)]
    if args.split == "confirm" and incomplete:
        print(f"the confirmatory analysis needs every item for: {', '.join(incomplete)}", file=sys.stderr)
        return 2
    rep = stage2_report(items, answers, primary, args.split)
    text = to_markdown(rep)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, body in ((f"stage2_analysis_{args.split}.json", json.dumps(rep, indent=2) + "\n"),
                           (f"stage2_analysis_{args.split}.md", text)):
            with open(out / name, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(body)
        print(f"wrote {out / f'stage2_analysis_{args.split}.md'}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
