"""Error analysis of generated answers on the changed items (no new generation).

    python -m experiments.medchange.error_analysis --split dev --out-dir experiments\\medchange\\results

Answers one question: when a system is wrong on a question whose verdict changed, where did it go
wrong? Per arm, every changed item is placed in exactly one group, in this order:

  correct         the parsed verdict equals the newest gold verdict
  parse_failure   no ``VERDICT:`` line could be parsed
  retrieval_miss  the frozen pool holds no passage from the update window (after the previous
                  review version, on or before the newest), so no admission rule could help
  admission_miss  the pool holds such a passage but the arm admitted none of them
  followed_or_ignored_evidence
                  the arm admitted at least one update-window passage and the answer is still
                  wrong (the model ignored it, misread it, or it did not settle the verdict)

Also reported: accuracy with and without admitted update-window evidence, accuracy by change type,
and which verdicts each arm predicts. The gold label of the newest review was written by a model and
can itself be wrong (not checked here). Writes ``error_analysis_<split>.json`` and ``.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import _in_window
from .generate_answers import load_jsonl

HERE = Path(__file__).resolve().parent
ARM_ORDER = ("B0", "B1", "B2", "B3", "P", "C1")
GROUPS = ("correct", "parse_failure", "retrieval_miss", "admission_miss",
          "followed_or_ignored_evidence")
LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION", None)


def window_pmids(item: dict, pool: dict) -> set:
    """PMIDs in the frozen pool that surely first appeared inside the update window."""
    prev, new = item["previous"]["date"], item["newest"]["date"]
    return {c["pmid"] for c in pool["candidates"] if _in_window(c, prev, new)}


def classify(item: dict, answer: dict, in_window: set) -> str:
    """The single group of one (item, arm) answer; see the module docstring."""
    verdict = answer.get("verdict")
    if verdict == item["newest"]["label"]:
        return "correct"
    if verdict is None:
        return "parse_failure"
    if not in_window:
        return "retrieval_miss"
    if not (set(answer["admitted"]) & in_window):
        return "admission_miss"
    return "followed_or_ignored_evidence"


def _rate(num: int, den: int):
    return round(num / den, 4) if den else None


def analyse(items: dict, pools: dict, answers: dict, arms: list) -> dict:
    changed = [i for i, it in items.items() if it["kind"] == "changed" and i in pools]
    report = {"n_changed_items": len(changed),
              "items_whose_pool_has_update_window_evidence":
                  _rate(sum(bool(window_pmids(items[i], pools[i])) for i in changed), len(changed)),
              "per_arm": {}}
    for arm in arms:
        ids = [i for i in changed if (i, arm) in answers]
        groups = dict.fromkeys(GROUPS, 0)
        with_ev, without_ev = [0, 0], [0, 0]          # [correct, total]
        by_type, predicted = {}, dict.fromkeys(("SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION", "none"), 0)
        outdated = 0
        for i in ids:
            it, ans = items[i], answers[(i, arm)]
            win = window_pmids(it, pools[i])
            g = classify(it, ans, win)
            groups[g] += 1
            ok = g == "correct"
            bucket = with_ev if set(ans["admitted"]) & win else without_ev
            bucket[0] += ok
            bucket[1] += 1
            kind = "involves_NEI" if "NOT ENOUGH" in (it.get("change_type") or "") else "decisive_flip"
            t = by_type.setdefault(kind, [0, 0])
            t[0] += ok
            t[1] += 1
            predicted[ans.get("verdict") or "none"] += 1
            outdated += ans.get("verdict") == it["previous"]["label"]
        report["per_arm"][arm] = {
            "n": len(ids), "groups": groups,
            "accuracy_with_admitted_update_window_evidence": {"n": with_ev[1], "accuracy": _rate(*with_ev)},
            "accuracy_without": {"n": without_ev[1], "accuracy": _rate(*without_ev)},
            "accuracy_by_change_type": {k: {"n": v[1], "accuracy": _rate(*v)} for k, v in sorted(by_type.items())},
            "predicted_verdicts": predicted,
            "outdated_verdict_rate": _rate(outdated, len(ids))}
    return report


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def to_markdown(split: str, report: dict) -> str:
    arms = list(report["per_arm"])
    out = [f"# Error analysis, {split} split, questions whose verdict changed (n = {report['n_changed_items']})", "",
           "Each wrong answer is assigned one cause (definitions: top of `experiments/medchange/error_analysis.py`).", "",
           f"The candidate pool contained at least one update-window passage for "
           f"{_pct(report['items_whose_pool_has_update_window_evidence'])} of these questions, "
           "so for the rest no admission rule could have helped.", "",
           "## Where each arm's answers end up (counts)", "",
           "| Arm | correct | parse failure | retrieval miss | admission miss | evidence admitted, still wrong |",
           "|---|---|---|---|---|---|"]
    for a in arms:
        g = report["per_arm"][a]["groups"]
        out.append(f"| {a} | " + " | ".join(str(g[k]) for k in GROUPS) + " |")
    out += ["", "## Does admitted update-window evidence help?", "",
            "| Arm | accuracy when it admitted such a passage (n) | accuracy when it did not (n) |", "|---|---|---|"]
    for a in arms:
        r = report["per_arm"][a]
        w, wo = r["accuracy_with_admitted_update_window_evidence"], r["accuracy_without"]
        out.append(f"| {a} | {_pct(w['accuracy'])} ({w['n']}) | {_pct(wo['accuracy'])} ({wo['n']}) |")
    out += ["", "## Accuracy by type of change", "", "| Arm | involves NOT ENOUGH INFORMATION | decisive flip |", "|---|---|---|"]
    for a in arms:
        t = report["per_arm"][a]["accuracy_by_change_type"]
        cell = lambda k: (f"{_pct(t[k]['accuracy'])} ({t[k]['n']})" if k in t else "n/a")
        out.append(f"| {a} | {cell('involves_NEI')} | {cell('decisive_flip')} |")
    out += ["", "## Which verdicts each arm gives (counts) and how often it repeats the outdated verdict", "",
            "| Arm | SUPPORTED | REFUTED | NOT ENOUGH INFORMATION | none | outdated-verdict rate |", "|---|---|---|---|---|---|"]
    for a in arms:
        p = report["per_arm"][a]["predicted_verdicts"]
        out.append(f"| {a} | {p['SUPPORTED']} | {p['REFUTED']} | {p['NOT ENOUGH INFORMATION']} | {p['none']} | "
                   f"{_pct(report['per_arm'][a]['outdated_verdict_rate'])} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm"))
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--answers", default=None)
    ap.add_argument("--out-dir", default=None, help="write error_analysis_<split>.json/.md here")
    args = ap.parse_args(argv)
    d = Path(args.data_dir)
    items = {r["item_id"]: r for r in load_jsonl(d / "benchmark.jsonl")
             if r["split"] == args.split and not r["likely_label_noise"]}
    pools = {r["item_id"]: r for r in load_jsonl(d / f"frozen_{args.split}.jsonl")}
    answers = {(r["item_id"], r["arm"]): r for r in load_jsonl(Path(args.answers or d / f"answers_{args.split}.jsonl"))}
    if not answers:
        print("no answers found", file=sys.stderr)
        return 2
    arms = sorted({a for _, a in answers}, key=ARM_ORDER.index)
    report = analyse(items, pools, answers, arms)
    text = to_markdown(args.split, report)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, body in ((f"error_analysis_{args.split}.json", json.dumps(report, indent=2) + "\n"),
                           (f"error_analysis_{args.split}.md", text)):
            with open(out / name, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(body)
        print(f"wrote {out / f'error_analysis_{args.split}.md'}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
