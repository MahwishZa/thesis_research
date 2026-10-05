"""Analysis of the realigned study: adapted RAG² (R2) and evidence-criteria verification (R2V).

    python -m experiments.medchange.analyze_rag2 --split dev        # exploratory
    python -m experiments.medchange.analyze_rag2 --split confirm    # confirmatory: run once, after the freeze

Inputs (``--data-dir``): ``benchmark.jsonl``, ``answers_<split>.jsonl`` (B0, B1), ``rag2_answers_<split>.jsonl``
(R2 family), and when present ``frozen_<split>.jsonl`` (evidence types of B1's abstracts),
``rag2_directness_<split>.jsonl`` (independent directness judgements) and the label audit
(``--label-audit``, for the label-stable subset). Output: ``rag2_analysis_<split>.json`` and ``.md``.

Pre-declared (docs/experiment_plan.md §7): the primary comparison is R2V - R2 over all items (exact McNemar,
paired bootstrap 95% interval); the requirement of +1.0 pp is read as *met and confirmed* (difference >= 1.0 pp,
p < .05, interval above 0), *met as a point estimate, not confirmed* (difference >= 1.0 pp otherwise) or
*not met*. Secondary comparisons are Holm-corrected among themselves, the ablations likewise. On the dev split
everything is exploratory and the requirement is never read as met.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

from . import rag2 as R
from .analyze_stage2 import ALPHA, _family, paired, stable_difference, summarize
from .arms import point_date
from .generate_answers import load_jsonl
from .synthesis import LABELS, study_type

HERE = Path(__file__).resolve().parent
ARM_ORDER = ("B0", "B1", "R2", "R2-RQ", "R2-BR", "R2-NF", "R2C", "R2V", "R2V-ND")
EVIDENCE_ARMS = ("B1",) + tuple(a for a in ARM_ORDER if a.startswith("R2"))
REQUIREMENT_PP = 1.0
PRIMARY = ("R2V", "R2", ("changed", "unchanged"))
SECONDARY = (("R2", "B1", ("changed", "unchanged")), ("R2V", "B1", ("changed", "unchanged")),
             ("R2C", "R2", ("changed", "unchanged")), ("R2V", "R2C", ("changed", "unchanged")),
             ("R2V", "R2V-ND", ("changed", "unchanged")), ("R2V", "R2", ("changed",)))
ABLATIONS = (("R2", "R2-RQ", ("changed", "unchanged")), ("R2", "R2-BR", ("changed", "unchanged")),
             ("R2", "R2-NF", ("changed", "unchanged")))
BOTH = ("changed", "unchanged")


def load(data: Path, split: str) -> tuple[dict, dict]:
    items = {r["item_id"]: r for r in load_jsonl(data / "benchmark.jsonl")
             if r["split"] == split and not r["likely_label_noise"]}
    answers = {}
    for name in (f"answers_{split}.jsonl", f"rag2_answers_{split}.jsonl"):
        for r in load_jsonl(data / name):
            if r["item_id"] in items and r["arm"] in ARM_ORDER:
                answers[(r["item_id"], r["arm"])] = r
    return items, answers


def present_arms(answers: dict) -> list[str]:
    arms = {arm for _, arm in answers}
    return [a for a in ARM_ORDER if a in arms]


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------

def hallucination_rows(items: dict, answers: dict, arms: Sequence[str]) -> dict:
    """Anachronism rate (all answers) and unsupported decisive verdicts (answers that had evidence)."""
    out = {}
    for arm in arms:
        ids = [i for i in items if (i, arm) in answers]
        if not ids:
            continue
        ana = [bool(R.anachronistic_years(answers[(i, arm)]["text"], items[i]["newest"]["date"])) for i in ids]
        flags = [R.unsupported_decisive(answers[(i, arm)]["verdict"], answers[(i, arm)]["text"],
                                        len(answers[(i, arm)].get("admitted", []))) for i in ids]
        flags = [f for f in flags if f is not None]
        out[arm] = {"n": len(ids), "anachronism_rate": round(sum(ana) / len(ids), 4),
                    "unsupported_decisive_rate": round(sum(flags) / len(flags), 4) if flags else None,
                    "answers_with_evidence": len(flags)}
    return out


def verifier_rows(items: dict, answers: dict, arms: Sequence[str]) -> dict:
    """For the criteria arms: valid outputs, how often the verdict differs from R2's, and whether a change
    fixed or broke the answer."""
    out = {}
    for arm in arms:
        if arm not in R.CRITERIA_ARMS:
            continue
        ids = [i for i in items if (i, arm) in answers and answers[(i, arm)].get("fallback") is None]
        if not ids:
            continue
        recs = [answers[(i, arm)] for i in ids]
        changed = [i for i, r in zip(ids, recs) if r["changed"]]
        gold = {i: items[i]["newest"]["label"] for i in ids}
        fixed = sum(answers[(i, arm)]["verdict"] == gold[i] and answers[(i, arm)]["draft_verdict"] != gold[i]
                    for i in changed)
        broke = sum(answers[(i, arm)]["verdict"] != gold[i] and answers[(i, arm)]["draft_verdict"] == gold[i]
                    for i in changed)
        transitions: dict = {}
        for i in changed:
            key = f"{answers[(i, arm)]['draft_verdict']} -> {answers[(i, arm)]['verdict']}"
            transitions[key] = transitions.get(key, 0) + 1
        out[arm] = {"n_with_evidence": len(ids), "valid_rate": round(sum(r["valid"] for r in recs) / len(ids), 4),
                    "changed_rate": round(len(changed) / len(ids), 4), "changes_fixed": fixed,
                    "changes_broke": broke, "transitions": dict(sorted(transitions.items()))}
    return out


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------

def _admitted_meta(item_id: str, rec: dict, frozen: dict) -> list[dict]:
    if "admitted_stratum" in rec:
        return [{"pmid": p, "lower": lo, "upper": up, "stratum": s}
                for p, lo, up, s in zip(rec["admitted"], rec["admitted_lower"], rec["admitted_upper"],
                                        rec["admitted_stratum"])]
    pool = {c["pmid"]: c for c in frozen.get(item_id, {}).get("candidates", [])}
    out = []
    for p in rec.get("admitted", []):
        c = pool.get(p)
        if c is not None:
            out.append({"pmid": p, "lower": c["lower"], "upper": c["upper"], "stratum": study_type(c.get("pubtypes"))})
    return out


def retrieval_rows(items: dict, answers: dict, arms: Sequence[str], frozen: dict,
                   directness: Optional[dict] = None) -> dict:
    """Per evidence arm: how much was admitted, of which evidence types, how recent, how much from the update
    window, overlap with B1, and (if judged) the share judged directly on the question."""
    out = {}
    for arm in arms:
        if arm not in EVIDENCE_ARMS:
            continue
        ids = [i for i in items if (i, arm) in answers]
        if not ids:
            continue
        counts, zero, mix, shares, any_window, ages, jac, direct = [], 0, {s: 0 for s in R.STRATA}, [], 0, [], [], []
        for i in ids:
            it, rec = items[i], answers[(i, arm)]
            meta = _admitted_meta(i, rec, frozen)
            counts.append(len(rec.get("admitted", [])))
            zero += not rec.get("admitted")
            for m in meta:
                mix[m["stratum"]] += 1
            flags = [it["previous"]["date"] < m["lower"] and m["upper"] <= it["newest"]["date"] for m in meta]
            if meta:
                shares.append(sum(flags) / len(meta))
            any_window += any(flags)
            cutoff = date.fromisoformat(it["newest"]["date"])
            ages += [max(0, (cutoff - point_date(m)).days) / 365.25 for m in meta]
            if arm != "B1" and (i, "B1") in answers:
                a_, b_ = set(rec.get("admitted", [])), set(answers[(i, "B1")].get("admitted", []))
                if a_ | b_:
                    jac.append(len(a_ & b_) / len(a_ | b_))
            if directness:
                judged = [directness[(i, p)] for p in rec.get("admitted", [])
                          if (i, p) in directness and directness[(i, p)]["valid"]]
                if judged:
                    direct.append(sum(j["p_yes"] >= R.FILTER_THRESHOLD for j in judged) / len(judged))
        total = sum(mix.values())
        out[arm] = {"n": len(ids), "mean_admitted": round(sum(counts) / len(ids), 2),
                    "share_without_evidence": round(zero / len(ids), 4),
                    "evidence_mix": {s: round(v / total, 4) for s, v in mix.items()} if total else None,
                    "update_window_share": round(sum(shares) / len(shares), 4) if shares else None,
                    "items_with_update_window_evidence": round(any_window / len(ids), 4),
                    "mean_age_years": round(sum(ages) / len(ages), 2) if ages else None,
                    "jaccard_with_B1": round(sum(jac) / len(jac), 4) if jac else None,
                    "directness_at_k": round(sum(direct) / len(direct), 4) if direct else None}
    return out


# --------------------------------------------------------------------------------------
# Comparisons and the reading of the requirement
# --------------------------------------------------------------------------------------

def compare(items: dict, answers: dict, specs) -> dict:
    out = {}
    for a, b, kinds in specs:
        r = paired(items, answers, a, b, kinds)
        if r:
            out[f"{a} vs {b}" + ("" if tuple(kinds) == BOTH else f" ({'+'.join(kinds)})")] = r
    return out


def primary_result(items: dict, answers: dict) -> Optional[dict]:
    r = paired(items, answers, *PRIMARY)
    if not r:
        return None
    confirmed = r["mcnemar_p"] < ALPHA and r["ci95"][0] > 0 and r["diff_a_minus_b"] > 0
    return dict(r, confirmed=bool(confirmed))


def requirement_reading(primary: Optional[dict], split: str) -> str:
    """The pre-declared reading of the +1.0 pp requirement (docs/experiment_plan.md §7)."""
    if primary is None:
        return "not run"
    if split != "confirm":
        return "dev estimate only (exploratory; the requirement is read on the confirmatory split)"
    if primary["diff_a_minus_b"] * 100 < REQUIREMENT_PP:
        return "not met"
    if primary["confirmed"]:
        return "met and confirmed"
    return "met as a point estimate, not confirmed"


def stable_ids_from_audit(path: Optional[Path]) -> Optional[list[str]]:
    if not path or not path.is_file():
        return None
    return sorted(r["item_id"] for r in load_jsonl(path) if r.get("which") == "newest" and r["gold"] == r["relabel"])


def case_study(items: dict, answers: dict, arms: Sequence[str]) -> dict:
    ids = sorted(i for i, it in items.items() if it.get("ad_related"))
    rows = {}
    for i in ids:
        rows[i] = {"kind": items[i]["kind"], "gold": items[i]["newest"]["label"],
                   "previous": items[i]["previous"]["label"],
                   "verdicts": {a: answers[(i, a)]["verdict"] for a in arms if (i, a) in answers}}
    return {"items": rows, "correct": {a: sum(answers[(i, a)]["verdict"] == items[i]["newest"]["label"]
                                              for i in ids if (i, a) in answers) for a in arms},
            "n": len(ids)}


def report(items: dict, answers: dict, split: str, frozen: Optional[dict] = None,
           directness: Optional[dict] = None, stable_ids: Optional[Sequence[str]] = None) -> dict:
    arms = present_arms(answers)
    primary = primary_result(items, answers)
    rep = {"split": split, "items": len(items), "arms": arms,
           "generation": summarize(items, answers, arms),
           "unsupported_answers": hallucination_rows(items, answers, arms),
           "verification": verifier_rows(items, answers, arms),
           "retrieval": retrieval_rows(items, answers, arms, frozen or {}, directness),
           "primary": primary, "requirement": requirement_reading(primary, split),
           "secondary": _family(compare(items, answers, SECONDARY)),
           "ablations": _family(compare(items, answers, ABLATIONS)),
           "case_study_alzheimers": case_study(items, answers, arms)}
    if stable_ids:
        rep["label_stable"] = {"n": len(stable_ids),
                               "R2V_minus_R2": stable_difference(items, answers, "R2V", "R2", stable_ids),
                               "R2_minus_B1": stable_difference(items, answers, "R2", "B1", stable_ids)}
    return rep


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------

def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _pp(x) -> str:
    return "—" if x is None else f"{100 * x:+.1f}"


def _comparison_table(rows: dict, family: bool) -> list[str]:
    head = "| Comparison | n | difference (pp) | 95% CI (pp) | a-only / b-only | p |" + (" Holm p | confirmed |" if family else "")
    lines = [head, "|---|---|---|---|---|---|" + ("---|---|" if family else "")]
    for name, r in rows.items():
        line = (f"| {name} | {r['n']} | {_pp(r['diff_a_minus_b'])} | {_pp(r['ci95'][0])} to {_pp(r['ci95'][1])} | "
                f"{r['a_only_correct']} / {r['b_only_correct']} | {r['mcnemar_p']:.4f} |")
        if family:
            line += f" {r['holm_p']:.4f} | {'yes' if r['confirmed'] else 'no'} |"
        lines.append(line)
    return lines


def to_markdown(rep: dict) -> str:
    kind = "confirmatory" if rep["split"] == "confirm" else "exploratory"
    L = [f"# Adapted RAG² and evidence-criteria verification, {rep['split']} split ({kind})", "",
         f"Items: {rep['items']}. Arms: {', '.join(rep['arms'])}. Protocol: `docs/experiment_plan.md`.", "",
         "## Generation: verdict accuracy", "",
         "| Arm | all (95% CI) | changed | unchanged | recall S / R / NEI | macro-F1 | answers NEI | outdated rate |",
         "|---|---|---|---|---|---|---|---|"]
    for arm, row in rep["generation"].items():
        a = row.get("all", {})
        rec = row["recall"]
        L.append(f"| {arm} | {_pct(a.get('accuracy'))} ({_pct(a.get('wilson95', [None])[0])}–"
                 f"{_pct(a.get('wilson95', [None, None])[1])}) | {_pct(row.get('changed', {}).get('accuracy'))} | "
                 f"{_pct(row.get('unchanged', {}).get('accuracy'))} | {_pct(rec[LABELS[0]])} / {_pct(rec[LABELS[1]])} / "
                 f"{_pct(rec[LABELS[2]])} | {_pct(row['macro_f1'])} | {_pct(row['predicted_share'][LABELS[2]])} | "
                 f"{_pct(row['outdated_verdict_rate_changed'])} |")
    L += ["", "## The requirement (+1.0 pp over the adapted RAG² baseline)", ""]
    p = rep["primary"]
    if p:
        L += [f"R2V − R2 = **{_pp(p['diff_a_minus_b'])} pp** (95% CI {_pp(p['ci95'][0])} to {_pp(p['ci95'][1])}; "
              f"{p['a_only_correct']} questions right only with R2V, {p['b_only_correct']} only with R2; exact McNemar "
              f"p = {p['mcnemar_p']:.4f}).", "", f"Reading (pre-declared): **{rep['requirement']}**."]
    else:
        L += ["R2V or R2 is missing: not run."]
    L += ["", "## Secondary comparisons (Holm among themselves)", ""]
    L += _comparison_table(rep["secondary"], True) if rep["secondary"] else ["None available."]
    if rep["ablations"]:
        L += ["", "## Ablations of the adapted RAG² baseline (exploratory)", ""] + _comparison_table(rep["ablations"], True)
    L += ["", "## Unsupported answers (automatic indicators)", "",
          "| Arm | anachronism rate | unsupported decisive verdicts (answers with evidence) |", "|---|---|---|"]
    for arm, row in rep["unsupported_answers"].items():
        L.append(f"| {arm} | {_pct(row['anachronism_rate'])} | {_pct(row['unsupported_decisive_rate'])} "
                 f"({row['answers_with_evidence']}) |")
    if rep["verification"]:
        L += ["", "## What the criteria arms changed (questions with evidence)", "",
              "| Arm | valid output | verdict differs from R2 | changes that fixed / broke an answer |", "|---|---|---|---|"]
        for arm, row in rep["verification"].items():
            L.append(f"| {arm} | {_pct(row['valid_rate'])} | {_pct(row['changed_rate'])} | "
                     f"{row['changes_fixed']} / {row['changes_broke']} |")
    L += ["", "## Retrieval (descriptive)", "",
          "| Arm | admitted | none admitted | SR/MA / RCT / other | update-window share | mean age (y) | overlap with B1 | directness@k |",
          "|---|---|---|---|---|---|---|---|"]
    for arm, row in rep["retrieval"].items():
        mix = row["evidence_mix"]
        mix_s = " / ".join(_pct(mix[s]) for s in R.STRATA) if mix else "—"
        L.append(f"| {arm} | {row['mean_admitted']} | {_pct(row['share_without_evidence'])} | {mix_s} | "
                 f"{_pct(row['update_window_share'])} | {row['mean_age_years'] if row['mean_age_years'] is not None else '—'} | "
                 f"{_pct(row['jaccard_with_B1'])} | {_pct(row['directness_at_k'])} |")
    if rep.get("label_stable"):
        s = rep["label_stable"]
        L += ["", f"Label-stable subset ({s['n']} items, descriptive): R2V − R2 = "
              f"{_pp((s['R2V_minus_R2'] or {}).get('diff_a_minus_b'))} pp; R2 − B1 = "
              f"{_pp((s['R2_minus_B1'] or {}).get('diff_a_minus_b'))} pp."]
    cs = rep["case_study_alzheimers"]
    if cs["n"]:
        L += ["", f"Alzheimer's case study ({cs['n']} items, descriptive): correct answers per arm: "
              + ", ".join(f"{a} {k}" for a, k in cs["correct"].items()) + "."]
    L += ["", "A result is *confirmed* only if the p-value (Holm-adjusted in a family) is below .05, the 95% interval "
          "excludes 0 and the difference is positive. With about 500 questions only differences of roughly 4–6 pp "
          "can be confirmed (docs/experiment_plan.md §7)."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm"))
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=None, help="write rag2_analysis_<split>.json/.md here")
    ap.add_argument("--label-audit", default=None, help="label_audit_<split>.jsonl (label-stable subset)")
    a = ap.parse_args(argv)
    data = Path(a.data_dir)
    items, answers = load(data, a.split)
    if not items or not any(arm.startswith("R2") for _, arm in answers):
        print(f"nothing to analyse: no benchmark items or no R2-family answers for the {a.split} split",
              file=sys.stderr)
        return 2
    frozen = {r["item_id"]: r for r in load_jsonl(data / f"frozen_{a.split}.jsonl")}
    judged = load_jsonl(data / f"rag2_directness_{a.split}.jsonl")
    directness = {(r["item_id"], r["pmid"]): r for r in judged} or None
    stable = stable_ids_from_audit(Path(a.label_audit) if a.label_audit else None)
    rep = report(items, answers, a.split, frozen, directness, stable)
    text = to_markdown(rep)
    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"rag2_analysis_{a.split}.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        (out / f"rag2_analysis_{a.split}.md").write_text(text, encoding="utf-8")
        print(f"wrote {out / f'rag2_analysis_{a.split}.md'}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
