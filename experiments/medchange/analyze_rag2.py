"""Analysis of the realigned study: adapted RAG² (R2) and evidence-criteria verification (R2V).

    python -m experiments.medchange.analyze_rag2 --split dev        # exploratory
    python -m experiments.medchange.analyze_rag2 --split confirm    # confirmatory: run once, after the freeze

Inputs (``--data-dir``): ``benchmark.jsonl``, ``answers_<split>.jsonl`` (B0, B1), ``rag2_answers_<split>.jsonl``
(R2 family), and when present ``frozen_<split>.jsonl`` (evidence types of B1's abstracts),
``rag2_directness_<split>.jsonl`` (independent directness judgements) and the label audit
(``--label-audit``, for the label-stable subset). Output: ``rag2_analysis_<split>.json`` and ``.md``.

Pre-declared (docs/protocol.md §1, docs/evaluation.md §5): the primary comparison is R2V - R2 over all items (exact McNemar,
paired bootstrap 95% interval), read as *confirmed* (p < .05 and the interval above 0), *positive but not confirmed* (the
difference is above 0 otherwise) or *no evidence* (the difference is not above 0). Secondary comparisons are Holm-corrected
among themselves, the ablations likewise. On the dev split everything is exploratory and nothing is read as a finding. The
Alzheimer's disease split ("ad") is a secondary held-out test, run once after the freeze; the difference is read on it in the
same way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import rag2 as R
from .analysis import build_report, load_split, stable_ids_from_audit
from .benchmark import LABELS
from .files import load_jsonl
from .scoring import detectable_range

HERE = Path(__file__).resolve().parent


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
    kind = {"confirm": "confirmatory", "ad": "secondary held-out test, Alzheimer's disease set"}.get(rep["split"], "exploratory")
    L = [f"# Adapted RAG² and evidence-criteria verification, {rep['split']} split ({kind})", "",
         f"Items: {rep['items']}. Arms: {', '.join(rep['arms'])}. Protocol: `docs/protocol.md`.", "",
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
    L += ["", "## Verdict accuracy: R2V against the adapted RAG² baseline (R2)", ""]
    p = rep["primary"]
    if p:
        L += [f"R2V − R2 = **{_pp(p['diff_a_minus_b'])} pp** (95% CI {_pp(p['ci95'][0])} to {_pp(p['ci95'][1])}; "
              f"{p['a_only_correct']} questions right only with R2V, {p['b_only_correct']} only with R2; exact McNemar "
              f"p = {p['mcnemar_p']:.4f}).", "", f"Reading (`docs/evaluation.md` §5): **{rep['reading']}**."]
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
    if rep.get("context_truncated"):
        L += ["", "Answers written after the abstracts were shortened because the prompt did not fit the context window: "
              + ", ".join(f"{a} {n}" for a, n in rep["context_truncated"].items()) + "."]
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
    if rep["split"] == "ad":
        L += ["", "The update-window and age columns of the retrieval table do not apply to this split: almost every "
              "review has a single version, so there is no earlier version to measure an update window from."]
    elif cs["n"]:
        L += ["", f"Alzheimer's case study ({cs['n']} items, descriptive): correct answers per arm: "
              + ", ".join(f"{a} {k}" for a, k in cs["correct"].items()) + "."]
    low, high = detectable_range(rep["items"])
    L += ["", "A result is *confirmed* only if the p-value (Holm-adjusted in a family) is below .05, the 95% interval "
          f"excludes 0 and the difference is positive. With {rep['items']} questions only differences of roughly "
          f"{low}–{high} pp can be confirmed (docs/evaluation.md §4)."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm", "ad"))
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=None, help="write rag2_analysis_<split>.json/.md here")
    ap.add_argument("--label-audit", default=None, help="label_audit_<split>.jsonl (label-stable subset)")
    a = ap.parse_args(argv)
    data = Path(a.data_dir)
    items, answers = load_split(data, a.split)
    if not items or not any(arm.startswith("R2") for _, arm in answers):
        print(f"nothing to analyse: no benchmark items or no R2-family answers for the {a.split} split",
              file=sys.stderr)
        return 2
    frozen = {r["item_id"]: r for r in load_jsonl(data / f"frozen_{a.split}.jsonl")}
    judged = load_jsonl(data / f"rag2_directness_{a.split}.jsonl")
    directness = {(r["item_id"], r["pmid"]): r for r in judged} or None
    stable = stable_ids_from_audit(Path(a.label_audit) if a.label_audit else None)
    rep = build_report(items, answers, a.split, frozen, directness, stable)
    text = to_markdown(rep)
    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, body in ((f"rag2_analysis_{a.split}.json", json.dumps(rep, indent=2) + "\n"),
                           (f"rag2_analysis_{a.split}.md", text)):
            with open(out / name, "w", encoding="utf-8", newline="\n") as h:
                h.write(body)
        print(f"wrote {out / f'rag2_analysis_{a.split}.md'}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
